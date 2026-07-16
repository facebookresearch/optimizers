"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import abc
import os
import unittest
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, cast, Dict

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.gpu_tests.distributor_test_utils import (
    DistributorOnEmptyParamTest,
)
from distributed_shampoo.distributor.shampoo_block_info import DTensorBlockInfo
from distributed_shampoo.distributor.shampoo_ddp_distributor import DDPDistributor
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    ClassicMomentumConfig,
    DDPDistributedConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSOAPConfig,
    EigendecomposedShampooPreconditionerConfig,
    GeneralizedPrimalAveragingConfig,
    IterateAveragingConfig,
    PreconditionerConfig,
    ScheduleFreeConfig,
    SingleDeviceDistributedConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
    construct_training_problem,
    generate_global_train_data,
    train_model,
)
from distributed_shampoo.utils.shampoo_utils import pack_upper_triangular
from torch import distributed as dist, nn, tensor
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import ParamsT
from torch.testing._comparison import default_tolerances
from torch.testing._internal.common_distributed import (
    DynamoDistributedMultiProcTestCase,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


PRECONDITIONER_DIM = 3


def _shampoo_factor_dtensor(
    local_tensor: torch.Tensor,
    device_mesh: DeviceMesh,
    use_symmetric_packing: bool,
) -> DTensor:
    """Builds an expected Shampoo factor / inverse factor matrix DTensor.

    Packs `local_tensor` to upper triangular format when symmetric packing is enabled.
    Shampoo state in this test always uses Replicate placement.
    """
    return DTensor.from_local(
        local_tensor=pack_upper_triangular(local_tensor)
        if use_symmetric_packing
        else local_tensor,
        device_mesh=device_mesh,
        placements=(Replicate(),),
    )


# Use outer class as wrapper to avoid running the abstract test.
class AbstractTest:
    @instantiate_parametrized_tests
    class ShampooDDPDistributorDeviceTest(abc.ABC, DynamoDistributedMultiProcTestCase):
        @property
        @abc.abstractmethod
        def _device(self) -> torch.device: ...

        def _init_distributed(self) -> None:
            if not dist.is_initialized():
                dist.init_process_group(
                    {
                        torch.device("cuda"): dist.Backend.NCCL,
                        torch.device("cpu"): dist.Backend.GLOO,
                    }[self._device],
                    init_method=f"file://{self.file_name}",
                    rank=self.rank,
                    world_size=self.world_size,
                )
            if self._device == torch.device("cuda"):
                torch.cuda.set_device(self.rank)

        @property
        def world_size(self) -> int:
            return 2

        @staticmethod
        def _shampoo_optim_factory(
            distributed_config: DDPDistributedConfig | SingleDeviceDistributedConfig,
            preconditioner_config: PreconditionerConfig = DefaultShampooConfig,
            iterate_averaging_config: IterateAveragingConfig | None = None,
            weight_decay: float = 0.0,
            weight_decay_type: WeightDecayType = WeightDecayType.DECOUPLED,
        ) -> Callable[[ParamsT], torch.optim.Optimizer]:
            return partial(
                DistributedShampoo,
                lr=0.001,
                betas=(0.9, 1.0),
                epsilon=1e-8,
                weight_decay=weight_decay,
                max_preconditioner_dim=PRECONDITIONER_DIM,
                precondition_frequency=1,
                start_preconditioning_step=2,
                weight_decay_type=weight_decay_type,
                grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
                distributed_config=distributed_config,
                preconditioner_config=preconditioner_config,
                iterate_averaging_config=iterate_averaging_config,
            )

        @parametrize(
            "preconditioner_config",
            (
                DefaultShampooConfig,
                EigendecomposedShampooPreconditionerConfig(),
                DefaultEigenvalueCorrectedShampooConfig,
                DefaultSOAPConfig,
            ),
        )
        @parametrize(
            "communication_dtype, communicate_params, rtol, atol",
            (
                # Expecting CommunicationDType.DEFAULT would have bitwise identical results (by setting rtol=atol=0.0).
                (torch.float32, False, 0.0, 0.0),
                (torch.float32, True, 0.0, 0.0),
                # Using FP16 for distributed parameters prohibitively lowers precision.
                (
                    torch.float16,
                    False,
                    # FP16 requires 2x tolerances than the original float16 tolerances.
                    *[2 * tol for tol in default_tolerances(torch.bfloat16)],
                ),
                (
                    torch.bfloat16,
                    False,
                    # BF16 requires 2x tolerances than the original bfloat16 tolerances.
                    *[2 * tol for tol in default_tolerances(torch.bfloat16)],
                ),
            ),
        )
        @parametrize("num_trainers_per_group", (-1, 1, 2))
        @parametrize(
            "iterate_averaging_config",
            (
                None,
                GeneralizedPrimalAveragingConfig(),
                ScheduleFreeConfig(),
            ),
        )
        def test_losses(
            self,
            num_trainers_per_group: int,
            communication_dtype: torch.dtype,
            communicate_params: bool,
            rtol: float,
            atol: float,
            preconditioner_config: PreconditionerConfig,
            iterate_averaging_config: IterateAveragingConfig | None,
        ) -> None:
            self._init_distributed()

            compare_two_optimizers_on_weight_and_loss(
                control_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DefaultSingleDeviceDistributedConfig,
                    preconditioner_config=preconditioner_config,
                    iterate_averaging_config=iterate_averaging_config,
                ),
                experimental_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig(
                        communication_dtype=communication_dtype,
                        num_trainers_per_group=num_trainers_per_group,
                        communicate_params=communicate_params,
                    ),
                    preconditioner_config=preconditioner_config,
                    iterate_averaging_config=iterate_averaging_config,
                ),
                model_linear_layers_dims=(
                    PRECONDITIONER_DIM * 4,
                    PRECONDITIONER_DIM * 2,
                    1,
                ),
                model_dead_layers_dims=(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                device=self._device,
                fill=0.01,
                rtol=rtol,
                atol=atol,
                rank=self.rank,
                world_size=self.world_size,
                # DDP wrapping is needed so gradients are all-reduced across ranks,
                # matching real DDP training semantics. find_unused_parameters=True
                # is required because the model's dead layers don't participate in
                # forward() and thus never receive gradients.
                experimental_post_model_decoration=partial(
                    DDP,
                    device_ids=[self.rank] if self._device.type == "cuda" else None,
                    find_unused_parameters=True,
                ),
            )

        @parametrize(
            "weight_decay_type",
            (
                WeightDecayType.DECOUPLED,
                WeightDecayType.CORRECTED,
                WeightDecayType.INDEPENDENT,
                WeightDecayType.L2,
            ),
        )
        @parametrize("num_trainers_per_group", (-1, 1, 2))
        @parametrize(
            "iterate_averaging_config",
            (
                None,
                GeneralizedPrimalAveragingConfig(),
                ScheduleFreeConfig(),
                ClassicMomentumConfig(),
            ),
        )
        def test_weight_decay_losses(
            self,
            num_trainers_per_group: int,
            weight_decay_type: WeightDecayType,
            iterate_averaging_config: IterateAveragingConfig | None,
        ) -> None:
            self._init_distributed()

            compare_two_optimizers_on_weight_and_loss(
                control_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DefaultSingleDeviceDistributedConfig,
                    iterate_averaging_config=iterate_averaging_config,
                    weight_decay=0.3,
                    weight_decay_type=weight_decay_type,
                ),
                experimental_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig(
                        communication_dtype=torch.float32,
                        num_trainers_per_group=num_trainers_per_group,
                    ),
                    iterate_averaging_config=iterate_averaging_config,
                    weight_decay=0.3,
                    weight_decay_type=weight_decay_type,
                ),
                model_linear_layers_dims=(
                    PRECONDITIONER_DIM * 4,
                    PRECONDITIONER_DIM * 2,
                    1,
                ),
                model_dead_layers_dims=(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                device=self._device,
                fill=0.01,
                rtol=0.0,
                atol=0.0,
                rank=self.rank,
                world_size=self.world_size,
                experimental_post_model_decoration=partial(
                    DDP,
                    device_ids=[self.rank] if self._device.type == "cuda" else None,
                    find_unused_parameters=True,
                ),
            )

        @parametrize(
            "preconditioner_config",
            (
                DefaultShampooConfig,
                EigendecomposedShampooPreconditionerConfig(),
                DefaultEigenvalueCorrectedShampooConfig,
                DefaultSOAPConfig,
            ),
        )
        @parametrize(
            "communication_dtype, communicate_params, rtol, atol",
            (
                # Expecting CommunicationDType.DEFAULT would have bitwise identical results (by setting rtol=atol=0.0).
                (torch.float32, False, 0.0, 0.0),
                (torch.float32, True, 0.0, 0.0),
                # Using FP16 for distributed parameters prohibitively lowers precision.
                (
                    torch.float16,
                    False,
                    # FP16 requires 2x tolerances than the original float16 tolerances.
                    *[2 * tol for tol in default_tolerances(torch.bfloat16)],
                ),
                (
                    torch.bfloat16,
                    False,
                    # BF16 requires 2x tolerances than the original bfloat16 tolerances.
                    *[2 * tol for tol in default_tolerances(torch.bfloat16)],
                ),
            ),
        )
        def test_can_run_with_empty_local_params(
            self,
            communication_dtype: torch.dtype,
            communicate_params: bool,
            rtol: float,
            atol: float,
            preconditioner_config: PreconditionerConfig,
        ) -> None:
            """
            A variant of test_losses() that tests the case where ranks can have empty local blocked parameters.
            """
            self._init_distributed()

            compare_two_optimizers_on_weight_and_loss(
                control_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DefaultSingleDeviceDistributedConfig,
                    preconditioner_config=preconditioner_config,
                ),
                experimental_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig(
                        communication_dtype=communication_dtype,
                        num_trainers_per_group=-1,
                        communicate_params=communicate_params,
                    ),
                    preconditioner_config=preconditioner_config,
                ),
                # Setting model_linear_layers_dims to (PRECONDITIONER_DIM, 1) creates an model with one linear layer with PRECONDITIONER_DIMx1 weight.
                # Because Shampoo's max_preconditioner_dim = PRECONDITIONER_DIM, there will be only one block.
                # In the case of two trainers per group, there will be one trainer has no params to work on.
                model_linear_layers_dims=(PRECONDITIONER_DIM, 1),
                model_dead_layers_dims=None,
                enable_learnable_scalar=False,
                device=self._device,
                fill=0.01,
                rtol=rtol,
                atol=atol,
                rank=self.rank,
                world_size=self.world_size,
                # DDP wrapping with find_unused_parameters=True (see test_losses comment).
                experimental_post_model_decoration=partial(
                    DDP,
                    device_ids=[self.rank] if self._device.type == "cuda" else None,
                    find_unused_parameters=True,
                ),
            )

        @parametrize("use_symmetric_packing", (False, True))
        def test_state_dict(self, use_symmetric_packing: bool) -> None:
            self._init_distributed()

            num_steps = 3
            global_train_data = generate_global_train_data(
                num_steps=num_steps,
                world_size=self.world_size,
                data_shape=(PRECONDITIONER_DIM * 2,),
                device=self._device,
            )
            model, _, _, _, optimizer = train_model(
                optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig(),
                    preconditioner_config=replace(
                        DefaultShampooConfig,
                        use_symmetric_packing=use_symmetric_packing,
                    ),
                ),
                # Setting model_linear_layers_dims to creates an model with one linear layer with (PRECONDITIONER_DIM * 2)xPRECONDITIONER_DIM weight.
                # Because Shampoo's max_preconditioner_dim = PRECONDITIONER_DIM, there will be two blocks; rank 0 has block 0 and rank 1 has block 1.
                model_factory=partial(
                    construct_training_problem,
                    model_linear_layers_dims=(
                        PRECONDITIONER_DIM * 2,
                        PRECONDITIONER_DIM,
                    ),
                    model_dead_layers_dims=None,
                    device=self._device,
                    fill=0.01,
                ),
                num_steps=num_steps,
                train_data=global_train_data[:, self.rank],
            )

            assert isinstance(optimizer, DistributedShampoo)

            mesh_0 = DeviceMesh(
                self._device.type,
                [
                    0,
                ],
            )
            mesh_1 = DeviceMesh(
                self._device.type,
                [
                    1,
                ],
            )
            factor = partial(
                _shampoo_factor_dtensor, use_symmetric_packing=use_symmetric_packing
            )

            # Define the expected state dictionary for each rank.
            # The state_dict is keyed by parameter index (0 and 1).
            # Parameter 0 has block_0 (1x1 tensors), Parameter 1 has block_1 (3x3 tensors).
            rank_to_expected_state_dict: Dict[int, Dict[int, Dict[str, Any]]] = {
                0: {
                    0: {
                        "block_0": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: factor(
                                        local_tensor=tensor([[0.0001959779328899458]]),
                                        device_mesh=mesh_0,
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: factor(
                                        local_tensor=tensor([[71.43077087402344]]),
                                        device_mesh=mesh_0,
                                    ),
                                },
                            },
                            "adagrad": DTensor.from_local(
                                local_tensor=tensor([0.0001959779328899458]),
                                device_mesh=mesh_0,
                                placements=(Replicate(),),
                            ),
                            "filtered_grad": DTensor.from_local(
                                local_tensor=tensor([0.0018590810941532254]),
                                device_mesh=mesh_0,
                                placements=(Replicate(),),
                            ),
                        },
                        "step": tensor(3),
                    },
                    1: {
                        "block_1": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    1.361092654406093e-05,
                                                    1.361092654406093e-05,
                                                    1.361092654406093e-05,
                                                ],
                                                [
                                                    1.361092654406093e-05,
                                                    1.3610928363050334e-05,
                                                    1.361092654406093e-05,
                                                ],
                                                [
                                                    1.361092654406093e-05,
                                                    1.361092654406093e-05,
                                                    1.361092654406093e-05,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_0,
                                    ),
                                    1: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    1.2197860996820964e-05,
                                                    1.3136921097611776e-06,
                                                    -1.6640237845422234e-06,
                                                ],
                                                [
                                                    1.3136921097611776e-06,
                                                    7.509043371101143e-06,
                                                    -1.2525374586402904e-05,
                                                ],
                                                [
                                                    -1.6640237845422234e-06,
                                                    -1.2525374586402904e-05,
                                                    2.112587753799744e-05,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_0,
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    70.83631896972656,
                                                    -29.16368293762207,
                                                    -29.16368293762207,
                                                ],
                                                [
                                                    -29.163679122924805,
                                                    70.83528900146484,
                                                    -29.162641525268555,
                                                ],
                                                [
                                                    -29.163679122924805,
                                                    -29.16264533996582,
                                                    70.83528900146484,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_0,
                                    ),
                                    1: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    16.980432510375977,
                                                    -1.1272867918014526,
                                                    -0.1816159188747406,
                                                ],
                                                [
                                                    -1.1272867918014526,
                                                    49.9351692199707,
                                                    21.47319793701172,
                                                ],
                                                [
                                                    -0.1816159337759018,
                                                    21.47319793701172,
                                                    26.421972274780273,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_0,
                                    ),
                                },
                            },
                            "adagrad": DTensor.from_local(
                                local_tensor=tensor(
                                    [
                                        [
                                            4.065953362442087e-06,
                                            2.503014229660039e-06,
                                            7.04195917933248e-06,
                                        ],
                                        [
                                            4.065953362442087e-06,
                                            2.5030146844073897e-06,
                                            7.04195917933248e-06,
                                        ],
                                        [
                                            4.065953362442087e-06,
                                            2.503014229660039e-06,
                                            7.04195917933248e-06,
                                        ],
                                    ]
                                ),
                                device_mesh=mesh_0,
                                placements=(Replicate(),),
                            ),
                            "filtered_grad": DTensor.from_local(
                                local_tensor=tensor(
                                    [
                                        [
                                            -0.0002331818686798215,
                                            -8.544711454305798e-05,
                                            0.00015910383081063628,
                                        ],
                                        [
                                            -0.0002331818686798215,
                                            -8.54471290949732e-05,
                                            0.00015910383081063628,
                                        ],
                                        [
                                            -0.0002331818686798215,
                                            -8.544711454305798e-05,
                                            0.00015910383081063628,
                                        ],
                                    ]
                                ),
                                device_mesh=mesh_0,
                                placements=(Replicate(),),
                            ),
                        },
                    },
                },
                1: {
                    1: {
                        "block_0": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    2.4930672225309536e-05,
                                                    2.4930672225309536e-05,
                                                    2.4930672225309536e-05,
                                                ],
                                                [
                                                    2.4930672225309536e-05,
                                                    2.4930672225309536e-05,
                                                    2.4930672225309536e-05,
                                                ],
                                                [
                                                    2.4930672225309536e-05,
                                                    2.4930672225309536e-05,
                                                    2.4930672225309536e-05,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_1,
                                    ),
                                    1: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    1.4635477782576345e-05,
                                                    -1.1405569239286706e-05,
                                                    -8.130889909807593e-06,
                                                ],
                                                [
                                                    -1.1405569239286706e-05,
                                                    4.607178925652988e-05,
                                                    2.4604041755083017e-05,
                                                ],
                                                [
                                                    -8.130889909807593e-06,
                                                    2.4604041755083017e-05,
                                                    1.4084745998843573e-05,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_1,
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    70.21800231933594,
                                                    -29.731094360351562,
                                                    -29.73412322998047,
                                                ],
                                                [
                                                    -29.731094360351562,
                                                    70.24112701416016,
                                                    -29.75722885131836,
                                                ],
                                                [
                                                    -29.734119415283203,
                                                    -29.75722885131836,
                                                    70.24415588378906,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_1,
                                    ),
                                    1: factor(
                                        local_tensor=tensor(
                                            [
                                                [
                                                    17.4156436920166,
                                                    0.022538283839821815,
                                                    3.622297525405884,
                                                ],
                                                [
                                                    0.02253795601427555,
                                                    16.994813919067383,
                                                    -10.457884788513184,
                                                ],
                                                [
                                                    3.622298002243042,
                                                    -10.4578857421875,
                                                    32.26253890991211,
                                                ],
                                            ]
                                        ),
                                        device_mesh=mesh_1,
                                    ),
                                },
                            },
                            "adagrad": DTensor.from_local(
                                local_tensor=tensor(
                                    [
                                        [
                                            4.878492745774565e-06,
                                            1.5357263691839762e-05,
                                            4.694914878200507e-06,
                                        ],
                                        [
                                            4.878493200521916e-06,
                                            1.5357263691839762e-05,
                                            4.694915332947858e-06,
                                        ],
                                        [
                                            4.878492745774565e-06,
                                            1.5357263691839762e-05,
                                            4.694914878200507e-06,
                                        ],
                                    ]
                                ),
                                device_mesh=mesh_1,
                                placements=(Replicate(),),
                            ),
                            "filtered_grad": DTensor.from_local(
                                local_tensor=tensor(
                                    [
                                        [
                                            0.00016447670350316912,
                                            8.83667089510709e-05,
                                            -4.004316360806115e-05,
                                        ],
                                        [
                                            0.00016447667439933866,
                                            8.836669439915568e-05,
                                            -4.004319998784922e-05,
                                        ],
                                        [
                                            0.0001644766889512539,
                                            8.836670167511329e-05,
                                            -4.004317815997638e-05,
                                        ],
                                    ]
                                ),
                                device_mesh=mesh_1,
                                placements=(Replicate(),),
                            ),
                        },
                    },
                    0: {"step": tensor(3)},
                },
            }

            expected_state_dict = rank_to_expected_state_dict[dist.get_rank()]

            # Helper function to get the local tensor from a DTensor or return the tensor itself.
            def local_tensor_getter(t: torch.Tensor | DTensor) -> torch.Tensor:
                return t.to_local() if isinstance(t, DTensor) else t

            # Recursively compare nested state dictionaries.
            def assert_state_dict_close(
                actual: dict[Any, Any],
                expected: dict[Any, Any],
                path: str = "",
            ) -> None:
                for key in expected:
                    current_path = f"{path}.{key}" if path else str(key)
                    self.assertIn(
                        key, actual, f"Key {current_path} not found in actual"
                    )
                    actual_val = actual[key]
                    expected_val = expected[key]

                    if isinstance(expected_val, dict):
                        self.assertIsInstance(
                            actual_val, dict, f"Expected dict at {current_path}"
                        )
                        assert_state_dict_close(actual_val, expected_val, current_path)
                    elif isinstance(expected_val, (torch.Tensor, DTensor)):
                        with self.subTest(path=current_path):
                            torch.testing.assert_close(
                                local_tensor_getter(actual_val),
                                local_tensor_getter(expected_val),
                                atol=1e-4,
                                rtol=2e-1,
                            )

            state_dict = optimizer.state_dict()["state"]
            assert_state_dict_close(state_dict, expected_state_dict)

        @parametrize("communicate_params", (False, True))
        def test_all_ranks_with_no_grads(self, communicate_params: bool) -> None:
            self._init_distributed()

            steps_without_gradients = 2
            global_train_data = generate_global_train_data(
                num_steps=steps_without_gradients,
                world_size=self.world_size,
                data_shape=(PRECONDITIONER_DIM * 4,),
                device=self._device,
            )
            with unittest.mock.patch.object(torch.Tensor, "backward") as mock_backward:
                # By mocking the backward() method, we're intercepting gradient calculation.
                # This effectively simulates running forward passes without computing gradients.
                train_model(
                    optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                        distributed_config=DDPDistributedConfig(
                            communicate_params=communicate_params
                        )
                    ),
                    # 4 * 2 blocks in total. Rank 0 and Rank 1 have 4 blocks each.
                    model_factory=partial(
                        construct_training_problem,
                        model_linear_layers_dims=(
                            PRECONDITIONER_DIM * 4,
                            PRECONDITIONER_DIM * 2,
                        ),
                        model_dead_layers_dims=None,
                        device=self._device,
                    ),
                    num_steps=steps_without_gradients,
                    train_data=global_train_data[:, self.rank],
                )

            # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
            self.assertEqual(mock_backward.call_count, steps_without_gradients)

        @parametrize("communicate_params", (False, True))
        def test_some_ranks_with_no_grads_due_to_dead_layers(
            self, communicate_params: bool
        ) -> None:
            self._init_distributed()

            num_steps = 3
            global_train_data = generate_global_train_data(
                num_steps=num_steps,
                world_size=self.world_size,
                data_shape=(PRECONDITIONER_DIM,),
                device=self._device,
            )
            model, _, _, _, optimizer = train_model(
                optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig(
                        communicate_params=communicate_params
                    )
                ),
                # Experiment setup: only two blocks in total, one rank gets one block with gradients and the other rank gets one block without gradients due to dead layer.
                model_factory=partial(
                    construct_training_problem,
                    model_linear_layers_dims=(PRECONDITIONER_DIM, 1),
                    model_dead_layers_dims=(PRECONDITIONER_DIM, 1),
                    enable_learnable_scalar=False,
                    device=self._device,
                ),
                num_steps=num_steps,
                train_data=global_train_data[:, self.rank],
            )

            assert isinstance(optimizer, DistributedShampoo)
            # For each rank, no matter getting gradients or not, the step should be updated.
            osd_with_fqn = get_optimizer_state_dict(model, optimizer)
            self.assertEqual(
                cast(Dict[str, Any], osd_with_fqn["state"])["linear_layers.0.weight"][
                    "step"
                ],
                num_steps,
            )

    @instantiate_parametrized_tests
    class DDPDistributorOnEmptyParamDeviceTest(
        DynamoDistributedMultiProcTestCase,
        DistributorOnEmptyParamTest.Interface,
        abc.ABC,
    ):
        @property
        @abc.abstractmethod
        def _device(self) -> torch.device: ...

        def _init_distributed(self) -> None:
            if not dist.is_initialized():
                dist.init_process_group(
                    {
                        torch.device("cuda"): dist.Backend.NCCL,
                        torch.device("cpu"): dist.Backend.GLOO,
                    }[self._device],
                    init_method=f"file://{self.file_name}",
                    rank=self.rank,
                    world_size=self.world_size,
                )
            if self._device == torch.device("cuda"):
                torch.cuda.set_device(self.rank)

        @property
        def world_size(self) -> int:
            return 2

        def _construct_model_and_distributor(self) -> tuple[nn.Module, DDPDistributor]:
            # Create a model with specific configuration:
            # - linear_layers contains empty parameters (second dimension is 0), creating one block
            # - dead_layers contains a larger tensor that will be partitioned into three blocks
            # - The model will have four blocks in total (1 from linear_layers + 3 from dead_layers)
            model = construct_training_problem(
                model_linear_layers_dims=(PRECONDITIONER_DIM, 0),
                model_dead_layers_dims=(PRECONDITIONER_DIM, 3 * PRECONDITIONER_DIM),
                enable_learnable_scalar=False,
                device=self._device,
                fill=0.01,
            )[0]
            distributed_config = DDPDistributedConfig(num_trainers_per_group=1)
            distributor = DDPDistributor(
                param_group=DistributedShampoo(
                    model.parameters(),
                    lr=0.001,
                    betas=(0.9, 1.0),
                    epsilon=1e-8,
                    weight_decay=0.0,
                    precondition_frequency=1,
                    start_preconditioning_step=-1,
                    max_preconditioner_dim=PRECONDITIONER_DIM,
                    distributed_config=distributed_config,
                ).param_groups[0],
            )

            # Get the weight of the linear layers (which is empty) and set its gradient
            linear_layers: nn.ModuleList = cast(nn.ModuleList, model.linear_layers)
            first_linear_layer_weight: torch.Tensor = cast(
                torch.Tensor, linear_layers[0].weight
            )
            assert first_linear_layer_weight.numel() == 0
            first_linear_layer_weight.grad = torch.ones_like(first_linear_layer_weight)

            # Get the weight of the dead layers and set its gradient to None to make sure this is a dead layer
            dead_layers: nn.ModuleList = cast(nn.ModuleList, model.dead_layers)
            first_dead_layer_weight: torch.Tensor = cast(
                torch.Tensor, dead_layers[0].weight
            )
            first_dead_layer_weight.grad = None

            return model, distributor

        @parametrize("use_masked_tensors", [True, False])
        def test_update_params(self, use_masked_tensors: bool) -> None:
            self._init_distributed()
            DistributorOnEmptyParamTest.Interface._test_update_params_impl(
                self, use_masked_tensors
            )

        @property
        def _expected_local_grad_selector(self) -> tuple[bool, ...]:
            return (False, False, False, False)

        def test_local_grad_selector(self) -> None:
            self._init_distributed()
            DistributorOnEmptyParamTest.Interface.test_local_grad_selector(self)

        @property
        def _expected_local_blocked_params(self) -> tuple[torch.Tensor, ...]:
            return (
                torch.zeros(
                    (0,),
                    dtype=torch.float,
                    device=self._device,
                ),
                torch.zeros(
                    (PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                    dtype=torch.float,
                    device=self._device,
                ),
                torch.zeros(
                    (PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                    dtype=torch.float,
                    device=self._device,
                ),
                torch.zeros(
                    (PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                    dtype=torch.float,
                    device=self._device,
                ),
            )

        def test_local_blocked_params(self) -> None:
            self._init_distributed()
            DistributorOnEmptyParamTest.Interface.test_local_blocked_params(self)

        def _expected_local_block_info_list(
            self, model: nn.Module
        ) -> tuple[DTensorBlockInfo, ...]:
            # Get the weight parameters from the first linear and dead layers
            linear_layers: nn.ModuleList = cast(nn.ModuleList, model.linear_layers)
            first_linear_layer_weight: torch.Tensor = cast(
                torch.Tensor, linear_layers[0].weight
            )
            dead_layers: nn.ModuleList = cast(nn.ModuleList, model.dead_layers)
            first_dead_layer_weight: torch.Tensor = cast(
                torch.Tensor, dead_layers[0].weight
            )

            # Define expected BlockInfo objects for each rank
            return (
                DTensorBlockInfo(
                    param=first_linear_layer_weight,
                    composable_block_ids=(0, "block_0"),
                ),
                DTensorBlockInfo(
                    param=first_dead_layer_weight,
                    composable_block_ids=(1, "block_0"),
                ),
                DTensorBlockInfo(
                    param=first_dead_layer_weight,
                    composable_block_ids=(1, "block_1"),
                ),
                DTensorBlockInfo(
                    param=first_dead_layer_weight,
                    composable_block_ids=(1, "block_2"),
                ),
            )

        def test_local_block_info_list(self) -> None:
            self._init_distributed()
            DistributorOnEmptyParamTest.Interface.test_local_block_info_list(self)

        @property
        def _expected_local_masked_block_grads(self) -> tuple[torch.Tensor, ...]:
            return ()

        def test_merge_and_block_gradients(self) -> None:
            self._init_distributed()
            DistributorOnEmptyParamTest.Interface.test_merge_and_block_gradients(self)


class ShampooDDPDistributorCPUTest(AbstractTest.ShampooDDPDistributorDeviceTest):
    @property
    def _device(self) -> torch.device:
        return torch.device("cpu")

    def setUp(self) -> None:
        # Set TORCH_GLOO_LAZY_INIT to prevent timeout in test_empty_local_blocked_params.
        os.environ["TORCH_GLOO_LAZY_INIT"] = "1"
        super().setUp()

    def tearDown(self) -> None:
        # Clean up the environment variable after the test.
        del os.environ["TORCH_GLOO_LAZY_INIT"]
        return super().tearDown()


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooDDPDistributorGPUTest(AbstractTest.ShampooDDPDistributorDeviceTest):
    @property
    def _device(self) -> torch.device:
        return torch.device("cuda")


class DDPDistributorOnEmptyParamCPUTest(
    AbstractTest.DDPDistributorOnEmptyParamDeviceTest
):
    @property
    def _device(self) -> torch.device:
        return torch.device("cpu")


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class DDPDistributorOnEmptyParamGPUTest(
    AbstractTest.DDPDistributorOnEmptyParamDeviceTest
):
    @property
    def _device(self) -> torch.device:
        return torch.device("cuda")
