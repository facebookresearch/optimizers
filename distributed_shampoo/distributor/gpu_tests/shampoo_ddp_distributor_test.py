"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import contextlib
import re
import unittest

from collections.abc import Callable
from functools import partial
from typing import cast
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.gpu_tests.distributor_test_utils import (
    DistributorOnEmptyParamTest,
)
from distributed_shampoo.distributor.shampoo_block_info import DTensorBlockInfo
from distributed_shampoo.distributor.shampoo_ddp_distributor import DDPDistributor
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DDPDistributedConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSOAPConfig,
    EigendecomposedShampooPreconditionerConfig,
    PreconditionerConfig,
    SingleDeviceDistributedConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
    construct_training_problem,
    train_model,
)

from torch import distributed as dist, nn, tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
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
        ) -> Callable[[ParamsT], torch.optim.Optimizer]:
            return partial(
                DistributedShampoo,
                lr=0.001,
                betas=(0.9, 1.0),
                epsilon=1e-8,
                momentum=0.9,
                weight_decay=0.0,
                max_preconditioner_dim=PRECONDITIONER_DIM,
                precondition_frequency=1,
                start_preconditioning_step=2,
                use_decoupled_weight_decay=True,
                grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
                distributed_config=distributed_config,
                preconditioner_config=preconditioner_config,
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
        def test_losses(
            self,
            num_trainers_per_group: int,
            communication_dtype: torch.dtype,
            communicate_params: bool,
            rtol: float,
            atol: float,
            preconditioner_config: PreconditionerConfig,
        ) -> None:
            self._init_distributed()

            compare_two_optimizers_on_weight_and_loss(
                control_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DefaultSingleDeviceDistributedConfig,
                    preconditioner_config=preconditioner_config,
                ),
                experimental_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig(
                        communication_dtype=communication_dtype,
                        num_trainers_per_group=num_trainers_per_group,
                        communicate_params=communicate_params,
                    ),
                    preconditioner_config=preconditioner_config,
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
            )

        def test_distributed_state_dict(self) -> None:
            self._init_distributed()

            num_steps = 3
            model, _, _, _, optimizer = train_model(
                optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                    distributed_config=DDPDistributedConfig()
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
            )

            assert isinstance(optimizer, DistributedShampoo)
            # Retrieve the distributed state dictionary of the first layer (i.e., the only layer) from the optimizer.
            distributed_state_dict = optimizer.distributed_state_dict(
                key_to_param=model.named_parameters(), save_param_groups=False
            )["state"]["linear_layers.0.weight"]

            # Define the expected distributed state dictionary for each rank.
            rank_to_expected_distributed_state_dict = {
                0: {
                    '["block_1", "shampoo", "factor_matrices", 0]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    7.3598799644969404e-05,
                                    7.3598806920927018e-05,
                                    7.3598806920927018e-05,
                                ],
                                [
                                    7.3598806920927018e-05,
                                    7.3598814196884632e-05,
                                    7.3598806920927018e-05,
                                ],
                                [
                                    7.3598806920927018e-05,
                                    7.3598806920927018e-05,
                                    7.3598806920927018e-05,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "shampoo", "factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    6.2865015934221447e-05,
                                    -4.1212708310922608e-05,
                                    -4.5267632231116295e-05,
                                ],
                                [
                                    -4.1212708310922608e-05,
                                    3.7400783185148612e-05,
                                    2.0580342606990598e-05,
                                ],
                                [
                                    -4.5267632231116295e-05,
                                    2.0580342606990598e-05,
                                    1.2053063255734742e-04,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 0]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    69.3880081176757812,
                                    -30.5801677703857422,
                                    -30.6043815612792969,
                                ],
                                [
                                    -30.5801677703857422,
                                    69.3745880126953125,
                                    -30.5909557342529297,
                                ],
                                [
                                    -30.6043853759765625,
                                    -30.5909519195556641,
                                    69.3988037109375000,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    14.4575786590576172,
                                    4.7331795692443848,
                                    1.7240729331970215,
                                ],
                                [
                                    4.7331795692443848,
                                    16.3652210235595703,
                                    0.1419535577297211,
                                ],
                                [
                                    1.7240731716156006,
                                    0.1419535428285599,
                                    9.9801998138427734,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "adagrad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    2.0955003492417745e-05,
                                    1.2466928637877572e-05,
                                    4.0176873881136999e-05,
                                ],
                                [
                                    2.0955008949385956e-05,
                                    1.2466928637877572e-05,
                                    4.0176877519115806e-05,
                                ],
                                [
                                    2.0955005311407149e-05,
                                    1.2466928637877572e-05,
                                    4.0176873881136999e-05,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "momentum"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    0.3861412107944489,
                                    -0.7968538999557495,
                                    2.0522127151489258,
                                ],
                                [
                                    0.3861400783061981,
                                    -0.7968541383743286,
                                    2.0522124767303467,
                                ],
                                [
                                    0.3861398994922638,
                                    -0.7968545556068420,
                                    2.0522127151489258,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "filtered_grad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    -0.0004798756854143,
                                    0.0001531048619654,
                                    0.0008914597565308,
                                ],
                                [
                                    -0.0004798757436220,
                                    0.0001531048910692,
                                    0.0008914598147385,
                                ],
                                [
                                    -0.0004798757145181,
                                    0.0001531048765173,
                                    0.0008914597565308,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                },
                1: {
                    '["block_0", "shampoo", "factor_matrices", 0]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    0.0001128265430452,
                                    0.0001128265430452,
                                    0.0001128265430452,
                                ],
                                [
                                    0.0001128265430452,
                                    0.0001128265430452,
                                    0.0001128265430452,
                                ],
                                [
                                    0.0001128265430452,
                                    0.0001128265430452,
                                    0.0001128265430452,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "shampoo", "factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    1.3368048530537635e-04,
                                    1.3935216702520847e-04,
                                    -1.0508949344512075e-05,
                                ],
                                [
                                    1.3935216702520847e-04,
                                    1.8916006956715137e-04,
                                    6.3602856243960559e-06,
                                ],
                                [
                                    -1.0508949344512075e-05,
                                    6.3602856243960559e-06,
                                    1.5639088815078139e-05,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 0]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    69.1099548339843750,
                                    -30.8445358276367188,
                                    -30.8929176330566406,
                                ],
                                [
                                    -30.8445396423339844,
                                    69.0552749633789062,
                                    -30.8382663726806641,
                                ],
                                [
                                    -30.8929176330566406,
                                    -30.8382663726806641,
                                    69.1036682128906250,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    13.0387792587280273,
                                    -4.4621400833129883,
                                    2.8714332580566406,
                                ],
                                [
                                    -4.4621400833129883,
                                    11.2216405868530273,
                                    -2.2768990993499756,
                                ],
                                [
                                    2.8714334964752197,
                                    -2.2769000530242920,
                                    17.8002052307128906,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "adagrad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    4.4560158130479977e-05,
                                    6.3053361373022199e-05,
                                    5.2130294534435961e-06,
                                ],
                                [
                                    4.4560158130479977e-05,
                                    6.3053361373022199e-05,
                                    5.2130299081909470e-06,
                                ],
                                [
                                    4.4560158130479977e-05,
                                    6.3053361373022199e-05,
                                    5.2130294534435961e-06,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "momentum"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    1.8189388513565063,
                                    1.9332338571548462,
                                    -0.6552859544754028,
                                ],
                                [
                                    1.8189374208450317,
                                    1.9332300424575806,
                                    -0.6552855968475342,
                                ],
                                [
                                    1.8189386129379272,
                                    1.9332307577133179,
                                    -0.6552851200103760,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "filtered_grad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [
                                    0.0008461118559353,
                                    0.0009362435666844,
                                    0.0001043296360876,
                                ],
                                [
                                    0.0008461119141430,
                                    0.0009362435666844,
                                    0.0001043296579155,
                                ],
                                [
                                    0.0008461119141430,
                                    0.0009362435666844,
                                    0.0001043296433636,
                                ],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                },
            }

            # Because DTensor does not support comparison, the verification has to be performed in two stages:
            # 1. Comparing the key sets are identical.
            # 2. Comparing the values are identical by converting it into local tensor.

            # Assert that the keys in the state dictionary match the expected keys for the current rank.
            self.assertEqual(
                distributed_state_dict.keys(),
                rank_to_expected_distributed_state_dict[dist.get_rank()].keys(),
                msg=f"{distributed_state_dict.keys() - rank_to_expected_distributed_state_dict[dist.get_rank()].keys()=} {rank_to_expected_distributed_state_dict[dist.get_rank()].keys() - distributed_state_dict.keys()=}",
            )

            # Helper function to get the local tensor from a DTensor or return the tensor itself.
            def local_tensor_getter(t: torch.Tensor | DTensor) -> torch.Tensor:
                return t.to_local() if isinstance(t, DTensor) else t

            # Compare each value in the state dictionary with the expected values.
            for key, actual_val in distributed_state_dict.items():
                expected_val = rank_to_expected_distributed_state_dict[dist.get_rank()][
                    key
                ]
                with self.subTest(
                    key=key, actual_val=actual_val, expected_val=expected_val
                ):
                    torch.testing.assert_close(
                        local_tensor_getter(actual_val),
                        local_tensor_getter(expected_val),
                        atol=1e-5,
                        rtol=2e-3,
                    )

        @parametrize("communicate_params", (False, True))
        def test_all_ranks_with_no_grads(self, communicate_params: bool) -> None:
            self._init_distributed()

            steps_without_gradients = 2
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
                )

            # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
            self.assertEqual(mock_backward.call_count, steps_without_gradients)

        @parametrize("communicate_params", (False, True))
        def test_some_ranks_with_no_grads_due_to_dead_layers(
            self, communicate_params: bool
        ) -> None:
            self._init_distributed()

            num_steps = 3
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
            )

            assert isinstance(optimizer, DistributedShampoo)
            # For each rank, no matter getting gradients or not, the step should be updated.
            self.assertEqual(
                optimizer.distributed_state_dict(key_to_param=model.named_parameters())[
                    "state"
                ]["linear_layers.0.weight"]['["step"]'].item(),
                num_steps,
            )

        # This mock is used to catch the number of calls to Shampoo's step(), which happened after __init__().
        # If there is no blocked params, __init__() will raise and step() should not be called.
        # Otherwise, step() will be called.
        @mock.patch.object(DistributedShampoo, "step")
        def test_empty_local_blocked_params(self, mock_step: mock.Mock) -> None:
            self._init_distributed()

            # The test setting is only rank 0 has params, so all other ranks have no parameters to work on.
            has_blocked_params = dist.get_rank() == 0
            with (
                contextlib.nullcontext()
                if has_blocked_params
                else self.assertRaisesRegex(
                    AssertionError,
                    re.escape("Some workers have no parameters to work on."),
                )
            ):
                train_model(
                    optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                        distributed_config=DDPDistributedConfig()
                    ),
                    # Setting model_linear_layers_dims to (PRECONDITIONER_DIM, 1) creates an model with one linear layer with PRECONDITIONER_DIMx1 weight.
                    # Because Shampoo's max_preconditioner_dim = PRECONDITIONER_DIM, there will be only one block.
                    # In the case of two trainers per group, there will be one trainer has no params to work on.
                    model_factory=partial(
                        construct_training_problem,
                        model_linear_layers_dims=(PRECONDITIONER_DIM, 1),
                        model_dead_layers_dims=None,
                        enable_learnable_scalar=False,
                        device=self._device,
                    ),
                )

            if has_blocked_params:
                mock_step.assert_called()
            else:
                mock_step.assert_not_called()

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
                    momentum=0.0,
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

        @property
        def _expected_masked_blocked_params(self) -> tuple[torch.Tensor, ...]:
            return ()

        def test_update_params(self) -> None:
            self._init_distributed()
            DistributorOnEmptyParamTest.Interface.test_update_params(self)

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
