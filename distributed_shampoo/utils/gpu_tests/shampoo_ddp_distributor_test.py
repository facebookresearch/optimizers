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
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    DDPShampooConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSOAPConfig,
    EigendecomposedShampooPreconditionerConfig,
    PreconditionerConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
    construct_training_problem,
    train_model,
)

from torch import distributed as dist, tensor
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
            distributed_config: DDPShampooConfig | None,
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
                grafting_config=AdaGradGraftingConfig(epsilon=1e-8),
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
                    *default_tolerances(torch.float16),
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
                    distributed_config=None,
                    preconditioner_config=preconditioner_config,
                ),
                experimental_optim_factory=self._shampoo_optim_factory(
                    distributed_config=DDPShampooConfig(
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
                    distributed_config=DDPShampooConfig()
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
                                [9.7451e-06, 9.7451e-06, 9.7451e-06],
                                [9.7451e-06, 9.7451e-06, 9.7451e-06],
                                [9.7451e-06, 9.7451e-06, 9.7451e-06],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "shampoo", "factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [1.1501e-06, -5.6068e-06, -9.3039e-07],
                                [-5.6068e-06, 2.7333e-05, 4.5356e-06],
                                [-9.3039e-07, 4.5356e-06, 7.5264e-07],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 0]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [71.1973, -28.8003, -28.7987],
                                [-28.8003, 71.1994, -28.8008],
                                [-28.7987, -28.8008, 71.1979],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [96.5949, 16.5691, 2.7488],
                                [16.5691, 19.2215, -13.4046],
                                [2.7488, -13.4046, 97.7756],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "adagrad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [3.8337e-07, 9.1109e-06, 2.5088e-07],
                                [3.8337e-07, 9.1109e-06, 2.5088e-07],
                                [3.8337e-07, 9.1108e-06, 2.5088e-07],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "momentum"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [-0.9379, 1.4338, 0.9135],
                                [-0.9379, 1.4338, 0.9135],
                                [-0.9379, 1.4338, 0.9135],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [0]),
                        placements=(Replicate(),),
                    ),
                    '["block_1", "filtered_grad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [1.5298e-05, -7.4576e-05, -1.2375e-05],
                                [1.5298e-05, -7.4576e-05, -1.2375e-05],
                                [1.5298e-05, -7.4576e-05, -1.2375e-05],
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
                                [1.3363e-06, 1.3363e-06, 1.3363e-06],
                                [1.3363e-06, 1.3363e-06, 1.3363e-06],
                                [1.3363e-06, 1.3363e-06, 1.3363e-06],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "shampoo", "factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [2.4575e-06, 9.4018e-07, 1.7113e-06],
                                [9.4018e-07, 3.5969e-07, 6.5472e-07],
                                [1.7113e-06, 6.5472e-07, 1.1917e-06],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 0]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [74.1114, -25.8886, -25.8885],
                                [-25.8886, 74.1112, -25.8882],
                                [-25.8885, -25.8882, 74.1110],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 1]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [52.3903, -18.2144, -33.1540],
                                [-18.2144, 93.0316, -12.6838],
                                [-33.1540, -12.6838, 76.9120],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "adagrad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [8.1917e-07, 1.1990e-07, 3.9724e-07],
                                [8.1917e-07, 1.1990e-07, 3.9724e-07],
                                [8.1917e-07, 1.1990e-07, 3.9724e-07],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "momentum"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [-1.3151, -1.0032, -1.1617],
                                [-1.3151, -1.0032, -1.1617],
                                [-1.3151, -1.0032, -1.1617],
                            ]
                        ),
                        device_mesh=DeviceMesh(str(self._device), [1]),
                        placements=(Replicate(),),
                    ),
                    '["block_0", "filtered_grad"]': DTensor.from_local(
                        local_tensor=tensor(
                            [
                                [2.2362e-05, 8.5550e-06, 1.5572e-05],
                                [2.2362e-05, 8.5550e-06, 1.5572e-05],
                                [2.2362e-05, 8.5550e-06, 1.5572e-05],
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

            steps_with_gradients = 2
            model, loss, data, target, optimizer = train_model(
                optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                    distributed_config=DDPShampooConfig(
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
                num_steps=steps_with_gradients,
            )

            steps_without_gradients = 3
            for _ in range(steps_without_gradients):
                objective = loss(model(data), target)
                objective.backward()

                # Experiment setup: all ranks get no gradients.
                optimizer.zero_grad()

                optimizer.step()

            assert isinstance(optimizer, DistributedShampoo)
            # For each rank, no matter getting gradients or not, the step should be updated.
            self.assertEqual(
                optimizer.distributed_state_dict(key_to_param=model.named_parameters())[
                    "state"
                ]["scalar"]['["step"]'].item(),
                steps_with_gradients + steps_without_gradients,
            )

        @parametrize("communicate_params", (False, True))
        def test_some_ranks_with_no_grads_due_to_dead_layers(
            self, communicate_params: bool
        ) -> None:
            self._init_distributed()

            num_steps = 3
            model, _, _, _, optimizer = train_model(
                optim_factory=AbstractTest.ShampooDDPDistributorDeviceTest._shampoo_optim_factory(
                    distributed_config=DDPShampooConfig(
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
                        distributed_config=DDPShampooConfig()
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


class ShampooDDPDistributorCPUTest(AbstractTest.ShampooDDPDistributorDeviceTest):
    @property
    def _device(self) -> torch.device:
        return torch.device("cpu")


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooDDPDistributorGPUTest(AbstractTest.ShampooDDPDistributorDeviceTest):
    @property
    def _device(self) -> torch.device:
        return torch.device("cuda")
