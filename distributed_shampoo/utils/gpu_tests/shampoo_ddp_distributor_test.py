"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import contextlib
import re
from collections.abc import Callable
from functools import partial
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSOAPConfig,
    PreconditionerConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
    construct_training_problem,
)

from torch import distributed as dist, tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.optim.optimizer import ParamsT
from torch.testing._comparison import default_tolerances
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    DynamoDistributedMultiProcTestCase,
)
from torch.testing._internal.common_utils import parametrize


PRECONDITIONER_DIM = 3


class ShampooDDPDistributorDeviceTest(DynamoDistributedMultiProcTestCase):
    def _train_model(
        self,
        optim_factory: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        device: torch.device,
        model_linear_layers_dims: tuple[int, ...] = (
            PRECONDITIONER_DIM * 4,
            PRECONDITIONER_DIM * 2,
            1,
        ),
        model_dead_layer_dims: tuple[int, ...] | None = (
            PRECONDITIONER_DIM,
            PRECONDITIONER_DIM,
        ),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layer_dims=model_dead_layer_dims,
            device=device,
            fill=0.01,
        )
        params = list(model.parameters())
        optimizer = optim_factory(params)
        for _ in range(5):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()
        return params[0], objective.detach()

    def _init_distributed(self, device: str) -> None:
        if not dist.is_initialized():
            dist.init_process_group(
                dist.Backend.NCCL if device == "cuda" else dist.Backend.GLOO,
                init_method=f"file://{self.file_name}",
                rank=self.rank,
                world_size=self.world_size,
            )
        if device == "cuda":
            torch.cuda.set_device(self.rank)

    @property
    def world_size(self) -> int:
        return 2

    def _shampoo_optim_factory(
        self,
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
            grafting_config=AdaGradGraftingConfig(
                epsilon=1e-8,
            ),
            distributed_config=distributed_config,
            preconditioner_config=preconditioner_config,
        )

    @parametrize(
        "preconditioner_config",
        (
            DefaultShampooConfig,
            DefaultEigenvalueCorrectedShampooConfig,
            DefaultSOAPConfig,
        ),
    )
    @parametrize(
        "communication_dtype, communicate_params, rtol, atol",
        (
            # Expecting CommunicationDType.DEFAULT would have bitwise identical results (by setting rtol=atol=0.0).
            (CommunicationDType.DEFAULT, False, 0.0, 0.0),
            (CommunicationDType.DEFAULT, True, 0.0, 0.0),
            # Using FP16 for distributed parameters prohibitively lowers precision.
            (
                CommunicationDType.FP16,
                False,
                *default_tolerances(torch.float16),
            ),
            (
                CommunicationDType.BF16,
                False,
                # BF16 requires 2x tolerances than the original bfloat16 tolerances.
                *[2 * tol for tol in default_tolerances(torch.bfloat16)],
            ),
        ),
    )
    @parametrize("num_trainers_per_group", (-1, 1, 2))
    def test_losses(
        self,
        device: str,
        num_trainers_per_group: int,
        communication_dtype: CommunicationDType,
        communicate_params: bool,
        rtol: float,
        atol: float,
        preconditioner_config: PreconditionerConfig,
    ) -> None:
        self._init_distributed(device=device)

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
            model_dead_layer_dims=(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            device=torch.device(device),
            fill=0.01,
            rtol=rtol,
            atol=atol,
        )

    def test_distributed_state_dict(self, device: str) -> None:
        self._init_distributed(device=device)

        model, loss, data, target = construct_training_problem(
            # Setting model_linear_layers_dims to creates an model with one linear layer with (PRECONDITIONER_DIM * 2)xPRECONDITIONER_DIM weight.
            # Because Shampoo's max_preconditioner_dim = PRECONDITIONER_DIM, there will be two blocks; rank 0 has block 0 and rank 1 has block 1.
            model_linear_layers_dims=(PRECONDITIONER_DIM * 2, PRECONDITIONER_DIM),
            model_dead_layer_dims=None,
            device=torch.device(device),
            fill=0.01,
        )
        params = list(model.parameters())
        optimizer = self._shampoo_optim_factory(distributed_config=DDPShampooConfig())(
            params
        )
        assert isinstance(optimizer, DistributedShampoo)
        for _ in range(3):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()

        # Retrieve the distributed state dictionary of the first layer (i.e., the only layer) from the optimizer.
        distributed_state_dict = optimizer.distributed_state_dict(
            key_to_param=model.named_parameters(), save_param_groups=False
        )["state"]["linear_layers.0.weight"]

        # Define the expected distributed state dictionary for each rank.
        rank_to_expected_distributed_state_dict = {
            1: {
                '["block_1", "shampoo", "factor_matrices", 0]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0004, 0.0004, 0.0004],
                            [0.0004, 0.0004, 0.0004],
                            [0.0004, 0.0004, 0.0004],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["block_1", "shampoo", "factor_matrices", 1]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0002, 0.0003, 0.0004],
                            [0.0003, 0.0004, 0.0005],
                            [0.0004, 0.0005, 0.0006],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["block_1", "shampoo", "inv_factor_matrices", 0]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [68.2723, -31.3830, -31.4793],
                            [-31.3830, 68.3608, -31.5677],
                            [-31.4793, -31.5677, 68.4570],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["block_1", "shampoo", "inv_factor_matrices", 1]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [82.9738, -22.7016, -28.3770],
                            [-22.7016, 69.6087, -37.7380],
                            [-28.3770, -37.7380, 52.6266],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["block_1", "adagrad"]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [7.0041e-05, 1.2452e-04, 1.9456e-04],
                            [7.0041e-05, 1.2452e-04, 1.9456e-04],
                            [7.0041e-05, 1.2452e-04, 1.9456e-04],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["block_1", "momentum"]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [1.6924, 1.9865, 2.2806],
                            [1.6924, 1.9865, 2.2806],
                            [1.6924, 1.9865, 2.2806],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["block_1", "filtered_grad"]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0013, 0.0017, 0.0021],
                            [0.0013, 0.0017, 0.0021],
                            [0.0013, 0.0017, 0.0021],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [1]),
                    placements=(Replicate(),),
                ),
                '["step"]': tensor(3),
            },
            0: {
                '["block_0", "shampoo", "factor_matrices", 0]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [3.8911e-05, 3.8911e-05, 3.8911e-05],
                            [3.8911e-05, 3.8911e-05, 3.8911e-05],
                            [3.8911e-05, 3.8911e-05, 3.8911e-05],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["block_0", "shampoo", "factor_matrices", 1]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0000e00, 0.0000e00, 0.0000e00],
                            [0.0000e00, 2.3347e-05, 4.6694e-05],
                            [0.0000e00, 4.6694e-05, 9.3387e-05],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["block_0", "shampoo", "inv_factor_matrices", 0]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [69.8735, -30.1266, -30.1266],
                            [-30.1266, 69.8674, -30.1204],
                            [-30.1266, -30.1204, 69.8673],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["block_0", "shampoo", "inv_factor_matrices", 1]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [100.0000, 0.0000, 0.0000],
                            [0.0000, 81.9241, -36.1519],
                            [0.0000, -36.1519, 27.6963],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["block_0", "adagrad"]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0000e00, 7.7823e-06, 3.1129e-05],
                            [0.0000e00, 7.7823e-06, 3.1129e-05],
                            [0.0000e00, 7.7823e-06, 3.1129e-05],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["block_0", "momentum"]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0000, 1.5694, 2.3289],
                            [0.0000, 1.5694, 2.3289],
                            [0.0000, 1.5694, 2.3289],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["block_0", "filtered_grad"]': DTensor.from_local(
                    local_tensor=tensor(
                        [
                            [0.0000, 0.0004, 0.0009],
                            [0.0000, 0.0004, 0.0009],
                            [0.0000, 0.0004, 0.0009],
                        ]
                    ),
                    device_mesh=DeviceMesh(str(device), [0]),
                    placements=(Replicate(),),
                ),
                '["step"]': tensor(3),
            },
        }

        # Because DTensor does not support comparison, the verification has to be performed in two stages:
        # 1. Comparing the key sets are identical.
        # 2. Comparing the values are identical by converting it into local tensor.

        # Assert that the keys in the state dictionary match the expected keys for the current rank.
        self.assertEqual(
            distributed_state_dict.keys(),
            rank_to_expected_distributed_state_dict[dist.get_rank()].keys(),
        )

        # Helper function to get the local tensor from a DTensor or return the tensor itself.
        def local_tensor_getter(t: torch.Tensor | DTensor) -> torch.Tensor:
            return t.to_local() if isinstance(t, DTensor) else t

        # Compare each value in the state dictionary with the expected values.
        for key, actual_val in distributed_state_dict.items():
            expected_val = rank_to_expected_distributed_state_dict[dist.get_rank()][key]
            with self.subTest(
                key=key, actual_val=actual_val, expected_val=expected_val
            ):
                torch.testing.assert_close(
                    local_tensor_getter(actual_val),
                    local_tensor_getter(expected_val),
                    atol=1e-4,
                    rtol=2e-1,
                )

    # This mock is used to catch the number of calls to Shampoo's step(), which happened after __init__().
    # If there is no blocked params, __init__() will raise and step() should not be called.
    # Otherwise, step() will be called.
    def test_empty_local_blocked_params(self, device: str) -> None:
        self._init_distributed(device=device)

        # The test setting is only rank 0 has params, so all other ranks have no parameters to work on.
        has_blocked_params = dist.get_rank() == 0
        with mock.patch.object(DistributedShampoo, "step") as mock_step, (
            contextlib.nullcontext()
            if has_blocked_params
            else self.assertRaisesRegex(
                AssertionError,
                re.escape("Some workers have no parameters to work on."),
            )
        ):
            self._train_model(
                self._shampoo_optim_factory(distributed_config=DDPShampooConfig()),
                device=torch.device(device),
                # Setting model_linear_layers_dims to (PRECONDITIONER_DIM, 1) creates an model with one linear layer with PRECONDITIONER_DIMx1 weight.
                # Because Shampoo's max_preconditioner_dim = PRECONDITIONER_DIM, there will be only one block.
                # In the case of two trainers per group, there will be one trainer has no params to work on.
                model_linear_layers_dims=(PRECONDITIONER_DIM, 1),
                model_dead_layer_dims=None,
            )

        if has_blocked_params:
            mock_step.assert_called()
        else:
            mock_step.assert_not_called()

    def test_unsupported_communication_dtype(self, device: str) -> None:
        self._init_distributed(device=device)

        with mock.patch.object(CommunicationDType, "__eq__", return_value=False):
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape(
                    "Unsupported communication dtype: CommunicationDType.DEFAULT"
                ),
                self._train_model,
                self._shampoo_optim_factory(distributed_config=DDPShampooConfig()),
                device=device,
            )


instantiate_device_type_tests(
    ShampooDDPDistributorDeviceTest,
    globals(),
    # Only test GPU and CPU.
    only_for=("cuda", "cpu"),
)
