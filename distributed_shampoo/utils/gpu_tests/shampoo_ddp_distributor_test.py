"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import contextlib
import pathlib
import re
import unittest
from itertools import product

from typing import Callable, Optional
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem

from torch import distributed as dist
from torch.nn.parameter import Parameter
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import MultiProcessTestCase


# Use outer class as wrapper to avoid running the abstract test.
class AbstractTest:
    class ShampooDDPDistributorDeviceTest(abc.ABC, MultiProcessTestCase):
        @property
        @abc.abstractmethod
        def _device(self) -> torch.device: ...

        @staticmethod
        def _train_model(
            optim_factory: Callable[
                [ParamsT],
                torch.optim.Optimizer,
            ],
            device: torch.device,
            model_linear_layers_dims: tuple[int, ...] = (80, 40, 1),
            model_dead_layer_dims: Optional[tuple[int, ...]] = (20, 20),
        ) -> tuple[Parameter, torch.Tensor]:
            model, loss, data, target = construct_training_problem(
                model_linear_layers_dims=model_linear_layers_dims,
                model_dead_layer_dims=model_dead_layer_dims,
                device=device,
                fill=0.01,
            )
            params = model.parameters()
            optimizer = optim_factory(params)
            for _ in range(5):
                optimizer.zero_grad()
                objective = loss(model(data), target)
                objective.backward()
                optimizer.step()
            return model.linear_layers[0].weight.data.cpu(), objective.detach().cpu()

        @staticmethod
        def _test_two_configs(
            optim_factory1: Callable[
                [ParamsT],
                torch.optim.Optimizer,
            ],
            optim_factory2: Callable[
                [ParamsT],
                torch.optim.Optimizer,
            ],
            device: torch.device,
        ) -> None:
            params1, loss1 = AbstractTest.ShampooDDPDistributorDeviceTest._train_model(
                optim_factory1,
                device=device,
            )
            params2, loss2 = AbstractTest.ShampooDDPDistributorDeviceTest._train_model(
                optim_factory2,
                device=device,
            )
            torch.testing.assert_close(loss1, loss2)
            torch.testing.assert_close(params1, params2)

        def setUp(self) -> None:
            super().setUp()
            self._spawn_processes()

        def tearDown(self) -> None:
            super().tearDown()
            pathlib.Path(self.file_name).unlink(missing_ok=True)

        def _init_distributed(self) -> None:
            if not dist.is_initialized():
                dist.init_process_group(
                    dist.Backend.NCCL
                    if self._device == torch.device("cuda")
                    else dist.Backend.GLOO,
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
            distributed_config: Optional[DDPShampooConfig],
        ) -> Callable[[ParamsT], torch.optim.Optimizer]:
            return lambda parameters: (
                lambda distributed_config: DistributedShampoo(
                    parameters,
                    lr=0.001,
                    betas=(0.9, 1.0),
                    epsilon=1e-8,
                    momentum=0.0,
                    weight_decay=0.0,
                    max_preconditioner_dim=20,
                    precondition_frequency=1,
                    start_preconditioning_step=2,
                    use_decoupled_weight_decay=True,
                    grafting_config=AdaGradGraftingConfig(
                        epsilon=1e-8,
                    ),
                    distributed_config=distributed_config,
                )
            )(
                distributed_config,
            )

        def test_losses(self) -> None:
            self._init_distributed()
            for num_trainers_per_group, (
                communication_dtype,
                communicate_params,
            ) in product(
                (-1, 1, 2),
                (
                    (CommunicationDType.DEFAULT, False),
                    (CommunicationDType.DEFAULT, True),
                    (CommunicationDType.FP16, False),
                    (CommunicationDType.BF16, False),
                    # Using FP16 for distributed parameters prohibitively lowers precision.
                ),
            ):
                with self.subTest(
                    communication_dtype=communication_dtype,
                    num_trainers_per_group=num_trainers_per_group,
                    communicate_params=communicate_params,
                ):
                    AbstractTest.ShampooDDPDistributorDeviceTest._test_two_configs(
                        self._shampoo_optim_factory(
                            distributed_config=None,
                        ),
                        self._shampoo_optim_factory(
                            distributed_config=DDPShampooConfig(
                                communication_dtype=communication_dtype,
                                num_trainers_per_group=num_trainers_per_group,
                                communicate_params=communicate_params,
                            )
                        ),
                        device=self._device,
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
                AbstractTest.ShampooDDPDistributorDeviceTest._train_model(
                    self._shampoo_optim_factory(distributed_config=DDPShampooConfig()),
                    device=self._device,
                    # Setting model_linear_layers_dims to (20, 1) creates an model with one linear layer with 20x1 weight.
                    # Because Shampoo's max_preconditioner_dim = 20, there will be only one block.
                    # In the case of two trainers per group, there will be one trainer has no params to work on.
                    model_linear_layers_dims=(20, 1),
                    model_dead_layer_dims=None,
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
