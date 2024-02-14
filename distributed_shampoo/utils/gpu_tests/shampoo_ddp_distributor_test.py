"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import os
import re
import unittest
from itertools import product

from typing import Callable, Iterable, Optional, Tuple
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
)

from torch import distributed as dist, nn
from torch.nn.parameter import Parameter
from torch.testing._internal.common_distributed import MultiProcessTestCase


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooDDPDistributorTest(MultiProcessTestCase):
    @staticmethod
    def _construct_model(
        device: torch.device,
    ) -> Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        data = torch.arange(80, dtype=torch.float, device=device)
        data /= torch.norm(data)
        model = nn.Sequential(
            nn.Linear(80, 40, bias=False),
            nn.Linear(40, 1, bias=False),
        ).to(device=device)
        model[0].weight.data.fill_(0.01)
        model[1].weight.data.fill_(0.01)
        model[1].requires_grad = False
        loss = nn.MSELoss()
        target = torch.tensor([0.0]).to(device=device)
        return model, loss, data, target

    @staticmethod
    def _train_model(
        optim_factory: Callable[
            [Iterable[Parameter]],
            torch.optim.Optimizer,
        ],
        device: torch.device,
    ) -> Tuple[Parameter, torch.Tensor]:
        model, loss, data, target = ShampooDDPDistributorTest._construct_model(
            device=device
        )
        params = model.parameters()
        optimizer = optim_factory(params)
        for _ in range(5):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()
        return model[0].weight.data.cpu(), objective.detach().cpu()

    @staticmethod
    def _test_two_configs(
        optim_factory1: Callable[
            [Iterable[Parameter]],
            torch.optim.Optimizer,
        ],
        optim_factory2: Callable[
            [Iterable[Parameter]],
            torch.optim.Optimizer,
        ],
        device: torch.device,
    ) -> None:
        params1, loss1 = ShampooDDPDistributorTest._train_model(
            optim_factory1,
            device=device,
        )
        params2, loss2 = ShampooDDPDistributorTest._train_model(
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
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_distributed(self) -> None:
        if not dist.is_initialized():
            dist.init_process_group(
                dist.Backend.NCCL,
                init_method=f"file://{self.file_name}",
                rank=self.rank,
                world_size=self.world_size,
            )
        torch.cuda.set_device(self.rank)

    @property
    def world_size(self) -> int:
        return 2

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: Optional[DDPShampooConfig],
    ) -> Callable[[Iterable[Parameter]], torch.optim.Optimizer]:
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
                self._test_two_configs(
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
                    device=torch.device("cuda"),
                )

    def test_invalid_num_trainers_per_group(self) -> None:
        self._init_distributed()
        with self.assertRaisesRegex(
            ValueError,
            re.escape("Invalid number of trainers per group:"),
        ):
            self._train_model(
                self._shampoo_optim_factory(
                    DDPShampooConfig(
                        num_trainers_per_group=3,
                    ),
                ),
                device=torch.device("cuda"),
            )

    @mock.patch("torch.distributed.get_world_size", return_value=8)
    def test_num_trainers_per_group_does_not_divide_world_size(
        self, mock_get_world_size: mock.Mock
    ) -> None:
        self._init_distributed()
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "distributed_config.num_trainers_per_group=3 must divide self._global_size=8!"
            ),
        ):
            self._train_model(
                self._shampoo_optim_factory(
                    DDPShampooConfig(
                        num_trainers_per_group=3,
                    ),
                ),
                device=torch.device("cuda"),
            )
        mock_get_world_size.assert_called()

    def test_dist_is_initialized(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("DDPDistributor needs torch.distributed to be initialized!"),
        ):
            ShampooDDPDistributorTest._train_model(
                self._shampoo_optim_factory(
                    DDPShampooConfig(
                        num_trainers_per_group=2,
                    ),
                ),
                device=torch.device("cuda"),
            )
