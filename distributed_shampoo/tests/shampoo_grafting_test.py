"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest
from typing import Callable, Iterable, Tuple

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_utils import GraftingType
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD


class DistributedShampooGraftingTest(unittest.TestCase):
    def _construct_quadratic(
        self,
        device: torch.device,
    ) -> Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        data = torch.arange(10, dtype=torch.float, device=device)
        model = nn.Sequential(
            nn.Linear(10, 1, bias=False),
        ).to(device=device)
        model[0].weight.data.fill_(1.0)
        loss = nn.MSELoss()
        target = torch.tensor([0.0]).to(device=device)
        return model, loss, data, target

    def _train_quadratic(
        self,
        optim_factory: Callable[
            [Iterable[Parameter], float],
            torch.optim.Optimizer,
        ],
        weight_decay: float,
        device: torch.device,
    ) -> Tuple[Parameter, torch.Tensor]:
        model, loss, data, target = self._construct_quadratic(device=device)
        params = model.parameters()
        optimizer = optim_factory(params, weight_decay)
        for _ in range(5):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()
        return model[0].weight.data.cpu(), objective.detach().cpu()

    def _test_baseline_and_shampoo(
        self,
        baseline_optim_factory: Callable[
            [Iterable[Parameter], float],
            torch.optim.Optimizer,
        ],
        shampoo_optim_factory: Callable[
            [Iterable[Parameter], float],
            torch.optim.Optimizer,
        ],
        device: torch.device,
        weight_decay: float,
    ) -> None:
        baseline_params, baseline_loss = self._train_quadratic(
            baseline_optim_factory,
            weight_decay=weight_decay,
            device=device,
        )
        shampoo_params, shampoo_loss = self._train_quadratic(
            shampoo_optim_factory,
            weight_decay,
            device=device,
        )
        torch.testing.assert_close(baseline_loss, shampoo_loss)
        torch.testing.assert_close(
            baseline_params,
            shampoo_params,
            rtol=torch.finfo(torch.float32).eps,
            atol=torch.finfo(torch.float32).eps,
        )

    def test_adagrad_grafting_on_quadratic(self) -> None:
        # construct optimizer factories for training
        def baseline_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return Adagrad(
                parameters,
                lr=0.01,
                eps=1e-10,
                weight_decay=weight_decay,
            )

        def shampoo_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return DistributedShampoo(
                parameters,
                lr=0.01,
                betas=(0.0, 1.0),
                epsilon=1e-10,
                momentum=0.0,
                weight_decay=weight_decay,
                max_preconditioner_dim=10,
                precondition_frequency=1,
                start_preconditioning_step=1000,
                num_trainers_per_group=1,
                use_decoupled_weight_decay=False,
                grafting_type=GraftingType.ADAGRAD,
                grafting_beta2=1.0,
                grafting_epsilon=1e-10,
            )

        # test with and without weight decay
        for weight_decay in [0.0, 0.3]:
            with self.subTest(weight_decay=weight_decay):
                self._test_baseline_and_shampoo(
                    baseline_optim_factory,
                    shampoo_optim_factory,
                    device=torch.device("cpu"),
                    weight_decay=weight_decay,
                )

    def test_adam_grafting_on_quadratic(self) -> None:
        # construct optimizer factories for training
        def baseline_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return Adam(
                parameters,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )

        def shampoo_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return DistributedShampoo(
                parameters,
                lr=0.001,
                betas=(0.9, 0.999),
                epsilon=1e-8,
                momentum=0.0,
                weight_decay=weight_decay,
                max_preconditioner_dim=10,
                precondition_frequency=1,
                start_preconditioning_step=1000,
                num_trainers_per_group=1,
                use_decoupled_weight_decay=False,
                grafting_type=GraftingType.ADAM,
                grafting_beta2=0.999,
                grafting_epsilon=1e-8,
            )

        # test with and without weight decay
        for weight_decay in [0.0, 0.3]:
            with self.subTest(weight_decay=weight_decay):
                self._test_baseline_and_shampoo(
                    baseline_optim_factory,
                    shampoo_optim_factory,
                    device=torch.device("cpu"),
                    weight_decay=weight_decay,
                )

    def test_adamw_grafting_on_quadratic(self) -> None:
        # construct optimizer factories for training
        def baseline_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return AdamW(
                parameters,
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )

        def shampoo_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return DistributedShampoo(
                parameters,
                lr=0.001,
                betas=(0.9, 0.999),
                epsilon=1e-8,
                momentum=0.0,
                weight_decay=weight_decay,
                max_preconditioner_dim=10,
                precondition_frequency=1,
                start_preconditioning_step=1000,
                num_trainers_per_group=1,
                use_decoupled_weight_decay=True,
                grafting_type=GraftingType.ADAM,
                grafting_beta2=0.999,
                grafting_epsilon=1e-8,
            )

        # test with and without weight decay
        for weight_decay in [0.0, 0.3]:
            with self.subTest(weight_decay=weight_decay):
                self._test_baseline_and_shampoo(
                    baseline_optim_factory,
                    shampoo_optim_factory,
                    device=torch.device("cpu"),
                    weight_decay=weight_decay,
                )

    def test_rmsprop_grafting_on_quadratic(self) -> None:
        # construct optimizer factories for training
        def baseline_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return RMSprop(
                parameters,
                lr=0.01,
                alpha=0.99,
                eps=1e-8,
                weight_decay=weight_decay,
            )

        def shampoo_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return DistributedShampoo(
                parameters,
                lr=0.01,
                betas=(0.0, 0.99),
                epsilon=1e-8,
                momentum=0.0,
                weight_decay=weight_decay,
                max_preconditioner_dim=10,
                precondition_frequency=1,
                start_preconditioning_step=1000,
                num_trainers_per_group=1,
                use_decoupled_weight_decay=False,
                grafting_type=GraftingType.RMSPROP,
                grafting_beta2=0.99,
                grafting_epsilon=1e-8,
            )

        # test with and without weight decay
        for weight_decay in [0.0, 0.3]:
            with self.subTest(weight_decay=weight_decay):
                self._test_baseline_and_shampoo(
                    baseline_optim_factory,
                    shampoo_optim_factory,
                    device=torch.device("cpu"),
                    weight_decay=weight_decay,
                )

    def test_sgd_grafting_on_quadratic(self) -> None:
        # construct optimizer factories for training
        def baseline_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return SGD(
                parameters,
                lr=0.1,
                momentum=0.9,
                weight_decay=weight_decay,
            )

        def shampoo_optim_factory(
            parameters: Iterable[Parameter], weight_decay: float
        ) -> torch.optim.Optimizer:
            return DistributedShampoo(
                parameters,
                lr=0.1,
                betas=(0.0, 0.9),
                epsilon=1e-10,
                momentum=0.9,
                weight_decay=weight_decay,
                max_preconditioner_dim=10,
                precondition_frequency=1,
                start_preconditioning_step=1000,
                num_trainers_per_group=1,
                use_decoupled_weight_decay=True,
                grafting_type=GraftingType.SGD,
                grafting_beta2=0.9,
                grafting_epsilon=1e-10,
            )

        # test with and without weight decay
        for weight_decay in [0.0, 0.3]:
            with self.subTest(weight_decay=weight_decay):
                self._test_baseline_and_shampoo(
                    baseline_optim_factory,
                    shampoo_optim_factory,
                    device=torch.device("cpu"),
                    weight_decay=weight_decay,
                )
