"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from functools import partial
from itertools import product
from typing import Any, Callable, Type

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    RMSpropGraftingConfig,
    SGDGraftingConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from torch.nn.parameter import Parameter
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import ParamsT
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD


class DistributedShampooGraftingTest(unittest.TestCase):
    @staticmethod
    def _train_quadratic(
        optim_factory: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        device: torch.device,
    ) -> tuple[Parameter, torch.Tensor]:
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=(10, 1, 1),
            device=device,
            fill=1.0,
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
    def _test_baseline_and_shampoo(
        baseline_optim_factory: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        shampoo_optim_factory: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        device: torch.device,
    ) -> None:
        (
            baseline_params,
            baseline_loss,
        ) = DistributedShampooGraftingTest._train_quadratic(
            baseline_optim_factory,
            device=device,
        )
        shampoo_params, shampoo_loss = DistributedShampooGraftingTest._train_quadratic(
            shampoo_optim_factory,
            device=device,
        )
        torch.testing.assert_close(shampoo_loss, baseline_loss)
        torch.testing.assert_close(
            shampoo_params,
            baseline_params,
        )

    @staticmethod
    def _optim_factory(
        parameters: ParamsT,
        optim_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        return optim_cls(parameters, **kwargs)

    def test_adagrad_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.01,
                weight_decay=weight_decay,
            )
            with self.subTest(weight_decay=weight_decay, device=device):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        optim_factory, optim_cls=Adagrad, eps=1e-10
                    ),
                    shampoo_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        betas=(0.0, 1.0),
                        epsilon=1e-10,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_decoupled_weight_decay=False,
                        grafting_config=AdaGradGraftingConfig(
                            epsilon=1e-10,
                        ),
                    ),
                    device=device,
                )

    def test_adam_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            with self.subTest(weight_decay=weight_decay, device=device):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        optim_factory, optim_cls=Adam, eps=1e-8
                    ),
                    shampoo_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        epsilon=1e-8,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_decoupled_weight_decay=False,
                        grafting_config=AdamGraftingConfig(
                            beta2=0.999,
                            epsilon=1e-8,
                        ),
                    ),
                    device=device,
                )

    def test_adamw_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            with self.subTest(weight_decay=weight_decay, device=device):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        optim_factory, optim_cls=AdamW, eps=1e-8
                    ),
                    shampoo_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        epsilon=1e-8,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_decoupled_weight_decay=True,
                        grafting_config=AdamGraftingConfig(
                            beta2=0.999,
                            epsilon=1e-8,
                        ),
                    ),
                    device=device,
                )

    def test_rmsprop_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.01,
                weight_decay=weight_decay,
            )
            with self.subTest(weight_decay=weight_decay, device=device):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        optim_factory,
                        optim_cls=RMSprop,
                        alpha=0.99,
                        eps=1e-8,
                    ),
                    shampoo_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        betas=(0.0, 0.99),
                        epsilon=1e-8,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_decoupled_weight_decay=False,
                        grafting_config=RMSpropGraftingConfig(
                            beta2=0.99,
                            epsilon=1e-8,
                        ),
                    ),
                    device=device,
                )

    def test_sgd_grafting_on_quadratic(self) -> None:
        # Test all the combinations of with and without weight decay, with and without nesterov, and with CPU or GPU.
        for weight_decay, use_nesterov, device in product(
            (0.0, 0.3),
            (True, False),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.1,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay, use_nesterov=use_nesterov, device=device
            ):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        optim_factory,
                        optim_cls=SGD,
                        nesterov=use_nesterov,
                    ),
                    shampoo_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        betas=(0.0, 0.9),
                        epsilon=1e-10,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_nesterov=use_nesterov,
                        use_decoupled_weight_decay=False,
                        grafting_config=SGDGraftingConfig(),
                    ),
                    device=device,
                )
