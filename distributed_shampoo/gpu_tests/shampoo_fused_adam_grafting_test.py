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
from typing import Any, Callable, Iterable, Tuple, Type

import torch
from caffe2.torch.fb.apex.optimizers import FusedAdam
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    RMSpropGraftingConfig,
)
from torch import nn
from torch.nn.parameter import Parameter


class DistributedShampooGraftingTest(unittest.TestCase):
    @staticmethod
    def _construct_quadratic(
        device: torch.device,
    ) -> Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        data = torch.arange(10, dtype=torch.float, device=device)
        data /= torch.norm(data)
        model = nn.Sequential(
            nn.Linear(10, 1, bias=False),
            nn.Linear(1, 1, bias=False),
        ).to(device=device)
        model[0].weight.data.fill_(1.0)
        model[1].weight.data.fill_(1.0)
        loss = nn.MSELoss()
        target = torch.tensor([0.0]).to(device=device)
        return model, loss, data, target

    @staticmethod
    def _train_quadratic(
        optim_factory: Callable[
            [Iterable[Parameter]],
            torch.optim.Optimizer,
        ],
        device: torch.device,
    ) -> Tuple[Parameter, torch.Tensor]:
        model, loss, data, target = DistributedShampooGraftingTest._construct_quadratic(
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
    def _test_baseline_and_shampoo(
        baseline_optim_factory: Callable[
            [Iterable[Parameter]],
            torch.optim.Optimizer,
        ],
        shampoo_optim_factory: Callable[
            [Iterable[Parameter]],
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
        parameters: Iterable[Parameter],
        optim_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        return optim_cls(parameters, **kwargs)

    def test_amsgradv3_with_beta2_equal_1_grafting_on_quadratic(self) -> None:
        # Test all the combinations of with and without weight decay, and use and don't use bias correction.
        for weight_decay, use_bias_correction in product((0.0, 0.3), (True, False)):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.001,
                betas=(0.9, 1.0),
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay, use_bias_correction=use_bias_correction
            ):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        optim_factory,
                        optim_cls=FusedAdam,
                        bias_correction=use_bias_correction,
                        eps=1e-8,
                        adam_w_mode=True,
                        amsgrad=False,
                        set_grad_none=True,
                        amsgrad_v2=False,
                        amsgrad_v3=True,
                        nesterov=False,
                    ),
                    # NOTE: AMSGrad (v3) with beta2 = 1.0 is equivalent to AdaGrad with
                    # decoupled weight decay and exponential moving averaging of the gradient.
                    shampoo_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        epsilon=1e-8,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_bias_correction=use_bias_correction,
                        use_decoupled_weight_decay=True,
                        grafting_config=AdaGradGraftingConfig(
                            epsilon=1e-8,
                        ),
                    ),
                    device=torch.device("cuda"),
                )

    def test_amsgradv3_with_beta2_less_1_grafting_on_quadratic(self) -> None:
        # Test all the combinations of with and without weight decay, and use and don't use bias correction.
        for weight_decay, use_bias_correction in product((0.0, 0.3), (True, False)):
            with self.subTest(
                weight_decay=weight_decay, use_bias_correction=use_bias_correction
            ):
                DistributedShampooGraftingTest._test_baseline_and_shampoo(
                    baseline_optim_factory=partial(
                        DistributedShampooGraftingTest._optim_factory,
                        optim_cls=FusedAdam,
                        lr=0.001,
                        bias_correction=use_bias_correction,
                        betas=(0.9, 0.99),
                        eps=1e-8,
                        adam_w_mode=True,
                        weight_decay=weight_decay,
                        amsgrad=False,
                        set_grad_none=True,
                        amsgrad_v2=False,
                        amsgrad_v3=True,
                        nesterov=False,
                    ),
                    # More details on the how Shampoo grafting setting is derived could be found
                    # in https://fb.workplace.com/notes/291113153574033.
                    shampoo_optim_factory=partial(
                        DistributedShampooGraftingTest._optim_factory,
                        optim_cls=DistributedShampoo,
                        lr=0.0001,  # lr has to be multiplied by sqrt(1 - beta2)
                        betas=(0.9, 0.99),
                        epsilon=1e-9,  # epsilon has to be multiplied by sqrt(1 - beta2)
                        momentum=0.0,
                        weight_decay=10
                        * weight_decay,  # weight decay has to be divided by sqrt(1 - beta2)
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=1000,
                        use_bias_correction=use_bias_correction,
                        use_decoupled_weight_decay=True,
                        grafting_config=RMSpropGraftingConfig(
                            beta2=0.99,
                            epsilon=1e-9,
                        ),
                    ),
                    device=torch.device("cuda"),
                )
