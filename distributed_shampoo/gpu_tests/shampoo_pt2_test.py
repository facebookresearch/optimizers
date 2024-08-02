"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3


import itertools
import unittest
from functools import partial
from typing import Callable, Iterable, Optional, Tuple

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdaGradGraftingConfig, GraftingConfig
from torch import nn
from torch.nn.parameter import Parameter


class DistributedShampooPytorchCompileTest(unittest.TestCase):
    @staticmethod
    def _construct_quadratic() -> (
        Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]
    ):
        data = torch.arange(10, dtype=torch.float, device=torch.device("cuda"))
        data /= torch.norm(data)
        model = nn.Sequential(
            nn.Linear(10, 1, bias=False),
            nn.Linear(1, 1, bias=False),
        ).to(device=torch.device("cuda"))
        model[0].weight.data.fill_(1.0)
        model[1].weight.data.fill_(1.0)
        model[1].requires_grad = False
        loss = nn.MSELoss()
        target = torch.tensor([0.0]).to(device=torch.device("cuda"))
        return model, loss, data, target

    @staticmethod
    def _train_quadratic(
        shampoo_optim_factory: Callable[[Iterable[Parameter]], torch.optim.Optimizer],
        total_steps: int = 5,
    ) -> Tuple[Parameter, torch.Tensor]:
        (
            model,
            loss,
            data,
            target,
        ) = DistributedShampooPytorchCompileTest._construct_quadratic()
        params = model.parameters()
        optimizer = shampoo_optim_factory(params)
        for _ in range(total_steps):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()
        return model[0].weight.data.cpu(), objective.detach().cpu()

    @staticmethod
    def _test_shampoo_baseline_and_pt2(
        baseline_optim_factory: Callable[[Iterable[Parameter]], torch.optim.Optimizer],
        pt2_optim_factory: Callable[[Iterable[Parameter]], torch.optim.Optimizer],
        total_steps: int = 5,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> None:
        (
            baseline_params,
            baseline_loss,
        ) = DistributedShampooPytorchCompileTest._train_quadratic(
            baseline_optim_factory,
            total_steps=total_steps,
        )
        pt2_params, pt2_loss = DistributedShampooPytorchCompileTest._train_quadratic(
            pt2_optim_factory,
            total_steps=total_steps,
        )
        torch.testing.assert_close(
            pt2_loss,
            baseline_loss,
            rtol=rtol,
            atol=atol,
        )
        torch.testing.assert_close(
            pt2_params,
            baseline_params,
            rtol=rtol,
            atol=atol,
        )

    @staticmethod
    def _shampoo_optim_factory(
        use_pytorch_compile: bool,
        precondition_frequency: int,
        start_preconditioning_step: int,
        weight_decay: float,
        betas: Tuple[float, float],
        grafting_config: GraftingConfig,
    ) -> Callable[[Iterable[Parameter]], torch.optim.Optimizer]:
        return lambda parameters: DistributedShampoo(
            parameters,
            lr=0.01,
            betas=betas,
            beta3=betas[0] * betas[0],
            epsilon=1e-10,
            momentum=0.9,
            dampening=0.9,
            weight_decay=weight_decay,
            max_preconditioner_dim=10,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            use_decoupled_weight_decay=True,
            use_pytorch_compile=use_pytorch_compile,
            grafting_config=grafting_config,
            pytorch_compile_backend="inductor",
        )

    def test_pt2_shampoo_before_preconditioning_on_quadratic(self) -> None:
        # Test all the combinations for weight decay, precondition frequency,
        # starting preconditioning step, and total steps.
        # Test on steps before start_preconditioning_step
        for (
            weight_decay,
            betas,
            grafting_config,
            (
                precondition_frequency,
                start_preconditioning_step,
                total_steps,
            ),
        ) in itertools.product(
            (0.0, 0.1),
            ((0.0, 1.0), (0.9, 0.999)),
            (
                None,
                AdaGradGraftingConfig(
                    epsilon=1e-10,
                ),
            ),
            ((1, 1000, 5), (10, 10, 5)),
        ):
            shampoo_optim_factory = partial(
                DistributedShampooPytorchCompileTest._shampoo_optim_factory,
                precondition_frequency=precondition_frequency,
                start_preconditioning_step=start_preconditioning_step,
                weight_decay=weight_decay,
                betas=betas,
                grafting_config=grafting_config,
            )
            with self.subTest(
                weight_decay=weight_decay,
                precondition_frequency=precondition_frequency,
                start_preconditioning_step=start_preconditioning_step,
                total_steps=total_steps,
                betas=betas,
                grafting_config=grafting_config,
            ):
                DistributedShampooPytorchCompileTest._test_shampoo_baseline_and_pt2(
                    baseline_optim_factory=shampoo_optim_factory(
                        use_pytorch_compile=False
                    ),
                    pt2_optim_factory=shampoo_optim_factory(use_pytorch_compile=True),
                    total_steps=total_steps,
                )

    def test_pt2_shampoo_after_preconditioning_on_quadratic(self) -> None:
        # NOTE: Test on steps after start_preconditioning_step
        #       b/c of PT2 compile with inductor + root inverse after starting preconditioning
        #       the numerical differences between non-pt2 baseline and pt2 has a bigger
        #       precision gap; however, this num diff would NOT cause NEX for (ads) model training.
        #       So we still want to add some numerical diff guardrails to prevent PT2 degradation.
        #       - It appears if torch.float16 precision tolerance is a good threshold:
        #       rtol = 1e-3; atol = 1e-5;
        #       - Test config specifc: changing other Shampoo param vals can lead to UT failure:
        #       e.g., increase total_steps to a big val (e.g., 10000)

        # Test all the combinations for weight decay, betas, grafting_config,
        # precondition frequency, starting preconditioning step, and total steps.
        for (
            weight_decay,
            betas,
            grafting_config,
            (
                precondition_frequency,
                start_preconditioning_step,
                total_steps,
            ),
        ) in itertools.product(
            (0.1,),
            ((0.9, 0.999),),
            (
                AdaGradGraftingConfig(
                    epsilon=1e-10,
                ),
            ),
            (
                (10, 100, 110),
                (10, 10, 20),
            ),
        ):
            shampoo_optim_factory = partial(
                DistributedShampooPytorchCompileTest._shampoo_optim_factory,
                precondition_frequency=precondition_frequency,
                start_preconditioning_step=start_preconditioning_step,
                weight_decay=weight_decay,
                betas=betas,
                grafting_config=grafting_config,
            )
            with self.subTest(
                weight_decay=weight_decay,
                precondition_frequency=precondition_frequency,
                start_preconditioning_step=start_preconditioning_step,
                total_steps=total_steps,
                betas=betas,
                grafting_config=grafting_config,
            ):
                DistributedShampooPytorchCompileTest._test_shampoo_baseline_and_pt2(
                    baseline_optim_factory=shampoo_optim_factory(
                        use_pytorch_compile=False
                    ),
                    pt2_optim_factory=shampoo_optim_factory(use_pytorch_compile=True),
                    total_steps=total_steps,
                    rtol=1.0e-3,
                    atol=1.0e-5,
                )
