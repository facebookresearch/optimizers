"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import itertools
import unittest
from collections.abc import Callable
from functools import partial

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    GraftingConfig,
    ShampooPT2CompileConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
)
from torch.optim.optimizer import ParamsT


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class DistributedShampooPytorchCompileTest(unittest.TestCase):
    @staticmethod
    def _shampoo_optim_factory(
        shampoo_pt2_compile_config: ShampooPT2CompileConfig | None,
        precondition_frequency: int,
        start_preconditioning_step: int,
        weight_decay: float,
        betas: tuple[float, float],
        grafting_config: GraftingConfig | None,
    ) -> Callable[[ParamsT], torch.optim.Optimizer]:
        return lambda parameters: DistributedShampoo(
            parameters,
            lr=0.01,
            betas=betas,
            # TODO: comment out beta3 to unblock quantization changes; need to fix PT2 FMA changes for this test
            # beta3=betas[0] * betas[0],
            epsilon=1e-10,
            momentum=0.9,
            dampening=0.9,
            weight_decay=weight_decay,
            max_preconditioner_dim=10,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            use_decoupled_weight_decay=True,
            shampoo_pt2_compile_config=shampoo_pt2_compile_config,
            grafting_config=grafting_config,
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
                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=shampoo_optim_factory(
                        shampoo_pt2_compile_config=None,
                    ),
                    experimental_optim_factory=shampoo_optim_factory(
                        shampoo_pt2_compile_config=ShampooPT2CompileConfig()
                    ),
                    device=torch.device("cuda"),
                    total_steps=total_steps,
                )

    def test_pt2_shampoo_after_preconditioning_on_quadratic(self) -> None:
        # NOTE: Test on steps after start_preconditioning_step.
        #       PT2 compilation with Inductor + root inverse introduces larger numerical differences
        #       compared to the non-PT2 baseline after preconditioning starts. However, these differences
        #       should NOT impact model quality.
        #       So we still want to add some numerical diff guardrails to prevent PT2 degradation.
        #       - It appears if torch.float16 precision tolerance is a good threshold:
        #       rtol = 1e-3; atol = 1e-5;
        #       - Test config specific: changing other Shampoo param vals can lead to UT failure:
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
                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=shampoo_optim_factory(
                        shampoo_pt2_compile_config=None,
                    ),
                    experimental_optim_factory=shampoo_optim_factory(
                        shampoo_pt2_compile_config=ShampooPT2CompileConfig()
                    ),
                    device=torch.device("cuda"),
                    total_steps=total_steps,
                    rtol=1.0e-3,
                    atol=1.0e-5,
                )
