"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest
from itertools import product

import torch
from distributed_shampoo import (
    DistributedShampoo,
    ScheduleFreeConfig,
    SGDPreconditionerConfig,
    ShampooPT2CompileConfig,
)


class ScheduleFreeZeroLearningRateTest(unittest.TestCase):
    def test_iterates_match_weighted_average_with_zero_learning_rates(self) -> None:
        # Equations (3)-(5) and (23) of https://arxiv.org/abs/2405.15682.
        # Keep the three sequences explicit to independently check the fused update.
        for power, compile_step in product((0, 1, 2), (False, True)):
            for rates, grad_scale in (
                ((0.0, 0.0, 0.1, 0.0, 0.2, 0.05), 1.0),
                ((0.1, 0.2, 0.05), 1.0),
                ((0.0, 1e-10, 2e-10), 1e10),
            ):
                with self.subTest(power=power, rates=rates, compile_step=compile_step):
                    param = torch.nn.Parameter(
                        torch.tensor([1.0, -2.0], dtype=torch.float64)
                    )
                    optimizer = DistributedShampoo(
                        [param],
                        lr=rates[0],
                        betas=(0.0, 1.0),
                        preconditioner_config=SGDPreconditionerConfig(),
                        shampoo_pt2_compile_config=(
                            ShampooPT2CompileConfig(backend="eager")
                            if compile_step
                            else None
                        ),
                        iterate_averaging_config=ScheduleFreeConfig(
                            train_interp_coeff=0.9,
                            eval_coeff_lr_power=power,
                        ),
                    )
                    optimizer.train()
                    x = param.detach().clone()
                    z = x.clone()
                    weight_sum = 0.0
                    for index, lr in enumerate(rates):
                        grad = (
                            torch.tensor([index + 1.0, -0.5], dtype=param.dtype)
                            * grad_scale
                        )
                        optimizer.param_groups[0]["lr"] = lr
                        param.grad = grad.clone()
                        optimizer.step()

                        weight = lr**power
                        weight_sum += weight
                        average_weight = weight / weight_sum if weight_sum else 0.0
                        z = z - lr * grad
                        x = (1.0 - average_weight) * x + average_weight * z
                        y = 0.9 * x + 0.1 * z
                        torch.testing.assert_close(param, y, rtol=1e-6, atol=1e-7)
                        self.assertAlmostEqual(
                            optimizer.state[param]["lr_sum"].item(),
                            weight_sum,
                            delta=max(1e-30, weight_sum * 1e-6),
                        )
                        optimizer.eval()
                        torch.testing.assert_close(param, x, rtol=1e-6, atol=1e-7)
                        optimizer.train()
                        torch.testing.assert_close(param, y, rtol=1e-6, atol=1e-7)

    def test_checkpoint_after_zero_rate_steps_can_resume(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = DistributedShampoo(
            [param],
            lr=0.0,
            iterate_averaging_config=ScheduleFreeConfig(),
        )
        optimizer.train()
        for _ in range(2):
            param.grad = torch.tensor([2.0, -3.0])
            optimizer.step()
        torch.testing.assert_close(param, torch.tensor([1.0, -2.0]))
        checkpoint = optimizer.state_dict()
        restored_param = torch.nn.Parameter(param.detach().clone())
        restored_optimizer = DistributedShampoo(
            [restored_param],
            lr=0.0,
            iterate_averaging_config=ScheduleFreeConfig(),
        )
        restored_optimizer.load_state_dict(checkpoint)
        for candidate, candidate_param in (
            (optimizer, param),
            (restored_optimizer, restored_param),
        ):
            candidate.param_groups[0]["lr"] = 0.1
            candidate_param.grad = torch.tensor([1.0, -1.0])
            candidate.step()
            self.assertTrue(candidate_param.isfinite().all().item())
        torch.testing.assert_close(restored_param, param)
