"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

GPU convergence tests for GPAAdamW optimizer.

This module contains convergence tests that run on both CPU and GPU using
device parameterization. Tests verify:
    - Convergence on various optimization problems
    - CPU/GPU equivalence (implicit via parameterization)
    - Mode switching behavior during training

Running tests:
    python -m unittest gpa.gpu_tests.gpa_adamw_numerics_test -v
"""

import unittest
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from gpa.gpa_adamw import GPAAdamW
from gpa.tests.gpa_test_utils import (
    AVAILABLE_DEVICES,
    create_gpa_optimizer,
    create_schedulefree_optimizer,
)
from parameterized import parameterized


# =============================================================================
# LR Scheduler Factory Functions
# =============================================================================


def create_linear_lr_scheduler(
    optimizer: GPAAdamW, num_steps: int
) -> torch.optim.lr_scheduler.LinearLR:
    """Create a linear LR decay scheduler from lr to 0 over num_steps."""
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=num_steps,
    )


# =============================================================================
# Test Case Definitions (explicitly typed for Pyre)
# =============================================================================

# Type alias for LR scheduler factory
LRSchedulerFactory = Callable[[GPAAdamW, int], torch.optim.lr_scheduler.LRScheduler]

# Device and optimizer mode test cases for convergence tests
# (device, optimizer_factory, mode_name, lr_scheduler_factory)
_schedulefree_cases: list[
    tuple[
        torch.device,
        Callable[[Any], GPAAdamW],
        str,
        Optional[LRSchedulerFactory],
    ]
] = [
    (d, create_schedulefree_optimizer, "Schedule-Free", None) for d in AVAILABLE_DEVICES
]
_gpa_cases: list[
    tuple[
        torch.device,
        Callable[[Any], GPAAdamW],
        str,
        Optional[LRSchedulerFactory],
    ]
] = [
    (d, create_gpa_optimizer, "GPA", create_linear_lr_scheduler)
    for d in AVAILABLE_DEVICES
]
DEVICE_MODE_TEST_CASES: list[
    tuple[
        torch.device,
        Callable[[Any], GPAAdamW],
        str,
        Optional[LRSchedulerFactory],
    ]
] = _schedulefree_cases + _gpa_cases


# =============================================================================
# Convergence Tests
# =============================================================================


class GPAConvergenceTest(unittest.TestCase):
    """
    Convergence tests for GPAAdamW optimizer on CPU and GPU.

    These tests verify the optimizer can minimize various objective functions.
    Each test runs on both CPU and CUDA (when available) via parameterization.
    """

    def _test_convergence(
        self,
        model_loss_fn: Callable[[], torch.Tensor],
        params: Any,
        optimizer_factory: Callable[[Any], GPAAdamW],
        num_steps: int,
        device: torch.device,
        problem_name: str,
        final_param: Optional[nn.Parameter] = None,
        opt_param: Optional[torch.Tensor] = None,
        atol: float = 0.1,
        rtol: float = 0.1,
        loss_reduction_factor: float = 0.01,
        lr_scheduler_factory: Optional[LRSchedulerFactory] = None,
    ) -> None:
        """
        Helper function for testing convergence on an example problem.

        Args:
            model_loss_fn: Zero-argument closure that computes and returns loss.
            params: Parameters to pass to the optimizer factory.
            optimizer_factory: Factory function to create the optimizer.
            num_steps: Number of optimization steps.
            device: Device to run on.
            problem_name: Name of the problem for error messages.
            final_param: The parameter to check against opt_param (if provided).
            opt_param: Optional optimal parameter values (if known).
            atol: Absolute tolerance for parameter comparison.
            rtol: Relative tolerance for parameter comparison.
            loss_reduction_factor: Expected loss reduction ratio (final/init).
            lr_scheduler_factory: Optional factory to create an LR scheduler.
        """
        optimizer = optimizer_factory(params)
        scheduler = (
            lr_scheduler_factory(optimizer, num_steps)
            if lr_scheduler_factory is not None
            else None
        )

        optimizer.train()
        init_loss: float = float(model_loss_fn().item())

        for _ in range(num_steps):
            optimizer.zero_grad()
            loss = model_loss_fn()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        optimizer.eval()
        final_loss: float = float(model_loss_fn().item())

        if opt_param is not None and final_param is not None:
            # Compute min_loss at the optimal parameters
            saved_data = final_param.data.clone()
            final_param.data = opt_param
            min_loss: float = float(model_loss_fn().item())
            final_param.data = saved_data

            torch.testing.assert_close(
                final_param,
                opt_param,
                atol=atol,
                rtol=rtol,
                msg=f"{problem_name} optimization did not converge on {device} "
                f"with final_param={final_param} and opt_param={opt_param}!",
            )
            self.assertLess(
                final_loss - min_loss,
                atol + rtol * abs(min_loss),
                f"{problem_name} optimization did not converge on {device} "
                f"with final_loss={final_loss:.6f} and min_loss={min_loss:.6f}!",
            )
        else:
            self.assertLess(
                final_loss,
                init_loss * loss_reduction_factor,
                f"{problem_name} optimization did not converge on {device} "
                f"with final_loss={final_loss:.6f} and init_loss={init_loss:.6f}!",
            )

    @parameterized.expand(DEVICE_MODE_TEST_CASES)
    def test_quadratic_convergence(
        self,
        device: torch.device,
        optimizer_factory: Callable[[Any], GPAAdamW],
        mode_name: str,
        lr_scheduler_factory: Optional[LRSchedulerFactory],
    ) -> None:
        """
        Test convergence on a convex quadratic: f(x) = 0.5 * x^T A x - b^T x.

        The optimizer should find x* = A^{-1} b.
        """
        torch.manual_seed(42)
        d = 10
        L: torch.Tensor = torch.randn(d, d, device=device)
        A: torch.Tensor = L @ L.T + 1.0 * torch.eye(d, device=device)
        b: torch.Tensor = torch.randn(d, device=device)
        param = nn.Parameter(torch.randn(d, device=device))
        optimal_param = torch.linalg.solve(A, b)

        self._test_convergence(
            model_loss_fn=lambda: 0.5 * param @ A @ param - b @ param,
            params=[param],
            optimizer_factory=optimizer_factory,
            num_steps=1000,
            device=device,
            problem_name=f"Quadratic ({mode_name})",
            final_param=param,
            opt_param=optimal_param,
            atol=0.1,
            rtol=0.1,
            lr_scheduler_factory=lr_scheduler_factory,
        )

    @parameterized.expand(DEVICE_MODE_TEST_CASES)
    def test_linear_regression_convergence(
        self,
        device: torch.device,
        optimizer_factory: Callable[[Any], GPAAdamW],
        mode_name: str,
        lr_scheduler_factory: Optional[LRSchedulerFactory],
    ) -> None:
        """Test convergence on linear regression: min_w ||Xw - y||^2."""
        torch.manual_seed(42)
        n_samples, n_features = 100, 10
        X: torch.Tensor = torch.randn(n_samples, n_features, device=device)
        true_w = torch.randn(n_features, device=device)
        y: torch.Tensor = X @ true_w + 0.1 * torch.randn(n_samples, device=device)
        param = nn.Parameter(torch.zeros(n_features, device=device))
        optimal_param = torch.linalg.lstsq(X, y).solution

        self._test_convergence(
            model_loss_fn=lambda: ((X @ param - y).pow(2)).mean(),
            params=[param],
            optimizer_factory=optimizer_factory,
            num_steps=1000,
            device=device,
            problem_name=f"Linear regression ({mode_name})",
            final_param=param,
            opt_param=optimal_param,
            atol=0.1,
            rtol=0.1,
            lr_scheduler_factory=lr_scheduler_factory,
        )

    @parameterized.expand(DEVICE_MODE_TEST_CASES)
    def test_mlp_convergence(
        self,
        device: torch.device,
        optimizer_factory: Callable[[Any], GPAAdamW],
        mode_name: str,
        lr_scheduler_factory: Optional[LRSchedulerFactory],
    ) -> None:
        """Test MLP convergence on a simple regression task (no optimal solution)."""
        torch.manual_seed(42)

        X = torch.randn(100, 10, device=device)
        y = torch.randn(100, 1, device=device)

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        ).to(device)

        self._test_convergence(
            model_loss_fn=lambda: ((model(X) - y).pow(2)).mean(),
            params=model.parameters(),
            optimizer_factory=optimizer_factory,
            num_steps=1000,
            device=device,
            problem_name=f"MLP ({mode_name})",
            loss_reduction_factor=0.1,
            lr_scheduler_factory=lr_scheduler_factory,
        )


if __name__ == "__main__":
    unittest.main()
