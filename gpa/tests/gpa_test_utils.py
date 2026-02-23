"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared test utilities for GPA optimizer tests.

This module provides common helper functions used across the GPA optimizer
test suite.

Usage:
    from gpa.tests.gpa_test_utils import (
        create_simple_model,
        set_deterministic_gradients,
    )
"""

from typing import Any

import torch
import torch.nn as nn
from gpa.gpa_adamw import GPAAdamW
from gpa.gpa_types import IterateAveragingType


# =============================================================================
# Device Utilities
# =============================================================================


def get_available_devices() -> tuple[torch.device, ...]:
    """
    Get tuple of available devices for parameterized testing.

    Returns CPU always, and CUDA if available.

    Returns:
        Tuple of torch.device objects.
    """
    devices = (torch.device("cpu"),)
    if torch.cuda.is_available():
        devices = devices + (torch.device("cuda"),)
    return devices


# Pre-computed for use in test classes
AVAILABLE_DEVICES: tuple[torch.device, ...] = get_available_devices()


# =============================================================================
# Model Creation Utilities
# =============================================================================


def create_simple_model(
    input_dim: int = 10,
    output_dim: int = 5,
    bias: bool = False,
) -> nn.Module:
    """
    Create a simple linear model for testing.

    This creates a minimal model suitable for unit testing optimizer behavior.
    The model has no bias by default to simplify gradient verification.

    Args:
        input_dim: Input dimension for the linear layer.
        output_dim: Output dimension for the linear layer.
        bias: Whether to include bias in the linear layer.

    Returns:
        A simple nn.Linear model.
    """
    return nn.Linear(input_dim, output_dim, bias=bias)


# =============================================================================
# Gradient Utilities
# =============================================================================


def set_deterministic_gradients(
    model: nn.Module,
    seed: int = 42,
) -> None:
    """
    Set deterministic gradients on all model parameters.

    This utility sets reproducible gradient values for testing,
    allowing verification of optimizer update equations.

    Args:
        model: Model to set gradients on.
        seed: Random seed for gradient generation.
    """
    torch.manual_seed(seed)
    for param in model.parameters():
        param.grad = torch.randn_like(param)


# =============================================================================
# Optimizer Factory Functions
# =============================================================================


def create_schedulefree_optimizer(params: Any) -> GPAAdamW:
    """Factory function for creating GPAAdamW in Schedule-Free mode."""
    return GPAAdamW(
        params,
        lr=0.1,
        train_interp_coeff=0.7,
        eval_interp_coeff=0.0,
        iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
    )


def create_gpa_optimizer(params: Any) -> GPAAdamW:
    """Factory function for creating GPAAdamW in standard GPA mode."""
    return GPAAdamW(
        params,
        lr=0.1,
        train_interp_coeff=0.7,
        eval_interp_coeff=0.9967,
        iterate_averaging_type=IterateAveragingType.GPA,
    )


# =============================================================================
# Optimizer Step Utilities
# =============================================================================


def run_optimizer_steps(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    gradients: list[torch.Tensor],
) -> None:
    """Run optimizer for len(gradients) steps with pre-generated gradients.

    Args:
        optimizer: The optimizer to step.
        model: The model whose parameters receive the gradients.
        gradients: Pre-generated gradient tensors, one per step.
    """
    for grad in gradients:
        optimizer.zero_grad()
        for param in model.parameters():
            param.grad = grad.clone()
        optimizer.step()
