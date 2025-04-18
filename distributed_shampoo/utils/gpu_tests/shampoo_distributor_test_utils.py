"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

from collections.abc import Callable
from functools import partial
from typing import TypeVar

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdaGradGraftingConfig, DistributedConfig

from torch import nn
from torch.optim.optimizer import ParamsT

# Type variable for model factory return types
ModelFactoryReturnT = TypeVar(
    "ModelFactoryReturnT",
    tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor],
    tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, bool],
)

# Type variable for train model return types
TrainModelReturnT = TypeVar(
    "TrainModelReturnT",
    tuple[list[torch.Tensor], torch.Tensor],
    tuple[list[torch.Tensor], torch.Tensor, torch.Tensor],
)


def shampoo_optim_factory(
    distributed_config: DistributedConfig | None,
) -> Callable[
    [ParamsT],
    torch.optim.Optimizer,
]:
    """
    Args:
        distributed_config: Configuration for distributed Shampoo.

    Returns:
        A callable that creates a DistributedShampoo optimizer.
    """
    return partial(
        DistributedShampoo,
        lr=0.001,
        betas=(0.9, 1.0),
        epsilon=1e-8,
        momentum=0.0,
        weight_decay=0.0,
        max_preconditioner_dim=4,
        precondition_frequency=1,
        start_preconditioning_step=2,
        use_decoupled_weight_decay=True,
        grafting_config=AdaGradGraftingConfig(
            epsilon=1e-8,
        ),
        distributed_config=distributed_config,
    )


def test_two_configs(
    train_model_func: Callable[
        [
            Callable[[ParamsT], torch.optim.Optimizer],
            Callable[[torch.device], ModelFactoryReturnT],
            torch.device,
        ],
        TrainModelReturnT,
    ],
    optim_factory1: Callable[
        [ParamsT],
        torch.optim.Optimizer,
    ],
    model_factory1: Callable[
        [torch.device],
        ModelFactoryReturnT,
    ],
    optim_factory2: Callable[
        [ParamsT],
        torch.optim.Optimizer,
    ],
    model_factory2: Callable[
        [torch.device],
        ModelFactoryReturnT,
    ],
    device: torch.device,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    """
    Common function to test two different configurations and verify they produce equivalent results.

    Args:
        train_model_func: Function to train the model and return parameters and loss.
        optim_factory1: First optimizer factory.
        model_factory1: First model factory.
        optim_factory2: Second optimizer factory.
        model_factory2: Second model factory.
        device: Device to run the test on.
        atol: Absolute tolerance for tensor comparison.
        rtol: Relative tolerance for tensor comparison.
    """
    params1, loss1, *_ = train_model_func(
        optim_factory1,
        model_factory1,
        device,
    )
    params2, loss2, *_ = train_model_func(
        optim_factory2,
        model_factory2,
        device,
    )

    torch.testing.assert_close(loss1, loss2, atol=atol, rtol=rtol)

    # Handle different parameter structures
    if isinstance(params1, list) and isinstance(params2, list):
        if len(params1) == len(params2):
            # Compare all parameters
            torch.testing.assert_close(params1, params2)
        else:
            # For cases like ShampooFullyShardDistributorTest where only specific parameters are compared
            # This is a simplified approach - specific test classes might need to override this behavior
            for p1, p2 in zip(params1, params2):
                torch.testing.assert_close(p1, p2)
