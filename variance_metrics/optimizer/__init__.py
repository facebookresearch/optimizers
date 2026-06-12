# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from typing import Generic, TypeVar

import torch.nn as nn
from components.configs import OptimizerConfig
from components.configurable import Configurable
from torch.optim import Optimizer

__all__ = ["OptimizersContainer"]


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Configurable, Generic[T]):
    """A wrapper around a single optimizer instantiated directly from a
    Python class and kwargs (``config.optimizer_cls`` /
    ``config.optimizer_kwargs``)."""

    Config = OptimizerConfig

    optimizer: T
    model: nn.Module

    def __init__(self, config: OptimizerConfig, *, model: nn.Module) -> None:
        assert config.optimizer_cls is not None, "optimizer.optimizer_cls must be set"
        self.model = model
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = config.optimizer_cls(params, **config.optimizer_kwargs)
        # Initialize Optimizer for hooks support.
        Optimizer.__init__(self, params, dict(config.optimizer_kwargs))

    def __iter__(self) -> Iterator[T]:
        return iter([self.optimizer])

    def __len__(self) -> int:
        return 1

    # pyrefly: ignore [bad-override]
    def step(self, *args, **kwargs) -> None:
        # pyrefly: ignore [missing-attribute]
        self.optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        # pyrefly: ignore [missing-attribute]
        self.optimizer.zero_grad(*args, **kwargs)
