# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable nn.Module base + leaf-layer primitives.

- ``Module``: base class combining ``nn.Module`` with ``Configurable``.
- ``ModuleDict``: container that recursively initializes its children.
- ``Linear`` / ``Embedding`` / ``RMSNorm``: parameter-bearing leaf primitives
  wrapping their PyTorch counterparts with custom init.
"""

from dataclasses import dataclass

import torch.nn as nn
from components.configurable import Configurable


class Module(nn.Module, Configurable):
    """Base class for all configurable nn.Module components.

    Subclasses with learnable parameters should override ``init_weights``;
    the default is a no-op.
    """

    def init_weights(self, **kwargs) -> None:
        pass


def _container_init_weights(self: "Module", **kwargs) -> None:
    """``init_weights`` for container modules: recursively init each child."""
    for child in self.children():
        assert isinstance(child, Module)
        child.init_weights(**kwargs)


class ModuleDict(nn.ModuleDict, Module):
    """Module-protocol-compatible version of ``nn.ModuleDict``."""

    init_weights = _container_init_weights


class Linear(nn.Linear, Module):
    """Configurable nn.Linear with truncated-normal init."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config, *, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def init_weights(self, init_std: float = 0.02, **kwargs) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=init_std)


class Embedding(nn.Embedding, Module):
    """Configurable nn.Embedding with normal init."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config, *, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def init_weights(self, **kwargs) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0)


class RMSNorm(nn.RMSNorm, Module):
    """Configurable nn.RMSNorm."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        eps: float = 1e-5

    def __init__(self, config: Config, *, normalized_shape: int):
        super().__init__(normalized_shape, eps=config.eps)

    def init_weights(self, **kwargs) -> None:
        self.reset_parameters()
