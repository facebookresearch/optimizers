# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import ClassVar


class Configurable:
    """Base class for configurable components.

    Each subclass defines a nested ``Config`` (a kw-only dataclass inheriting
    from ``Configurable.Config``) and an ``__init__(self, config: Config,
    **runtime_kwargs)``. ``some_config.build(**runtime_kwargs)`` is sugar
    for ``OwningClass(config=some_config, **runtime_kwargs)``; the owner is
    auto-wired by ``__init_subclass__``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        _owner: ClassVar[type | None] = None

        def build(self, **kwargs):
            assert self._owner is not None, (
                f"{type(self).__name__} has no _owner — "
                "define Config inside a Configurable subclass."
            )
            return self._owner(config=self, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        config_cls = cls.__dict__.get("Config")
        if config_cls is not None and issubclass(config_cls, Configurable.Config):
            config_cls._owner = cls
