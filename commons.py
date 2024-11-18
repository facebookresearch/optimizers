"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AbstractDataclass:
    def __new__(cls, *args: Any, **kwargs: Any) -> "AbstractDataclass":
        if cls == AbstractDataclass or AbstractDataclass in cls.__bases__:
            raise TypeError(f"Cannot instantiate abstract class: {cls.__name__}.")
        return super().__new__(cls)
