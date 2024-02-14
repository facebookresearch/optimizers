"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest
from typing import Type, Union

from distributed_shampoo.shampoo_types import (
    AbstractDataclass,
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    DistributedConfig,
    GraftingConfig,
    RMSpropGraftingConfig,
)


class InvalidAbstractDataclassInitTest(unittest.TestCase):
    def test_invalid_init(self) -> None:
        for abstract_cls in (AbstractDataclass, DistributedConfig, GraftingConfig):
            with self.subTest(abstract_cls=abstract_cls), self.assertRaisesRegex(
                TypeError,
                re.escape(
                    f"Cannot instantiate abstract class: {abstract_cls.__name__}."
                ),
            ):
                abstract_cls()


class AdaGradGraftingConfigTest(unittest.TestCase):
    def test_illegal_epsilon(self) -> None:
        epsilon = 0.0
        grafting_config_type = self._get_grafting_config_type()
        with self.subTest(
            grafting_config_type=grafting_config_type
        ), self.assertRaisesRegex(
            ValueError,
            re.escape(f"Invalid epsilon value: {epsilon}. Must be > 0.0."),
        ):
            grafting_config_type(epsilon=epsilon)

    def _get_grafting_config_type(
        self,
    ) -> Union[
        Type[AdaGradGraftingConfig],
        Type[RMSpropGraftingConfig],
        Type[AdamGraftingConfig],
    ]:
        return AdaGradGraftingConfig


class RMSpropGraftingConfigTest(AdaGradGraftingConfigTest):
    def test_illegal_beta2(
        self,
    ) -> None:
        grafting_config_type = self._get_grafting_config_type()
        for beta2 in (-1.0, 0.0, 1.3):
            with self.subTest(
                grafting_config_type=grafting_config_type, beta2=beta2
            ), self.assertRaisesRegex(
                ValueError,
                re.escape(
                    f"Invalid grafting beta2 parameter: {beta2}. Must be in (0.0, 1.0]."
                ),
            ):
                grafting_config_type(beta2=beta2)

    def _get_grafting_config_type(
        self,
    ) -> Union[Type[RMSpropGraftingConfig], Type[AdamGraftingConfig]]:
        return RMSpropGraftingConfig


class AdamGraftingConfigTest(RMSpropGraftingConfigTest):
    def _get_grafting_config_type(
        self,
    ) -> Union[Type[RMSpropGraftingConfig], Type[AdamGraftingConfig]]:
        return AdamGraftingConfig
