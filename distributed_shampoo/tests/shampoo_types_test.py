"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import re
import unittest

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
)


class AdaGradGraftingConfigSubclassesTest(unittest.TestCase):
    def test_illegal_epsilon(self) -> None:
        epsilon = 0.0
        for cls in AdaGradGraftingConfig.__subclasses__():
            with self.subTest(cls=cls):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(f"Invalid epsilon value: {epsilon}. Must be > 0.0."),
                    cls,
                    epsilon=epsilon,
                )


class RMSpropGraftingConfigSubclassesTest(AdaGradGraftingConfigSubclassesTest):
    def test_illegal_beta2(
        self,
    ) -> None:
        for cls, beta2 in itertools.product(
            RMSpropGraftingConfig.__subclasses__(), (-1.0, 0.0, 1.3)
        ):
            with self.subTest(cls=cls, beta2=beta2):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid grafting beta2 parameter: {beta2}. Must be in (0.0, 1.0]."
                    ),
                    cls,
                    beta2=beta2,
                )


class PreconditionerConfigSubclassesTest(unittest.TestCase):
    def test_illegal_num_tolerated_failed_amortized_computations(self) -> None:
        num_tolerated_failed_amortized_computations = -1
        for cls in PreconditionerConfig.__subclasses__():
            with self.subTest(cls=cls):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid num_tolerated_failed_amortized_computations value: "
                        f"{num_tolerated_failed_amortized_computations}. Must be >= 0."
                    ),
                    cls,
                    num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
                )
