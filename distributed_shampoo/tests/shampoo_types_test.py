"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import re
import unittest

from commons import get_all_subclasses

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
)


class AdaGradGraftingConfigSubclassesTest(unittest.TestCase):
    def test_illegal_epsilon(self) -> None:
        epsilon = 0.0
        for cls in get_all_subclasses(AdaGradGraftingConfig):
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
            get_all_subclasses(RMSpropGraftingConfig),
            (-1.0, 0.0, 1.3),
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
        # Not testing for the base class PreconditionerConfig because it is an abstract class.
        for cls in get_all_subclasses(PreconditionerConfig, include_cls_self=False):
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

    def test_illegal_ignored_dims(self) -> None:
        ignored_dims = [1, 2, 3, 1]
        # Not testing for the base class PreconditionerConfig because it is an abstract class.
        for cls in get_all_subclasses(PreconditionerConfig, include_cls_self=False):
            with self.subTest(cls=cls):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid ignored_dims value: {ignored_dims}. Must be a list of unique dimensions."
                    ),
                    cls,
                    ignored_dims=ignored_dims,
                )
