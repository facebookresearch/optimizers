"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import re
import unittest
from functools import reduce
from operator import or_
from typing import TypeVar

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdaptiveAmortizedComputationFrequencyConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
    ShampooPreconditionerConfig,
)


SubclassesType = TypeVar("SubclassesType")


def get_all_subclasses(
    cls: SubclassesType, include_cls_self: bool = True
) -> list[SubclassesType]:
    """
    Retrieves all subclasses of a given class, optionally including the class itself.

    This function uses a helper function to recursively find all unique subclasses
    of the specified class.

    Args:
        cls (SubclassesType): The class for which to find subclasses.
        include_cls_self (bool): Whether to include the class itself in the result. (Default: True)

    Returns:
        list[SubclassesType]: A list of all unique subclasses of the given class.
    """

    def get_all_unique_subclasses(cls: SubclassesType) -> set[SubclassesType]:
        """Gets all unique subclasses of a given class recursively."""
        assert (
            subclasses := getattr(cls, "__subclasses__", lambda: None)()
        ) is not None, f"{cls} does not have __subclasses__."
        return reduce(or_, map(get_all_unique_subclasses, subclasses), {cls})

    return list(get_all_unique_subclasses(cls) - (set() if include_cls_self else {cls}))


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


class AdaptiveAmortizedComputationFrequencyConfigSubclassesTest(unittest.TestCase):
    def test_illegal_tolerance(self) -> None:
        tolerance = -1.0  # Negative tolerance is illegal.
        for cls in get_all_subclasses(AdaptiveAmortizedComputationFrequencyConfig):
            with self.subTest(cls=cls):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(f"Invalid tolerance value: {tolerance}. Must be >= 0.0."),
                    cls,
                    tolerance=tolerance,
                )


class ShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    def test_illegal_amortized_computation_frequency_config(self) -> None:
        for cls in get_all_subclasses(ShampooPreconditionerConfig):
            with self.subTest(cls=cls):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        "Invalid amortized_computation_frequency_config: AdaptiveAmortizedComputationFrequencyConfig(tolerance=0.0001). "
                        "Must be ConstantAmortizedComputationFrequencyConfig."
                    ),
                    cls,
                    amortized_computation_frequency_config=AdaptiveAmortizedComputationFrequencyConfig(
                        tolerance=1e-4
                    ),
                )
