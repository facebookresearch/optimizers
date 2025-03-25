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
    EigenvalueCorrectedShampooPreconditionerConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
    ShampooPreconditionerConfig,
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


class ShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    def test_illegal_inverse_exponent_override(self) -> None:
        for cls in get_all_subclasses(
            ShampooPreconditionerConfig, include_cls_self=True
        ):
            with self.subTest(cls=cls):
                non_positive_orders_config: dict[int, dict[int, float]] = {
                    -1: {},
                    -2: {},
                }
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid orders in self.inverse_exponent_override={non_positive_orders_config}: [-1, -2]. All orders must be >= 0."
                    ),
                    cls,
                    inverse_exponent_override=non_positive_orders_config,
                )

                # illegal_dimensions_config[1] is the problematic one.
                illegal_dimensions_config: dict[int, dict[int, float]] = {
                    0: {0: 0.2},
                    1: {0: 0.3, 1: 0.2},
                }
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid dimensions in self.inverse_exponent_override[order]={illegal_dimensions_config[1]}: [1]. All dimensions must be within [0, 0]."
                    ),
                    cls,
                    inverse_exponent_override=illegal_dimensions_config,
                )

                # non_positive_overrides_config[1] is the problematic one.
                non_positive_overrides_config: dict[int, dict[int, float]] = {
                    1: {0: -0.3},
                    2: {0: 0.2, 1: 0.5},
                }
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid override value in self.inverse_exponent_override[order]={non_positive_overrides_config[1]}: [-0.3]. All overrides must be >= 0."
                    ),
                    cls,
                    inverse_exponent_override=non_positive_overrides_config,
                )


class EigenvalueCorrectedShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    def test_illegal_ignored_basis_change_dims(self) -> None:
        for cls in get_all_subclasses(
            EigenvalueCorrectedShampooPreconditionerConfig, include_cls_self=True
        ):
            with self.subTest(cls=cls):
                non_positive_orders_config: dict[int, list[int]] = {-1: [0], -2: [0, 1]}
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid orders in self.ignored_basis_change_dims={non_positive_orders_config}: [-1, -2]. All orders must be >= 0."
                    ),
                    cls,
                    ignored_basis_change_dims=non_positive_orders_config,
                )

                # illegal_dimensions_config[1] is the problematic one.
                illegal_ignored_dimensions_config: dict[int, list[int]] = {
                    0: [0],
                    1: [0, 1],
                }
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid dimensions in self.ignored_basis_change_dims[order]={illegal_ignored_dimensions_config[1]}: [1]. All dimensions must be within [0, 0]."
                    ),
                    cls,
                    ignored_basis_change_dims=illegal_ignored_dimensions_config,
                )

                # duplicate_ignored_basis_change_dims_config[1] is the problematic one.
                duplicate_ignored_basis_change_dims_config: dict[int, list[int]] = {
                    0: [0],
                    1: [0, 0],
                }
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid ignored dimensions in self.ignored_basis_change_dims[order]={duplicate_ignored_basis_change_dims_config[1]}. Duplicate dimensions found in {duplicate_ignored_basis_change_dims_config[1]}. All dimensions must be unique."
                    ),
                    cls,
                    ignored_basis_change_dims=duplicate_ignored_basis_change_dims_config,
                )

    def test_illegal_inverse_exponent_override(self) -> None:
        for cls in get_all_subclasses(
            EigenvalueCorrectedShampooPreconditionerConfig, include_cls_self=True
        ):
            with self.subTest(cls=cls):
                non_positive_orders_config: dict[int, float] = {-1: 0.5, -2: 0.2}
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid orders in self.inverse_exponent_override={non_positive_orders_config}: [-1, -2]. All orders must be >= 0."
                    ),
                    cls,
                    inverse_exponent_override=non_positive_orders_config,
                )

                # non_positive_overrides_config[1] is the problematic one.
                non_positive_overrides_config: dict[int, float] = {1: 0.0, 2: 0.2}
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid override value in self.inverse_exponent_override[order]={non_positive_overrides_config[1]}: 0.0. All overrides must be > 0."
                    ),
                    cls,
                    inverse_exponent_override=non_positive_overrides_config,
                )

                # negative_overrides_config[2] is the problematic one.
                negative_overrides_config: dict[int, float] = {1: 0.2, 2: -0.2}
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid override value in self.inverse_exponent_override[order]={negative_overrides_config[2]}: -0.2. All overrides must be > 0."
                    ),
                    cls,
                    inverse_exponent_override=negative_overrides_config,
                )
