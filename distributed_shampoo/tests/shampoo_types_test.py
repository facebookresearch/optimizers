"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

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
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class AdaGradGraftingConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[AdaGradGraftingConfig]] = get_all_subclasses(
        AdaGradGraftingConfig
    )

    @parametrize("epsilon", (0.0, -1.0))
    @parametrize("cls", subclasses_types)
    def test_illegal_epsilon(
        self, cls: type[AdaGradGraftingConfig], epsilon: float
    ) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Invalid epsilon value: {epsilon}. Must be > 0.0."),
            cls,
            epsilon=epsilon,
        )


@instantiate_parametrized_tests
class RMSpropGraftingConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[RMSpropGraftingConfig]] = get_all_subclasses(
        RMSpropGraftingConfig
    )

    @parametrize("beta2", (-1.0, 0.0, 1.3))
    @parametrize("cls", subclasses_types)
    def test_illegal_beta2(
        self, cls: type[RMSpropGraftingConfig], beta2: float
    ) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid grafting beta2 parameter: {beta2}. Must be in (0.0, 1.0]."
            ),
            cls,
            beta2=beta2,
        )


@instantiate_parametrized_tests
class PreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[PreconditionerConfig]] = get_all_subclasses(
        PreconditionerConfig, include_cls_self=False
    )

    # Not testing for the base class PreconditionerConfig because it is an abstract class.
    @parametrize("cls", subclasses_types)
    def test_illegal_num_tolerated_failed_amortized_computations(
        self, cls: type[PreconditionerConfig]
    ) -> None:
        num_tolerated_failed_amortized_computations = -1
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid num_tolerated_failed_amortized_computations value: "
                f"{num_tolerated_failed_amortized_computations}. Must be >= 0."
            ),
            cls,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
        )


@instantiate_parametrized_tests
class ShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[ShampooPreconditionerConfig]] = get_all_subclasses(
        ShampooPreconditionerConfig, include_cls_self=True
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_inverse_exponent_override(
        self, cls: type[ShampooPreconditionerConfig]
    ) -> None:
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


@instantiate_parametrized_tests
class EigenvalueCorrectedShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[EigenvalueCorrectedShampooPreconditionerConfig]] = (
        get_all_subclasses(
            EigenvalueCorrectedShampooPreconditionerConfig, include_cls_self=True
        )
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_ignored_basis_change_dims(
        self, cls: type[EigenvalueCorrectedShampooPreconditionerConfig]
    ) -> None:
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

    @parametrize("cls", subclasses_types)
    def test_illegal_inverse_exponent_override(
        self, cls: type[EigenvalueCorrectedShampooPreconditionerConfig]
    ) -> None:
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
