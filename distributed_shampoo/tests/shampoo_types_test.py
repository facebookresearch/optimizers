"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest
from typing import Any
from unittest.mock import MagicMock

from distributed_shampoo.preconditioner.matrix_functions_types import (
    EighEigendecompositionConfig,
    PseudoInverseConfig,
)

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AmortizedPreconditionerConfig,
    DistributedConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    FSDPShampooConfig,
    HSDPShampooConfig,
    HybridShardShampooConfig,
    RMSpropGraftingConfig,
    ShampooPreconditionerConfig,
)

from distributed_shampoo.utils.commons import get_all_non_abstract_subclasses
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class AdaGradGraftingConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[AdaGradGraftingConfig]] = list(
        get_all_non_abstract_subclasses(AdaGradGraftingConfig)
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
    subclasses_types: list[type[RMSpropGraftingConfig]] = list(
        get_all_non_abstract_subclasses(RMSpropGraftingConfig)
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
class AmortizedPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[AmortizedPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(AmortizedPreconditionerConfig)  # type: ignore[type-abstract]
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_num_tolerated_failed_amortized_computations(
        self, cls: type[AmortizedPreconditionerConfig]
    ) -> None:
        num_tolerated_failed_amortized_computations = -1
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Invalid num_tolerated_failed_amortized_computations value: "
                f"{num_tolerated_failed_amortized_computations}. Must be >= 0."
            ),
            cls,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
        )


@instantiate_parametrized_tests
class ShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[ShampooPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(
            ShampooPreconditionerConfig,  # type: ignore[type-abstract]
        )
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_inverse_exponent_override(
        self, cls: type[ShampooPreconditionerConfig]
    ) -> None:
        non_positive_orders_config: dict[int, dict[int, float] | float] = {
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
        illegal_dimensions_config: dict[int, dict[int, float] | float] = {
            0: 0.2,
            1: {0: 0.3, 1: 0.2},
        }
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid dimensions in self.inverse_exponent_override[1]={illegal_dimensions_config[1]}: [1]. All dimensions must be within [0, 0]."
            ),
            cls,
            inverse_exponent_override=illegal_dimensions_config,
        )

        # non_positive_dim_overrides_config[1] is the problematic one.
        non_positive_dim_overrides_config: dict[int, dict[int, float] | float] = {
            1: {0: -0.3},
            2: 0.2,
        }
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid override value in self.inverse_exponent_override[1]={non_positive_dim_overrides_config[1]}: [-0.3]. All overrides must be >= 0."
            ),
            cls,
            inverse_exponent_override=non_positive_dim_overrides_config,
        )

        non_positive_universal_overrides_config: dict[int, dict[int, float] | float] = {
            1: -0.2,
            2: {0: 0.3, 1: 0.2},
        }
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid override value in self.inverse_exponent_override[1]={non_positive_universal_overrides_config[1]}: {non_positive_universal_overrides_config[1]}. All overrides must be >= 0."
            ),
            cls,
            inverse_exponent_override=non_positive_universal_overrides_config,
        )


@instantiate_parametrized_tests
class EigenvalueCorrectedShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[EigenvalueCorrectedShampooPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(EigenvalueCorrectedShampooPreconditionerConfig)
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

    @parametrize("cls", subclasses_types)
    def test_illegal_rank_deficient_stability_config(
        self, cls: type[EigenvalueCorrectedShampooPreconditionerConfig]
    ) -> None:
        invalid_amortized_computation_config = EighEigendecompositionConfig(
            rank_deficient_stability_config=PseudoInverseConfig(),
        )
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"PseudoInverseConfig is an invalid rank_deficient_stability_config for {cls.__name__}."
                " Please use an instance of PerturbationConfig instead."
            ),
            cls,
            amortized_computation_config=invalid_amortized_computation_config,
        )


@instantiate_parametrized_tests
class DistributedConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[DistributedConfig]] = list(
        get_all_non_abstract_subclasses(DistributedConfig)  # type: ignore[type-abstract]
    )

    @parametrize("cls", subclasses_types)
    @parametrize(
        "target_parameter_dimensionality, error_msg",
        [
            (-1, "Must be >= 1."),
            (0, "Must be >= 1."),
            (0.1, "Must be an integer or math.inf."),
        ],
    )
    def test_illegal_target_parameter_dimensionality(
        self,
        cls: type[DistributedConfig],
        target_parameter_dimensionality: int,
        error_msg: str,
    ) -> None:
        # Create required arguments for specific config classes.
        kwargs: dict[str, Any] = {
            "target_parameter_dimensionality": target_parameter_dimensionality
        }
        if cls in (FSDPShampooConfig, HSDPShampooConfig):
            kwargs["param_to_metadata"] = {}
        if cls in (HSDPShampooConfig, HybridShardShampooConfig):
            # Mock DeviceMesh to avoid distributed initialization.
            kwargs["device_mesh"] = MagicMock()

        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid self.{target_parameter_dimensionality=} value. {error_msg}"
            ),
            cls,
            **kwargs,
        )
