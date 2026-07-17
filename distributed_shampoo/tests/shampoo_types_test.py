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

import torch
from distributed_shampoo.preconditioner.matrix_functions_types import (
    EighEigendecompositionConfig,
    PseudoInverseConfig,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    BaseShampooPreconditionerConfig,
    ClassicMomentumConfig,
    ClassicShampooPreconditionerConfig,
    DistributedConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    FSDPDistributedConfig,
    FullyShardDistributedConfig,
    GeneralizedPrimalAveragingConfig,
    HSDPDistributedConfig,
    HybridShardDistributedConfig,
    IterateAveragingConfig,
    RMSpropPreconditionerConfig,
    SignDescentPreconditionerConfig,
)
from distributed_shampoo.utils.commons import get_all_non_abstract_subclasses
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class AdaGradPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[AdaGradPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(AdaGradPreconditionerConfig)
    )

    @parametrize("epsilon", (0.0, -1.0))
    @parametrize("cls", subclasses_types)
    def test_illegal_epsilon(
        self, cls: type[AdaGradPreconditionerConfig], epsilon: float
    ) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Invalid epsilon value: {epsilon}. Must be > 0.0."),
            cls,
            epsilon=epsilon,
        )


@instantiate_parametrized_tests
class RMSpropPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[RMSpropPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(RMSpropPreconditionerConfig)
    )

    @parametrize("beta2", (-1.0, 0.0, 1.3))
    @parametrize("cls", subclasses_types)
    def test_illegal_beta2(
        self, cls: type[RMSpropPreconditionerConfig], beta2: float
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
class BaseShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[BaseShampooPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(BaseShampooPreconditionerConfig)  # type: ignore[type-abstract]
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_num_tolerated_failed_amortized_computations(
        self, cls: type[BaseShampooPreconditionerConfig]
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
class ClassicShampooPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[ClassicShampooPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(
            ClassicShampooPreconditionerConfig,  # type: ignore[type-abstract]
        )
    )

    @parametrize("cls", subclasses_types)
    def test_illegal_inverse_exponent_override(
        self, cls: type[ClassicShampooPreconditionerConfig]
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
class SignDescentPreconditionerConfigSubclassesTest(unittest.TestCase):
    subclasses_types: list[type[SignDescentPreconditionerConfig]] = list(
        get_all_non_abstract_subclasses(SignDescentPreconditionerConfig)
    )

    @parametrize("cls", subclasses_types)
    def test_default_scale_fn(self, cls: type[SignDescentPreconditionerConfig]) -> None:
        # Test default scale_fn returns 1.0 for any input
        config = cls()
        grad = torch.randn(3, 4)
        self.assertEqual(config.scale_fn(grad), 1.0)

    @parametrize("cls", subclasses_types)
    def test_custom_scale_fn(self, cls: type[SignDescentPreconditionerConfig]) -> None:
        # Define a custom scale function
        def l1_norm_scale_fn(grad: torch.Tensor) -> float:
            return grad.abs().sum().item()

        config = cls(scale_fn=l1_norm_scale_fn)
        grad = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        # Expected result is 10.0 because L1 norm = |1.0| + |-2.0| + |3.0| + |-4.0| = 1 + 2 + 3 + 4 = 10.0
        self.assertEqual(config.scale_fn(grad), 10.0)


@instantiate_parametrized_tests
class IterateAveragingConfigSubclassesTest(unittest.TestCase):
    # Only test subclasses that have a train_interp_coeff field.
    subclasses_types: list[type[IterateAveragingConfig]] = [
        cls  # type: ignore[type-abstract]
        for cls in get_all_non_abstract_subclasses(IterateAveragingConfig)
        if hasattr(cls, "train_interp_coeff")
    ]

    @parametrize("train_interp_coeff", (-0.1, 0.0, 1.5))
    @parametrize("cls", subclasses_types)
    def test_illegal_train_interp_coeff(
        self, cls: type[IterateAveragingConfig], train_interp_coeff: float
    ) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid self.train_interp_coeff={train_interp_coeff}. Must be within (0.0, 1.0]."
            ),
            cls,
            train_interp_coeff=train_interp_coeff,
        )


@instantiate_parametrized_tests
class GeneralizedPrimalAveragingConfigTest(unittest.TestCase):
    @parametrize("eval_interp_coeff", (-0.1, 1.0, 1.5))
    def test_illegal_eval_interp_coeff(self, eval_interp_coeff: float) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid self.eval_interp_coeff={eval_interp_coeff}. Must be within [0.0, 1.0)."
            ),
            GeneralizedPrimalAveragingConfig,
            eval_interp_coeff=eval_interp_coeff,
        )


@instantiate_parametrized_tests
class ClassicMomentumConfigTest(unittest.TestCase):
    @parametrize("momentum", (-0.1, 0.0, 1.0, 1.5))
    def test_illegal_momentum(self, momentum: float) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Invalid self.momentum={momentum}. Must be within (0.0, 1.0)."),
            ClassicMomentumConfig,
            momentum=momentum,
        )

    @parametrize("dampening", (-0.1, 1.0, 1.5))
    def test_illegal_dampening(self, dampening: float) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"Invalid self.dampening={dampening}. Must be within [0.0, 1.0)."
            ),
            ClassicMomentumConfig,
            dampening=dampening,
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
        if cls in (FSDPDistributedConfig, HSDPDistributedConfig):
            kwargs["param_to_metadata"] = {}
        if cls in (HSDPDistributedConfig, HybridShardDistributedConfig):
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


@instantiate_parametrized_tests
class FullyShardDistributedConfigTest(unittest.TestCase):
    @parametrize("num_sub_groups", (0, -1))
    def test_illegal_num_sub_groups(self, num_sub_groups: int) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Invalid self.num_sub_groups={num_sub_groups}. Must be >= 1."),
            FullyShardDistributedConfig,
            num_sub_groups=num_sub_groups,
        )


@instantiate_parametrized_tests
class HybridShardDistributedConfigTest(unittest.TestCase):
    @parametrize("num_sub_groups", (0, -1))
    def test_illegal_num_sub_groups(self, num_sub_groups: int) -> None:
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Invalid self.num_sub_groups={num_sub_groups}. Must be >= 1."),
            HybridShardDistributedConfig,
            device_mesh=MagicMock(),
            num_sub_groups=num_sub_groups,
        )
