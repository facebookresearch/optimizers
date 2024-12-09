"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest
from abc import ABC, abstractmethod
from typing import Generic, Type, TypeVar

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
    ShampooPreconditionerConfig,
)
from matrix_functions_types import (
    DefaultEigenConfig,
    DefaultEighConfig,
    EigenvectorConfig,
    MatrixFunctionConfig,
    RootInvConfig,
)


class AdaGradGraftingConfigTest(unittest.TestCase):
    def test_illegal_epsilon(self) -> None:
        epsilon = 0.0
        grafting_config_type = self._get_grafting_config_type()
        with (
            self.subTest(grafting_config_type=grafting_config_type),
            self.assertRaisesRegex(
                ValueError,
                re.escape(f"Invalid epsilon value: {epsilon}. Must be > 0.0."),
            ),
        ):
            grafting_config_type(epsilon=epsilon)

    def _get_grafting_config_type(
        self,
    ) -> (
        Type[AdaGradGraftingConfig]
        | Type[RMSpropGraftingConfig]
        | Type[AdamGraftingConfig]
    ):
        return AdaGradGraftingConfig


class RMSpropGraftingConfigTest(AdaGradGraftingConfigTest):
    def test_illegal_beta2(
        self,
    ) -> None:
        grafting_config_type = self._get_grafting_config_type()
        for beta2 in (-1.0, 0.0, 1.3):
            with (
                self.subTest(grafting_config_type=grafting_config_type, beta2=beta2),
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid grafting beta2 parameter: {beta2}. Must be in (0.0, 1.0]."
                    ),
                ),
            ):
                grafting_config_type(beta2=beta2)

    def _get_grafting_config_type(
        self,
    ) -> Type[RMSpropGraftingConfig] | Type[AdamGraftingConfig]:
        return RMSpropGraftingConfig


class AdamGraftingConfigTest(RMSpropGraftingConfigTest):
    def _get_grafting_config_type(
        self,
    ) -> Type[RMSpropGraftingConfig] | Type[AdamGraftingConfig]:
        return AdamGraftingConfig


PreconditionerConfigType = TypeVar(
    "PreconditionerConfigType", bound=Type[PreconditionerConfig]
)
AmortizedComputationConfigType = TypeVar(
    "AmortizedComputationConfigType", bound=MatrixFunctionConfig
)


class AbstractPreconditionerConfigTest:
    class PreconditionerConfigTest(
        ABC,
        unittest.TestCase,
        Generic[PreconditionerConfigType, AmortizedComputationConfigType],
    ):
        def test_illegal_num_tolerated_failed_amortized_computations(self) -> None:
            num_tolerated_failed_amortized_computations = -1
            with (
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Invalid num_tolerated_failed_amortized_computations value: "
                        f"{num_tolerated_failed_amortized_computations}. Must be >= 0."
                    ),
                ),
            ):
                self._get_preconditioner_config_type()(
                    amortized_computation_config=self._get_amortized_computation_config(),
                    num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
                )

        @abstractmethod
        def _get_preconditioner_config_type(
            self,
        ) -> PreconditionerConfigType: ...

        @abstractmethod
        def _get_amortized_computation_config(
            self,
        ) -> AmortizedComputationConfigType: ...


class ShampooPreconditionerConfigTest(
    AbstractPreconditionerConfigTest.PreconditionerConfigTest[
        Type[ShampooPreconditionerConfig], RootInvConfig
    ]
):
    def _get_amortized_computation_config(self) -> RootInvConfig:
        return DefaultEigenConfig

    def _get_preconditioner_config_type(
        self,
    ) -> Type[ShampooPreconditionerConfig]:
        return ShampooPreconditionerConfig


class EigenvalueCorrectedShampooPreconditionerConfigTest(
    AbstractPreconditionerConfigTest.PreconditionerConfigTest[
        Type[EigenvalueCorrectedShampooPreconditionerConfig], EigenvectorConfig
    ]
):
    def _get_amortized_computation_config(self) -> EigenvectorConfig:
        return DefaultEighConfig

    def _get_preconditioner_config_type(
        self,
    ) -> Type[EigenvalueCorrectedShampooPreconditionerConfig]:
        return EigenvalueCorrectedShampooPreconditionerConfig
