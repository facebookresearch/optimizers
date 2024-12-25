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
    "PreconditionerConfigType", bound=PreconditionerConfig
)


class AbstractPreconditionerConfigTest:
    class PreconditionerConfigTest(
        ABC,
        unittest.TestCase,
        Generic[PreconditionerConfigType],
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
                    num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
                )

        @abstractmethod
        def _get_preconditioner_config_type(
            self,
        ) -> Type[PreconditionerConfigType]: ...


class ShampooPreconditionerConfigTest(
    AbstractPreconditionerConfigTest.PreconditionerConfigTest[
        ShampooPreconditionerConfig
    ]
):
    def _get_preconditioner_config_type(
        self,
    ) -> Type[ShampooPreconditionerConfig]:
        return ShampooPreconditionerConfig


class EigenvalueCorrectedShampooPreconditionerConfigTest(
    AbstractPreconditionerConfigTest.PreconditionerConfigTest[
        EigenvalueCorrectedShampooPreconditionerConfig
    ]
):
    def _get_preconditioner_config_type(
        self,
    ) -> Type[EigenvalueCorrectedShampooPreconditionerConfig]:
        return EigenvalueCorrectedShampooPreconditionerConfig
