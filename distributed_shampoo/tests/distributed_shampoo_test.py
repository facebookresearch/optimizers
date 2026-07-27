"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import abc
import gc
import logging
import re
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.preconditioner.matrix_functions_types import (
    DefaultNewtonSchulzOrthogonalizationConfig,
    EigenConfig,
    OrthogonalizationConfig,
    PseudoInverseConfig,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    BaseShampooPreconditionerConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSpectralDescentPreconditionerConfig,
    DistributedConfig,
    EigendecomposedKLShampooPreconditionerConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    GeneralizedPrimalAveragingConfig,
    IterateAveragingConfig,
    LR_SUM,
    PreconditionerConfig,
    RootInvKLShampooPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    ScheduleFreeConfig,
    ShampooPT2CompileConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
    STEP,
    TRAIN_MODE,
    WeightDecayType,
)
from distributed_shampoo.utils.shampoo_utils import pack_upper_triangular
from torch import nn, Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


def _pack_if_enabled(
    config: PreconditionerConfig, matrix: torch.Tensor
) -> torch.Tensor:
    """Pack matrix to upper triangular format if symmetric packing is enabled."""
    return (
        pack_upper_triangular(matrix)
        if isinstance(config, BaseShampooPreconditionerConfig)
        and config.use_symmetric_packing
        else matrix
    )


@instantiate_parametrized_tests
class DistributedShampooInitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )

    def test_invalid_preconditioner_config(self) -> None:
        @dataclass
        class NotSupportedPreconditionerConfig(PreconditionerConfig):
            """A dummy preconditioner config that is not supported."""

            unsupported_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            r"preconditioner_config=.*\.NotSupportedPreconditionerConfig\(.*\) not supported!",
            DistributedShampoo,
            self._model.parameters(),
            preconditioner_config=NotSupportedPreconditionerConfig(),
        )

    @parametrize(
        "incorrect_hyperparameter_setting, expected_error_msg",
        [
            (
                {"lr": -0.1},
                "Invalid param_group[LR]=-0.1. Must be >= 0.0.",
            ),
            (
                {"betas": (-0.1, 1.0)},
                "Invalid param_group[BETAS][0]=-0.1. Must be in [0.0, 1.0).",
            ),
            (
                {"betas": (0.9, -0.1)},
                "Invalid param_group[BETAS][1]=-0.1. Must be in [0.0, 1.0].",
            ),
            (
                {"beta3": -0.1},
                "Invalid param_group[BETA3]=-0.1. Must be in [0.0, 1.0).",
            ),
            (
                {
                    "epsilon": 0.1,
                    "preconditioner_config": RootInvShampooPreconditionerConfig(
                        amortized_computation_config=EigenConfig(
                            rank_deficient_stability_config=PseudoInverseConfig()
                        )
                    ),
                },
                "Invalid param_group[EPSILON]=0.1. Must be == 0.0 when PseudoInverseConfig is used.",
            ),
            (
                {"epsilon": 0.0},
                "Invalid param_group[EPSILON]=0.0. Must be > 0.0.",
            ),
            (
                {"weight_decay": -0.1},
                "Invalid param_group[WEIGHT_DECAY]=-0.1. Must be >= 0.0.",
            ),
            (
                {"max_preconditioner_dim": 3.14},
                "Invalid param_group[MAX_PRECONDITIONER_DIM]=3.14. Must be an integer or math.inf.",
            ),
            (
                {"max_preconditioner_dim": 0},
                "Invalid param_group[MAX_PRECONDITIONER_DIM]=0. Must be >= 1.",
            ),
            (
                {"precondition_frequency": 0},
                "Invalid param_group[PRECONDITION_FREQUENCY]=0. Must be >= 1.",
            ),
            (
                {"start_preconditioning_step": -2},
                "Invalid param_group[START_PRECONDITIONING_STEP]=-2. Must be >= -1.",
            ),
            (
                {"start_preconditioning_step": 10, "precondition_frequency": 100},
                "Invalid param_group[START_PRECONDITIONING_STEP]=10. Must be >= param_group[PRECONDITION_FREQUENCY]=100.",
            ),
        ],
    )
    def test_invalid_with_incorrect_hyperparameter_setting(
        self, incorrect_hyperparameter_setting: dict[str, Any], expected_error_msg: str
    ) -> None:
        # Test the incorrect hyperparameter setting in the default hyperparameter setting.
        self.assertRaisesRegex(
            ValueError,
            re.escape(expected_error_msg),
            DistributedShampoo,
            self._model.parameters(),
            **incorrect_hyperparameter_setting,
        )

        # Test the incorrect hyperparameter setting in the param_group setting.
        with self.assertLogs(level="INFO") as cm:
            self.assertRaisesRegex(
                ValueError,
                re.escape(expected_error_msg),
                DistributedShampoo,
                [
                    {"params": []},  # param_group 0 is valid
                    {
                        "params": self._model.parameters(),
                        **incorrect_hyperparameter_setting,  # We intentionally let param_group 1 fail to test error detection
                    },
                    {"params": []},  # param_group 2 is valid
                ],
            )

            msgs = [r.msg for r in cm.records if r.levelname == "INFO"]

        self.assertEqual(
            msgs,
            [
                "Checking param_group 0 hyperparameters...",
                "Checking param_group 1 hyperparameters...",
                # We don't see param_group 2 message because validation stops after finding the first invalid param_group
            ],
        )

    @parametrize(
        "noop_hyperparameter_setting, expected_warning_msgs",
        [
            (
                {
                    "betas": (0.9, 0.999),
                    "epsilon": 1e-8,
                    "precondition_frequency": 100,
                    "preconditioner_config": DefaultSpectralDescentPreconditionerConfig,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=1,
                    ),
                },
                [
                    "param_group[BETAS][1]=0.999 does not have any effect when SpectralDescentPreconditionerConfig is used.",
                    "param_group[EPSILON]=1e-08 does not have any effect when SpectralDescentPreconditionerConfig is used.",
                    "param_group[PRECONDITION_FREQUENCY]=100 does not have any effect when SpectralDescentPreconditionerConfig is used. Setting precondition_frequency to 1...",
                    "param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality=1 is not equal to 2. Setting target_parameter_dimensionality to 2...",
                ],
            ),
            (
                {
                    "betas": (0.9, 0.999),
                    "epsilon": 1e-8,
                    "precondition_frequency": 100,
                    "preconditioner_config": DefaultSignDescentPreconditionerConfig,
                },
                [
                    "param_group[BETAS][1]=0.999 does not have any effect when SignDescentPreconditionerConfig is used.",
                    "param_group[EPSILON]=1e-08 does not have any effect when SignDescentPreconditionerConfig is used.",
                    "param_group[PRECONDITION_FREQUENCY]=100 does not have any effect when SignDescentPreconditionerConfig is used. Setting precondition_frequency to 1...",
                ],
            ),
        ],
    )
    def test_noop_hyperparameter_setting_warnings(
        self,
        noop_hyperparameter_setting: dict[str, Any],
        expected_warning_msgs: list[str],
    ) -> None:
        with self.assertLogs(level="WARNING") as cm:
            DistributedShampoo(
                self._model.parameters(),
                **noop_hyperparameter_setting,
            )
            recorded_warning_msgs = [r.msg for r in cm.records]
            for expected_warning_msg in expected_warning_msgs:
                with self.subTest(
                    noop_hyperparameter_setting=noop_hyperparameter_setting,
                    expected_warning_msg=expected_warning_msg,
                    recorded_warning_msgs=recorded_warning_msgs,
                ):
                    self.assertIn(
                        expected_warning_msg,
                        recorded_warning_msgs,
                    )

    def test_invalid_distributed_config(self) -> None:
        @dataclass
        class NotSupportedDistributedConfig(DistributedConfig):
            """A dummy distributed config that is not supported."""

            unsupported_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            r"distributed_config=.*\.NotSupportedDistributedConfig\(.*\) not supported!",
            DistributedShampoo,
            params=self._model.parameters(),
            distributed_config=NotSupportedDistributedConfig(),
        )


class DistributedShampooTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            # Explicitly set grafting_config=None to test the case that no grafting config is used.
            grafting_config=None,
        )

    def test_step_with_closure(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        # Test the case without closure, the loss returned by step() is None.
        self._optimizer.zero_grad()
        layer_weight.grad = torch.rand_like(layer_weight)
        self.assertIsNone(self._optimizer.step(closure=None))

        # Test the case that the closure returns a scalar.
        def closure() -> float:
            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            return 1.0

        self.assertEqual(self._optimizer.step(closure=closure), 1.0)

    def test_optimizer_zero_grad(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        layer_weight.grad = torch.ones_like(layer_weight)

        # Store the data pointer of the current gradient to check if it gets freed later.
        grad_data_ptr = layer_weight.grad.data_ptr()

        self._optimizer.step()

        # Call zero_grad with set_to_none=True to explicitly release gradient memory rather than just zeroing it out.
        self._optimizer.zero_grad(set_to_none=True)

        # Verify that the gradient has been set to None.
        self.assertIsNone(layer_weight.grad)

        # Get all tensor objects currently tracked by the garbage collector.
        all_alive_tensors = tuple(
            obj
            for obj in gc.get_objects()
            # Using type(obj) here to prevent the garbage collector from including non-real tensors like FakeTensor.
            if type(obj) in (torch.Tensor, nn.Parameter)
        )

        # Check that the stored gradient data pointer is not in the list of alive tensors, ensuring it was freed.
        self.assertNotIn(
            grad_data_ptr,
            (t.data_ptr() for t in all_alive_tensors),
            msg="Found gradients space is still not freed, check Shampoo code for properly free gradients pointers.",
        )


class AbstractTest:
    class StateDictTestBase(abc.ABC, unittest.TestCase):
        @property
        @abc.abstractmethod
        def _preconditioner_config(self) -> PreconditionerConfig: ...

        @property
        @abc.abstractmethod
        def _ref_state_dict(self) -> dict[str, Any]: ...

        def setUp(self) -> None:
            self._model = nn.Sequential(
                nn.Linear(5, 10, bias=False),
            )
            # Initialize weights to zeros to ensure deterministic state dict values.
            with torch.no_grad():
                cast(torch.Tensor, self._model[0].weight).zero_()
            self._optimizer = DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                betas=(0.9, 1.0),
                epsilon=1e-12,
                weight_decay=0.0,
                max_preconditioner_dim=5,
                precondition_frequency=1,
                start_preconditioning_step=-1,
                distributed_config=replace(
                    DefaultSingleDeviceDistributedConfig,
                    # distributed_config.target_parameter_dimensionality=2 is necessary to prevent SpectralDescentPreconditionerConfig assertion error.
                    target_parameter_dimensionality=2,
                ),
                grafting_config=AdaGradPreconditionerConfig(
                    epsilon=0.001,
                ),
                preconditioner_config=self._preconditioner_config,
            )

        def test_setstate_call(self) -> None:
            """Test that __setstate__ is properly called during load_state_dict operation."""

            class MockDistributedShampoo(DistributedShampoo):
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    super().__init__(*args, **kwargs)
                    # Flag to track if __setstate__ was called
                    self._shampoo_setstate_called = False

                def __setstate__(self, state: dict[str, Any]) -> None:
                    # Mark that __setstate__ was invoked
                    self._shampoo_setstate_called = True
                    super().__setstate__(state)

            # Create a mock optimizer instance
            mocked_shampoo_optimizer = MockDistributedShampoo(self._model.parameters())
            # Get the current state dictionary
            optim_state_dict = mocked_shampoo_optimizer.state_dict()

            # Load the state dictionary, which should trigger __setstate__
            mocked_shampoo_optimizer.load_state_dict(optim_state_dict)

            # Verify that __setstate__ was called during load_state_dict
            self.assertTrue(mocked_shampoo_optimizer._shampoo_setstate_called, True)

        def test_state_dict(self) -> None:
            """
            Test that the state dict is correct by comparing
            optimizer.state_dict() and the reference state dict.
            """
            state_dict = self._optimizer.state_dict()
            ref_state_dict = self._ref_state_dict
            self.assertEqual(state_dict.keys(), {"state", "param_groups"})

            torch.testing.assert_close(
                state_dict["state"],
                ref_state_dict["state"],
            )
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

        def test_load_state_dict(self) -> None:
            """
            Test that load_state_dict() loads the correct state dict by comparing
            optimizer.state_dict() and the reference state dict. Note that load_state_dict()
            calls __setstate__, which we override in Shampoo.
            """
            ref_state_dict = self._ref_state_dict
            self._optimizer.load_state_dict(
                state_dict=ref_state_dict,
            )

            state_dict = self._optimizer.state_dict()

            self.assertEqual(state_dict.keys(), ref_state_dict.keys())
            torch.testing.assert_close(state_dict["state"], ref_state_dict["state"])
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

    class NoPreconditionerStateDictTestBase(StateDictTestBase):
        """A base class for methods that do not have a preconditioner."""

        @property
        def _ref_state_dict(self) -> dict[str, Any]:
            return {
                "state": {
                    0: {
                        "block_0": {
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "block_1": {
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "step": torch.tensor(0),
                    }
                },
                "param_groups": [
                    {
                        "lr": 0.01,
                        "betas": (0.9, 1.0),
                        "beta3": 0.9,
                        "epsilon": 1e-12,
                        "weight_decay": 0.0,
                        "peak_lr": 0.01,
                        "weight_decay_type": WeightDecayType.DECOUPLED,
                        "max_preconditioner_dim": 5,
                        "precondition_frequency": 1,
                        "start_preconditioning_step": 1,
                        "use_bias_correction": True,
                        "iterate_averaging_config": None,
                        "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                        "use_pin_memory": False,
                        "distributed_config": SingleDeviceDistributedConfig(
                            target_parameter_dimensionality=2
                        ),
                        "preconditioner_config": self._preconditioner_config,
                        "params": [0],
                    }
                ],
            }


class ShampooStateDictTest(AbstractTest.StateDictTestBase):
    @property
    def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return DefaultShampooConfig

    @property
    def _ref_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                0: {
                    "block_0": {
                        "shampoo": {
                            "factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                            },
                            "inv_factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.eye(5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.eye(5)
                                ),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "block_1": {
                        "shampoo": {
                            "factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                            },
                            "inv_factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.eye(5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.eye(5)
                                ),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "step": torch.tensor(0),
                }
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "betas": (0.9, 1.0),
                    "beta3": 0.9,
                    "epsilon": 1e-12,
                    "weight_decay": 0.0,
                    "peak_lr": 0.01,
                    "weight_decay_type": WeightDecayType.DECOUPLED,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_bias_correction": True,
                    "iterate_averaging_config": None,
                    "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                    "use_pin_memory": False,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=2
                    ),
                    "preconditioner_config": self._preconditioner_config,
                    "params": [0],
                }
            ],
        }


class EigendecomposedShampooStateDictTest(AbstractTest.StateDictTestBase):
    @property
    def _preconditioner_config(self) -> EigendecomposedShampooPreconditionerConfig:
        return EigendecomposedShampooPreconditionerConfig()

    @property
    def _ref_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                0: {
                    "block_0": {
                        "shampoo": {
                            "factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvalues": {
                                0: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                                1: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "block_1": {
                        "shampoo": {
                            "factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "factor_matrices_eigenvalues": {
                                0: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                                1: torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]),
                            },
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "step": torch.tensor(0),
                }
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "betas": (0.9, 1.0),
                    "beta3": 0.9,
                    "epsilon": 1e-12,
                    "weight_decay": 0.0,
                    "peak_lr": 0.01,
                    "weight_decay_type": WeightDecayType.DECOUPLED,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_bias_correction": True,
                    "iterate_averaging_config": None,
                    "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                    "use_pin_memory": False,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=2
                    ),
                    "preconditioner_config": self._preconditioner_config,
                    "params": [0],
                }
            ],
        }


class EigenvalueCorrectedShampooStateDictTest(AbstractTest.StateDictTestBase):
    @property
    def _preconditioner_config(self) -> EigenvalueCorrectedShampooPreconditionerConfig:
        return DefaultEigenvalueCorrectedShampooConfig

    @property
    def _ref_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                0: {
                    "block_0": {
                        "shampoo": {
                            "factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "corrected_eigenvalues": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "block_1": {
                        "shampoo": {
                            "factor_matrices": {
                                0: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                                1: _pack_if_enabled(
                                    self._preconditioner_config, torch.zeros(5, 5)
                                ),
                            },
                            "factor_matrices_eigenvectors": {
                                0: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 1.0],
                                    ]
                                ),
                            },
                            "corrected_eigenvalues": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "adagrad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        "filtered_grad": torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                    "step": torch.tensor(0),
                }
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "betas": (0.9, 1.0),
                    "beta3": 0.9,
                    "epsilon": 1e-12,
                    "weight_decay": 0.0,
                    "peak_lr": 0.01,
                    "weight_decay_type": WeightDecayType.DECOUPLED,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_bias_correction": True,
                    "iterate_averaging_config": None,
                    "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                    "use_pin_memory": False,
                    "distributed_config": SingleDeviceDistributedConfig(
                        target_parameter_dimensionality=2
                    ),
                    "preconditioner_config": self._preconditioner_config,
                    "params": [0],
                }
            ],
        }


class RootInvKLShampooStateDictTest(ShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> RootInvKLShampooPreconditionerConfig:
        return RootInvKLShampooPreconditionerConfig()


class EigendecomposedKLShampooStateDictTest(EigendecomposedShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> EigendecomposedKLShampooPreconditionerConfig:
        return EigendecomposedKLShampooPreconditionerConfig()


# ---- Unpacked (use_symmetric_packing=False) state dict test variants ----


class ShampooStateDictUnpackedTest(ShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return replace(DefaultShampooConfig, use_symmetric_packing=False)


class EigendecomposedShampooStateDictUnpackedTest(EigendecomposedShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> EigendecomposedShampooPreconditionerConfig:
        return replace(
            EigendecomposedShampooPreconditionerConfig(),
            use_symmetric_packing=False,
        )


class EigenvalueCorrectedShampooStateDictUnpackedTest(
    EigenvalueCorrectedShampooStateDictTest,
):
    @property
    def _preconditioner_config(self) -> EigenvalueCorrectedShampooPreconditionerConfig:
        return replace(
            DefaultEigenvalueCorrectedShampooConfig,
            use_symmetric_packing=False,
        )


class RootInvKLShampooStateDictUnpackedTest(RootInvKLShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> RootInvKLShampooPreconditionerConfig:
        return replace(
            RootInvKLShampooPreconditionerConfig(), use_symmetric_packing=False
        )


class EigendecomposedKLShampooStateDictUnpackedTest(
    EigendecomposedKLShampooStateDictTest,
):
    @property
    def _preconditioner_config(self) -> EigendecomposedKLShampooPreconditionerConfig:
        return replace(
            EigendecomposedKLShampooPreconditionerConfig(),
            use_symmetric_packing=False,
        )


class SignDescentStateDictTest(AbstractTest.NoPreconditionerStateDictTestBase):
    @property
    def _preconditioner_config(self) -> SignDescentPreconditionerConfig:
        return DefaultSignDescentPreconditionerConfig

    def test_state_dict_warning(self) -> None:
        """
        When Shampoo's `post_state_dict_hook` is fired during
        `state_dict()` call, it should issue a warning if a lambda function is detected,
        since it cannot pickled. This test checks that the warning is issued.
        """
        osd = self._optimizer.state_dict()
        self.assertCountEqual(osd.keys(), ["state", "param_groups"])

        @dataclass(kw_only=True)
        class SignDescentPreconditionerConfigWithLambda(
            SignDescentPreconditionerConfig
        ):
            """
            Creating a preconditioner config with a dummy lambda function to make sure the
            warning from `_post_state_dict_hook` emit.
            """

            scale_fn: Callable[[Tensor], float | Tensor] = lambda grad: 1.0

        self._optimizer.param_groups[0]["preconditioner_config"] = (
            SignDescentPreconditionerConfigWithLambda()
        )
        logger = logging.getLogger("distributed_shampoo.distributed_shampoo")
        with self.assertLogs(logger, level="WARNING") as cm:
            osd = self._optimizer.state_dict()
        self.assertIn(
            "Note that lambda function cannot be pickled. torch.save() cannot serialize lambda functions, "
            "because it relies on Python's pickle module for serialization, and pickle does not support lambda functions",
            cm.output[0],
        )


class SpectralDescentStateDictTest(AbstractTest.NoPreconditionerStateDictTestBase):
    @property
    def _preconditioner_config(self) -> SpectralDescentPreconditionerConfig:
        return DefaultSpectralDescentPreconditionerConfig

    def test_state_dict_warning(self) -> None:
        """
        When Shampoo's `post_state_dict_hook` is fired during
        `state_dict()` call, it should issue a warning if a lambda function is detected,
        since it cannot pickled. This test checks that the warning is issued.
        """
        osd = self._optimizer.state_dict()
        self.assertCountEqual(osd.keys(), ["state", "param_groups"])

        @dataclass(kw_only=True)
        class SpectralDescentPreconditionerConfigWithLambda(PreconditionerConfig):
            """
            Creating a orthogonalization config with a dummy lambda function to make sure the
            warning from `_post_state_dict_hook` emit.
            """

            orthogonalization_config: OrthogonalizationConfig = field(
                default_factory=lambda: DefaultNewtonSchulzOrthogonalizationConfig
            )

        self._optimizer.param_groups[0]["orthogonalization_config"] = (
            SpectralDescentPreconditionerConfigWithLambda()
        )
        logger = logging.getLogger("distributed_shampoo.distributed_shampoo")
        with self.assertLogs(logger, level="WARNING") as cm:
            osd = self._optimizer.state_dict()
        self.assertIn(
            "Note that lambda function cannot be pickled. torch.save() cannot serialize lambda functions, "
            "because it relies on Python's pickle module for serialization, and pickle does not support lambda functions",
            cm.output[0],
        )


class AbstractIterateAveragingTest:
    """Abstract base classes for testing iterate averaging configurations (GPA and Schedule-Free)."""

    class IterateAveragingStateDictTestBase(abc.ABC, unittest.TestCase):
        """Base class for testing state dict with iterate averaging enabled.

        When iterate averaging is enabled, the optimizer stores a weight_buffer
        for each parameter block that contains the "z" sequence.
        """

        @property
        @abc.abstractmethod
        def _iterate_averaging_config(self) -> IterateAveragingConfig: ...

        @property
        def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
            return DefaultShampooConfig

        def setUp(self) -> None:
            self._model = nn.Sequential(
                nn.Linear(5, 10, bias=False),
            )
            # Initialize weights to zeros to ensure deterministic state dict values.
            with torch.no_grad():
                cast(torch.Tensor, self._model[0].weight).zero_()
            self._optimizer = DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                betas=(0.9, 1.0),
                epsilon=1e-12,
                weight_decay=0.0,
                max_preconditioner_dim=5,
                precondition_frequency=1,
                start_preconditioning_step=-1,
                iterate_averaging_config=self._iterate_averaging_config,
                distributed_config=replace(
                    DefaultSingleDeviceDistributedConfig,
                    target_parameter_dimensionality=2,
                ),
                grafting_config=AdaGradPreconditionerConfig(
                    epsilon=0.001,
                ),
                preconditioner_config=self._preconditioner_config,
            )

        @property
        def _ref_state_dict(self) -> dict[str, Any]:
            return {
                "state": {
                    0: {
                        "block_0": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: _pack_if_enabled(
                                        self._preconditioner_config, torch.zeros(5, 5)
                                    ),
                                    1: _pack_if_enabled(
                                        self._preconditioner_config, torch.zeros(5, 5)
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: _pack_if_enabled(
                                        self._preconditioner_config, torch.eye(5)
                                    ),
                                    1: _pack_if_enabled(
                                        self._preconditioner_config, torch.eye(5)
                                    ),
                                },
                            },
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "weight_buffer": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "block_1": {
                            "shampoo": {
                                "factor_matrices": {
                                    0: _pack_if_enabled(
                                        self._preconditioner_config, torch.zeros(5, 5)
                                    ),
                                    1: _pack_if_enabled(
                                        self._preconditioner_config, torch.zeros(5, 5)
                                    ),
                                },
                                "inv_factor_matrices": {
                                    0: _pack_if_enabled(
                                        self._preconditioner_config, torch.eye(5)
                                    ),
                                    1: _pack_if_enabled(
                                        self._preconditioner_config, torch.eye(5)
                                    ),
                                },
                            },
                            "adagrad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "filtered_grad": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                            "weight_buffer": torch.tensor(
                                [
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0],
                                ]
                            ),
                        },
                        "step": torch.tensor(0),
                        "train_mode": torch.tensor(True),
                        "lr_sum": torch.tensor(0.0),
                    }
                },
                "param_groups": [
                    {
                        "lr": 0.01,
                        "betas": (0.9, 1.0),
                        "beta3": 0.9,
                        "epsilon": 1e-12,
                        "weight_decay": 0.0,
                        "peak_lr": 0.01,
                        "weight_decay_type": WeightDecayType.DECOUPLED,
                        "max_preconditioner_dim": 5,
                        "precondition_frequency": 1,
                        "start_preconditioning_step": 1,
                        "use_bias_correction": True,
                        "iterate_averaging_config": self._iterate_averaging_config,
                        "grafting_config": AdaGradPreconditionerConfig(epsilon=0.001),
                        "use_pin_memory": False,
                        "distributed_config": SingleDeviceDistributedConfig(
                            target_parameter_dimensionality=2
                        ),
                        "preconditioner_config": self._preconditioner_config,
                        "params": [0],
                    }
                ],
            }

        def test_state_dict(self) -> None:
            """Test that the state dict contains weight_buffer when iterate averaging is enabled."""
            state_dict = self._optimizer.state_dict()
            ref_state_dict = self._ref_state_dict
            self.assertEqual(state_dict.keys(), {"state", "param_groups"})

            torch.testing.assert_close(
                state_dict["state"],
                ref_state_dict["state"],
            )
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

        def test_load_state_dict(self) -> None:
            """Test that load_state_dict() correctly restores weight_buffer."""
            ref_state_dict = self._ref_state_dict
            self._optimizer.load_state_dict(
                state_dict=ref_state_dict,
            )

            state_dict = self._optimizer.state_dict()

            self.assertEqual(state_dict.keys(), ref_state_dict.keys())
            torch.testing.assert_close(state_dict["state"], ref_state_dict["state"])
            self.assertEqual(
                state_dict["param_groups"],
                ref_state_dict["param_groups"],
            )

        def test_weight_buffer_in_state(self) -> None:
            """Test that weight_buffer is present in each block's state."""
            state_dict = self._optimizer.state_dict()
            for block_key in ["block_0", "block_1"]:
                self.assertIn(
                    "weight_buffer",
                    state_dict["state"][0][block_key],
                    f"weight_buffer should be present in {block_key} when iterate averaging is enabled",
                )


class GPAShampooStateDictTest(
    AbstractIterateAveragingTest.IterateAveragingStateDictTestBase
):
    """Test state dict with Generalized Primal Averaging (GPA) enabled.

    See https://arxiv.org/pdf/2512.17131 for details on GPA.
    """

    @property
    def _iterate_averaging_config(self) -> GeneralizedPrimalAveragingConfig:
        return GeneralizedPrimalAveragingConfig(
            eval_interp_coeff=0.5,
            train_interp_coeff=0.9,
        )


class ScheduleFreeShampooStateDictTest(
    AbstractIterateAveragingTest.IterateAveragingStateDictTestBase
):
    """Test state dict with Schedule-Free enabled.

    See https://arxiv.org/abs/2405.15682 for details on Schedule-Free.
    """

    @property
    def _iterate_averaging_config(self) -> ScheduleFreeConfig:
        return ScheduleFreeConfig(
            train_interp_coeff=0.9,
        )


class GPAShampooStateDictUnpackedTest(GPAShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return replace(DefaultShampooConfig, use_symmetric_packing=False)


class ScheduleFreeShampooStateDictUnpackedTest(ScheduleFreeShampooStateDictTest):
    @property
    def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return replace(DefaultShampooConfig, use_symmetric_packing=False)


@instantiate_parametrized_tests
class DistributedShampooTrainEvalModeTest(unittest.TestCase):
    """Test train/eval mode switching with iterate averaging configurations."""

    @parametrize(
        "iterate_averaging_config",
        (
            GeneralizedPrimalAveragingConfig(),
            ScheduleFreeConfig(),
        ),
    )
    def test_train_eval_mode_switching(
        self,
        iterate_averaging_config: IterateAveragingConfig,
    ) -> None:
        """
        Test that train() and eval() mode switching works correctly with iterate averaging.
        This verifies that the mode switching updates the train_mode flag appropriately.
        """
        model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=iterate_averaging_config,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

        # Take a few optimizer steps
        for _ in range(3):
            optimizer.zero_grad()
            layer_weight: torch.Tensor = cast(torch.Tensor, model[0].weight)
            layer_weight.grad = torch.rand_like(layer_weight)
            optimizer.step()

        # Get the full state dictionary.
        initial_state = optimizer.state_dict()["state"]

        # Find all parameters that contain train_mode keys.
        # We expect exactly 1 parameter (the first in each group) to have train_mode stored.
        train_mode_param_keys = []
        all_train_mode_keys: dict[str, list[str]] = {}
        for param_key, param_state in initial_state.items():
            keys_with_train_mode = [
                key for key in param_state.keys() if "train_mode" in key
            ]
            if keys_with_train_mode:
                train_mode_param_keys.append(param_key)
                all_train_mode_keys[param_key] = keys_with_train_mode

        # There should be exactly 1 parameter with train_mode keys (one per param group).
        self.assertEqual(
            len(train_mode_param_keys),
            1,
            msg=f"Expected exactly 1 parameter with train_mode keys, got {len(train_mode_param_keys)}",
        )

        train_mode_param_key = train_mode_param_keys[0]
        train_mode_keys = all_train_mode_keys[train_mode_param_key]

        # Verify initial training mode (should be True after training)
        for key in train_mode_keys:
            self.assertTrue(
                initial_state[train_mode_param_key][key].item(),
                msg="Expected train_mode to be True after training",
            )

        # Switch to eval mode
        optimizer.eval()

        # Verify eval mode
        eval_state = optimizer.state_dict()["state"]
        for key in train_mode_keys:
            self.assertFalse(
                eval_state[train_mode_param_key][key].item(),
                msg="Expected train_mode to be False after eval()",
            )

        # Switch back to train mode
        optimizer.train()

        # Verify train mode again
        train_state = optimizer.state_dict()["state"]
        for key in train_mode_keys:
            self.assertTrue(
                train_state[train_mode_param_key][key].item(),
                msg="Expected train_mode to be True after train()",
            )

    @parametrize(
        "iterate_averaging_config",
        (
            GeneralizedPrimalAveragingConfig(),
            ScheduleFreeConfig(),
        ),
    )
    @parametrize(
        "save_in_eval,load_in_eval",
        (
            (False, False),  # Save in train, load in train -> stay in train.
            (True, False),  # Save in eval, load in train -> stay in train.
            (False, True),  # Save in train, load in eval -> stay in eval.
            (True, True),  # Save in eval, load in eval -> stay in eval.
        ),
    )
    def test_load_state_dict_preserves_mode(
        self,
        iterate_averaging_config: IterateAveragingConfig,
        save_in_eval: bool,
        load_in_eval: bool,
    ) -> None:
        """Test that load_state_dict() preserves the caller's train/eval mode."""
        model = nn.Sequential(nn.Linear(5, 10, bias=False))
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=iterate_averaging_config,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

        # Take a few optimizer steps to populate state.
        for _ in range(3):
            optimizer.zero_grad()
            layer_weight: torch.Tensor = cast(torch.Tensor, model[0].weight)
            layer_weight.grad = torch.rand_like(layer_weight)
            optimizer.step()

        # Save checkpoint in the specified mode.
        if save_in_eval:
            optimizer.eval()
        state_dict = optimizer.state_dict()

        # Switch to the load mode.
        if load_in_eval:
            optimizer.eval()
        else:
            optimizer.train()

        # Load the checkpoint.
        optimizer.load_state_dict(state_dict)

        # Verify the optimizer preserved the load mode (not the save mode).
        expected_train_mode = not load_in_eval
        state = optimizer.state_dict()["state"]
        for param_state in state.values():
            for key in param_state:
                if "train_mode" in key:
                    self.assertEqual(
                        param_state[key].item(),
                        expected_train_mode,
                        msg=f"Expected train_mode to be {expected_train_mode} "
                        f"(save_in_eval={save_in_eval}, load_in_eval={load_in_eval})",
                    )

    def test_train_eval_mode_without_iterate_averaging(self) -> None:
        """
        Test that train() and eval() are no-ops when iterate_averaging_config is None.
        This verifies that calling these methods doesn't raise a KeyError.
        """
        model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=None,  # No iterate averaging
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

        # Take a few optimizer steps
        for _ in range(3):
            optimizer.zero_grad()
            layer_weight: torch.Tensor = cast(torch.Tensor, model[0].weight)
            layer_weight.grad = torch.rand_like(layer_weight)
            optimizer.step()

        # Calling train() and eval() should not raise any errors
        optimizer.train()
        optimizer.eval()
        optimizer.train()

        # Verify state_dict works without iterate averaging
        state_dict = optimizer.state_dict()
        self.assertIn("state", state_dict)
        self.assertIn("param_groups", state_dict)


class DistributedShampooNoneGradTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            shampoo_pt2_compile_config=ShampooPT2CompileConfig(backend="eager"),
            distributed_config=DefaultSingleDeviceDistributedConfig,
            # Explicitly set grafting_config=None to test the case that no grafting config is used.
            grafting_config=None,
        )

    def test_step_with_consistent_grads(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        with self.assertNoLogs(level="WARNING"):
            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            self._optimizer.step()

            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            self._optimizer.step()

    def test_step_with_none_grads(self) -> None:
        layer_weight: torch.Tensor = cast(torch.Tensor, self._model[0].weight)
        expected_msg = "PT2 will recompile because the gradient selection of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model."
        ending_msg = "Changed gradient selector indices: [0, 1]"
        with self.assertLogs(level="WARNING") as cm:
            self._optimizer.zero_grad()
            layer_weight.grad = torch.rand_like(layer_weight)
            self._optimizer.step()

            self._optimizer.zero_grad()  # Implicitly set grad=None in second step
            self._optimizer.step()
            msgs = [r.msg for r in cm.records]

        self.assertEqual(len(msgs), 1)
        self.assertIn(expected_msg, msgs[0])
        self.assertIn(ending_msg, msgs[0])


class CheckpointPerParamScalarStateTest(unittest.TestCase):
    """Per-param checkpointing of the shared per-group scalars step / lr_sum / train_mode.

    The optimizer keeps ``step`` (and, with iterate averaging, ``lr_sum`` /
    ``train_mode``) as a single tensor shared (aliased) across all params in a
    group. These are checkpointed under EVERY param's state (the PyTorch per-param
    convention) rather than only under the group's first param, so the on-disk
    ``state_dict`` key set is param-keyed and invariant to how params are
    partitioned into groups.

    ``num_sub_groups > 1`` only applies to FSDP2 / HSDP2
    (``FullyShardDistributedConfig`` / ``HybridShardDistributedConfig``), which
    require a live ``device_mesh`` and cannot be constructed CPU-only.
    ``split_param_groups`` turns one qualifying group into N sub-groups over the
    same params, which is structurally identical to supplying N param groups over
    those params. These CPU-only tests therefore use 1 vs 2 param groups over the
    same params as a faithful proxy for ``num_sub_groups`` = 1 vs > 1.
    """

    def _make_model(self) -> nn.Sequential:
        model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
            nn.Linear(10, 4, bias=False),
        )
        # Zero the weights for deterministic state values.
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        return model

    def _make_optimizer(
        self,
        params: Any,
        iterate_averaging_config: IterateAveragingConfig | None = None,
    ) -> DistributedShampoo:
        return DistributedShampoo(
            params,
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            iterate_averaging_config=iterate_averaging_config,
            distributed_config=replace(
                DefaultSingleDeviceDistributedConfig,
                target_parameter_dimensionality=2,
            ),
            grafting_config=AdaGradPreconditionerConfig(epsilon=0.001),
        )

    @staticmethod
    def _gpa_config() -> GeneralizedPrimalAveragingConfig:
        return GeneralizedPrimalAveragingConfig(
            eval_interp_coeff=0.5,
            train_interp_coeff=0.9,
        )

    @staticmethod
    def _param_ids_with_scalar(optimizer: DistributedShampoo, key: str) -> set[int]:
        """Param indices whose serialized state contains the given scalar key."""
        state = optimizer.state_dict()["state"]
        return {param_id for param_id, sub in state.items() if key in sub}

    def test_step_stored_under_every_param(self) -> None:
        # Setup: multi-param model in a single param group.
        model = self._make_model()
        params = list(model.parameters())
        self.assertEqual(len(params), 2)
        optimizer = self._make_optimizer(params)

        # Assert: step is checkpointed under EVERY param (not just the first).
        self.assertEqual(self._param_ids_with_scalar(optimizer, STEP), {0, 1})

        # Assert: all params alias one shared step tensor (no extra memory / hot-path cost).
        self.assertIs(
            optimizer.state[params[0]][STEP],
            optimizer.state[params[1]][STEP],
        )

    def test_lr_sum_and_train_mode_stored_under_every_param(self) -> None:
        # Setup: multi-param model with iterate averaging (adds lr_sum / train_mode).
        model = self._make_model()
        params = list(model.parameters())
        optimizer = self._make_optimizer(
            params, iterate_averaging_config=self._gpa_config()
        )

        # Assert: both scalars are checkpointed under EVERY param.
        self.assertEqual(self._param_ids_with_scalar(optimizer, LR_SUM), {0, 1})
        self.assertEqual(self._param_ids_with_scalar(optimizer, TRAIN_MODE), {0, 1})

        # Assert: all params alias one shared tensor for each scalar.
        self.assertIs(
            optimizer.state[params[0]][LR_SUM],
            optimizer.state[params[1]][LR_SUM],
        )
        self.assertIs(
            optimizer.state[params[0]][TRAIN_MODE],
            optimizer.state[params[1]][TRAIN_MODE],
        )

    def test_state_dict_key_set_invariant_to_param_group_structure(self) -> None:
        gpa_config = self._gpa_config()

        # num_sub_groups = 1 analog: both params in ONE group.
        single_group = self._make_optimizer(
            list(self._make_model().parameters()), gpa_config
        )
        # num_sub_groups = 2 analog: each param in its OWN group.
        multi_params = list(self._make_model().parameters())
        multi_group = self._make_optimizer(
            [{"params": [multi_params[0]]}, {"params": [multi_params[1]]}],
            gpa_config,
        )

        single_state = single_group.state_dict()["state"]
        multi_state = multi_group.state_dict()["state"]

        # Both layouts expose the same per-param key set (same param ids).
        self.assertEqual(single_state.keys(), multi_state.keys())

        # Each scalar is present under exactly the same (every) param in both layouts,
        # so the checkpoint key set does not depend on the param-group structure.
        for key in (STEP, LR_SUM, TRAIN_MODE):
            single_ids = {pid for pid, sub in single_state.items() if key in sub}
            multi_ids = {pid for pid, sub in multi_state.items() if key in sub}
            self.assertEqual(
                single_ids,
                multi_ids,
                msg=f"{key!r} key set differs between 1-group and 2-group layouts",
            )
            self.assertEqual(
                multi_ids,
                set(multi_state.keys()),
                msg=f"{key!r} should be stored under every param",
            )

    def test_param_keyed_cross_regroup_resume_preserves_scalars(self) -> None:
        """Emulates a distributed-checkpoint (param-keyed) resume across a change in
        param-group structure (num_sub_groups 1 -> 2).

        DCP matches optimizer state by param key, so every param must independently
        carry step / lr_sum / train_mode for a differently-grouped optimizer to be
        repopulated on resume. (Positional ``torch.optim.load_state_dict`` cannot
        cross group counts, so the per-param scalars are merged by key here, mirroring
        DCP resharding.)
        """
        gpa_config = self._gpa_config()

        # Save side: num_sub_groups = 1 analog (both params in one group).
        src_params = list(self._make_model().parameters())
        src = self._make_optimizer(src_params, gpa_config)
        # Set distinctive, non-default values on the shared scalar tensors.
        src.state[src_params[0]][STEP].fill_(7)
        src.state[src_params[0]][LR_SUM].fill_(0.5)
        saved = src.state_dict()

        # The param-keyed checkpoint carries the scalars under EVERY param -- this is
        # what makes a differently-grouped resume possible.
        for sub in saved["state"].values():
            self.assertIn(STEP, sub)
            self.assertIn(LR_SUM, sub)
            self.assertIn(TRAIN_MODE, sub)

        # Load side: num_sub_groups = 2 analog (each param its own group).
        dst_params = list(self._make_model().parameters())
        dst = self._make_optimizer(
            [{"params": [dst_params[0]]}, {"params": [dst_params[1]]}],
            gpa_config,
        )
        target = dst.state_dict()
        # DCP-style param-keyed merge: copy each param's saved scalars into the
        # (differently grouped) target checkpoint, then load with matching structure.
        for param_id, sub in target["state"].items():
            for key in (STEP, LR_SUM, TRAIN_MODE):
                sub[key] = saved["state"][param_id][key]
        dst.load_state_dict(target)

        # Every param recovers the saved step / lr_sum values; train_mode is present
        # under every param (its value follows the loader's mode by design).
        loaded = dst.state_dict()["state"]
        self.assertEqual(set(loaded.keys()), {0, 1})
        for sub in loaded.values():
            self.assertEqual(sub[STEP].item(), 7)
            self.assertAlmostEqual(sub[LR_SUM].item(), 0.5, places=6)
            self.assertIn(TRAIN_MODE, sub)

    def test_shared_scalars_track_runtime_mutations(self) -> None:
        """The aliased scalars are one tensor, so hot-path mutations (an optimizer
        step, an eval/train toggle) are visible identically through EVERY param's
        state -- the per-param checkpoint keys never diverge at runtime."""
        model = self._make_model()
        params = list(model.parameters())
        optimizer = self._make_optimizer(params, self._gpa_config())

        for param in params:
            param.grad = torch.ones_like(param)
        optimizer.step()

        # step advanced to 1 under every param, and lr_sum reads identically everywhere.
        self.assertTrue(all(optimizer.state[p][STEP].item() == 1 for p in params))
        self.assertEqual(len({optimizer.state[p][LR_SUM].item() for p in params}), 1)

        # train/eval toggles propagate through every param's shared train_mode.
        optimizer.eval()
        self.assertTrue(all(not optimizer.state[p][TRAIN_MODE].item() for p in params))
        optimizer.train()
        self.assertTrue(all(optimizer.state[p][TRAIN_MODE].item() for p in params))

    def test_divergent_aliased_scalar_raises_on_load(self) -> None:
        """A checkpoint storing DIVERGENT values for the same aliased per-group scalar
        under two params in one group must fail loudly on the native torch.optim load
        path (where the __setstate__ guard runs)."""
        params = list(self._make_model().parameters())
        optimizer = self._make_optimizer(params, self._gpa_config())
        state_dict = optimizer.state_dict()
        # Both params share one aliased lr_sum tensor; store differing values.
        state_dict["state"][0][LR_SUM] = torch.tensor(0.5)
        state_dict["state"][1][LR_SUM] = torch.tensor(0.9)
        self.assertRaisesRegex(
            ValueError,
            "aliased across parameters sharing a group",
            optimizer.load_state_dict,
            state_dict,
        )

    def test_consistent_aliased_scalar_loads_without_error(self) -> None:
        """Consistent duplicate values for an aliased scalar must load without error."""
        params = list(self._make_model().parameters())
        optimizer = self._make_optimizer(params, self._gpa_config())
        state_dict = optimizer.state_dict()
        # Same value under every param -> consistent, must not raise.
        for sub in state_dict["state"].values():
            sub[LR_SUM] = torch.tensor(0.5)
        optimizer.load_state_dict(state_dict)
        loaded = optimizer.state_dict()["state"]
        for sub in loaded.values():
            self.assertAlmostEqual(sub[LR_SUM].item(), 0.5, places=6)

    def test_schedule_free_nonzero_lr_sum_survives_round_trip(self) -> None:
        """ScheduleFree actually accumulates lr_sum (unlike GPA, which never updates it),
        so a NON-ZERO lr_sum must survive a save -> load round trip under every param."""
        sf_config = ScheduleFreeConfig(train_interp_coeff=0.9)
        src_params = list(self._make_model().parameters())
        src = self._make_optimizer(src_params, sf_config)
        # Take steps so lr_sum accumulates to a non-zero value.
        for _ in range(3):
            for param in src_params:
                param.grad = torch.ones_like(param)
            src.step()
        lr_sum_val = src.state[src_params[0]][LR_SUM].item()
        self.assertGreater(lr_sum_val, 0.0)
        saved = src.state_dict()

        dst_params = list(self._make_model().parameters())
        dst = self._make_optimizer(dst_params, sf_config)
        dst.load_state_dict(saved)

        loaded = dst.state_dict()["state"]
        for sub in loaded.values():
            self.assertAlmostEqual(sub[LR_SUM].item(), lr_sum_val, places=6)


class MultiParamStepStateTest(unittest.TestCase):
    """Verify that the STEP tensor is stored under every parameter's state
    when a group contains multiple parameters."""

    def setUp(self) -> None:
        # Two linear layers (bias=False) -> two distinct parameters in one group.
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
            nn.Linear(10, 3, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

    def test_step_tensor_present_in_all_params(self) -> None:
        """Every parameter in the group must have a 'step' key in its state."""
        for param in self._model.parameters():
            self.assertIn(
                "step",
                self._optimizer.state[param],
                msg="Expected 'step' key in every parameter's optimizer state",
            )

    def test_step_tensor_aliased_across_params(self) -> None:
        """All parameters in the same group must share the exact same step tensor
        (aliased, not copied) so that incrementing the counter is visible everywhere."""
        params = list(self._model.parameters())
        step_tensors = [self._optimizer.state[p]["step"] for p in params]

        # All step tensors should point to the same underlying storage.
        for i in range(1, len(step_tensors)):
            self.assertEqual(
                step_tensors[0].data_ptr(),
                step_tensors[i].data_ptr(),
                msg=f"Step tensor for param {i} is not aliased with param 0",
            )

    def test_step_increments_visible_to_all_params(self) -> None:
        """After an optimizer step, the shared step counter should be visible
        through every parameter's state."""
        # Provide gradients and take a step.
        for param in self._model.parameters():
            tensor = cast(torch.Tensor, param)
            tensor.grad = torch.rand_like(tensor)
        self._optimizer.step()

        params = list(self._model.parameters())
        for param in params:
            step_val = self._optimizer.state[param]["step"].item()
            self.assertEqual(
                step_val,
                1,
                msg="Step counter should be 1 after one optimizer step",
            )

    def test_step_in_state_dict_for_all_params(self) -> None:
        """The state_dict() output must include 'step' for every parameter index."""
        state_dict = self._optimizer.state_dict()
        for param_idx in state_dict["state"]:
            self.assertIn(
                "step",
                state_dict["state"][param_idx],
                msg=f"'step' missing from state_dict for param index {param_idx}",
            )


class MultiParamScheduleFreeStateTest(unittest.TestCase):
    """Verify that TRAIN_MODE and LR_SUM are stored under every parameter's
    state when using ScheduleFreeConfig with multiple parameters."""

    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
            nn.Linear(10, 3, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            iterate_averaging_config=ScheduleFreeConfig(train_interp_coeff=0.9),
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

    def test_train_mode_present_in_all_params(self) -> None:
        """Every parameter must have 'train_mode' in its optimizer state."""
        for param in self._model.parameters():
            self.assertIn(
                "train_mode",
                self._optimizer.state[param],
                msg="Expected 'train_mode' in every parameter's optimizer state",
            )

    def test_lr_sum_present_in_all_params(self) -> None:
        """Every parameter must have 'lr_sum' in its optimizer state."""
        for param in self._model.parameters():
            self.assertIn(
                "lr_sum",
                self._optimizer.state[param],
                msg="Expected 'lr_sum' in every parameter's optimizer state",
            )

    def test_train_mode_aliased_across_params(self) -> None:
        """All parameters must share the same train_mode tensor (aliased)."""
        params = list(self._model.parameters())
        train_mode_tensors = [self._optimizer.state[p]["train_mode"] for p in params]
        for i in range(1, len(train_mode_tensors)):
            self.assertEqual(
                train_mode_tensors[0].data_ptr(),
                train_mode_tensors[i].data_ptr(),
                msg=f"train_mode tensor for param {i} is not aliased with param 0",
            )

    def test_lr_sum_aliased_across_params(self) -> None:
        """All parameters must share the same lr_sum tensor (aliased)."""
        params = list(self._model.parameters())
        lr_sum_tensors = [self._optimizer.state[p]["lr_sum"] for p in params]
        for i in range(1, len(lr_sum_tensors)):
            self.assertEqual(
                lr_sum_tensors[0].data_ptr(),
                lr_sum_tensors[i].data_ptr(),
                msg=f"lr_sum tensor for param {i} is not aliased with param 0",
            )

    def test_train_eval_mode_visible_to_all_params(self) -> None:
        """Switching to eval mode should be reflected in every parameter's state."""
        # Take a step first to populate state.
        for param in self._model.parameters():
            tensor = cast(torch.Tensor, param)
            tensor.grad = torch.rand_like(tensor)
        self._optimizer.step()

        # Verify initial train mode.
        for param in self._model.parameters():
            self.assertTrue(
                self._optimizer.state[param]["train_mode"].item(),
                msg="Expected train_mode=True after training step",
            )

        # Switch to eval.
        self._optimizer.eval()
        for param in self._model.parameters():
            self.assertFalse(
                self._optimizer.state[param]["train_mode"].item(),
                msg="Expected train_mode=False after eval()",
            )

        # Switch back to train.
        self._optimizer.train()
        for param in self._model.parameters():
            self.assertTrue(
                self._optimizer.state[param]["train_mode"].item(),
                msg="Expected train_mode=True after train()",
            )

    def test_state_dict_contains_train_mode_and_lr_sum_for_all_params(self) -> None:
        """state_dict() must include train_mode and lr_sum for every parameter."""
        state_dict = self._optimizer.state_dict()
        for param_idx in state_dict["state"]:
            self.assertIn(
                "train_mode",
                state_dict["state"][param_idx],
                msg=f"'train_mode' missing from state_dict for param {param_idx}",
            )
            self.assertIn(
                "lr_sum",
                state_dict["state"][param_idx],
                msg=f"'lr_sum' missing from state_dict for param {param_idx}",
            )


class MultiParamGPAStateTest(unittest.TestCase):
    """Verify that TRAIN_MODE and LR_SUM are stored under every parameter's
    state when using GeneralizedPrimalAveragingConfig with multiple parameters."""

    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
            nn.Linear(10, 3, bias=False),
        )
        self._optimizer = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            iterate_averaging_config=GeneralizedPrimalAveragingConfig(
                eval_interp_coeff=0.5,
                train_interp_coeff=0.9,
            ),
            distributed_config=DefaultSingleDeviceDistributedConfig,
            grafting_config=None,
        )

    def test_train_mode_and_lr_sum_present_in_all_params(self) -> None:
        """With GPA config, every parameter must have train_mode and lr_sum."""
        for param in self._model.parameters():
            state = self._optimizer.state[param]
            self.assertIn("train_mode", state)
            self.assertIn("lr_sum", state)

    def test_shared_state_aliased_across_params_with_gpa(self) -> None:
        """train_mode and lr_sum must be aliased (same tensor) across all params."""
        params = list(self._model.parameters())
        first_train_mode = self._optimizer.state[params[0]]["train_mode"]
        first_lr_sum = self._optimizer.state[params[0]]["lr_sum"]

        for param in params[1:]:
            self.assertEqual(
                first_train_mode.data_ptr(),
                self._optimizer.state[param]["train_mode"].data_ptr(),
                msg="train_mode tensors must be aliased across parameters",
            )
            self.assertEqual(
                first_lr_sum.data_ptr(),
                self._optimizer.state[param]["lr_sum"].data_ptr(),
                msg="lr_sum tensors must be aliased across parameters",
            )

    def test_gpa_optimizer_step_updates_shared_state(self) -> None:
        """After an optimizer step, the shared lr_sum is visible identically through
        every parameter's state (the aliased tensor is mutated in place)."""
        for param in self._model.parameters():
            tensor = cast(torch.Tensor, param)
            tensor.grad = torch.rand_like(tensor)
        self._optimizer.step()

        params = list(self._model.parameters())
        lr_sum_val = self._optimizer.state[params[0]]["lr_sum"].item()

        # All params observe the same lr_sum value because they alias one tensor.
        for param in params[1:]:
            self.assertEqual(
                self._optimizer.state[param]["lr_sum"].item(),
                lr_sum_val,
                msg="lr_sum should be identical across all parameters",
            )
