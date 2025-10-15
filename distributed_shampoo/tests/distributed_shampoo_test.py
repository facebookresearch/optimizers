"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import gc
import logging
import re
import unittest
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.preconditioner.matrix_functions_types import (
    EigenConfig,
    PseudoInverseConfig,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSpectralDescentPreconditionerConfig,
    DistributedConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    PreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    ShampooPT2CompileConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
)
from torch import nn, Tensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
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
                {"momentum": 3.14},
                "Invalid param_group[MOMENTUM]=3.14. Must be [0.0, 1.0).",
            ),
            (
                {"dampening": -0.1},
                "Invalid param_group[DAMPENING]=-0.1. Must be [0.0, 1.0).",
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
                {"momentum": 0.0, "use_nesterov": True},
                [
                    "Nesterov flag is enabled but momentum parameter is zero! Continuing without using momentum or Nesterov acceleration..."
                ],
            ),
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
            r"group\[DISTRIBUTED_CONFIG\]=.*\.NotSupportedDistributedConfig\(.*\) not supported!",
            DistributedShampoo,
            params=self._model.parameters(),
            distributed_config=NotSupportedDistributedConfig(),
        )

        self.assertRaisesRegex(
            NotImplementedError,
            r"group\[DISTRIBUTED_CONFIG\]=.*FullyShardDistributedConfig\(.*ROUND_ROBIN.*\) not supported!",
            DistributedShampoo,
            params=self._model.parameters(),
            distributed_config=FullyShardDistributedConfig(
                param_assignment_strategy=FSDPParamAssignmentStrategy.ROUND_ROBIN
            ),
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
            momentum=0.0,
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
            self._optimizer = DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                betas=(0.9, 1.0),
                epsilon=1e-12,
                momentum=0.9,
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
                            "momentum": torch.tensor(
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
                            "momentum": torch.tensor(
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
                        "momentum": 0.9,
                        "dampening": 0.0,
                        "weight_decay": 0.0,
                        "max_preconditioner_dim": 5,
                        "precondition_frequency": 1,
                        "start_preconditioning_step": 1,
                        "use_nesterov": False,
                        "use_bias_correction": True,
                        "use_decoupled_weight_decay": True,
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
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrix_indices": {},
                            "inv_factor_matrices": {
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
                        "momentum": torch.tensor(
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
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrix_indices": {},
                            "inv_factor_matrices": {
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
                        "momentum": torch.tensor(
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
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_nesterov": False,
                    "use_bias_correction": True,
                    "use_decoupled_weight_decay": True,
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
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrix_indices": {},
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
                        "momentum": torch.tensor(
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
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrix_indices": {},
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
                        "momentum": torch.tensor(
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
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_nesterov": False,
                    "use_bias_correction": True,
                    "use_decoupled_weight_decay": True,
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
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrix_indices": {},
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
                        "momentum": torch.tensor(
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
                                0: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                                1: torch.tensor(
                                    [
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 0.0, 0.0],
                                    ]
                                ),
                            },
                            "factor_matrix_indices": {},
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
                        "momentum": torch.tensor(
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
                    "momentum": 0.9,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "use_nesterov": False,
                    "use_bias_correction": True,
                    "use_decoupled_weight_decay": True,
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
        class SpectralDescentPreconditionerConfigWithLambda(
            SpectralDescentPreconditionerConfig
        ):
            """
            Creating a preconditioner config with a dummy lambda function to make sure the
            warning from `_post_state_dict_hook` emit.
            """

            scale_fn: Callable[[Tensor], float | Tensor] = lambda grad: 1.0

        self._optimizer.param_groups[0]["preconditioner_config"] = (
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
            momentum=0.0,
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
        expected_msg = "PT2 will recompile because the gradient selction of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model."
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
