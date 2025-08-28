"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import abc
import copy
import gc
import re
import unittest
from dataclasses import dataclass, replace
from itertools import chain
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
from torch import nn
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
    class DistributedStateDictTestBase(abc.ABC, unittest.TestCase):
        @property
        @abc.abstractmethod
        def _preconditioner_config(self) -> PreconditionerConfig: ...

        @property
        @abc.abstractmethod
        def _distributed_state_dict(self) -> dict[str, Any]: ...

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

        def test_state_dict(self) -> None:
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape(
                    "Distributed Shampoo does not support the standard state_dict() method for checkpointing!"
                ),
                self._optimizer.state_dict,
            )

        def test_load_state_dict(self) -> None:
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape(
                    "Distributed Shampoo does not support the standard load_state_dict() method for checkpointing!"
                ),
                self._optimizer.load_state_dict,
                state_dict={},
            )

        def test_distributed_state_dict(self) -> None:
            state_dict_with_param_groups = self._optimizer.distributed_state_dict(
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )
            self.assertEqual(
                state_dict_with_param_groups.keys(), {"state", "param_groups"}
            )

            torch.testing.assert_close(
                state_dict_with_param_groups["state"],
                self._distributed_state_dict["state"],
            )
            self.assertEqual(
                state_dict_with_param_groups["param_groups"],
                self._distributed_state_dict["param_groups"],
            )

            state_dict_without_param_groups = self._optimizer.distributed_state_dict(
                key_to_param=self._model.named_parameters(),
                save_param_groups=False,
            )
            self.assertEqual(state_dict_without_param_groups.keys(), {"state"})

            torch.testing.assert_close(
                state_dict_without_param_groups["state"],
                self._distributed_state_dict["state"],
            )

        def test_load_distributed_state_dict(self) -> None:
            expected_distributed_state_dict = copy.deepcopy(
                self._distributed_state_dict
            )

            self._optimizer.load_distributed_state_dict(
                state_dict=self._distributed_state_dict,
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )

            actual_state_dict = self._optimizer.distributed_state_dict(
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )

            self.assertEqual(
                actual_state_dict.keys(), expected_distributed_state_dict.keys()
            )
            torch.testing.assert_close(
                actual_state_dict["state"], expected_distributed_state_dict["state"]
            )
            self.assertEqual(
                actual_state_dict["param_groups"],
                expected_distributed_state_dict["param_groups"],
            )

        def test_load_distributed_state_dict_with_mismatch_param_groups(self) -> None:
            # Add "1.weight" so param_groups_to_load has two fields (i.e., "0.weight" and "1.weight")
            # but param_groups only needs one (i.e., "0.weight").
            distributed_state_dict_copy = copy.deepcopy(self._distributed_state_dict)
            distributed_state_dict_copy["param_groups"]["1.weight"] = {}

            self.assertRaisesRegex(
                ValueError,
                re.escape("Different param_groups count: 1 vs 2"),
                self._optimizer.load_distributed_state_dict,
                state_dict=distributed_state_dict_copy,
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )

            # Remove "0.weight" so param_groups_to_load has "1.weight" only but param_groups needs "0.weight".
            del distributed_state_dict_copy["param_groups"]["0.weight"]

            self.assertRaisesRegex(
                ValueError,
                re.escape("Param group 0.weight not found in param_groups_to_load!"),
                self._optimizer.load_distributed_state_dict,
                state_dict=distributed_state_dict_copy,
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )

        def test_load_distributed_state_dict_with_missing_param_key(self) -> None:
            self.assertRaisesRegex(
                KeyError,
                re.escape("Parameter key 0.weight not found in key_to_param mapping!"),
                self._optimizer.load_distributed_state_dict,
                state_dict=self._distributed_state_dict,
                # Instead of providing self._model.named_parameters(), we provide an empty list
                # to trigger the missing key check error.
                key_to_param=iter([]),
                save_param_groups=False,
                enable_missing_key_check=True,
            )

            with self.assertLogs(
                level="WARNING",
            ) as cm:
                self._optimizer.load_distributed_state_dict(
                    state_dict=self._distributed_state_dict,
                    # Instead of providing self._model.named_parameters(), we provide an empty list
                    # to trigger the missing key check warning.
                    key_to_param=iter([]),
                    save_param_groups=False,
                    enable_missing_key_check=False,
                )
                self.assertCountEqual(
                    [r.msg for r in cm.records],
                    ["Parameter key 0.weight not found in key_to_param mapping!"],
                )

        def test_load_distributed_state_dict_with_missing_param(self) -> None:
            # Instead of providing self._distributed_state_dict and self._model.named_parameters()
            # (which contains parameter "0.weight"), we provide an additional param (i.e., "1.weight")
            # to trigger the missing key error and warning.
            state_dict_to_load_copy = copy.deepcopy(self._distributed_state_dict)
            state_dict_to_load_copy["state"]["1.weight"] = torch.tensor(0)
            key_to_param_copy = chain(
                self._model.named_parameters(), iter([("1.weight", torch.tensor(1))])
            )
            self.assertRaisesRegex(
                KeyError,
                re.escape("Parameter 1 not found in state!"),
                self._optimizer.load_distributed_state_dict,
                state_dict=state_dict_to_load_copy,
                key_to_param=key_to_param_copy,
                save_param_groups=False,
                enable_missing_key_check=True,
            )

            # Re-populate key_to_param_copy because it is an iterator that was consumed by the previous call.
            key_to_param_copy = chain(
                self._model.named_parameters(), iter([("1.weight", torch.tensor(1))])
            )
            with self.assertLogs(
                level="WARNING",
            ) as cm:
                self._optimizer.load_distributed_state_dict(
                    state_dict=state_dict_to_load_copy,
                    key_to_param=key_to_param_copy,
                    save_param_groups=False,
                    enable_missing_key_check=False,
                )
                self.assertCountEqual(
                    [r.msg for r in cm.records],
                    ["Parameter 1 not found in state!"],
                )

    class NoPreconditionerStateDictTestBase(DistributedStateDictTestBase):
        """A base class for methods that do not have a preconditioner."""

        @property
        def _distributed_state_dict(self) -> dict[str, Any]:
            return {
                "state": {
                    "0.weight": {
                        '["step"]': torch.tensor(0),
                        '["block_0", "adagrad"]': torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        '["block_1", "adagrad"]': torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        '["block_0", "momentum"]': torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        '["block_1", "momentum"]': torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        '["block_0", "filtered_grad"]': torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                        '["block_1", "filtered_grad"]': torch.tensor(
                            [
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                            ]
                        ),
                    },
                },
                "param_groups": {
                    "0.weight": {
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
                        "grafting_config": AdaGradPreconditionerConfig(
                            epsilon=0.001,
                        ),
                        "use_pin_memory": False,
                        "distributed_config": replace(
                            DefaultSingleDeviceDistributedConfig,
                            target_parameter_dimensionality=2,
                        ),
                        "preconditioner_config": self._preconditioner_config,
                    }
                },
            }


class ShampooDistributedStateDictTest(AbstractTest.DistributedStateDictTestBase):
    @property
    def _preconditioner_config(self) -> RootInvShampooPreconditionerConfig:
        return DefaultShampooConfig

    @property
    def _distributed_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                "0.weight": {
                    '["step"]': torch.tensor(0),
                    '["block_0", "shampoo", "factor_matrices", 0]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices", 1]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 0]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 1]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices", 0]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices", 1]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 0]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 1]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_0", "adagrad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "adagrad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "momentum"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "momentum"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "filtered_grad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "filtered_grad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            },
            "param_groups": {
                "0.weight": {
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
                    "grafting_config": AdaGradPreconditionerConfig(
                        epsilon=0.001,
                    ),
                    "use_pin_memory": False,
                    "distributed_config": replace(
                        DefaultSingleDeviceDistributedConfig,
                        target_parameter_dimensionality=2,
                    ),
                    "preconditioner_config": self._preconditioner_config,
                }
            },
        }


class EigendecomposedShampooDistributedStateDictTest(
    AbstractTest.DistributedStateDictTestBase
):
    @property
    def _preconditioner_config(self) -> EigendecomposedShampooPreconditionerConfig:
        return EigendecomposedShampooPreconditionerConfig()

    @property
    def _distributed_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                "0.weight": {
                    '["step"]': torch.tensor(0),
                    '["block_0", "shampoo", "factor_matrices", 0]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices", 1]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices_eigenvectors", 0]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices_eigenvectors", 1]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices_eigenvalues", 0]': torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ),
                    '["block_0", "shampoo", "factor_matrices_eigenvalues", 1]': torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ),
                    '["block_1", "shampoo", "factor_matrices", 0]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices", 1]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices_eigenvectors", 0]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices_eigenvectors", 1]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices_eigenvalues", 0]': torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ),
                    '["block_1", "shampoo", "factor_matrices_eigenvalues", 1]': torch.tensor(
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ),
                    '["block_0", "adagrad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "adagrad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "momentum"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "momentum"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "filtered_grad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "filtered_grad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            },
            "param_groups": {
                "0.weight": {
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
                    "grafting_config": AdaGradPreconditionerConfig(
                        epsilon=0.001,
                    ),
                    "use_pin_memory": False,
                    "distributed_config": replace(
                        DefaultSingleDeviceDistributedConfig,
                        target_parameter_dimensionality=2,
                    ),
                    "preconditioner_config": self._preconditioner_config,
                }
            },
        }


class EigenvalueCorrectedShampooDistributedStateDictTest(
    AbstractTest.DistributedStateDictTestBase
):
    @property
    def _preconditioner_config(self) -> EigenvalueCorrectedShampooPreconditionerConfig:
        return DefaultEigenvalueCorrectedShampooConfig

    @property
    def _distributed_state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                "0.weight": {
                    '["step"]': torch.tensor(0),
                    '["block_0", "shampoo", "factor_matrices", 0]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices", 1]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices_eigenvectors", 0]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices_eigenvectors", 1]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_0", "shampoo", "corrected_eigenvalues"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices", 0]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices", 1]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices_eigenvectors", 0]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices_eigenvectors", 1]': torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    '["block_1", "shampoo", "corrected_eigenvalues"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "adagrad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "adagrad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "momentum"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "momentum"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "filtered_grad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "filtered_grad"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                },
            },
            "param_groups": {
                "0.weight": {
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
                    "grafting_config": AdaGradPreconditionerConfig(
                        epsilon=0.001,
                    ),
                    "use_pin_memory": False,
                    "distributed_config": replace(
                        DefaultSingleDeviceDistributedConfig,
                        target_parameter_dimensionality=2,
                    ),
                    "preconditioner_config": self._preconditioner_config,
                }
            },
        }


class SignDescentDistributedStateDictTest(
    AbstractTest.NoPreconditionerStateDictTestBase
):
    @property
    def _preconditioner_config(self) -> SignDescentPreconditionerConfig:
        return DefaultSignDescentPreconditionerConfig


class SpectralDescentDistributedStateDictTest(
    AbstractTest.NoPreconditionerStateDictTestBase
):
    @property
    def _preconditioner_config(self) -> SpectralDescentPreconditionerConfig:
        return DefaultSpectralDescentPreconditionerConfig


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
