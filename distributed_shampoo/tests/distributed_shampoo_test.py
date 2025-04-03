"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import copy
import re
import unittest
from itertools import chain
from typing import Any
from unittest import mock

import torch
from distributed_shampoo import distributed_shampoo
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    DDPShampooConfig,
    DefaultShampooConfig,
    DistributedConfig,
    GraftingConfig,
    PreconditionerConfig,
    SGDGraftingConfig,
    ShampooPreconditionerConfig,
)
from matrix_functions_types import EigenConfig, EigendecompositionConfig, RootInvConfig
from torch import nn


class DistributedShampooInitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )

    def test_invalid_preconditioner_config(self) -> None:
        with mock.patch.object(
            distributed_shampoo,
            "type",
            side_effect=lambda object: {
                ShampooPreconditionerConfig: PreconditionerConfig
            }.get(type(object), type(object)),
        ):
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape("group[PRECONDITIONER_CONFIG]=ShampooPreconditionerConfig"),
                DistributedShampoo,
                self._model.parameters(),
                preconditioner_config=DefaultShampooConfig,
            )

        with mock.patch.object(
            distributed_shampoo,
            "isinstance",
            side_effect=lambda object, classinfo: False
            if classinfo in (RootInvConfig, EigendecompositionConfig)
            else None,
        ):
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape(
                    "group[PRECONDITIONER_CONFIG].amortized_computation_config=EigenConfig(retry_double_precision=True, eigendecomposition_offload_device='', exponent_multiplier=1.0, enhance_stability=False) not supported!"
                ),
                DistributedShampoo,
                self._model.parameters(),
                preconditioner_config=DefaultShampooConfig,
            )

    def test_invalid_grafting_config(self) -> None:
        with mock.patch.object(
            distributed_shampoo,
            "type",
            side_effect=lambda object: {SGDGraftingConfig: GraftingConfig}.get(
                type(object), type(object)
            ),
        ):
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape("group[GRAFTING_CONFIG]=SGDGraftingConfig"),
                DistributedShampoo,
                self._model.parameters(),
                grafting_config=SGDGraftingConfig(),  # type: ignore[abstract]
            )

    def test_invalid_with_incorrect_hyperparameter_setting(self) -> None:
        incorrect_hyperparameter_setting_and_expected_error_msg: list[
            tuple[dict[str, Any], str]
        ] = [
            (
                {"lr": -0.1},
                "Invalid learning rate: -0.1. Must be >= 0.0.",
            ),
            (
                {"betas": (-0.1, 1.0)},
                "Invalid beta parameter at index 0: -0.1. Must be in [0.0, 1.0).",
            ),
            (
                {"betas": (0.9, 0.0)},
                "Invalid beta parameter at index 1: 0.0. Must be in (0.0, 1.0].",
            ),
            (
                {"beta3": -0.1},
                "Invalid beta3 parameter: -0.1. Must be in [0.0, 1.0).",
            ),
            (
                {"epsilon": 0.0},
                "Invalid epsilon value: 0.0. Must be > 0.0.",
            ),
            (
                {"momentum": 3.14},
                "Invalid momentum parameter: 3.14. Must be [0.0, 1.0).",
            ),
            (
                {"dampening": -0.1},
                "Invalid damping parameter: -0.1. Must be [0.0, 1.0).",
            ),
            (
                {"weight_decay": -0.1},
                "Invalid weight_decay value: -0.1. Must be >= 0.0.",
            ),
            (
                {"max_preconditioner_dim": 0},
                "Invalid max preconditioner dimension: 0. Must be >= 1.",
            ),
            (
                {"precondition_frequency": 0},
                "Invalid precondition frequency: 0. Must be >= 1.",
            ),
            (
                {"start_preconditioning_step": -2},
                "Invalid start preconditioning step: -2. Must be >= -1.",
            ),
            (
                {"start_preconditioning_step": 10, "precondition_frequency": 100},
                "Invalid start_preconditioning_step value: 10. Must be >= precondition_frequency=100.",
            ),
            (
                {
                    "preconditioner_config": ShampooPreconditionerConfig(
                        amortized_computation_config=EigenConfig(
                            exponent_multiplier=0.5
                        )
                    )
                },
                "preconditioner_config.amortized_computation_config.exponent_multiplier is not supported. Please use PreconditionerConfig.inverse_exponent_override instead.",
            ),
        ]

        for (
            incorrect_hyperparameter_setting,
            expected_error_msg,
        ) in incorrect_hyperparameter_setting_and_expected_error_msg:
            with self.subTest(
                incorrect_hyperparameter_setting=incorrect_hyperparameter_setting,
                expected_error_msg=expected_error_msg,
            ):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(expected_error_msg),
                    DistributedShampoo,
                    self._model.parameters(),
                    **incorrect_hyperparameter_setting,
                )

    def test_nesterov_and_zero_momentum(self) -> None:
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            DistributedShampoo(
                self._model.parameters(),
                momentum=0.0,
                use_nesterov=True,
            )

            self.assertIn(
                "Nesterov flag is enabled but momentum parameter is zero! "
                "Continuing without using momentum or Nesterov acceleration...",
                [r.msg for r in cm.records],
            )

    def test_invalid_distributed_config(self) -> None:
        with mock.patch.object(
            distributed_shampoo,
            "type",
            side_effect=lambda object: DistributedConfig,
        ):
            self.assertRaisesRegex(
                NotImplementedError,
                re.escape(
                    "distributed_config=DDPShampooConfig(communication_dtype=<CommunicationDType.DEFAULT: 1>, "
                    "num_trainers_per_group=-1, communicate_params=False) not supported!"
                ),
                DistributedShampoo,
                params=self._model.parameters(),
                distributed_config=DDPShampooConfig(),
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
            distributed_config=None,
            # Explicity set grafting_config=None to test the case that no grafting is used.
            grafting_config=None,
        )

    def test_step_with_closure(self) -> None:
        # Test the case without closure, the loss returned by step() is None.
        self._optimizer.zero_grad()
        self._model[0].weight.grad = torch.rand_like(self._model[0].weight)
        self.assertIsNone(self._optimizer.step(closure=None))

        # Test the case that the closure returns a scalar.
        def closure() -> float:
            self._optimizer.zero_grad()
            self._model[0].weight.grad = torch.rand_like(self._model[0].weight)
            return 1.0

        self.assertEqual(self._optimizer.step(closure=closure), 1.0)

    def test_step_with_empty_grad_list(self) -> None:
        # Because the grad_list is empty, after taking five steps, the internal step should be 0.
        for _ in range(5):
            self._optimizer.zero_grad()
            self._optimizer.step()

        actual_step = self._optimizer.distributed_state_dict(
            key_to_param=self._model.named_parameters(),
            save_param_groups=True,
        )["state"]["0.weight"]['["step"]']
        torch.testing.assert_close(
            actual_step,
            torch.as_tensor(0),
            rtol=0,
            atol=0,
        )


class DistributedShampooStateDictTest(unittest.TestCase):
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
            distributed_config=None,
            grafting_config=AdaGradGraftingConfig(
                epsilon=0.001,
            ),
        )
        self._distributed_state_dict: dict[str, Any] = {
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
                    "grafting_config": AdaGradGraftingConfig(
                        epsilon=0.001,
                    ),
                    "use_merge_dims": True,
                    "preconditioner_dtype": torch.float32,
                    "preconditioner_config": DefaultShampooConfig,
                }
            },
        }

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
        self.assertEqual(state_dict_with_param_groups.keys(), {"state", "param_groups"})

        torch.testing.assert_close(
            state_dict_with_param_groups["state"], self._distributed_state_dict["state"]
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
        expected_distributed_state_dict = copy.deepcopy(self._distributed_state_dict)

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
        self._distributed_state_dict["param_groups"]["1.weight"] = {}

        self.assertRaisesRegex(
            ValueError,
            re.escape("Different param_groups count: 1 vs 2"),
            self._optimizer.load_distributed_state_dict,
            state_dict=self._distributed_state_dict,
            key_to_param=self._model.named_parameters(),
            save_param_groups=True,
        )

        # Remove "0.weight" so param_groups_to_load has "1.weight" only but param_groups needs "0.weight".
        del self._distributed_state_dict["param_groups"]["0.weight"]

        self.assertRaisesRegex(
            ValueError,
            re.escape("Param group 0.weight not found in param_groups_to_load!"),
            self._optimizer.load_distributed_state_dict,
            state_dict=self._distributed_state_dict,
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
            distributed_config=None,
            # Explicity set grafting_config=None to test the case that no grafting is used.
            grafting_config=None,
        )

    def test_step_with_consistent_grads(self) -> None:
        with self.assertNoLogs(level="WARNING"):
            self._optimizer.zero_grad()
            self._model[0].weight.grad = torch.rand_like(self._model[0].weight)
            self._optimizer.step()

            self._optimizer.zero_grad()
            self._model[0].weight.grad = torch.rand_like(self._model[0].weight)
            self._optimizer.step()

    def test_step_with_none_grads(self) -> None:
        expected_msg = "PT2 will recompile because the gradient selction of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model."
        ending_msg = "Changed gradient selector indices: [0, 1]"
        with self.assertLogs(level="WARNING") as cm:
            self._optimizer.zero_grad()
            self._model[0].weight.grad = torch.rand_like(self._model[0].weight)
            self._optimizer.step()

            self._optimizer.zero_grad()  # Implicitly set grad=None in second step
            self._optimizer.step()
            msgs = [r.msg for r in cm.records]

        self.assertEqual(len(msgs), 1)
        self.assertIn(expected_msg, msgs[0])
        self.assertIn(ending_msg, msgs[0])
