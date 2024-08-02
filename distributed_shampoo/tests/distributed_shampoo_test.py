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
from typing import Any, Dict, List, Tuple
from unittest import mock

import torch
from distributed_shampoo import distributed_shampoo
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    DDPShampooConfig,
    GRAFTING_PRECONDITIONER_LIST,
    MASKED_FILTERED_GRAD_LIST,
    MASKED_MOMENTUM_LIST,
    PrecisionConfig,
    SGDGraftingConfig,
    SHAMPOO_PRECONDITIONER_LIST,
)
from distributed_shampoo.utils.shampoo_preconditioner_list import (
    ShampooPreconditionerList,
)
from distributed_shampoo.utils.shampoo_quantization import QuantizedTensorList
from torch import nn


class DistributedShampooInitTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )

    def test_invalid_grafting_config(self) -> None:
        with mock.patch.object(
            distributed_shampoo,
            "isinstance",
            side_effect=lambda object, classinfo: (
                False if classinfo == SGDGraftingConfig else None
            ),
        ), self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Unsupported grafting config: group[GRAFTING_CONFIG]=SGDGraftingConfig"
            ),
        ):
            DistributedShampoo(
                self._model.parameters(),
                grafting_config=SGDGraftingConfig(),
            )

    def test_invalid_with_incorrect_hyperparameter_setting(self) -> None:
        incorrect_hyperparameter_setting_and_expected_error_msg: List[
            Tuple[Dict[str, Any], str]
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
                {"inv_root_override": [-1, 2, 3]},
                "Invalid exponent override list: [-1, 2, 3]. All values must be >= 0.",
            ),
            (
                {"inv_root_override": -1},
                "Invalid exponent override: -1. Must be >= 0.",
            ),
            (
                {"start_preconditioning_step": 10, "precondition_frequency": 100},
                "Invalid start_preconditioning_step value: 10. Must be >= precondition_frequency=100.",
            ),
        ]

        for (
            incorrect_hyperparameter_setting,
            expected_error_msg,
        ) in incorrect_hyperparameter_setting_and_expected_error_msg:
            with self.subTest(
                incorrect_hyperparameter_setting=incorrect_hyperparameter_setting,
                expected_error_msg=expected_error_msg,
            ), self.assertRaisesRegex(ValueError, re.escape(expected_error_msg)):
                DistributedShampoo(
                    self._model.parameters(),
                    **incorrect_hyperparameter_setting,
                )

    def test_invalid_pytorch_compile_setting(self) -> None:
        with mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ), self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Backend does NOT support Pytorch 2.0 compile. Switch to use_pytorch_compile=False."
            ),
        ):
            DistributedShampoo(
                self._model.parameters(),
                use_pytorch_compile=True,
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
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "self._distributed_config=DDPShampooConfig(communication_dtype=<CommunicationDType.DEFAULT: 0>, "
                "num_trainers_per_group=-1, communicate_params=False) not supported!"
            ),
        ), mock.patch.object(
            distributed_shampoo,
            "isinstance",
            side_effect=lambda object, classinfo: (
                False
                if classinfo == DDPShampooConfig
                else isinstance(object, classinfo)
            ),
        ):
            DistributedShampoo(
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
        self._model[0].weight.grad = torch.rand(10, 5)
        self.assertIsNone(self._optimizer.step(closure=None))

        # Test the case that the closure returns a scalar.
        def closure() -> float:
            self._optimizer.zero_grad()
            self._model[0].weight.grad = torch.rand(10, 5)
            return 1.0

        self.assertEqual(self._optimizer.step(closure=closure), 1.0)

    def test_step_with_empty_grad_list(self) -> None:
        # Test the case that the grad_list is empty.
        self._optimizer.zero_grad()
        with mock.patch.object(
            ShampooPreconditionerList, "update_preconditioners"
        ) as mock_upgrade_preconditioners:
            self._optimizer.step()
            # Because the gradient list is empty, the preconditioners should not be updated.
            mock_upgrade_preconditioners.assert_not_called()


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
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            distributed_config=None,
            grafting_config=AdaGradGraftingConfig(
                epsilon=0.001,
            ),
        )
        self._distributed_state_dict: Dict[str, Any] = {
            "state": {
                "0.weight": {
                    '["step"]': torch.tensor(0),
                    '["block_0", "shampoo", "factor_matrices", 0, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "factor_matrices", 1, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 0, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "inv_factor_matrices", 1, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "shampoo", "is_factor_matrices_diagonal", 0]': torch.tensor(
                        True
                    ),
                    '["block_0", "shampoo", "is_factor_matrices_diagonal", 1]': torch.tensor(
                        True
                    ),
                    '["block_1", "shampoo", "factor_matrices", 0, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "factor_matrices", 1, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 0, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "inv_factor_matrices", 1, "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "shampoo", "is_factor_matrices_diagonal", 0]': torch.tensor(
                        True
                    ),
                    '["block_1", "shampoo", "is_factor_matrices_diagonal", 1]': torch.tensor(
                        True
                    ),
                    '["block_0", "adagrad", "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "adagrad", "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_0", "filtered_grad", "quantized_values"]': torch.tensor(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                    '["block_1", "filtered_grad", "quantized_values"]': torch.tensor(
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
                    "momentum": 0.0,
                    "dampening": 0.0,
                    "weight_decay": 0.0,
                    "max_preconditioner_dim": 5,
                    "precondition_frequency": 1,
                    "start_preconditioning_step": 1,
                    "inv_root_override": 0,
                    "exponent_multiplier": 1.0,
                    "use_nesterov": False,
                    "use_bias_correction": True,
                    "use_decoupled_weight_decay": True,
                    "grafting_config": AdaGradGraftingConfig(
                        epsilon=0.001,
                    ),
                    "use_merge_dims": True,
                    "preconditioner_dtype": None,
                    "precision_config": PrecisionConfig(),
                }
            },
        }

    def test_state_dict(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Distributed Shampoo does not support the standard state_dict() method for checkpointing!"
            ),
        ):
            self._optimizer.state_dict()

    def test_load_state_dict(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Distributed Shampoo does not support the standard load_state_dict() method for checkpointing!"
            ),
        ):
            self._optimizer.load_state_dict(state_dict={})

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

        with self.assertRaisesRegex(
            ValueError, re.escape("Different param_groups count: 1 vs 2")
        ):
            self._optimizer.load_distributed_state_dict(
                state_dict=self._distributed_state_dict,
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )

        # Remove "0.weight" so param_groups_to_load has "1.weight" only but param_groups needs "0.weight".
        del self._distributed_state_dict["param_groups"]["0.weight"]

        with self.assertRaisesRegex(
            ValueError,
            re.escape("Param group 0.weight not found in param_groups_to_load!"),
        ):
            self._optimizer.load_distributed_state_dict(
                state_dict=self._distributed_state_dict,
                key_to_param=self._model.named_parameters(),
                save_param_groups=True,
            )

    def test_load_distributed_state_dict_with_missing_param_key(self) -> None:
        with self.assertRaisesRegex(
            KeyError,
            re.escape("Parameter key 0.weight not found in key_to_param mapping!"),
        ):
            self._optimizer.load_distributed_state_dict(
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
        with self.assertRaisesRegex(
            KeyError, re.escape("Parameter 1 not found in state!")
        ):
            self._optimizer.load_distributed_state_dict(
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


class DistributedShampooTrackRootInvResidualsTest(unittest.TestCase):
    def _get_track_root_inverse_residuals_output(self, dtype: torch.dtype) -> List[str]:
        # Create a model and a DistributedShampoo optimizer with enabled track_root_inv_residuals and corresponding dtype.
        # The dtype of the model and the optimizer are the same.
        model = nn.Sequential(nn.Linear(2, 1, bias=False))
        model[0].weight.data = torch.tensor([1.0, 2.0], dtype=dtype)
        optimizer = DistributedShampoo(
            params=model.parameters(),
            precondition_frequency=2,
            start_preconditioning_step=2,
            precision_config=PrecisionConfig(
                computation_dtype=dtype,
                factor_matrix_dtype=dtype,
                inv_factor_matrix_dtype=dtype,
                filtered_grad_dtype=dtype,
                momentum_dtype=dtype,
                grafting_state_dtype=dtype,
            ),
            track_root_inv_residuals=True,
        )

        # Run two steps of the optimizer to compute the root inverse residuals.
        # Because precondition_frequency and start_preconditioning_step are both 2, there should be one call of
        # _compute_and_log_root_inverse_residuals().
        with self.assertLogs(level="DEBUG") as cm:
            model[0].weight.grad = torch.tensor([1.0, 0.0], dtype=dtype)
            optimizer.step()
            model[0].weight.grad = torch.tensor([0.0, 1.0], dtype=dtype)
            optimizer.step()
            return [r.msg for r in cm.records]

    def test_compute_and_log_root_inverse_residuals(self) -> None:
        # Test the cases that tracking root inverse residuals support both float32 and float64.
        for dtype, expected_relative_error in [
            (torch.float32, 1e-3),
            (torch.float64, 1e-7),
        ]:
            with self.subTest(dtype=dtype):
                msgs = self._get_track_root_inverse_residuals_output(dtype=dtype)
                self.assertIn("Group Index: 0", msgs)
                self.assertIn(
                    f"Expect Relative Error <= {expected_relative_error}", msgs
                )

        # Test the case that tracking root inverse residuals does not support float16.
        msgs = self._get_track_root_inverse_residuals_output(dtype=torch.float16)
        self.assertEqual(
            msgs,
            [
                "Expected relative error/residual not supported for precision lower than float32."
            ],
        )


class DistributedShampooPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(5, 10, bias=False),
        )

    def _instantiate_optimizer(
        self, precision_config: PrecisionConfig
    ) -> DistributedShampoo:
        return DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            momentum=0.99,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=1,
            distributed_config=None,
            grafting_config=AdaGradGraftingConfig(
                epsilon=0.001,
            ),
            precision_config=precision_config,
        )

    def _assert_equal_state_dtype(
        self,
        quantized_tensor_list: QuantizedTensorList,
        computation_dtype: torch.dtype,
        quantized_dtype: torch.dtype,
    ) -> None:
        self.assertEqual(quantized_tensor_list.computation_dtype, computation_dtype)
        self.assertEqual(quantized_tensor_list.quantized_dtype, quantized_dtype)
        self.assertIsNone(quantized_tensor_list.dequantized_value_list)

    def _assert_state_list_dtype(
        self, state_list: Dict[str, Any], precision_config: PrecisionConfig
    ) -> None:
        # TODO: is it possible to avoid accessing private field _masked_kronecker_factors_list?
        for kronecker_factor in state_list[
            SHAMPOO_PRECONDITIONER_LIST
        ]._masked_kronecker_factors_list:
            self._assert_equal_state_dtype(
                kronecker_factor.factor_matrices,
                precision_config.computation_dtype,
                precision_config.factor_matrix_dtype,
            )
            self._assert_equal_state_dtype(
                kronecker_factor.inv_factor_matrices,
                precision_config.computation_dtype,
                precision_config.inv_factor_matrix_dtype,
            )
        self._assert_equal_state_dtype(
            state_list[GRAFTING_PRECONDITIONER_LIST]._masked_preconditioner_list,
            precision_config.computation_dtype,
            precision_config.grafting_state_dtype,
        )
        self._assert_equal_state_dtype(
            state_list[MASKED_FILTERED_GRAD_LIST],
            precision_config.computation_dtype,
            precision_config.filtered_grad_dtype,
        )
        self._assert_equal_state_dtype(
            state_list[MASKED_MOMENTUM_LIST],
            precision_config.computation_dtype,
            precision_config.momentum_dtype,
        )

    def test_precision_configs(self) -> None:
        precision_configs = [
            PrecisionConfig(computation_dtype=torch.float16),
            PrecisionConfig(factor_matrix_dtype=torch.float16),
            PrecisionConfig(inv_factor_matrix_dtype=torch.float16),
            PrecisionConfig(filtered_grad_dtype=torch.float16),
            PrecisionConfig(momentum_dtype=torch.float16),
            PrecisionConfig(grafting_state_dtype=torch.float16),
            PrecisionConfig(
                factor_matrix_dtype=torch.float16, inv_factor_matrix_dtype=torch.float16
            ),
            PrecisionConfig(
                factor_matrix_dtype=torch.float16,
                inv_factor_matrix_dtype=torch.float16,
                grafting_state_dtype=torch.float16,
                filtered_grad_dtype=torch.float16,
                momentum_dtype=torch.float16,
            ),
        ]

        for precision_config in precision_configs:
            with self.subTest(precision_config=precision_config):
                optimizer = self._instantiate_optimizer(
                    precision_config=precision_config
                )
                for state_list in optimizer._per_group_state_lists:
                    self._assert_state_list_dtype(state_list, precision_config)

                for _ in range(2):
                    optimizer.step()
                    for state_list in optimizer._per_group_state_lists:
                        self._assert_state_list_dtype(state_list, precision_config)

    def test_setting_both_preconditioner_dtype_and_precision_config(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Both preconditioner_dtype and precision_config are provided. Please use only precision_config as preconditioner_dtype is deprecated."
            ),
        ):
            DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                preconditioner_dtype=torch.float16,
                precision_config=PrecisionConfig(),
            )

    def test_setting_preconditioner_dtype_only(self) -> None:
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                preconditioner_dtype=torch.float16,
                precision_config=None,
            )

            self.assertIn(
                "preconditioner_dtype is deprecated. Please use precision_config instead.",
                [r.msg for r in cm.records],
            )
