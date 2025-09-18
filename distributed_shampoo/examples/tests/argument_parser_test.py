"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch
from distributed_shampoo import FSDPParamAssignmentStrategy

from distributed_shampoo.examples.argument_parser import (
    OptimizerType,
    Parser,
    PreconditionerComputationType,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
class TestArgumentParser(unittest.TestCase):
    sys_module: ModuleType = sys

    def test_invalid_optimizer_type_argument(self) -> None:
        """Test that invalid optimizer type argument raises proper error."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--optimizer-type", "INVALID_OPTIMIZER"],
        ):
            with self.assertRaises(SystemExit):
                Parser.get_args()

    def test_invalid_preconditioner_computation_type_argument(self) -> None:
        """Test that invalid preconditioner computation type argument raises proper error."""
        with patch.object(
            self.sys_module,
            "argv",
            [
                "test_program",
                "--preconditioner-computation-type",
                "INVALID_PRECONDITIONER",
            ],
        ):
            with self.assertRaises(SystemExit):
                Parser.get_args()

    def test_invalid_grafting_type_argument(self) -> None:
        """Test that invalid grafting type argument raises proper error."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--grafting-type", "INVALID_GRAFTING"],
        ):
            with self.assertRaises(SystemExit):
                Parser.get_args()

    def test_invalid_param_assignment_strategy_argument(self) -> None:
        """Test that invalid param assignment strategy argument raises proper error."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--param-assignment-strategy", "INVALID_STRATEGY"],
        ):
            with self.assertRaises(SystemExit):
                Parser.get_args()

    @parametrize("optimizer_type", OptimizerType)
    def test_get_args_with_optimizer_type(self, optimizer_type: OptimizerType) -> None:
        """Test get_args with optimizer type argument."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--optimizer-type", optimizer_type.name],
        ):
            args = Parser.get_args()
            self.assertEqual(args.optimizer_type, optimizer_type)

    @parametrize("preconditioner_type", PreconditionerComputationType)
    def test_get_args_with_preconditioner_computation_type(
        self, preconditioner_type: PreconditionerComputationType
    ) -> None:
        """Test get_args with preconditioner computation type argument."""
        with patch.object(
            self.sys_module,
            "argv",
            [
                "test_program",
                "--preconditioner-computation-type",
                preconditioner_type.name,
            ],
        ):
            args = Parser.get_args()
            self.assertEqual(args.preconditioner_computation_type, preconditioner_type)

    @parametrize("grafting_type", PreconditionerComputationType)
    def test_get_args_with_grafting_type(
        self, grafting_type: PreconditionerComputationType
    ) -> None:
        """Test get_args with grafting type argument."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--grafting-type", grafting_type.name],
        ):
            args = Parser.get_args()
            self.assertEqual(args.grafting_type, grafting_type)

    def test_get_args_with_batch_size(self) -> None:
        """Test get_args with custom batch size."""
        with patch.object(
            self.sys_module, "argv", ["test_program", "--batch-size", "256"]
        ):
            args = Parser.get_args()
            self.assertEqual(args.batch_size, 256)

    def test_get_args_default_values(self) -> None:
        """Test get_args with default values."""
        with patch.object(self.sys_module, "argv", ["test_program"]):
            args = Parser.get_args()

        # Test some key default values
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.window_size, 1)
        self.assertEqual(args.seed, 2022)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.beta1, 0.9)
        self.assertEqual(args.beta2, 0.999)
        self.assertEqual(args.beta3, -1.0)
        self.assertEqual(args.epsilon, 1e-12)
        self.assertEqual(args.weight_decay, 0.0)
        self.assertEqual(args.momentum, 0.0)
        self.assertEqual(args.dampening, 0.0)
        self.assertEqual(args.max_preconditioner_dim, 1024)
        self.assertEqual(args.precondition_frequency, 1)
        self.assertEqual(args.start_preconditioning_step, -1)
        self.assertEqual(args.inv_root_override, 0)
        self.assertEqual(
            args.preconditioner_computation_type,
            PreconditionerComputationType.EIGEN_ROOT_INV,
        )
        self.assertEqual(args.grafting_type, PreconditionerComputationType.SGD)
        self.assertEqual(args.grafting_epsilon, 1e-8)
        self.assertEqual(args.grafting_beta2, 0.999)
        self.assertEqual(args.communication_dtype, torch.float32)
        self.assertEqual(args.num_trainers_per_group, -1)
        self.assertEqual(args.local_batch_size, 128)
        self.assertEqual(args.num_trainers, 2)
        self.assertEqual(args.backend, "nccl")
        self.assertEqual(args.data_path, "./data")
        self.assertEqual(args.checkpoint_dir, None)
        self.assertEqual(args.dp_replicate_degree, 2)
        self.assertEqual(
            args.param_assignment_strategy, FSDPParamAssignmentStrategy.DEFAULT
        )
        self.assertEqual(args.metrics_dir, None)

    def test_get_args_boolean_flags(self) -> None:
        """Test get_args with boolean flags."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--use-nesterov", "--use-bias-correction"],
        ):
            args = Parser.get_args()
            self.assertTrue(args.use_nesterov)
            self.assertTrue(args.use_bias_correction)
            self.assertFalse(args.use_decoupled_weight_decay)  # Not set
            self.assertFalse(args.use_merge_dims)  # Not set
            self.assertFalse(args.communicate_params)  # Not set

    @parametrize("param_assignment_strategy", FSDPParamAssignmentStrategy)
    def test_get_args_with_param_assignment_strategy(
        self, param_assignment_strategy: FSDPParamAssignmentStrategy
    ) -> None:
        """Test get_args with FSDP parameter assignment strategy."""
        with patch.object(
            self.sys_module,
            "argv",
            [
                "test_program",
                "--param-assignment-strategy",
                param_assignment_strategy.name,
            ],
        ):
            args = Parser.get_args()
            self.assertEqual(args.param_assignment_strategy, param_assignment_strategy)

    def test_get_args_with_communication_dtype(self) -> None:
        """Test get_args with communication dtype."""
        with patch.object(
            self.sys_module,
            "argv",
            ["test_program", "--communication-dtype", "float16"],
        ):
            args = Parser.get_args()
            self.assertEqual(args.communication_dtype, torch.float16)

    def test_get_args_with_backend(self) -> None:
        """Test get_args with distributed backend."""
        with patch.object(
            self.sys_module, "argv", ["test_program", "--backend", "gloo"]
        ):
            args = Parser.get_args()
            self.assertEqual(args.backend, "gloo")

    def test_get_args_with_optimizer_parameters(self) -> None:
        """Test get_args with optimizer-specific parameters."""
        with patch.object(
            self.sys_module,
            "argv",
            [
                "test_program",
                "--lr",
                "0.001",
                "--beta1",
                "0.95",
                "--beta2",
                "0.99",
                "--epsilon",
                "1e-10",
                "--weight-decay",
                "0.01",
            ],
        ):
            args = Parser.get_args()
            self.assertEqual(args.lr, 0.001)
            self.assertEqual(args.beta1, 0.95)
            self.assertEqual(args.beta2, 0.99)
            self.assertEqual(args.epsilon, 1e-10)
            self.assertEqual(args.weight_decay, 0.01)

    def test_get_args_with_directories(self) -> None:
        """Test get_args with directory paths."""
        with patch.object(
            self.sys_module,
            "argv",
            [
                "test_program",
                "--metrics-dir",
                "/tmp/metrics",
                "--checkpoint-dir",
                "/tmp/checkpoints",
            ],
        ):
            args = Parser.get_args()
            self.assertEqual(args.metrics_dir, "/tmp/metrics")
            self.assertEqual(args.checkpoint_dir, "/tmp/checkpoints")

    def test_parser_argument_completeness(self) -> None:
        """Test that all expected arguments are defined in the parser."""
        with patch.object(self.sys_module, "argv", ["test_program"]):
            args = Parser.get_args()

            # Verify all expected attributes exist
            expected_attrs = {
                "optimizer_type",
                "batch_size",
                "epochs",
                "window_size",
                "seed",
                "lr",
                "beta1",
                "beta2",
                "beta3",
                "epsilon",
                "weight_decay",
                "momentum",
                "dampening",
                "max_preconditioner_dim",
                "precondition_frequency",
                "start_preconditioning_step",
                "inv_root_override",
                "use_nesterov",
                "use_bias_correction",
                "use_decoupled_weight_decay",
                "use_merge_dims",
                "preconditioner_computation_type",
                "grafting_type",
                "grafting_epsilon",
                "grafting_beta2",
                "communication_dtype",
                "num_trainers_per_group",
                "communicate_params",
                "local_batch_size",
                "num_trainers",
                "backend",
                "data_path",
                "checkpoint_dir",
                "dp_replicate_degree",
                "param_assignment_strategy",
                "metrics_dir",
            }

            actual_attrs = set(vars(args).keys())
            missing_attrs = expected_attrs - actual_attrs

            self.assertEqual(
                missing_attrs, set(), f"Missing attributes: {missing_attrs}"
            )
            # Allow extra attributes as they may be added in future versions
            # Just ensure we have all the expected ones
