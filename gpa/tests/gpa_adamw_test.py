"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Consolidated unit tests for GPAAdamW optimizer.

Test classes:
1. GPAAdamWInitializationTest - Hyperparameter validation and state initialization
2. GPAAdamWStepAndModeTest - Train/eval mode switching and step behavior
3. GPAAdamWAvgCoeffTest - Schedule-Free vs GPA mode (distinct from train/eval)
4. GPAAdamWStateDictTest - State dict save/load and checkpoint compatibility

Running tests:
    python -m unittest gpa_adamw_test -v
"""

import unittest
from typing import Any, Dict

import torch
from gpa.gpa_adamw import GPAAdamW
from gpa.gpa_types import (
    EXP_AVG,
    EXP_AVG_SQ,
    ITERATE_AVERAGING_TYPE,
    IterateAveragingType,
    LR_MAX,
    STEP,
    TRAIN_MODE,
    WEIGHT_SUM,
    Z_BUFFER,
)

from .gpa_test_utils import create_simple_model, set_deterministic_gradients


class GPAAdamWInitializationTest(unittest.TestCase):
    """
    Tests for GPAAdamW initialization and parameter validation.

    Verifies:
    - Default and custom hyperparameters are stored correctly
    - Invalid hyperparameters raise appropriate ValueError exceptions
    - State is initialized correctly on first step
    - Multiple parameter groups work correctly
    """

    def setUp(self) -> None:
        """Set up a simple model for each test."""
        self.model = create_simple_model(input_dim=10, output_dim=5)

    def test_default_initialization(self) -> None:
        """Test that optimizer initializes correctly with default parameters."""
        optimizer = GPAAdamW(self.model.parameters())

        self.assertEqual(len(optimizer.param_groups), 1)
        group = optimizer.param_groups[0]

        self.assertEqual(group["lr"], 1.0)
        self.assertEqual(group["eps"], 1e-8)
        self.assertEqual(group["train_interp_coeff"], 0.7)
        self.assertEqual(group["beta1"], 0.9)
        self.assertEqual(group["beta2"], 0.999)
        self.assertEqual(group["weight_decay"], 0)
        self.assertEqual(group["weight_pow_coeff"], 0.0)
        self.assertEqual(group["weight_lr_power"], 2)
        self.assertEqual(group["eval_interp_coeff"], 0.9967)
        self.assertEqual(group[ITERATE_AVERAGING_TYPE], IterateAveragingType.GPA)

    def test_custom_hyperparameters(self) -> None:
        """Test that custom hyperparameters are correctly stored in param_groups."""
        optimizer = GPAAdamW(
            self.model.parameters(),
            lr=0.01,
            eps=1e-6,
            train_interp_coeff=0.8,
            beta1=0.85,
            beta2=0.995,
            weight_decay=0.01,
            weight_pow_coeff=1.0,
            weight_lr_power=1.5,
            eval_interp_coeff=0.5,
        )

        group = optimizer.param_groups[0]

        self.assertEqual(group["lr"], 0.01)
        self.assertEqual(group["eps"], 1e-6)
        self.assertEqual(group["train_interp_coeff"], 0.8)
        self.assertEqual(group["beta1"], 0.85)
        self.assertEqual(group["beta2"], 0.995)
        self.assertEqual(group["weight_decay"], 0.01)
        self.assertEqual(group["weight_pow_coeff"], 1.0)
        self.assertEqual(group["weight_lr_power"], 1.5)
        self.assertEqual(group["eval_interp_coeff"], 0.5)

    def test_invalid_hyperparameters(self) -> None:
        """Test that invalid hyperparameters raise ValueError with descriptive messages."""
        with self.assertRaisesRegex(ValueError, "Invalid learning rate: -0.01"):
            GPAAdamW(self.model.parameters(), lr=-0.01)

        with self.assertRaisesRegex(ValueError, "Invalid epsilon value: -1e-08"):
            GPAAdamW(self.model.parameters(), eps=-1e-8)

        # beta1 must be in [0, 1)
        with self.assertRaisesRegex(ValueError, "Invalid beta1 value: -0.1"):
            GPAAdamW(self.model.parameters(), beta1=-0.1)

        with self.assertRaisesRegex(ValueError, "Invalid beta1 value: 1.0"):
            GPAAdamW(self.model.parameters(), beta1=1.0)

        # beta2 must be in [0, 1)
        with self.assertRaisesRegex(ValueError, "Invalid beta2 value: -0.1"):
            GPAAdamW(self.model.parameters(), beta2=-0.1)

        with self.assertRaisesRegex(ValueError, "Invalid beta2 value: 1.0"):
            GPAAdamW(self.model.parameters(), beta2=1.0)

        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -0.01"):
            GPAAdamW(self.model.parameters(), weight_decay=-0.01)

        # iterate_averaging_type must be an IterateAveragingType enum
        invalid_type: Any = "gpa"
        with self.assertRaisesRegex(
            ValueError, r"iterate_averaging_type must be an IterateAveragingType enum"
        ):
            GPAAdamW(self.model.parameters(), iterate_averaging_type=invalid_type)

        # train_interp_coeff must be in (0, 1] - exclusive 0, inclusive 1
        with self.assertRaisesRegex(
            ValueError, r"train_interp_coeff must be in range \(0, 1\]: 0.0"
        ):
            GPAAdamW(self.model.parameters(), train_interp_coeff=0.0)

        with self.assertRaisesRegex(
            ValueError, r"train_interp_coeff must be in range \(0, 1\]: 1.5"
        ):
            GPAAdamW(self.model.parameters(), train_interp_coeff=1.5)

        # eval_interp_coeff must be in [0, 1] - inclusive both ends
        with self.assertRaisesRegex(
            ValueError, r"eval_interp_coeff must be in range \[0, 1\]: -0.1"
        ):
            GPAAdamW(self.model.parameters(), eval_interp_coeff=-0.1)

        with self.assertRaisesRegex(
            ValueError, r"eval_interp_coeff must be in range \[0, 1\]: 1.5"
        ):
            GPAAdamW(self.model.parameters(), eval_interp_coeff=1.5)

        with self.assertRaisesRegex(ValueError, "Invalid weight_pow_coeff value: -1.0"):
            GPAAdamW(self.model.parameters(), weight_pow_coeff=-1.0)

        with self.assertRaisesRegex(ValueError, "Invalid weight_lr_power value: -1.0"):
            GPAAdamW(self.model.parameters(), weight_lr_power=-1.0)

    def test_state_initialization(self) -> None:
        """Test that shared state and per-parameter state are correctly initialized."""
        optimizer = GPAAdamW(
            self.model.parameters(),
            lr=0.01,
            train_interp_coeff=0.7,
            eval_interp_coeff=0.9967,
        )

        optimizer.train()
        set_deterministic_gradients(self.model)
        optimizer.step()

        first_param = optimizer.param_groups[0]["params"][0]
        state = optimizer.state[first_param]

        # Verify shared state exists
        self.assertIn(TRAIN_MODE, state)
        self.assertIn(LR_MAX, state)
        self.assertIn(WEIGHT_SUM, state)
        self.assertIn(STEP, state)
        self.assertEqual(state[STEP].item(), 1)

        # Verify per-parameter state for all parameters
        for param in self.model.parameters():
            param_state = optimizer.state[param]
            self.assertIn(Z_BUFFER, param_state)
            self.assertIn(EXP_AVG, param_state)
            self.assertIn(EXP_AVG_SQ, param_state)
            self.assertEqual(param_state[EXP_AVG].dtype, torch.float32)
            self.assertEqual(param_state[EXP_AVG_SQ].dtype, torch.float32)

    def test_multiple_param_groups(self) -> None:
        """Test that optimizer correctly handles multiple parameter groups."""
        model1 = create_simple_model(10, 5)
        model2 = create_simple_model(5, 3)

        optimizer = GPAAdamW(
            [
                {"params": model1.parameters(), "lr": 0.01},
                {"params": model2.parameters(), "lr": 0.001},
            ]
        )

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.001)


class GPAAdamWStepAndModeTest(unittest.TestCase):
    """
    Tests for GPAAdamW train/eval mode switching and step() behavior.

    Verifies:
    - train() and eval() correctly toggle TRAIN_MODE
    - step() requires train mode
    - Parameters are correctly transformed between y-sequence and x-sequence
    - Mode roundtrip preserves parameter values
    - z_buffer is preserved during mode switching
    """

    def setUp(self) -> None:
        """Set up model and optimizer for each test (uninitialized state)."""
        self.model = create_simple_model(input_dim=10, output_dim=5)
        self.optimizer = GPAAdamW(
            self.model.parameters(),
            lr=0.01,
            train_interp_coeff=0.7,
            eval_interp_coeff=0.9967,
        )

    def _initialize_optimizer(self) -> None:
        """Initialize optimizer by running one step. Call explicitly when needed."""
        self.optimizer.train()
        set_deterministic_gradients(self.model)
        self.optimizer.step()

    def test_train_eval_mode_switching(self) -> None:
        """Test that train() and eval() correctly toggle TRAIN_MODE."""
        self._initialize_optimizer()
        first_param = self.optimizer.param_groups[0]["params"][0]

        # After initialization, optimizer should be in train mode
        self.assertTrue(self.optimizer.state[first_param][TRAIN_MODE].item())

        # Switch to eval mode
        self.optimizer.eval()
        self.assertFalse(self.optimizer.state[first_param][TRAIN_MODE].item())

        # Switch back to train mode
        self.optimizer.train()
        self.assertTrue(self.optimizer.state[first_param][TRAIN_MODE].item())

    def test_step_requires_train_mode(self) -> None:
        """Test that step() raises RuntimeError when not in train mode."""
        set_deterministic_gradients(self.model)

        with self.assertRaisesRegex(
            RuntimeError,
            "Optimizer was not in train mode when step is called. "
            "Please insert .train\\(\\) and .eval\\(\\) calls on the "
            "optimizer. See documentation for details.",
        ):
            self.optimizer.step()

    def test_step_updates_state(self) -> None:
        """Test that step() correctly updates exp_avg, exp_avg_sq, z_buffer, and lr_max.

        This test uses explicit gradient values and hard-coded expected results
        to verify the optimizer's numerical correctness without duplicating
        the algorithm's computation logic.

        Given:
            - gradient = [[1.0, 2.0, ...], [-1.0, -2.0, ...], ...]  (5x10 tensor)
            - beta1 = 0.9, beta2 = 0.999 (default values)

        After first step:
            - exp_avg = (1 - beta1) * grad = 0.1 * grad
            - exp_avg_sq = (1 - beta2) * grad^2 = 0.001 * grad^2
            - z_buffer is updated: z_new = z_old - lr * grad_normalized
        """
        self.optimizer.train()

        # Use simple explicit gradient values for easy manual verification
        # Shape: (output_dim=5, input_dim=10) matching model from setUp
        # Row 0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # Row 1: [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10] (negative to test sign)
        # Row 2: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Row 3: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # Row 4: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        explicit_grad = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            ],
            dtype=torch.float32,
        )

        param = list(self.model.parameters())[0]
        z_buffer_before = param.clone()  # z_buffer is initialized to param value
        param.grad = explicit_grad.clone()

        self.optimizer.step()

        state = self.optimizer.state[param]

        # Hard-coded expected values (manually computed):
        # exp_avg = (1 - 0.9) * grad = 0.1 * grad
        # Row 0: 0.1 * [1,2,3,4,5,6,7,8,9,10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # Row 1: 0.1 * [-1,-2,...,-10] = [-0.1, -0.2, ..., -1.0] (signs preserved)
        # Row 2: 0.1 * [0.1,0.2,...,1.0] = [0.01, 0.02, ..., 0.1]
        # Row 3: 0.1 * [1,1,...,1] = [0.1, 0.1, ..., 0.1]
        # Row 4: 0.1 * [2,2,...,2] = [0.2, 0.2, ..., 0.2]
        expected_exp_avg = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0],
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            state[EXP_AVG], expected_exp_avg, rtol=1e-5, atol=1e-8
        )

        # Hard-coded expected values (manually computed):
        # exp_avg_sq = (1 - 0.999) * grad^2 = 0.001 * grad^2
        # Row 0: 0.001 * [1,4,9,16,25,36,49,64,81,100] = [0.001, 0.004, 0.009, ...]
        # Row 1: 0.001 * [1,4,9,...,100] = same as row 0 (squaring removes sign)
        # Row 2: 0.001 * [0.01,0.04,0.09,...,1.0] = [0.00001, 0.00004, ...]
        # Row 3: 0.001 * [1,1,...,1] = [0.001, 0.001, ..., 0.001]
        # Row 4: 0.001 * [4,4,...,4] = [0.004, 0.004, ..., 0.004]
        expected_exp_avg_sq = torch.tensor(
            [
                [0.001, 0.004, 0.009, 0.016, 0.025, 0.036, 0.049, 0.064, 0.081, 0.100],
                [0.001, 0.004, 0.009, 0.016, 0.025, 0.036, 0.049, 0.064, 0.081, 0.100],
                [
                    0.00001,
                    0.00004,
                    0.00009,
                    0.00016,
                    0.00025,
                    0.00036,
                    0.00049,
                    0.00064,
                    0.00081,
                    0.001,
                ],
                [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
                [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            state[EXP_AVG_SQ], expected_exp_avg_sq, rtol=1e-5, atol=1e-8
        )

        # lr_max should be updated
        first_param = self.optimizer.param_groups[0]["params"][0]
        torch.testing.assert_close(
            self.optimizer.state[first_param][LR_MAX],
            torch.tensor(0.01),
            rtol=1e-5,
            atol=1e-8,
        )

        # Verify z_buffer was updated (z_new = z_old - lr * grad_normalized)
        # z_buffer should be different from initial value after step
        self.assertFalse(
            torch.allclose(state[Z_BUFFER], z_buffer_before),
            "z_buffer should be updated after step()",
        )

    def test_mode_roundtrip_preserves_params_and_z_buffer(self) -> None:
        """Test that train() -> eval() -> train() preserves parameters and z_buffer."""
        # Initialize optimizer first (state including z_buffer only exists after step)
        self._initialize_optimizer()

        # Store original values as lists (consistent with list(self.model.parameters()))
        original_y = [param.clone() for param in self.model.parameters()]
        original_z = [
            self.optimizer.state[param][Z_BUFFER].clone()
            for param in self.model.parameters()
        ]

        # Roundtrip: train -> eval -> train
        self.optimizer.eval()
        self.optimizer.train()

        # Parameters should be back to original
        torch.testing.assert_close(
            list(self.model.parameters()),
            original_y,
            rtol=1e-5,
            atol=1e-8,
        )

        # z_buffer should be unchanged
        torch.testing.assert_close(
            [
                self.optimizer.state[param][Z_BUFFER]
                for param in self.model.parameters()
            ],
            original_z,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_double_mode_call_idempotent(self) -> None:
        """Test that calling train()/eval() twice doesn't double-transform parameters."""
        # Store params after first train (as list, consistent with list(self.model.parameters()))
        params_after_train = [param.clone() for param in self.model.parameters()]

        # Call train again - should be idempotent (exact equality)
        self.optimizer.train()

        torch.testing.assert_close(
            list(self.model.parameters()),
            params_after_train,
            atol=0.0,
            rtol=0.0,
        )

        # Switch to eval and store
        self.optimizer.eval()
        params_after_eval = [param.clone() for param in self.model.parameters()]

        # Call eval again - should be idempotent (exact equality)
        self.optimizer.eval()

        torch.testing.assert_close(
            list(self.model.parameters()),
            params_after_eval,
            atol=0.0,
            rtol=0.0,
        )


class GPAAdamWAvgCoeffTest(unittest.TestCase):
    """
    Tests for GPAAdamW averaging coefficient computation.

    These tests verify the compute_avg_coeff method behavior in:
    - Schedule-Free mode (eval_interp_coeff=0): Uses polynomial weighting
    - GPA mode (eval_interp_coeff>0): Uses (1 - eval_interp_coeff)

    Note: The "mode" here refers to Schedule-Free vs GPA mode (based on eval_interp_coeff),
    NOT train vs eval mode (which is tested in GPAAdamWStepAndModeTest).
    """

    def test_schedule_free_mode(self) -> None:
        """
        Test polynomial weighting in Schedule-Free mode.

        In Schedule-Free mode, avg_coeff is computed as:
        weight = k^weight_pow_coeff * lr_max^weight_lr_power
        weight_sum += weight
        avg_coeff = weight / weight_sum
        """
        weight_sum = torch.tensor(0.0)

        # First step (k=1)
        avg_coeff = GPAAdamW.compute_avg_coeff(
            iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
            eval_interp_coeff=0.0,
            k=1,
            weight_pow_coeff=0.0,  # k^0 = 1
            lr_max=0.01,
            weight_lr_power=2.0,  # lr_max^2
            weight_sum_ref=weight_sum,
        )

        # weight = 1^0 * 0.01^2 = 0.0001
        # weight_sum = 0 + 0.0001 = 0.0001
        # avg_coeff = 0.0001 / 0.0001 = 1.0
        self.assertAlmostEqual(avg_coeff, 1.0, places=6)

        # Second step (k=2)
        avg_coeff = GPAAdamW.compute_avg_coeff(
            iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
            eval_interp_coeff=0.0,
            k=2,
            weight_pow_coeff=0.0,
            lr_max=0.01,
            weight_lr_power=2.0,
            weight_sum_ref=weight_sum,
        )

        # weight = 1^0 * 0.01^2 = 0.0001
        # weight_sum = 0.0001 + 0.0001 = 0.0002
        # avg_coeff = 0.0001 / 0.0002 = 0.5
        self.assertAlmostEqual(avg_coeff, 0.5, places=6)

    def test_gpa_mode(self) -> None:
        """
        Test that avg_coeff = 1 - eval_interp_coeff in GPA mode.

        In GPA mode, the averaging coefficient is simply (1 - mu_x),
        independent of step count or lr_max.
        """
        weight_sum = torch.tensor(0.0)

        # Test representative values: small, medium, large
        for eval_interp_coeff in [0.0, 0.1, 0.5, 0.9]:
            with self.subTest(eval_interp_coeff=eval_interp_coeff):
                avg_coeff = GPAAdamW.compute_avg_coeff(
                    iterate_averaging_type=IterateAveragingType.GPA,
                    eval_interp_coeff=eval_interp_coeff,
                    k=1,
                    weight_pow_coeff=0.0,
                    lr_max=0.01,
                    weight_lr_power=2.0,
                    weight_sum_ref=weight_sum,
                )

                expected = 1 - eval_interp_coeff
                self.assertAlmostEqual(
                    avg_coeff,
                    expected,
                    places=6,
                )

    def test_weight_sum_accumulation(self) -> None:
        """Test that weight_sum accumulates correctly in Schedule-Free mode."""
        weight_sum = torch.tensor(0.0)

        for k in range(1, 6):
            GPAAdamW.compute_avg_coeff(
                iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
                eval_interp_coeff=0.0,
                k=k,
                weight_pow_coeff=0.0,
                lr_max=1.0,  # Simplify: lr_max^2 = 1
                weight_lr_power=2.0,
                weight_sum_ref=weight_sum,
            )

        # Each step adds weight = 1^0 * 1^2 = 1
        # After 5 steps: weight_sum = 5
        self.assertAlmostEqual(weight_sum.item(), 5.0, places=6)

    def test_weight_pow_coeff_effect(self) -> None:
        """
        Test that weight_pow_coeff affects weight calculation.

        With weight_pow_coeff > 0, later steps have higher weights.
        """
        weight_sum = torch.tensor(0.0)

        # With weight_pow_coeff=1, weight = k^1 * lr_max^power
        # k=1: weight=1, sum=1, avg=1
        avg_coeff_k1 = GPAAdamW.compute_avg_coeff(
            iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
            eval_interp_coeff=0.0,
            k=1,
            weight_pow_coeff=1.0,
            lr_max=1.0,
            weight_lr_power=0.0,  # lr_max^0 = 1
            weight_sum_ref=weight_sum,
        )

        # k=2: weight=2, sum=3, avg=2/3
        avg_coeff_k2 = GPAAdamW.compute_avg_coeff(
            iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
            eval_interp_coeff=0.0,
            k=2,
            weight_pow_coeff=1.0,
            lr_max=1.0,
            weight_lr_power=0.0,
            weight_sum_ref=weight_sum,
        )

        self.assertAlmostEqual(avg_coeff_k1, 1.0, places=6)
        self.assertAlmostEqual(avg_coeff_k2, 2.0 / 3.0, places=6)

    def test_weight_lr_power_effect(self) -> None:
        """Test that weight_lr_power affects weight calculation."""
        weight_sum = torch.tensor(0.0)

        # lr_max=2, weight_lr_power=2 -> lr_max^2 = 4
        GPAAdamW.compute_avg_coeff(
            iterate_averaging_type=IterateAveragingType.SCHEDULE_FREE,
            eval_interp_coeff=0.0,
            k=1,
            weight_pow_coeff=0.0,
            lr_max=2.0,
            weight_lr_power=2.0,
            weight_sum_ref=weight_sum,
        )

        # weight = 1 * 4 = 4
        self.assertAlmostEqual(weight_sum.item(), 4.0, places=6)


class GPAAdamWStateDictTest(unittest.TestCase):
    """
    Tests for GPAAdamW state dict save and load operations.

    Uses the reference state dict pattern (similar to distributed_shampoo_test.py)
    to verify state dict structure and roundtrip behavior.
    """

    def setUp(self) -> None:
        """Set up model and optimizer for each test."""
        torch.manual_seed(42)
        self.model = create_simple_model(input_dim=5, output_dim=3)
        self.optimizer = GPAAdamW(
            self.model.parameters(),
            lr=0.01,
            train_interp_coeff=0.7,
            eval_interp_coeff=0.9967,
        )

    def _initialize_and_run_steps(self, steps: int = 3) -> None:
        """Helper to initialize optimizer and run several steps."""
        self.optimizer.train()
        for i in range(steps):
            set_deterministic_gradients(self.model, seed=i)
            self.optimizer.step()

    @property
    def _ref_state_dict(self) -> Dict[str, Any]:
        """
        Reference state dict for comparison.

        This represents the expected structure after initialization and 3 steps
        with torch.manual_seed(42) for model initialization and seeds 0, 1, 2
        for deterministic gradients.

        The reference pattern ensures state dict format doesn't change unexpectedly.
        """
        return {
            "state": {
                0: {
                    Z_BUFFER: torch.tensor(
                        [
                            [
                                0.3145188391,
                                0.3842400312,
                                -0.0823936909,
                                0.3916080892,
                                -0.0779593736,
                            ],
                            [
                                0.1157064661,
                                -0.2204549909,
                                0.2341007739,
                                0.4234150052,
                                -0.2985607982,
                            ],
                            [
                                0.4085050523,
                                0.0626114607,
                                0.3574149311,
                                0.0544980019,
                                0.2324079126,
                            ],
                        ]
                    ),
                    EXP_AVG: torch.tensor(
                        [
                            [
                                0.2235720605,
                                -0.0221009720,
                                -0.2028810084,
                                -0.0185422208,
                                -0.0240715072,
                            ],
                            [
                                -0.1915607303,
                                -0.0470674001,
                                0.1563264281,
                                -0.1899352223,
                                -0.1876134723,
                            ],
                            [
                                0.0032281273,
                                0.0842663422,
                                -0.2196345776,
                                -0.1218880117,
                                -0.0590292960,
                            ],
                        ]
                    ),
                    EXP_AVG_SQ: torch.tensor(
                        [
                            [
                                0.0029607685,
                                0.0002070865,
                                0.0048435149,
                                0.0021602320,
                                0.0024687566,
                            ],
                            [
                                0.0023807085,
                                0.0028073201,
                                0.0011390453,
                                0.0017248112,
                                0.0015662514,
                            ],
                            [
                                0.0028894001,
                                0.0005954248,
                                0.0025301611,
                                0.0036812474,
                                0.0013339740,
                            ],
                        ]
                    ),
                    STEP: torch.tensor(3),
                    LR_MAX: torch.tensor(0.01),
                    WEIGHT_SUM: torch.tensor(0.0),
                    TRAIN_MODE: torch.tensor(True),
                },
            },
            "param_groups": [
                {
                    "lr": 0.01,
                    "eps": 1e-8,
                    "train_interp_coeff": 0.7,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "weight_decay": 0,
                    "weight_pow_coeff": 0.0,
                    "weight_lr_power": 2,
                    "eval_interp_coeff": 0.9967,
                    "iterate_averaging_type": IterateAveragingType.GPA,
                    "params": [0],
                },
            ],
        }

    def _assert_state_dicts_equal(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """
        Compare two optimizer state dicts for equality.

        This helper method performs a comprehensive comparison of two state dicts:
        1. Verifies both have the expected top-level structure (state, param_groups)
        2. Compares all state tensors (z_buffer, exp_avg, exp_avg_sq, etc.) using
           torch.testing.assert_close with configurable tolerances
        3. Compares param_groups settings (excluding 'params' which contains
           object references that differ between optimizer instances)

        Args:
            actual: The state dict to verify (e.g., from optimizer.state_dict())
            expected: The reference state dict to compare against
            rtol: Relative tolerance for tensor comparison (default: 1e-6)
            atol: Absolute tolerance for tensor comparison (default: 1e-6)

        Raises:
            AssertionError: If state dicts differ in structure or values
        """
        # Verify top-level structure
        self.assertEqual(
            set(actual.keys()),
            {"state", "param_groups"},
            "State dict must have 'state' and 'param_groups' keys",
        )
        self.assertEqual(
            set(expected.keys()),
            {"state", "param_groups"},
            "Reference state dict must have 'state' and 'param_groups' keys",
        )

        # Compare state tensors - iterate manually to handle dtype differences
        # for boolean tensors (train_mode may have different dtype after load)
        self.assertEqual(
            set(actual["state"].keys()),
            set(expected["state"].keys()),
            "State keys must match",
        )

        for param_idx in actual["state"]:
            actual_state = actual["state"][param_idx]
            expected_state = expected["state"][param_idx]

            self.assertEqual(
                set(actual_state.keys()),
                set(expected_state.keys()),
                f"State[{param_idx}] keys must match",
            )

            for key in actual_state:
                actual_val = actual_state[key]
                expected_val = expected_state[key]

                if isinstance(actual_val, torch.Tensor) and isinstance(
                    expected_val, torch.Tensor
                ):
                    # Handle train_mode specially - compare values, not dtype
                    if key == TRAIN_MODE:
                        self.assertEqual(
                            bool(actual_val.item()),
                            bool(expected_val.item()),
                            f"State[{param_idx}][{key}] values differ",
                        )
                    else:
                        torch.testing.assert_close(
                            actual_val,
                            expected_val,
                            rtol=rtol,
                            atol=atol,
                            msg=f"State[{param_idx}][{key}] differs",
                        )
                else:
                    self.assertEqual(
                        actual_val,
                        expected_val,
                        f"State[{param_idx}][{key}] differs",
                    )

        # Compare param_groups (excluding 'params' which has object references)
        self.assertEqual(
            len(actual["param_groups"]),
            len(expected["param_groups"]),
            "Number of param_groups must match",
        )
        for i, (actual_pg, expected_pg) in enumerate(
            zip(actual["param_groups"], expected["param_groups"])
        ):
            actual_pg_copy = actual_pg.copy()
            expected_pg_copy = expected_pg.copy()
            actual_pg_copy.pop("params")
            expected_pg_copy.pop("params")
            self.assertEqual(
                actual_pg_copy,
                expected_pg_copy,
                f"param_groups[{i}] settings differ",
            )

    def test_state_dict_structure(self) -> None:
        """
        Test that the state dict has the expected structure and tensor values.

        This test verifies state dict matches the reference state dict with
        hard-coded tensor values after running 3 deterministic steps.
        """
        self._initialize_and_run_steps()

        state_dict = self.optimizer.state_dict()
        ref_state_dict = self._ref_state_dict

        self._assert_state_dicts_equal(state_dict, ref_state_dict)

    def test_state_dict_roundtrip(self) -> None:
        """
        Test state dict save and load preserves all optimizer state.

        Verifies that load_state_dict correctly restores optimizer state
        by saving state from one optimizer and loading into a fresh optimizer,
        then comparing the resulting state dicts.
        """
        self._initialize_and_run_steps(steps=3)
        self.optimizer.eval()

        # Save state from original optimizer
        original_state_dict = self.optimizer.state_dict()

        # Create new optimizer and load state
        torch.manual_seed(42)
        new_model = create_simple_model(input_dim=5, output_dim=3)
        new_optimizer = GPAAdamW(
            new_model.parameters(),
            lr=0.01,
            train_interp_coeff=0.7,
            eval_interp_coeff=0.9967,
        )
        new_optimizer.load_state_dict(original_state_dict)

        # Get state dict from new optimizer after loading
        loaded_state_dict = new_optimizer.state_dict()

        # Compare state dicts - they should be identical after roundtrip
        self._assert_state_dicts_equal(loaded_state_dict, original_state_dict)


if __name__ == "__main__":
    unittest.main()
