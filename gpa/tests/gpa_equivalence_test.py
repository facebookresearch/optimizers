"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Equivalence tests for GPA optimizers.

This module tests that GPA reduces to known optimizers under specific
hyperparameter settings:
    - When μ_x → 0, GPA approximates the base optimizer (AdamW)

From the paper (https://arxiv.org/abs/2512.17131):
"When μ_x = 0, x^{(t)} = y^{(t)} = z^{(t)} for any choice of μ_y,
and GPA reverts to the base optimizer."

Running tests:
    python -m unittest gpa.tests.gpa_equivalence_test -v
"""

import unittest

import torch
import torch.nn as nn
from gpa.gpa_adamw import GPAAdamW
from gpa.gpa_types import IterateAveragingType, Z_BUFFER
from gpa.tests.gpa_test_utils import run_optimizer_steps


class GPAEquivalenceTest(unittest.TestCase):
    """
    Tests that GPA reduces to known optimizers under specific settings.

    From the paper: "When μ_x = 0, x^{(t)} = y^{(t)} = z^{(t)} for any choice
    of μ_y, and GPA reverts to the base optimizer."
    """

    def setUp(self) -> None:
        self.lr = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.num_steps = 10
        self.seed = 42

    def _assert_gpa_matches_adamw(self, weight_decay: float) -> None:
        """Assert GPA with mu_x=0 matches AdamW for a given weight_decay."""
        device = torch.device("cpu")

        # Create GPA optimizer with μ_x=0
        torch.manual_seed(self.seed)
        model_gpa = nn.Linear(10, 5, bias=False).to(device)
        optimizer_gpa = GPAAdamW(
            model_gpa.parameters(),
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=weight_decay,
            train_interp_coeff=0.9,
            eval_interp_coeff=0.0,
            iterate_averaging_type=IterateAveragingType.GPA,
        )

        # Create AdamW optimizer with same initial weights
        torch.manual_seed(self.seed)
        model_adamw = nn.Linear(10, 5, bias=False).to(device)
        optimizer_adamw = torch.optim.AdamW(
            model_adamw.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=weight_decay,
        )

        # Generate identical gradients for both optimizers
        torch.manual_seed(100)
        gradients = [torch.randn(5, 10, device=device) for _ in range(self.num_steps)]

        # Run both optimizers with identical gradients
        optimizer_gpa.train()
        run_optimizer_steps(optimizer_gpa, model_gpa, gradients)
        run_optimizer_steps(optimizer_adamw, model_adamw, gradients)

        # Compare z-sequence of GPA to AdamW weights
        optimizer_gpa.eval()
        for param_gpa, param_adamw in zip(
            model_gpa.parameters(), model_adamw.parameters()
        ):
            z_buffer = optimizer_gpa.state[param_gpa][Z_BUFFER]
            torch.testing.assert_close(
                z_buffer,
                param_adamw,
                atol=1e-6,
                rtol=1e-6,
                msg=f"z-buffer should match AdamW on {device}"
                f" (weight_decay={weight_decay})",
            )

    def test_gpa_with_mu_x_zero_equals_adamw(self) -> None:
        """
        Test that GPA with μ_x=0 matches AdamW to floating point precision.

        When eval_interp_coeff=0 in GPA mode (not Schedule-Free), the x-update
        becomes:
            x^{(t+1)} = 0 * x^{(t)} + 1 * z^{(t+1)} = z^{(t+1)}

        This means x = z at every step, so y = mu_y * x + (1 - mu_y) * z = z
        regardless of mu_y. The optimizer reduces to AdamW mathematically.

        Tested with both weight_decay=0 and weight_decay>0. Not bitwise equal
        due to different floating point operation ordering between GPA and
        PyTorch's AdamW implementation (e.g., GPA applies weight decay
        by adding it to the search direction while AdamW multiplies it to the parameters directly).
        """
        for weight_decay in [0.0, 0.01]:
            with self.subTest(weight_decay=weight_decay):
                self._assert_gpa_matches_adamw(weight_decay)


if __name__ == "__main__":
    unittest.main()
