"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import unittest
from math import sqrt
from typing import cast

import torch
import torch.nn as nn

from distributed_shampoo.utils.shampoo_model_utils import CombinedLinear


class CombinedLinearTest(unittest.TestCase):
    def _init_weights(self, m, seed) -> None:
        torch.random.manual_seed(seed)
        if type(m) == nn.Linear:
            bound = 1 / sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif type(m) == CombinedLinear:
            bound = 1 / sqrt(m.in_features)
            if m.bias:
                torch.nn.init.uniform_(m.combined_weight[:, :-1], -bound, bound)
                torch.nn.init.uniform_(m.combined_weight[:, -1], -bound, bound)
            else:
                torch.nn.init.uniform_(m.combined_weight, -bound, bound)

    def _test_linear_forward_backward(
        self,
        feature_vector: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: bool,
        seed: int,
    ) -> None:
        # generate linear layers and initialize
        original_linear = nn.Linear(in_features, out_features, bias=bias)
        combined_linear = CombinedLinear(in_features, out_features, bias=bias)

        self._init_weights(original_linear, seed)
        self._init_weights(combined_linear, seed)

        # confirm weights are initialized equally
        if bias:
            assert torch.equal(
                original_linear.weight, combined_linear.combined_weight[:, :-1]
            )
            assert torch.equal(
                original_linear.bias, combined_linear.combined_weight[:, -1]
            )
        else:
            assert torch.equal(original_linear.weight, combined_linear.combined_weight)

        # perform forward pass
        original_output = original_linear(feature_vector)
        combined_output = combined_linear(feature_vector)

        # compute backward of sum of output
        torch.sum(original_output).backward()
        torch.sum(combined_output).backward()

        # check values are equal
        with self.subTest("Test forward"):
            torch.testing.assert_close(original_output, combined_output)

        with self.subTest("Test backward"):
            if bias:
                torch.testing.assert_close(
                    cast(torch.Tensor, original_linear.weight.grad),
                    cast(torch.Tensor, combined_linear.combined_weight.grad)[
                        :, :-1
                    ],
                )
                torch.testing.assert_close(
                    cast(torch.Tensor, original_linear.bias.grad),
                    cast(torch.Tensor, combined_linear.combined_weight.grad)[:, -1],
                )
            else:
                torch.testing.assert_close(
                    cast(torch.Tensor, original_linear.weight.grad),
                    cast(torch.Tensor, combined_linear.combined_weight.grad),
                )

    def test_linear_forward_backward(self):
        dims = [2, 10]
        biases = [False, True]
        seeds = [920, 2022]

        for in_features, out_features, bias, seed in itertools.product(
            dims, dims, biases, seeds
        ):
            with self.subTest(
                f"Test with in_features = {in_features}, out_features = {out_features}, bias = {bias}, seed = {seed}"
            ):
                torch.random.manual_seed(seed)
                feature_vector = torch.rand(in_features)
                self._test_linear_forward_backward(
                    feature_vector, in_features, out_features, bias, seed
                )

    def test_initialization(self):
        in_features = 10
        out_features = 20
        biases = [False, True]
        seeds = [920, 2022]

        for bias, seed in itertools.product(biases, seeds):
            with self.subTest(
                f"Test with in_features = {in_features}, out_features = {out_features}, bias = {bias}, seed = {seed}"
            ):
                # generate linear layers and initialize
                torch.random.manual_seed(seed)
                original_linear = nn.Linear(in_features, out_features, bias=bias)
                torch.random.manual_seed(seed)
                combined_linear = CombinedLinear(in_features, out_features, bias=bias)

                # confirm weights are initialized equally
                if bias:
                    torch.testing.assert_close(
                        original_linear.weight, combined_linear.combined_weight[:, :-1]
                    )
                    torch.testing.assert_close(
                        original_linear.bias, combined_linear.combined_weight[:, -1]
                    )
                else:
                    torch.testing.assert_close(
                        original_linear.weight, combined_linear.combined_weight
                    )
