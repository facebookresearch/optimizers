"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest

import torch
from distributed_shampoo.examples.convnet import ConvNet


class ConvNetTest(unittest.TestCase):
    def test_forward_pass_cifar10_size(self) -> None:
        """Test forward pass with CIFAR-10 input size (32x32)."""
        model = ConvNet(height=32, width=32)
        input_tensor = torch.randn(4, 3, 32, 32)
        output = model(input_tensor)
        self.assertEqual(output.shape, (4, 10))

    def test_forward_pass_mnist_size(self) -> None:
        """Test forward pass with MNIST-like input size (28x28)."""
        model = ConvNet(height=28, width=28)
        input_tensor = torch.randn(2, 3, 28, 28)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 10))

    def test_forward_pass_larger_image(self) -> None:
        """Test forward pass with larger image (64x64)."""
        model = ConvNet(height=64, width=64)
        input_tensor = torch.randn(1, 3, 64, 64)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10))

    def test_forward_pass_rectangular_image(self) -> None:
        """Test forward pass with rectangular image (48x32)."""
        model = ConvNet(height=48, width=32)
        input_tensor = torch.randn(2, 3, 48, 32)
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 10))

    def test_forward_pass_mismatched_input_size(self) -> None:
        """Test that forward pass fails with mismatched input size."""
        model = ConvNet(height=32, width=32)
        # Input with different size than model expects
        input_tensor = torch.randn(2, 3, 28, 28)

        with self.assertRaises(RuntimeError):
            model(input_tensor)

    def test_model_parameters(self) -> None:
        """Test that model parameters have correct shapes."""
        height, width = 32, 32
        model = ConvNet(height=height, width=width)

        parameters = list(model.parameters())

        # Should have 3 parameters: conv weight, linear weight, linear bias
        # (conv has bias=False)
        self.assertEqual(len(parameters), 3)

        # Check conv weight shape: (out_channels, in_channels, kernel_h, kernel_w)
        conv_weight = parameters[0]
        self.assertEqual(conv_weight.shape, (64, 3, 3, 3))

        # Check linear weight and bias shapes
        linear_weight = parameters[1]
        linear_bias = parameters[2]
        expected_linear_input_size = height * width * 64
        self.assertEqual(linear_weight.shape, (10, expected_linear_input_size))
        self.assertEqual(linear_bias.shape, (10,))

    def test_model_with_small_dimensions(self) -> None:
        """Test model with small edge case dimensions (16x16)."""
        model = ConvNet(height=16, width=16)
        input_tensor = torch.randn(1, 3, 16, 16)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10))

    def test_model_with_very_small_dimensions(self) -> None:
        """Test model with very small dimensions (5x5)."""
        model = ConvNet(height=5, width=5)
        input_tensor = torch.randn(1, 3, 5, 5)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 10))
