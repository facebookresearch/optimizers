"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import math
from typing import List

import torch
import torch.nn as nn


def infer_conv_output_shape(
    input_shape: List[int], kernel_size: int, stride: int, padding: int
) -> List[int]:
    output_shape = []
    for input_length in input_shape:
        output_length = (input_length - kernel_size + 2 * padding) / stride + 1
        if not output_length.is_integer():
            raise ValueError(
                f"Stride {stride} is not compatible with input shape {input_shape}, kernel size {kernel_size} and padding {padding}!"
            )
        output_shape.append(int(output_length))
    return output_shape


class ConvNet(nn.Module):
    """Simple two-layer convolutional network for image classification.
    Takes in image represented by an order-3 tensor. Used for testing optimizers.

    Args:
        height (int): Height of image.
        width (int): Width of image.
        channels (int): Channels of image.
        use_combined_linear (bool): Uses CombinedLinear module in place of nn.Linear.

    """

    def __init__(
        self, height: int, width: int, channels: int, use_combined_linear=False
    ):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(
            math.prod(
                infer_conv_output_shape(
                    [height, width], kernel_size=3, stride=1, padding=1
                )
            )
            * 64,
            10,
        )

    def forward(self, x):
        return self.linear(torch.flatten(self.activation(self.conv(x)), 1))
