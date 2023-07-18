"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import unittest

import numpy as np

import torch
import torch.distributed as dist

try:
    # DTensor requires PyTorch 2.1 nightly build.
    import torch.distributed._tensor as dtensor

    ENABLE_DTENSOR = True
except ImportError:
    ENABLE_DTENSOR = False

from torch.testing._internal.common_distributed import spawn_threads_and_init_comms

from distributed_shampoo.shampoo_fsdp_utils import convex_split

logger: logging.Logger = logging.getLogger(__name__)

if not ENABLE_DTENSOR:
    logger.warning(
        "DTensor is not available and was not imported. Continuing with Tensor..."
    )


class ConvexSplitTest(unittest.TestCase):
    def _test_convex_split(self, tensor, split_tensors, start_idx, end_idx) -> None:
        split_tensors.sort(key=lambda x: x[(0,) * x.ndim])
        results = convex_split(
            tensor.flatten()[start_idx : end_idx + 1],
            tensor.size(),
            start_idx,
            end_idx,
        )
        assert len(results) != 0
        results.sort(key=lambda x: x[(0,) * x.ndim])
        for idx, t in enumerate(results):
            with self.subTest(f"Test with idx = {idx}"):
                torch.testing.assert_close(split_tensors[idx].squeeze(), t.squeeze())

    def test_convex_split_for_one_dim(self) -> None:
        tensor = torch.arange(10)
        self._test_convex_split(tensor, [tensor], 0, 9)

        split_tensors = [torch.arange(6) + 2]
        self._test_convex_split(tensor, split_tensors, 2, 7)

    def test_convex_split_for_two_dim(self) -> None:
        tensor = torch.arange(48).reshape(6, 8)
        self._test_convex_split(tensor, [tensor], 0, 47)

        split_tensors = [
            torch.tensor([8, 9, 10]),
            torch.arange(8).reshape(1, 8),
        ]
        self._test_convex_split(tensor, split_tensors, 0, 10)

        split_tensors = [
            torch.tensor([11, 12, 13, 14, 15]),
            torch.arange(32).reshape(4, 8) + 16,
        ]
        self._test_convex_split(tensor, split_tensors, 11, 47)

        split_tensors = [
            torch.tensor([11, 12, 13, 14, 15]),
            torch.tensor([24, 25, 26, 27, 28, 29]),
            torch.arange(8).reshape(1, 8) + 16,
        ]
        self._test_convex_split(tensor, split_tensors, 11, 29)

        split_tensors = [
            torch.tensor([11, 12, 13, 14, 15]),
            torch.tensor([16]),
        ]
        self._test_convex_split(tensor, split_tensors, 11, 16)

    def test_convex_split_for_three_dim(self) -> None:
        tensor = torch.arange(27).reshape(3, 3, 3)
        self._test_convex_split(tensor, [tensor], 0, 26)

        split_tensors = [
            torch.tensor([9, 10]),
            torch.arange(9).reshape(3, 3),
        ]
        self._test_convex_split(tensor, split_tensors, 0, 10)

        split_tensors = [
            torch.tensor([[21, 22, 23], [24, 25, 26]]),
        ]
        self._test_convex_split(tensor, split_tensors, 21, 26)

        split_tensors = [
            torch.arange(9).reshape(3, 3) + 9,
        ]
        self._test_convex_split(tensor, split_tensors, 9, 17)

        split_tensors = [
            torch.tensor([8]),
            torch.tensor([18]),
            torch.arange(9).reshape(3, 3) + 9,
        ]
        self._test_convex_split(tensor, split_tensors, 8, 18)

        split_tensors = [
            torch.tensor([5]),
            torch.tensor([6]),
        ]
        self._test_convex_split(tensor, split_tensors, 5, 6)

    def test_convex_split_for_four_dim(self) -> None:
        tensor = torch.arange(81).reshape(3, 3, 3, 3)
        self._test_convex_split(tensor, [tensor], 0, 80)

        split_tensors = [
            torch.tensor([9, 10]),
            torch.arange(9).reshape(3, 3),
        ]
        self._test_convex_split(tensor, split_tensors, 0, 10)

        split_tensors = [
            torch.tensor([[21, 22, 23], [24, 25, 26]]),
        ]
        self._test_convex_split(tensor, split_tensors, 21, 26)

        split_tensors = [
            torch.arange(9).reshape(3, 3) + 9,
        ]
        self._test_convex_split(tensor, split_tensors, 9, 17)

        split_tensors = [
            torch.tensor([8]),
            torch.tensor([18]),
            torch.arange(9).reshape(3, 3) + 9,
        ]
        self._test_convex_split(tensor, split_tensors, 8, 18)
