"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import re
import unittest
from typing import List

import torch

from distributed_shampoo.utils.shampoo_fsdp_distributor import FSDPDistributor
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


class SplitTensorBlockRecoveryTest(unittest.TestCase):
    def _test_split_tensor_block_recovery(
        self,
        original_tensor: Tensor,
        expected_split_tensors: List[Tensor],
        start_idx: int,
        end_idx: int,
    ) -> None:
        actual_split_tensors = FSDPDistributor._split_tensor_block_recovery(
            original_tensor.flatten()[start_idx:end_idx],
            original_tensor.size(),
            start_idx,
            end_idx,
        )

        self.assertNotEqual(len(actual_split_tensors), 0)
        torch.testing.assert_close(actual_split_tensors, expected_split_tensors)

    def test_illegal_tensor_shard_size(self) -> None:
        with self.assertRaisesRegex(ValueError, re.escape("Input tensor is not flat")):
            FSDPDistributor._split_tensor_block_recovery(
                tensor_shard=torch.randn((3, 4)),
                original_shape=torch.Size((3, 4)),
                start_idx=0,
                end_idx=16,
            )

    def test_split_tensor_block_recovery_for_one_dim(self) -> None:
        original_tensor = torch.arange(5)
        with self.subTest("Test tensor without modification"):
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=[torch.arange(5)],
                start_idx=0,
                end_idx=5,
            )
        with self.subTest("Test tensor with indices [1, 4)"):
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=[torch.arange(1, 4)],
                start_idx=1,
                end_idx=4,
            )

    def test_split_tensor_block_recovery_for_two_dim(self) -> None:
        original_tensor = torch.arange(15).reshape(3, 5)

        with self.subTest("Test with indices [0, 11)"):
            actual_split_tensors = [
                torch.arange(10).reshape(2, 5),
                torch.tensor([10]),
            ]
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=actual_split_tensors,
                start_idx=0,
                end_idx=11,
            )

        with self.subTest("Test with indices [3, 15)"):
            actual_split_tensors = [
                torch.arange(3, 5),
                torch.arange(5, 15).reshape(2, 5),
            ]
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=actual_split_tensors,
                start_idx=3,
                end_idx=15,
            )

        with self.subTest("Test with indices [3, 4)"):
            actual_split_tensors = [
                torch.tensor([3]),
            ]
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=actual_split_tensors,
                start_idx=3,
                end_idx=4,
            )

    def test_split_tensor_block_recovery_for_three_dim(self) -> None:
        original_tensor = torch.arange(27).reshape(3, 3, 3)

        with self.subTest("Test with indices [0, 9)"):
            actual_split_tensors = [
                torch.arange(9).reshape(1, 3, 3),
            ]
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=actual_split_tensors,
                start_idx=0,
                end_idx=9,
            )

        with self.subTest("Test with indices [8, 10)"):
            actual_split_tensors = [
                torch.tensor([8]),
                torch.tensor([9]),
            ]
            self._test_split_tensor_block_recovery(
                original_tensor=original_tensor,
                expected_split_tensors=actual_split_tensors,
                start_idx=8,
                end_idx=10,
            )
