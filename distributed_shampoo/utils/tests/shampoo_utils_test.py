"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest

import torch

from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    generate_pairwise_indices,
    get_dtype_size,
    merge_small_dims,
    multi_dim_split,
)


class MergeSmallDimsTest(unittest.TestCase):
    def test_merge_all_small_dims(self) -> None:
        dims = (1, 2, 5, 1)
        merged_dims = (10,)
        threshold = 10
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

    def test_merge_some_small_dims(self) -> None:
        dims = (1, 2, 5, 1)
        merged_dims = (2, 5)
        threshold = 1
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

    def test_merge_small_dims_for_single_dim(self) -> None:
        dims = torch.Size([2])
        merged_dims = (2,)
        threshold = 10
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

    def test_merge_small_dims_all_ones(self) -> None:
        dims = (1, 1, 1, 1)
        merged_dims = (1,)

        threshold = 10
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

        threshold = 1
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

    def test_merge_small_dims_empty(self) -> None:
        dims = (0,)
        merged_dims = (0,)

        threshold = 10
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)


class MultiDimSplitTest(unittest.TestCase):
    def test_multi_dim_split_for_one_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        expected_split_grad = (
            torch.arange(6).reshape(3, 2),
            torch.arange(6, 10).reshape(2, 2),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=3), expected_split_grad
        )

    def test_multi_dim_split_for_two_dim(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (
            torch.tensor([[0, 1], [3, 4]]),
            torch.tensor([[2], [5]]),
            torch.tensor([[6, 7], [9, 10]]),
            torch.tensor([[8], [11]]),
            torch.tensor([[12, 13]]),
            torch.tensor([[14]]),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=2), expected_split_grad
        )

    def test_multi_dim_split_without_spliting(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (
            torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=5), expected_split_grad
        )


class CompressListTest(unittest.TestCase):
    def test_compress_list(self) -> None:
        self.assertTupleEqual(compress_list([1, 2, 3], (True, True, False)), (1, 2))
        self.assertTupleEqual(compress_list([1, 2, 3], (False, True, True)), (2, 3))
        self.assertTupleEqual(compress_list([1, 2, 3], (True, False, True)), (1, 3))

    def test_compress_list_with_different_size(self) -> None:
        with self.assertRaisesRegex(AssertionError, re.escape("Inconsistent lengths")):
            compress_list(complete_list=[1, 2, 3], selector=(True, False))


class GetDTypeSizeTest(unittest.TestCase):
    def test_get_dtype_size(self) -> None:
        self.assertEqual(get_dtype_size(torch.int64), 8)
        self.assertEqual(get_dtype_size(torch.float32), 4)
        self.assertEqual(get_dtype_size(torch.bfloat16), 2)
        self.assertEqual(get_dtype_size(torch.bool), 1)


class GeneratePairwiseIndicesTest(unittest.TestCase):
    def test_generate_pairwise_indices(self) -> None:
        input_tuple = (1, 3, 2)
        expected_pairwise_indices = [(0, 1), (1, 4), (4, 6)]
        self.assertListEqual(
            list(generate_pairwise_indices(input_tuple)), expected_pairwise_indices
        )

    def test_generate_pairwise_indices_with_empty_list(self) -> None:
        input_tuple = ()
        expected_pairwise_indices = []
        self.assertListEqual(
            list(generate_pairwise_indices(input_tuple)), expected_pairwise_indices
        )
