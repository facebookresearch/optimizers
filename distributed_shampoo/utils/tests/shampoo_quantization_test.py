"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest

import torch

from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_quantization import (
    QuantizedTensor,
    QuantizedTensorList,
)


class QuantizedTensorTest(unittest.TestCase):
    def test_init_float_from_dequantized(self) -> None:
        base_tensor = torch.rand(10)
        quantized_tensor = QuantizedTensor.init_from_dequantized_tensor(
            base_tensor,
            torch.float16,
            BlockInfo(torch.zeros(10), (0, "dummy")),
        )
        torch.testing.assert_allclose(quantized_tensor.quantized_values, base_tensor)
        self.assertIsNone(quantized_tensor.min_value)
        self.assertIsNone(quantized_tensor.max_value)


class QuantizedTensorListTest(unittest.TestCase):
    def setUp(self) -> None:
        self._base_tensors = tuple(
            torch.randn(10, 10, dtype=torch.float16) for _ in range(5)
        )
        self._quantized_tensors = QuantizedTensorList(
            tuple((torch.clone(tensor), None, None) for tensor in self._base_tensors),
            quantized_dtype=torch.float16,
            computation_dtype=torch.float64,
        )

    def test_init_from_quantized_tensors(self) -> None:
        quantized_tensors = [
            QuantizedTensor(
                torch.ones(10, dtype=torch.float16) * i,
                BlockInfo(torch.zeros(10), (i, "dummy")),
            )
            for i in range(5)
        ]
        quantized_tensor_list = QuantizedTensorList(
            quantized_tensors, torch.float16, torch.float64
        )
        for i, tensor in enumerate(quantized_tensor_list.quantized_value):
            self.assertFalse(torch.any(torch.nonzero(tensor - i)))

    def test_dequantize_quantize(self) -> None:
        deq_tensors = self._quantized_tensors.dequantize()
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(deq_tensors, self._base_tensors, strict=True):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_allclose(deq_tensor, tensor)

        delta = 1.0
        torch._foreach_add_(deq_tensors, delta)

        self._quantized_tensors.quantize(deq_tensors)
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for quantized_tensor, tensor in zip(
            self._quantized_tensors.quantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(quantized_tensor=quantized_tensor, base_tensor=tensor):
                self.assertEqual(quantized_tensor.dtype, torch.float16)
                torch.testing.assert_allclose(quantized_tensor, tensor + delta)

    def test_inplace_dequantize_quantize(self) -> None:
        self._quantized_tensors.dequantize_()
        self.assertTrue(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(
            self._quantized_tensors.dequantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_allclose(deq_tensor, tensor)

        delta = 1.0
        torch._foreach_add_(self._quantized_tensors.dequantized_value, delta)

        self._quantized_tensors.quantize_()
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for quantized_tensor, tensor in zip(
            self._quantized_tensors.quantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(quantized_tensor=quantized_tensor, base_tensor=tensor):
                self.assertEqual(quantized_tensor.dtype, torch.float16)
                torch.testing.assert_allclose(quantized_tensor, tensor + delta)
