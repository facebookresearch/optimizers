"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import re
import unittest
from unittest import mock

import torch

from distributed_shampoo.utils import shampoo_quantization
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
        torch.testing.assert_close(
            quantized_tensor.quantized_values, base_tensor, check_dtype=False
        )
        self.assertIsNone(quantized_tensor.min_value)
        self.assertIsNone(quantized_tensor.max_value)


class QuantizedTensorListInitTest(unittest.TestCase):
    def test_invalid_quantized_data_type(self) -> None:
        with mock.patch.object(
            shampoo_quantization,
            "isinstance",
            side_effect=lambda object, classinfo: False,
        ), self.assertRaisesRegex(
            TypeError,
            re.escape(
                "quantized_data must be typing.Union[typing.Sequence[typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor], typing.Optional[torch.Tensor]]], typing.Sequence[distributed_shampoo.utils.shampoo_quantization.QuantizedTensor]] but get <class 'list'>"
            ),
        ):
            QuantizedTensorList(
                quantized_data=[
                    (torch.randn(2, 2, dtype=torch.float16), None, None)
                    for _ in range(5)
                ],
                quantized_dtype=torch.float16,
                computation_dtype=torch.float64,
            )

    def test_invalid_computation_dtype(self) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            re.escape(
                "computation_dtype=torch.int64 is not supported! It must be one of (torch.float16, torch.bfloat16, torch.float32, torch.float64)!"
            ),
        ):
            QuantizedTensorList(
                quantized_data=[
                    (torch.randn(2, 2, dtype=torch.float16), None, None)
                    for _ in range(5)
                ],
                quantized_dtype=torch.float16,
                computation_dtype=torch.int64,
            )


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
                torch.testing.assert_close(deq_tensor, tensor, check_dtype=False)

        delta = 1.0
        torch._foreach_add_(deq_tensors, delta)

        self._quantized_tensors.quantize(deq_tensors)
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for quantized_tensor, tensor in zip(
            self._quantized_tensors.quantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(quantized_tensor=quantized_tensor, base_tensor=tensor):
                self.assertEqual(quantized_tensor.dtype, torch.float16)
                torch.testing.assert_close(quantized_tensor, tensor + delta)

        # Calling quantize() while dequantize_value_list exists should trigger warning message.
        self._quantized_tensors.dequantize_()
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            self._quantized_tensors.quantize(deq_tensors)
            self.assertIn(
                "Existing stored dequantized values.\nWriting quantized values with input tensor_list without using these stored dequantized values...",
                [r.msg for r in cm.records],
            )

    def test_inplace_dequantize_quantize(self) -> None:
        # Calling quantize_() before dequantize_() should trigger warning message.
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            self._quantized_tensors.quantize_()
            self.assertIn(
                "No stored dequantized values self.dequantized_value_list=None. Must first call dequantize_().",
                [r.msg for r in cm.records],
            )

        self._quantized_tensors.dequantize_()
        self.assertTrue(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(
            self._quantized_tensors.dequantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_close(deq_tensor, tensor, check_dtype=False)

        # Calling dequantize_() before consuming already stored dequantized value should trigger warning message.
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            self._quantized_tensors.dequantize_()
            self.assertIn(
                "Dequantized values are already stored; overwriting these values...",
                [r.msg for r in cm.records],
            )
        # All dequantized values should be there without any change because no changes in quantized values.
        self.assertTrue(self._quantized_tensors.is_dequantized_stored())
        for deq_tensor, tensor in zip(
            self._quantized_tensors.dequantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(deq_tensor=deq_tensor, base_tensor=tensor):
                self.assertEqual(deq_tensor.dtype, torch.float64)
                torch.testing.assert_close(deq_tensor, tensor, check_dtype=False)

        delta = 1.0
        torch._foreach_add_(self._quantized_tensors.dequantized_value, delta)

        self._quantized_tensors.quantize_()
        self.assertFalse(self._quantized_tensors.is_dequantized_stored())
        for quantized_tensor, tensor in zip(
            self._quantized_tensors.quantized_value, self._base_tensors, strict=True
        ):
            with self.subTest(quantized_tensor=quantized_tensor, base_tensor=tensor):
                self.assertEqual(quantized_tensor.dtype, torch.float16)
                torch.testing.assert_close(quantized_tensor, tensor + delta)
