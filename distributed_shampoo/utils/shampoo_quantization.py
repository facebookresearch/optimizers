"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import typing
from typing import Optional, Sequence, Tuple, Union

import torch
from distributed_shampoo.utils.shampoo_block_info import BlockInfo

from distributed_shampoo.utils.shampoo_utils import compress_list

from optimizer_modules import OptimizerModule
from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


_FLOAT_DTYPES: Tuple[torch.dtype, ...] = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


class QuantizedTensor(OptimizerModule):
    def __init__(
        self,
        quantized_values: Tensor,
        block_info: BlockInfo,
        min_value: Optional[Tensor] = None,
        max_value: Optional[Tensor] = None,
    ) -> None:
        self.quantized_values = quantized_values
        self.min_value = min_value
        self.max_value = max_value
        self.block_info = block_info

        if self.quantized_values.dtype in _FLOAT_DTYPES:
            assert min_value is None and max_value is None
        elif min_value is None and max_value is None:
            assert torch.count_nonzero(self.quantized_values) == 0
            self.min_value = torch.zeros(1)
            self.max_value = torch.zeros(1)

    @classmethod
    def init_from_dequantized_tensor(
        cls,
        dequantized_values: Tensor,
        quantized_dtype: torch.dtype,
        block_info: BlockInfo,
    ) -> "QuantizedTensor":
        quantized_values = block_info.allocate_zeros_tensor(
            shape=dequantized_values.shape,
            dtype=quantized_dtype,
            device=dequantized_values.device,
        )
        min_value, max_value = QuantizedTensor._quantize_and_return_metadata(
            dequantized_values=block_info.get_tensor(dequantized_values),
            quantized_values=block_info.get_tensor(quantized_values),
        )
        return cls(quantized_values, block_info, min_value, max_value)

    def dequantize(self, dequantized_dtype: torch.dtype) -> Tensor:
        if self.quantized_values.dtype == dequantized_dtype:
            return self.quantized_values
        elif self.quantized_values.dtype in _FLOAT_DTYPES:
            dequantized_values = torch.zeros_like(
                self.quantized_values, dtype=dequantized_dtype
            )
            QuantizedTensor._convert_float_to_float(
                dequantized_values, self.quantized_values
            )
            return dequantized_values
        else:
            raise NotImplementedError(
                f"Quantization for {self.quantized_values.dtype} is not yet supported!"
            )

    def quantize(self, dequantized_tensor: Tensor) -> None:
        self.min_value, self.max_value = self._quantize_and_return_metadata(
            dequantized_tensor, self.quantized_values
        )

    @staticmethod
    def _quantize_and_return_metadata(
        dequantized_values: Tensor,
        quantized_values: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        quantized_dtype = quantized_values.dtype
        if quantized_dtype in _FLOAT_DTYPES:
            QuantizedTensor._convert_float_to_float(
                dequantized_values, quantized_values
            )
            return None, None
        else:
            raise NotImplementedError(
                f"Quantization for {quantized_dtype} is not yet supported!"
            )

    @staticmethod
    def _convert_float_to_float(src: torch.Tensor, dest: torch.Tensor) -> None:
        dest.copy_(src)

    @staticmethod
    def _convert_float_to_bucket(src: torch.Tensor, dest: torch.Tensor) -> None:
        target_dtype = dest.dtype
        raise NotImplementedError(
            f"Quantization for {target_dtype} is not yet supported!"
        )


class QuantizedTensorList:
    def __init__(
        self,
        quantized_data: Union[
            Sequence[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]],
            Sequence[QuantizedTensor],
        ],
        quantized_dtype: torch.dtype,
        computation_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.quantized_value_list: Tuple[Tensor, ...]
        self._min_values: Tuple[Optional[Tensor], ...]
        self._max_values: Tuple[Optional[Tensor], ...]

        if all(isinstance(x, QuantizedTensor) for x in quantized_data):
            self.quantized_value_list = tuple(
                x.block_info.get_tensor(x.quantized_values)
                for x in quantized_data
                if isinstance(x, QuantizedTensor)
                # NOTE: this is a hack to make the type checker happy
            )
            self._min_values = tuple(
                x.min_value for x in quantized_data if isinstance(x, QuantizedTensor)
            )
            self._max_values = tuple(
                x.max_value for x in quantized_data if isinstance(x, QuantizedTensor)
            )
        elif all(isinstance(x, tuple) for x in quantized_data):
            self.quantized_value_list, self._min_values, self._max_values = zip(
                *quantized_data
            )
        else:
            raise TypeError(
                f"quantized_data must be {typing.get_type_hints(QuantizedTensorList.__init__)['quantized_data']} but get {type(quantized_data)}"
            )

        self.dequantized_value_list: Optional[Tuple[Tensor, ...]] = None

        assert all(
            value.dtype == quantized_dtype for value in self.quantized_value_list
        )
        self.quantized_dtype = quantized_dtype
        assert (
            computation_dtype in _FLOAT_DTYPES
        ), f"{computation_dtype=} is not supported! It must be one of {_FLOAT_DTYPES}!"
        self.computation_dtype = computation_dtype

        # All min/max values should be None, or no min/max values are None
        assert all(
            a is None and b is None
            for a, b in zip(self._min_values, self._max_values, strict=True)
        ) or not any(
            None in (a, b)
            for a, b in zip(self._min_values, self._max_values, strict=True)
        )

    def __len__(self) -> int:
        return len(self.quantized_value_list)

    def dequantize(self) -> Tuple[Tensor, ...]:
        if self.quantized_dtype == self.computation_dtype:
            return self.quantized_value_list
        elif self.quantized_dtype in _FLOAT_DTYPES:
            return self._convert_float_to_float(
                src_list=self.quantized_value_list, target_dtype=self.computation_dtype
            )
        else:
            raise NotImplementedError(
                f"Quantization for {self.quantized_dtype} is not yet supported!"
            )

    def dequantize_(self) -> None:
        if self.dequantized_value_list is not None:
            logger.warning(
                "Dequantized values are already stored; overwriting these values..."
            )

        self.dequantized_value_list = self.dequantize()

    def quantize(self, tensor_list: Tuple[Tensor, ...]) -> None:
        if (
            tensor_list is not self.dequantized_value_list
            and self.is_dequantized_stored()
        ):
            logger.warning(
                "Existing stored dequantized values.\nWriting quantized values with input tensor_list without using these stored dequantized values..."
            )

        if self.quantized_dtype in _FLOAT_DTYPES:
            self._convert_float_to_float(
                src_list=tensor_list,
                target_dtype=self.quantized_dtype,
                dest_list=self.quantized_value_list,
            )
        else:
            raise NotImplementedError(
                f"Quantization for {self.quantized_dtype} is not yet supported!"
            )

    def quantize_(self) -> None:
        if self.dequantized_value_list is None:
            logger.warning(
                f"No stored dequantized values {self.dequantized_value_list=}. Must first call dequantize_()."
            )
            return

        if self.quantized_dtype != self.computation_dtype:
            assert self.dequantized_value_list is not None  # make type checker happy
            self.quantize(self.dequantized_value_list)
            del self.dequantized_value_list
            torch.cuda.empty_cache()
        self.dequantized_value_list = None

    @property
    def dequantized_value(self) -> Tuple[Tensor, ...]:
        assert self.dequantized_value_list is not None
        return self.dequantized_value_list

    @property
    def quantized_value(self) -> Tuple[Tensor, ...]:
        return self.quantized_value_list

    def is_dequantized_stored(self) -> bool:
        return self.dequantized_value_list is not None

    def compress(self, selector: Tuple[bool, ...]) -> "QuantizedTensorList":
        assert self.dequantized_value_list is None
        masked_quantized_value_list = compress_list(self.quantized_value_list, selector)
        masked_min_values = compress_list(self._min_values, selector)
        masked_max_values = compress_list(self._max_values, selector)
        return QuantizedTensorList(
            tuple(
                zip(masked_quantized_value_list, masked_min_values, masked_max_values)
            ),
            self.quantized_dtype,
            self.computation_dtype,
        )

    def _convert_float_to_float(
        self,
        src_list: Tuple[Tensor, ...],
        target_dtype: torch.dtype,
        dest_list: Optional[Tuple[Tensor, ...]] = None,
    ) -> Tuple[Tensor]:
        if dest_list is None:
            dest_list = tuple(
                torch.zeros_like(tensor, dtype=target_dtype) for tensor in src_list
            )
        torch._foreach_copy_(dest_list, src_list)
        return dest_list

    def _convert_bucket_to_float(self) -> Tuple[Tensor, ...]:
        raise NotImplementedError(
            f"Quantization for {self.quantized_dtype} is not yet supported!"
        )

    def _convert_float_to_bucket(self) -> Tuple[Tensor, ...]:
        raise NotImplementedError(
            f"Quantization for {self.quantized_dtype} is not yet supported!"
        )
