"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import math
from itertools import accumulate, chain, compress, pairwise
from typing import Iterator, Sequence, Tuple, TypeVar

import torch
from torch import Tensor


def merge_small_dims(tensor_shape: Sequence[int], threshold: int) -> Tuple[int, ...]:
    """Reshapes tensor by merging small dimensions.

    Args:
        tensor_shape (Sequence[int]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (List[int]): New tensor shape.

    """

    # Squeeze tensor shape to remove dimension with 1; if all dimensions are 1,
    # then add a 1 to the tensor shape.
    squeezed_tensor_shape = list(filter(lambda t: t != 1, tensor_shape)) or [1]
    new_tensor_shape = [squeezed_tensor_shape[0]]
    for next_tensor_shape in squeezed_tensor_shape[1:]:
        if (new_dimension := new_tensor_shape[-1] * next_tensor_shape) <= threshold:
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)
    return tuple(new_tensor_shape)


def multi_dim_split(tensor: Tensor, split_size: int) -> Tuple[Tensor, ...]:
    """Chunks tensor across multiple dimensions based on splits.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        split_size (int): Size of a single chunk.

    Returns:
        split_grad (List[Tensor]): List of tensors.

    """
    split_tensors = (tensor,)
    if all(s <= split_size for s in tensor.size()):
        return split_tensors

    for dim in range(tensor.dim()):
        split_tensors = tuple(
            s for t in split_tensors for s in torch.split(t, split_size, dim=dim)
        )
    return split_tensors


CompressListType = TypeVar("CompressListType")


def compress_list(
    complete_list: Sequence[CompressListType], selector: Sequence[bool]
) -> Tuple[CompressListType, ...]:
    """Compresses sequence based on selector.

    NOTE: Despite the name, this function can compress both lists and tuples, but will always return
    a tuple in order to ensure downstream compatibility.

    Args:
        complete_list (Sequence[CompressListType]): Complete tuple of candidates.
        selector (Sequence[bool]): Mask that is True if state is active, False otherwise.

    Returns:
        compressed_tuple (Tuple[CompressListType, ...]): Compressed list of candidates based on selector.

    """
    assert len(complete_list) == len(
        selector
    ), f"Inconsistent lengths between complete_list {len(complete_list)} and selector {len(selector)}!"
    return tuple(compress(complete_list, selector))


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size (bytes) of a given data type."""
    if dtype is torch.bool:
        return 1
    return math.ceil(
        (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits / 8.0
    )


def generate_pairwise_indices(input_list: Sequence[int]) -> Iterator[Tuple[int, int]]:
    """Generates accumulated pairwise indices for a given input list.

    For example, if input_list = (1, 3, 2), then this will output [(0, 1), (1, 4), (4, 6)].
    This is useful for generating interval indices for iterating through a list given the
    number of blocks within each parameter.

    Args:
        input_list (Sequence[int]): A list of intergers specifying the number of elements within each partition.

    Returns:
        partition_indices: Iterator[Tuple[int, int]]: An iterator containing pairs of indices which specify
            the start and the ending indices of each partition specified in the input_list.

    """
    return pairwise(accumulate(chain([0], input_list)))
