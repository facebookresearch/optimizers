"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import heapq
import operator
from collections.abc import Callable, Iterator, Sequence
from functools import cache, partial, reduce
from itertools import accumulate, chain, compress, islice, pairwise
from types import TracebackType
from typing import TypeVar

import torch
from torch import Tensor


@cache
def merge_small_dims(tensor_shape: tuple[int, ...], threshold: int) -> tuple[int, ...]:
    """Reshapes tensor by merging small dimensions.

    Args:
        tensor_shape (tuple[int, ...]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (tuple[int, ...]): New tensor shape.

    """

    # Squeeze tensor shape to remove dimension with 1; if all dimensions are 1,
    # then add a 1 to the tensor shape.
    squeezed_tensor_shape = list(filter(lambda t: t != 1, tensor_shape)) or [1]
    new_tensor_shape = [squeezed_tensor_shape[0]]
    for next_tensor_shape in islice(squeezed_tensor_shape, 1, None):
        if (new_dimension := new_tensor_shape[-1] * next_tensor_shape) <= threshold:
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)
    return tuple(new_tensor_shape)


def multi_dim_split(tensor: Tensor, split_size: int) -> tuple[Tensor, ...]:
    """Chunks tensor across multiple dimensions based on splits.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        split_size (int): Size of a single chunk.

    Returns:
        split_tensors (tuple[Tensor, ...]): List of tensors.

    """
    return reduce(
        lambda split_tensors, dim: tuple(
            s for t in split_tensors for s in torch.split(t, split_size, dim=dim)
        ),
        range(tensor.dim()),
        (tensor,),
    )


_CompressListType = TypeVar("_CompressListType")


def compress_list(
    complete_list: Sequence[_CompressListType], selector: Sequence[bool]
) -> tuple[_CompressListType, ...]:
    """Compresses sequence based on selector.

    NOTE: Despite the name, this function can compress both lists and tuples, but will always return
    a tuple in order to ensure downstream compatibility.

    Example:
        complete_list = ['a', 'b', 'c', 'd'] and selector = [True, False, True, False]:
        Result: ('a', 'c')

        Only elements from complete_list where the corresponding selector is True are included.

    Args:
        complete_list (Sequence[CompressListType]): Complete tuple of candidates.
        selector (Sequence[bool]): Mask that is True if state is active, False otherwise.

    Returns:
        compressed_tuple (tuple[CompressListType, ...]): Compressed list of candidates based on selector.

    """
    assert (
        len(complete_list) == len(selector)
    ), f"Inconsistent lengths between complete_list {len(complete_list)} and selector {len(selector)}!"
    return tuple(compress(complete_list, selector))


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size (bytes) of a given data type."""
    if dtype is torch.bool:
        return 1
    # Fast ceiling of bits/8 using (bits + 7) // 8
    return (
        (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits + 7
    ) // 8


def generate_pairwise_indices(input_list: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Generates accumulated pairwise indices for a given input list.

    For example, if input_list = (1, 3, 2),
        - First element (1) generates index range [0, 1)
        - Second element (3) generates index range [1, 4)
        - Third element (2) generates index range [4, 6)

    then this will output [(0, 1), (1, 4), (4, 6)].

    This is useful for generating interval indices for iterating through a list given the
    number of blocks within each parameter.

    Args:
        input_list (Sequence[int]): A list of integers specifying the number of elements within each partition.

    Returns:
        partition_indices: Iterator[tuple[int, int]]: An iterator containing pairs of indices which specify
            the start and the ending indices of each partition specified in the input_list.

    """
    return pairwise(accumulate(chain([0], input_list)))


_ParameterizeEnterExitContextType = TypeVar("_ParameterizeEnterExitContextType")


class ParameterizeEnterExitContext:
    """ParameterizeEnterExitContext is used for automatically invoking the enter and exit methods on the input within this context.

    Args:
        input_with_enter_exit_context (ParameterizeEnterExitContextType): Input whose state will be changed while entering and exiting the context by enter_method_caller and exit_method_caller and exit_method_caller respectively.
        enter_method_caller (Callable[[ParameterizeEnterExitContextType], None]): Method caller for entering the context.
        exit_method_caller (Callable[[ParameterizeEnterExitContextType], None]): Method caller for exiting the context.

    """

    def __init__(
        self,
        input_with_enter_exit_context: _ParameterizeEnterExitContextType,
        enter_method_caller: Callable[[_ParameterizeEnterExitContextType], None],
        exit_method_caller: Callable[[_ParameterizeEnterExitContextType], None],
    ) -> None:
        self._enter_method: Callable[[], None] = partial(
            enter_method_caller, input_with_enter_exit_context
        )
        self._exit_method: Callable[[], None] = partial(
            exit_method_caller, input_with_enter_exit_context
        )

    def __enter__(self) -> "ParameterizeEnterExitContext":
        self._enter_method()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._exit_method()


def distribute_buffer_sizes(
    buffer_sizes: tuple[int, ...],
    group_size: int,
) -> tuple[tuple[int, int], ...]:
    """Distribute given buffer sizes across ranks in a group.

    Buffer sizes will be rounded up for memory allocation. Buffers are distributed such that
    total buffer sizes of each rank are as even as possible. This is currently performed
    using a greedy algorithm. We do not currently consider computational cost
    or kernel launching overheads.

    Note: A better distribution strategy should try to minimize the delta of buffer sizes
    between the most and the least allocated groups.

    Args:
        buffer_sizes (tuple[int, ...]): Buffer sizes of blocks to be distributed.
        group_size (int): Number of groups to distribute across.

    Returns:
        buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
            buffer size for each block and its assigned rank.

    Example:
        Assuming ALIGNMENT_BYTES = 64, given buffer_sizes = [128, 64, 500, 256], group_size = 2
        -> buffer_size_ranks = [(128, 1), (64, 1), (512, 0), (256, 1)]

        This means buffer at index 0 (size 128) is assigned to rank 1,
        buffer at index 1 (size 64) is assigned to rank 1,
        buffer at index 2 (size 512) is assigned to rank 0, and
        buffer at index 3 (size 256) is assigned to rank 1.
    """
    ALIGNMENT_BYTES = (
        64  # necessary for determining buffer size, possibly hardware-dependent
    )

    # Convert each of buffer_sizes into smallest multiple of ALIGNMENT_BYTES that is >= buffer size.
    aligned_buffer_sizes = [
        (buffer_size + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES
        for buffer_size in buffer_sizes
    ]
    buffer_size_ranks = [(-1, -1)] * len(buffer_sizes)
    allocated_buffer_sizes = [(0, group_index) for group_index in range(group_size)]
    heapq.heapify(allocated_buffer_sizes)

    for index, aligned_buffer_size in sorted(
        enumerate(aligned_buffer_sizes),
        key=operator.itemgetter(1),
        reverse=True,
    ):
        # Greedily find the group with the least allocated buffer size and its group index
        # in order to allocate buffers on that group.
        (
            min_allocated_buffer_size,
            min_allocated_buffer_size_group_index,
        ) = heapq.heappop(allocated_buffer_sizes)

        heapq.heappush(
            allocated_buffer_sizes,
            (
                min_allocated_buffer_size + aligned_buffer_size,
                min_allocated_buffer_size_group_index,
            ),
        )
        buffer_size_ranks[index] = (
            aligned_buffer_size,
            min_allocated_buffer_size_group_index,
        )

    return tuple(buffer_size_ranks)
