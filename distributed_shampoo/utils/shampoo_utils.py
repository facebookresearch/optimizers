"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import copy
import heapq
import logging
import math
import operator
from collections.abc import Callable, Iterator, Sequence
from functools import cache, partial, reduce
from itertools import accumulate, chain, compress, islice, pairwise
from types import TracebackType
from typing import Any, TypeGuard, TypeVar

import torch
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    HybridShardDistributedConfig,
    LoadBalancingConfig,
    OWNER_RANKS,
    PARAMS,
)
from distributed_shampoo.utils.load_balancing_utils import DefaultCostModel
from torch import distributed as dist, Tensor
from torch.distributed.tensor import DTensor

logger: logging.Logger = logging.getLogger(__name__)


@cache
def merge_small_dims(
    tensor_shape: tuple[int, ...],
    threshold: int,
    target_tensor_dimensionality: int | float,
) -> tuple[int, ...]:
    """Reshapes tensor by merging small dimensions.

    This function merges adjacent dimensions of a tensor when their product is below
    the specified threshold, which helps optimize operations on tensors with many
    small dimensions.

    Note:
    - Shampoo will promote 0D tensor (torch.Size([]) into an 1D tensor (torch.Size([1])).
    - Empty tensors (with a dimension of size 0) will return a shape of (0,).
    - Dimensions of size 1 are removed (squeezed) before merging.
    - If all dimensions are 1, it returns (1,).
    - Dimensions are merged in reverse order to accommodate PyTorch's tensor layout.

    Args:
        tensor_shape (tuple[int, ...]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.
        target_tensor_dimensionality (int | float): Target dimensionality of the tensor. Only merge until the target dimensionality is reached.
            If target_tensor_dimensionality > len(tensor_shape), then no merging occurs. The only float that can be used is math.inf.
            Note that the output tensor will NOT necessarily have dimension equal to target_tensor_dimensionality.
            The merging will stop before reaching target_tensor_dimensionality if the threshold is small.

    Returns:
        new_tensor_shape (tuple[int, ...]): New tensor shape after merging dimensions.

    Raises:
        AssertionError: If target_tensor_dimensionality is a float but not math.inf.

    Example:
        - merge_small_dims((1, 2, 5, 1), threshold=10, target_tensor_dimensionality=1) -> (10,)
          All dimensions are merged as their product (10) is equal to the threshold.

        - merge_small_dims((1, 2, 5, 1), threshold=1, target_tensor_dimensionality=1) -> (2, 5)
          Dimensions of size 1 are removed, and no merging occurs as 2*5 > threshold.

        - merge_small_dims((32, 3, 64, 64), threshold=8192, target_tensor_dimensionality=1) -> (96, 4096)
          For convolution-like dimensions, merges into (32*3, 64*64) as 96 < threshold
          but 96*4096 > threshold.

        - merge_small_dims((32, 3, 64, 64), threshold=1_000_000, target_tensor_dimensionality=2) -> (32, 12_288)
          For convolution-like dimensions, merges into (32, 3*64*64) despite 32*3*64*64 < threshold because
          target_tensor_dimensionality is 2. This is useful for spectral descent methods like Muon.

    """
    if 0 in tensor_shape:
        return (0,)

    if isinstance(target_tensor_dimensionality, float):
        assert target_tensor_dimensionality == math.inf, (
            f"{target_tensor_dimensionality=} has to be an integer or math.inf."
        )
        return tensor_shape

    # Squeeze tensor shape to remove dimension with 1; if all dimensions are 1,
    # then add a 1 to the tensor shape.
    # We merge dimensions in reverse order to accommodate PyTorch's general tensor layout.
    # This is particularly useful for convolution operations where kernel sizes are typically
    # placed at the end of the tensor shape.
    squeezed_tensor_shape = list(filter(lambda t: t != 1, reversed(tensor_shape))) or [
        1
    ]
    squeezed_dimensionality = len(squeezed_tensor_shape)
    new_tensor_shape = [squeezed_tensor_shape[0]]
    for num_processed_dimensions, next_tensor_shape in enumerate(
        islice(squeezed_tensor_shape, 1, None), start=1
    ):
        current_dimensionality = len(new_tensor_shape)
        remaining_dimensions = squeezed_dimensionality - num_processed_dimensions
        potential_dimensionality_before_merge = (
            current_dimensionality + remaining_dimensions
        )
        if (
            potential_dimensionality_before_merge > target_tensor_dimensionality
            and (new_dimension := new_tensor_shape[-1] * next_tensor_shape) <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)
    return tuple(reversed(new_tensor_shape))


def multi_dim_split(tensor: Tensor, split_size: int | float) -> tuple[Tensor, ...]:
    """Chunks tensor across multiple dimensions based on splits.

    This function recursively splits a tensor along all of its dimensions using the
    specified split size. It applies torch.split() to each dimension sequentially,
    resulting in a tuple of smaller tensors.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        split_size (int | float): Size of a single chunk along each dimension.
            If math.inf is provided, no splitting occurs.

    Returns:
        split_tensors (tuple[Tensor, ...]): Tuple of tensors after splitting.
            If split_size is greater than or equal to any dimension size,
            no splitting occurs along that dimension.

    Example:
        - multi_dim_split(tensor of shape (5, 2), split_size=3):
          Returns (tensor([0, 1, 2], [0, 1]), tensor([3, 4], [0, 1]))
          Splits only along dimension 0 since split_size > dimension 1 size.

        - multi_dim_split(tensor of shape (5, 3), split_size=2):
          First splits along dimension 0:
          [(0-1, 0-2), (2-3, 0-2), (4, 0-2)]

          Then splits each chunk along dimension 1:
          [(0-1, 0-1), (0-1, 2), (2-3, 0-1), (2-3, 2), (4, 0-1), (4, 2)]

          Returns 6 smaller tensors.

        - multi_dim_split(tensor of shape (5, 3), split_size=5):
          Returns (original tensor,) since split_size ≥ all dimensions.

        - multi_dim_split(tensor of shape (5, 3), split_size=math.inf):
          Returns (original tensor,) since math.inf means no splitting.

    """
    if isinstance(split_size, float):
        assert split_size == math.inf, (
            f"{split_size=} has to be an integer or math.inf."
        )
        return (tensor,)

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

    Args:
        complete_list (Sequence[CompressListType]): Complete tuple of candidates.
        selector (Sequence[bool]): Mask that is True if state is active, False otherwise.

    Returns:
        compressed_tuple (tuple[CompressListType, ...]): Compressed list of candidates based on selector.

    Example:
        complete_list = ['a', 'b', 'c', 'd'] and selector = [True, False, True, False]:
        Result: ('a', 'c')

        Only elements from complete_list where the corresponding selector is True are included.

    """
    assert len(complete_list) == len(selector), (
        f"Inconsistent lengths between complete_list {len(complete_list)} and selector {len(selector)}!"
    )
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

    This is useful for generating interval indices for iterating through a list given the
    number of blocks within each parameter.

    Args:
        input_list (Sequence[int]): A list of integers specifying the number of elements within each partition.

    Returns:
        partition_indices (Iterator[tuple[int, int]]): An iterator containing pairs of indices which specify
            the start and the ending indices of each partition specified in the input_list.

    Example:
        If input_list = (1, 3, 2),
            - First element (1) generates index range [0, 1)
            - Second element (3) generates index range [1, 4)
            - Third element (2) generates index range [4, 6)

        then this will output [(0, 1), (1, 4), (4, 6)].

    """
    return pairwise(accumulate(chain([0], input_list)))


_ParameterizeEnterExitContextType = TypeVar("_ParameterizeEnterExitContextType")


class ParameterizeEnterExitContext:
    """ParameterizeEnterExitContext is used for automatically invoking the enter and exit methods on the input within this context.

    Args:
        input_with_enter_exit_context (ParameterizeEnterExitContextType): Input whose state will be changed while entering and exiting the context by enter_method_caller and exit_method_caller respectively.
        enter_method_caller (Callable[[ParameterizeEnterExitContextType], Any]): Method caller for entering the context.
        exit_method_caller (Callable[[ParameterizeEnterExitContextType], Any]): Method caller for exiting the context.

    """

    def __init__(
        self,
        input_with_enter_exit_context: _ParameterizeEnterExitContextType,
        enter_method_caller: Callable[[_ParameterizeEnterExitContextType], Any],
        exit_method_caller: Callable[[_ParameterizeEnterExitContextType], Any],
    ) -> None:
        self._enter_method: Callable[[], Any] = partial(
            enter_method_caller, input_with_enter_exit_context
        )
        self._exit_method: Callable[[], Any] = partial(
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
    blocked_params: tuple[Tensor, ...],
    group_size: int,
    load_balancing_config: LoadBalancingConfig,
) -> tuple[tuple[int, int], ...]:
    """Distribute given param blocks across ranks in a group.

    Param blocks are distributed such that the total assigned load of each rank is as even as possible.
    The load of a param block is determined by ``load_balance_config``.
    By default, the load is measured purely by buffer size. If ``load_balance_config`` specifies
    a compute-based strategy, the distribution will instead weigh each buffer by its estimated
    computational cost (e.g., cost of processing or kernel execution time) rather than size alone.
    This is currently performed using a greedy algorithm.

    Note: A better distribution strategy should try to minimize the delta of buffer sizes
    between the most and the least allocated groups.

    Args:
        blocked_params (tuple[Tensor, ...]): A list of blocked parameters.
        group_size (int): Number of groups to distribute across.
        load_balancing_config (LoadBalancingConfig): Memory or compute load balancing config.

    Returns:
        buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
            buffer size for each block and its assigned rank.
    """
    buffer_sizes_aligned = tuple(
        int(DefaultCostModel.cost(blocked_param)) for blocked_param in blocked_params
    )

    param_block_costs = tuple(
        load_balancing_config.cost_model.cost(block) for block in blocked_params
    )
    param_block_ranks = [-1] * len(blocked_params)
    allocated_load_sizes = [(0.0, group_index) for group_index in range(group_size)]
    heapq.heapify(allocated_load_sizes)

    for index, block_cost in sorted(
        enumerate(param_block_costs),
        key=operator.itemgetter(1),
        reverse=True,
    ):
        # Greedily find the group with the least allocated load and its group index
        # in order to allocate buffers on that group.
        (
            min_allocated_load,
            min_allocated_load_group_index,
        ) = heapq.heappop(allocated_load_sizes)

        new_load_size = min_allocated_load + block_cost

        heapq.heappush(
            allocated_load_sizes,
            (
                new_load_size,
                min_allocated_load_group_index,
            ),
        )
        param_block_ranks[index] = min_allocated_load_group_index

    buffer_size_ranks = tuple(zip(buffer_sizes_aligned, param_block_ranks, strict=True))

    return buffer_size_ranks


def prepare_update_param_buffers(
    params: tuple[DTensor, ...], group_size: int
) -> list[Tensor]:
    """Allocates a persistent shadow copy of updated parameters."""
    if any(p.dtype != params[0].dtype for p in params):
        raise NotImplementedError(
            "When using round-robin assignment in FSDP Shampoo, parameters of "
            "different dtypes are not currently supported."
        )

    param_sizes = [p.to_local().numel() for p in params]
    buffer_size = sum(param_sizes)
    buffer = params[0].to_local().new_zeros(buffer_size)
    buffer_offsets = list(accumulate(param_sizes))

    def round_up_to_multiple_of(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    pad_len = round_up_to_multiple_of(len(buffer_offsets), group_size) - len(
        buffer_offsets
    )

    # The padding logic below handles when a rank has some parameters but fewer than group size.
    # For example, if group size is 4 and there are 3 parameters, it will pad a 0-sized tensor at the end.
    # Example:
    #   Assume we have 3 parameters and group size is 4. param0: 100, param1: 200, param2: 300.
    #   buffer_offsets = [100, 300, 600, 600] (note that the last element is 600)
    #   This buffer for communication have 4 chunks.
    #   - Rank 0: [0, 100)
    #   - Rank 1: [100, 300)
    #   - Rank 2: [300, 600)
    #   - Rank 3: [600, 600) (empty tensor)
    # Pad the list with empty tensors to ensure each rank participates in all-to-all.
    buffer_offsets.extend([buffer_size] * pad_len)
    # Drop the last element as torch.tensor_split takes indices as split points.
    buffer_offsets = buffer_offsets[:-1]

    return list(torch.tensor_split(buffer, buffer_offsets))


def redistribute_and_update_params(
    params: tuple[DTensor, ...],
    local_full_params: list[Tensor],
    update_param_buffers: list[Tensor],
    dist_group: torch.distributed.ProcessGroup,
) -> None:
    """Redistributes updated parameters to each parameter's rank."""
    group_size = dist_group.size()

    # Run all-to-all collectives to exchange the updated parameters across
    # ranks in group. This implementation runs multiple rounds of a2a ops
    # if the number of parameters is larger than the world size.
    a2a_rounds = range(len(update_param_buffers) // group_size)
    logger.info(f"Running {len(a2a_rounds)} rounds of a2a ops.")
    for a2a_round in a2a_rounds:
        # Send either a valid full parameter, or a padding zero tensor.
        send_param = (
            local_full_params[a2a_round]
            if a2a_round < len(local_full_params)
            else params[0].to_local().new_zeros(0)
        )
        # Chunk the send_param to exactly group_size slices to distribute to
        # all ranks. We need to manually pad the result of torch.chunk since
        # it does not guarantee that the result has the desired chunks.
        send_list = [t.flatten() for t in torch.chunk(send_param, group_size, dim=0)]
        if len(send_list) < group_size:
            # NOTE: Intentionally use `torch.tensor_split` here to do a trivial
            # split to ensure that the padding is in contiguous memory space as
            # is required for all-to-all collectives.
            append_len = group_size - len(send_list)
            last_t = send_list[-1]
            split_indices = [send_list[-1].shape[0]] * append_len
            send_list.extend(torch.tensor_split(last_t, split_indices, dim=0)[1:])
        assert len(send_list) == group_size

        # Specify receive list as a range of update_param_buffers.
        recv_list = update_param_buffers[
            a2a_round * group_size : (a2a_round + 1) * group_size
        ]

        dist.all_to_all(recv_list, send_list, dist_group)

    torch._foreach_copy_(
        [p.to_local().flatten() for p in params], update_param_buffers[: len(params)]
    )


def _compute_chunk_sizes(numel: int, num_chunks: int) -> list[int]:
    """Compute chunk sizes that torch.chunk would produce.

    torch.chunk(tensor, n, dim) semantics for n chunks along dim with size d:
    - Computes chunk_size = ceil(d / n)
    - First (n-1) chunks get chunk_size elements, last chunk gets remainder
    - If d < n, only d chunks are produced (each of size 1)

    Args:
        numel (int): Size of the dimension to split.
        num_chunks (int): Number of chunks to split into.

    Returns:
        sizes (list[int]): List of chunk sizes, one per chunk.
    """
    if numel == 0:
        return [0] * num_chunks

    if numel >= num_chunks:
        # torch.chunk computes ceil(numel / num_chunks) as the chunk size
        chunk_size = (numel + num_chunks - 1) // num_chunks  # This is ceil division
        # First chunks get chunk_size, last chunk gets remainder
        sizes = []
        remaining = numel
        for _i in range(num_chunks):
            if remaining <= 0:
                sizes.append(0)
            elif remaining >= chunk_size:
                sizes.append(chunk_size)
                remaining -= chunk_size
            else:
                sizes.append(remaining)
                remaining = 0
        return sizes
    else:
        # numel < num_chunks: each element becomes its own chunk
        # Ranks beyond numel get empty chunks (size 0)
        sizes = [1] * numel + [0] * (num_chunks - numel)
        return sizes


def _compute_param_chunk_sizes(
    params: tuple[DTensor, ...], group_size: int
) -> list[list[int]]:
    """Compute per-rank chunk sizes for all params based on FSDP dim-0 sharding.

    For each parameter, computes how many elements each rank holds after
    FSDP's dim-0 sharding (torch.chunk semantics).

    Args:
        params (tuple[DTensor, ...]): Tuple of DTensor parameters.
        group_size (int): Number of ranks in the process group.

    Returns:
        param_chunk_sizes (list[list[int]]): List of lists, where param_chunk_sizes[i][r] is the
            number of elements rank r holds for parameter i.
    """
    param_chunk_sizes: list[list[int]] = []
    for param in params:
        global_shape = tuple(param.shape)

        if len(global_shape) == 0:
            # Scalar tensor - all on rank 0
            chunk_sizes = [1] + [0] * (group_size - 1)
        else:
            dim0_size = global_shape[0]
            remaining_numel = math.prod(global_shape[1:])

            # Compute dim-0 chunk sizes using torch.chunk semantics
            dim0_chunks = _compute_chunk_sizes(dim0_size, group_size)
            # Multiply by remaining dimensions to get actual element counts
            chunk_sizes = [c * remaining_numel for c in dim0_chunks]

        param_chunk_sizes.append(chunk_sizes)

    return param_chunk_sizes


def _build_buffer_views(buffer: Tensor, sizes: list[int]) -> list[Tensor]:
    """Build contiguous views into a flat buffer given per-section sizes."""
    views: list[Tensor] = []
    offset = 0
    for size in sizes:
        views.append(buffer[offset : offset + size])
        offset += size
    return views


class RedistributeParamsContext:
    """Context for optimized parameter redistribution using all_to_all.

    This class precomputes all static information needed for efficient parameter
    redistribution during initialization, then uses that information to perform
    fast redistribution using a single all_to_all collective instead of
    multiple point-to-point calls.

    The key insight is that torch.chunk(tensor, group_size, dim=0) is used to split
    full params, and FSDP shards along dim=0. So chunk[i] corresponds to rank i's
    local shard.

    TODO(irisz): Currently assumes dim-0 sharding only. FSDP2 supports non-dim-0 sharding
    but it is not enabled on APS. Extend to support arbitrary shard dimensions if needed.

    Args:
        params (tuple[DTensor, ...]): Tuple of DTensor parameters to redistribute.
        assigned_params_mask (tuple[bool, ...]): Boolean mask indicating which params this rank computes.
        dist_group (torch.distributed.ProcessGroup): Process group for communication.

    Example:
        # During initialization:
        ctx = RedistributeParamsContext(params, assigned_mask, dist_group)

        # During each optimizer step:
        ctx.redistribute_and_update_params(local_full_params)
    """

    @torch.no_grad()
    def __init__(
        self,
        params: tuple[DTensor, ...],
        assigned_params_mask: tuple[bool, ...],
        dist_group: torch.distributed.ProcessGroup,
    ) -> None:
        self._params: tuple[DTensor, ...] = params
        self._assigned_params_mask: tuple[bool, ...] = assigned_params_mask
        self._dist_group: torch.distributed.ProcessGroup = dist_group
        self._group_size: int = dist_group.size()
        self._rank: int = dist.get_rank(group=dist_group)

        if not params:
            raise ValueError("params cannot be empty")

        if len(params) != len(assigned_params_mask):
            raise ValueError(
                f"len(params) ({len(params)}) != "
                f"len(assigned_params_mask) ({len(assigned_params_mask)})"
            )

        self._dtype: torch.dtype = params[0].to_local().dtype
        self._device: torch.device = params[0].to_local().device

        # Validate all params have same dtype
        if any(p.to_local().dtype != self._dtype for p in params):
            raise NotImplementedError(
                "When using optimized redistribution, parameters of "
                "different dtypes are not currently supported."
            )

        # Precompute static metadata
        self._precompute_metadata()

    def _precompute_metadata(self) -> None:
        """Precompute all static information needed for redistribution.

        Key insight: FSDP2 uses torch.chunk for dim-0 sharding. We can compute
        chunk sizes mathematically during initialization.

        torch.chunk(tensor, n) semantics:
        - If numel >= n: returns n chunks, each of size ceil(numel/n) except last
        - If numel < n: returns numel chunks of size 1 (fewer than n chunks!)
        """
        group_size = self._group_size

        # Store the local shard numel for each param (what FSDP actually shards to)
        self._param_local_numels: list[int] = [
            p.to_local().numel() for p in self._params
        ]
        # Store the global numel for each param
        self._param_global_numels: list[int] = [p.numel() for p in self._params]

        # Compute which params each rank owns (based on round-robin assignment)
        self._local_param_indices: list[int] = [
            i for i, assigned in enumerate(self._assigned_params_mask) if assigned
        ]

        # Compute chunk sizes using FSDP's dim-0 sharding semantics
        self._param_chunk_sizes: list[list[int]] = _compute_param_chunk_sizes(
            self._params, group_size
        )

        # Compute send sizes: sum of chunk sizes for owned params going to each peer
        self._send_sizes: list[int] = []
        for peer_rank in range(group_size):
            send_size = sum(
                self._param_chunk_sizes[param_idx][peer_rank]
                for param_idx in self._local_param_indices
            )
            self._send_sizes.append(send_size)
        self._total_send_size = sum(self._send_sizes)

        # Compute recv sizes: sum of chunk sizes for params owned by each peer
        self._recv_sizes: list[int] = []
        for peer_rank in range(group_size):
            peer_param_indices = [
                i for i in range(len(self._params)) if i % group_size == peer_rank
            ]
            recv_size = sum(
                self._param_chunk_sizes[param_idx][self._rank]
                for param_idx in peer_param_indices
            )
            self._recv_sizes.append(recv_size)
        self._total_recv_size = sum(self._recv_sizes)

        # Pre-allocate persistent send and recv buffers for all_to_all.
        # Memory overhead: O(total_params / world_size) each.
        #   - send_buffer holds chunked assigned (owned) params = total_params / world_size
        #   - recv_buffer holds received shards for all params from all peers,
        #     but each param's local shard is ~1/world_size of its full size,
        #     so total recv ≈ total_params / world_size as well.
        self._send_buffer = torch.empty(
            self._total_send_size, dtype=self._dtype, device=self._device
        )
        self._recv_buffer = torch.empty(
            self._total_recv_size, dtype=self._dtype, device=self._device
        )

        # Pre-compute send_list and recv_list as contiguous views into buffers.
        self._send_list = _build_buffer_views(self._send_buffer, self._send_sizes)
        self._recv_list = _build_buffer_views(self._recv_buffer, self._recv_sizes)

        # Pre-compute send and recv views for batched operations.
        # Forward-declare attributes set by helper methods for Pyre.
        self._send_dst_views: list[Tensor] = []
        self._send_param_indices: list[int] = []
        self._send_peer_ranks: list[int] = []
        self._param_to_local_idx: dict[int, int] = {}
        self._local_param_global_shapes: list[tuple[int, ...]] = []
        self._param_recv_info: list[tuple[int, int]] = []
        self._recv_src_views: list[Tensor] = []
        self._recv_dst_views: list[Tensor] = []
        self._compute_send_views()
        self._compute_recv_views()

        logger.info(
            f"RedistributeParamsContext initialized: "
            f"rank={self._rank}, group_size={group_size}, "
            f"num_params={len(self._params)}, "
            f"local_params={len(self._local_param_indices)}, "
            f"send_sizes={self._send_sizes}, "
            f"recv_sizes={self._recv_sizes}, "
            f"param_local_numels={self._param_local_numels}, "
            f"param_global_numels={self._param_global_numels}, "
            f"param_chunk_sizes={self._param_chunk_sizes}"
        )

    def _compute_send_views(self) -> None:
        """Pre-compute per-(local_param, peer_rank) destination views into send buffer."""
        group_size = self._group_size
        send_offset = 0
        for peer_rank in range(group_size):
            for param_idx in self._local_param_indices:
                chunk_size = self._param_chunk_sizes[param_idx][peer_rank]
                if chunk_size > 0:
                    self._send_dst_views.append(
                        self._send_buffer[send_offset : send_offset + chunk_size]
                    )
                    self._send_param_indices.append(param_idx)
                    self._send_peer_ranks.append(peer_rank)
                    send_offset += chunk_size

        # Build a mapping from param_idx to local_idx for fast lookup.
        self._param_to_local_idx.update(
            {
                param_idx: local_idx
                for local_idx, param_idx in enumerate(self._local_param_indices)
            }
        )

        # Pre-compute global shapes for owned params (used for chunking).
        self._local_param_global_shapes.extend(
            tuple(self._params[param_idx].shape)
            for param_idx in self._local_param_indices
        )

    def _compute_recv_views(self) -> None:
        """Pre-compute recv unpacking info and copy-out views."""
        group_size = self._group_size

        # Precompute recv unpacking info
        # Recv buffer layout: [data from rank 0][data from rank 1]...
        # _param_recv_info[param_idx] = (offset in recv_buffer, chunk_size)
        self._param_recv_info = [(-1, -1)] * len(self._params)
        recv_offset = 0
        for peer_rank in range(group_size):
            peer_param_indices = [
                i for i in range(len(self._params)) if i % group_size == peer_rank
            ]
            for param_idx in peer_param_indices:
                chunk_size = self._param_chunk_sizes[param_idx][self._rank]
                self._param_recv_info[param_idx] = (recv_offset, chunk_size)
                recv_offset += chunk_size

        # Pre-compute recv copy-out views for batched copy in
        # redistribute_and_update_params().
        for param_idx in range(len(self._params)):
            recv_offset, chunk_size = self._param_recv_info[param_idx]
            if chunk_size > 0:
                local_param = self._params[param_idx].to_local()
                local_numel = local_param.numel()
                assert chunk_size == local_numel, (
                    f"chunk_size ({chunk_size}) != local_numel ({local_numel}) "
                    f"for param {param_idx}"
                )
                self._recv_src_views.append(
                    self._recv_buffer[recv_offset : recv_offset + chunk_size]
                )
                self._recv_dst_views.append(local_param.flatten())

    @torch.no_grad()
    def redistribute_and_update_params(
        self,
        local_full_params: list[Tensor],
    ) -> None:
        """Redistribute updated parameters using a SINGLE coalesced all_to_all.

        This combines ALL params into one collective call instead of multiple rounds,
        reducing the number of collective operations.

        Args:
            local_full_params (list[Tensor]): List of full updated params computed by this rank.
                Must match the params indicated by assigned_params_mask.
        """
        assert len(local_full_params) == len(self._local_param_indices), (
            f"Expected {len(self._local_param_indices)} local params, "
            f"got {len(local_full_params)}"
        )

        group_size = self._group_size

        # Pre-chunk all local full params by peer rank for source view construction.
        # param_chunks[local_idx][peer_rank] = flattened chunk tensor
        param_chunks: list[list[Tensor]] = []
        for local_idx in range(len(self._local_param_indices)):
            full_param = local_full_params[local_idx]
            global_shape = self._local_param_global_shapes[local_idx]

            if len(global_shape) == 0:
                # Scalar tensor - all goes to rank 0.
                # Note: FSDP2 does not produce scalar params in practice;
                # this branch is defensive.
                chunks = [full_param.flatten()]
                while len(chunks) < group_size:
                    chunks.append(
                        torch.empty(0, dtype=full_param.dtype, device=full_param.device)
                    )
            else:
                full_param_reshaped = full_param.view(global_shape)
                dim0_chunks = torch.chunk(full_param_reshaped, group_size, dim=0)
                chunks = [c.flatten() for c in dim0_chunks]
                while len(chunks) < group_size:
                    chunks.append(
                        torch.empty(0, dtype=full_param.dtype, device=full_param.device)
                    )
            param_chunks.append(chunks)

        # Build source list matching the pre-computed destination views and
        # batch-copy into the send buffer.
        src_list: list[Tensor] = []
        dst_list: list[Tensor] = []
        for dst_view, param_idx, peer_rank in zip(
            self._send_dst_views,
            self._send_param_indices,
            self._send_peer_ranks,
            strict=True,
        ):
            local_idx = self._param_to_local_idx[param_idx]
            src_list.append(param_chunks[local_idx][peer_rank])
            dst_list.append(dst_view)

        if dst_list:
            torch._foreach_copy_(dst_list, src_list)

        # Execute SINGLE all_to_all - this is the key optimization!
        dist.all_to_all(self._recv_list, self._send_list, group=self._dist_group)

        # Batch-copy received chunks to local param shards in one fused kernel.
        if self._recv_dst_views:
            torch._foreach_copy_(self._recv_dst_views, self._recv_src_views)


class GatherGradientsContext:
    """Context for gathering gradients to owning ranks using all_to_all.

    This is the mirror of RedistributeParamsContext. While RedistributeParamsContext
    sends full params from owning ranks to all ranks' local shards,
    GatherGradientsContext gathers local gradient shards from all ranks to the
    owning rank to reconstruct full gradients.

    Data flow:
        Each rank has local grad shards for ALL params (from FSDP).
        Each rank sends its local shard of param i to the rank that owns param i.
        The owning rank concatenates received shards along dim-0 to get the full gradient.

    This replaces per-param full_tensor() all-gathers with a single all_to_all,
    reducing peak memory from O(total_params) to O(total_params / world_size).

    Args:
        params (tuple[DTensor, ...]): Tuple of DTensor parameters.
        assigned_params_mask (tuple[bool, ...]): Boolean mask indicating which params this rank owns.
        dist_group (torch.distributed.ProcessGroup): Process group for communication.

    Example:
        # During initialization:
        ctx = GatherGradientsContext(params, assigned_mask, dist_group)

        # During each optimizer step:
        full_grads = ctx.gather_gradients()
    """

    @torch.no_grad()
    def __init__(
        self,
        params: tuple[DTensor, ...],
        assigned_params_mask: tuple[bool, ...],
        dist_group: torch.distributed.ProcessGroup,
    ) -> None:
        self._params = params
        self._assigned_params_mask = assigned_params_mask
        self._dist_group = dist_group
        self._group_size: int = dist_group.size()
        self._rank: int = dist.get_rank(group=dist_group)

        if not params:
            raise ValueError("params cannot be empty")

        if len(params) != len(assigned_params_mask):
            raise ValueError(
                f"len(params) ({len(params)}) != "
                f"len(assigned_params_mask) ({len(assigned_params_mask)})"
            )

        self._dtype: torch.dtype = params[0].to_local().dtype
        self._device: torch.device = params[0].to_local().device

        # Validate all params have same dtype
        if any(p.to_local().dtype != self._dtype for p in params):
            raise NotImplementedError(
                "When using gradient gathering, parameters of "
                "different dtypes are not currently supported."
            )

        self._precompute_metadata()

    def _precompute_metadata(self) -> None:
        """Precompute all static information needed for gradient gathering.

        The send/recv directions are swapped relative to RedistributeParamsContext:
        - Send: this rank sends its local grad shard of param i to the rank owning param i
        - Recv: this rank receives grad shards from all ranks for params it owns

        The send buffer layout (per peer rank) groups params by owner:
          send_to_peer_r = [local_shard_of_param_j for j owned by peer r]

        The recv buffer layout (per peer rank) groups by sender:
          recv_from_peer_r = [shard_from_rank_r_for_param_j for j owned by this rank]
        """
        group_size = self._group_size

        # Compute which params this rank owns
        self._local_param_indices: list[int] = [
            i for i, assigned in enumerate(self._assigned_params_mask) if assigned
        ]

        # Compute chunk sizes using shared utility
        self._param_chunk_sizes: list[list[int]] = _compute_param_chunk_sizes(
            self._params, group_size
        )

        # Store global shapes for assigned params (used to reconstruct full grads)
        self._param_global_shapes: list[tuple[int, ...]] = [
            tuple(p.shape) for p in self._params
        ]

        # Send sizes: for each peer rank, sum of this rank's local shard sizes
        # for params owned by that peer.
        # "This rank sends its chunk[self._rank] of param i to the rank that owns param i."
        self._send_sizes: list[int] = []
        for peer_rank in range(group_size):
            # Params owned by peer_rank
            peer_param_indices = [
                i for i in range(len(self._params)) if i % group_size == peer_rank
            ]
            send_size = sum(
                self._param_chunk_sizes[param_idx][self._rank]
                for param_idx in peer_param_indices
            )
            self._send_sizes.append(send_size)
        self._total_send_size = sum(self._send_sizes)

        # Recv sizes: for each peer rank, sum of that peer's shard sizes
        # for params owned by this rank.
        # "This rank receives chunk[peer_rank] of param j from peer_rank, for all j this rank owns."
        self._recv_sizes: list[int] = []
        for peer_rank in range(group_size):
            recv_size = sum(
                self._param_chunk_sizes[param_idx][peer_rank]
                for param_idx in self._local_param_indices
            )
            self._recv_sizes.append(recv_size)
        self._total_recv_size = sum(self._recv_sizes)

        # Pre-allocate persistent send and recv buffers.
        # Memory overhead: O(total_params / world_size) each.
        #   - send_buffer holds local grad shards for all params ≈ total_params / world_size
        #   - recv_buffer holds received grad shards for assigned (owned) params
        #     from all peers = total_params / world_size
        self._send_buffer = torch.empty(
            self._total_send_size, dtype=self._dtype, device=self._device
        )
        self._recv_buffer = torch.empty(
            self._total_recv_size, dtype=self._dtype, device=self._device
        )

        # Pre-compute send_list and recv_list as contiguous views into buffers.
        self._send_list = _build_buffer_views(self._send_buffer, self._send_sizes)
        self._recv_list = _build_buffer_views(self._recv_buffer, self._recv_sizes)

        # Precompute recv unpacking info for each assigned param.
        # After all_to_all, for assigned param j, its shards from each rank are
        # scattered across the recv buffer (one piece per peer rank section).
        # We precompute (offset, size) pairs for each (param, peer_rank) so we
        # can efficiently gather them into a full gradient.
        #
        # _param_recv_info[local_idx] = list of (offset, size) for each peer rank
        # where local_idx indexes into self._local_param_indices
        self._param_recv_info: list[list[tuple[int, int]]] = []
        # First compute offsets within each peer_rank's recv section
        # recv_offsets_per_peer[peer_rank] tracks the running offset within
        # the recv section for peer_rank
        recv_section_starts: list[int] = []
        offset = 0
        for recv_size in self._recv_sizes:
            recv_section_starts.append(offset)
            offset += recv_size

        # For each local param, compute where each peer's shard lands in recv_buffer
        # The recv section for peer_rank contains shards in local_param_indices order
        peer_offsets = list(recv_section_starts)  # mutable copy
        # We need to iterate local_param_indices in order for each peer_rank
        # to compute offsets correctly
        self._param_local_idx_map: dict[int, int] = {
            param_idx: local_idx
            for local_idx, param_idx in enumerate(self._local_param_indices)
        }
        # Initialize recv info for each local param
        for _ in self._local_param_indices:
            self._param_recv_info.append([(-1, -1)] * group_size)

        for peer_rank in range(group_size):
            for param_idx in self._local_param_indices:
                local_idx = self._param_local_idx_map[param_idx]
                chunk_size = self._param_chunk_sizes[param_idx][peer_rank]
                self._param_recv_info[local_idx][peer_rank] = (
                    peer_offsets[peer_rank],
                    chunk_size,
                )
                peer_offsets[peer_rank] += chunk_size

        # Pre-compute views for batched operations.
        # Forward-declare attributes set by helper methods for Pyre.
        self._full_grad_buffers: list[Tensor] = []
        self._unpack_src_views: list[Tensor] = []
        self._unpack_dst_views: list[Tensor] = []
        self._send_dst_views: list[Tensor] = []
        self._send_src_param_indices: list[int] = []
        self._send_src_chunk_sizes: list[int] = []
        self._compute_unpack_views()
        self._compute_send_dst_views()

        logger.info(
            f"GatherGradientsContext initialized: "
            f"rank={self._rank}, group_size={group_size}, "
            f"num_params={len(self._params)}, "
            f"local_params={len(self._local_param_indices)}, "
            f"send_sizes={self._send_sizes}, "
            f"recv_sizes={self._recv_sizes}"
        )

    def _compute_unpack_views(self) -> None:
        """Pre-allocate full gradient buffers and pre-compute recv unpack views."""
        for local_idx in range(len(self._local_param_indices)):
            total_size = sum(
                size for _, size in self._param_recv_info[local_idx] if size > 0
            )
            grad_buffer = torch.empty(
                total_size, dtype=self._dtype, device=self._device
            )
            self._full_grad_buffers.append(grad_buffer)
            dst_offset = 0
            for recv_offset, size in self._param_recv_info[local_idx]:
                if size > 0:
                    self._unpack_src_views.append(
                        self._recv_buffer[recv_offset : recv_offset + size]
                    )
                    self._unpack_dst_views.append(
                        grad_buffer[dst_offset : dst_offset + size]
                    )
                    dst_offset += size

    def _compute_send_dst_views(self) -> None:
        """Pre-compute destination views into the send buffer for batched copy."""
        group_size = self._group_size
        send_offset = 0
        for peer_rank in range(group_size):
            for param_idx in range(len(self._params)):
                if param_idx % group_size != peer_rank:
                    continue
                chunk_size = self._param_chunk_sizes[param_idx][self._rank]
                if chunk_size > 0:
                    self._send_dst_views.append(
                        self._send_buffer[send_offset : send_offset + chunk_size]
                    )
                    self._send_src_param_indices.append(param_idx)
                    self._send_src_chunk_sizes.append(chunk_size)
                    send_offset += chunk_size

    def _build_send_list(self, has_grad: list[bool]) -> None:
        """Pack local grad shards into the persistent send buffer.

        For each pre-computed (param_idx, dst_view) pair, copies the local
        gradient shard into the send buffer using foreach_copy_ for efficiency.

        Args:
            has_grad: Boolean mask indicating which params have gradients.
        """
        src_list: list[Tensor] = []
        dst_list: list[Tensor] = []

        for dst, param_idx, chunk_size in zip(
            self._send_dst_views,
            self._send_src_param_indices,
            self._send_src_chunk_sizes,
            strict=True,
        ):
            if has_grad[param_idx]:
                local_grad = self._params[param_idx].grad
                assert local_grad is not None
                src_list.append(
                    local_grad.to_local().flatten()[:chunk_size]  # type: ignore
                )
                dst_list.append(dst)
            else:
                dst.zero_()

        if dst_list:
            torch._foreach_copy_(dst_list, src_list)

    def _unpack_recv_buffer(self, has_grad: list[bool]) -> list[Tensor | None]:
        """Unpack the recv buffer into full gradients for assigned params.

        Uses pre-computed views and foreach_copy_ to batch-copy all shards from
        the recv buffer into pre-allocated full gradient buffers, then reshapes
        each to the original global shape.

        Args:
            has_grad (list[bool]): Boolean mask indicating which params have gradients.

        Returns:
            full_grads (list[Tensor | None]): List of full gradients (same length as self._params).
                None for params without gradients or unassigned params.
        """
        # Batch-copy all recv shards into pre-allocated full grad buffers.
        if self._unpack_dst_views:
            torch._foreach_copy_(self._unpack_dst_views, self._unpack_src_views)

        full_grads: list[Tensor | None] = [None for _ in self._params]
        for local_idx, param_idx in enumerate(self._local_param_indices):
            if not has_grad[param_idx]:
                continue
            global_shape = self._param_global_shapes[param_idx]
            if len(global_shape) == 0:
                full_grads[param_idx] = self._full_grad_buffers[local_idx].squeeze()
            else:
                full_grads[param_idx] = self._full_grad_buffers[local_idx].view(
                    global_shape
                )

        return full_grads

    def _execute_all_to_all(self) -> None:
        """Execute all_to_all using persistent send/recv buffers."""
        dist.all_to_all(self._recv_list, self._send_list, group=self._dist_group)

    @torch.no_grad()
    def gather_gradients(self) -> list[Tensor | None]:
        """Gather gradients from all ranks using a single all_to_all.

        Each rank sends its local grad shard of each param to the rank that owns
        that param. The owning rank then concatenates the received shards along
        dim-0 to reconstruct the full gradient.

        Returns:
            full_grads: List of full gradients for ALL params (same length as
                self._params). Entries are None for params not assigned to this
                rank, regardless of whether they have gradients. Only params
                assigned to this rank will have non-None values.
        """
        has_grad = [p.grad is not None for p in self._params]

        self._build_send_list(has_grad)
        self._execute_all_to_all()

        result = self._unpack_recv_buffer(has_grad)
        # Clone to detach from _full_grad_buffers, which will be overwritten
        # by subsequent gather_gradients() calls.
        return [t.clone() if t is not None else None for t in result]

    @torch.no_grad()
    def gather_params(self) -> list[Tensor | None]:
        """Gather full parameter values from all ranks using a single all_to_all.

        Similar to gather_gradients(), but operates on parameter values instead
        of gradients. Every parameter always has a value, so no None handling
        is needed on the send side.

        Returns:
            full_params: List of full parameter tensors (same length as
                self._params). Entries are None for params not assigned to this
                rank. Only params assigned to this rank will have non-None values.
        """
        # Pack param values into persistent send buffer using foreach_copy_.
        src_list: list[Tensor] = []
        dst_list: list[Tensor] = []

        for dst, param_idx, chunk_size in zip(
            self._send_dst_views,
            self._send_src_param_indices,
            self._send_src_chunk_sizes,
            strict=True,
        ):
            src_list.append(self._params[param_idx].to_local().flatten()[:chunk_size])
            dst_list.append(dst)

        if dst_list:
            torch._foreach_copy_(dst_list, src_list)

        self._execute_all_to_all()

        result = self._unpack_recv_buffer(has_grad=[True] * len(self._params))
        # Clone to detach from _full_grad_buffers, which will be overwritten
        # by subsequent gather_gradients() calls.
        return [t.clone() if t is not None else None for t in result]


def _device_key(device: torch.device) -> tuple[str, int]:
    """Canonicalizes a torch.device into a hashable (type, index) tuple.

    Resolves cuda devices without an explicit index to the current device, so
    torch.device('cuda') and torch.device('cuda:0') normalize to the same key
    and don't produce duplicate cache entries in `_get_triu_indices`.
    """
    if device.type == "cuda" and device.index is None:
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index or 0)


@cache
def _get_triu_indices(d: int, device_type: str, device_index: int) -> Tensor:
    """Returns cached upper-triangular index pairs as a (2, d*(d+1)/2) int32 tensor.

    Args:
        d (int): Dimension of the square matrix.
        device_type (str): Device type, e.g. "cpu" or "cuda".
        device_index (int): Resolved device index. Use `_device_key` to
            normalize from a torch.device.

    Returns:
        indices (Tensor): A (2, d*(d+1)/2) int32 tensor of (row, col) index
            pairs for the upper triangle in row-major order. The returned
            tensor is cached by reference and MUST NOT be mutated by callers.

    Notes:
        Cached process-wide via functools.cache. int32 is sufficient since
        matrix dim d is bounded by max_preconditioner_dim (default 1024, up
        to 4096 in known production workloads), well under int32's 2**31
        range; using int32 instead of torch.triu_indices' int64 default
        halves the cache footprint.
    """
    return torch.triu_indices(d, d, device=torch.device(device_type, device_index)).to(
        torch.int32
    )


def pack_upper_triangular(matrix: Tensor) -> Tensor:
    """Packs the upper triangle of a symmetric matrix into a flat 1D tensor.

    For a (d, d) symmetric matrix, extracts the upper triangular elements
    (including diagonal) into a contiguous 1D tensor of size d*(d+1)/2.

    Args:
        matrix (Tensor): A 2D symmetric matrix of shape (d, d).

    Returns:
        packed (Tensor): A 1D tensor of size d*(d+1)/2 containing the upper
            triangular elements in row-major order.
    """
    rows, cols = _get_triu_indices(matrix.shape[0], *_device_key(matrix.device))
    return matrix[rows, cols].contiguous()


def unpack_upper_triangular(packed: Tensor, dim: int) -> Tensor:
    """Reconstructs a symmetric matrix from its packed upper triangle.

    Args:
        packed (Tensor): A 1D tensor of size dim*(dim+1)/2 containing the
            upper triangular elements in row-major order.
        dim (int): The dimension of the original square matrix.

    Returns:
        matrix (Tensor): A 2D symmetric matrix of shape (dim, dim).
    """
    matrix = torch.zeros(dim, dim, dtype=packed.dtype, device=packed.device)
    rows, cols = _get_triu_indices(dim, *_device_key(packed.device))
    matrix[rows, cols] = packed  # Fill upper triangle.
    matrix[cols, rows] = packed  # Mirror to lower triangle (swap row/col indices).
    return matrix


_T = TypeVar("_T")


def greedy_bin_pack(
    items: Sequence[_T],
    num_bins: int,
    cost_fn: Callable[[_T], int],
) -> tuple[list[list[_T]], list[int]]:
    """Partition items into bins using greedy LPT (Longest Processing Time) bin-packing.

    Sorts items by cost (largest first) and assigns each to the bin with the
    smallest total cost. This balances total cost across bins.

    Args:
        items (Sequence[_T]): Items to partition.
        num_bins (int): Number of bins.
        cost_fn (Callable[[_T], int]): Function returning the cost of an item.

    Returns:
        A tuple of (bins, bin_costs) where bins[i] is the list of items in bin i
        and bin_costs[i] is the total cost of bin i. Empty bins are included.
    """
    bins: list[list[_T]] = [[] for _ in range(num_bins)]
    heap: list[tuple[int, int]] = [(0, i) for i in range(num_bins)]
    heapq.heapify(heap)
    # Use enumerate as tiebreaker to avoid comparing items that may not
    # support < (e.g., torch.nn.Parameter).
    for cost, _, item in sorted(
        ((cost_fn(item), i, item) for i, item in enumerate(items)),
        reverse=True,
    ):
        min_cost, min_idx = heapq.heappop(heap)
        bins[min_idx].append(item)
        heapq.heappush(heap, (min_cost + cost, min_idx))
    bin_costs = [0] * num_bins
    for cost, idx in heap:
        bin_costs[idx] = cost
    return bins, bin_costs


def split_param_groups(
    param_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split each FSDP/HSDP lossless param group with ``num_sub_groups > 1``
    into N independent sub-groups, each with its own copy of the distributed
    config (and thus its own distributor + NCCL communicator).

    Sub-groups can then run on dedicated CUDA streams in parallel threads
    (see ``DistributedShampoo._run_threaded_step``) to overlap compute and
    communication.

    Within a qualifying group, params are distributed across sub-groups via
    greedy bin-packing on numel to balance element count.

    For the ``ROUND_ROBIN`` assignment strategy, a per-parameter owner-rank map is
    computed here via greedy LPT bin-packing on per-param cost and attached to each
    resulting group under the ``OWNER_RANKS`` key:

    * Non-split path (``num_sub_groups == 1``): a single GLOBAL map over the full
      parameter set, so per-owner-rank byte load is balanced across the whole model.
    * Split path (``num_sub_groups > 1``): a PER-BUCKET map (Option C, see
      ``_compute_per_bucket_owner_ranks``) that assigns owners within each sub-group
      while carrying accumulated per-rank load across buckets. This keeps global byte
      balance AND gives every shard rank coverage within each sub-group, avoiding the
      residue-class collapse that slicing one global map per bucket would cause at
      ``num_sub_groups > 1`` (which leaves most shard ranks empty per sub-group).

    A map is always produced (even at ``num_sub_groups == 1``) so the single
    distributor still gets one.

    Currently scoped to ``FullyShardDistributedConfig`` (and its subclass
    ``HybridShardDistributedConfig``); other distributor types pass through
    unchanged.

    Args:
        param_groups (list[dict[str, Any]]): Optimizer param groups to split.

    Returns:
        new_param_groups (list[dict[str, Any]]): New param groups list. Groups
            that don't opt into splitting pass through (with an ``OWNER_RANKS``
            map attached for ROUND_ROBIN); qualifying groups are replaced by N
            sub-groups with the params evenly distributed and each carrying its
            own per-bucket owner map (Option C: per-bucket LPT with carried load).

    TODO(irisz): Support automatic determination of optimal num_sub_groups
    based on param count, compute cost, and communicator memory overhead.
    """
    new_param_groups: list[dict[str, Any]] = []
    for group in param_groups:
        distributed_config = group[DISTRIBUTED_CONFIG]
        # ROUND_ROBIN is load-balanced via greedy LPT. The non-split branch below
        # attaches a single GLOBAL (all-params) owner map; the split branch attaches
        # a PER-BUCKET map with carried load (Option C). Both balance per-owner-rank
        # byte load across the whole model. (Positional ``i % shard`` is just this
        # LPT with a uniform cost model; the default AlignedMemoryCostModel makes it
        # size-balanced.) None for REPLICATE / non-FSDP configs.
        is_round_robin: bool = (
            isinstance(distributed_config, FullyShardDistributedConfig)
            and distributed_config.param_assignment_strategy
            == FSDPParamAssignmentStrategy.ROUND_ROBIN
        )
        if not _should_split(distributed_config):
            # No sub-group split: attach the global owner map for ROUND_ROBIN so
            # the single distributor consumes it; otherwise pass through unchanged.
            if is_round_robin:
                new_param_groups.append(
                    {
                        **group,
                        OWNER_RANKS: _compute_global_owner_ranks(
                            group[PARAMS], distributed_config
                        ),
                    }
                )
            else:
                new_param_groups.append(group)
            continue
        # Validate BEFORE computing the owner map so a bad num_sub_groups raises
        # cheaply (and without needing a process group for the world size).
        _validate_num_sub_groups(distributed_config, len(group[PARAMS]))
        # Option C: assign owners PER BUCKET with per-rank load carried across
        # buckets (inside _create_sub_groups), instead of slicing one global owner
        # map. Global-map-then-slice collapsed each sub-group's owners onto a
        # residue class mod num_sub_groups (leaving shard ranks empty per sub-group
        # -> the "Some workers have no parameters" path); per-bucket-with-carry
        # keeps global per-rank balance AND covers all shard ranks in every
        # sub-group.
        sub_groups = _create_sub_groups(
            group, distributed_config.num_sub_groups, is_round_robin=is_round_robin
        )
        new_param_groups.extend(sub_groups)
        logger.info(
            "Split param group (%d params) into %d sub-groups (numel per group: %s).",
            len(group[PARAMS]),
            len(sub_groups),
            [sum(p.numel() for p in g[PARAMS]) for g in sub_groups],
        )
    return new_param_groups


def _compute_global_owner_ranks(
    params: Sequence[Tensor],
    distributed_config: FullyShardDistributedConfig,
) -> list[int]:
    """Compute a per-parameter owner-rank map balanced GLOBALLY across all params.

    Uses the DDP distributor's LPT primitive (``distribute_buffer_sizes``) over the
    FULL parameter set, greedily assigning each param to the currently least-loaded
    owner rank by cost (aligned bytes by default). Returns a list aligned to
    ``params``: entry ``i`` is the owner rank of param ``i``.

    The owner space is the shard dimension: ``device_mesh.size(1)`` for HSDP,
    else the world size for FSDP. ``distribute_buffer_sizes`` is deterministic, so
    the map is identical on every rank and stable across checkpoints for a fixed
    topology (shard size) and cost config -- changing the shard size or the
    ``load_balancing_config`` cost model recomputes a different ownership map.
    """
    shard_size = (
        distributed_config.device_mesh.size(1)
        if isinstance(distributed_config, HybridShardDistributedConfig)
        else dist.get_world_size()
    )
    buffer_size_ranks = distribute_buffer_sizes(
        blocked_params=tuple(params),
        group_size=shard_size,
        load_balancing_config=distributed_config.load_balancing_config,
    )
    return [rank for _, rank in buffer_size_ranks]


def _should_split(
    distributed_config: Any,
) -> TypeGuard[FullyShardDistributedConfig]:
    """True when the config opts into sub-group splitting (num_sub_groups > 1)."""
    return (
        isinstance(distributed_config, FullyShardDistributedConfig)
        and distributed_config.num_sub_groups > 1
    )


def _validate_num_sub_groups(
    distributed_config: FullyShardDistributedConfig, num_params: int
) -> None:
    """Raise ValueError if num_sub_groups is too large for this group's params.

    This is a SOFT efficiency guard, not a correctness requirement. It keeps each
    sub-group large enough (>= ``shard_size`` params) that every shard rank tends
    to own at least one param, avoiding degenerate tiny sub-groups that still spin
    up a full per-sub-group distributor + NCCL communicator + CUDA stream for
    little work. A shard rank that ends up owning zero params in a sub-group does
    NOT deadlock: the lossless all_to_all pads zero-owner ranks with empty tensors
    so they still participate (see ``shampoo_fully_shard_utils.py`` --
    ``prepare_update_param_buffers`` pads the buffer offsets, and
    ``redistribute_and_update_params`` sends a zero-size tensor for a missing
    owner). For FSDP this collapses to ``shard_size=1`` (i.e., n <= num_params).
    """
    n = distributed_config.num_sub_groups
    shard_size = (
        distributed_config.device_mesh.size(1)
        if isinstance(distributed_config, HybridShardDistributedConfig)
        else 1
    )
    max_groups = max(1, num_params // shard_size)
    if n > max_groups:
        raise ValueError(
            f"num_sub_groups={n} is too large for {num_params} params with "
            f"shard_group_size={shard_size}: each sub-group should hold at least "
            f"{shard_size} params to avoid degenerate tiny sub-groups (an "
            f"efficiency guard, not a correctness/deadlock requirement -- "
            f"zero-owner shard ranks are handled by all_to_all padding). "
            f"Maximum num_sub_groups={max_groups}."
        )


def _create_sub_groups(
    group: dict[str, Any],
    n: int,
    is_round_robin: bool = False,
) -> list[dict[str, Any]]:
    """Split ``group``'s params into N sub-groups via greedy bin-packing on
    numel; each carries a fresh copy of the distributed config so each gets
    its own distributor and NCCL communicator.

    Empty bins are dropped (can occur when one param's numel dominates the
    total enough to leave a bin unfilled).

    For HSDP, additionally applies the same SOFT efficiency guard as
    ``_validate_num_sub_groups``: every non-empty bin should hold at least
    ``shard_size`` params. Greedy LPT can place a single dominant-numel param
    alone in a bin, leaving fewer than ``shard_size`` params there even though
    ``_validate_num_sub_groups`` (which assumes contiguous chunking) accepted the
    count. This is NOT a deadlock condition: a shard rank owning zero params in a
    bin still participates in the all_to_all via padding (see
    ``shampoo_fully_shard_utils.py``); the guard just rejects degenerate tiny
    sub-groups whose per-sub-group communicator/stream overhead isn't worth it.

    Option C: for ROUND_ROBIN (``is_round_robin``), owner ranks are assigned PER
    BUCKET via ``_compute_per_bucket_owner_ranks`` -- LPT over each bucket's params
    across all shard ranks, with per-rank load carried across buckets. This
    replaces the global-map-then-slice, which collapsed each sub-group's owners
    onto a residue class (leaving shard ranks empty per sub-group). Per-bucket
    assignment aims to cover all shard ranks per sub-group (coverage is best-effort
    -- a heavily skewed carried load can still leave a rank without an owner,
    handled by all_to_all padding) while the carried load preserves global per-rank
    balance (the largest param of every bucket is NOT re-stacked onto rank 0).
    """
    params = group[PARAMS]
    distributed_config = group[DISTRIBUTED_CONFIG]
    base_keys = {k: v for k, v in group.items() if k not in (PARAMS, OWNER_RANKS)}
    # Bin-pack over indices (not the param objects). Using the index as the item
    # reproduces the previous params-as-items packing exactly (the sort key's
    # unique enumerate tiebreaker means the item is never compared).
    indices = list(range(len(params)))
    buckets, _ = greedy_bin_pack(
        items=indices, num_bins=n, cost_fn=lambda i: params[i].numel()
    )
    if isinstance(distributed_config, HybridShardDistributedConfig):
        shard_size = distributed_config.device_mesh.size(1)
        for i, bucket in enumerate(buckets):
            if bucket and len(bucket) < shard_size:
                raise ValueError(
                    f"HSDP sub-group {i} got {len(bucket)} params after "
                    f"greedy bin-packing, fewer than shard_size={shard_size}. "
                    f"This happens when one or more params dominate by numel, "
                    f"leaving a degenerate tiny sub-group. This is an efficiency "
                    f"guard, not a correctness/deadlock requirement -- shard ranks "
                    f"owning zero params still participate via all_to_all padding. "
                    f"Reduce num_sub_groups or rebalance the param shapes."
                )
    # Option C: per-bucket owner map (aligned to each bucket's param order).
    per_bucket_owner_ranks: list[list[int]] | None = (
        _compute_per_bucket_owner_ranks(buckets, params, distributed_config)
        if is_round_robin
        else None
    )
    sub_groups: list[dict[str, Any]] = []
    for b, bucket in enumerate(buckets):
        if not bucket:
            continue
        new_group: dict[str, Any] = {
            **base_keys,
            PARAMS: [params[i] for i in bucket],
            DISTRIBUTED_CONFIG: copy.copy(distributed_config),
        }
        if per_bucket_owner_ranks is not None:
            new_group[OWNER_RANKS] = per_bucket_owner_ranks[b]
        sub_groups.append(new_group)
    return sub_groups


def _compute_per_bucket_owner_ranks(
    buckets: list[list[int]],
    params: Sequence[Tensor],
    distributed_config: FullyShardDistributedConfig,
) -> list[list[int]]:
    """Option C: per-bucket owner assignment with carried load.

    For each bucket (processed in order), greedily assign its params to the
    currently least-loaded shard rank via LPT -- but SEED the per-rank load with
    the accumulated load from all prior buckets (the "carry"). This aims to spread
    owners across shard ranks; note coverage is best-effort -- a heavily skewed
    carried load from prior buckets can still leave a rank without an owner in a
    given bucket (padding in the all_to_all handles zero-owner ranks). The carry
    keeps the global per-rank byte load balanced (the largest param of every
    bucket is not re-stacked onto rank 0). Mirrors ``distribute_buffer_sizes`` per
    bucket (same cost model, sort order, and heap tiebreak) so at
    ``num_sub_groups == 1`` it is identical to the global map.

    Returns a list aligned to ``buckets``: entry ``b`` is a list of owner ranks
    aligned to ``buckets[b]`` (owner of the p-th param of bucket b).
    """
    shard_size = (
        distributed_config.device_mesh.size(1)
        if isinstance(distributed_config, HybridShardDistributedConfig)
        else dist.get_world_size()
    )
    cost_model = distributed_config.load_balancing_config.cost_model
    carried_load: list[float] = [0.0] * shard_size
    per_bucket_owner_ranks: list[list[int]] = []
    for bucket in buckets:
        owners: list[int] = [-1] * len(bucket)
        # Min-heap seeded with the load carried from prior buckets so this bucket
        # fills the globally-lightest ranks first (keeps the global map balanced).
        heap: list[tuple[float, int]] = [
            (carried_load[rank], rank) for rank in range(shard_size)
        ]
        heapq.heapify(heap)
        # Assign this bucket's params in cost-descending order (LPT), matching
        # distribute_buffer_sizes' ordering and tiebreaks.
        costs = [
            (float(cost_model.cost(params[i])), pos) for pos, i in enumerate(bucket)
        ]
        for cost, pos in sorted(costs, key=operator.itemgetter(0), reverse=True):
            load, rank = heapq.heappop(heap)
            heapq.heappush(heap, (load + cost, rank))
            owners[pos] = rank
        # Persist this bucket's resulting per-rank load as the carry for the next.
        for load, rank in heap:
            carried_load[rank] = load
        per_bucket_owner_ranks.append(owners)
    return per_bucket_owner_ranks
