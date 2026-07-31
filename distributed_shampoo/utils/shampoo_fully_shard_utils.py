"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import accumulate

import torch
from torch import distributed as dist, Tensor
from torch.distributed.tensor import DTensor

logger: logging.Logger = logging.getLogger(__name__)


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


# NOTE: Plans group parallel per-entry lists; the lists themselves stay flat
# so the hot path passes them directly to torch._foreach_copy_.


@dataclass
class _RedistributeSendPlan:
    """Per-entry send metadata for RedistributeParamsContext.

    Source data comes from the caller (local_full_params); only destination
    views and routing metadata are stored here.
    """

    param_indices: list[int] = field(default_factory=list)
    peer_ranks: list[int] = field(default_factory=list)
    dst_views: list[Tensor] = field(default_factory=list)


@dataclass
class _RedistributeRecvPlan:
    """Per-entry recv metadata for RedistributeParamsContext.

    Source views point into the recv buffer; destination views point into the
    local param shards. Iteration is src -> dst.
    """

    src_views: list[Tensor] = field(default_factory=list)
    dst_views: list[Tensor] = field(default_factory=list)


@dataclass
class _GatherSendPlan:
    """Per-entry send metadata for GatherGradientsContext.

    Source data comes from a callable (gradient or param value); only the
    routing metadata (param indices, chunk sizes) and destination send-buffer
    views are stored here.
    """

    param_indices: list[int] = field(default_factory=list)
    chunk_sizes: list[int] = field(default_factory=list)
    dst_views: list[Tensor] = field(default_factory=list)


@dataclass
class _GatherUnpackPlan:
    """Per-entry recv-buffer-unpack metadata for GatherGradientsContext.

    Source views point into the recv buffer; destination views point into the
    pre-allocated full-grad buffers. Iteration is src -> dst.
    """

    src_views: list[Tensor] = field(default_factory=list)
    dst_views: list[Tensor] = field(default_factory=list)


class AllToAllContext(ABC):
    """Base class for coalesced all_to_all communication contexts.

    Provides shared infrastructure for precomputing FSDP dim-0 sharding metadata,
    allocating persistent send/recv buffers, and executing a single all_to_all
    collective. Subclasses implement the direction-specific send/recv size
    computation and additional metadata precomputation.

    TODO(irisz): Currently assumes dim-0 sharding only. FSDP2 supports non-dim-0 sharding
    but it is not enabled on APS. Extend to support arbitrary shard dimensions if needed.

    Args:
        params (tuple[DTensor, ...]): Tuple of DTensor parameters.
        assigned_params_mask (tuple[bool, ...]): Boolean mask indicating which params this rank owns.
        dist_group (torch.distributed.ProcessGroup): Process group for communication.
        owner_ranks (list[int] | None): Per-param owner-rank map; entry ``i`` is the rank that owns
            param ``i``. This is the single source of truth for ownership at every send/recv site.
            The lossless distributors pass the load-balanced global map here (see OWNER_RANKS). When
            ``None`` (default), it falls back to a positional ``[i % group_size for i in range(len(params))]``
            map -- convenient for direct construction (e.g. unit tests) that supply a positional
            ``assigned_params_mask``. Must be consistent with ``assigned_params_mask`` (i.e.
            ``owner_ranks[i] == this_rank`` iff ``assigned_params_mask[i]``) and identical across all
            ranks in the group.
    """

    @torch.no_grad()
    def __init__(
        self,
        params: tuple[DTensor, ...],
        assigned_params_mask: tuple[bool, ...],
        dist_group: torch.distributed.ProcessGroup,
        owner_ranks: list[int] | None = None,
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

        # Single source of truth for ownership across all send/recv sites. None ->
        # positional (i % group_size) default for direct construction; the lossless
        # distributors always pass the load-balanced global map explicitly.
        self._owner_ranks: list[int] = (
            list(owner_ranks)
            if owner_ranks is not None
            else [i % self._group_size for i in range(len(params))]
        )
        if len(self._owner_ranks) != len(params):
            raise ValueError(
                f"len(owner_ranks) ({len(self._owner_ranks)}) != "
                f"len(params) ({len(params)})"
            )

        self._dtype: torch.dtype = params[0].to_local().dtype
        self._device: torch.device = params[0].to_local().device

        # Validate all params have same dtype
        if any(p.to_local().dtype != self._dtype for p in params):
            raise NotImplementedError(
                "Parameters of different dtypes are not currently supported."
            )

        self._precompute_metadata()

    def _precompute_metadata(self) -> None:
        """Precompute shared metadata for all_to_all communication."""
        group_size = self._group_size

        # Compute which params this rank owns (based on round-robin assignment)
        self._local_param_indices: list[int] = [
            i for i, assigned in enumerate(self._assigned_params_mask) if assigned
        ]

        # Compute chunk sizes using FSDP's dim-0 sharding semantics
        self._param_chunk_sizes: list[list[int]] = _compute_param_chunk_sizes(
            self._params, group_size
        )

        # Compute direction-specific send/recv sizes
        self._send_sizes, self._recv_sizes = self._compute_send_recv_sizes()
        self._total_send_size = sum(self._send_sizes)
        self._total_recv_size = sum(self._recv_sizes)

        # Pre-allocate persistent send and recv buffers for all_to_all.
        self._send_buffer = torch.empty(
            self._total_send_size, dtype=self._dtype, device=self._device
        )
        self._recv_buffer = torch.empty(
            self._total_recv_size, dtype=self._dtype, device=self._device
        )

        # Pre-compute send_list and recv_list as contiguous views into buffers.
        self._send_list = _build_buffer_views(self._send_buffer, self._send_sizes)
        self._recv_list = _build_buffer_views(self._recv_buffer, self._recv_sizes)

        # Delegate subclass-specific precomputation.
        self._precompute_subclass_metadata()

    @abstractmethod
    def _compute_send_recv_sizes(self) -> tuple[list[int], list[int]]:
        """Compute per-peer send and recv sizes. Subclasses swap the direction."""
        ...

    @abstractmethod
    def _precompute_subclass_metadata(self) -> None:
        """Hook for subclass-specific metadata precomputation."""
        ...

    def _execute_all_to_all(self) -> None:
        """Execute all_to_all using persistent send/recv buffers."""
        dist.all_to_all(self._recv_list, self._send_list, group=self._dist_group)


class RedistributeParamsContext(AllToAllContext):
    """Context for optimized parameter redistribution using all_to_all.

    Sends full params from owning ranks to all ranks' local shards.
    Precomputes all static information at init, then uses a single all_to_all
    collective per step instead of multiple point-to-point calls.

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
        owner_ranks: list[int] | None = None,
    ) -> None:
        # Forward-declare subclass-specific attributes for Pyre before
        # super().__init__() calls _precompute_subclass_metadata().
        self._param_local_numels: list[int] = []
        self._param_global_numels: list[int] = []
        self._param_to_local_idx: dict[int, int] = {}
        self._local_param_global_shapes: list[tuple[int, ...]] = []
        self._param_recv_info: list[tuple[int, int]] = []
        self._send_plan = _RedistributeSendPlan()
        self._recv_plan = _RedistributeRecvPlan()
        super().__init__(params, assigned_params_mask, dist_group, owner_ranks)

    def _compute_send_recv_sizes(self) -> tuple[list[int], list[int]]:
        """Owner-to-all: send owned params' chunks to each peer, recv each peer's chunks."""
        group_size = self._group_size

        # Send sizes: sum of chunk sizes for owned params going to each peer
        send_sizes: list[int] = []
        for peer_rank in range(group_size):
            send_size = sum(
                self._param_chunk_sizes[param_idx][peer_rank]
                for param_idx in self._local_param_indices
            )
            send_sizes.append(send_size)

        # Recv sizes: sum of chunk sizes for params owned by each peer
        recv_sizes: list[int] = []
        for peer_rank in range(group_size):
            peer_param_indices = [
                i for i in range(len(self._params)) if self._owner_ranks[i] == peer_rank
            ]
            recv_size = sum(
                self._param_chunk_sizes[param_idx][self._rank]
                for param_idx in peer_param_indices
            )
            recv_sizes.append(recv_size)

        return send_sizes, recv_sizes

    def _precompute_subclass_metadata(self) -> None:
        """Precompute redistribute-specific views and metadata."""
        self._param_local_numels = [p.to_local().numel() for p in self._params]
        self._param_global_numels = [p.numel() for p in self._params]
        self._compute_send_views()
        self._compute_recv_views()

        logger.info(
            f"RedistributeParamsContext initialized: "
            f"rank={self._rank}, group_size={self._group_size}, "
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
                    self._send_plan.param_indices.append(param_idx)
                    self._send_plan.peer_ranks.append(peer_rank)
                    self._send_plan.dst_views.append(
                        self._send_buffer[send_offset : send_offset + chunk_size]
                    )
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
                i for i in range(len(self._params)) if self._owner_ranks[i] == peer_rank
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
                self._recv_plan.src_views.append(
                    self._recv_buffer[recv_offset : recv_offset + chunk_size]
                )
                self._recv_plan.dst_views.append(local_param.flatten())

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
        for param_idx, peer_rank, dst_view in zip(
            self._send_plan.param_indices,
            self._send_plan.peer_ranks,
            self._send_plan.dst_views,
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
        if self._recv_plan.dst_views:
            torch._foreach_copy_(self._recv_plan.dst_views, self._recv_plan.src_views)


class GatherGradientsContext(AllToAllContext):
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
        owner_ranks: list[int] | None = None,
    ) -> None:
        # Forward-declare subclass-specific attributes for Pyre before
        # super().__init__() calls _precompute_subclass_metadata().
        self._param_global_shapes: list[tuple[int, ...]] = []
        self._param_recv_info: list[list[tuple[int, int]]] = []
        self._param_local_idx_map: dict[int, int] = {}
        self._full_grad_buffers: list[Tensor] = []
        self._send_plan = _GatherSendPlan()
        self._unpack_plan = _GatherUnpackPlan()
        super().__init__(params, assigned_params_mask, dist_group, owner_ranks)

    def _compute_send_recv_sizes(self) -> tuple[list[int], list[int]]:
        """All-to-owner: send this rank's chunks to each owner, recv all peers' chunks for owned params."""
        group_size = self._group_size

        # Send sizes: for each peer rank, sum of this rank's local shard sizes
        # for params owned by that peer.
        send_sizes: list[int] = []
        for peer_rank in range(group_size):
            peer_param_indices = [
                i for i in range(len(self._params)) if self._owner_ranks[i] == peer_rank
            ]
            send_size = sum(
                self._param_chunk_sizes[param_idx][self._rank]
                for param_idx in peer_param_indices
            )
            send_sizes.append(send_size)

        # Recv sizes: for each peer rank, sum of that peer's shard sizes
        # for params owned by this rank.
        recv_sizes: list[int] = []
        for peer_rank in range(group_size):
            recv_size = sum(
                self._param_chunk_sizes[param_idx][peer_rank]
                for param_idx in self._local_param_indices
            )
            recv_sizes.append(recv_size)

        return send_sizes, recv_sizes

    def _precompute_subclass_metadata(self) -> None:
        """Precompute gather-specific views and metadata."""
        group_size = self._group_size

        # Store global shapes for all params (used to reconstruct full grads)
        self._param_global_shapes = [tuple(p.shape) for p in self._params]

        # Precompute recv unpacking info for each assigned param.
        # _param_recv_info[local_idx] = list of (offset, size) for each peer rank
        recv_section_starts: list[int] = []
        offset = 0
        for recv_size in self._recv_sizes:
            recv_section_starts.append(offset)
            offset += recv_size

        peer_offsets = list(recv_section_starts)  # mutable copy
        self._param_local_idx_map = {
            param_idx: local_idx
            for local_idx, param_idx in enumerate(self._local_param_indices)
        }
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
                    self._unpack_plan.src_views.append(
                        self._recv_buffer[recv_offset : recv_offset + size]
                    )
                    self._unpack_plan.dst_views.append(
                        grad_buffer[dst_offset : dst_offset + size]
                    )
                    dst_offset += size

    def _compute_send_dst_views(self) -> None:
        """Pre-compute destination views into the send buffer for batched copy."""
        group_size = self._group_size
        send_offset = 0
        for peer_rank in range(group_size):
            for param_idx in range(len(self._params)):
                if self._owner_ranks[param_idx] != peer_rank:
                    continue
                chunk_size = self._param_chunk_sizes[param_idx][self._rank]
                if chunk_size > 0:
                    self._send_plan.param_indices.append(param_idx)
                    self._send_plan.chunk_sizes.append(chunk_size)
                    self._send_plan.dst_views.append(
                        self._send_buffer[send_offset : send_offset + chunk_size]
                    )
                    send_offset += chunk_size

    def _pack_send_buffer(self, data_source: Callable[[int], Tensor | None]) -> None:
        """Pack data into the persistent send buffer using foreach_copy_.

        Args:
            data_source (Callable[[int], Tensor | None]): Callable that takes a param index
                and returns the local tensor to send (flattened and sliced to chunk_size),
                or None to zero the destination region.
        """
        src_list: list[Tensor] = []
        dst_list: list[Tensor] = []

        for param_idx, chunk_size, dst in zip(
            self._send_plan.param_indices,
            self._send_plan.chunk_sizes,
            self._send_plan.dst_views,
            strict=True,
        ):
            data = data_source(param_idx)
            if data is not None:
                src_list.append(data[:chunk_size])
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
        if self._unpack_plan.dst_views:
            torch._foreach_copy_(
                self._unpack_plan.dst_views, self._unpack_plan.src_views
            )

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
        has_grad: list[bool] = [p.grad is not None for p in self._params]

        def _grad_source(param_idx: int) -> Tensor | None:
            if not has_grad[param_idx]:
                return None
            local_grad = self._params[param_idx].grad
            assert local_grad is not None
            return local_grad.to_local().flatten()  # type: ignore

        self._pack_send_buffer(_grad_source)
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
        self._pack_send_buffer(
            lambda param_idx: self._params[param_idx].to_local().flatten()
        )
        self._execute_all_to_all()

        result = self._unpack_recv_buffer(has_grad=[True] * len(self._params))
        # Clone to detach from _full_grad_buffers, which will be overwritten
        # by subsequent gather_gradients() calls.
        return [t.clone() if t is not None else None for t in result]
