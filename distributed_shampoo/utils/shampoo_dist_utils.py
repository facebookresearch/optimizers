"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

try:
    # DTensor requires PyTorch 2.1 nightly build.
    import torch.distributed._tensor as dtensor
    from torch.distributed._tensor import zeros as dtensor_zeros

    # Flag that DTensor is enabled.
    ENABLE_DTENSOR = True

    # Cache for device meshes for allocating distributed tensors.
    _device_mesh_cache: Dict[str, dtensor.DeviceMesh] = {}

except ImportError:
    # If we encounter an import error, turns off DTensor.
    ENABLE_DTENSOR = False

ALIGNMENT_BYTES = (
    64  # necessary for determining buffer size, possibly hardware-dependent
)

logger: logging.Logger = logging.getLogger(__name__)

if not ENABLE_DTENSOR:
    logger.warning(
        "DTensor is not available and was not imported. Continuing with Tensor..."
    )


###### HELPER FUNCTIONS ######
def distribute_buffer_sizes(
    buffer_sizes: List[int], group_size: int
) -> List[Tuple[int, int]]:
    """Distribute given buffer sizes across ranks in a group.

    Buffer sizes will be rounded up for memory allocation. Buffers are distributed such that
    total buffer sizes of each rank are as even as possible. This is currently performed
    using a greedy algorithm. We do not currently consider computational cost
    or kernel launching overheads.

    TODO: Explore a better distribution strategy.

    Args:
        buffer_sizes (List[int]): buffer sizes
        group_size (int): the size of groups.

    Returns:
        buffer_size_ranks (List[Tuple[int, int]]): a list of pairs of buffer size and an assigned rank.

    Example:
        buffer_sizes = [128, 64, 500, 256], group_size = 2
        -> buffer_size_ranks = [(128, 1), (64, 1), (512, 0), (256, 1)]

    """

    # Allocate them greedily (note: Python's "sorted" is stable)
    buffer_size_ranks = [(-1, -1)] * len(buffer_sizes)
    buffer_size_sums = [0] * group_size
    for index, buffer_size in sorted(
        enumerate(buffer_sizes),
        key=lambda t: t[1],
        reverse=True,
    ):
        # computes smallest multiple of ALIGNMENT_BYTES that is >= buffer size
        aligned_buffer_size = (
            (buffer_size + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES
        )
        rank = buffer_size_sums.index(min(buffer_size_sums))
        buffer_size_sums[rank] += aligned_buffer_size
        buffer_size_ranks[index] = (aligned_buffer_size, rank)

    return buffer_size_ranks


def split_local_dist_buffers(
    buffer_size_ranks: List[Tuple[int, int]],
    local_dist_buffers: Union[Tuple[torch.Tensor], List[torch.Tensor]],
) -> List[Tuple[torch.Tensor, int]]:
    """Split given buffers according to a list of pairs of buffer size and an assigned rank.

    Args:
        buffer_size_ranks (List[Tuple[int, int]]): a list of pairs of buffer size and an assigned rank.
        local_dist_buffers (Union[Tuple[torch.Tensor], List[torch.Tensor]]): a list of tensors to be split

    Returns:
        buffer_ranks (List[Tuple[torch.Tensor, int]]): A list of pairs of a view tensor and an assigned rank

    Example:
        tensor0 = tensor(1024)
        tensor1 = tensor(1024)
        buffer_size_ranks = [(128, 0), (64, 0), (512, 1), (256, 0)]
        local_dist_buffers = [tensor0, tensor1]
        -> buffer_ranks = [
             (tensor0's view(  0-128 bytes), 0),
             (tensor0's view(128-192 bytes), 0),
             (tensor1's view(  0-512 bytes), 1),
             (tensor0's view(192-448 bytes), 0),
           ]

    """

    # Create list of lists containing local views of each split tensor for each rank.
    split_tensors_list = []
    for rank, local_dist_buffer in enumerate(local_dist_buffers):
        buffer_sizes = [s for s, r in buffer_size_ranks if r == rank]
        remainder_size = local_dist_buffer.size(0) - sum(buffer_sizes)
        split_tensors = torch.split(local_dist_buffer, buffer_sizes + [remainder_size])
        split_tensors_list.append(split_tensors)

    # Obtain ordered buffer ranks containing (view of local buffer, rank).
    buffer_ranks = []
    buffer_indices = [0] * len(
        local_dist_buffers
    )  # index counter for each rank for obtaining right buffer
    for _, rank in buffer_size_ranks:
        buffer_ranks.append((split_tensors_list[rank][buffer_indices[rank]], rank))
        buffer_indices[rank] += 1

    return buffer_ranks


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size (bytes) of a given data type."""
    return math.ceil(
        (torch.finfo if dtype.is_floating_point else torch.iinfo)(dtype).bits / 8.0
    )


def allocate_distributed_tensor(
    shape,
    dtype: torch.dtype,
    device: torch.device,
    device_mesh_ranks: Optional[List[int]] = None,
    use_dtensor: bool = True,
) -> torch.Tensor:
    """Instantiates distributed tensor using Tensor or DTensor.

    Args:
        shape (List[int]): Shape of desired tensor.
        dtype (torch.dtype): DType of desired tensor.
        device (torch.device): Device of desired tensor.
        device_mesh_ranks (Optional[List[int]]): Ranks to use in device mesh of desired tensor.
        use_dtensor (bool): Flag for using DTensor. If True and available, uses DTensor.  Otherwise, uses Tensor.

    Returns:
        out (Tensor): Desired tensor or DTensor.

    """
    if (
        ENABLE_DTENSOR
        and dist.is_initialized()
        and use_dtensor
        and device_mesh_ranks is not None
    ):
        global _device_mesh_cache

        key = repr(device_mesh_ranks)
        if key not in _device_mesh_cache:
            _device_mesh_cache[key] = dtensor.DeviceMesh(
                device_type=device.type, mesh=device_mesh_ranks
            )
        device_mesh = _device_mesh_cache[key]

        return dtensor_zeros(
            shape,
            dtype=dtype,
            device_mesh=device_mesh,
            placements=[dtensor.Replicate()],
        )
    else:
        return torch.zeros(shape, dtype=dtype, device=device)


def use_local_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
    """Uses local tensor if input is a DTensor."""
    return (
        input_tensor.to_local()
        if ENABLE_DTENSOR and isinstance(input_tensor, dtensor.DTensor)
        else input_tensor
    )
