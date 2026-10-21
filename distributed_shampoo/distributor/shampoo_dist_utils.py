"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from contextlib import contextmanager
from functools import cache, partial
from typing import Generator

import torch
from torch import distributed as dist
from torch.autograd import profiler
from torch.distributed.device_mesh import DeviceMesh


@contextmanager
def shampoo_comm_profiler(name: str) -> Generator[None, None, None]:
    """Context manager that profiles communication operations in Shampoo distributors.

    Args:
        name (str): The name to use for profiling (e.g., "ClassName::method_name").

    Example:
        with shampoo_comm_profiler("HybridShardShampooDistributor::all_gather_into_tensor"):
            dist.all_gather_into_tensor(...)

    """
    # TODO(irisz): Investigate adding CUDA NVTX ranges (torch.cuda.nvtx.range_push/pop)
    # so annotations appear on the GPU timeline even when launched from worker threads.
    with profiler.record_function(name):
        yield


@contextmanager
def cuda_stream_context(stream: torch.cuda.Stream) -> Generator[None, None, None]:
    """Run the with-block on ``stream``, ordered after the device's current stream.

    Caller must afterward sync ``stream`` back to the default stream (e.g.
    ``torch.cuda.current_stream(stream.device).wait_stream(stream)`` from the
    main thread once worker threads have joined) so subsequent default-stream
    work observes this stream's results.

    Args:
        stream (torch.cuda.Stream): The pre-created per-group stream.

    Example:
        with cuda_stream_context(state_lists[CUDA_STREAM]):
            run_step_body()

    """
    stream.wait_stream(torch.cuda.current_stream(stream.device))
    with torch.cuda.stream(stream):
        yield


@cache
def get_device_mesh(
    device_type: str,
    mesh: tuple[tuple[int, ...], ...] | tuple[int, ...],
    mesh_dim_names: tuple[str, ...] | None = None,
) -> DeviceMesh:
    """Returns device mesh from provided device type, mesh, and mesh dim names.
    This function will cache previous meshes according to the input.

    Args:
        device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
        mesh (tuple[tuple[int, ...], ...] | tuple[int, ...]):  A multi-dimensional array describing the layout
                of devices, where the IDs are global IDs of the default process group.
        mesh_dim_names (tuple[str, ...] | None): Names of mesh dimensions. (Default: None)

    Returns:
        device_mesh (DeviceMesh): Device mesh.


    """
    return DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)


@cache
def _get_optimizer_state_device_mesh(
    device_type: str,
    mesh: tuple[tuple[int, ...], ...],
) -> DeviceMesh:
    """Create a canonical optimizer-state mesh from a globally ordered call site."""
    return DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=("replicate", "shard"),
    )


def create_hybrid_shard_process_groups(
    device_mesh: DeviceMesh,
    dist_group_size: int,
) -> tuple[dist.ProcessGroup, DeviceMesh]:
    """Create the comms process group and its containing optimizer device mesh.

    Splits replicated rank groups into sub-groups of size dist_group_size
    and returns the process group for the current rank's sub-group along
    the shard dimension.

    Args:
        device_mesh: The hybrid shard device mesh with (replicate, shard) dims.
        dist_group_size: Size of each distribution sub-group.

    Returns:
        The process group for communication along the shard dimension and the
        per-column device mesh containing that group for the current rank.
    """
    mesh_dim_names = device_mesh.mesh_dim_names
    assert mesh_dim_names is not None, "DeviceMesh must have mesh_dim_names"
    shard_dim_name = mesh_dim_names[1]
    ranks_in_all_replicated_groups = device_mesh.mesh.T
    comms_dist_group: dist.ProcessGroup | None = None
    local_optimizer_state_mesh: DeviceMesh | None = None
    for ranks_in_replicated_group in ranks_in_all_replicated_groups:
        optimizer_mesh = tuple(
            map(
                partial(tuple),
                ranks_in_replicated_group.view(-1, dist_group_size).tolist(),
            )
        )
        comms_device_mesh = get_device_mesh(
            device_type=device_mesh.device_type,
            mesh=optimizer_mesh,
            mesh_dim_names=mesh_dim_names,
        )
        # Optimizer-state meshes must have stable names across optimizer
        # instances for checkpoint save/load. Construct them here, where all
        # ranks visit every replicated group in the same order. The cache makes
        # later optimizer instances reuse these exact meshes and process groups.
        optimizer_state_mesh = _get_optimizer_state_device_mesh(
            device_type=device_mesh.device_type,
            mesh=optimizer_mesh,
        )
        if dist.get_rank() in ranks_in_replicated_group:
            comms_dist_group = comms_device_mesh.get_group(shard_dim_name)
            local_optimizer_state_mesh = optimizer_state_mesh
    assert comms_dist_group is not None and local_optimizer_state_mesh is not None
    return comms_dist_group, local_optimizer_state_mesh
