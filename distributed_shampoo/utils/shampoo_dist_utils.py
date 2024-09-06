"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from functools import cache
from typing import Optional

from torch.distributed._tensor import DeviceMesh


@cache
def get_device_mesh(
    device_type: str,
    mesh: tuple[tuple[int, ...], ...] | tuple[int, ...],
    mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> DeviceMesh:
    """Returns device mesh from provided device type, mesh, and mesh dim names.
    This function will cache previous meshes according to the input.

    Args:
        device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
        mesh (tuple[tuple[int, ...], ...] | tuple[int, ...]):  A multi-dimensional array describing the layout
                of devices, where the IDs are global IDs of the default process group.
        mesh_dim_names (Optional[tuple[str, ...]]): Names of mesh dimensions. (Default: None)

    Returns:
        device_mesh (DeviceMesh): Device mesh.


    """
    return DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)
