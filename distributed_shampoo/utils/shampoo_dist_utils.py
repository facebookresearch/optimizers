"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from functools import cache
from typing import Optional

import torch

from torch.distributed import _tensor as dtensor
from torch.distributed._tensor import DeviceMesh


@cache
def get_device_mesh(
    device_type: str,
    mesh: torch.Tensor | tuple[int, ...],
    mesh_dim_names: Optional[tuple[str, ...]] = None,
) -> dtensor.DeviceMesh:
    """Returns device mesh from provided ranks. This function will cache previous meshes according to the input ranks.

    Args:
        device_type (str): The device type of the mesh. Currently supports: "cpu", "cuda/cuda-like".
        mesh (torch.Tensor | tuple[int, ...]):  A multi-dimensional array or an integer tensor describing the layout
                of devices, where the IDs are global IDs of the default process group.
        mesh_dim_names (Optional[tuple[str, ...]]): Names of mesh dimensions.

    Returns:
        device_mesh (dtensor.DeviceMesh): Device mesh.


    """
    return DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)
