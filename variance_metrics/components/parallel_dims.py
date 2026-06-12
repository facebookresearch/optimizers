# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

from components.utils import device_type, logger
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    tp: int
    world_size: int

    _meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _global_meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        for d in (self.dp_replicate, self.tp):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"
        assert self.dp_shard == -1 or self.dp_shard >= 1, "dp_shard must be -1 or >=1."
        if self.dp_shard < 0:
            self.dp_shard = self.world_size // (self.dp_replicate * self.tp)
        assert self.dp_shard >= 1
        assert self.dp_replicate * self.dp_shard * self.tp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({self.dp_replicate}) * "
            f"dp_shard({self.dp_shard}) * tp({self.tp}) != "
            f"WORLD_SIZE({self.world_size})"
        )

    def _mesh_exists(self, degree: int) -> bool:
        return degree > 1

    def build_mesh(self) -> DeviceMesh:
        """Build the device mesh.

        Created dimensions:
            batch:        ``dp_replicate`` * ``dp_shard`` (data loading +
                            global-batch sizing).
            loss:         ``dp_replicate`` * ``dp_shard`` (all-reduce group
                            for loss reduction).
            dp_replicate: DDP / HSDP replicate dimension.
            fsdp:         ``dp_shard`` (FSDP shard dimension).
            tp:           Tensor Parallelism (TP).
        """

        def unflatten(world_mesh, dim_names, dim_degrees):
            backend_override = {
                name: "fake"
                for name, deg in zip(dim_names, dim_degrees, strict=True)
                if not self._mesh_exists(deg)
            }
            return world_mesh._unflatten(
                0,
                dim_degrees,
                dim_names,
                # pyrefly: ignore [bad-argument-type]
                backend_override=backend_override,
            )

        logger.info(
            f"Building device mesh: "
            f"dp_replicate={self.dp_replicate}, dp_shard={self.dp_shard}, "
            f"tp={self.tp}"
        )

        batch = self.dp_replicate * self.dp_shard
        world_mesh = init_device_mesh(
            device_type, (self.world_size,), mesh_dim_names=("world",)
        )
        self._world_mesh = world_mesh
        dataloading_mesh = unflatten(world_mesh, ("batch", "tp"), (batch, self.tp))
        loss_mesh = dataloading_mesh["batch"]._flatten("loss_mesh")
        dense_mesh = unflatten(
            world_mesh,
            ("dp_replicate", "fsdp", "tp"),
            (self.dp_replicate, self.dp_shard, self.tp),
        )

        self._global_meshes = {
            "dataloading": dataloading_mesh,
            "loss": loss_mesh,
            "dense": dense_mesh,
        }

        self._meshes = {
            "batch": dataloading_mesh["batch"],
            "loss": loss_mesh,
            "dp_replicate": dense_mesh["dp_replicate"],
            "fsdp": dense_mesh["fsdp"],
            "tp": dataloading_mesh["tp"],
        }

        for name, expected in {
            "batch": self.dp_replicate * self.dp_shard,
            "loss": self.dp_replicate * self.dp_shard,
            "dp_replicate": self.dp_replicate,
            "fsdp": self.dp_shard,
            "tp": self.tp,
        }.items():
            actual = self._meshes[name].size()
            assert actual == expected, (
                f"Mesh '{name}' has unexpected size: expected {expected}, got {actual}"
            )

        logger.info(
            f"Active mesh dims: {list(self.get_all_one_dimensional_meshes().keys())}"
        )
        return world_mesh

    def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
        """Get a device mesh by name(s); returns None if size is 1.

        Valid names: 'batch', 'loss', 'dp_replicate', 'fsdp', 'tp'.
        """
        if not self._meshes:
            self.build_mesh()
        if isinstance(dims, str):
            dims = [dims]
        for name in dims:
            if name not in self._meshes:
                raise ValueError(
                    f"Invalid mesh dim: {name!r}. Valid: {list(self._meshes.keys())}"
                )
        if any(not self._mesh_exists(self._meshes[d].size()) for d in dims):
            return None
        if len(dims) == 1:
            return self._meshes[dims[0]]
        for global_mesh in self._global_meshes.values():
            assert global_mesh.mesh_dim_names is not None
            if set(dims).issubset(set(global_mesh.mesh_dim_names)):
                return global_mesh[tuple(dims)]
        raise ValueError(f"Invalid mesh name combinations {dims}.")

    def get_mesh(self, dims: str | list[str]) -> DeviceMesh:
        mesh = self.get_optional_mesh(dims)
        if mesh is None:
            raise ValueError(f"Mesh {dims!r} not available (parallelism not enabled).")
        return mesh

    def get_all_one_dimensional_meshes(self) -> dict[str, DeviceMesh]:
        if not self._meshes:
            self.build_mesh()
        return {k: v for k, v in self._meshes.items() if v.ndim == 1 and v.size() > 1}

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def fsdp_enabled(self):
        return self.dp_shard > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def seq_len_divisor(self) -> int:
        # Sequence parallel requires seq_len divisible by TP degree.
        return self.tp
