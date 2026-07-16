"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest

import numpy as np
import torch
from distributed_shampoo.utils.shampoo_fully_shard_utils import (
    _compute_chunk_sizes,
    _compute_param_chunk_sizes,
    GatherGradientsContext,
    prepare_update_param_buffers,
    redistribute_and_update_params,
    RedistributeParamsContext,
)
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


def generate_param_shapes(num_params: int) -> list[tuple[int, ...]]:
    """Generate parameter shapes for testing.

    For N parameters, we generate the following shapes:
        [(1, 2), (2, 3), (3, 4), ..., (N, N + 1)].
    """
    return [(i, i + 1) for i in range(1, num_params + 1)]


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class RedistributeAndUpdateParamsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("num_params", (1, 4, 7))
    def test_redistribute_and_update_params(self, num_params: int) -> None:
        device_mesh = init_device_mesh("cuda", (4,))
        shapes = generate_param_shapes(num_params)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )

        update_buffers = prepare_update_param_buffers(dtensor_params, self.world_size)
        self.assertEqual(
            len(update_buffers),
            int(np.ceil(num_params / self.world_size) * self.world_size),
        )
        for i, buffer in enumerate(update_buffers):
            if i < num_params:
                self.assertEqual(buffer.numel(), dtensor_params[i].to_local().numel())
            else:
                self.assertEqual(buffer.numel(), 0)

        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()
        # Fill the locally assigned parameters with the rank as value.
        local_full_params = [
            torch.zeros(s, device="cuda").fill_(rank)
            for i, s in enumerate(shapes)
            if i % self.world_size == rank
        ]
        redistribute_and_update_params(
            dtensor_params, local_full_params, update_buffers, dist_group
        )
        for i, param in enumerate(dtensor_params):
            np.testing.assert_allclose(
                param.to_local().cpu().numpy(), i % self.world_size
            )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class RedistributeParamsContextTest(DTensorTestBase):
    """Tests for RedistributeParamsContext class.

    Validates initialization, metadata precomputation, and parameter redistribution
    using a single coalesced all_to_all collective.
    """

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_init_empty_params_raises_value_error(self) -> None:
        """Verify that empty params raises ValueError."""
        dist_group = dist.distributed_c10d._get_default_group()
        with self.assertRaises(ValueError):
            RedistributeParamsContext(
                params=(),
                assigned_params_mask=(),
                dist_group=dist_group,
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_init_mismatched_dtypes_raises_not_implemented(self) -> None:
        """Verify that params with different dtypes raises NotImplementedError."""
        device_mesh = init_device_mesh("cuda", (4,))
        param_f32 = distribute_tensor(
            torch.zeros(4, 4, device="cuda", dtype=torch.float32),
            device_mesh,
            [Shard(0)],
        )
        param_f64 = distribute_tensor(
            torch.zeros(4, 4, device="cuda", dtype=torch.float64),
            device_mesh,
            [Shard(0)],
        )
        dist_group = dist.distributed_c10d._get_default_group()
        with self.assertRaises(NotImplementedError):
            RedistributeParamsContext(
                params=(param_f32, param_f64),
                assigned_params_mask=(True, False),
                dist_group=dist_group,
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_precomputed_metadata_chunk_sizes(self) -> None:
        """Verify that chunk sizes are correctly precomputed for different param shapes."""
        device_mesh = init_device_mesh("cuda", (4,))
        # Create params with dim-0 size of 8 (evenly divisible by 4 ranks)
        param1 = distribute_tensor(
            torch.zeros(8, 3, device="cuda"), device_mesh, [Shard(0)]
        )
        # Create params with dim-0 size of 5 (not evenly divisible by 4 ranks)
        param2 = distribute_tensor(
            torch.zeros(5, 2, device="cuda"), device_mesh, [Shard(0)]
        )
        dist_group = dist.distributed_c10d._get_default_group()
        ctx = RedistributeParamsContext(
            params=(param1, param2),
            assigned_params_mask=(True, False),
            dist_group=dist_group,
        )

        # param1: shape (8, 3), dim-0 chunked into 4 → [2, 2, 2, 2] * 3 = [6, 6, 6, 6]
        self.assertEqual(ctx._param_chunk_sizes[0], [6, 6, 6, 6])
        # param2: shape (5, 2), dim-0 chunked into 4 → [2, 2, 1, 0] * 2 = [4, 4, 2, 0]
        # (torch.chunk(5, 4) → ceil(5/4)=2, chunks of [2,2,1])
        # Actually: ceil(5/4) = 2, so chunks = [2, 2, 1, 0] → *2 = [4, 4, 2, 0]
        expected_dim0_chunks = _compute_chunk_sizes(5, 4)
        expected_chunk_sizes_param2 = [s * 2 for s in expected_dim0_chunks]
        self.assertEqual(ctx._param_chunk_sizes[1], expected_chunk_sizes_param2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_precomputed_metadata_send_recv_sizes(self) -> None:
        """Verify send and recv sizes are correctly computed."""
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        shapes = generate_param_shapes(4)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )
        # Round-robin: rank 0 owns param 0, rank 1 owns param 1, etc.
        assigned_mask = tuple(i % 4 == rank for i in range(4))
        dist_group = dist.distributed_c10d._get_default_group()

        ctx = RedistributeParamsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )

        # Verify send_sizes sum equals total_send_size
        self.assertEqual(sum(ctx._send_sizes), ctx._total_send_size)
        # Verify recv_sizes sum equals total_recv_size
        self.assertEqual(sum(ctx._recv_sizes), ctx._total_recv_size)
        # Verify recv buffer is pre-allocated with correct size
        self.assertEqual(ctx._recv_buffer.numel(), ctx._total_recv_size)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_precomputed_metadata_local_param_indices(self) -> None:
        """Verify local param indices are correctly identified from assigned_params_mask."""
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        shapes = generate_param_shapes(7)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )
        # Round-robin assignment
        assigned_mask = tuple(i % 4 == rank for i in range(7))
        dist_group = dist.distributed_c10d._get_default_group()

        ctx = RedistributeParamsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )

        expected_indices = [i for i in range(7) if i % 4 == rank]
        self.assertEqual(ctx._local_param_indices, expected_indices)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("num_params", (1, 4, 7))
    def test_redistribute_params_context_correctness(self, num_params: int) -> None:
        """Verify RedistributeParamsContext produces correct results.

        Each rank fills its assigned params with its rank value, then redistributes.
        After redistribution, each param's local shard should contain the value of the
        rank that owns that param (param i is owned by rank i % world_size).
        """
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        shapes = generate_param_shapes(num_params)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )

        # Round-robin assignment
        assigned_mask = tuple(i % self.world_size == rank for i in range(num_params))
        dist_group = dist.distributed_c10d._get_default_group()

        ctx = RedistributeParamsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )

        # Fill locally assigned parameters with rank value
        local_full_params = [
            torch.zeros(s, device="cuda").fill_(rank)
            for i, s in enumerate(shapes)
            if i % self.world_size == rank
        ]

        ctx.redistribute_and_update_params(local_full_params)

        # Verify each param's local shard has the correct value
        for i, param in enumerate(dtensor_params):
            owning_rank = i % self.world_size
            local_data = param.to_local().cpu().numpy()
            np.testing.assert_allclose(
                local_data,
                owning_rank,
                err_msg=f"Param {i} (owned by rank {owning_rank}) has incorrect values "
                f"on rank {rank}",
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_redistribute_params_context_wrong_num_local_params_asserts(
        self,
    ) -> None:
        """Verify assertion when local_full_params count doesn't match."""
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        shapes = generate_param_shapes(4)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )
        assigned_mask = tuple(i % 4 == rank for i in range(4))
        dist_group = dist.distributed_c10d._get_default_group()

        ctx = RedistributeParamsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )

        # Provide wrong number of local params
        wrong_local_params = [
            torch.zeros(1, device="cuda"),
            torch.zeros(1, device="cuda"),
        ]
        with self.assertRaises(AssertionError):
            ctx.redistribute_and_update_params(wrong_local_params)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_redistribute_params_context_matches_legacy(self) -> None:
        """Verify RedistributeParamsContext produces identical results to the legacy
        redistribute_and_update_params function."""
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        num_params = 7
        shapes = generate_param_shapes(num_params)

        # Create two sets of identical DTensor params
        params_legacy = [torch.zeros(s, device="cuda") for s in shapes]
        params_new = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params_legacy = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params_legacy
        )
        dtensor_params_new = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params_new
        )

        dist_group = dist.distributed_c10d._get_default_group()
        assigned_mask = tuple(i % self.world_size == rank for i in range(num_params))

        # Create the same local_full_params for both paths
        # Use distinct values per param to verify correctness
        local_full_params_legacy = [
            torch.zeros(s, device="cuda").fill_(float(rank * 100 + i))
            for i, s in enumerate(shapes)
            if i % self.world_size == rank
        ]
        local_full_params_new = [
            torch.zeros(s, device="cuda").fill_(float(rank * 100 + i))
            for i, s in enumerate(shapes)
            if i % self.world_size == rank
        ]

        # Run legacy path
        update_buffers = prepare_update_param_buffers(
            dtensor_params_legacy, self.world_size
        )
        redistribute_and_update_params(
            dtensor_params_legacy,
            local_full_params_legacy,
            update_buffers,
            dist_group,
        )

        # Run new path
        ctx = RedistributeParamsContext(
            params=dtensor_params_new,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )
        ctx.redistribute_and_update_params(local_full_params_new)

        # Compare results
        for i in range(num_params):
            legacy_vals = dtensor_params_legacy[i].to_local().cpu().numpy()
            new_vals = dtensor_params_new[i].to_local().cpu().numpy()
            np.testing.assert_allclose(
                new_vals,
                legacy_vals,
                err_msg=f"Param {i} mismatch between legacy and new redistribution on rank {rank}",
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_recv_buffer_reuse(self) -> None:
        """Verify that the recv buffer is reused across multiple redistribute calls."""
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        shapes = generate_param_shapes(4)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )
        assigned_mask = tuple(i % 4 == rank for i in range(4))
        dist_group = dist.distributed_c10d._get_default_group()

        ctx = RedistributeParamsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )

        # Capture the recv buffer's data_ptr
        recv_buffer_ptr = ctx._recv_buffer.data_ptr()

        # Run redistribute twice
        for _ in range(2):
            local_full_params = [
                torch.zeros(s, device="cuda").fill_(rank)
                for i, s in enumerate(shapes)
                if i % 4 == rank
            ]
            ctx.redistribute_and_update_params(local_full_params)

        # Verify the same buffer is used (no re-allocation)
        self.assertEqual(ctx._recv_buffer.data_ptr(), recv_buffer_ptr)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_param_recv_info_completeness(self) -> None:
        """Verify _param_recv_info has valid entries for all params."""
        device_mesh = init_device_mesh("cuda", (4,))
        rank = dist.get_rank()
        num_params = 5
        shapes = generate_param_shapes(num_params)
        params = [torch.zeros(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )
        assigned_mask = tuple(i % 4 == rank for i in range(num_params))
        dist_group = dist.distributed_c10d._get_default_group()

        ctx = RedistributeParamsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_mask,
            dist_group=dist_group,
        )

        # Every param should have a valid recv info entry (offset >= 0)
        for param_idx in range(num_params):
            offset, chunk_size = ctx._param_recv_info[param_idx]
            self.assertGreaterEqual(  # type: ignore
                offset, 0, f"Param {param_idx} has invalid recv offset {offset}"
            )
            self.assertGreaterEqual(
                chunk_size,
                0,
                f"Param {param_idx} has invalid recv chunk_size {chunk_size}",
            )


@instantiate_parametrized_tests
class ComputeParamChunkSizesTest(DTensorTestBase):
    """Tests for _compute_param_chunk_sizes with actual DTensors."""

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("num_params", (1, 4, 7))
    def test_chunk_sizes_match_local_numels(self, num_params: int) -> None:
        """Verify computed chunk sizes match actual local shard numels."""
        device_mesh = init_device_mesh("cuda", (4,))
        shapes = generate_param_shapes(num_params)
        params = [torch.randn(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )

        chunk_sizes = _compute_param_chunk_sizes(dtensor_params, self.world_size)

        rank = dist.get_rank()
        for i, param in enumerate(dtensor_params):
            local_numel = param.to_local().numel()
            self.assertEqual(
                chunk_sizes[i][rank],
                local_numel,
                f"Chunk size mismatch for param {i} on rank {rank}",
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_chunk_sizes_sum_to_global_numel(self) -> None:
        """Verify chunk sizes across all ranks sum to global numel."""
        device_mesh = init_device_mesh("cuda", (4,))
        shapes = [(8, 4), (3, 5), (1, 2)]
        params = [torch.randn(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )

        chunk_sizes = _compute_param_chunk_sizes(dtensor_params, self.world_size)

        for i, param in enumerate(dtensor_params):
            global_numel = param.numel()
            computed_total = sum(chunk_sizes[i])
            self.assertEqual(
                computed_total,
                global_numel,
                f"Chunk sizes for param {i} sum to {computed_total}, expected {global_numel}",
            )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class GatherGradientsContextTest(DTensorTestBase):
    """Tests for GatherGradientsContext gradient gathering using all_to_all."""

    @property
    def world_size(self) -> int:
        return 4

    def _create_dtensor_params_with_grads(
        self,
        shapes: list[tuple[int, ...]],
        device_mesh: torch.distributed.device_mesh.DeviceMesh,
    ) -> tuple[torch.distributed.tensor.DTensor, ...]:
        """Create DTensor params and set their gradients via a dummy backward pass."""
        dtensor_params = []
        for shape in shapes:
            global_tensor = torch.randn(shape, device="cuda", requires_grad=True)
            dtensor = distribute_tensor(global_tensor, device_mesh, [Shard(0)])
            dtensor_params.append(dtensor)

        loss = sum(p.sum() for p in dtensor_params)
        loss.backward()  # type: ignore

        return tuple(dtensor_params)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("num_params", (1, 4, 7))
    def test_gather_gradients_matches_full_tensor(self, num_params: int) -> None:
        """Verify gathered grads match per-param full_tensor() for assigned params."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = generate_param_shapes(num_params)
        dtensor_params = self._create_dtensor_params_with_grads(shapes, device_mesh)

        # Compute expected full grads using full_tensor()
        expected_full_grads = [
            None if p.grad is None else p.grad.full_tensor()  # type: ignore
            for p in dtensor_params
        ]

        # Build round-robin assignment mask
        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(num_params)
        )

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )
        gathered_grads = ctx.gather_gradients()

        # Verify length
        self.assertEqual(len(gathered_grads), num_params)

        # Verify correctness for assigned params
        for i in range(num_params):
            if i % self.world_size == rank:
                self.assertIsNotNone(  # type: ignore
                    gathered_grads[i],
                    f"Assigned param {i} should have a gathered gradient",
                )
                torch.testing.assert_close(
                    gathered_grads[i],
                    expected_full_grads[i],
                    msg=f"Gathered gradient for param {i} does not match full_tensor()",
                )
            else:
                # Unassigned params should be None
                self.assertIsNone(  # type: ignore
                    gathered_grads[i],
                    f"Unassigned param {i} should be None on rank {rank}",
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_gradients_with_none_grads(self) -> None:
        """Verify params without gradients return None."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = generate_param_shapes(4)
        # Create params WITHOUT gradients
        dtensor_params = tuple(
            distribute_tensor(torch.randn(s, device="cuda"), device_mesh, [Shard(0)])
            for s in shapes
        )

        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(len(shapes))
        )

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )
        gathered_grads = ctx.gather_gradients()

        # All should be None since no gradients were set
        for i, grad in enumerate(gathered_grads):
            self.assertIsNone(  # type: ignore
                grad, f"Param {i} has no grad, should be None"
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_gradients_with_partial_none_grads(self) -> None:
        """Verify correct handling when some params have grads and some don't."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = generate_param_shapes(4)
        dtensor_params_list = [
            distribute_tensor(
                torch.randn(s, device="cuda", requires_grad=True),
                device_mesh,
                [Shard(0)],
            )
            for s in shapes
        ]

        # Only set gradients on even-indexed params (0, 2)
        for i, p in enumerate(dtensor_params_list):
            if i % 2 == 0:
                fake_loss = p.sum()
                fake_loss.backward()

        dtensor_params = tuple(dtensor_params_list)

        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(len(shapes))
        )

        # Compute expected grads via full_tensor() for params that have grads
        expected_full_grads = [
            None if p.grad is None else p.grad.full_tensor()  # type: ignore
            for p in dtensor_params
        ]

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )
        gathered_grads = ctx.gather_gradients()

        for i in range(len(shapes)):
            if i % self.world_size == rank:
                if expected_full_grads[i] is not None:
                    self.assertIsNotNone(gathered_grads[i])  # type: ignore
                    torch.testing.assert_close(
                        gathered_grads[i], expected_full_grads[i]
                    )
                else:
                    self.assertIsNone(gathered_grads[i])  # type: ignore
            else:
                self.assertIsNone(gathered_grads[i])  # type: ignore

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_gradients_preserves_shape(self) -> None:
        """Verify gathered gradients have the correct global shape."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = [(8, 4), (12, 3), (5, 7, 2)]
        # pyrefly: ignore [bad-argument-type]
        dtensor_params = self._create_dtensor_params_with_grads(shapes, device_mesh)

        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(len(shapes))
        )

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )
        gathered_grads = ctx.gather_gradients()

        for i in range(len(shapes)):
            if i % self.world_size == rank:
                self.assertIsNotNone(gathered_grads[i])  # type: ignore
                self.assertEqual(
                    gathered_grads[i].shape,  # type: ignore
                    torch.Size(shapes[i]),
                    f"Shape mismatch for param {i}: expected {shapes[i]}, got {gathered_grads[i].shape}",  # type: ignore
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_gather_gradients_multiple_calls(self) -> None:
        """Verify gather_gradients works correctly across multiple calls."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = generate_param_shapes(4)
        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(len(shapes))
        )

        # First call with initial gradients
        dtensor_params = self._create_dtensor_params_with_grads(shapes, device_mesh)

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )

        first_grads = ctx.gather_gradients()

        # Zero gradients and do another backward with different values
        for p in dtensor_params:
            if p.grad is not None:
                p.grad.zero_()

        loss = sum(2.0 * p.sum() for p in dtensor_params)
        loss.backward()  # type: ignore

        expected_second_grads = [
            None if p.grad is None else p.grad.full_tensor()  # type: ignore
            for p in dtensor_params
        ]

        second_grads = ctx.gather_gradients()

        # Verify second call produces correct results (different from first)
        for i in range(len(shapes)):
            if i % self.world_size == rank:
                self.assertIsNotNone(second_grads[i])  # type: ignore
                torch.testing.assert_close(second_grads[i], expected_second_grads[i])
                # Verify second call differs from first
                # (2x gradient vs 1x gradient for sum)
                self.assertFalse(  # type: ignore
                    torch.equal(first_grads[i], second_grads[i]),  # type: ignore
                    f"Second gather should differ from first for param {i}",
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_empty_params_raises_error(self) -> None:
        """Verify empty params raises ValueError."""
        dist_group = dist.distributed_c10d._get_default_group()

        with self.assertRaises(ValueError):
            GatherGradientsContext(
                params=(),
                assigned_params_mask=(),
                dist_group=dist_group,
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_metadata_precomputation(self) -> None:
        """Verify precomputed metadata is consistent."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = generate_param_shapes(4)
        dtensor_params = tuple(
            distribute_tensor(torch.randn(s, device="cuda"), device_mesh, [Shard(0)])
            for s in shapes
        )

        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(len(shapes))
        )

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )

        # Verify send/recv sizes have correct count
        self.assertEqual(len(ctx._send_sizes), self.world_size)
        self.assertEqual(len(ctx._recv_sizes), self.world_size)

        # Total send size should equal sum of this rank's chunk sizes for all params
        expected_total_send = sum(
            ctx._param_chunk_sizes[i][rank] for i in range(len(shapes))
        )
        self.assertEqual(ctx._total_send_size, expected_total_send)

        # Verify recv buffer is pre-allocated with correct size
        self.assertEqual(ctx._recv_buffer.numel(), ctx._total_recv_size)

        # Verify local param indices match assignment mask
        expected_local_indices = [
            i for i, assigned in enumerate(assigned_params_mask) if assigned
        ]
        self.assertEqual(ctx._local_param_indices, expected_local_indices)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("num_params", (1, 4, 7))
    def test_gather_params_matches_full_tensor(self, num_params: int) -> None:
        """Verify gather_params() returns correct full parameter tensors."""
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        rank = dist.get_rank()
        dist_group = dist.distributed_c10d._get_default_group()

        shapes = generate_param_shapes(num_params)
        # Create params with distinct values so we can verify correctness
        params = [torch.randn(s, device="cuda") for s in shapes]
        dtensor_params = tuple(
            distribute_tensor(t, device_mesh, [Shard(0)]) for t in params
        )

        # Compute expected full params using full_tensor()
        expected_full_params = [p.full_tensor() for p in dtensor_params]

        assigned_params_mask = tuple(
            i % self.world_size == rank for i in range(num_params)
        )

        ctx = GatherGradientsContext(
            params=dtensor_params,
            assigned_params_mask=assigned_params_mask,
            dist_group=dist_group,
        )
        gathered_params = ctx.gather_params()

        # Verify length
        self.assertEqual(len(gathered_params), num_params)

        # Verify correctness for assigned params
        for i in range(num_params):
            if i % self.world_size == rank:
                self.assertIsNotNone(  # type: ignore
                    gathered_params[i],
                    f"Assigned param {i} should have a gathered value",
                )
                torch.testing.assert_close(
                    gathered_params[i],
                    expected_full_params[i],
                    msg=f"Gathered param {i} does not match full_tensor()",
                )
            else:
                self.assertIsNone(  # type: ignore
                    gathered_params[i],
                    f"Unassigned param {i} should be None on rank {rank}",
                )
