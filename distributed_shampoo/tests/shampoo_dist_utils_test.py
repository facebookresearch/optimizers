"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest

import torch
import torch.distributed as dist
import torch.distributed._tensor as dtensor

from distributed_shampoo.utils.shampoo_dist_utils import (
    allocate_distributed_tensor,
    distribute_buffer_sizes,
    get_dtype_size,
    split_local_dist_buffers,
    use_local_tensor,
)
from torch.testing._internal.common_distributed import spawn_threads_and_init_comms


class DistributeBufferSizesTest(unittest.TestCase):
    def test_distribute_buffer_sizes(self) -> None:
        buffer_sizes = [128, 64, 500, 256]
        group_size = 2
        buffer_size_ranks = [(128, 1), (64, 1), (512, 0), (256, 1)]
        self.assertEqual(
            distribute_buffer_sizes(buffer_sizes, group_size), buffer_size_ranks
        )


class SplitLocalDistBuffersTest(unittest.TestCase):
    def test_split_local_dist_buffers(self) -> None:
        buffer_size_ranks = [(128, 0), (64, 0), (512, 1), (256, 0)]
        local_dist_buffers = [
            torch.zeros(1024, dtype=torch.int8),
            torch.zeros(1024, dtype=torch.int8),
        ]
        expected_buffer_ranks = [
            (torch.zeros(128, dtype=torch.int8), 0),
            (torch.zeros(64, dtype=torch.int8), 0),
            (torch.zeros(512, dtype=torch.int8), 1),
            (torch.zeros(256, dtype=torch.int8), 0),
        ]
        torch.testing.assert_close(
            expected_buffer_ranks,
            split_local_dist_buffers(buffer_size_ranks, local_dist_buffers),
        )


class GetDTypeSizeTest(unittest.TestCase):
    def test_get_dtype_size(self) -> None:
        self.assertEqual(get_dtype_size(torch.float32), 4)


class AllocateDistributedTensorTest(unittest.TestCase):
    def test_allocate_distributed_tensor_with_tensor(self) -> None:
        shape = (4, 4)
        device = torch.device("cpu")
        device_mesh_ranks = [0]
        expected_tensor = torch.zeros(shape, device=device)
        torch.testing.assert_close(
            allocate_distributed_tensor(
                shape,
                dtype=torch.float32,
                device=device,
                device_mesh_ranks=device_mesh_ranks,
                use_dtensor=False,
            ),
            expected_tensor,
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_allocate_distributed_tensor_with_dtensor(self) -> None:
        shape = (4, 4)
        device = torch.device("cpu")
        # TODO: Change device_mesh_ranks = [0, 1] after bug fix with torch.testing.assert_close and torch.equal.
        device_mesh_ranks = [0, 1, 2, 3]
        expected_tensor = dtensor.zeros(
            shape,
            dtype=torch.float32,
            device_mesh=dtensor.DeviceMesh("cpu", device_mesh_ranks),
            placements=[dtensor.Replicate()],
        )
        self.assertTrue(
            torch.equal(
                allocate_distributed_tensor(
                    shape,
                    dtype=torch.float32,
                    device=device,
                    device_mesh_ranks=device_mesh_ranks,
                    use_dtensor=True,
                ),
                expected_tensor,
            )
        )


class UseLocalTensorTest(unittest.TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_use_local_tensor_with_dtensor(self) -> torch.Tensor:
        device_mesh_ranks = [0, 1]
        device_mesh = dtensor.DeviceMesh("cpu", device_mesh_ranks)
        tensor = dtensor.zeros(
            (4, 4),
            dtype=torch.float,
            device_mesh=device_mesh,
            placements=[dtensor.Replicate()],
        )
        expected_tensor = torch.zeros(4, 4)
        # pyre-ignore [7]
        torch.testing.assert_close(
            use_local_tensor(tensor),
            expected_tensor
            if dist.get_rank() in device_mesh_ranks
            else torch.tensor([]),
        )

    def test_use_local_tensor_with_tensor(self) -> torch.Tensor:
        expected_tensor: torch.Tensor = torch.randn(4, 4)
        # pyre-ignore [7]
        torch.testing.assert_close(use_local_tensor(expected_tensor), expected_tensor)
