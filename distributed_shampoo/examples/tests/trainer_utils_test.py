"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import random
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.cuda
from distributed_shampoo.examples.trainer_utils import set_seed, setup_distribution


class SetSeedTest(unittest.TestCase):
    """Test cases for the set_seed() function."""

    def test_set_seed_deterministic_torch(self) -> None:
        """Test that torch.manual_seed is called and produces deterministic results."""
        seed = 42
        set_seed(seed)

        # Generate some random tensors after setting seed
        tensor1 = torch.randn(5, 5)
        tensor2 = torch.randn(5, 5)

        # Reset seed and generate again - should be identical
        set_seed(seed)
        tensor3 = torch.randn(5, 5)
        tensor4 = torch.randn(5, 5)

        # Check tensors are identical when seed is reset
        torch.testing.assert_close(tensor1, tensor3)
        torch.testing.assert_close(tensor2, tensor4)

    def test_set_seed_deterministic_numpy(self) -> None:
        """Test that numpy random seed is set and produces deterministic results."""
        seed = 123
        set_seed(seed)

        # Generate numpy random numbers
        arr1 = np.random.random((3, 3))
        arr2 = np.random.random((3, 3))

        # Reset seed and generate again
        set_seed(seed)
        arr3 = np.random.random((3, 3))
        arr4 = np.random.random((3, 3))

        # Check arrays are identical when seed is reset
        np.testing.assert_array_equal(arr1, arr3)
        np.testing.assert_array_equal(arr2, arr4)

    def test_set_seed_deterministic_python_random(self) -> None:
        """Test that Python's random module seed is set and produces deterministic results."""
        seed = 456
        set_seed(seed)

        # Generate random numbers using Python's random module
        nums1 = [random.random() for _ in range(10)]

        # Reset seed and generate again
        set_seed(seed)
        nums2 = [random.random() for _ in range(10)]

        # Check lists are identical when seed is reset
        self.assertEqual(nums1, nums2)

    def test_set_seed_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different random results."""
        # Set first seed and generate tensors
        set_seed(42)
        tensor1 = torch.randn(5, 5)

        # Set different seed and generate tensor
        set_seed(999)
        tensor2 = torch.randn(5, 5)

        # Tensors should be different (with very high probability)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(tensor1, tensor2)


class SetupDistributionTest(unittest.TestCase):
    """Test cases for the setup_distribution() function."""

    torch_distributed_module: ModuleType = torch.distributed
    torch_cuda_module: ModuleType = torch.cuda

    @patch.object(torch_distributed_module, "init_process_group")
    @patch.object(torch_cuda_module, "is_available", return_value=True)
    @patch.object(torch_cuda_module, "set_device")
    def test_setup_distribution_with_cuda(
        self,
        mock_set_device: MagicMock,
        mock_cuda_available: MagicMock,
        mock_init_process_group: MagicMock,
    ) -> None:
        """Test setup_distribution when CUDA is available."""

        backend = "nccl"
        world_rank = 1
        world_size = 4
        local_rank = 2

        device = setup_distribution(backend, world_rank, world_size, local_rank)

        # Verify process group initialization
        mock_init_process_group.assert_called_once_with(
            backend=backend,
            init_method="env://",
            rank=world_rank,
            world_size=world_size,
        )

        # Verify CUDA device setup
        mock_set_device.assert_called_once_with(local_rank)

        # Verify returned device
        expected_device = torch.device("cuda", local_rank)
        self.assertEqual(device, expected_device)

    @patch.object(torch_distributed_module, "init_process_group")
    @patch.object(torch_cuda_module, "is_available", return_value=False)
    @patch.object(torch_cuda_module, "set_device")
    def test_setup_distribution_without_cuda(
        self,
        mock_set_device: MagicMock,
        mock_cuda_available: MagicMock,
        mock_init_process_group: MagicMock,
    ) -> None:
        """Test setup_distribution when CUDA is not available."""

        backend = "gloo"
        world_rank = 0
        world_size = 2
        local_rank = 0

        device = setup_distribution(backend, world_rank, world_size, local_rank)

        # Verify process group initialization
        mock_init_process_group.assert_called_once_with(
            backend=backend,
            init_method="env://",
            rank=world_rank,
            world_size=world_size,
        )

        # Verify CUDA device setup is NOT called
        mock_set_device.assert_not_called()

        # Verify returned device is CPU
        expected_device = torch.device("cpu", local_rank)
        self.assertEqual(device, expected_device)
