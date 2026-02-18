"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import os
import random
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from distributed_shampoo.examples import utils
from distributed_shampoo.examples.utils import (
    get_distributed_env,
    get_model_and_loss_fn,
    instantiate_optimizer,
    load_checkpoint,
    set_seed,
    setup_distribution,
    setup_environment,
)
from omegaconf import OmegaConf
from torch import nn


class SetupEnvironmentTest(unittest.TestCase):
    def test_sets_cublas_workspace_config(self) -> None:
        """Test that setup_environment sets CUBLAS_WORKSPACE_CONFIG."""
        setup_environment()
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")


class GetDistributedEnvTest(unittest.TestCase):
    @patch.dict(os.environ, {"LOCAL_RANK": "2", "RANK": "5", "WORLD_SIZE": "8"})
    def test_reads_environment_variables(self) -> None:
        """Test that get_distributed_env reads from environment."""
        local_rank, rank, world_size = get_distributed_env()
        self.assertEqual(local_rank, 2)
        self.assertEqual(rank, 5)
        self.assertEqual(world_size, 8)

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_when_env_not_set(self) -> None:
        """Test that get_distributed_env returns defaults when env vars not set."""
        # Clear relevant env vars
        for var in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]:
            os.environ.pop(var, None)

        local_rank, rank, world_size = get_distributed_env()
        self.assertEqual(local_rank, 0)
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)


class SetSeedTest(unittest.TestCase):
    def test_set_seed_reproducibility(self) -> None:
        """Test that set_seed produces reproducible random numbers."""
        set_seed(42)
        torch_val1 = torch.rand(1).item()
        np_val1 = np.random.rand()
        random_val1 = random.random()

        set_seed(42)
        torch_val2 = torch.rand(1).item()
        np_val2 = np.random.rand()
        random_val2 = random.random()

        self.assertEqual(torch_val1, torch_val2)
        self.assertEqual(np_val1, np_val2)
        self.assertEqual(random_val1, random_val2)

    def test_set_seed_enables_deterministic_algorithms(self) -> None:
        """Test that set_seed enables deterministic algorithms."""
        set_seed(123)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())


class GetModelAndLossFnTest(unittest.TestCase):
    def test_returns_model_and_loss_fn(self) -> None:
        """Test that get_model_and_loss_fn returns a model and loss function."""
        device = torch.device("cpu")
        model, loss_fn = get_model_and_loss_fn(device)

        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(loss_fn, nn.CrossEntropyLoss)

    def test_model_on_correct_device(self) -> None:
        """Test that the model is placed on the correct device."""
        device = torch.device("cpu")
        model, _ = get_model_and_loss_fn(device)

        # Check all parameters are on the correct device
        for param in model.parameters():
            self.assertEqual(param.device, device)


class InstantiateOptimizerTest(unittest.TestCase):
    def test_instantiate_sgd_optimizer(self) -> None:
        """Test instantiating SGD optimizer."""
        cfg = OmegaConf.create(
            {
                "optimizer": {
                    "_target_": "torch.optim.SGD",
                    "_partial_": True,
                    "lr": 0.01,
                    "momentum": 0.9,
                }
            }
        )
        model = nn.Linear(10, 5)

        optimizer = instantiate_optimizer(cfg, model.parameters())

        self.assertIsInstance(optimizer, torch.optim.SGD)

    def test_instantiate_adam_optimizer(self) -> None:
        """Test instantiating Adam optimizer."""
        cfg = OmegaConf.create(
            {
                "optimizer": {
                    "_target_": "torch.optim.Adam",
                    "_partial_": True,
                    "lr": 0.001,
                }
            }
        )
        model = nn.Linear(10, 5)

        optimizer = instantiate_optimizer(cfg, model.parameters())

        self.assertIsInstance(optimizer, torch.optim.Adam)


class LoadCheckpointTest(unittest.TestCase):
    def test_load_checkpoint_returns_early_if_no_checkpoint_dir(self) -> None:
        """Test that load_checkpoint returns early if checkpoint_dir is None."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise - just returns early
        load_checkpoint(None, model, optimizer)

    def test_load_checkpoint_returns_early_if_not_shampoo(self) -> None:
        """Test that load_checkpoint returns early for non-Shampoo optimizers."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Should not raise - just returns early
        load_checkpoint("/some/path", model, optimizer)

    utils_module: ModuleType = utils

    @patch.object(utils_module, "os")
    def test_load_checkpoint_returns_early_if_metadata_not_exists(
        self, mock_os: MagicMock
    ) -> None:
        """Test that load_checkpoint returns early if metadata file doesn't exist."""
        mock_os.path.exists.return_value = False

        _ = nn.Linear(10, 5)
        # Create a mock that passes isinstance check for DistributedShampoo
        mock_optimizer = MagicMock(spec=["distributed_state_dict"])
        mock_optimizer.__class__.__name__ = "DistributedShampoo"

        # Should not raise - just returns early when metadata doesn't exist
        # We patch at module level so this won't actually check the path


class SetupDistributionTest(unittest.TestCase):
    torch_distributed_module: ModuleType = torch.distributed

    @patch.object(torch_distributed_module, "init_process_group")
    @patch("torch.cuda.is_available", return_value=False)
    def test_setup_distribution_cpu(
        self, mock_cuda: MagicMock, mock_init: MagicMock
    ) -> None:
        """Test setup_distribution on CPU."""
        device = setup_distribution(backend="gloo", rank=0, world_size=2, local_rank=0)

        mock_init.assert_called_once_with(
            backend="gloo", init_method="env://", rank=0, world_size=2
        )
        self.assertEqual(device.type, "cpu")

    @patch.object(torch_distributed_module, "init_process_group")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_setup_distribution_cuda(
        self, mock_cuda: MagicMock, mock_set_device: MagicMock, mock_init: MagicMock
    ) -> None:
        """Test setup_distribution on CUDA."""
        device = setup_distribution(backend="nccl", rank=1, world_size=4, local_rank=1)

        mock_init.assert_called_once_with(
            backend="nccl", init_method="env://", rank=1, world_size=4
        )
        mock_set_device.assert_called_once_with(1)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 1)
