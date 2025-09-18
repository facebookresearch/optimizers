"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import argparse
import os
from functools import partial

import torch
import torch.distributed as dist

from distributed_shampoo import DistributedConfig, HybridShardDistributedConfig
from distributed_shampoo.examples.argument_parser import Parser
from distributed_shampoo.examples.trainer_utils import (
    create_model_and_optimizer_and_loss_fn,
    get_data_loader_and_sampler,
    set_seed,
    setup_distribution,
    train_model,
)

from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule
from torchvision.datasets import VisionDataset

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# get local and world rank and world size
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def main(
    args: argparse.Namespace,
    device_mesh: DeviceMesh | None,
    distributed_config: DistributedConfig,
) -> None:
    # set seed for reproducibility
    set_seed(args.seed)

    # initialize distributed process group
    device: torch.device = setup_distribution(
        backend=args.backend,
        world_rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        local_rank=LOCAL_RANK,
    )

    model: nn.Module | FSDPModule
    optimizer: torch.optim.Optimizer
    loss_fn: nn.Module
    model, optimizer, loss_fn = create_model_and_optimizer_and_loss_fn(
        args=args,
        device=device,
        distributed_config=distributed_config,
        post_model_decoration=partial(fully_shard, mesh=device_mesh),
    )

    # instantiate data loader
    data_loader: torch.utils.data.DataLoader[VisionDataset]
    sampler: torch.utils.data.distributed.DistributedSampler[
        torch.utils.data.Dataset[VisionDataset]
    ]
    data_loader, sampler = get_data_loader_and_sampler(
        args.data_path, WORLD_SIZE, WORLD_RANK, args.local_batch_size
    )

    # train model
    train_model(
        model,
        WORLD_SIZE,
        loss_fn,
        sampler,
        data_loader,
        optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        window_size=args.window_size,
        local_rank=LOCAL_RANK,
        metrics_dir=args.metrics_dir if WORLD_RANK == 0 else None,
    )

    # clean up process group
    dist.destroy_process_group()


if __name__ == "__main__":
    """Multi-GPU CIFAR-10 Per-parameter Hybrid Sharded Data Parallel (a.k.a HSDP2) Training Example Script

    Uses torch.distributed to launch distributed training run.

    Requirements:
        - Python 3.12 or above
        - PyTorch / TorchVision

    To run this training script with a single node, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hybrid_shard_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hybrid_shard_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting, precondition frequency = 100):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hybrid_shard_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --num-trainers-per-group -1 --use-bias-correction --use-decoupled-weight-decay --use-merge-dims --dp-replicate-degree 2

    To use distributed checkpointing on Distributed Shampoo, append the flag with --checkpoint-dir argument.

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """

    args: argparse.Namespace = Parser.get_args()
    # Instantiate device mesh for Hybrid Sharded Data Parallel Shampoo.
    # For example, with 8 GPUs and dp_replicate_degree set to 2, the device mesh will be:
    # ([[0, 1, 2, 3], [4, 5, 6, 7]])
    device_mesh: DeviceMesh = init_device_mesh(
        "cuda",
        (args.dp_replicate_degree, WORLD_SIZE // args.dp_replicate_degree),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )

    main(
        args=args,
        device_mesh=device_mesh,
        distributed_config=HybridShardDistributedConfig(
            device_mesh=device_mesh,
            num_trainers_per_group=args.num_trainers_per_group,
        ),
    )
