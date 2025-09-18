"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import argparse
import os
from collections.abc import Callable
from functools import partial

import torch

import torch.distributed as dist

from distributed_shampoo import DistributedConfig, HSDPDistributedConfig
from distributed_shampoo.distributor.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from distributed_shampoo.examples.argument_parser import Parser
from distributed_shampoo.examples.trainer_utils import (
    get_data_loader_and_sampler,
    get_model_and_loss_fn,
    instantiate_optimizer,
    set_seed,
    setup_distribution,
    train_model,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
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
    partial_distributed_config: Callable[..., DistributedConfig],
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

    # instantiate model and loss function
    model: nn.Module
    loss_function: nn.Module
    model, loss_function = get_model_and_loss_fn(
        device=device,
        post_model_decoration=partial(
            FSDP,
            device_mesh=device_mesh,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
        ),
    )

    # instantiate data loader
    data_loader: torch.utils.data.DataLoader[VisionDataset]
    sampler: torch.utils.data.distributed.DistributedSampler[
        torch.utils.data.Dataset[VisionDataset]
    ]
    data_loader, sampler = get_data_loader_and_sampler(
        args.data_path, WORLD_SIZE, WORLD_RANK, args.local_batch_size
    )

    # instantiate optimizer (SGD, Adam, DistributedShampoo)
    optimizer: torch.optim.Optimizer = instantiate_optimizer(
        args.optimizer_type,
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        beta3=args.beta3,
        epsilon=args.epsilon,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_nesterov=args.use_nesterov,
        use_bias_correction=args.use_bias_correction,
        use_decoupled_weight_decay=args.use_decoupled_weight_decay,
        grafting_type=args.grafting_type,
        grafting_epsilon=args.grafting_epsilon,
        grafting_beta2=args.grafting_beta2,
        distributed_config=partial_distributed_config(
            param_to_metadata=compile_fsdp_parameter_metadata(model)
        ),
        preconditioner_computation_type=args.preconditioner_computation_type,
    )

    # train model
    train_model(
        model,
        WORLD_SIZE,
        loss_function,
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
    """Multi-GPU CIFAR-10 Distributed Data Parallel Training Example Script

    Uses torch.distributed to launch distributed training run.

    Requirements:
        - Python 3.12 or above
        - PyTorch / TorchVision

    To run this training script with a single node, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hsdp_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hsdp_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting, precondition frequency = 100):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hsdp_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --num-trainers-per-group 2 --use-bias-correction --use-decoupled-weight-decay --use-merge-dims

    To use distributed checkpointing on Distributed Shampoo, append the flag with --checkpoint-dir argument.

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """
    args: argparse.Namespace = Parser.get_args()
    # Instantiate device mesh for HSDP Shampoo.
    # For example, with 8 GPUs and dp_replicate_degree set to 2, the device mesh will be:
    # ([[0, 1, 2, 3], [4, 5, 6, 7]])
    device_mesh: DeviceMesh = init_device_mesh(
        "cuda", (args.dp_replicate_degree, WORLD_SIZE // args.dp_replicate_degree)
    )

    main(
        args=args,
        device_mesh=device_mesh,
        partial_distributed_config=partial(
            HSDPDistributedConfig,
            device_mesh=device_mesh,
            num_trainers_per_group=args.num_trainers_per_group,
        ),
    )
