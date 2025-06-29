"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import argparse
import logging
import os
from functools import partial
from typing import Any

import torch

import torch.distributed as dist
import torch.distributed.checkpoint as dist_checkpoint

from distributed_shampoo import DDPShampooConfig, DistributedShampoo
from distributed_shampoo.examples.trainer_utils import (
    get_data_loader_and_sampler,
    get_model_and_loss_fn,
    instantiate_optimizer,
    Parser,
    set_seed,
    setup_distribution,
    train_model,
)
from torch import nn
from torchvision.datasets import VisionDataset  # type: ignore[import-untyped]

logging.basicConfig(
    format="[%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
logger: logging.Logger = logging.getLogger(__name__)

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# get local and world rank and world size
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


if __name__ == "__main__":
    """Multi-GPU CIFAR-10 Distributed Data Parallel Training Example Script

    Uses torch.distributed to launch distributed training run.

    Requirements:
        - Python 3.12 or above
        - PyTorch / TorchVision

    To run this training script with a single node, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.ddp_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.ddp_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting, precondition frequency = 100):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.ddp_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --num-trainers-per-group -1 --use-bias-correction --use-decoupled-weight-decay --use-merge-dims

    To use distributed checkpointing, append the flag --use-distributed-checkpoint with optional --checkpoint-dir argument.

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """

    args: argparse.Namespace = Parser.get_args()

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
            nn.parallel.DistributedDataParallel,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK,
        )
        if args.backend == "nccl"
        else nn.parallel.DistributedDataParallel,
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
        exponent_multiplier=args.exponent_multiplier,
        use_nesterov=args.use_nesterov,
        use_bias_correction=args.use_bias_correction,
        use_decoupled_weight_decay=args.use_decoupled_weight_decay,
        grafting_type=args.grafting_type,
        grafting_beta2=args.grafting_beta2,
        grafting_epsilon=args.grafting_epsilon,
        use_merge_dims=args.use_merge_dims,
        distributed_config=DDPShampooConfig(
            communication_dtype=args.communication_dtype,
            num_trainers_per_group=args.num_trainers_per_group,
            communicate_params=args.communicate_params,
        ),
        preconditioner_dtype=args.preconditioner_dtype,
        preconditioner_computation_type=args.preconditioner_computation_type,
    )

    # checks for checkpointing
    if args.use_distributed_checkpoint and not isinstance(
        optimizer, DistributedShampoo
    ):
        raise ValueError(
            "Distributed checkpointing is only supported with DistributedShampoo!"
        )
    if args.use_distributed_checkpoint and args.checkpoint_dir is None:
        raise ValueError(
            "Trying to use distributed checkpointing but checkpoint directory is not provided!"
        )

    # load optimizer and model checkpoint if using Distributed Shampoo optimizer
    if (
        args.use_distributed_checkpoint
        and isinstance(optimizer, DistributedShampoo)
        and os.path.exists(args.checkpoint_dir + "/.metadata")
    ):
        state_dict: dict[str, Any] = {
            "model": model.state_dict(),
            "optim": optimizer.distributed_state_dict(
                key_to_param=model.named_parameters()
            ),
        }
        dist_checkpoint.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_checkpoint.FileSystemReader(args.checkpoint_dir),
        )

        model.load_state_dict(state_dict["model"])
        optimizer.load_distributed_state_dict(
            state_dict["optim"], key_to_param=model.named_parameters()
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
        use_distributed_checkpoint=args.use_distributed_checkpoint,
        metrics_dir=args.metrics_dir if WORLD_RANK == 0 else None,
    )

    # clean up process group
    dist.destroy_process_group()
