"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import argparse
import os

import torch
from distributed_shampoo import DefaultSingleDeviceDistributedConfig
from distributed_shampoo.examples.argument_parser import Parser
from distributed_shampoo.examples.trainer_utils import (
    create_model_and_optimizer_and_loss_fn,
    get_data_loader_and_sampler,
    set_seed,
    train_model,
)
from torch import nn
from torchvision.datasets import VisionDataset

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if __name__ == "__main__":
    """Single GPU CIFAR-10 Training Example Script

    Trains a simple convolutional network with a single GPU.

    Requirements:
        - Python 3.12 or above
        - PyTorch / TorchVision

    To run this simple training script, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        python -m distributed_shampoo.examples.default_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        python -m distributed_shampoo.examples.default_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting and precondition frequency = 100):
        python -m distributed_shampoo.examples.default_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --use-bias-correction --use-decoupled-weight-decay --use-merge-dims

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """

    # parse arguments
    args: argparse.Namespace = Parser.get_args()

    # set seed for reproducibility
    set_seed(args.seed)

    # check cuda availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model and loss function
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_function: nn.Module
    model, optimizer, loss_function = create_model_and_optimizer_and_loss_fn(
        args=args,
        device=device,
        distributed_config=DefaultSingleDeviceDistributedConfig,
    )

    # instantiate data loader. Note that this is a single GPU training example,
    # so we do not need to instantiate a sampler.
    data_loader: torch.utils.data.DataLoader[VisionDataset]
    # type: ignore
    data_loader, _ = get_data_loader_and_sampler(args.data_path, 1, 0, args.batch_size)

    train_model(
        model,
        0,
        loss_function,
        None,
        data_loader,
        optimizer,
        device,
        checkpoint_dir=None,
        epochs=args.epochs,
        window_size=args.window_size,
        metrics_dir=args.metrics_dir,
    )
