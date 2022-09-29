"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import os
import random
from typing import Tuple

import numpy as np

import torch
from torch import nn
from torchvision import datasets, transforms

try:
    from ai_codesign.optimizers.distributed_shampoo.distributed_shampoo import (
        DistributedShampoo,
    )
    from ai_codesign.optimizers.distributed_shampoo.examples.convnet import ConvNet
    from ai_codesign.optimizers.distributed_shampoo.shampoo_utils import (
        ArgTypeMixin,
        GraftingType,
        LargeDimMethod,
        RootInvStrategy,
    )

except ImportError:
    from convnet import ConvNet

    from ..distributed_shampoo import DistributedShampoo
    from ..shampoo_utils import ArgTypeMixin, GraftingType, LargeDimMethod, RootInvStrategy

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class DType(ArgTypeMixin, enum.Enum):
    FLOAT = 0
    FLOAT64 = 1


class OptimizerType(ArgTypeMixin, enum.Enum):
    SGD = 0
    ADAM = 1
    DISTRIBUTED_SHAMPOO = 2


def instantiate_optimizer(
    optimizer_type: OptimizerType,
    model: nn.Module,
    lr: float,
    betas: Tuple[float, float],
    epsilon: float,
    momentum: float,
    weight_decay: float,
    max_preconditioner_dim: int,
    precondition_frequency: int,
    start_preconditioning_step: int,
    use_nesterov: bool,
    use_bias_correction: bool,
    use_decoupled_weight_decay: bool,
    large_dim_method: LargeDimMethod,
    root_inv_strategy: RootInvStrategy,
    grafting_type: GraftingType,
    grafting_epsilon: float,
    grafting_beta2: float,
) -> torch.optim.Optimizer:
    if optimizer_type == OptimizerType.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=use_nesterov,
        )
    elif optimizer_type == OptimizerType.ADAM:
        if use_decoupled_weight_decay:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=epsilon,
                weight_decay=weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=epsilon,
                weight_decay=weight_decay,
            )
    elif optimizer_type == OptimizerType.DISTRIBUTED_SHAMPOO:
        # since only working with a single GPU, root_inv_strategy = RootInvStrategy.NONE
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=lr,
            betas=betas,
            epsilon=epsilon,
            momentum=momentum,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            use_nesterov=use_nesterov,
            use_bias_correction=use_bias_correction,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            large_dim_method=large_dim_method,
            root_inv_strategy=root_inv_strategy,
            grafting_type=grafting_type,
            grafting_epsilon=grafting_epsilon,
            grafting_beta2=grafting_beta2,
        )
    else:
        raise ValueError(f"Invalid OptimizerType {optimizer_type}!")

    return optimizer


def train_single_gpu_model(
    model: nn.Module,
    loss_function: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 1,
    window_size: int = 100,
) -> Tuple[float, float, int]:
    """Constructs the main training loop."""

    # initialize metrics
    iteration = 0
    window_loss = 0.0
    accumulated_loss = 0.0
    lifetime_loss = 0.0

    # main training loop
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            # print intermediate results
            if iteration % window_size == 0:
                accumulated_loss += window_loss
                window_loss /= window_size
                lifetime_loss = (
                    accumulated_loss / iteration if iteration > 0 else accumulated_loss
                )
                print(
                    f"Epoch: {epoch} | Iteration: {iteration} | Lifetime Loss: {lifetime_loss} | Window Loss: {window_loss}"
                )
                window_loss = 0.0

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            # compute running metrics
            loss = loss.cpu().detach().numpy()
            iteration += 1
            window_loss += loss

    print(
        f"Final Epoch: {epoch} | Iteration: {iteration} | Lifetime Loss: {lifetime_loss} | Window Loss: {window_loss}"
    )

    return lifetime_loss, window_loss, iteration


if __name__ == "__main__":
    """Single GPU CIFAR-10 Example Script

    To run this simple training script, one can run:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        python single_gpu_cifar10_example.py --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        python single_gpu_cifar10_example.py --optimizer-type Adam

    Distributed Shampoo (with default Adam grafting and precondition frequency = 100):
        python single_gpu_cifar10_example.py --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --use-bias-correction --use-decoupled-weight-decay

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """

    import argparse

    parser = argparse.ArgumentParser(description="Arguments for Shampoo run.")

    # arguments for training script
    parser.add_argument(
        "--optimizer-type",
        type=OptimizerType.argtype,
        default="SGD",
        help="Optimizer type.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs.")
    parser.add_argument(
        "--window-size", type=int, default=1, help="Window size for tracking loss."
    )
    parser.add_argument("--seed", type=int, default=2022, help="Seed.")

    # arguments for optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for gradient filtering."
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for exponential moving average of second moment.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-12, help="Epsilon for Adam and Shampoo."
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum parameter for SGD and Shampoo.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--max-preconditioner-dim",
        type=int,
        default=1024,
        help="Max preconditioner dimension for Shampoo.",
    )
    parser.add_argument(
        "--precondition-frequency",
        type=int,
        default=1,
        help="Precondition frequency for Shampoo.",
    )
    parser.add_argument(
        "--start-preconditioning-step",
        type=int,
        default=-1,
        help="Start preconditioning step for Shampoo.",
    )
    parser.add_argument(
        "--use-nesterov",
        action="store_true",
        help="Use Nesterov momentum for SGD and Shampoo.",
    )
    parser.add_argument(
        "--use-bias-correction",
        action="store_false",
        help="Use bias correction for Shampoo.",
    )
    parser.add_argument(
        "--use-decoupled-weight-decay",
        action="store_true",
        help="Use decoupled weight decay for Adam and Shampoo.",
    )
    parser.add_argument(
        "--preconditioner-dtype",
        type=DType.argtype,
        default="FLOAT",
        help="Preconditioner dtype for Shampoo.",
    )
    parser.add_argument(
        "--large-dim-method",
        type=LargeDimMethod.argtype,
        default="BLOCKING",
        help="Large dimensional method for Shampoo.",
    )
    parser.add_argument(
        "--grafting-type",
        type=GraftingType.argtype,
        default="SGD",
        help="Grafted method for Shampoo.",
    )
    parser.add_argument(
        "--grafting-epsilon",
        type=float,
        default=1e-8,
        help="Grafting epsilon parameter for Shampoo.",
    )
    parser.add_argument(
        "--grafting-beta2",
        type=float,
        default=0.999,
        help="Grafting beta2 parameter for Shampoo.",
    )

    args = parser.parse_args()

    # set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # check cuda availability and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # instantiate model and loss function
    model = ConvNet(32, 32, 3).to(device)
    loss_function = nn.CrossEntropyLoss()

    # instantiate data loader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    # instantiate optimizer (SGD, Adam, DistributedShampoo)
    optimizer = instantiate_optimizer(
        args.optimizer_type,
        model,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        epsilon=args.epsilon,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        use_nesterov=args.use_nesterov,
        use_bias_correction=args.use_bias_correction,
        use_decoupled_weight_decay=args.use_decoupled_weight_decay,
        large_dim_method=args.large_dim_method,
        root_inv_strategy=RootInvStrategy.NONE,
        grafting_type=args.grafting_type,
        grafting_epsilon=args.grafting_epsilon,
        grafting_beta2=args.grafting_beta2,
    )

    # train model
    train_single_gpu_model(
        model,
        loss_function,
        data_loader,
        optimizer,
        device,
        epochs=args.epochs,
        window_size=args.window_size,
    )
