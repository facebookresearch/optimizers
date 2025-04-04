"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import os

import torch

from distributed_shampoo.examples.trainer_utils import (
    get_data_loader_and_sampler,
    get_model_and_loss_fn,
    instantiate_optimizer,
    LossMetrics,
    Parser,
    set_seed,
)
from torch import nn

logging.basicConfig(
    format="[%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


###### TRAINING LOOP ######
def train_default_model(
    model: nn.Module,
    loss_function: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    window_size: int = 100,
) -> tuple[float, float, int]:
    """Constructs the main training loop."""

    # initialize metrics
    metrics = LossMetrics(window_size=window_size, device=device)

    # main training loop
    for epoch in range(epochs):
        metrics._epoch = epoch
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            metrics.update(loss)
            metrics.log()

    return (
        metrics._lifetime_loss.item(),
        metrics._window_loss.item(),
        metrics._iteration,
    )


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
    args = Parser.get_args()

    # set seed for reproducibility
    set_seed(args.seed)

    # check cuda availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model and loss function
    model, loss_function = get_model_and_loss_fn(device)

    # instantiate data loader. Note that this is a single GPU training example,
    # so we do not need to instantiate a sampler.
    data_loader, _ = get_data_loader_and_sampler(args.data_path, 1, 0, args.batch_size)

    # instantiate optimizer (SGD, Adam, DistributedShampoo)
    optimizer = instantiate_optimizer(
        args.optimizer_type,
        model,
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
        grafting_epsilon=args.grafting_epsilon,
        grafting_beta2=args.grafting_beta2,
        use_merge_dims=args.use_merge_dims,
        distributed_config=None,
        preconditioner_dtype=args.preconditioner_dtype,
        preconditioner_computation_type=args.preconditioner_computation_type,
    )

    # train model
    train_default_model(
        model,
        loss_function,
        data_loader,
        optimizer,
        device,
        epochs=args.epochs,
        window_size=args.window_size,
    )
