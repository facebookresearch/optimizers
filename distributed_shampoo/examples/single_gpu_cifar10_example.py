"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import argparse
import enum
import logging
import os
import random
from abc import ABC
from typing import Tuple

import numpy as np

import torch
import torch.distributed as dist

from distributed_shampoo.distributed_shampoo import DistributedShampoo

from distributed_shampoo.examples.convnet import ConvNet
from distributed_shampoo.utils.shampoo_utils import GraftingType, LargeDimMethod
from torch import nn
from torchvision import datasets, transforms

logging.basicConfig(
    format="[%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


###### ENUM CLASSES ######
class DType(enum.Enum):
    FLOAT = 0
    FLOAT64 = 1


class OptimizerType(enum.Enum):
    SGD = 0
    ADAM = 1
    DISTRIBUTED_SHAMPOO = 2


###### ARGPARSER ######
def enum_type_parse(s: str, enum_type: enum.Enum):
    try:
        return enum_type[s]
    except KeyError:
        raise argparse.ArgumentTypeError(
            "Use one of {}".format(", ".join([t.name for t in enum_type]))
        )


class Parser:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="Arguments for Shampoo run.")

        # arguments for training script
        parser.add_argument(
            "--optimizer-type",
            type=lambda t: enum_type_parse(t, OptimizerType),
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
            "--exponent-override",
            type=int,
            default=0,
            help="Exponent override for Shampoo root inverse.",
        )
        parser.add_argument(
            "--use-nesterov",
            action="store_true",
            help="Use Nesterov momentum for SGD and Shampoo.",
        )
        parser.add_argument(
            "--use-bias-correction",
            action="store_true",
            help="Use bias correction for Shampoo.",
        )
        parser.add_argument(
            "--use-decoupled-weight-decay",
            action="store_true",
            help="Use decoupled weight decay for Adam and Shampoo.",
        )
        parser.add_argument(
            "--preconditioner-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FLOAT,
            help="Preconditioner dtype for Shampoo.",
        )
        parser.add_argument(
            "--large-dim-method",
            type=lambda t: enum_type_parse(t, LargeDimMethod),
            default=LargeDimMethod.BLOCKING,
            help="Large dimensional method for Shampoo.",
        )
        parser.add_argument(
            "--grafting-type",
            type=lambda t: enum_type_parse(t, GraftingType),
            default=GraftingType.SGD,
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
        parser.add_argument(
            "--use-protected-eigh",
            action="store_true",
            help="Uses protected eigendecomposition.",
        )
        parser.add_argument(
            "--use-dtensor", action="store_true", help="Use DTensor if available."
        )
        parser.add_argument(
            "--debug-mode",
            action="store_true",
            help="Use debug mode for examining root inverse residuals.",
        )

        # arguments for distributed training
        # not used if using single GPU training
        parser.add_argument(
            "--local-batch-size", type=int, default=128, help="Local batch size."
        )
        parser.add_argument(
            "--num-trainers", type=int, default=2, help="Number of trainers."
        )
        parser.add_argument(
            "--backend",
            type=str,
            default="nccl",
            choices=["nccl", "gloo"],
            help="Distributed backend.",
        )
        parser.add_argument(
            "--num-trainers-per-group",
            type=int,
            default=-1,
            help="Number of GPUs per distributed process group.",
        )
        parser.add_argument(
            "--data-path",
            type=str,
            default="./data",
            help="Path to CIFAR-10 dataset.",
        )
        parser.add_argument(
            "--use-distributed-checkpoint",
            action="store_true",
            help="Toggle distributed checkpoint testing.",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default="./checkpoints",
            help="Directory to save checkpoints and logs.",
        )

        return parser.parse_args()


###### METRICS CLASSES ######
class Metrics(ABC):
    def log(self):
        pass

    def reset(self):
        pass

    def update(self):
        pass


class LossMetrics(Metrics):
    def __init__(
        self,
        window_size: int = 100,
        device: torch.device = torch.device("cpu"),
        world_size: int = 0,
    ):
        super().__init__()
        self._world_size = world_size
        self._window_size = window_size
        self._epoch = 0
        self._iteration = 0
        self._window_losses = []
        self._window_loss = torch.tensor(0.0, device=device)
        self._accumulated_loss = torch.tensor(0.0, device=device)
        self._lifetime_loss = torch.tensor(0.0, device=device)

        if self._world_size > 1:
            self._global_window_loss = torch.tensor(0.0, device=device)
            self._global_lifetime_loss = torch.tensor(0.0, device=device)

    def reset(self):
        self._epoch = 0
        self._iteration = 0
        self._window_losses = []
        self._window_loss = torch.tensor(0.0, device=device)
        self._accumulated_loss = torch.tensor(0.0, device=device)
        self._lifetime_loss = torch.tensor(0.0, device=device)

    def update(self, loss: torch.Tensor):
        self._iteration += 1
        self._window_losses.append(loss)
        if len(self._window_losses) > self._window_size:
            self._window_losses.pop(0)
        self._window_loss = torch.mean(torch.stack(self._window_losses))
        self._accumulated_loss += loss
        self._lifetime_loss = self._accumulated_loss / self._iteration

    def log(self):
        logger.info(
            f"Epoch: {self._epoch} | Iteration: {self._iteration} | Local Lifetime Loss: {self._lifetime_loss} | Local Window Loss: {self._window_loss}"
        )

    def update_global_metrics(self):
        if dist.is_initialized() and self._world_size > 1:
            self._global_window_loss = self._window_loss / self._world_size
            self._global_lifetime_loss = self._lifetime_loss / self._world_size
            dist.all_reduce(self._global_window_loss, op=dist.reduce_op.SUM)
            dist.all_reduce(self._global_lifetime_loss, op=dist.reduce_op.SUM)
        else:
            pass

    def log_global_metrics(self):
        if self._world_size > 1:
            logger.info(
                f"Epoch: {self._epoch} | Iteration: {self._iteration} | Global Lifetime Loss: {self._global_lifetime_loss} | Global Window Loss: {self._global_window_loss}"
            )
        else:
            pass


###### OPTIMIZER INSTANTIATION ######
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
    exponent_override: int,
    use_nesterov: bool,
    use_bias_correction: bool,
    use_decoupled_weight_decay: bool,
    preconditioner_dtype: DType,
    large_dim_method: LargeDimMethod,
    num_trainers_per_group: int,
    grafting_type: GraftingType,
    grafting_epsilon: float,
    grafting_beta2: float,
    use_protected_eigh: bool,
    use_dtensor: bool,
    debug_mode: bool,
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
            exponent_override=exponent_override,
            use_nesterov=use_nesterov,
            use_bias_correction=use_bias_correction,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            preconditioner_dtype=preconditioner_dtype,
            large_dim_method=large_dim_method,
            num_trainers_per_group=num_trainers_per_group,
            grafting_type=grafting_type,
            grafting_epsilon=grafting_epsilon,
            grafting_beta2=grafting_beta2,
            use_protected_eigh=use_protected_eigh,
            use_dtensor=use_dtensor,
            debug_mode=debug_mode,
        )
    else:
        raise ValueError(f"Invalid OptimizerType {optimizer_type}!")

    return optimizer


###### TRAINING LOOP ######
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

    return metrics._lifetime_loss, metrics._window_loss, metrics._iteration


if __name__ == "__main__":
    """Single GPU CIFAR-10 Training Example Script

    Trains a simple convolutional network with a single GPU.

    Requirements:
        - Python 3.8 or above
        - PyTorch / TorchVision

    To run this simple training script, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        python -m distributed_shampoo.examples.single_gpu_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        python -m distributed_shampoo.examples.single_gpu_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting and precondition frequency = 100):
        python -m distributed_shampoo.examples.single_gpu_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --use-bias-correction --use-decoupled-weight-decay

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """

    # parse arguments
    args = Parser.get_args()

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
    dataset = datasets.CIFAR10(
        args.data_path, train=True, download=True, transform=transform
    )
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
        exponent_override=args.exponent_override,
        use_nesterov=args.use_nesterov,
        use_bias_correction=args.use_bias_correction,
        use_decoupled_weight_decay=args.use_decoupled_weight_decay,
        preconditioner_dtype=torch.float
        if args.preconditioner_dtype == DType.FLOAT
        else torch.float64,
        large_dim_method=args.large_dim_method,
        num_trainers_per_group=0,
        grafting_type=args.grafting_type,
        grafting_epsilon=args.grafting_epsilon,
        grafting_beta2=args.grafting_beta2,
        use_protected_eigh=args.use_protected_eigh,
        use_dtensor=args.use_dtensor,
        debug_mode=args.debug_mode,
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
