"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import argparse
import enum
import importlib
import logging
import random
import shutil
from collections.abc import Callable
from functools import partial
from operator import attrgetter
from pathlib import Path
from typing import overload, Type

import numpy as np

import torch
import torch.distributed as dist

from distributed_shampoo import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultSOAPConfig,
    DistributedConfig,
    DistributedShampoo,
    GraftingConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
    RootInvShampooPreconditionerConfig,
    SGDGraftingConfig,
)
from distributed_shampoo.examples.convnet import ConvNet

from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    EigenConfig,
)

from torch import nn
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.optim.optimizer import ParamsT
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

logger: logging.Logger = logging.getLogger(__name__)

# create default device
default_device = torch.device("cpu")

CIFAR_10_DATASET_FILENAME = "cifar-10-python.tar.gz"


###### ENUM CLASSES ######
@enum.unique
class OptimizerType(enum.Enum):
    SGD = enum.auto()
    ADAM = enum.auto()
    DISTRIBUTED_SHAMPOO = enum.auto()


@enum.unique
class GraftingType(enum.Enum):
    NONE = enum.auto()
    SGD = enum.auto()
    ADAGRAD = enum.auto()
    RMSPROP = enum.auto()
    ADAM = enum.auto()


@enum.unique
class PreconditionerComputationType(enum.Enum):
    EIGEN_ROOT_INV = enum.auto()
    COUPLED_NEWTON_ROOT_INV = enum.auto()
    COUPLED_HIGHER_ORDER_ROOT_INV = enum.auto()
    EIGH_EIGENVALUE_CORRECTION = enum.auto()
    QR_EIGENVALUE_CORRECTION = enum.auto()


###### ARGPARSER ######
def enum_type_parse(s: str, enum_type: Type[enum.Enum]) -> enum.Enum:
    try:
        return enum_type[s]  # type: ignore[index]
    except KeyError:
        raise argparse.ArgumentTypeError(
            "Use one of {}".format(", ".join([t.name for t in enum_type]))  # type: ignore[attr-defined]
        )


class Parser:
    @staticmethod
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Arguments for Shampoo run.")

        # Arguments for training script.
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

        # Arguments for optimizer.
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
            "--beta3",
            type=float,
            default=-1.0,
            help="Beta3 for taking the exponential moving average of the gradient only at the current iteration.",
        )
        parser.add_argument(
            "--epsilon", type=float, default=1e-12, help="Epsilon for Adam and Shampoo."
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=0.0,
            help="Weight decay.",
        )

        # Arguments for Shampoo.
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.0,
            help="Momentum parameter for SGD and Shampoo.",
        )
        parser.add_argument(
            "--dampening",
            type=float,
            default=0.0,
            help="Dampening parameter for SGD and Shampoo in momentum.",
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
            "--inv-root-override",
            type=int,
            default=0,
            help="Inverse root override for Shampoo root inverse.",
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
            "--use-merge-dims",
            action="store_true",
            help="Use merge dims for Shampoo.",
        )
        parser.add_argument(
            "--preconditioner-computation-type",
            type=lambda t: enum_type_parse(t, PreconditionerComputationType),
            default=PreconditionerComputationType.EIGEN_ROOT_INV,
            help="Preconditioner computation method for Shampoo.",
        )

        # Arguments for grafting.
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

        # Arguments for DDP Shampoo.
        parser.add_argument(
            "--communication-dtype",
            type=lambda t: attrgetter(t)(torch),
            default=torch.float32,
            help="Communication dtype for Shampoo.",
        )
        parser.add_argument(
            "--num-trainers-per-group",
            type=int,
            default=-1,
            help="Number of GPUs per distributed process group.",
        )
        parser.add_argument(
            "--communicate-params",
            action="store_true",
            help="Communicate parameters for Shampoo.",
        )

        # Arguments for Distributed Training.
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
            "--data-path",
            type=str,
            default="./data",
            help="Path to CIFAR-10 dataset.",
        )
        parser.add_argument(
            "--checkpoint-dir",
            type=str,
            default=None,
            help="Directory to save checkpoints for DistributedShampoo if this value is not None; otherwise, no checkpoints will be saved.",
        )
        parser.add_argument(
            "--dp-replicate-degree",
            type=int,
            default=2,
            help="Default HSDP replicate degree.",
        )

        # Arguments for metrics logging.
        parser.add_argument(
            "--metrics-dir",
            type=str,
            default=None,
            help="Directory to save metrics logs if set.",
        )

        return parser.parse_args()


###### METRICS CLASSES ######
class LossMetrics:
    def __init__(
        self,
        window_size: int = 100,
        device: torch.device = default_device,
        world_size: int = 0,
        metrics_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._world_size = world_size
        self._window_size = window_size
        self._device = device
        self._epoch = 0
        self._iteration = 0
        self._window_losses: list[torch.Tensor] = []
        self._window_loss: torch.Tensor = torch.tensor(0.0, device=device)
        self._accumulated_loss: torch.Tensor = torch.tensor(0.0, device=device)
        self._lifetime_loss: torch.Tensor = torch.tensor(0.0, device=device)

        if self._world_size > 1:
            self._global_window_loss: torch.Tensor = torch.tensor(0.0, device=device)
            self._global_lifetime_loss: torch.Tensor = torch.tensor(0.0, device=device)

        self._metrics_writer: SummaryWriter | None = (
            SummaryWriter(log_dir=metrics_dir) if metrics_dir else None
        )

    def reset(self) -> None:
        self._epoch = 0
        self._iteration = 0
        self._window_losses = []
        self._window_loss = torch.tensor(0.0, device=self._device)
        self._accumulated_loss = torch.tensor(0.0, device=self._device)
        self._lifetime_loss = torch.tensor(0.0, device=self._device)

    def update(self, loss: torch.Tensor) -> None:
        self._iteration += 1
        self._window_losses.append(loss)
        if len(self._window_losses) > self._window_size:
            self._window_losses.pop(0)
        self._window_loss = torch.mean(torch.stack(self._window_losses))
        self._accumulated_loss += loss
        self._lifetime_loss = self._accumulated_loss / self._iteration

    def log(self) -> None:
        logger.info(
            f"Epoch: {self._epoch} | Iteration: {self._iteration} | Local Lifetime Loss: {self._lifetime_loss} | Local Window Loss: {self._window_loss}"
        )
        if self._metrics_writer is not None:
            self._metrics_writer.add_scalars(
                "Local Loss",
                {"Lifetime": self._lifetime_loss, "Window": self._window_loss},
                self._iteration,
            )

    def update_global_metrics(self) -> None:
        if dist.is_initialized() and self._world_size > 1:
            self._global_window_loss = self._window_loss / self._world_size
            self._global_lifetime_loss = self._lifetime_loss / self._world_size
            dist.all_reduce(self._global_window_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._global_lifetime_loss, op=dist.ReduceOp.SUM)

    def log_global_metrics(self) -> None:
        if self._world_size > 1:
            logger.info(
                f"Epoch: {self._epoch} | Iteration: {self._iteration} | Global Lifetime Loss: {self._global_lifetime_loss} | Global Window Loss: {self._global_window_loss}"
            )
            if self._metrics_writer is not None:
                self._metrics_writer.add_scalars(
                    "Global Loss",
                    {
                        "Lifetime": self._global_lifetime_loss,
                        "Window": self._global_window_loss,
                    },
                    self._iteration,
                )

    def flush(self) -> None:
        if self._metrics_writer is not None:
            self._metrics_writer.flush()


###### OPTIMIZER INSTANTIATION ######
def instantiate_optimizer(
    optimizer_type: OptimizerType,
    parameters: ParamsT,
    lr: float,
    betas: tuple[float, float],
    beta3: float,
    epsilon: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    max_preconditioner_dim: int,
    precondition_frequency: int,
    start_preconditioning_step: int,
    use_nesterov: bool,
    use_bias_correction: bool,
    use_decoupled_weight_decay: bool,
    grafting_type: GraftingType,
    grafting_beta2: float,
    grafting_epsilon: float,
    use_merge_dims: bool,
    distributed_config: DistributedConfig | None,
    preconditioner_computation_type: PreconditionerComputationType,
) -> torch.optim.Optimizer:
    if optimizer_type == OptimizerType.SGD:
        optimizer_cls: Callable[..., torch.optim.Optimizer] = partial(
            torch.optim.SGD,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=use_nesterov,
        )
    elif optimizer_type == OptimizerType.ADAM:
        optimizer_cls = partial(
            torch.optim.AdamW if use_decoupled_weight_decay else torch.optim.Adam,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
        )
    elif optimizer_type == OptimizerType.DISTRIBUTED_SHAMPOO:
        optimizer_cls = partial(
            DistributedShampoo,
            betas=betas,
            beta3=beta3,
            epsilon=epsilon,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            use_nesterov=use_nesterov,
            use_bias_correction=use_bias_correction,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            grafting_config=instantiate_grafting_config(
                grafting_type, grafting_beta2, grafting_epsilon
            ),
            use_merge_dims=use_merge_dims,
            distributed_config=distributed_config,
            preconditioner_config=instantiate_preconditioner_config(
                preconditioner_computation_type=preconditioner_computation_type,
            ),
        )
    else:
        raise ValueError(f"Invalid OptimizerType {optimizer_type}!")

    return optimizer_cls(parameters, lr=lr)


def instantiate_grafting_config(
    grafting_type: GraftingType,
    grafting_beta2: float,
    grafting_epsilon: float,
) -> GraftingConfig | None:
    if grafting_type == GraftingType.NONE:
        return None
    elif grafting_type == GraftingType.SGD:
        return SGDGraftingConfig()  # type: ignore[abstract]
    elif grafting_type == GraftingType.ADAGRAD:
        return AdaGradGraftingConfig(
            epsilon=grafting_epsilon,
        )
    elif grafting_type == GraftingType.RMSPROP:
        return RMSpropGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    elif grafting_type == GraftingType.ADAM:
        return AdamGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    else:
        raise ValueError(f"Invalid GraftingType {grafting_type}!")


def instantiate_preconditioner_config(
    preconditioner_computation_type: PreconditionerComputationType,
) -> PreconditionerConfig:
    if preconditioner_computation_type == PreconditionerComputationType.EIGEN_ROOT_INV:
        return RootInvShampooPreconditionerConfig(
            amortized_computation_config=EigenConfig()
        )
    elif (
        preconditioner_computation_type
        == PreconditionerComputationType.COUPLED_NEWTON_ROOT_INV
    ):
        return RootInvShampooPreconditionerConfig(
            amortized_computation_config=CoupledNewtonConfig(),
        )
    elif (
        preconditioner_computation_type
        == PreconditionerComputationType.COUPLED_HIGHER_ORDER_ROOT_INV
    ):
        return RootInvShampooPreconditionerConfig(
            amortized_computation_config=CoupledHigherOrderConfig(
                rel_epsilon=0.0, abs_epsilon=0.0
            ),
        )
    elif (
        preconditioner_computation_type
        == PreconditionerComputationType.EIGH_EIGENVALUE_CORRECTION
    ):
        return DefaultEigenvalueCorrectedShampooConfig
    elif (
        preconditioner_computation_type
        == PreconditionerComputationType.QR_EIGENVALUE_CORRECTION
    ):
        return DefaultSOAPConfig
    else:
        raise ValueError(
            f"Invalid PreconditionerComputationType {preconditioner_computation_type}!"
        )


###### DATA LOADER ######
def get_data_loader_and_sampler(
    data_path: Path, world_size: int, rank: int, local_batch_size: int
) -> tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.distributed.DistributedSampler[torch.utils.data.Dataset],
]:
    # instantiate data loader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    data_path = Path(data_path) / str(rank)
    # If data is available as a packaged resource, skip download and use it directly.
    with importlib.resources.path(
        __package__, CIFAR_10_DATASET_FILENAME
    ) as resource_path:
        if resource_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(resource_path, data_path)

    dataset = datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform
    )
    sampler: torch.utils.data.distributed.DistributedSampler[
        torch.utils.data.Dataset
    ] = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            num_workers=2,
        ),
        sampler,
    )


###### SET UP ######
def set_seed(seed: int) -> None:
    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)


def setup_distribution(
    backend: str, world_rank: int, world_size: int, local_rank: int
) -> torch.device:
    # initialize distributed process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=world_rank,
        world_size=world_size,
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", local_rank)

    if use_cuda:
        # Necessary to ensure DTensor's local tensors are instantiated
        # on the correct device.
        #
        # TODO: DTensor zeros instantiation needs to be fixed.
        torch.cuda.set_device(local_rank)

    return device


@overload
def get_model_and_loss_fn(
    device: torch.device,
    post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
) -> tuple[nn.Module, nn.Module]: ...


@overload
def get_model_and_loss_fn(
    device: torch.device,
    post_model_decoration: Callable[[nn.Module], FSDPModule] = lambda x: fully_shard(x),
) -> tuple[FSDPModule, nn.Module]: ...


def get_model_and_loss_fn(
    device: torch.device,
    post_model_decoration: Callable[[nn.Module], nn.Module | FSDPModule] = lambda x: x,
) -> tuple[nn.Module | FSDPModule, nn.Module]:
    """
    Creates and returns a model and loss function for training.

    Args:
        device (torch.device): The device (CPU/GPU) where the model should be placed.
        post_model_decoration (Callable[[nn.Module], nn.Module | FSDPModule]): Optional function to apply additional modifications to the model after creation (e.g., for distributed training). (Default: identity function)

    Returns:
        model (nn.Module | FSDPModule): The instantiated ConvNet model moved to the specified device and with any post-decoration applied.
        loss_fn (nn.Module): The CrossEntropyLoss function for training.
    """
    # instantiate model and loss function
    model = ConvNet(height=32, width=32).to(device)
    loss_fn = nn.CrossEntropyLoss()

    return post_model_decoration(model), loss_fn


###### TRAIN LOOP ######
def train_model(
    model: nn.Module,
    world_size: int,
    loss_function: nn.Module,
    sampler: torch.utils.data.Sampler,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_dir: str,
    epochs: int = 1,
    window_size: int = 100,
    local_rank: int = 0,
    metrics_dir: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # initialize metrics
    metrics = LossMetrics(
        window_size=window_size,
        device=device,
        world_size=world_size,
        metrics_dir=metrics_dir,
    )

    # main training loop
    for epoch in range(epochs):
        metrics._epoch = epoch
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(epoch)

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()

            optimizer.step()
            metrics.update(loss.detach())
            metrics.log()
            metrics.update_global_metrics()
            if local_rank == 0:
                metrics.log_global_metrics()

    # checkpoint optimizer and model using distributed checkpointing solution
    if checkpoint_dir is not None and isinstance(optimizer, DistributedShampoo):
        state_dict = {
            "model": model.state_dict(),
            "optim": optimizer.distributed_state_dict(
                key_to_param=model.named_parameters()
            ),
        }
        dist_checkpoint.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_checkpoint.FileSystemWriter(checkpoint_dir),
        )

    metrics.flush()
    return metrics._lifetime_loss, metrics._window_loss, metrics._iteration
