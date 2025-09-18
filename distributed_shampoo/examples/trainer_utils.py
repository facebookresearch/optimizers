"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import argparse
import importlib
import logging
import random
import shutil
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import overload

import numpy as np

import torch
import torch.distributed as dist

from distributed_shampoo import (
    AdaGradPreconditionerConfig,
    AdamPreconditionerConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultSOAPConfig,
    DistributedConfig,
    DistributedShampoo,
    PreconditionerConfig,
    RMSpropPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    SGDPreconditionerConfig,
)
from distributed_shampoo.examples.argument_parser import (
    OptimizerType,
    PreconditionerComputationType,
)
from distributed_shampoo.examples.convnet import ConvNet
from distributed_shampoo.examples.loss_metrics import LossMetrics

from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    EigenConfig,
)

from torch import nn
from torch.distributed import checkpoint as dist_checkpoint
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.optim.optimizer import ParamsT
from torchvision import datasets, transforms

logger: logging.Logger = logging.getLogger(__name__)

CIFAR_10_DATASET_FILENAME = "cifar-10-python.tar.gz"


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
    grafting_type: PreconditionerComputationType,
    grafting_beta2: float,
    grafting_epsilon: float,
    distributed_config: DistributedConfig,
    preconditioner_computation_type: PreconditionerComputationType,
) -> torch.optim.Optimizer:
    def instantiate_preconditioner_config(
        preconditioner_computation_type: PreconditionerComputationType,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 0.0,
    ) -> PreconditionerConfig | None:
        if preconditioner_computation_type == PreconditionerComputationType.NONE:
            return None
        elif preconditioner_computation_type == PreconditionerComputationType.SGD:
            return SGDPreconditionerConfig()  # type: ignore[abstract]
        elif preconditioner_computation_type == PreconditionerComputationType.ADAGRAD:
            return AdaGradPreconditionerConfig(
                epsilon=grafting_epsilon,
            )
        elif preconditioner_computation_type == PreconditionerComputationType.RMSPROP:
            return RMSpropPreconditionerConfig(
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        elif preconditioner_computation_type == PreconditionerComputationType.ADAM:
            return AdamPreconditionerConfig(
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        elif (
            preconditioner_computation_type
            == PreconditionerComputationType.EIGEN_ROOT_INV
        ):
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
            raise ValueError(f"Invalid {preconditioner_computation_type=}!")

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
        assert (
            preconditioner_config := instantiate_preconditioner_config(
                preconditioner_computation_type=preconditioner_computation_type,
            )
        ) is not None
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
            grafting_config=instantiate_preconditioner_config(
                preconditioner_computation_type=grafting_type,
                grafting_beta2=grafting_beta2,
                grafting_epsilon=grafting_epsilon,
            ),
            distributed_config=distributed_config,
            preconditioner_config=preconditioner_config,
        )
    else:
        raise ValueError(f"Invalid OptimizerType {optimizer_type}!")

    return optimizer_cls(parameters, lr=lr)


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
    sampler: torch.utils.data.Sampler | None,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_dir: str | None,
    epochs: int = 1,
    window_size: int = 100,
    local_rank: int | None = 0,
    metrics_dir: str | None = None,
) -> tuple[float, float, int]:
    # initialize metrics
    metrics = LossMetrics(
        window_size=window_size,
        device=device,
        world_size=world_size,
        metrics_dir=metrics_dir,
    )

    # main training loop
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
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
    return (
        metrics._lifetime_loss.item(),
        metrics._window_loss.item(),
        metrics._iteration,
    )


def create_model_and_optimizer_and_loss_fn(
    args: argparse.Namespace,
    device: torch.device,
    distributed_config: DistributedConfig,
    post_model_decoration: Callable[[nn.Module], nn.Module | FSDPModule] = lambda x: x,
) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    # instantiate model and loss function
    model, loss_function = get_model_and_loss_fn(
        device=device,
        post_model_decoration=post_model_decoration,  # type: ignore
    )
    assert isinstance(model, nn.Module)

    # instantiate optimizer (SGD, Adam, DistributedShampoo)
    optimizer = instantiate_optimizer(
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
        distributed_config=distributed_config,
        preconditioner_computation_type=args.preconditioner_computation_type,
    )
    return model, optimizer, loss_function
