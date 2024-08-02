"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import argparse
import enum
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    CommunicationDType,
    DistributedConfig,
    GraftingConfig,
    PrecisionConfig,
    RMSpropGraftingConfig,
    SGDGraftingConfig,
)
from torch import nn

logger = logging.getLogger(__name__)

# create default device
default_device = torch.device("cpu")


###### ENUM CLASSES ######
class DType(enum.Enum):
    BF16 = torch.bfloat16
    FP16 = torch.float16
    FP32 = torch.float32
    FP64 = torch.float64


class OptimizerType(enum.Enum):
    SGD = 0
    ADAM = 1
    DISTRIBUTED_SHAMPOO = 2


class GraftingType(enum.Enum):
    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    ADAM = 4


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
            "--exponent-multiplier",
            type=float,
            default=1.0,
            help="Exponent multiplier for Shampoo root inverse.",
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
            "--use-pytorch-compile",
            action="store_true",
            help="Use PyTorch compile for Shampoo.",
        )
        parser.add_argument(
            "--use-protected-eigh",
            action="store_true",
            help="Uses protected eigendecomposition.",
        )
        parser.add_argument(
            "--track-root-inv-residuals",
            action="store_true",
            help="Use debug mode for examining root inverse residuals.",
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

        # Arguments for mixed-precision.
        parser.add_argument(
            "--computation-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FP32,
            help="Data type for all computation in Shampoo.",
        )
        parser.add_argument(
            "--factor-matrix-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FP32,
            help="Data type for storing Shampoo factor matrices.",
        )
        parser.add_argument(
            "--inv-factor-matrix-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FP32,
            help="Data type for storing Shampoo inverse factor matrices.",
        )
        parser.add_argument(
            "--filtered-grad-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FP32,
            help="Data type for storing filtered gradients.",
        )
        parser.add_argument(
            "--momentum-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FP32,
            help="Data type for storing momentum states.",
        )
        parser.add_argument(
            "--grafting-state-dtype",
            type=lambda t: enum_type_parse(t, DType),
            default=DType.FP32,
            help="Data type for storing grafting preconditioners.",
        )

        # Arguments for DDP Shampoo.
        parser.add_argument(
            "--communication-dtype",
            type=lambda t: enum_type_parse(t, CommunicationDType),
            default=CommunicationDType.FP32,
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
    @abstractmethod
    def log(self): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self): ...


class LossMetrics(Metrics):
    def __init__(
        self,
        window_size: int = 100,
        device: torch.device = default_device,
        world_size: int = 0,
    ):
        super().__init__()
        self._world_size = world_size
        self._window_size = window_size
        self._device = device
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
        self._window_loss = torch.tensor(0.0, device=self._device)
        self._accumulated_loss = torch.tensor(0.0, device=self._device)
        self._lifetime_loss = torch.tensor(0.0, device=self._device)

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
    beta3: float,
    epsilon: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    max_preconditioner_dim: int,
    precondition_frequency: int,
    start_preconditioning_step: int,
    inv_root_override: int,
    exponent_multiplier: float,
    use_nesterov: bool,
    use_bias_correction: bool,
    use_decoupled_weight_decay: bool,
    grafting_type: GraftingType,
    grafting_beta2: float,
    grafting_epsilon: float,
    use_merge_dims: bool,
    use_pytorch_compile: bool,
    distributed_config: Optional[DistributedConfig],
    precision_config: Optional[PrecisionConfig],
    use_protected_eigh: bool,
    track_root_inv_residuals: bool,
) -> torch.optim.Optimizer:
    if optimizer_type == OptimizerType.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
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
            beta3=beta3,
            epsilon=epsilon,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            max_preconditioner_dim=max_preconditioner_dim,
            precondition_frequency=precondition_frequency,
            start_preconditioning_step=start_preconditioning_step,
            inv_root_override=inv_root_override,
            exponent_multiplier=exponent_multiplier,
            use_nesterov=use_nesterov,
            use_bias_correction=use_bias_correction,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            grafting_config=instantiate_grafting_config(
                grafting_type, grafting_beta2, grafting_epsilon
            ),
            use_merge_dims=use_merge_dims,
            use_pytorch_compile=use_pytorch_compile,
            distributed_config=distributed_config,
            precision_config=precision_config,
            use_protected_eigh=use_protected_eigh,
            track_root_inv_residuals=track_root_inv_residuals,
        )
    else:
        raise ValueError(f"Invalid OptimizerType {optimizer_type}!")

    return optimizer


def instantiate_grafting_config(
    grafting_type: GraftingType,
    grafting_beta2: float,
    grafting_epsilon: float,
) -> Optional[GraftingConfig]:
    if grafting_type == GraftingType.NONE:
        return None
    elif grafting_type == GraftingType.ADAGRAD:
        return AdaGradGraftingConfig(
            epsilon=grafting_epsilon,
        )
    elif grafting_type == GraftingType.ADAM:
        return AdamGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    elif grafting_type == GraftingType.RMSPROP:
        return RMSpropGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    elif grafting_type == GraftingType.SGD:
        return SGDGraftingConfig(
            beta2=grafting_beta2,
            epsilon=grafting_epsilon,
        )
    else:
        raise ValueError(f"Invalid GraftingType {grafting_type}!")
