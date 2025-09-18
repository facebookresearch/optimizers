"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import argparse
import enum
from operator import attrgetter
from typing import Type

import torch

from distributed_shampoo import FSDPParamAssignmentStrategy


###### ENUM CLASSES ######
@enum.unique
class OptimizerType(enum.Enum):
    SGD = enum.auto()
    ADAM = enum.auto()
    DISTRIBUTED_SHAMPOO = enum.auto()


@enum.unique
class PreconditionerComputationType(enum.Enum):
    NONE = enum.auto()
    SGD = enum.auto()
    ADAGRAD = enum.auto()
    RMSPROP = enum.auto()
    ADAM = enum.auto()
    EIGEN_ROOT_INV = enum.auto()
    COUPLED_NEWTON_ROOT_INV = enum.auto()
    COUPLED_HIGHER_ORDER_ROOT_INV = enum.auto()
    EIGH_EIGENVALUE_CORRECTION = enum.auto()
    QR_EIGENVALUE_CORRECTION = enum.auto()


###### ARGPARSER ######
class Parser:
    @staticmethod
    def _enum_type_parse(s: str, enum_type: Type[enum.Enum]) -> enum.Enum:
        try:
            return enum_type[s]  # type: ignore[index]
        except KeyError:
            raise argparse.ArgumentTypeError(
                "Use one of {}".format(", ".join([t.name for t in enum_type]))  # type: ignore[attr-defined]
            )

    @staticmethod
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Arguments for Shampoo run.")

        # Arguments for training script.
        parser.add_argument(
            "--optimizer-type",
            type=lambda t: Parser._enum_type_parse(t, OptimizerType),
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
            type=lambda t: Parser._enum_type_parse(t, PreconditionerComputationType),
            default=PreconditionerComputationType.EIGEN_ROOT_INV,
            help="Preconditioner computation method for Shampoo.",
        )

        # Arguments for grafting.
        parser.add_argument(
            "--grafting-type",
            type=lambda t: Parser._enum_type_parse(t, PreconditionerComputationType),
            default=PreconditionerComputationType.SGD,
            help="Grafting method for Shampoo.",
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
        parser.add_argument(
            "--param-assignment-strategy",
            type=lambda t: Parser._enum_type_parse(t, FSDPParamAssignmentStrategy),
            default=FSDPParamAssignmentStrategy.DEFAULT,
            help="Parameter assignment strategy in FSDP / HSDP distributor.",
        )

        # Arguments for metrics logging.
        parser.add_argument(
            "--metrics-dir",
            type=str,
            default=None,
            help="Directory to save metrics logs if set.",
        )

        return parser.parse_args()
