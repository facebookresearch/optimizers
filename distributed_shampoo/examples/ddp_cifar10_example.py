"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import os
import random
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_checkpoint

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.examples.convnet import ConvNet
from distributed_shampoo.examples.trainer_utils import (
    instantiate_optimizer,
    LossMetrics,
    Parser,
)
from distributed_shampoo.shampoo_types import DDPShampooConfig, PrecisionConfig
from torch import nn

from torchvision import datasets, transforms

logging.basicConfig(
    format="[%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# for reproducibility, set environmental variable for CUBLAS
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# get local and world rank and world size
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def train_ddp_model(
    model: nn.Module,
    world_size: int,
    loss_function: nn.Module,
    sampler: torch.utils.data.Sampler,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device],
    epochs: int = 1,
    window_size: int = 100,
    use_distributed_checkpoint: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Tuple[float, float, int]:
    """Constructs the main training loop.

    Assumes torch.distributed is initialized.

    """

    # checks for checkpointing
    if use_distributed_checkpoint and not isinstance(optimizer, DistributedShampoo):
        raise ValueError(
            "Distributed checkpointing is only supported with DistributedShampoo!"
        )
    if use_distributed_checkpoint and checkpoint_dir is None:
        raise ValueError(
            "Trying to use distributed checkpointing but checkpoint directory is not provided!"
        )

    # initialize metrics
    metrics = LossMetrics(window_size=window_size, device=device, world_size=world_size)

    # load optimizer and model checkpoint if using Distributed Shampoo optimizer
    if (
        use_distributed_checkpoint
        and isinstance(optimizer, DistributedShampoo)
        and os.path.exists(checkpoint_dir + "/.metadata")
    ):
        state_dict = {
            "model": model.state_dict(),
            "optim": optimizer.distributed_state_dict(
                key_to_param=model.named_parameters()
            ),
        }
        dist_checkpoint.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_checkpoint.FileSystemReader(checkpoint_dir),
        )

        model.load_state_dict(state_dict["model"])
        optimizer.load_distributed_state_dict(
            state_dict["optim"], key_to_param=model.named_parameters()
        )

    # main training loop
    for epoch in range(epochs):
        metrics._epoch = epoch
        sampler.set_epoch(epoch)

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = loss_function(output, labels)
            loss.backward()

            optimizer.step()
            metrics.update(loss)
            metrics.log()
            metrics.update_global_metrics()
            if LOCAL_RANK == 0:
                metrics.log_global_metrics()

    # checkpoint optimizer and model using distributed checkpointing solution
    if use_distributed_checkpoint and isinstance(optimizer, DistributedShampoo):
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

    return metrics._lifetime_loss, metrics._window_loss, metrics._iteration


if __name__ == "__main__":
    """Multi-GPU CIFAR-10 Distributed Data Parallel Training Example Script

    Uses torch.distributed to launch distributed training run.

    Requirements:
        - Python 3.8 or above
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

    args = Parser.get_args()

    # set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # initialize distributed process group
    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
    )
    device = torch.device("cuda:{}".format(LOCAL_RANK))

    # Necessary to ensure DTensor's local tensors are instantiated
    # on the correct device.
    #
    # TODO: DTensor zeros instantiation needs to be fixed.
    torch.cuda.set_device(LOCAL_RANK)

    # instantiate model and loss function
    model = ConvNet(32, 32, 3).to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
    )
    loss_function = nn.CrossEntropyLoss()

    # instantiate data loader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.CIFAR10(
        args.data_path, train=True, download=True, transform=transform
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=WORLD_RANK, shuffle=True
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.local_batch_size,
        sampler=sampler,
        num_workers=2,
    )

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
        inv_root_override=args.inv_root_override,
        exponent_multiplier=args.exponent_multiplier,
        use_nesterov=args.use_nesterov,
        use_bias_correction=args.use_bias_correction,
        use_decoupled_weight_decay=args.use_decoupled_weight_decay,
        grafting_type=args.grafting_type,
        grafting_beta2=args.grafting_beta2,
        grafting_epsilon=args.grafting_epsilon,
        use_merge_dims=args.use_merge_dims,
        use_pytorch_compile=args.use_pytorch_compile,
        distributed_config=DDPShampooConfig(
            communication_dtype=args.communication_dtype,
            num_trainers_per_group=args.num_trainers_per_group,
            communicate_params=args.communicate_params,
        ),
        precision_config=PrecisionConfig(
            computation_dtype=args.computation_dtype.value,
            factor_matrix_dtype=args.factor_matrix_dtype.value,
            inv_factor_matrix_dtype=args.inv_factor_matrix_dtype.value,
            filtered_grad_dtype=args.filtered_grad_dtype.value,
            momentum_dtype=args.momentum_dtype.value,
            grafting_state_dtype=args.grafting_state_dtype.value,
        ),
        use_protected_eigh=args.use_protected_eigh,
        track_root_inv_residuals=args.track_root_inv_residuals,
    )

    # train model
    train_ddp_model(
        model,
        WORLD_SIZE,
        loss_function,
        sampler,
        data_loader,
        optimizer,
        device=device,
        epochs=args.epochs,
        window_size=args.window_size,
        use_distributed_checkpoint=args.use_distributed_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
    )

    # clean up process group
    dist.destroy_process_group()
