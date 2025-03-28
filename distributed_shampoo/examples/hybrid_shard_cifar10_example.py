"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_checkpoint

from distributed_shampoo import DistributedShampoo, HybridShardShampooConfig
from distributed_shampoo.examples.trainer_utils import (
    get_data_loader_and_sampler,
    get_model_and_loss_fn,
    instantiate_optimizer,
    LossMetrics,
    Parser,
    set_seed,
    setup_distribution,
)

from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

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


def train_hybrid_shard_model(
    model: nn.Module,
    world_size: int,
    loss_function: nn.Module,
    sampler: torch.utils.data.Sampler,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    window_size: int = 100,
    use_distributed_checkpoint: bool = False,
    checkpoint_dir: str | None = None,
) -> tuple[float, float, int]:
    """Constructs the main training loop.

    Assumes torch.distributed is initialized.

    """

    # initialize metrics
    metrics = LossMetrics(window_size=window_size, device=device, world_size=world_size)

    # main training loop
    for epoch in range(epochs):
        metrics._epoch = epoch
        sampler.set_epoch(epoch)  # type: ignore[attr-defined]

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
        assert checkpoint_dir is not None
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

    return (
        metrics._lifetime_loss.item(),
        metrics._window_loss.item(),
        metrics._iteration,
    )


def create_model_and_optimizer_and_loss_fn(args, device, device_mesh):
    # instantiate model and loss function
    model, loss_function = get_model_and_loss_fn(device)

    model = fully_shard(model, mesh=device_mesh)
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
        distributed_config=HybridShardShampooConfig(
            device_mesh=device_mesh,
            num_trainers_per_group=args.num_trainers_per_group,
        ),
        preconditioner_dtype=args.preconditioner_dtype,
        preconditioner_computation_type=args.preconditioner_computation_type,
    )
    return model, optimizer, loss_function


if __name__ == "__main__":
    """Multi-GPU CIFAR-10 Per-parameter Hybrid Sharded Data Parallel (a.k.a HSDP2) Training Example Script

    Uses torch.distributed to launch distributed training run.

    Requirements:
        - Python 3.10 or above
        - PyTorch / TorchVision

    To run this training script with a single node, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hybrid_shard_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hybrid_shard_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting, precondition frequency = 100):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.hybrid_shard_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --num-trainers-per-group -1 --use-bias-correction --use-decoupled-weight-decay --use-merge-dims --dp-replicate-degree 2 

    To use distributed checkpointing, append the flag --use-distributed-checkpoint with optional --checkpoint-dir argument.

    The script will produce lifetime and window loss values retrieved from the forward pass over the data.
    Guaranteed reproducibility on a single GPU.

    """

    args = Parser.get_args()

    # set seed for reproducibility
    set_seed(args.seed)

    # initialize distributed process group
    device = setup_distribution(
        backend=args.backend,
        world_rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        local_rank=LOCAL_RANK,
    )

    # initialize device_mesh for hybrid shard data parallel
    device_mesh = init_device_mesh(
        "cuda",
        (args.dp_replicate_degree, WORLD_RANK // args.dp_replicate_degree),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )

    model, optimizer, loss_fn = create_model_and_optimizer_and_loss_fn(
        args, device, device_mesh
    )

    # instantiate data loader
    data_loader, sampler = get_data_loader_and_sampler(
        args.data_path, WORLD_SIZE, WORLD_RANK, args.local_batch_size
    )

    # train model
    train_hybrid_shard_model(
        model,
        WORLD_SIZE,
        loss_fn,
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
