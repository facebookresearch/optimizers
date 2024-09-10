"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import os

import torch.distributed as dist
from distributed_shampoo.examples.trainer_utils import (
    get_data_loader_and_sampler,
    get_model_and_loss_fn,
    instantiate_optimizer,
    Parser,
    set_seed,
    setup_distribution,
    train_model,
)

from distributed_shampoo.shampoo_types import FSDPShampooConfig, PrecisionConfig
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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


if __name__ == "__main__":
    """Multi-GPU CIFAR-10 Distributed Data Parallel Training Example Script

    Uses torch.distributed to launch distributed training run.

    Requirements:
        - Python 3.8 or above
        - PyTorch / TorchVision

    To run this training script with a single node, one can run from the optimizers directory:

    SGD (with learning rate = 1e-2, momentum = 0.9):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.fsdp_cifar10_example --optimizer-type SGD --lr 1e-2 --momentum 0.9

    Adam (with default parameters):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.fsdp_cifar10_example --optimizer-type ADAM

    Distributed Shampoo (with default Adam grafting, precondition frequency = 100):
        torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_TRAINERS -m distributed_shampoo.examples.fsdp_cifar10_example --optimizer-type DISTRIBUTED_SHAMPOO --precondition-frequency 100 --grafting-type ADAM --num-trainers-per-group -1 --use-bias-correction --use-decoupled-weight-decay --use-merge-dims

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

    # instantiate model and loss function
    model, loss_function = get_model_and_loss_fn(device)
    model = FSDP(model, use_orig_params=True)

    # instantiate data loader
    data_loader, sampler = get_data_loader_and_sampler(
        args.data_path, WORLD_SIZE, WORLD_RANK, args.local_batch_size
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
        grafting_epsilon=args.grafting_epsilon,
        grafting_beta2=args.grafting_beta2,
        use_merge_dims=args.use_merge_dims,
        use_pytorch_compile=args.use_pytorch_compile,
        distributed_config=FSDPShampooConfig(
            param_to_metadata=compile_fsdp_parameter_metadata(model),
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
    train_model(
        model,
        WORLD_SIZE,
        loss_function,
        sampler,
        data_loader,
        optimizer,
        device=device,
        epochs=args.epochs,
        window_size=args.window_size,
        local_rank=LOCAL_RANK,
    )

    # clean up process group
    dist.destroy_process_group()
