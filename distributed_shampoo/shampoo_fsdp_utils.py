"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
from copy import deepcopy
from math import prod
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from distributed_shampoo.shampoo_utils import (
    AdagradPreconditioner,
    BlockShampooPreconditioner,
    DistributedPreconditioner,
    GraftingType,
    LargeDimMethod,
    ShampooPreconditioner,
)

logger = logging.getLogger(__name__)


# TODO: think about enum names


class ConvexShapeRecoveryMethod(enum.Enum):
    SPLIT = 0
    COMM = 1


class CommunicationType(enum.Enum):
    SEND = 0
    RECV = 1
    NONE = 2


class CommunicationDirection(enum.Enum):
    FORWARD = 0
    BACKWARD = 1


def convex_split(
    tensor: Tensor,
    orig_shape: torch.Size,
    start_idx: int,
    end_idx: int,
    include_placeholder: bool = False,
) -> List[Tensor]:
    """Chunks tensor across dimensions row-wise to be convex.

    Starting from the leftmost dimension, the largest possible slices in each dimension
    (with the remaining dimensions on the right retaining the original shape) are split off.

    2D example:
     _______________                  _______________
    |       ________|                |       ________|
    |______|        |                |______|________|
    |               |                |               |
    |    shard  ____|       ->       |_______________|
    |__________|    |                |__________|    |
    |               |                |               |
    |_______________|                |_______________|

    Args:
        tensor (Tensor): Flattened gradient or tensor to split.
        orig_shape (torch.Size): Shape of original tensor that tensor is a slice of.
        start_idx (int): Flattened index in original tensor where tensor starts.
        end_idx (int): Flattened index in original tensor where tensor ends (inclusive).
        include_placeholder (bool): Whether to include empty partitioned tensors as placeholders. (Default: False)

    Returns:
        split_tensors (List[Tensor]): List of tensors.

    """
    if len(tensor.size()) != 1:
        logger.info(
            f"Input tensor is not flat, has shape {tensor.size()}. Continuing without splitting."
        )
        return tensor

    end_idx += 1  # correct off-by-one (FSDP shard_param_info provides inclusive index)
    assert (
        end_idx - start_idx == tensor.size()[0]
    ), f"Start/end indices do not match tensor size: start {start_idx} end {end_idx}, tensor size {tensor.size()}"

    # maintains the order that partitions had in the flat tensor
    split_tensors_left = []
    split_tensors_right = []
    left_idx = None
    right_idx = None
    center_partition = False
    for i in range(1, len(orig_shape) + 1):
        remaining_size = prod(orig_shape[i:])
        left_idx_new = int(np.ceil(start_idx / remaining_size)) * remaining_size
        right_idx_new = end_idx // remaining_size * remaining_size  # floor

        # first iteration (largest convex partition in the center)
        if not center_partition:
            if left_idx_new <= right_idx_new:
                if left_idx_new < right_idx_new:
                    split_tensors_left.append(
                        torch.narrow(
                            tensor,
                            0,
                            left_idx_new - start_idx,
                            right_idx_new - left_idx_new,
                        )
                        .view([-1] + list(orig_shape[i:]))
                        .squeeze()
                    )
                elif include_placeholder:
                    split_tensors_left.append(torch.tensor([]).to(tensor.device))
                left_idx = left_idx_new
                right_idx = right_idx_new
                center_partition = True
            continue

        # add partition to left of current partitions
        if left_idx_new < left_idx:
            split_tensors_left.append(
                torch.narrow(
                    tensor, 0, left_idx_new - start_idx, left_idx - left_idx_new
                )
                .view([-1] + list(orig_shape[i:]))
                .squeeze()
            )
            left_idx = left_idx_new
        elif left_idx_new == left_idx and include_placeholder:
            split_tensors_left.append(torch.tensor([]).to(tensor.device))

        # add partition to right of current partitions
        if right_idx < right_idx_new:
            split_tensors_right.append(
                torch.narrow(
                    tensor, 0, right_idx - start_idx, right_idx_new - right_idx
                )
                .view([-1] + list(orig_shape[i:]))
                .squeeze()
            )
            right_idx = right_idx_new
        elif right_idx == right_idx_new and include_placeholder:
            split_tensors_right.append(torch.tensor([]).to(tensor.device))

    split_tensors_left.reverse()
    return split_tensors_left + split_tensors_right


class SplitShampooPreconditioner(DistributedPreconditioner):
    """Shampoo with split function (currently row-wise convex split, see function convex_split) applied to the parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        metadata (Tuple): FSDP shard metadata of parameter. Contains fqn, original shape, original numels, and shard param info.
            See FSDP class FlatParameter for more details.
            https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/flat_param.py#L190
        large_dim_method (LargeDimMethod): method for handling large scale tensors. (Default: LargeDimMethod.BLOCKING)
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        exponent_override (int): Exponent override for taking the root of the matrix. If exponent_override = 0, uses
            2 * order of the tensor. (Default: 0)
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root. (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        block_size (int): Block size for blocking large tensors. (Default: 1024)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        use_merge_dims (bool): Denotes whether or not dimensions are merged. (Default: True)
        start_preconditioning_step (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (Default: 0)
        grafting_type (LayerwiseGraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)

    """

    def __init__(
        self,
        param,
        metadata: Tuple,
        large_dim_method: LargeDimMethod = LargeDimMethod.BLOCKING,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        max_preconditioner_dim: int = 1024,
        dtype: torch.dtype = torch.float,
        idx: Union[None, int, str] = None,
        use_merge_dims: bool = True,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
        use_protected_eigh: bool = True,
        use_dtensor: bool = False,
    ):
        super(SplitShampooPreconditioner, self).__init__(
            param,
        )

        # Set parameters.
        self._large_dim_method = large_dim_method
        self._beta2 = beta2
        self._epsilon = epsilon
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._use_bias_correction = use_bias_correction
        self._max_preconditioner_dim = max_preconditioner_dim
        self._dtype = dtype
        self._idx = idx
        self._start_preconditioning_step = start_preconditioning_step
        self._use_merge_dims = use_merge_dims

        _, orig_shape, _, shard_param_info = metadata
        start_idx = shard_param_info.intra_param_start_idx
        end_idx = shard_param_info.intra_param_end_idx
        self._orig_shape = orig_shape
        self._start_idx = start_idx
        self._end_idx = end_idx

        # Initialize exponential moving average
        self.bias_correction1 = torch.as_tensor(1.0)
        self.exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)

        # Construct multiple preconditioners for each block
        self._split_preconditioners = []

        split_param = convex_split(param, orig_shape, start_idx, end_idx)
        for i, p in enumerate(split_param):
            dims = torch.as_tensor(p.shape)
            split_idx = str(idx) + "." + str(i)

            # Blocks the tensor and applies Shampoo to each block, with block
            # size equal to the max_preconditioner_dim; see feature above.
            if self._large_dim_method == LargeDimMethod.BLOCKING:
                preconditioner = BlockShampooPreconditioner(
                    p,
                    beta2=beta2,
                    epsilon=epsilon,
                    exponent_override=exponent_override,
                    exponent_multiplier=exponent_multiplier,
                    use_bias_correction=use_bias_correction,
                    block_size=max_preconditioner_dim,
                    dtype=dtype,
                    idx=split_idx,
                    use_merge_dims=use_merge_dims,
                    start_preconditioning_step=start_preconditioning_step,
                    grafting_type=grafting_type,
                    grafting_beta2=grafting_beta2,
                    grafting_epsilon=grafting_epsilon,
                    use_protected_eigh=use_protected_eigh,
                    use_dtensor=use_dtensor,
                )

            # Uses Adagrad preconditioner if any dimension is larger than
            # the max_preconditioner_dim; see features above.
            elif self._large_dim_method == LargeDimMethod.ADAGRAD:
                preconditioner = (
                    AdagradPreconditioner(
                        p,
                        beta2=beta2,
                        epsilon=epsilon,
                        use_bias_correction=use_bias_correction,
                        idx=split_idx,
                        use_dtensor=use_dtensor,
                    )
                    if torch.any(dims > self._max_preconditioner_dim)
                    else ShampooPreconditioner(
                        p,
                        beta2=beta2,
                        epsilon=epsilon,
                        exponent_override=exponent_override,
                        exponent_multiplier=exponent_multiplier,
                        use_bias_correction=use_bias_correction,
                        diagonal_threshold=max_preconditioner_dim,
                        dtype=dtype,
                        idx=split_idx,
                        start_preconditioning_step=start_preconditioning_step,
                        grafting_type=grafting_type,
                        grafting_beta2=grafting_beta2,
                        grafting_epsilon=grafting_epsilon,
                        use_protected_eigh=use_protected_eigh,
                        use_dtensor=use_dtensor,
                    )
                )

            # Uses diagonal Shampoo preconditioner in place of full Shampoo
            # preconditioner if dimension is larger than max_preconditioner_dim; see feature
            # above.
            elif self._large_dim_method == LargeDimMethod.DIAGONAL:
                preconditioner = ShampooPreconditioner(
                    p,
                    beta2=beta2,
                    epsilon=epsilon,
                    exponent_override=exponent_override,
                    exponent_multiplier=exponent_multiplier,
                    use_bias_correction=use_bias_correction,
                    diagonal_threshold=max_preconditioner_dim,
                    dtype=dtype,
                    idx=split_idx,
                    start_preconditioning_step=start_preconditioning_step,
                    grafting_type=grafting_type,
                    grafting_beta2=grafting_beta2,
                    grafting_epsilon=grafting_epsilon,
                    use_protected_eigh=use_protected_eigh,
                    use_dtensor=use_dtensor,
                )

            else:
                raise ValueError(
                    "Large dim method "
                    + str(self._large_dim_method)
                    + " is not implemented!"
                )

            self._split_preconditioners.append(preconditioner)
            self._parameter_count += preconditioner.parameter_count

    def update_exp_avg(
        self,
        grad: Tensor,
        iteration: Tensor,
        beta1: float,
    ) -> Tensor:
        # TODO: maybe this could be moved to a lower level preconditioner class, but not sure about efficiency of passing everything

        # Compute bias corrections if necessary.
        if self._use_bias_correction:
            self.bias_correction1 = 1.0 - beta1**iteration

        # Compute exponential moving average of the gradient (with
        # potential bias correction).
        self.exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        return self.exp_avg / self.bias_correction1

    def apply_split(self, tensor: Tensor):
        return convex_split(tensor, self._orig_shape, self._start_idx, self._end_idx)

    def update_preconditioners(self, grad: Tensor, iteration: Tensor):
        split_grad = self.apply_split(grad)
        assert len(split_grad) == len(
            self._split_preconditioners
        ), f"split shampoo preconditioner {self._idx} has {len(self._split_preconditioners)} preconditioners but grad was split into {len(split_grad)}"
        for p, g in zip(self._split_preconditioners, split_grad):
            p.update_preconditioners(g, iteration)

    def precondition(self, grad: Tensor, iteration: Tensor) -> Tensor:
        split_grad = self.apply_split(grad)
        assert len(self._split_preconditioners) == len(
            split_grad
        ), f"split shampoo preconditioner {self._idx} has {len(self._split_preconditioners)} preconditioners but grad was split into {len(split_grad)}"

        split_preconditioned_grad = []
        for p, g in zip(self._split_preconditioners, split_grad):
            split_preconditioned_grad.append(p.precondition(g, iteration))

        return split_preconditioned_grad

    def compute_root_inverse(self) -> None:
        for preconditioner in self._split_preconditioners:
            if isinstance(
                preconditioner, (ShampooPreconditioner, BlockShampooPreconditioner)
            ):
                preconditioner.compute_root_inverse()

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        relative_errors = []
        relative_residuals = []

        for preconditioner in self._split_preconditioners:
            if isinstance(
                preconditioner, (ShampooPreconditioner, BlockShampooPreconditioner)
            ):
                (
                    relative_errors_temp,
                    relative_residuals_temp,
                ) = preconditioner.compute_root_inverse_residuals()

                relative_errors += relative_errors_temp
                relative_residuals += relative_residuals_temp

        return (
            relative_errors,
            relative_residuals,
        )

    def compute_norm(self, grad: Tensor, iteration: Tensor) -> Tensor:
        return torch.linalg.norm(self.precondition(grad, iteration))

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            for preconditioner in self._split_preconditioners:
                preconditioner.to(device=device)

    def num_preconditioners(self) -> int:
        # returns total number of preconditioners (where block preconditioners are considered to contain multiple preconditioners)
        return sum(
            preconditioner.num_preconditioners()
            for preconditioner in self._split_preconditioners
        )

    def reset_preconditioners(self) -> None:
        for preconditioner in self._split_preconditioners:
            preconditioner.reset_preconditioners()


class CommunicationShampooPreconditioner(DistributedPreconditioner):
    """Shampoo with communication of small partitions of the parameters for convex shape recovery with FSDP.
    Currently only tested on the case where sending is from left to right in terms of rank number.
    TODO: come up with a better name

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        metadata (Tuple): FSDP shard metadata of parameter. Contains fqn, original shape, original numels, and shard param info.
            See FSDP class FlatParameter for more details.
            https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/flat_param.py#L190
        large_dim_method (LargeDimMethod): method for handling large scale tensors. (Default: LargeDimMethod.BLOCKING)
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        exponent_override (int): Exponent override for taking the root of the matrix. If exponent_override = 0, uses
            2 * order of the tensor. (Default: 0)
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root. (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        block_size (int): Block size for blocking large tensors. (Default: 1024)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        use_merge_dims (bool): Denotes whether or not dimensions are merged. (Default: True)
        start_preconditioning_step (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (Default: 0)
        grafting_type (LayerwiseGraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        use_dtensor (bool): Flag for using DTensor. Requires PyTorch 2 nightly. Otherwise, uses Tensor. (Default: True)
        left_comm (CommunicationType): Indicates how param should communicate left side. (Default: CommunicationType.NONE)
        right_comm (CommunicationType): Indicates how param should communicate right side. (Default: CommunicationType.NONE)

    """

    def __init__(
        self,
        param,
        metadata: Tuple,
        large_dim_method: LargeDimMethod = LargeDimMethod.BLOCKING,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        max_preconditioner_dim: int = 1024,
        dtype: torch.dtype = torch.float,
        idx: Union[None, int, str] = None,
        use_merge_dims: bool = True,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
        use_protected_eigh: bool = True,
        use_dtensor: bool = False,
        left_comm: CommunicationType = CommunicationType.NONE,
        right_comm: CommunicationType = CommunicationType.NONE,
    ):
        super(CommunicationShampooPreconditioner, self).__init__(
            param,
        )

        # Set parameters.
        self._large_dim_method = large_dim_method
        self._beta2 = beta2
        self._epsilon = epsilon
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._use_bias_correction = use_bias_correction
        self._max_preconditioner_dim = max_preconditioner_dim
        self._dtype = dtype
        self._idx = idx
        self._start_preconditioning_step = start_preconditioning_step
        self._use_merge_dims = use_merge_dims

        _, orig_shape, orig_numels, shard_param_info = metadata
        start_idx = shard_param_info.intra_param_start_idx
        end_idx = shard_param_info.intra_param_end_idx
        self._orig_shape = orig_shape
        self._start_idx = start_idx
        self._end_idx = end_idx

        self._left_comm = left_comm
        self._right_comm = right_comm

        # Initialize exponential moving average
        self.bias_correction1 = torch.as_tensor(1.0)
        self.exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)

        # Initialize communication buffers
        # TODO: decide whether to keep checks on Nones or update left_comm and right_comm
        self.left_send_buffer = None
        self.right_send_buffer = None
        self.left_recv_buffer = None
        self.right_recv_buffer = None
        # TODO: play around with ordering of ops in batch to see whether all these lists are necessary
        # if they are necessary, there may be a cleaner way of keeping track of them than lists
        self.forward_send_ops = []
        self.forward_recv_ops = []
        self.backward_send_ops = []
        self.backward_recv_ops = []
        split_param = convex_split(
            param, orig_shape, start_idx, end_idx, include_placeholder=True
        )

        if (
            left_comm == CommunicationType.SEND
            # TODO: these checks could be moved into the main optimizer, but that may get messy
            # reconsider if a more complex assignment function is written
            and start_idx != 0
            and split_param[0].numel() != 0
        ):
            self.left_send_buffer = torch.zeros_like(split_param[0]).to(param.device)
            # TODO: not sure if calling dist.get_rank() is good practice; may need to pass rank into the class
            self.forward_send_ops.append(
                dist.P2POp(
                    dist.isend, self.left_send_buffer, dist.get_rank() - 1, tag=idx
                )
            )
            self.backward_recv_ops.append(
                dist.P2POp(
                    dist.irecv,
                    self.left_send_buffer,
                    dist.get_rank() - 1,
                    tag=idx,
                )
            )
            split_param = split_param[1:]
        elif (
            left_comm == CommunicationType.RECV
            and start_idx != 0
            and split_param[0].numel() != 0
        ):
            # TODO: write more complex merging function
            self.left_recv_buffer = torch.zeros(
                orig_shape[-1] - split_param[0].size()[0]
            ).to(param.device)
            self.left_recv_buffer_exp_avg = torch.zeros_like(self.left_recv_buffer)
            self.forward_recv_ops.append(
                dist.P2POp(
                    dist.irecv,
                    self.left_recv_buffer,
                    dist.get_rank() - 1,
                    tag=idx,
                )
            )
            self.backward_send_ops.append(
                dist.P2POp(
                    dist.isend,
                    self.left_recv_buffer,
                    dist.get_rank() - 1,
                    tag=idx,
                )
            )
            split_param[0] = torch.cat([self.left_recv_buffer, split_param[0]])

        if (
            right_comm == CommunicationType.SEND
            and end_idx != orig_numels - 1
            and split_param[-1].numel() != 0
        ):
            self.right_send_buffer = torch.zeros_like(split_param[-1]).to(param.device)
            # TODO: not sure if calling dist.get_rank() is good practice; may need to pass rank into the class
            self.forward_send_ops.append(
                dist.P2POp(
                    dist.isend,
                    self.right_send_buffer,
                    dist.get_rank() + 1,
                    tag=idx,
                )
            )
            self.backward_recv_ops.append(
                dist.P2POp(
                    dist.irecv,
                    self.right_send_buffer,
                    dist.get_rank() + 1,
                    tag=idx,
                )
            )
            split_param = split_param[:-1]
        elif (
            right_comm == CommunicationType.RECV
            and end_idx != orig_numels - 1
            and split_param[-1].numel() != 0
        ):
            # TODO: write more complex merging function
            self.right_recv_buffer = torch.zeros(
                orig_shape[-1] - split_param[-1].size()[0]
            ).to(param.device)
            self.right_recv_buffer_exp_avg = torch.zeros_like(self.right_recv_buffer)
            self.forward_recv_ops.append(
                dist.P2POp(
                    dist.irecv,
                    self.right_recv_buffer,
                    dist.get_rank() + 1,
                    tag=idx,
                )
            )
            self.backward_send_ops.append(
                dist.P2POp(
                    dist.isend,
                    self.right_recv_buffer,
                    dist.get_rank() + 1,
                    tag=idx,
                )
            )
            split_param[-1] = torch.cat([split_param[-1], self.right_recv_buffer])

        # Construct multiple preconditioners for each block
        self._split_preconditioners = []
        self._split_preconditioned_grad = None

        for i, p in enumerate(split_param):
            dims = torch.as_tensor(p.shape)
            split_idx = str(idx) + "." + str(i)
            if p.numel() == 0:
                continue

            # Blocks the tensor and applies Shampoo to each block, with block
            # size equal to the max_preconditioner_dim; see feature above.
            if self._large_dim_method == LargeDimMethod.BLOCKING:
                preconditioner = BlockShampooPreconditioner(
                    p,
                    beta2=beta2,
                    epsilon=epsilon,
                    exponent_override=exponent_override,
                    exponent_multiplier=exponent_multiplier,
                    use_bias_correction=use_bias_correction,
                    block_size=max_preconditioner_dim,
                    dtype=dtype,
                    idx=split_idx,
                    use_merge_dims=use_merge_dims,
                    start_preconditioning_step=start_preconditioning_step,
                    grafting_type=grafting_type,
                    grafting_beta2=grafting_beta2,
                    grafting_epsilon=grafting_epsilon,
                    use_protected_eigh=use_protected_eigh,
                    use_dtensor=use_dtensor,
                )

            # Uses Adagrad preconditioner if any dimension is larger than
            # the max_preconditioner_dim; see features above.
            elif self._large_dim_method == LargeDimMethod.ADAGRAD:
                preconditioner = (
                    AdagradPreconditioner(
                        p,
                        beta2=beta2,
                        epsilon=epsilon,
                        use_bias_correction=use_bias_correction,
                        idx=split_idx,
                        use_dtensor=use_dtensor,
                    )
                    if torch.any(dims > self._max_preconditioner_dim)
                    else ShampooPreconditioner(
                        p,
                        beta2=beta2,
                        epsilon=epsilon,
                        exponent_override=exponent_override,
                        exponent_multiplier=exponent_multiplier,
                        use_bias_correction=use_bias_correction,
                        diagonal_threshold=max_preconditioner_dim,
                        dtype=dtype,
                        idx=split_idx,
                        start_preconditioning_step=start_preconditioning_step,
                        grafting_type=grafting_type,
                        grafting_beta2=grafting_beta2,
                        grafting_epsilon=grafting_epsilon,
                        use_protected_eigh=use_protected_eigh,
                        use_dtensor=use_dtensor,
                    )
                )

            # Uses diagonal Shampoo preconditioner in place of full Shampoo
            # preconditioner if dimension is larger than max_preconditioner_dim; see feature
            # above.
            elif self._large_dim_method == LargeDimMethod.DIAGONAL:
                preconditioner = ShampooPreconditioner(
                    p,
                    beta2=beta2,
                    epsilon=epsilon,
                    exponent_override=exponent_override,
                    exponent_multiplier=exponent_multiplier,
                    use_bias_correction=use_bias_correction,
                    diagonal_threshold=max_preconditioner_dim,
                    dtype=dtype,
                    idx=split_idx,
                    start_preconditioning_step=start_preconditioning_step,
                    grafting_type=grafting_type,
                    grafting_beta2=grafting_beta2,
                    grafting_epsilon=grafting_epsilon,
                    use_protected_eigh=use_protected_eigh,
                    use_dtensor=use_dtensor,
                )

            else:
                raise ValueError(
                    "Large dim method "
                    + str(self._large_dim_method)
                    + " is not implemented!"
                )

            self._split_preconditioners.append(preconditioner)
            self._parameter_count += preconditioner.parameter_count

    def update_exp_avg(
        self,
        grad: Tensor,
        iteration: Tensor,
        beta1: float,
    ) -> Tensor:
        # TODO: maybe this could be moved to a lower preconditioner class, but not sure about efficiency of passing everything

        # Compute bias corrections if necessary.
        if self._use_bias_correction:
            self.bias_correction1 = 1.0 - beta1**iteration

        # Compute exponential moving average of the gradient (with
        # potential bias correction).
        self.exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        if self.left_recv_buffer is not None:
            self.left_recv_buffer_exp_avg.mul_(beta1).add_(
                self.left_recv_buffer, alpha=1 - beta1
            )
            self.left_recv_buffer.copy_(
                self.left_recv_buffer_exp_avg / self.bias_correction1
            )
        if self.right_recv_buffer is not None:
            self.right_recv_buffer_exp_avg.mul_(beta1).add_(
                self.right_recv_buffer, alpha=1 - beta1
            )
            self.right_recv_buffer.copy_(
                self.right_recv_buffer_exp_avg / self.bias_correction1
            )

        # TODO: an alternative to passing this back is to also store this within the class
        # however, this may take up some more memory, since this gets copied to p.grad in main
        return self.exp_avg / self.bias_correction1

    def get_forward_ops(self, grad):
        split_grad = convex_split(
            grad,
            self._orig_shape,
            self._start_idx,
            self._end_idx,
            include_placeholder=True,
        )

        if self.left_send_buffer is not None:
            self.left_send_buffer.copy_(split_grad[0])
        if self.right_send_buffer is not None:
            self.right_send_buffer.copy_(split_grad[-1])

        return self.forward_send_ops, self.forward_recv_ops

    def get_backward_ops(self):
        # NOTE: recv buffers should be filled from precondition_and_store already!
        return self.backward_send_ops, self.backward_recv_ops

    def apply_split(self, tensor: Tensor, merge_buffer: bool = False):
        split = convex_split(
            tensor,
            self._orig_shape,
            self._start_idx,
            self._end_idx,
            include_placeholder=True,
        )

        if merge_buffer:
            if self.left_send_buffer is not None:
                split = split[1:]
            elif self.left_recv_buffer is not None:
                # TODO: write more complex merging function
                split[0] = torch.cat([self.left_recv_buffer, split[0]])

            if self.right_send_buffer is not None:
                split = split[:-1]
            elif self.right_recv_buffer is not None:
                # TODO: write more complex merging function
                split[-1] = torch.cat([split[-1], self.right_recv_buffer])

        split = [t for t in split if t.numel() != 0]
        return split

    def update_preconditioners(self, grad: Tensor, iteration: Tensor):
        split_grad = self.apply_split(grad, merge_buffer=True)
        assert len(split_grad) == len(
            self._split_preconditioners
        ), f"split shampoo preconditioner {self._idx} has {len(self._split_preconditioners)} preconditioners but grad was split into {len(split_grad)}"
        for p, g in zip(self._split_preconditioners, split_grad):
            p.update_preconditioners(g, iteration)

    def precondition(self, grad: Tensor, iteration: Tensor) -> Tensor:
        split_grad = self.apply_split(grad, merge_buffer=True)
        assert len(self._split_preconditioners) == len(
            split_grad
        ), f"split shampoo preconditioner {self._idx} has {len(self._split_preconditioners)} preconditioners but grad was split into {len(split_grad)}"

        split_preconditioned_grad = []
        for p, g in zip(self._split_preconditioners, split_grad):
            split_preconditioned_grad.append(p.precondition(g, iteration))

        return split_preconditioned_grad

    def precondition_and_store(self, grad: Tensor, iteration: Tensor) -> None:
        # TODO: come up with a better name for this function
        # might not want to use preconditioned_grad_to_buffer since that's used by DDP Shampoo?
        split_preconditioned_grad = self.precondition(grad, iteration)

        if self.left_recv_buffer is not None:
            self.left_recv_buffer.copy_(
                split_preconditioned_grad[0][: self.left_recv_buffer.size()[0]]
            )
            split_preconditioned_grad[0] = split_preconditioned_grad[0][
                self.left_recv_buffer.size()[0] :
            ]

        if self.right_recv_buffer is not None:
            self.right_recv_buffer.copy_(
                split_preconditioned_grad[-1][-self.right_recv_buffer.size()[0] :]
            )
            split_preconditioned_grad[-1] = split_preconditioned_grad[-1][
                : -self.right_recv_buffer.size()[0]
            ]

        self._split_preconditioned_grad = split_preconditioned_grad

    def retrieve_preconditioned_grad(self) -> Tensor:
        left = [self.left_send_buffer] if self.left_send_buffer is not None else []
        right = [self.right_send_buffer] if self.right_send_buffer is not None else []
        return left + self._split_preconditioned_grad + right

    def compute_root_inverse(self) -> None:
        for preconditioner in self._split_preconditioners:
            if isinstance(
                preconditioner, (ShampooPreconditioner, BlockShampooPreconditioner)
            ):
                preconditioner.compute_root_inverse()

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        relative_errors = []
        relative_residuals = []

        for preconditioner in self._split_preconditioners:
            if isinstance(
                preconditioner, (ShampooPreconditioner, BlockShampooPreconditioner)
            ):
                (
                    relative_errors_temp,
                    relative_residuals_temp,
                ) = preconditioner.compute_root_inverse_residuals()

                relative_errors += relative_errors_temp
                relative_residuals += relative_residuals_temp

        return (
            relative_errors,
            relative_residuals,
        )

    def compute_norm(self, grad: Tensor, iteration: Tensor) -> Tensor:
        return torch.linalg.norm(self.precondition(grad, iteration))

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            for preconditioner in self._split_preconditioners:
                preconditioner.to(device=device)

    def num_preconditioners(self) -> int:
        # returns total number of preconditioners (where block preconditioners are considered to contain multiple preconditioners)
        return sum(
            preconditioner.num_preconditioners()
            for preconditioner in self._split_preconditioners
        )

    def reset_preconditioners(self) -> None:
        for preconditioner in self._split_preconditioners:
            preconditioner.reset_preconditioners()
