"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
import math
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from distributed_shampoo.matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
)
from torch import Tensor
from torch.distributed.distributed_c10d import GroupMember

# For backward compatibility, since get_group_rank / get_global_rank
# are not yet officially released in PyTorch.
if not hasattr(dist, "get_group_rank"):
    from torch.distributed.distributed_c10d import _get_group_rank

    dist.get_group_rank = (
        lambda group, global_rank: global_rank
        if group is GroupMember.WORLD
        else _get_group_rank(group, global_rank)
    )

if not hasattr(dist, "get_global_rank"):
    from torch.distributed.distributed_c10d import _get_global_rank

    dist.get_global_rank = (
        lambda group, group_rank: group_rank
        if group is GroupMember.WORLD
        else _get_global_rank(group, group_rank)
    )


logger = logging.getLogger(__name__)

###### ENUM CLASSES ######
class PreconditionerType(enum.Enum):
    FULL = 0
    DIAGONAL = 1


class GraftingType(enum.Enum):
    NONE = 0
    SGD = 1
    ADAGRAD = 2
    RMSPROP = 3
    ADAM = 4
    ADAGRAD_NORMALIZED = 5
    RMSPROP_NORMALIZED = 6
    ADAM_NORMALIZED = 7


class LargeDimMethod(enum.Enum):
    DIAGONAL = 0
    ADAGRAD = 1
    BLOCKING = 2


class RootInvStrategy(enum.Enum):
    NONE = 0
    CROSS_NODE = 1
    INTRA_NODE_ONLY = 2


###### MERGING AND BLOCKING HELPER FUNCTIONS ######
def merge_small_dims(tensor_shape: List[int], threshold: int) -> List[int]:
    """Reshapes tensor by merging small dimensions.

    Args:
        tensor_shape (List[int]): The shape of the tensor.
        threshold (int): Threshold on the maximum size of each dimension.

    Returns:
        new_tensor_shape (List[int]): New tensor shape.

    """

    new_tensor_shape = [tensor_shape[0]]
    for next_tensor_shape in tensor_shape[1:]:
        new_dimension = new_tensor_shape[-1] * next_tensor_shape
        if (
            new_tensor_shape[-1] == 1
            or next_tensor_shape == 1
            or new_dimension <= threshold
        ):
            new_tensor_shape[-1] = new_dimension
        else:
            new_tensor_shape.append(next_tensor_shape)

    return new_tensor_shape


def multi_dim_split(tensor: Tensor, splits: List[int]) -> List[Tensor]:
    """Chunks tensor across multiple dimensions based on splits.

    Args:
        tensor (Tensor): Gradient or tensor to split.
        splits (List[int]): List of sizes for each block or chunk along each dimension.

    Returns:
        split_grad (List[Tensor]): List of tensors.

    """
    split_tensors = [tensor]
    for dim, split in enumerate(splits):
        split_tensors = [
            s for t in split_tensors for s in torch.split(t, split, dim=dim)
        ]
    return split_tensors


def multi_dim_cat(split_tensors: List[Tensor], num_splits: List[int]) -> Tensor:
    """Concatenates multiple tensors to form single tensor across multiple dimensions.

    Args:
        split_tensor (List[Tensor]): List of tensor splits or blocks.
        num_splits (List[int]): Number of splits/blocks.

    Returns:
        merged_tensor (Tensor): Merged tensor.

    """
    merged_tensor = split_tensors
    for dim, split in reversed(list(enumerate(num_splits))):
        if split > 0:
            merged_tensor = [
                torch.cat(merged_tensor[i : i + split], dim=dim)
                for i in range(0, len(merged_tensor), split)
            ]
    assert len(merged_tensor) == 1
    return merged_tensor[0]


###### PRECONDITIONER CLASSES ######
class Preconditioner(ABC):
    """Preconditioner base class."""

    def __init__(self):
        self._parameter_count = 0

    def update_preconditioners(self, grad: Tensor) -> None:
        pass

    def precondition(self, grad: Tensor) -> Tensor:
        pass

    def precondition_and_update(
        self,
        param,
        grad: Tensor,
        lr: Union[float, Tensor],
    ) -> None:
        pass

    def compute_norm(self, grad: Tensor) -> Tensor:
        pass

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def broadcast(self, group: Optional[dist.ProcessGroup]):
        return

    def to(self, device: Union[None, torch.device] = None):
        pass


class AdagradPreconditioner(Preconditioner):
    """Adagrad / Adam / RMSProp preconditioner for a generic layer.

    Stores preconditioner using same format as parameter p. Operations are performed in-place.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)
        idx (Union[None, str, int]): Layer index (for logging purposes). (Default: None)

    """

    def __init__(
        self,
        param,
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
        idx: Union[None, str, int] = None,
    ):
        super(AdagradPreconditioner, self).__init__()
        self._beta2 = beta2
        self._epsilon = epsilon
        self._preconditioner = torch.zeros_like(param)
        self._idx = idx
        self._num_updates = torch.as_tensor(0)
        self._use_bias_correction = use_bias_correction
        self._bias_correction2 = torch.as_tensor(1.0)
        self._parameter_count += (
            torch.prod(torch.as_tensor(self._preconditioner.shape)).cpu().numpy()
        )

        if self._idx is not None:
            self._preconditioner_idx = str(self._idx) + "." + str(0)
            logger.info(
                f"Diagonal Adagrad Preconditioner {self._preconditioner_idx} with Parameter {self._idx}"
            )

    def update_preconditioners(self, grad: Tensor) -> None:
        if self._beta2 == 1.0:
            self._preconditioner.addcmul_(grad, grad, value=1)
        else:
            self._preconditioner.mul_(self._beta2).addcmul_(
                grad, grad, value=1 - self._beta2
            )

        self._num_updates += 1
        if self._use_bias_correction and self._beta2 < 1.0:
            self._bias_correction2 = 1.0 - self._beta2**self._num_updates

    def precondition(self, grad: Tensor) -> Tensor:
        denom = (
            (self._preconditioner / self._bias_correction2).sqrt().add_(self._epsilon)
        )
        grad.div_(denom)
        return grad

    def precondition_and_update(
        self,
        param,
        grad: Tensor,
        lr: Union[float, Tensor],
    ) -> None:
        denom = (
            (self._preconditioner / self._bias_correction2).sqrt().add_(self._epsilon)
        )
        param.addcdiv_(grad, denom, value=-lr)

    def compute_norm(self, grad: Tensor):
        denom = (
            (self._preconditioner / self._bias_correction2).sqrt().add_(self._epsilon)
        )
        adagrad_nrm = torch.linalg.norm(grad / denom)
        return adagrad_nrm

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self._preconditioner = self._preconditioner.to(device=device)
            self._bias_correction2 = self._bias_correction2.to(device=device)


class ShampooKroneckerFactor:
    """Shampoo Kronecker Factor Matrix / Preconditioner data class."""

    def __init__(
        self,
        preconditioner_type: PreconditionerType,
        factor_matrix: Tensor,
        inv_factor_matrix: Optional[Tensor] = None,
        group_source_rank: Optional[int] = None,
        index: Optional[str] = None,
        is_diagonal: bool = True,
    ):
        self.preconditioner_type = preconditioner_type
        self.factor_matrix = factor_matrix
        self.inv_factor_matrix = inv_factor_matrix
        self.group_source_rank = group_source_rank
        self.index = index
        self.is_diagonal = torch.as_tensor(is_diagonal)

    def to(self, device):
        self.factor_matrix = self.factor_matrix.to(device)
        if self.inv_factor_matrix is not None:
            self.inv_factor_matrix = self.inv_factor_matrix.to(device)


class ShampooPreconditioner(Preconditioner):
    """Shampoo preconditioners for some generic layer.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        exponent_override (int): Exponent override for taking the root of the matrix. If exponent_override = 0, uses
            2 * order of the tensor. (Default: 0)
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root. (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        diagonal_threshold (int): Threshold for using diagonal preconditioners. If None, disabled. (Default: None)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        root_inv_strategy (RootInvStrategy): Strategy for assigning root inverse computations. (Default: RootInvStrategy.INTRA_NODE_ONLY)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        start_preconditioning_step (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (default: 0)
        grafting_type (GraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)

    """

    def __init__(
        self,
        param,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        diagonal_threshold: Union[None, int] = None,
        dtype: torch.dtype = torch.float,
        root_inv_strategy: RootInvStrategy = RootInvStrategy.INTRA_NODE_ONLY,
        idx: Union[None, int, str] = None,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
    ):

        super(ShampooPreconditioner, self).__init__()

        # Initialize parameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._diagonal_threshold = diagonal_threshold
        self._dtype = dtype
        self._num_updates = torch.as_tensor(0)
        self._use_bias_correction = use_bias_correction
        self._bias_correction2 = torch.as_tensor(1.0)
        self._dims = torch.as_tensor(param.shape).numpy()
        self._order = param.dim()
        self._root_inv_strategy = root_inv_strategy
        self._idx = idx
        self._grafting_type = grafting_type
        self._start_preconditioning_step = start_preconditioning_step

        # Compute root.
        self._root = (
            2 * self._order if self._exponent_override == 0 else self._exponent_override
        )

        # Initialize lists for preconditioners, inverse preconditioners, types, and ranks.
        self._preconditioners = []

        for k, dim in enumerate(self._dims):
            index = str(self._idx) + "." + str(k) if self._idx else None

            # Creates a diagonal Shampoo preconditioner if dimension is larger than
            # self._diagonal_threshold.
            if self._diagonal_threshold is not None and dim > self._diagonal_threshold:
                preconditioner_type = PreconditionerType.DIAGONAL
                factor_matrix = torch.zeros(dim, dtype=param.dtype, device=param.device)
                inv_factor_matrix = None
                group_source_rank = None

                num_params = dim
                if self._idx is not None:
                    logger.info(
                        f"Diagonal Preconditioner {index} with Parameter {self._idx}, Order {k}, Dim {dim}, Number of Params {num_params}, DType {self._dtype}"
                    )

            # Otherwise, generates a full Shampoo preconditioner.
            else:
                preconditioner_type = PreconditionerType.FULL
                factor_matrix = torch.zeros(
                    (dim, dim), dtype=self._dtype, device=param.device
                )
                inv_factor_matrix = torch.zeros(
                    (dim, dim), dtype=param.dtype, device=param.device
                )
                group_source_rank = -1

                num_params = 2 * dim**2
                if self._idx is not None:
                    logger.info(
                        f"Full Matrix Preconditioner {index} with Parameter {self._idx}, Order {k}, Dim {dim}, Number of Params {num_params}, DType {self._dtype}"
                    )

            # Counts parameters and adds to lists.
            self._parameter_count += num_params
            self._preconditioners.append(
                ShampooKroneckerFactor(
                    preconditioner_type,
                    factor_matrix,
                    inv_factor_matrix,
                    group_source_rank,
                    index,
                )
            )

        # Initialize grafting method.
        if self._grafting_type == GraftingType.NONE:
            self._grafting = None
        elif self._grafting_type == GraftingType.SGD:
            self._grafting = SGDGrafting(param)
        elif self._grafting_type == GraftingType.ADAGRAD:
            self._grafting = AdagradGrafting(param, epsilon=grafting_epsilon)
        elif self._grafting_type == GraftingType.RMSPROP:
            self._grafting = RMSPropGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        elif self._grafting_type == GraftingType.ADAM:
            self._grafting = AdamGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        elif self._grafting_type == GraftingType.ADAGRAD_NORMALIZED:
            self._grafting = AdagradNormalizedGrafting(
                param,
                epsilon=grafting_epsilon,
            )
        elif self._grafting_type == GraftingType.RMSPROP_NORMALIZED:
            self._grafting = RMSPropNormalizedGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        elif self._grafting_type == GraftingType.ADAM_NORMALIZED:
            self._grafting = AdamNormalizedGrafting(
                param,
                beta2=grafting_beta2,
                epsilon=grafting_epsilon,
            )
        else:
            raise ValueError(f"Invalid Grafting Type {self._grafting_type}!")

        # Counts parameters for grafted method.
        self._parameter_count += getattr(self._grafting, "parameter_count", 0)

    def update_preconditioners(self, grad: Tensor) -> None:
        for k, (dim, preconditioner) in enumerate(
            zip(self._dims, self._preconditioners)
        ):
            if self._beta2 != 1.0:
                preconditioner.factor_matrix.mul_(self._beta2)

            # Update diagonal Shampoo preconditioner.
            if preconditioner.preconditioner_type == PreconditionerType.DIAGONAL:
                diagonal_or_outer_product = torch.linalg.norm(
                    grad.transpose(0, k).contiguous().view(dim, -1),
                    dim=1,
                ).pow(2)

            # Update full Shampoo preconditioner.
            else:
                contract_idx = [*range(k)] + [*range(k + 1, self._order)]
                diagonal_or_outer_product = torch.tensordot(
                    grad,
                    grad,
                    dims=(contract_idx, contract_idx),
                )
                if diagonal_or_outer_product.dtype != self._dtype:
                    diagonal_or_outer_product = diagonal_or_outer_product.to(
                        dtype=self._dtype
                    )

                # For tracking diagonality of the preconditioner.
                # Checks if the preconditioner is currently diagonal, then checks whether or not
                # the update matrix is diagonal.
                if preconditioner.is_diagonal and not check_diagonal(
                    diagonal_or_outer_product
                ):
                    preconditioner.is_diagonal = torch.as_tensor(False)
                    logger.info(
                        f"Preconditioner {preconditioner.index} is not diagonal."
                    )

            preconditioner.factor_matrix.add_(
                diagonal_or_outer_product,
                alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
            )

        # Update grafting preconditioner.
        if self._grafting_type != GraftingType.NONE:
            self._grafting.update_preconditioners(grad)

        self._num_updates += 1
        if self._use_bias_correction and self._beta2 < 1.0:
            self._bias_correction2 = 1.0 - self._beta2**self._num_updates

    def shampoo_precondition(self, grad: Tensor) -> Tensor:
        preconditioned_grad = grad.clone()
        for k, preconditioner in enumerate(self._preconditioners):

            # To handle diagonal case, requires not transposing the tensor.
            if self._diagonal_threshold is not None:

                # Precondition using diagonal preconditioner.
                if preconditioner.preconditioner_type == PreconditionerType.DIAGONAL:
                    denom = (
                        preconditioner.factor_matrix / self._bias_correction2
                    ).add_(self._epsilon)
                    preconditioned_grad.div_(
                        denom.pow(-self._exponent_multiplier / self._root)[
                            (None,) * k + (...,) + (None,) * (self._order - k - 1)
                        ]
                    )

                # Precondition using full Shampoo preconditioner.
                # Uses einsum in order to avoid transposing.
                else:
                    gradient_idx = [*range(1, self._order + 1)]
                    matrix_product_idx = deepcopy(gradient_idx)
                    matrix_product_idx[k] = 0
                    preconditioned_grad = torch.einsum(
                        preconditioner.inv_factor_matrix,
                        [0, k + 1],
                        preconditioned_grad,
                        gradient_idx,
                        matrix_product_idx,
                    )

            # Handles full Shampoo preconditioner case more efficiently but
            # transposes the tensor continually.
            else:
                preconditioned_grad = torch.tensordot(
                    preconditioned_grad, preconditioner.inv_factor_matrix, [[0], [0]]
                )

        # Apply grafting.
        if self._grafting_type != GraftingType.NONE:
            grafting_norm = self._grafting.direction_norm(grad)
            shampoo_norm = torch.linalg.norm(preconditioned_grad)
            preconditioned_grad.mul_(grafting_norm).div_(shampoo_norm + 1e-16)

        return preconditioned_grad

    def graft_precondition(self, grad: Tensor) -> Tensor:
        return (
            self._grafting.precondition(grad)
            if self._grafting_type != GraftingType.NONE
            else grad
        )

    def precondition(self, grad: Tensor) -> Tensor:
        return (
            self.graft_precondition
            if self._num_updates <= self._start_preconditioning_step
            else self.shampoo_precondition
        )(grad)

    def compute_root_inverse(
        self,
        rank: int = -1,
        group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        # Get group rank.
        group_rank = dist.get_group_rank(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            group if group is not None else dist.distributed_c10d.GroupMember.WORLD,
            rank,
        )

        for k, preconditioner in enumerate(self._preconditioners):

            # Check that this is a full Shampoo preconditioner.
            if preconditioner.preconditioner_type == PreconditionerType.FULL and (
                group_rank == preconditioner.group_source_rank
                if self._root_inv_strategy != RootInvStrategy.NONE
                else True
            ):
                # Add epsilon term and incorporate bias correction.
                bias_corrected_preconditioner = (
                    preconditioner.factor_matrix / self._bias_correction2
                )

                # Check for nan or inf values.
                if torch.any(torch.isnan(bias_corrected_preconditioner)):
                    logger.warning(
                        f"Encountered nan values in preconditioner {self._idx}.{k}!"
                    )
                elif torch.any(torch.isinf(bias_corrected_preconditioner)):
                    logger.warning(
                        f"Encountered inf values in preconditioner {self._idx}.{k}!"
                    )

                # Compute inverse preconditioner.
                inv_factor_matrix = matrix_inverse_root(
                    A=bias_corrected_preconditioner,
                    root=self._root,
                    epsilon=self._epsilon,
                    exponent_multiplier=self._exponent_multiplier,
                    is_diagonal=preconditioner.is_diagonal,
                )

                if inv_factor_matrix.dtype != preconditioner.inv_factor_matrix.dtype:
                    inv_factor_matrix = inv_factor_matrix.to(
                        dtype=preconditioner.inv_factor_matrix.dtype
                    )
                preconditioner.inv_factor_matrix = inv_factor_matrix

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        relative_errors = []
        relative_residuals = []

        for preconditioner in self._preconditioners:
            bias_corrected_preconditioner = (
                preconditioner.factor_matrix / self._bias_correction2
            )
            (
                relative_error,
                relative_residual,
            ) = compute_matrix_root_inverse_residuals(
                bias_corrected_preconditioner,
                preconditioner.inv_factor_matrix,
                self._root,
                self._epsilon,
                self._exponent_multiplier,
            )

            relative_errors.append(relative_error)
            relative_residuals.append(relative_residual)

        return (
            relative_errors,
            relative_residuals,
        )

    def precondition_and_update(
        self, param, grad: Tensor, lr: Union[float, Tensor]
    ) -> None:
        if self._num_updates <= self._start_preconditioning_step:
            self._grafting.precondition_and_update(param, grad, lr)
        else:
            preconditioned_grad = self.precondition(grad)
            param.add_(preconditioned_grad, alpha=-lr)

    def compute_norm(self, grad: Tensor) -> Tensor:
        return torch.linalg.norm(self.precondition(grad))

    def broadcast(self, group: Optional[dist.ProcessGroup] = None):
        for preconditioner in self._preconditioners:
            if preconditioner.preconditioner_type == PreconditionerType.FULL:
                global_source_rank = dist.get_global_rank(
                    # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
                    #  `Optional[ProcessGroup]`.
                    group=group
                    if group is not None
                    else dist.distributed_c10d.GroupMember.WORLD,
                    group_rank=preconditioner.group_source_rank,
                )
                dist.broadcast(
                    preconditioner.inv_factor_matrix,
                    src=global_source_rank,
                    group=group,
                )

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self._bias_correction2 = self._bias_correction2.to(device=device)
            for preconditioner in self._preconditioners:
                preconditioner.to(device)
            if self._grafting is not None:
                self._grafting.to(device=device)

    def assign_preconditioners_rank(
        self,
        preconditioner_count: int,
        group_size: int,
    ) -> int:
        for preconditioner in self._preconditioners:
            if preconditioner.preconditioner_type == PreconditionerType.FULL:
                preconditioner.group_source_rank = preconditioner_count % group_size
                preconditioner_count += 1
                if self._idx is not None:
                    logger.info(
                        f"Assigned Preconditioner {preconditioner.index} to rank {preconditioner.group_source_rank} in group"
                    )
        return preconditioner_count

    def reset_preconditioners(self) -> None:
        for preconditioner in self._preconditioners:
            preconditioner.factor_matrix.zero_()


class BlockShampooPreconditioner(Preconditioner):
    """Shampoo with blocking applied to the parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        exponent_override (int): Exponent override for taking the root of the matrix. If exponent_override = 0, uses
            2 * order of the tensor. (Default: 0)
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root. (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        block_size (int): Block size for blocking large tensors. (Default: 1024)
        dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        root_inv_strategy (RootInvStrategy): Strategy for assigning root inverse computations. (Default: RootInvStrategy.INTRA_NODE_ONLY)
        idx (Union[None, int, str]): Layer index (for logging purposes). (Default: None)
        use_merge_dims (bool): Denotes whether or not dimensions are merged. (Default: True)
        start_preconditioning_step (int): initial delay before starting to compute root inverse. Applies grafting method beforehand. (Default: 0)
        grafting_type (LayerwiseGraftingType): Selects grafting method. (Default: GraftingType.NONE)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)

    """

    def __init__(
        self,
        param,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        block_size: int = 1024,
        dtype: torch.dtype = torch.float,
        root_inv_strategy: RootInvStrategy = RootInvStrategy.INTRA_NODE_ONLY,
        idx: Union[None, int, str] = None,
        use_merge_dims: bool = True,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
    ):
        super(BlockShampooPreconditioner, self).__init__()

        # Set parameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._use_bias_correction = use_bias_correction
        self._block_size = block_size
        self._dtype = dtype
        self._idx = idx
        self._root_inv_strategy = root_inv_strategy
        self._start_preconditioning_step = start_preconditioning_step
        self._use_merge_dims = use_merge_dims
        self._original_dims = [*torch.as_tensor(param.shape).numpy()]
        self._merged_dims = (
            merge_small_dims(self._original_dims, self._block_size)
            if self._block_size is not None and use_merge_dims
            else self._original_dims
        )

        # Construct splits for blocking
        self._splits = [block_size] * len(self._merged_dims)
        self._num_splits = [math.ceil(dim / block_size) for dim in self._merged_dims]

        # Construct multiple preconditioners for each block
        self._split_preconditioners = []
        self._split_sizes = []

        split_param = self._combine_and_split_dims(param)
        for i, p in enumerate(split_param):
            self._split_sizes.append(torch.as_tensor(p.shape))
            split_idx = str(idx) + "." + str(i)
            preconditioner = ShampooPreconditioner(
                p,
                beta2=beta2,
                epsilon=epsilon,
                exponent_override=exponent_override,
                exponent_multiplier=exponent_multiplier,
                use_bias_correction=use_bias_correction,
                dtype=dtype,
                root_inv_strategy=root_inv_strategy,
                idx=split_idx,
                start_preconditioning_step=start_preconditioning_step,
                grafting_type=grafting_type,
                grafting_beta2=grafting_beta2,
                grafting_epsilon=grafting_epsilon,
            )
            self._split_preconditioners.append(preconditioner)
            self._parameter_count += preconditioner.parameter_count

    def _combine_and_split_dims(self, p: Tensor):
        if self._use_merge_dims:
            p = p.view(self._merged_dims)
        return multi_dim_split(p, self._splits)

    def update_preconditioners(self, grad: Tensor):
        split_grad = self._combine_and_split_dims(grad)
        assert len(split_grad) == len(self._split_preconditioners)
        for block_preconditioner, block_grad in zip(
            self._split_preconditioners, split_grad
        ):
            block_preconditioner.update_preconditioners(block_grad)

    def shampoo_precondition(self, grad: Tensor) -> Tensor:
        split_grad = self._combine_and_split_dims(grad)
        assert len(self._split_preconditioners) == len(split_grad)
        split_preconditioned_grad = [
            p.shampoo_precondition(g)
            for p, g in zip(self._split_preconditioners, split_grad)
        ]
        preconditioned_grad = multi_dim_cat(split_preconditioned_grad, self._num_splits)
        return (
            preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims
            else preconditioned_grad
        )

    def graft_precondition(self, grad: Tensor) -> Tensor:
        split_grad = self._combine_and_split_dims(grad)
        assert len(self._split_preconditioners) == len(split_grad)
        split_preconditioned_grad = [
            p.graft_precondition(g)
            for p, g in zip(self._split_preconditioners, split_grad)
        ]
        preconditioned_grad = multi_dim_cat(split_preconditioned_grad, self._num_splits)
        return (
            preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims
            else preconditioned_grad
        )

    def precondition(self, grad: Tensor) -> Tensor:
        split_grad = self._combine_and_split_dims(grad)
        assert len(self._split_preconditioners) == len(split_grad)
        split_preconditioned_grad = [
            p.precondition(g) for p, g in zip(self._split_preconditioners, split_grad)
        ]
        preconditioned_grad = multi_dim_cat(split_preconditioned_grad, self._num_splits)
        return (
            preconditioned_grad.view(self._original_dims)
            if self._use_merge_dims
            else preconditioned_grad
        )

    def compute_root_inverse(
        self, rank: int = -1, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        for preconditioner in self._split_preconditioners:
            preconditioner.compute_root_inverse(rank=rank, group=group)

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        relative_errors = []
        relative_residuals = []

        for preconditioner in self._split_preconditioners:
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

    def precondition_and_update(
        self,
        param,
        grad: Tensor,
        lr: Union[Tensor, float],
    ) -> None:
        preconditioned_grad = self.precondition(grad)
        param.add_(preconditioned_grad, alpha=-lr)

    def compute_norm(self, grad: Tensor) -> Tensor:
        return torch.linalg.norm(self.precondition(grad))

    def broadcast(self, group: Optional[dist.ProcessGroup] = None):
        for preconditioner in self._split_preconditioners:
            preconditioner.broadcast(group=group)

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            for preconditioner in self._split_preconditioners:
                preconditioner.to(device=device)

    def assign_preconditioners_rank(
        self,
        preconditioner_count: int,
        group_size: int,
    ) -> int:
        for preconditioner in self._split_preconditioners:
            preconditioner_count = preconditioner.assign_preconditioners_rank(
                preconditioner_count, group_size
            )
        return preconditioner_count

    def reset_preconditioners(self) -> None:
        for preconditioner in self._split_preconditioners:
            preconditioner.reset_preconditioners()


###### GRAFTING CLASSES ######
class Grafting(ABC):
    """Grafting base class.

    We graft the method by storing and maintaining the preconditioner for the grafted method.
    Therefore, any additional modifications including gradient EMA/filtering and momentum are
    not included in grafting.

    """

    def __init__(self, param: Tensor):
        self._parameter_count = 0

    def update_preconditioners(self, grad: Tensor):
        pass

    def precondition(self, grad: Tensor) -> Tensor:
        pass

    def direction_norm(self, grad: Tensor) -> Tensor:
        pass

    def precondition_and_update(
        self,
        param: Tensor,
        grad: Tensor,
        lr: Union[float, Tensor],
    ):
        pass

    @property
    def parameter_count(self):
        return self._parameter_count

    def to(self, device: Union[None, torch.device] = None):
        return


class SGDGrafting(Grafting):
    """SGD grafting.

    Grafts the stochastic gradient method by returning the norm of the gradient.

    Args:
        param (Tensor): Parameter of interest.

    """

    def __init__(self, param: Tensor):
        super(SGDGrafting, self).__init__(param)

    def precondition(self, grad: Tensor) -> Tensor:
        return grad

    def direction_norm(self, grad: Tensor) -> Tensor:
        return torch.linalg.norm(grad)

    def precondition_and_update(self, param, grad: Tensor, lr: Union[float, Tensor]):
        param.add_(grad, alpha=-lr)


class AdagradGrafting(Grafting):
    """Adagrad grafting.

    Supports RMSProp and Adam by determining beta2 and use_bias_correction.

    Note: beta1 is not included since that is shared between both Shampoo and the grafted optimizer.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)
        normalize_gradient (bool): Flag for normalizing the gradient. (Default: False)

    """

    def __init__(
        self,
        param: Tensor,
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
        normalize_gradient: bool = False,
    ):
        super(AdagradGrafting, self).__init__(param)
        self._preconditioner = AdagradPreconditioner(
            param, beta2=beta2, epsilon=epsilon, use_bias_correction=use_bias_correction
        )
        self.normalize_gradient = normalize_gradient
        self._parameter_count += self._preconditioner.parameter_count

    def _normalize_grad(self, grad: Tensor) -> Tensor:
        return grad / torch.norm(grad) if self.normalize_gradient else grad

    def update_preconditioners(self, grad: Tensor):
        self._preconditioner.update_preconditioners(self._normalize_grad(grad))

    def precondition(self, grad: Tensor) -> Tensor:
        return self._preconditioner.precondition(grad)

    def direction_norm(self, grad: Tensor) -> Tensor:
        return self._preconditioner.compute_norm(grad)

    def precondition_and_update(self, param, grad: Tensor, lr: Union[float, Tensor]):
        self._preconditioner.precondition_and_update(param, grad, lr)

    def to(self, device: Union[None, torch.device] = None):
        if device is not None:
            self._preconditioner.to(device=device)
        return


class RMSPropGrafting(AdagradGrafting):
    """RMSProp grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)

    """

    def __init__(self, param, beta2: float = 0.99, epsilon: float = 1e-8):
        super(RMSPropGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=False,
            normalize_gradient=False,
        )


class AdamGrafting(AdagradGrafting):
    """Adam grafting.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)

    """

    def __init__(self, param, beta2: float = 0.999, epsilon: float = 1e-8):
        super(AdamGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=True,
            normalize_gradient=False,
        )


class AdagradNormalizedGrafting(AdagradGrafting):
    """RMSProp grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)

    """

    def __init__(self, param, epsilon: float = 1e-10):
        super(AdagradNormalizedGrafting, self).__init__(
            param=param,
            beta2=1.0,
            epsilon=epsilon,
            use_bias_correction=False,
            normalize_gradient=True,
        )


class RMSPropNormalizedGrafting(AdagradGrafting):
    """RMSProp grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)

    """

    def __init__(self, param, beta2: float = 0.99, epsilon: float = 1e-8):
        super(RMSPropNormalizedGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=False,
            normalize_gradient=True,
        )


class AdamNormalizedGrafting(AdagradGrafting):
    """Adam grafting with per-parameter normalized gradients.

    Args:
        param (Tensor): Parameter of interest.
        beta2 (float): Exponential moving average factor. If beta2 = 1., will use Adagrad update. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-8)

    """

    def __init__(self, param, beta2: float = 0.999, epsilon: float = 1e-8):
        super(AdamNormalizedGrafting, self).__init__(
            param=param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=True,
            normalize_gradient=True,
        )
