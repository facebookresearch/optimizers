"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial, reduce

from itertools import chain
from typing import Any, cast, Generic, TypeVar

import torch
from distributed_shampoo.shampoo_types import (
    PreconditionerConfig,
    PreconditionerValueError,
)
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size
from matrix_functions import check_diagonal, matrix_eigenvectors, matrix_inverse_root

from matrix_functions_types import EigenvectorConfig, RootInvConfig
from optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

ADAGRAD = "adagrad"
SHAMPOO = "shampoo"


class PreconditionerList(ABC):
    """Preconditioner base class.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._numel_list: tuple[int, ...] = (0,) * len(block_list)
        self._dims_list: tuple[torch.Size, ...] = tuple(
            block.size() for block in block_list
        )
        self._num_bytes_list: tuple[int, ...] = (0,) * len(block_list)

    @abstractmethod
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None: ...

    @abstractmethod
    def precondition(
        self, masked_grad_list: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]: ...

    @abstractmethod
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None: ...

    @property
    def numel_list(self) -> tuple[int, ...]:
        return self._numel_list

    @property
    def dims_list(self) -> tuple[torch.Size, ...]:
        return self._dims_list

    @property
    def num_bytes_list(self) -> tuple[int, ...]:
        return self._num_bytes_list

    def numel(self) -> int:
        return sum(self._numel_list)

    def num_bytes(self) -> int:
        return sum(self._num_bytes_list)


class SGDPreconditionerList(PreconditionerList):
    """SGD (identity) preconditioners for a list of parameters.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
    ) -> None:
        super().__init__(block_list)

    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        return

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return masked_grad_list

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        return


class AdagradPreconditionerList(PreconditionerList):
    """Adagrad / Adam / RMSProp preconditioners for a list of parameters.

    Operations are performed in-place with foreach operators.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSProp, set beta2 = 0.999.
    To enable Adam, set beta2 = 0.999, use_bias_correction = True.

    Other variants can also be specified.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, Any]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        beta2 (float): Exponential moving average factor for Adam/RMSprop second moment state. If beta2 = 1., will use
            unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        # type: ignore
        state: Mapping[Tensor, Any],
        block_info_list: tuple[BlockInfo, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        use_bias_correction: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Instantiate scalar hyperparameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Instantiate (blocked) AdaGrad preconditioners and construct preconditioner list.
        # NOTE: We need to instantiate the AdaGrad preconditioner states within the optimizer's state dictionary,
        # and do not explicitly store them as AdagradPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but AdagradPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        preconditioner_list: list[Tensor] = []
        for block, block_info in zip(block_list, block_info_list, strict=True):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Instantiate AdaGrad optimizer state for this block.
            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[ADAGRAD] = block_info.allocate_zeros_tensor(
                size=block.size(),
                dtype=block.dtype,
                device=block.device,
            )
            preconditioner_list.append(block_info.get_tensor(block_state[ADAGRAD]))

            logger.info(
                f"Instantiated Adagrad Preconditioner {preconditioner_index} ({block_state[ADAGRAD].shape} with dtype {block_state[ADAGRAD].dtype}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._local_preconditioner_list: tuple[Tensor, ...] = tuple(preconditioner_list)
        self._masked_preconditioner_list: tuple[Tensor, ...] = (
            self._local_preconditioner_list
        )

        # Construct lists of numels and bytes for logging purposes.
        self._numel_list: tuple[int, ...] = tuple(
            preconditioner.numel() for preconditioner in self._local_preconditioner_list
        )
        self._num_bytes_list: tuple[int, ...] = tuple(
            preconditioner.numel() * preconditioner.element_size()
            for preconditioner in self._local_preconditioner_list
        )

    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            if self._beta2 == 1.0:
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list,
                    masked_grad_list,
                    masked_grad_list,
                    value=1.0,
                )
            else:
                torch._foreach_mul_(self._masked_preconditioner_list, self._beta2)
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list,
                    masked_grad_list,
                    masked_grad_list,
                    value=1 - self._beta2,
                )

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions the gradient list using the AdaGrad preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A tuple of gradients with None values removed.

        Returns:
            tuple[Tensor, ...]: A tuple of preconditioned gradients.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            masked_bias_corrected_preconditioner_list = torch._foreach_div(
                self._masked_preconditioner_list,
                self._bias_correction2,
            )
            torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
            torch._foreach_add_(
                masked_bias_corrected_preconditioner_list, self._epsilon
            )
            return torch._foreach_div(
                masked_grad_list, masked_bias_corrected_preconditioner_list
            )

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_preconditioner_list = compress_list(
                self._local_preconditioner_list, local_grad_selector
            )


@dataclass
class BaseShampooKroneckerFactors(OptimizerModule):
    """Base class for Shampoo Kronecker factors."""

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]
    is_factor_matrices_diagonal: tuple[Tensor, ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)
        self.is_factor_matrices_diagonal = tuple(
            torch.tensor(True) for _ in range(len(self.factor_matrices))
        )


@dataclass
class ShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Shampoo Kronecker factors (wrapped) for storing in the optimizer state."""

    inv_factor_matrices: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass
class ShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Shampoo Kronecker factors (unwrapped) for operations during optimizer computation."""

    inv_factor_matrices: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass
class EigenvalueCorrectedShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Eigenvalue-corrected Shampoo Kronecker factors (wrapped) for storing in the optimizer state."""

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


@dataclass
class EigenvalueCorrectedShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Eigenvalue-corrected Shampoo Kronecker factors (unwrapped) for operations during optimizer computation."""

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


ShampooKroneckerFactorsListType = TypeVar(
    "ShampooKroneckerFactorsListType",
    ShampooKroneckerFactorsList,
    EigenvalueCorrectedShampooKroneckerFactorsList,
)


class BaseShampooPreconditionerList(
    PreconditionerList, Generic[ShampooKroneckerFactorsListType]
):
    """Base class for Shampoo preconditioners.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, Any]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        preconditioner_config (PreconditionerConfig): Configuration for preconditioner computation. (Default: DefaultShampooConfig)
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        inv_root_override (int | tuple[int, ...]): Inverse root to use in Shampoo. If a list [l0, l1, l2, ..., lp], then we will
            use -1 / l0 for 0-D tensors (scalars), -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on.
            If the order of the tensor exceeds the length of the list, we revert to using the default value. If 0 is used, uses the
            default inverse root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        factor_matrix_dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        # type: ignore
        state: Mapping[Tensor, Any],
        block_info_list: tuple[BlockInfo, ...],
        preconditioner_config: PreconditionerConfig,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        inv_root_override: int | tuple[int, ...] = 0,
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._preconditioner_config = preconditioner_config
        self._beta2 = beta2
        self._epsilon = epsilon
        self._inv_root_override = inv_root_override
        self._factor_matrix_dtype = factor_matrix_dtype
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Create the Kronecker factors.
        kronecker_factors_list: list[ShampooKroneckerFactorsListType] = (
            self._create_kronecker_factors_state(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
            )
        )

        # Initialize state lists.
        self._initialize_state_lists(
            block_list=block_list,
            kronecker_factors_list=kronecker_factors_list,
        )

    def _create_base_kronecker_factors(
        self,
        block_info: BlockInfo,
        dims: torch.Size,
    ) -> BaseShampooKroneckerFactors:
        """
        Creates a BaseShampooKroneckerFactor object for a given block.

        Args:
            block_info (BlockInfo): The BlockInfo object containing information about the block.
            dims (torch.Size): The dimensions of the block.

        Returns:
            kronecker_factors_state (BaseShampooKroneckerFactors): An object containing the Kronecker factors for the block.
        """
        factor_matrices = tuple(
            block_info.allocate_zeros_tensor(
                size=(dim, dim),
                dtype=self._factor_matrix_dtype,
                device=block_info.param.device,
            )
            for dim in dims
        )

        param_index, block_index = block_info.composable_block_ids
        factor_matrix_indices = tuple(
            ".".join((str(param_index), str(block_index), str(k)))
            for k in range(len(dims))
        )
        return BaseShampooKroneckerFactors(
            factor_matrices=factor_matrices,
            factor_matrix_indices=factor_matrix_indices,
        )

    @abstractmethod
    def _create_kronecker_factors_state_for_block(
        self,
        block: Tensor,
        block_info: BlockInfo,
        dims: torch.Size,
    ) -> ShampooKroneckerFactorsState | EigenvalueCorrectedShampooKroneckerFactorsState:
        """
        Creates a Kronecker factors state object for a given block.

        Args:
            block (Tensor): The block of the parameter.
            block_info (BlockInfo): The BlockInfo object containing information about the block.
            dims (torch.Size): The dimensions of the block.

        Returns:
            kronecker_factors_state (ShampooKroneckerFactorsState | EigenvalueCorrectedShampooKroneckerFactorsState): An object containing the Kronecker factors for the block.
        """
        ...

    @abstractmethod
    def _create_kronecker_factors_list(
        self,
        kronecker_factors_state: ShampooKroneckerFactorsState
        | EigenvalueCorrectedShampooKroneckerFactorsState,
        block_info: BlockInfo,
    ) -> ShampooKroneckerFactorsListType:
        """
        Creates a ShampooKroneckerFactorsList object from the given ShampooKroneckerFactorsState.

        Args:
            kronecker_factors_state (ShampooKroneckerFactorsState | EigenvalueCorrectedShampooKroneckerFactorsState): The state containing the Kronecker factors.
            block_info (BlockInfo): The BlockInfo object containing information about the block.

        Returns:
            kronecker_factors_list (ShampooKroneckerFactorsListType): A list of ShampooKroneckerFactors objects.
        """
        ...

    def _create_kronecker_factors_state(
        self,
        block_list: tuple[Tensor, ...],
        # type: ignore
        state: Mapping[Tensor, Any],
        block_info_list: tuple[BlockInfo, ...],
    ) -> list[ShampooKroneckerFactorsListType]:
        # Instantiate (blocked) Kronecker factors and construct list of Kronecker factors.
        # NOTE: We need to instantiate the Kronecker factor states within the optimizer's state dictionary,
        # and do not explicitly store them as ShampooPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but ShampooPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        kronecker_factors_list = []
        for block, block_info, dims in zip(
            block_list, block_info_list, self._dims_list, strict=True
        ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            block_state[SHAMPOO] = self._create_kronecker_factors_state_for_block(
                block=block, block_info=block_info, dims=dims
            )

            kronecker_factors_list.append(
                self._create_kronecker_factors_list(block_state[SHAMPOO], block_info)
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {str(param_index) + '.' + str(block_index)} for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        return kronecker_factors_list

    @abstractmethod
    def _get_inverse_roots_from_override(
        self,
        inv_root_override: int | Sequence[int],
        order_list: tuple[int, ...],
    ) -> tuple[int, ...]:
        """
        Retrieves the inverse roots from the override parameter.

        Args:
            inv_root_override (int | Sequence[int]): The override value for the inverse root.
            order_list (tuple[int, ...]): A list of orders for each tensor in the preconditioner.

        Returns:
            tuple[int, ...]: A list of inverse roots for each tensor in the preconditioner.
        """
        ...

    @staticmethod
    def _get_inverse_roots_from_override_with_high_order_default(
        inv_root_override: int | Sequence[int],
        order_list: tuple[int, ...],
        high_order_default: Callable[[int], int],
    ) -> tuple[int, ...]:
        """Retrieves the appropriate root from the inverse root override parameter
        for a list of tensor orders.

        For example, suppose inv_root_override = (2, 1, 4, 3).
        If order = 0, then we will return 2;
        If order = 1, then we will return 1;
        If order = 2, then we will return 4;
        If order = 3, then we will return 3;
        If order > 3, then we will return high_order_default(order).

        Args:
            inv_root_override (int | Sequence[int]): Inverse root override int or list.
            order_list (tuple[int, ...]): List of orders for their corresponding tensors.
            higher_order_default (Callable[[int], int]): Function for computing the inverse root for orders greater than the length of the inverse root override list.

        Returns:
            root_list (int): Inverse roots to use in Shampoo for a list of tensors.

        """
        if isinstance(inv_root_override, Sequence):
            return tuple(
                (
                    high_order_default(order)
                    if order >= len(inv_root_override)
                    else inv_root_override[order]
                )
                for order in order_list
            )
        else:
            return (
                tuple(high_order_default(order) for order in order_list)
                if inv_root_override == 0
                else (inv_root_override,) * len(order_list)
            )

    @abstractmethod
    def _amortized_computation(self) -> None:
        """
        Computes the amortized computation needed for each Shampoo preconditioner implementation.
        This amortized computation is computation heavy work that cannot be done for each step.
        """
        ...

    @staticmethod
    def _check_factor_matrix_for_diagonality_nan_and_inf(
        factor_matrix: Tensor,
        is_factor_matrix_diagonal: Tensor,
        factor_matrix_index: str,
    ) -> None:
        # For tracking diagonality of the factor matrix.
        # Checks if the factor matrix is currently diagonal, then checks whether or not
        # the update factor matrix is diagonal.
        if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
            is_factor_matrix_diagonal.copy_(torch.tensor(False))
            logger.debug(f"Factor matrix {factor_matrix_index} is not diagonal.")

        # Check for nan or inf values.
        if torch.isnan(factor_matrix).any():
            raise PreconditionerValueError(
                f"Encountered nan values in factor matrix {factor_matrix_index}! "
                f"To mitigate, check if nan inputs are being passed into the network or nan gradients "
                f"are being passed to the optimizer."
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )
        if torch.isinf(factor_matrix).any():
            raise PreconditionerValueError(
                f"Encountered inf values in factor matrix {factor_matrix_index}! "
                f"In some cases, this may be due to divergence of the algorithm. "
                f"To mitigate, try decreasing the learning rate or increasing grafting epsilon."
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )

    def _raise_exception_if_failure_tolerance_exceeded(
        self,
        success_tracker: list[bool],
        preconditioner_index: int,
        exception: Exception,
    ) -> None:
        """Raises an exception if the number of failed amortized computations exceeds the tolerance.

        Resets the counter at the given index when all amortized computations are successful.

        Args:
            success_tracker (list[bool]): A list of booleans indicating whether the amortized computation was successful.
            preconditioner_index (int): The index of the preconditioner.
            exception (Exception): The exception to raise.

        Raises:
            exception (Exception): The exception to raise.

        """
        if all(success_tracker):
            # Reset counter for failed amortized computations.
            self._masked_failed_amortized_computation_counter_list[
                preconditioner_index
            ] = 0
        else:
            # Increment counter for failed amortized computations.
            self._masked_failed_amortized_computation_counter_list[
                preconditioner_index
            ] += 1
            # Raise the exception if the tolerance at the given index is exceeded.
            failure_counter = self._masked_failed_amortized_computation_counter_list[
                preconditioner_index
            ]
            tolerance = (
                self._preconditioner_config.num_tolerated_failed_amortized_computations
            )
            if failure_counter > tolerance:
                raise exception

    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        """
        Updates the preconditioners.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.
            step (Tensor): The current step.
            perform_amortized_computation (bool): Whether to perform an amortized computation.

        Returns:
            None
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            # Update the Kronecker factor matrices.
            self._update_factor_matrices(masked_grad_list=masked_grad_list)

            # Update bias correction term based on step.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

            # In Shampoo, this is equivalent to computing the inverse factor matrix.
            # In Eigenvalue-Corrected Shampoo, this is equivalent to computing the eigenvector of the factor matrix.
            if perform_amortized_computation:
                self._amortized_computation()

    def _initialize_state_lists(
        self,
        block_list: tuple[Tensor, ...],
        kronecker_factors_list: list[ShampooKroneckerFactorsListType],
    ) -> None:
        # Initialize local lists.
        self._local_kronecker_factors_list: tuple[
            ShampooKroneckerFactorsListType,
            ...,
        ] = tuple(kronecker_factors_list)
        self._local_order_list: tuple[int, ...] = tuple(
            block.dim() for block in block_list
        )
        self._local_root_list: tuple[int, ...] = self._get_inverse_roots_from_override(
            self._inv_root_override,
            self._local_order_list,
        )
        self._local_failed_amortized_computation_counter_list: list[int] = [0] * len(
            self._local_kronecker_factors_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: tuple[int, ...] = self._local_order_list
        self._masked_root_list: tuple[int, ...] = self._local_root_list
        self._masked_failed_amortized_computation_counter_list: list[int] = (
            self._local_failed_amortized_computation_counter_list
        )
        self._masked_kronecker_factors_list: tuple[
            ShampooKroneckerFactorsListType,
            ...,
        ] = self._local_kronecker_factors_list

        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._numel_list: tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list: tuple[int, ...] = tuple(
            numel
            * (get_dtype_size(self._factor_matrix_dtype) + get_dtype_size(block.dtype))
            // 2
            for numel, block in zip(self._numel_list, block_list, strict=True)
        )

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_order_list: tuple[int, ...] = compress_list(  # type: ignore[no-redef]
                self._local_order_list, local_grad_selector
            )
            self._masked_root_list: tuple[int, ...] = compress_list(  # type: ignore[no-redef]
                self._local_root_list, local_grad_selector
            )
            self._masked_failed_amortized_computation_counter_list: list[int] = (  # type: ignore[no-redef]
                list(
                    compress_list(
                        self._local_failed_amortized_computation_counter_list,
                        local_grad_selector,
                    )
                )
            )
            self._masked_kronecker_factors_list: tuple[  # type: ignore[no-redef]
                ShampooKroneckerFactorsListType,
                ...,
            ] = compress_list(self._local_kronecker_factors_list, local_grad_selector)

    def _update_factor_matrices(self, masked_grad_list: tuple[Tensor, ...]) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self._update_factor_matrices.__name__} ##"
        ):
            # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
            # We apply foreach operators onto the list of Kronecker factor matrices (as opposed to the
            # full list of gradients/optimizer states).
            for grad, order, kronecker_factors in zip(
                masked_grad_list,
                self._masked_order_list,
                self._masked_kronecker_factors_list,
                strict=True,
            ):
                # Scale Kronecker factors as a list.
                if self._beta2 != 1.0:
                    torch._foreach_mul_(kronecker_factors.factor_matrices, self._beta2)

                # Construct outer product list for updating Kronecker factors.
                outer_product_list = tuple(
                    torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(k), range(k + 1, order))]] * 2,  # type: ignore[has-type]
                    )
                    for k in range(order)
                )

                # Update Kronecker factors.
                torch._foreach_add_(
                    kronecker_factors.factor_matrices,
                    outer_product_list,
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )

    @staticmethod
    def _precondition_grad(
        grad: Tensor,
        preconditioner_list: tuple[Tensor, ...],
        dims: tuple[list[int], list[int]] = ([0], [0]),
    ) -> Tensor:
        return reduce(partial(torch.tensordot, dims=dims), preconditioner_list, grad)


class ShampooPreconditionerList(
    BaseShampooPreconditionerList[ShampooKroneckerFactorsList]
):
    """Shampoo preconditioners for list of parameters."""

    def _create_kronecker_factors_state_for_block(
        self, block: Tensor, block_info: BlockInfo, dims: torch.Size
    ) -> ShampooKroneckerFactorsState:
        inv_factor_matrices = tuple(
            block_info.allocate_zeros_tensor(
                size=(dim, dim),
                dtype=block.dtype,
                device=block_info.param.device,
            )
            for dim in dims
        )

        base_kronecker_factors = self._create_base_kronecker_factors(
            block_info=block_info, dims=dims
        )
        return ShampooKroneckerFactorsState(
            factor_matrices=base_kronecker_factors.factor_matrices,
            factor_matrix_indices=base_kronecker_factors.factor_matrix_indices,
            inv_factor_matrices=inv_factor_matrices,
        )

    def _create_kronecker_factors_list(
        self,
        kronecker_factors_state: ShampooKroneckerFactorsState
        | EigenvalueCorrectedShampooKroneckerFactorsState,
        block_info: BlockInfo,
    ) -> ShampooKroneckerFactorsList:
        assert isinstance(kronecker_factors_state, ShampooKroneckerFactorsState)
        return ShampooKroneckerFactorsList(
            factor_matrices=tuple(
                block_info.get_tensor(t)
                for t in kronecker_factors_state.factor_matrices
            ),
            inv_factor_matrices=tuple(
                block_info.get_tensor(t)
                for t in kronecker_factors_state.inv_factor_matrices
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
        )

    def _get_inverse_roots_from_override(
        self,
        inv_root_override: int | Sequence[int],
        order_list: tuple[int, ...],
    ) -> tuple[int, ...]:
        return BaseShampooPreconditionerList._get_inverse_roots_from_override_with_high_order_default(
            inv_root_override, order_list, high_order_default=lambda order: 2 * order
        )

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the Shampoo preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            tuple[Tensor, ...]: A list of preconditioned gradients.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            return tuple(
                self._precondition_grad(
                    grad=masked_grad,
                    preconditioner_list=kronecker_factors.inv_factor_matrices,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )

    @torch.compiler.disable
    def _amortized_computation(self) -> None:
        # NOTE: This function currently only computes the matrix root inverse based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self._amortized_computation.__name__} ##"
        ):
            for idx, (kronecker_factors, root) in enumerate(
                zip(
                    self._masked_kronecker_factors_list,
                    self._masked_root_list,
                    strict=True,
                )
            ):
                success_tracker: list[bool] = []
                for (
                    factor_matrix,
                    inv_factor_matrix,
                    is_factor_matrix_diagonal,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.inv_factor_matrices,
                    kronecker_factors.is_factor_matrices_diagonal,
                    kronecker_factors.factor_matrix_indices,
                    strict=True,
                ):
                    # Add epsilon term and incorporate bias correction.
                    bias_corrected_factor_matrix = (
                        factor_matrix / self._bias_correction2
                    )

                    BaseShampooPreconditionerList._check_factor_matrix_for_diagonality_nan_and_inf(
                        factor_matrix=bias_corrected_factor_matrix,
                        is_factor_matrix_diagonal=is_factor_matrix_diagonal,
                        factor_matrix_index=factor_matrix_index,
                    )

                    # Compute inverse preconditioner.
                    root_inv_config = cast(
                        RootInvConfig,
                        self._preconditioner_config.amortized_computation_config,
                    )
                    try:
                        computed_inv_factor_matrix = matrix_inverse_root(
                            A=bias_corrected_factor_matrix,
                            root=Fraction(
                                root
                                / getattr(
                                    root_inv_config,
                                    "exponent_multiplier",
                                    1,
                                )
                            ),
                            root_inv_config=root_inv_config,
                            epsilon=self._epsilon,
                            is_diagonal=bool(is_factor_matrix_diagonal),
                        ).to(dtype=inv_factor_matrix.dtype)
                        # Add success to success tracker.
                        success_tracker.append(True)
                    except Exception as exception:
                        # Add failure to success tracker.
                        success_tracker.append(False)
                        logger.warning(
                            f"Matrix computation failed for factor matrix {factor_matrix_index} "
                            f"with {exception=}. Using previous inverted factor matrix and continuing..."
                        )
                        # Define computed_inv_factor_matrix to prevent undefined local variable error.
                        computed_inv_factor_matrix = inv_factor_matrix

                    # Check if we encounter NaN or inf values in computed inverse matrix.
                    if (
                        torch.isnan(computed_inv_factor_matrix).any()
                        or torch.isinf(computed_inv_factor_matrix).any()
                    ):
                        torch.set_printoptions(threshold=100_000)
                        raise PreconditionerValueError(
                            f"Encountered nan or inf values in inverse factor matrix {factor_matrix_index}! "
                            f"To mitigate, check factor matrix before the matrix computation: {bias_corrected_factor_matrix=}"
                        )
                    inv_factor_matrix.copy_(computed_inv_factor_matrix)

                # Only reuse previous inverse roots if tolerance is not exceeded.
                self._raise_exception_if_failure_tolerance_exceeded(
                    success_tracker=success_tracker,
                    preconditioner_index=idx,
                    exception=ValueError(
                        f"The number of failed inverse root computations for factors {kronecker_factors.factor_matrix_indices} exceeded the allowed tolerance."
                    ),
                )


class EigenvalueCorrectedShampooPreconditionerList(
    BaseShampooPreconditionerList[EigenvalueCorrectedShampooKroneckerFactorsList]
):
    """Eigenvalue-corrected Shampoo preconditioners for list of parameters."""

    def _create_kronecker_factors_state_for_block(
        self, block: Tensor, block_info: BlockInfo, dims: torch.Size
    ) -> EigenvalueCorrectedShampooKroneckerFactorsState:
        factor_matrices_eigenvectors = tuple(
            block_info.allocate_zeros_tensor(
                size=(dim, dim),
                dtype=block.dtype,
                device=block_info.param.device,
            )
            for dim in dims
        )
        corrected_eigenvalues = block_info.allocate_zeros_tensor(
            size=tuple(dims),
            dtype=block.dtype,
            device=block_info.param.device,
        )

        base_kronecker_factors = self._create_base_kronecker_factors(
            block_info=block_info, dims=dims
        )
        return EigenvalueCorrectedShampooKroneckerFactorsState(
            factor_matrices=base_kronecker_factors.factor_matrices,
            factor_matrices_eigenvectors=factor_matrices_eigenvectors,
            corrected_eigenvalues=corrected_eigenvalues,
            factor_matrix_indices=base_kronecker_factors.factor_matrix_indices,
        )

    def _create_kronecker_factors_list(
        self,
        kronecker_factors_state: ShampooKroneckerFactorsState
        | EigenvalueCorrectedShampooKroneckerFactorsState,
        block_info: BlockInfo,
    ) -> EigenvalueCorrectedShampooKroneckerFactorsList:
        assert isinstance(
            kronecker_factors_state, EigenvalueCorrectedShampooKroneckerFactorsState
        )
        return EigenvalueCorrectedShampooKroneckerFactorsList(
            factor_matrices=tuple(
                block_info.get_tensor(t)
                for t in kronecker_factors_state.factor_matrices
            ),
            factor_matrices_eigenvectors=tuple(
                block_info.get_tensor(t)
                for t in kronecker_factors_state.factor_matrices_eigenvectors
            ),
            corrected_eigenvalues=block_info.get_tensor(
                kronecker_factors_state.corrected_eigenvalues
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
        )

    def _get_inverse_roots_from_override(
        self,
        inv_root_override: int | Sequence[int],
        order_list: tuple[int, ...],
    ) -> tuple[int, ...]:
        return BaseShampooPreconditionerList._get_inverse_roots_from_override_with_high_order_default(
            inv_root_override, order_list, high_order_default=lambda order: 2
        )

    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool,
    ) -> None:
        """
        Updates the preconditioners.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.
            step (Tensor): The current step.
            perform_amortized_computation (bool): Whether to perform an amortized computation.

        Returns:
            None
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            super().update_preconditioners(
                masked_grad_list=masked_grad_list,
                step=step,
                perform_amortized_computation=perform_amortized_computation,
            )
            # Update the eigenvalue corrections of Shampoo's preconditioner.
            self._update_eigenvalue_corrections(masked_grad_list=masked_grad_list)

    def _update_eigenvalue_corrections(
        self, masked_grad_list: tuple[Tensor, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self._update_eigenvalue_corrections.__name__} ##"
        ):
            # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
            for grad, kronecker_factors in zip(
                masked_grad_list,
                self._masked_kronecker_factors_list,
                strict=True,
            ):
                factor_eigenvectors = kronecker_factors.factor_matrices_eigenvectors
                if factor_eigenvectors[0].any():
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioner_list=factor_eigenvectors,
                    )
                # Scale corrected eigenvalues.
                # NOTE: The case when self._beta2 == 1.0 is not well tested and might not be stable.
                if self._beta2 != 1.0:
                    kronecker_factors.corrected_eigenvalues.mul_(self._beta2)
                # Update corrected eigenvalues (squared gradient in eigenbasis of Shampoo preconditioner).
                kronecker_factors.corrected_eigenvalues.add_(
                    grad.square(),
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the Eigenvalue-Corrected Shampoo preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            tuple[Tensor, ...]: A list of preconditioned gradients.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            preconditioned_grad_list = []
            for masked_grad, kronecker_factors, root in zip(
                masked_grad_list,
                self._masked_kronecker_factors_list,
                self._masked_root_list,
                strict=True,
            ):
                factor_eigenvectors = kronecker_factors.factor_matrices_eigenvectors
                corrected_eigenvalues = kronecker_factors.corrected_eigenvalues
                use_eigenbasis = factor_eigenvectors[0].any()
                grad = masked_grad.clone()
                if use_eigenbasis:
                    # Convert to eigenbasis of Shampoo factor matrices.
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioner_list=factor_eigenvectors,
                    )

                # Precondition with inverse root of corrected eigenvalues.
                grad.div_(
                    corrected_eigenvalues.div(self._bias_correction2)
                    .add_(self._epsilon)
                    .pow_(1 / root)
                )
                if use_eigenbasis:
                    # Convert back to basis of the parameters.
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioner_list=factor_eigenvectors,
                        dims=([0], [1]),
                    )
                preconditioned_grad_list.append(grad)
            return tuple(preconditioned_grad_list)

    @torch.compiler.disable
    def _amortized_computation(self) -> None:
        # NOTE: This function currently only computes the preconditioner eigenvectors based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self._amortized_computation.__name__} ##"
        ):
            for idx, kronecker_factors in enumerate(
                self._masked_kronecker_factors_list
            ):
                success_tracker: list[bool] = []
                for (
                    factor_matrix,
                    factor_matrix_eigenvectors,
                    is_factor_matrix_diagonal,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.factor_matrices_eigenvectors,
                    kronecker_factors.is_factor_matrices_diagonal,
                    kronecker_factors.factor_matrix_indices,
                    strict=True,
                ):
                    BaseShampooPreconditionerList._check_factor_matrix_for_diagonality_nan_and_inf(
                        factor_matrix=factor_matrix,
                        is_factor_matrix_diagonal=is_factor_matrix_diagonal,
                        factor_matrix_index=factor_matrix_index,
                    )

                    # Compute eigenvectors of factor matrix.
                    eigenvector_computation_config = cast(
                        EigenvectorConfig,
                        self._preconditioner_config.amortized_computation_config,
                    )
                    try:
                        computed_eigenvectors = matrix_eigenvectors(
                            A=factor_matrix,
                            eigenvectors_estimate=factor_matrix_eigenvectors,
                            eigenvector_computation_config=eigenvector_computation_config,
                            is_diagonal=bool(is_factor_matrix_diagonal),
                        )
                        # Add success to success tracker.
                        success_tracker.append(True)
                    except Exception as exception:
                        # Add failure to success tracker.
                        success_tracker.append(False)
                        logger.warning(
                            f"Matrix computation failed for factor matrix {factor_matrix_index} "
                            f"with {exception=}. Using previous factor matrix eigenvectors and continuing..."
                        )
                        # Define computed_eigenvectors to prevent undefined local variable error.
                        computed_eigenvectors = factor_matrix_eigenvectors

                    # Check if we encounter NaN or inf values in computed eigenvectors.
                    if (
                        torch.isnan(computed_eigenvectors).any()
                        or torch.isinf(computed_eigenvectors).any()
                    ):
                        torch.set_printoptions(threshold=100_000)
                        raise PreconditionerValueError(
                            f"Encountered nan or inf values in eigenvectors of factor matrix {factor_matrix_index}! "
                            f"To mitigate, check factor matrix before the matrix computation: {factor_matrix=}"
                        )
                    factor_matrix_eigenvectors.copy_(computed_eigenvectors)

                # Only reuse previous eigenvectors if tolerance is not exceeded.
                self._raise_exception_if_failure_tolerance_exceeded(
                    success_tracker=success_tracker,
                    preconditioner_index=idx,
                    exception=ValueError(
                        f"The number of failed eigenvector computations for factors {kronecker_factors.factor_matrix_indices} exceeded the allowed tolerance."
                    ),
                )
