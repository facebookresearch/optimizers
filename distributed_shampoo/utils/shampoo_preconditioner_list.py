"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from itertools import chain
from types import TracebackType
from typing import Any, DefaultDict, Optional, Sequence, Tuple, Type, Union

import torch
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_quantization import (
    QuantizedTensor,
    QuantizedTensorList,
)
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size

from matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
)
from optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

ADAGRAD = "adagrad"
SHAMPOO = "shampoo"


class PreconditionerList(ABC):
    """Preconditioner base class.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__()
        self._numel_list: Tuple[int, ...] = (0,) * len(block_list)
        self._dims_list: Tuple[torch.Size, ...] = tuple(
            block.size() for block in block_list
        )
        self._num_bytes_list: Tuple[int, ...] = (0,) * len(block_list)

    @abstractmethod
    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None: ...

    @abstractmethod
    def precondition(
        self, masked_grad_list: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]: ...

    @abstractmethod
    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None: ...

    @abstractmethod
    def dequantize_preconditioners(self) -> None: ...

    @abstractmethod
    def quantize_preconditioners(self) -> None: ...

    @property
    def numel_list(self) -> Tuple[int, ...]:
        return self._numel_list

    @property
    def dims_list(self) -> Tuple[torch.Size, ...]:
        return self._dims_list

    @property
    def num_bytes_list(self) -> Tuple[int, ...]:
        return self._num_bytes_list

    def numel(self) -> int:
        return sum(self._numel_list)

    def num_bytes(self) -> int:
        return sum(self._num_bytes_list)


class SGDPreconditionerList(PreconditionerList):
    """SGD (identity) preconditioners for a list of parameters.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
    ) -> None:
        super().__init__(block_list)

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        return

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        return masked_grad_list

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        return

    def dequantize_preconditioners(self) -> None:
        return

    def quantize_preconditioners(self) -> None:
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
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Parameter, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Adam/RMSprop second moment state. If beta2 = 1., will use
            unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        use_bias_correction (bool): Flag for using bias correction. (Default: False)

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-10,
        preconditioner_dtype: torch.dtype = torch.float32,
        computation_dtype: torch.dtype = torch.float32,
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
        preconditioner_list = []
        for block, block_info in zip(block_list, block_info_list, strict=True):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]

            # Instantiate AdaGrad optimizer state for this block.
            preconditioner_index = str(param_index) + "." + str(block_index)
            block_state[ADAGRAD] = QuantizedTensor(
                block_info.allocate_zeros_tensor(
                    block.size(), preconditioner_dtype, block.device
                ),
                block_info,
            )
            preconditioner_list.append(block_state[ADAGRAD])

            logger.info(
                f"Instantiated Adagrad Preconditioner {preconditioner_index} ({block_state[ADAGRAD].quantized_values.shape} with dtype {block_state[ADAGRAD].quantized_values.dtype}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._local_preconditioner_list = QuantizedTensorList(
            compress_list(preconditioner_list, distributor_selector),
            preconditioner_dtype,
            computation_dtype,
        )
        self._masked_preconditioner_list: QuantizedTensorList = (
            self._local_preconditioner_list
        )

        # Construct lists of bytes and numels for logging purposes.
        self._numel_list: Tuple[int, ...] = tuple(
            preconditioner.quantized_values.numel()
            for preconditioner in preconditioner_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            preconditioner.quantized_values.numel()
            * preconditioner.quantized_values.element_size()
            for preconditioner in preconditioner_list
        )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            if self._beta2 == 1.0:
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list.dequantized_value,
                    masked_grad_list,
                    masked_grad_list,
                    value=1.0,
                )
            else:
                torch._foreach_mul_(
                    self._masked_preconditioner_list.dequantized_value, self._beta2
                )
                torch._foreach_addcmul_(
                    self._masked_preconditioner_list.dequantized_value,
                    masked_grad_list,
                    masked_grad_list,
                    value=1 - self._beta2,
                )

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            masked_bias_corrected_preconditioner_list = torch._foreach_div(
                self._masked_preconditioner_list.dequantized_value,
                self._bias_correction2,
            )
            torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
            torch._foreach_add_(
                masked_bias_corrected_preconditioner_list, self._epsilon
            )
            return torch._foreach_div(
                masked_grad_list, masked_bias_corrected_preconditioner_list
            )

    def dequantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.dequantize_preconditioners.__name__} ##"
        ):
            self._masked_preconditioner_list.dequantize_()

    def quantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.quantize_preconditioners.__name__} ##"
        ):
            self._masked_preconditioner_list.quantize_()

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_preconditioner_list = self._local_preconditioner_list.compress(
                local_grad_selector
            )


@dataclass
class ShampooKroneckerFactorsState(OptimizerModule):
    """Shampoo Kronecker Factors (wrapped) for storing in the optimizer state."""

    factor_matrices: Tuple[QuantizedTensor, ...]
    inv_factor_matrices: Tuple[QuantizedTensor, ...]
    factor_matrix_indices: Tuple[str, ...]
    is_factor_matrices_diagonal: Tuple[Tensor, ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        assert (
            len(self.factor_matrices)
            == len(self.inv_factor_matrices)
            == len(self.factor_matrix_indices)
        )
        self.is_factor_matrices_diagonal = tuple(
            torch.tensor(True) for _ in range(len(self.factor_matrices))
        )


@dataclass
class ShampooKroneckerFactorsList(OptimizerModule):
    """Shampoo Kronecker Factors (unwrapped) for operations during optimizer computation."""

    factor_matrices: QuantizedTensorList
    inv_factor_matrices: QuantizedTensorList
    factor_matrix_indices: Tuple[str, ...]
    is_factor_matrices_diagonal: Tuple[Tensor, ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        assert (
            len(self.factor_matrices)
            == len(self.inv_factor_matrices)
            == len(self.factor_matrix_indices)
        )
        self.is_factor_matrices_diagonal = tuple(
            torch.tensor(True) for _ in range(len(self.factor_matrices))
        )


class ShampooPreconditionerList(PreconditionerList):
    """Shampoo preconditioners for list of parameters.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Parameter, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        inv_root_override (int, Tuple[int, ...]): Inverse root to use in Shampoo. If a list [l0, l1, l2, ..., lp], then we will
            use -1 / l0 for 0-D tensors (scalars), -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on.
            If the order of the tensor exceeds the length of the list, we revert to using the default value. If 0 is used, uses the
            default inverse root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        factor_matrix_dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.

    """

    def __init__(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
        distributor_selector: Tuple[bool, ...],
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        inv_root_override: Union[int, Tuple[int, ...]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
        inv_factor_matrix_dtype: torch.dtype = torch.float,
        computation_dtype: torch.dtype = torch.float,
        use_protected_eigh: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._beta2 = beta2
        self._epsilon = epsilon
        self._inv_root_override = inv_root_override
        self._exponent_multiplier = exponent_multiplier
        self._factor_matrix_dtype = factor_matrix_dtype
        self._inv_factor_matrix_dtype = inv_factor_matrix_dtype
        self._computation_dtype = computation_dtype
        self._use_bias_correction = use_bias_correction
        self._use_protected_eigh = use_protected_eigh
        self._bias_correction2: Tensor = torch.tensor(1.0)

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

            # Instantiate ShampooKroneckerFactors for this block.
            # The factor matrices are instantiated using the determined dtype.
            factor_matrices = tuple(
                QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (dim, dim),
                        self._factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )
            # The inverse factor matrices are instantiated using the dtype of the block / gradient.
            inv_factor_matrices = tuple(
                QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (dim, dim),
                        self._inv_factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )

            preconditioner_index = str(param_index) + "." + str(block_index)
            factor_matrix_indices = tuple(
                preconditioner_index + "." + str(k) for k in range(len(dims))
            )
            block_state[SHAMPOO] = ShampooKroneckerFactorsState(
                factor_matrices=factor_matrices,
                inv_factor_matrices=inv_factor_matrices,
                factor_matrix_indices=factor_matrix_indices,
            )
            kronecker_factors_list.append(
                ShampooKroneckerFactorsList(
                    factor_matrices=QuantizedTensorList(
                        factor_matrices,
                        self._factor_matrix_dtype,
                        self._computation_dtype,
                    ),
                    inv_factor_matrices=QuantizedTensorList(
                        inv_factor_matrices,
                        self._inv_factor_matrix_dtype,
                        self._computation_dtype,
                    ),
                    factor_matrix_indices=factor_matrix_indices,
                )
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                f"({[(factor_matrix.quantized_values.shape, factor_matrix.quantized_values.dtype) for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        # Initialize local lists.
        local_block_list = compress_list(block_list, distributor_selector)
        self._local_kronecker_factors_list: Tuple[ShampooKroneckerFactorsList, ...] = (
            compress_list(kronecker_factors_list, distributor_selector)
        )
        self._local_order_list: Tuple[int, ...] = tuple(
            block.dim() for block in local_block_list
        )
        self._local_root_list: Tuple[int, ...] = self._get_inverse_roots_from_override(
            self._inv_root_override, self._local_order_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: Tuple[int, ...] = self._local_order_list
        self._masked_root_list: Tuple[int, ...] = self._local_root_list
        self._masked_kronecker_factors_list: Tuple[ShampooKroneckerFactorsList, ...] = (
            self._local_kronecker_factors_list
        )

        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._numel_list: Tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            numel
            * (get_dtype_size(self._factor_matrix_dtype) + get_dtype_size(block.dtype))
            // 2
            for numel, block in zip(self._numel_list, local_block_list)
        )

    @staticmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]], order_list: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Retrieves the appropriate root from the inverse root override parameter
        for a list of tensor orders.

        For example, suppose inv_root_override = (2, 1, 4, 3).
        If order = 0, then we will return 2;
        If order = 1, then we will return 1;
        If order = 2, then we will return 4;
        If order = 3, then we will return 3;
        If order > 3, then we will return 2 * order.

        Args:
            inv_root_override (int, Sequence[int]): Inverse root override int or list.
            order_list (Tuple[int, ...]): List of orders for their corresponding tensors.

        Returns:
            root_list (int): Inverse roots to use in Shampoo for a list of tensors.

        """
        if isinstance(inv_root_override, Sequence):
            return tuple(
                (
                    2 * order
                    if order >= len(inv_root_override)
                    else inv_root_override[order]
                )
                for order in order_list
            )
        else:
            return (
                tuple(2 * order for order in order_list)
                if inv_root_override == 0
                else (inv_root_override,) * len(order_list)
            )

    def update_preconditioners(
        self, masked_grad_list: Tuple[Tensor, ...], step: Tensor
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
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
                    torch._foreach_mul_(
                        kronecker_factors.factor_matrices.dequantized_value, self._beta2
                    )

                # Construct outer product list for updating Kronecker factors.
                outer_product_list = tuple(
                    torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(k), range(k + 1, order))]] * 2,
                    )
                    for k in range(order)
                )

                # Update Kronecker factors.
                torch._foreach_add_(
                    kronecker_factors.factor_matrices.dequantized_value,
                    outer_product_list,
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )

            # Update bias correction term based on step list.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):

            def precondition_masked_grad(
                masked_grad: Tensor,
                inv_factor_matrices: Tuple[Tensor, ...],
            ) -> Tensor:
                for inv_factor_matrix in inv_factor_matrices:
                    masked_grad = torch.tensordot(
                        masked_grad, inv_factor_matrix, [[0], [0]]
                    )
                return masked_grad

            return tuple(
                precondition_masked_grad(
                    masked_grad=masked_grad,
                    inv_factor_matrices=kronecker_factors.inv_factor_matrices.dequantized_value,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )

    def compute_root_inverse(self) -> None:
        # NOTE: This function currently only computes the matrix root inverse based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compute_root_inverse.__name__} ##"
        ):
            for kronecker_factors, root in zip(
                self._masked_kronecker_factors_list,
                self._masked_root_list,
                strict=True,
            ):
                for (
                    factor_matrix,
                    inv_factor_matrix,
                    is_factor_matrix_diagonal,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices.dequantized_value,
                    kronecker_factors.inv_factor_matrices.dequantized_value,
                    kronecker_factors.is_factor_matrices_diagonal,
                    kronecker_factors.factor_matrix_indices,
                    strict=True,
                ):
                    # For tracking diagonality of the preconditioner.
                    # Checks if the preconditioner is currently diagonal, then checks whether or not
                    # the update matrix is diagonal.
                    if is_factor_matrix_diagonal and not check_diagonal(factor_matrix):
                        is_factor_matrix_diagonal.copy_(torch.tensor(False))
                        logger.debug(
                            f"Factor matrix {factor_matrix_index} is not diagonal."
                        )

                    # Add epsilon term and incorporate bias correction.
                    bias_corrected_factor_matrix = (
                        factor_matrix / self._bias_correction2
                    )

                    # Check for nan or inf values.
                    if torch.isnan(bias_corrected_factor_matrix).any():
                        raise ValueError(
                            f"Encountered nan values in bias-corrected factor matrix {factor_matrix_index}! "
                            f"To mitigate, check if nan inputs are being passed into the network or nan gradients "
                            f"are being passed to the optimizer."
                            f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                            f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                            f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
                        )
                    if torch.isinf(bias_corrected_factor_matrix).any():
                        raise ValueError(
                            f"Encountered inf values in bias-corrected factor matrix {factor_matrix_index}! "
                            f"In some cases, this may be due to divergence of the algorithm. "
                            f"To mitigate, try decreasing the learning rate or increasing grafting epsilon."
                            f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                            f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                            f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
                        )

                    # Compute inverse preconditioner.
                    # If reuse_previous_inv_factor_matrix is True, will reuse previous matrix if matrix
                    # inverse root computation fails.
                    try:
                        computed_inv_factor_matrix = matrix_inverse_root(
                            A=bias_corrected_factor_matrix,
                            root=root,
                            epsilon=self._epsilon,
                            exponent_multiplier=self._exponent_multiplier,
                            is_diagonal=is_factor_matrix_diagonal,
                            retry_double_precision=self._use_protected_eigh,
                        ).to(dtype=inv_factor_matrix.dtype)

                        # Check if we encounter NaN or inf values in computed inverse matrix.
                        if (
                            torch.isnan(computed_inv_factor_matrix).any()
                            or torch.isinf(computed_inv_factor_matrix).any()
                        ):
                            torch.set_printoptions(threshold=100_000)
                            raise ValueError(
                                f"Encountered nan or inf values in inverse factor matrix {factor_matrix_index}! "
                                f"To mitigate, check factor matrix before matrix inverse root computation: "
                                f"{bias_corrected_factor_matrix=}"
                            )

                        inv_factor_matrix.copy_(computed_inv_factor_matrix)

                    except Exception as exception:
                        if (
                            not self._use_protected_eigh
                            or "Encountered nan or inf values in inverse factor matrix"
                            in str(exception)
                        ):
                            raise exception
                        else:
                            logger.warning(
                                f"Matrix inverse root computation failed for factor matrix {factor_matrix_index} "
                                f"with exception {exception}. Using previous inv_factor_matrix and continuing..."
                            )

    def dequantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.dequantize_preconditioners.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.factor_matrices.dequantize_()
                kronecker_factors.inv_factor_matrices.dequantize_()

    def quantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.quantize_preconditioners.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.factor_matrices.quantize_()
                kronecker_factors.inv_factor_matrices.quantize_()

    def compress_preconditioner_list(
        self, local_grad_selector: Tuple[bool, ...]
    ) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compress_preconditioner_list.__name__} ##"
        ):
            self._masked_order_list = compress_list(
                self._local_order_list, local_grad_selector
            )
            self._masked_root_list = compress_list(
                self._local_root_list, local_grad_selector
            )
            self._masked_kronecker_factors_list: Tuple[
                ShampooKroneckerFactorsList, ...
            ] = compress_list(self._local_kronecker_factors_list, local_grad_selector)

    def compute_root_inverse_residuals(
        self,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        relative_errors = []
        relative_residuals = []

        for kronecker_factors, root in zip(
            self._masked_kronecker_factors_list,
            self._masked_root_list,
            strict=True,
        ):
            for factor_matrix, inv_factor_matrix in zip(
                kronecker_factors.factor_matrices.dequantized_value,
                kronecker_factors.inv_factor_matrices.dequantized_value,
                strict=True,
            ):
                bias_corrected_factor_matrix = factor_matrix / self._bias_correction2
                (
                    relative_error,
                    relative_residual,
                ) = compute_matrix_root_inverse_residuals(
                    bias_corrected_factor_matrix,
                    inv_factor_matrix,
                    root,
                    self._epsilon,
                    self._exponent_multiplier,
                )
                relative_errors.append(relative_error)
                relative_residuals.append(relative_residual)

        return (
            tuple(relative_errors),
            tuple(relative_residuals),
        )


class DequantizePreconditionersContext:
    """DequantizePreconditionersContext is used for automatically dequantize and then quantize the preconditioners used within this context.

    Args:
        preconditioner_list (PreconditionerList): Preconditioner list which contains the preconditioners to be dequantized and quantized.

    Examples:
        >>> with DequantizePreconditionersContext(preconditioner_list):
        >>>     # Do something with the preconditioners, and preconditioner_list will be dequantized.
        >>> # After the context is exited, the preconditioners will be quantized.

    """

    def __init__(self, preconditioner_list: PreconditionerList) -> None:
        self._preconditioner_list = preconditioner_list

    def __enter__(self) -> "DequantizePreconditionersContext":
        self._preconditioner_list.dequantize_preconditioners()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._preconditioner_list.quantize_preconditioners()
