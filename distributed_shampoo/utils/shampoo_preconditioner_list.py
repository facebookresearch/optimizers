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
from operator import methodcaller
from typing import Any, DefaultDict, List, Optional, Sequence, Tuple, Union

import torch
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_quantization import (
    QuantizedTensor,
    QuantizedTensorList,
)
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    get_dtype_size,
    ParameterizeEnterExitContext,
)
from distributed_shampoo.shampoo_types import PrecisionConfig

from matrix_functions import (
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_eigenvectors,
    matrix_inverse_root,
)

from matrix_functions_types import DefaultEigenConfig, RootInvConfig
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
        compute_root_inverse_or_eigenvectors: Optional[bool],
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
        compute_root_inverse_or_eigenvectors: Optional[bool],
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
        state (DefaultDict[Tensor, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        beta2 (float): Exponential moving average factor for Adam/RMSprop second moment state. If beta2 = 1., will use
            unweighted sum. (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-10)
        precision_config (PrecisionConfig): Data types for optimizer states. (Default: all fields torch.float.)
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
        precision_config: PrecisionConfig = PrecisionConfig(),
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
                    block.size(),
                    precision_config.grafting_state_dtype,
                    block.device,
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
            precision_config.grafting_state_dtype,
            precision_config.computation_dtype,
        )
        self._masked_preconditioner_list: QuantizedTensorList = (
            self._local_preconditioner_list
        )

        # Construct lists of dims, bytes, and numels for logging purposes.
        self._dims_list: Tuple[torch.Size, ...] = compress_list(
            self._dims_list, distributor_selector
        )
        self._numel_list: Tuple[int, ...] = tuple(
            quantized_preconditioner.numel()
            for quantized_preconditioner in self._local_preconditioner_list.quantized_value
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            quantize_preconditioner.numel() * quantize_preconditioner.element_size()
            for quantize_preconditioner in self._local_preconditioner_list.quantized_value
        )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor, ...],
        step: Tensor,
        compute_root_inverse_or_eigenvectors: Optional[bool] = None,
    ) -> None:
        if compute_root_inverse_or_eigenvectors is not None:
            raise ValueError(
                "compute_root_inverse_or_eigenvectors has to be None."
            )
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
class BaseShampooKroneckerFactors(OptimizerModule):
    """Base class for Shampoo Kronecker factors."""

    factor_matrices: Union[Tuple[QuantizedTensor, ...], QuantizedTensorList]
    factor_matrix_indices: Tuple[str, ...]
    is_factor_matrices_diagonal: Tuple[Tensor, ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__init__()
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)
        self.is_factor_matrices_diagonal = tuple(
            torch.tensor(True) for _ in range(len(self.factor_matrices))
        )


@dataclass
class ShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Shampoo Kronecker factors (wrapped) for storing in the optimizer state."""

    inv_factor_matrices: Tuple[QuantizedTensor, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass
class ShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Shampoo Kronecker factors (unwrapped) for operations during optimizer computation."""

    inv_factor_matrices: QuantizedTensorList

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass
class EigenvalueCorrectedShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Eigenvalue-corrected Shampoo Kronecker factors (wrapped) for storing in the optimizer state."""

    factor_matrices_eigenvectors: Tuple[QuantizedTensor, ...]
    corrected_eigenvalues: QuantizedTensor

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


@dataclass
class EigenvalueCorrectedShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Eigenvalue-corrected Shampoo Kronecker factors (unwrapped) for operations during optimizer computation."""

    factor_matrices_eigenvectors: QuantizedTensorList
    corrected_eigenvalues: QuantizedTensorList

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


class BaseShampooPreconditionerList(PreconditionerList):
    """Base class for Shampoo preconditioners.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (Tuple[Tensor, ...]): List of (blocks of) parameters.
        state (DefaultDict[Tensor, Any]): Dictionary containing optimizer state.
        block_info_list (Tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        distributor_selector (Tuple[bool, ...]): Distributor selector is a boolean list indicating whether a blocked parameter
            is selected by the current Distributor.
        root_inv_config (RootInvConfig): Configuration for root inverse computation. (Default: DefaultEigenConfig)
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        inv_root_override (Union[int, Tuple[int, ...]]): Inverse root to use in Shampoo. If a list [l0, l1, l2, ..., lp], then we will
            use -1 / l0 for 0-D tensors (scalars), -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on.
            If the order of the tensor exceeds the length of the list, we revert to using the default value. If 0 is used, uses the
            default inverse root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        precision_config (PrecisionConfig): Data types for optimizer states. (Default: all fields torch.float.)
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
        root_inv_config: RootInvConfig = DefaultEigenConfig,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        inv_root_override: Union[int, Tuple[int, ...]] = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        precision_config: PrecisionConfig = PrecisionConfig(),
        use_protected_eigh: bool = True,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._root_inv_config = root_inv_config
        self._beta2 = beta2
        self._epsilon = epsilon
        self._inv_root_override = inv_root_override
        self._exponent_multiplier = exponent_multiplier
        self._precision_config = precision_config
        self._use_bias_correction = use_bias_correction
        self._use_protected_eigh = use_protected_eigh
        self._bias_correction2: Tensor = torch.tensor(1.0)

        # Create the Kronecker factors.
        kronecker_factors_list = self._create_kronecker_factors_state(
            block_list=block_list,
            state=state,
            block_info_list=block_info_list,
        )

        # Initialize state lists.
        self._initialize_state_lists(
            block_list=block_list,
            kronecker_factors_list=kronecker_factors_list,
            distributor_selector=distributor_selector,
        )

    @abstractmethod
    def _create_kronecker_factors_state(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
    ) -> List[
            Union[
                ShampooKroneckerFactorsList,
                EigenvalueCorrectedShampooKroneckerFactorsList,
            ]
        ]:
        ...

    @staticmethod
    @abstractmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]],
        order_list: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        ...

    def _initialize_state_lists(
        self,
        block_list: Tuple[Tensor, ...],
        kronecker_factors_list: List[
            Union[
                ShampooKroneckerFactorsList,
                EigenvalueCorrectedShampooKroneckerFactorsList,
            ]
        ],
        distributor_selector: Tuple[bool, ...],
    ) -> None:
        # Initialize local lists.
        local_block_list = compress_list(block_list, distributor_selector)
        self._local_kronecker_factors_list: Tuple[
            Union[
                ShampooKroneckerFactorsList,
                EigenvalueCorrectedShampooKroneckerFactorsList,
            ],
            ...,
        ] = compress_list(kronecker_factors_list, distributor_selector)
        self._local_order_list: Tuple[int, ...] = tuple(
            block.dim() for block in local_block_list
        )
        self._local_root_list: Tuple[int, ...] = self._get_inverse_roots_from_override(
            self._inv_root_override,
            self._local_order_list,
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: Tuple[int, ...] = self._local_order_list
        self._masked_root_list: Tuple[int, ...] = self._local_root_list
        self._masked_kronecker_factors_list: Tuple[
            Union[
                ShampooKroneckerFactorsList,
                EigenvalueCorrectedShampooKroneckerFactorsList,
            ],
            ...,
        ] = self._local_kronecker_factors_list

        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._dims_list: Tuple[torch.Size, ...] = compress_list(
            self._dims_list, distributor_selector
        )
        self._numel_list: Tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in dims) for dims in self._dims_list
        )
        self._num_bytes_list: Tuple[int, ...] = tuple(
            numel
            * (
                get_dtype_size(self._precision_config.factor_matrix_dtype)
                + get_dtype_size(block.dtype)
            )
            // 2
            for numel, block in zip(self._numel_list, local_block_list, strict=True)
        )

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
                Union[
                    ShampooKroneckerFactorsList,
                    EigenvalueCorrectedShampooKroneckerFactorsList,
                ],
                ...,
            ] = compress_list(self._local_kronecker_factors_list, local_grad_selector)

    def _update_factor_matrices(
        self, masked_grad_list: Tuple[Tensor, ...]
    ) -> None:
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

    @staticmethod
    def _precondition_grad(
        grad: Tensor,
        preconditioner_list: Tuple[Tensor, ...],
        dims: List[List[int]] = [[0], [0]],
    ) -> Tensor:
        for preconditioner in preconditioner_list:
            grad = torch.tensordot(grad, preconditioner, dims)
        return grad

    @staticmethod
    def _check_factor_matrix_for_nan_and_inf(
        factor_matrix: Tensor, factor_matrix_index: str
    ) -> None:
        # Check for nan or inf values.
        if torch.isnan(factor_matrix).any():
            raise ValueError(
                f"Encountered nan values in factor matrix {factor_matrix_index}! "
                f"To mitigate, check if nan inputs are being passed into the network or nan gradients "
                f"are being passed to the optimizer."
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )
        if torch.isinf(factor_matrix).any():
            raise ValueError(
                f"Encountered inf values in factor matrix {factor_matrix_index}! "
                f"In some cases, this may be due to divergence of the algorithm. "
                f"To mitigate, try decreasing the learning rate or increasing grafting epsilon."
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )


class ShampooPreconditionerList(BaseShampooPreconditionerList):
    """Shampoo preconditioners for list of parameters."""

    def _create_kronecker_factors_state(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
    ) -> List[ShampooKroneckerFactorsList]:
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
                        self._precision_config.factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )
            inv_factor_matrices = tuple(
                QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (dim, dim),
                        self._precision_config.inv_factor_matrix_dtype,
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
                        self._precision_config.factor_matrix_dtype,
                        self._precision_config.computation_dtype,
                    ),
                    inv_factor_matrices=QuantizedTensorList(
                        inv_factor_matrices,
                        self._precision_config.inv_factor_matrix_dtype,
                        self._precision_config.computation_dtype,
                    ),
                    factor_matrix_indices=factor_matrix_indices,
                )
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                f"({[(factor_matrix.quantized_values.shape, factor_matrix.quantized_values.dtype) for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        return kronecker_factors_list

    @staticmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]],
        order_list: Tuple[int, ...],
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
        self,
        masked_grad_list: Tuple[Tensor],
        step: Tensor,
        compute_root_inverse_or_eigenvectors: Optional[bool],
    ) -> None:
        if compute_root_inverse_or_eigenvectors is None:
            raise ValueError(
                "compute_root_inverse_or_eigenvectors has to be boolean."
            )
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            # Update the Kronecker factor matrices.
            self._update_factor_matrices(masked_grad_list=masked_grad_list)

            # Update bias correction term based on step.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

            # Update the inverse roots of the factor matrices.
            if compute_root_inverse_or_eigenvectors:
                self.compute_root_inverse()

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            return tuple(
                self._precondition_grad(
                    grad=masked_grad,
                    preconditioner_list=kronecker_factors.inv_factor_matrices.dequantized_value,
                )
                for masked_grad, kronecker_factors in zip(
                    masked_grad_list, self._masked_kronecker_factors_list, strict=True
                )
            )

    @torch.compiler.disable
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

                    self._check_factor_matrix_for_nan_and_inf(
                        bias_corrected_factor_matrix,
                        factor_matrix_index,
                    )

                    # Compute inverse preconditioner.
                    # If reuse_previous_inv_factor_matrix is True, will reuse previous matrix if matrix
                    # inverse root computation fails.
                    try:
                        computed_inv_factor_matrix = matrix_inverse_root(
                            A=bias_corrected_factor_matrix,
                            root=root,
                            root_inv_config=self._root_inv_config,
                            epsilon=self._epsilon,
                            exponent_multiplier=self._exponent_multiplier,
                            is_diagonal=is_factor_matrix_diagonal,
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

    @torch.compiler.disable
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
                    A=bias_corrected_factor_matrix,
                    X_hat=inv_factor_matrix,
                    root=root,
                    epsilon=self._epsilon,
                    exponent_multiplier=self._exponent_multiplier,
                    root_inv_config=self._root_inv_config,
                )
                relative_errors.append(relative_error)
                relative_residuals.append(relative_residual)

        return (
            tuple(relative_errors),
            tuple(relative_residuals),
        )


class EigenvalueCorrectedShampooPreconditionerList(BaseShampooPreconditionerList):
    """Eigenvalue-corrected Shampoo preconditioners for list of parameters."""

    def _create_kronecker_factors_state(
        self,
        block_list: Tuple[Tensor, ...],
        state: DefaultDict[Tensor, Any],
        block_info_list: Tuple[BlockInfo, ...],
    ) -> List[EigenvalueCorrectedShampooKroneckerFactorsList]:
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
                        self._precision_config.factor_matrix_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )
            factor_matrices_eigenvectors = tuple(
                QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        (dim, dim),
                        self._precision_config.factor_matrix_eigenvectors_dtype,
                        block_info.param.device,
                    ),
                    block_info,
                )
                for dim in dims
            )
            corrected_eigenvalues = QuantizedTensor(
                block_info.allocate_zeros_tensor(
                    tuple(dims),
                    self._precision_config.corrected_eigenvalues_dtype,
                    block_info.param.device,
                ),
                block_info,
            )

            preconditioner_index = str(param_index) + "." + str(block_index)
            factor_matrix_indices = tuple(
                preconditioner_index + "." + str(k) for k in range(len(dims))
            )
            block_state[SHAMPOO] = EigenvalueCorrectedShampooKroneckerFactorsState(
                factor_matrices=factor_matrices,
                factor_matrices_eigenvectors=factor_matrices_eigenvectors,
                corrected_eigenvalues=corrected_eigenvalues,
                factor_matrix_indices=factor_matrix_indices,
            )
            kronecker_factors_list.append(
                EigenvalueCorrectedShampooKroneckerFactorsList(
                    factor_matrices=QuantizedTensorList(
                        factor_matrices,
                        self._precision_config.factor_matrix_dtype,
                        self._precision_config.computation_dtype,
                    ),
                    factor_matrices_eigenvectors=QuantizedTensorList(
                        factor_matrices_eigenvectors,
                        self._precision_config.factor_matrix_eigenvectors_dtype,
                        self._precision_config.computation_dtype,
                    ),
                    corrected_eigenvalues=QuantizedTensorList(
                        (corrected_eigenvalues,),
                        self._precision_config.corrected_eigenvalues_dtype,
                        self._precision_config.computation_dtype,
                    ),
                    factor_matrix_indices=factor_matrix_indices,
                )
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {preconditioner_index} "
                f"({[(factor_matrix.quantized_values.shape, factor_matrix.quantized_values.dtype) for factor_matrix in block_state[SHAMPOO].factor_matrices]}) "
                f"for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        return kronecker_factors_list

    @staticmethod
    def _get_inverse_roots_from_override(
        inv_root_override: Union[int, Sequence[int]],
        order_list: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Retrieves the appropriate root from the inverse root override parameter
        for a list of tensor orders.

        For example, suppose inv_root_override = (2, 1, 4, 3).
        If order = 0, then we will return 2;
        If order = 1, then we will return 1;
        If order = 2, then we will return 4;
        If order = 3, then we will return 3;
        If order > 3, then we will return 2.

        Args:
            inv_root_override (int, Sequence[int]): Inverse root override int or list.
            order_list (Tuple[int, ...]): List of orders for their corresponding tensors.

        Returns:
            root_list (int): Inverse roots to use in Shampoo for a list of tensors.

        """
        if isinstance(inv_root_override, Sequence):
            return tuple(
                (
                    2
                    if order >= len(inv_root_override)
                    else inv_root_override[order]
                )
                for order in order_list
            )
        else:
            return (
                (2 if inv_root_override == 0 else inv_root_override,)
                * len(order_list)
            )

    def update_preconditioners(
        self,
        masked_grad_list: Tuple[Tensor],
        step: Tensor,
        compute_root_inverse_or_eigenvectors: Optional[bool],
    ) -> None:
        if compute_root_inverse_or_eigenvectors is None:
            raise ValueError(
                "compute_root_inverse_or_eigenvectors has to be boolean."
            )
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.update_preconditioners.__name__} ##"
        ):
            # Update the Kronecker factor matrices.
            self._update_factor_matrices(masked_grad_list=masked_grad_list)

            # Update bias correction term based on step.
            if self._use_bias_correction and self._beta2 < 1.0:
                self._bias_correction2 = torch.tensor(1.0) - self._beta2**step

            # Update the eigenvectors of the factor matrices.
            if compute_root_inverse_or_eigenvectors:
                self.compute_preconditioner_eigenvectors()

            # Update the eigenvalue corrections of Shampoo's preconditioner.
            self._update_eigenvalue_corrections(masked_grad_list=masked_grad_list)

    def _update_eigenvalue_corrections(self, masked_grad_list: Tuple[Tensor, ...]) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self._update_eigenvalue_corrections.__name__} ##"
        ):
            # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
            for grad, kronecker_factors in zip(
                masked_grad_list,
                self._masked_kronecker_factors_list,
                strict=True,
            ):
                factor_eigenvectors = kronecker_factors.factor_matrices_eigenvectors.dequantized_value
                if factor_eigenvectors[0].any():
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioner_list=factor_eigenvectors,
                    )
                # Scale corrected eigenvalues.
                # NOTE: The case when self._beta2 == 1.0 is not well tested and might not be stable.
                if self._beta2 != 1.0:
                    kronecker_factors.corrected_eigenvalues.dequantized_value[0].mul_(
                        self._beta2
                    )
                # Update corrected eigenvalues (squared gradient in eigenbasis of Shampoo preconditioner).
                kronecker_factors.corrected_eigenvalues.dequantized_value[0].add_(
                    grad.square(),
                    alpha=1 - self._beta2 if self._beta2 != 1.0 else 1.0,
                )

    def precondition(self, masked_grad_list: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
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
                factor_eigenvectors = kronecker_factors.factor_matrices_eigenvectors.dequantized_value
                corrected_eigenvalues = kronecker_factors.corrected_eigenvalues.dequantized_value[0]
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
                    corrected_eigenvalues.div(self._bias_correction2).add_(
                        self._epsilon
                    ).pow_(self._exponent_multiplier / root)
                )
                if use_eigenbasis:
                    # Convert back to basis of the parameters.
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioner_list=factor_eigenvectors,
                        dims=[[0], [1]],
                    )
                preconditioned_grad_list.append(grad)
            return tuple(preconditioned_grad_list)

    @torch.compiler.disable
    def compute_preconditioner_eigenvectors(self) -> None:
        # NOTE: This function currently only computes the preconditioner eigenvectors based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.compute_preconditioner_eigenvectors.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                for (
                    factor_matrix,
                    factor_matrix_eigenvectors,
                    is_factor_matrix_diagonal,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices.dequantized_value,
                    kronecker_factors.factor_matrices_eigenvectors.dequantized_value,
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

                    self._check_factor_matrix_for_nan_and_inf(
                        factor_matrix,
                        factor_matrix_index,
                    )

                    # Compute eigenvectors of factor matrix.
                    # If reuse_previous_inv_factor_matrix is True, will reuse previous matrix if matrix
                    # inverse root computation fails.
                    try:
                        computed_eigenvectors = matrix_eigenvectors(
                            A=factor_matrix,
                            is_diagonal=is_factor_matrix_diagonal,
                            eigenvector_computation_config=self._root_inv_config,
                        )

                        # Check if we encounter NaN or inf values in computed inverse matrix.
                        if (
                            torch.isnan(computed_eigenvectors).any()
                            or torch.isinf(computed_eigenvectors).any()
                        ):
                            torch.set_printoptions(threshold=100_000)
                            raise ValueError(
                                f"Encountered nan or inf values in eigenvectors of factor matrix {factor_matrix_index}! "
                                f"To mitigate, check factor matrix before matrix eigendecomposition: "
                                f"{factor_matrix=}"
                            )

                        if is_factor_matrix_diagonal:
                            # Eigenvectors matrix is identity -> only store zero tensor which will be treated
                            # as identity when changing basis of gradient.
                            computed_eigenvectors = torch.as_tensor(
                                [0.0],
                                dtype=self._precision_config.factor_matrix_eigenvectors_dtype,
                                device=computed_eigenvectors.device,
                            )
                        factor_matrix_eigenvectors.copy_(
                            computed_eigenvectors.to(
                                dtype=self._precision_config.factor_matrix_eigenvectors_dtype
                            )
                        )

                    except Exception as exception:
                        if (
                            not self._use_protected_eigh
                            or "Encountered nan or inf values in eigenvectors of factor matrix"
                            in str(exception)
                        ):
                            raise exception
                        else:
                            logger.warning(
                                f"Eigenvector computation failed for factor matrix {factor_matrix_index} "
                                f"with exception {exception}. Using previous factor_matrices_eigenvectors and continuing..."
                            )

    def dequantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.dequantize_preconditioners.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.factor_matrices.dequantize_()
                kronecker_factors.factor_matrices_eigenvectors.dequantize_()
                kronecker_factors.corrected_eigenvalues.dequantize_()

    def quantize_preconditioners(self) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.quantize_preconditioners.__name__} ##"
        ):
            for kronecker_factors in self._masked_kronecker_factors_list:
                kronecker_factors.factor_matrices.quantize_()
                kronecker_factors.factor_matrices_eigenvectors.quantize_()
                kronecker_factors.corrected_eigenvalues.quantize_()


class DequantizePreconditionersContext(ParameterizeEnterExitContext):
    """DequantizePreconditionersContext is used for automatically dequantize and then quantize the preconditioners used within this context.

    Args:
        preconditioner_list (PreconditionerList): Preconditioner list which contains the preconditioners to be dequantized and quantized.

    Examples:
        >>> with DequantizePreconditionersContext(preconditioner_list):
        >>>     # Do something with the preconditioners which are dequantized.
        >>> # After the context is exited, the preconditioners will be quantized.

    """

    def __init__(self, preconditioner_list: PreconditionerList) -> None:
        super().__init__(
            input_with_enter_exit_context=preconditioner_list,
            enter_method_caller=methodcaller("dequantize_preconditioners"),
            exit_method_caller=methodcaller("quantize_preconditioners"),
        )
