"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import reduce

from itertools import chain
from operator import attrgetter
from typing import Any, cast, Generic, get_args, TypeVar

import torch
from distributed_shampoo.shampoo_types import (
    PreconditionerConfig,
    PreconditionerValueError,
)
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size
from matrix_functions import matrix_eigendecomposition, matrix_inverse_root

from matrix_functions_types import (
    EigendecompositionConfig,
    QREigendecompositionConfig,
    RootInvConfig,
)
from optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

ADAGRAD = "adagrad"
SHAMPOO = "shampoo"
INVERSE_EXPONENT_OVERRIDE = "inverse_exponent_override"


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
    """Adagrad / Adam / RMSprop preconditioners for a list of parameters.

    Operations are performed in-place with foreach operators.

    NOTE: Does not support sparse gradients at this time.

    To enable Adagrad, set beta2 = 1.0.
    To enable RMSprop, set beta2 = 0.999.
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
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
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
    """Base class for Shampoo Kronecker factors.

    Attributes:
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "BaseShampooKroneckerFactors":
        """
        Creates a BaseShampooKroneckerFactor object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.

        Returns:
            kronecker_factors_state (BaseShampooKroneckerFactors): An object containing the Kronecker factors for the block.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]

        return cls(
            factor_matrices=tuple(
                block_info.allocate_zeros_tensor(
                    size=(dim, dim),
                    dtype=factor_matrix_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            factor_matrix_indices=tuple(
                ".".join((*map(str, block_info.composable_block_ids), str(k)))
                for k in range(len(preconditioned_dims))
            ),
        )

    def __post_init__(self) -> None:
        super().__init__()
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)


@dataclass(kw_only=True)
class ShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "ShampooKroneckerFactorsState":
        """
        Creates a ShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.
            block_dtype (torch.dtype): Data type for the block.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        block_dtype: torch.dtype = kwargs["block_dtype"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactors.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize inv_factor_matrices as identity matrices.
            inv_factor_matrices=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=block_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass(kw_only=True)
class ShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "ShampooKroneckerFactorsList":
        """
        Creates a ShampooKroneckerFactorsList object from a block.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): Function to unwrap tensors.
            kronecker_factors_state (ShampooKroneckerFactorsState): State containing factor matrices and their inverses.

        Returns:
            ShampooKroneckerFactorsList: An instance of ShampooKroneckerFactorsList.
        """
        unwrapped_tensor_getter: Callable[[Tensor], Tensor] = kwargs[
            "unwrapped_tensor_getter"
        ]
        kronecker_factors_state: ShampooKroneckerFactorsState = kwargs[
            "kronecker_factors_state"
        ]

        return cls(
            factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices,
                )
            ),
            inv_factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.inv_factor_matrices,
                )
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.inv_factor_matrices)


@dataclass(kw_only=True)
class EigendecomposedShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Eigendecomposed Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        factor_matrices_eigenvalues (tuple[Tensor, ...]): A tuple of tensors representing the eigenvalues of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    factor_matrices_eigenvalues: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "EigendecomposedShampooKroneckerFactorsState":
        """
        Creates an EigendecomposedShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.
            block_dtype (torch.dtype): Data type for the block.

        Returns:
            EigendecomposedShampooKroneckerFactorsState: An instance of EigendecomposedShampooKroneckerFactorsState.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        block_dtype: torch.dtype = kwargs["block_dtype"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactors.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize factor_matrices_eigenvectors as identity matrices.
            factor_matrices_eigenvectors=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=block_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            # Initialize factor_matrices_eigenvalues all ones.
            factor_matrices_eigenvalues=tuple(
                block_info.allocate_ones_tensor(
                    size=(dim,),
                    dtype=block_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.factor_matrices)
            == len(self.factor_matrices_eigenvectors)
            == len(self.factor_matrices_eigenvalues)
        )


@dataclass(kw_only=True)
class EigendecomposedShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Eigendecomposed Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        factor_matrices_eigenvalues (tuple[Tensor, ...]): A tuple of tensors representing the eigenvalues of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    factor_matrices_eigenvalues: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "EigendecomposedShampooKroneckerFactorsList":
        """
        Creates an EigendecomposedShampooKroneckerFactorsList object from a block.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): Function to unwrap tensors.
            kronecker_factors_state (EigendecomposedShampooKroneckerFactorsState): State containing factor matrices, eigenvectors, and eigenvalues.

        Returns:
            EigendecomposedShampooKroneckerFactorsList: An instance of EigendecomposedShampooKroneckerFactorsList.
        """
        unwrapped_tensor_getter: Callable[[Tensor], Tensor] = kwargs[
            "unwrapped_tensor_getter"
        ]
        kronecker_factors_state: EigendecomposedShampooKroneckerFactorsState = kwargs[
            "kronecker_factors_state"
        ]

        return cls(
            factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices,
                )
            ),
            factor_matrices_eigenvectors=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvectors,
                )
            ),
            factor_matrices_eigenvalues=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvalues,
                )
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.factor_matrices)
            == len(self.factor_matrices_eigenvectors)
            == len(self.factor_matrices_eigenvalues)
        )


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooKroneckerFactorsState(BaseShampooKroneckerFactors):
    """Eigenvalue-corrected Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        corrected_eigenvalues (Tensor): A tensor representing the corrected eigenvalues.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    @classmethod
    def from_block(
        cls, **kwargs: Any
    ) -> "EigenvalueCorrectedShampooKroneckerFactorsState":
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        block_dtype: torch.dtype = kwargs["block_dtype"]
        dims: tuple[int, ...] = kwargs["dims"]

        return EigenvalueCorrectedShampooKroneckerFactorsState(
            **asdict(
                BaseShampooKroneckerFactors.from_block(
                    block_info=block_info,
                    factor_matrix_dtype=factor_matrix_dtype,
                    preconditioned_dims=preconditioned_dims,
                )
            ),
            # Initialize factor_matrices_eigenvectors as identity matrices.
            factor_matrices_eigenvectors=tuple(
                block_info.allocate_eye_tensor(
                    n=dim,
                    dtype=block_dtype,
                    device=block_info.param.device,
                )
                for dim in preconditioned_dims
            ),
            corrected_eigenvalues=block_info.allocate_zeros_tensor(
                # Note that the corrected eigenvalues are not affected by the preconditioned_dims.
                size=tuple(dims),
                dtype=block_dtype,
                device=block_info.param.device,
            ),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooKroneckerFactorsList(BaseShampooKroneckerFactors):
    """Eigenvalue-corrected Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        corrected_eigenvalues (Tensor): A tensor representing the corrected eigenvalues.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    @classmethod
    def from_block(
        cls, **kwargs: Any
    ) -> "EigenvalueCorrectedShampooKroneckerFactorsList":
        """
        Creates an EigenvalueCorrectedShampooKroneckerFactorsList object from a block.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): Function to unwrap tensors.
            kronecker_factors_state (EigenvalueCorrectedShampooKroneckerFactorsState): State containing factor matrices, eigenvectors, and corrected eigenvalues.

        Returns:
            EigenvalueCorrectedShampooKroneckerFactorsList: An instance of EigenvalueCorrectedShampooKroneckerFactorsList.
        """
        unwrapped_tensor_getter: Callable[[Tensor], Tensor] = kwargs[
            "unwrapped_tensor_getter"
        ]
        kronecker_factors_state: EigenvalueCorrectedShampooKroneckerFactorsState = (
            kwargs["kronecker_factors_state"]
        )

        return cls(
            factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices,
                )
            ),
            factor_matrices_eigenvectors=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvectors,
                )
            ),
            corrected_eigenvalues=unwrapped_tensor_getter(
                kronecker_factors_state.corrected_eigenvalues
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)


ShampooKroneckerFactorsStateType = TypeVar(
    "ShampooKroneckerFactorsStateType",
    ShampooKroneckerFactorsState,
    EigendecomposedShampooKroneckerFactorsState,
    EigenvalueCorrectedShampooKroneckerFactorsState,
)
ShampooKroneckerFactorsListType = TypeVar(
    "ShampooKroneckerFactorsListType",
    ShampooKroneckerFactorsList,
    EigendecomposedShampooKroneckerFactorsList,
    EigenvalueCorrectedShampooKroneckerFactorsList,
)


class BaseShampooPreconditionerList(
    PreconditionerList,
    Generic[ShampooKroneckerFactorsStateType, ShampooKroneckerFactorsListType],
):
    """Base class for Shampoo preconditioners.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, Any]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        preconditioner_config (PreconditionerConfig): Configuration for preconditioner computation.
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
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
        use_bias_correction: bool = True,
        factor_matrix_dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._preconditioner_config = preconditioner_config
        self._beta2 = beta2
        self._epsilon = epsilon
        self._factor_matrix_dtype = factor_matrix_dtype
        self._use_bias_correction = use_bias_correction
        self._bias_correction2: Tensor = torch.tensor(1.0)

        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = tuple(
            self._create_preconditioned_dims_selector(dims)
            # Traverse through each block's dims.
            for dims in self._dims_list
        )
        preconditioned_dims_list: tuple[tuple[int, ...], ...] = tuple(
            compress_list(dims, preconditioned_dims_selector)
            for dims, preconditioned_dims_selector in zip(
                self._dims_list, preconditioned_dims_selector_list, strict=True
            )
        )

        # Create the Kronecker factors.
        kronecker_factors_list: list[ShampooKroneckerFactorsListType] = (
            self._create_kronecker_factors_state(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                preconditioned_dims_list=preconditioned_dims_list,
            )
        )

        # Initialize state lists.
        self._initialize_state_lists(
            block_list=block_list,
            kronecker_factors_list=kronecker_factors_list,
            preconditioned_dims_list=preconditioned_dims_list,
            preconditioned_dims_selector_list=preconditioned_dims_selector_list,
        )

    @abstractmethod
    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        """
        Creates a preconditioned dimensions selectors for a block.

        Args:
            dims (torch.Size): The dimensions of the block.

        Returns:
            preconditioned_dims_selector (tuple[bool, ...]): A preconditioned dimensions selectors for a block.
        """
        ...

    def _create_kronecker_factors_state(
        self,
        block_list: tuple[Tensor, ...],
        # type: ignore
        state: Mapping[Tensor, Any],
        block_info_list: tuple[BlockInfo, ...],
        preconditioned_dims_list: tuple[tuple[int, ...], ...],
    ) -> list[ShampooKroneckerFactorsListType]:
        # Instantiate (blocked) Kronecker factors and construct list of Kronecker factors.
        # NOTE: We need to instantiate the Kronecker factor states within the optimizer's state dictionary,
        # and do not explicitly store them as ShampooPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but ShampooPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        kronecker_factors_list = []
        for block, block_info, dims, preconditioned_dims in zip(
            block_list,
            block_info_list,
            self._dims_list,
            preconditioned_dims_list,
            strict=True,
        ):
            param_index, block_index = block_info.composable_block_ids
            if block_index not in state[block_info.param]:
                state[block_info.param][block_index] = {}
            block_state = state[block_info.param][block_index]
            kronecker_factors_state_type, kronecker_factors_state_list_type = get_args(
                attrgetter("__orig_bases__")(self)[0]
            )
            block_state[SHAMPOO] = kronecker_factors_state_type.from_block(
                block_info=block_info,
                factor_matrix_dtype=self._factor_matrix_dtype,
                preconditioned_dims=preconditioned_dims,
                block_dtype=block.dtype,
                dims=dims,
            )
            kronecker_factors_list.append(
                kronecker_factors_state_list_type.from_block(
                    kronecker_factors_state=block_state[SHAMPOO],
                    unwrapped_tensor_getter=block_info.get_tensor,
                )
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {str(param_index) + '.' + str(block_index)} for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        return kronecker_factors_list

    @abstractmethod
    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        """
        Retrieves the inverse roots from the override parameter for a block.

        For a block, we compute the inverse root from the inverse exponent override parameter according to its order.
        If the order is not present in the inverse exponent override parameter, the default value is used for the inverse exponent override.
        The inverse root is then computed as 1 / inverse exponent override.

        Args:
            preconditioned_dims_selector (tuple[bool, ...]): A selector indicating which dimensions are preconditioned for a block.

        Returns:
            inverse_roots (tuple[float, ...]): Inverse roots for each preconditioner of a block.
        """
        ...

    @abstractmethod
    def _amortized_computation(self) -> None:
        """
        Computes the amortized computation needed for each Shampoo preconditioner implementation.
        This amortized computation is computation heavy work that cannot be done for each step.
        """
        ...

    @staticmethod
    def _check_factor_matrix_for_nan_and_inf(
        factor_matrix: Tensor,
        factor_matrix_index: str,
    ) -> None:
        # Check for nan or inf values.
        if not torch.isfinite(factor_matrix).all():
            raise PreconditionerValueError(
                f"Encountered nan/inf values in factor matrix {factor_matrix_index}! "
                f"To mitigate, check if nan inputs are being passed into the network or nan gradients are being passed to the optimizer. "
                f"Otherwise, in some cases, this may be due to divergence of the algorithm. To mitigate, try decreasing the learning rate or increasing grafting epsilon. "
                f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                f"{torch.min(factor_matrix)=}, {torch.max(factor_matrix)=}, "
                f"{factor_matrix.isinf().any()=}, {factor_matrix.isnan().any()=}."
            )

    def _raise_exception_if_failure_tolerance_exceeded(
        self,
        success_tracker: bool,
        preconditioner_index: int,
        exception: Exception,
    ) -> None:
        """Raises an exception if the number of failed amortized computations exceeds the tolerance.

        Resets the counter at the given index when all amortized computations are successful.

        Args:
            success_tracker (bool): A boolean indicating whether the amortized computation was successful.
            preconditioner_index (int): The index of the preconditioner.
            exception (Exception): The exception to raise.

        Raises:
            exception (Exception): The exception to raise.

        """
        if success_tracker:
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
            # In eigenvalue-corrected Shampoo, this is equivalent to computing the eigenvectors of the factor matrix.
            if perform_amortized_computation:
                self._amortized_computation()

    def _initialize_state_lists(
        self,
        block_list: tuple[Tensor, ...],
        kronecker_factors_list: list[ShampooKroneckerFactorsListType],
        preconditioned_dims_list: tuple[tuple[int, ...], ...],
        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...],
    ) -> None:
        # Initialize local lists.
        self._local_kronecker_factors_list: tuple[
            ShampooKroneckerFactorsListType,
            ...,
        ] = tuple(kronecker_factors_list)
        self._local_order_list: tuple[int, ...] = tuple(
            block.dim() for block in block_list
        )
        self._local_roots_list: tuple[tuple[float, ...], ...] = tuple(
            self._get_inverse_roots_from_override(preconditioned_dims_selector)
            # Traverse through each block's preconditioned_dims_selector_list.
            for preconditioned_dims_selector in preconditioned_dims_selector_list
        )
        self._local_failed_amortized_computation_counter_list: list[int] = [0] * len(
            self._local_kronecker_factors_list
        )
        self._local_preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = (
            preconditioned_dims_selector_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: tuple[int, ...] = self._local_order_list
        self._masked_roots_list: tuple[tuple[float, ...], ...] = self._local_roots_list
        self._masked_failed_amortized_computation_counter_list: list[int] = (
            self._local_failed_amortized_computation_counter_list
        )
        self._masked_kronecker_factors_list: tuple[
            ShampooKroneckerFactorsListType,
            ...,
        ] = self._local_kronecker_factors_list
        self._masked_preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = (
            self._local_preconditioned_dims_selector_list
        )

        # Construct lists of bytes and numels for logging purposes.
        # NOTE: These lists are constructed across all blocked parameters.
        self._numel_list: tuple[int, ...] = tuple(
            sum(2 * dim**2 for dim in preconditioned_dims)
            for preconditioned_dims in preconditioned_dims_list
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
            self._masked_roots_list: tuple[tuple[float, ...], ...] = compress_list(  # type: ignore[no-redef]
                self._local_roots_list, local_grad_selector
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
            self._masked_preconditioned_dims_selector_list = compress_list(  # type: ignore[no-redef]
                self._local_preconditioned_dims_selector_list, local_grad_selector
            )

    def _update_factor_matrices(self, masked_grad_list: tuple[Tensor, ...]) -> None:
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self._update_factor_matrices.__name__} ##"
        ):
            # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
            # We apply foreach operators onto the list of Kronecker factor matrices (as opposed to the
            # full list of gradients/optimizer states).
            for grad, order, preconditioned_dims_selector, kronecker_factors in zip(
                masked_grad_list,
                self._masked_order_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_list,
                strict=True,
            ):
                # Because of preconditioned_dims_selector, we may have no factor matrices to update.
                if not kronecker_factors.factor_matrices:
                    continue

                # Construct outer product list for updating Kronecker factors.
                outer_product_list = tuple(
                    torch.tensordot(
                        grad,
                        grad,
                        # Contracts across all dimensions except for k.
                        dims=[[*chain(range(k), range(k + 1, order))]] * 2,  # type: ignore[has-type]
                    )
                    for k in compress_list(range(order), preconditioned_dims_selector)
                )

                # Scale Kronecker factors as a list if beta2 is not 1.0
                if self._beta2 != 1.0:
                    alpha = 1 - self._beta2
                    # Update Kronecker factors.
                    torch._foreach_mul_(kronecker_factors.factor_matrices, self._beta2)
                    torch._foreach_add_(
                        kronecker_factors.factor_matrices,
                        outer_product_list,
                        alpha=alpha,
                    )
                else:
                    # Update Kronecker factors without scaling.
                    torch._foreach_add_(
                        kronecker_factors.factor_matrices, outer_product_list, alpha=1.0
                    )

    @staticmethod
    def _precondition_grad(
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        preconditioner_list: tuple[Tensor, ...],
        dims: tuple[list[int], list[int]] = ([0], [0]),
    ) -> Tensor:
        # TODO: Need to refactor this function to be more efficient. Ideally eliminate those branches.
        # Might consider einsum?
        assert (
            sum(preconditioned_dims_selector) == len(preconditioner_list)
        ), f"The number of dimensions to precondition ({sum(preconditioned_dims_selector)}) must match the number of preconditioners ({len(preconditioner_list)})."
        preconditioner_list_iter = iter(preconditioner_list)
        return reduce(
            lambda grad, should_precondition: torch.tensordot(
                grad, next(preconditioner_list_iter), dims=dims
            )
            if should_precondition
            # Perform a left rotation on grad if not preconditioned.
            else grad.permute(*range(1, grad.ndim), 0),
            preconditioned_dims_selector,
            grad,
        )


class ShampooPreconditionerList(
    BaseShampooPreconditionerList[
        ShampooKroneckerFactorsState, ShampooKroneckerFactorsList
    ]
):
    """Shampoo preconditioners for list of parameters."""

    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        return tuple(
            attrgetter(INVERSE_EXPONENT_OVERRIDE)(self._preconditioner_config)
            .get((order := len(dims)), {})
            .get(d, 1 / (2 * max(order, 1)))
            != 0.0
            # Traverse through each dim of a block.
            for d in range(len(dims))
        )

    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        return tuple(
            # Compute the inverse root, 1 / inverse_exponent{_override}, accordingly for each required dim.
            1
            / attrgetter(INVERSE_EXPONENT_OVERRIDE)(self._preconditioner_config)
            .get((order := len(preconditioned_dims_selector)), {})
            .get(k, 1 / (2 * max(order, 1)))
            # Traverse through each dim of a block that requires precondition.
            for k, should_precondition in enumerate(preconditioned_dims_selector)
            if should_precondition
        )

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the Shampoo preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            return tuple(
                self._precondition_grad(
                    grad=masked_grad,
                    preconditioned_dims_selector=preconditioned_dims_selector,
                    preconditioner_list=kronecker_factors.inv_factor_matrices,
                )
                for masked_grad, preconditioned_dims_selector, kronecker_factors in zip(
                    masked_grad_list,
                    self._masked_preconditioned_dims_selector_list,
                    self._masked_kronecker_factors_list,
                    strict=True,
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
            for idx, (kronecker_factors, roots) in enumerate(
                zip(
                    self._masked_kronecker_factors_list,
                    self._masked_roots_list,
                    strict=True,
                )
            ):
                success_tracker: bool = True
                for (
                    factor_matrix,
                    inv_factor_matrix,
                    factor_matrix_index,
                    root,
                ) in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.inv_factor_matrices,
                    kronecker_factors.factor_matrix_indices,
                    roots,
                    strict=True,
                ):
                    # Incorporate bias correction.
                    bias_corrected_factor_matrix = (
                        factor_matrix / self._bias_correction2
                    )

                    BaseShampooPreconditionerList._check_factor_matrix_for_nan_and_inf(
                        factor_matrix=bias_corrected_factor_matrix,
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
                            root=Fraction(root),
                            root_inv_config=root_inv_config,
                            epsilon=self._epsilon,
                            is_diagonal=False,
                        ).to(dtype=inv_factor_matrix.dtype)
                        # Add success to success tracker.
                        success_tracker &= True
                    except Exception as exception:
                        # Add failure to success tracker.
                        success_tracker &= False
                        logger.warning(
                            f"Matrix computation failed for factor matrix {factor_matrix_index} "
                            f"with {exception=}. Using previous inverted factor matrix and continuing..."
                        )
                        # Define computed_inv_factor_matrix to prevent undefined local variable error.
                        computed_inv_factor_matrix = inv_factor_matrix

                    # Check if we encounter NaN or inf values in computed inverse matrix.
                    if not torch.isfinite(computed_inv_factor_matrix).all():
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


class EigendecomposedShampooPreconditionerList(
    BaseShampooPreconditionerList[
        EigendecomposedShampooKroneckerFactorsState,
        EigendecomposedShampooKroneckerFactorsList,
    ]
):
    """Eigendecomposed Shampoo preconditioners for list of parameters."""

    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        return tuple(
            attrgetter(INVERSE_EXPONENT_OVERRIDE)(self._preconditioner_config)
            .get((order := len(dims)), {})
            .get(d, 1 / (2 * order))
            != 0.0
            # Traverse through each dim of a block.
            for d in range(len(dims))
        )

    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        return tuple(
            # Compute the inverse root, 1 / inverse_exponent{_override}, accordingly for each required dim.
            1
            / attrgetter(INVERSE_EXPONENT_OVERRIDE)(self._preconditioner_config)
            .get((order := len(preconditioned_dims_selector)), {})
            .get(k, 1 / (2 * order))
            # Traverse through each dim of a block that requires precondition.
            for k, should_precondition in enumerate(preconditioned_dims_selector)
            if should_precondition
        )

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the eigendecomposed Shampoo preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            return tuple(
                self._precondition_grad(
                    grad=masked_grad,
                    preconditioned_dims_selector=preconditioned_dims_selector,
                    preconditioner_list=tuple(
                        eigenvectors
                        * eigenvalues.add(self._epsilon).pow_(-1.0 / root).unsqueeze(0)
                        @ eigenvectors.T
                        for eigenvectors, eigenvalues, root in zip(
                            kronecker_factors.factor_matrices_eigenvectors,
                            kronecker_factors.factor_matrices_eigenvalues,
                            roots,
                            strict=True,
                        )
                    ),
                )
                for masked_grad, preconditioned_dims_selector, kronecker_factors, roots in zip(
                    masked_grad_list,
                    self._masked_preconditioned_dims_selector_list,
                    self._masked_kronecker_factors_list,
                    self._masked_roots_list,
                    strict=True,
                )
            )

    @torch.compiler.disable
    def _amortized_computation(self) -> None:
        # NOTE: This function currently only computes the eigendecomposition based on
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
                success_tracker: bool = True
                for (
                    factor_matrix,
                    factor_matrix_eigenvectors,
                    factor_matrix_eigenvalues,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.factor_matrices_eigenvectors,
                    kronecker_factors.factor_matrices_eigenvalues,
                    kronecker_factors.factor_matrix_indices,
                    strict=True,
                ):
                    # Incorporate bias correction.
                    bias_corrected_factor_matrix = (
                        factor_matrix / self._bias_correction2
                    )

                    BaseShampooPreconditionerList._check_factor_matrix_for_nan_and_inf(
                        factor_matrix=bias_corrected_factor_matrix,
                        factor_matrix_index=factor_matrix_index,
                    )

                    # Compute inverse preconditioner.
                    eigendecomposition_config = cast(
                        EigendecompositionConfig,
                        self._preconditioner_config.amortized_computation_config,
                    )
                    if isinstance(
                        eigendecomposition_config, QREigendecompositionConfig
                    ):
                        # Due to the use of QR algorithm, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., bias_corrected_factor_matrix.
                        eigendecomposition_config.eigenvectors_estimate = (
                            factor_matrix_eigenvectors
                        ).to(dtype=bias_corrected_factor_matrix.dtype)
                    try:
                        computed_eigenvalues, computed_eigenvectors = (
                            matrix_eigendecomposition(
                                A=bias_corrected_factor_matrix,
                                eigendecomposition_config=eigendecomposition_config,
                                is_diagonal=False,
                            )
                        )
                        computed_eigenvalues.to(dtype=factor_matrix_eigenvalues.dtype)
                        computed_eigenvectors.to(dtype=factor_matrix_eigenvectors.dtype)
                        # Add success to success tracker.
                        success_tracker &= True
                    except Exception as exception:
                        # Add failure to success tracker.
                        success_tracker &= False
                        logger.warning(
                            f"Matrix computation failed for factor matrix {factor_matrix_index} "
                            f"with {exception=}. Using previous inverted factor matrix and continuing..."
                        )
                        # Define computed_eigenvalues and computed_eigenvectors to prevent undefined local variable error.
                        computed_eigenvalues = factor_matrix_eigenvalues
                        computed_eigenvectors = factor_matrix_eigenvectors

                    # Check if we encounter NaN or inf values in computed quantities.
                    for computed_quantity, target in (
                        (computed_eigenvalues, factor_matrix_eigenvalues),
                        (computed_eigenvectors, factor_matrix_eigenvectors),
                    ):
                        if not torch.isfinite(computed_quantity).all():
                            quantity_name = f"{computed_quantity=}".split("=")[0].split(
                                "_"
                            )[-1]
                            torch.set_printoptions(threshold=100_000)
                            raise PreconditionerValueError(
                                f"Encountered nan or inf values in {quantity_name} of factor matrix {factor_matrix_index}! "
                                f"To mitigate, check factor matrix before the matrix computation: {bias_corrected_factor_matrix=}"
                            )
                        target.copy_(computed_quantity)

                # Only reuse previous inverse roots if tolerance is not exceeded.
                self._raise_exception_if_failure_tolerance_exceeded(
                    success_tracker=success_tracker,
                    preconditioner_index=idx,
                    exception=ValueError(
                        f"The number of failed eigendecompositions for factors {kronecker_factors.factor_matrix_indices} exceeded the allowed tolerance."
                    ),
                )


class EigenvalueCorrectedShampooPreconditionerList(
    BaseShampooPreconditionerList[
        EigenvalueCorrectedShampooKroneckerFactorsState,
        EigenvalueCorrectedShampooKroneckerFactorsList,
    ]
):
    """Eigenvalue-corrected Shampoo preconditioners for list of parameters."""

    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        return tuple(
            d
            not in attrgetter("ignored_basis_change_dims")(
                self._preconditioner_config
            ).get(len(dims), [])
            # Traverse through each dim of a block.
            for d in range(len(dims))
        )

    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        # NOTE: In eigenvalue-corrected Shampoo, there is only a single inverse root that is applied to the corrected eigenvalues.
        return (
            # Compute the inverse root, 1 / eigenvalue_inverse_exponent{_override}.
            1
            / attrgetter(INVERSE_EXPONENT_OVERRIDE)(self._preconditioner_config).get(
                len(preconditioned_dims_selector), 1 / 2
            ),
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
            for grad, preconditioned_dims_selector, kronecker_factors in zip(
                masked_grad_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_list,
                strict=True,
            ):
                factor_eigenvectors = kronecker_factors.factor_matrices_eigenvectors
                if factor_eigenvectors and factor_eigenvectors[0].any():
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioned_dims_selector=preconditioned_dims_selector,
                        preconditioner_list=factor_eigenvectors,
                    )
                # Update corrected eigenvalues (squared gradient in eigenbasis of Shampoo preconditioner).
                if self._beta2 != 1.0:
                    kronecker_factors.corrected_eigenvalues.mul_(self._beta2)
                    kronecker_factors.corrected_eigenvalues.addcmul_(
                        grad,
                        grad,
                        value=1 - self._beta2,
                    )
                else:
                    # NOTE: The case when self._beta2 == 1.0 is not well tested and might not be stable.
                    kronecker_factors.corrected_eigenvalues.addcmul_(grad, grad)

    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the eigenvalue-corrected Shampoo preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        with profiler.record_function(
            f"## {self.__class__.__name__}:{self.precondition.__name__} ##"
        ):
            preconditioned_grad_list = []
            for (
                masked_grad,
                preconditioned_dims_selector,
                kronecker_factors,
                roots,
            ) in zip(
                masked_grad_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_list,
                self._masked_roots_list,
                strict=True,
            ):
                factor_eigenvectors = kronecker_factors.factor_matrices_eigenvectors
                corrected_eigenvalues = kronecker_factors.corrected_eigenvalues
                use_eigenbasis = factor_eigenvectors and factor_eigenvectors[0].any()
                grad = masked_grad.clone()
                if use_eigenbasis:
                    # Convert to eigenbasis of Shampoo factor matrices.
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioned_dims_selector=preconditioned_dims_selector,
                        preconditioner_list=factor_eigenvectors,
                    )

                # Verify that the number of roots is 1 in Eigenvalue-Corrected Shampoo preconditioner.
                assert len(roots) == 1, f"{len(roots)=} != 1"
                # Precondition with inverse root of corrected eigenvalues.
                grad.div_(
                    corrected_eigenvalues.div(self._bias_correction2)
                    .add_(self._epsilon)
                    .pow_(1 / roots[0])
                )
                if use_eigenbasis:
                    # Convert back to basis of the parameters.
                    grad = self._precondition_grad(
                        grad=grad,
                        preconditioned_dims_selector=preconditioned_dims_selector,
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
                success_tracker: bool = True
                for (
                    factor_matrix,
                    factor_matrix_eigenvectors,
                    factor_matrix_index,
                ) in zip(
                    kronecker_factors.factor_matrices,
                    kronecker_factors.factor_matrices_eigenvectors,
                    kronecker_factors.factor_matrix_indices,
                    strict=True,
                ):
                    BaseShampooPreconditionerList._check_factor_matrix_for_nan_and_inf(
                        factor_matrix=factor_matrix,
                        factor_matrix_index=factor_matrix_index,
                    )

                    # Compute eigenvectors of factor matrix.
                    eigendecomposition_config = cast(
                        EigendecompositionConfig,
                        self._preconditioner_config.amortized_computation_config,
                    )
                    if isinstance(
                        eigendecomposition_config, QREigendecompositionConfig
                    ):
                        # Due to the use of QR algorithm, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., factor_matrix.
                        eigendecomposition_config.eigenvectors_estimate = (
                            factor_matrix_eigenvectors
                        ).to(dtype=factor_matrix.dtype)
                    try:
                        computed_eigenvectors = matrix_eigendecomposition(
                            A=factor_matrix,
                            eigendecomposition_config=eigendecomposition_config,
                            is_diagonal=False,
                        )[1].to(dtype=factor_matrix_eigenvectors.dtype)
                        # Add success to success tracker.
                        success_tracker &= True
                    except Exception as exception:
                        # Add failure to success tracker.
                        success_tracker &= False
                        logger.warning(
                            f"Matrix computation failed for factor matrix {factor_matrix_index} "
                            f"with {exception=}. Using previous factor matrix eigenvectors and continuing..."
                        )
                        # Define computed_eigenvectors to prevent undefined local variable error.
                        computed_eigenvectors = factor_matrix_eigenvectors

                    # Check if we encounter NaN or inf values in computed eigenvectors.
                    if not torch.isfinite(computed_eigenvectors).all():
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
