"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import reduce, wraps
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import Any, cast, Generic, get_args, NoReturn, overload, TypeVar

import torch
from distributed_shampoo.shampoo_types import (
    AmortizedPreconditionerConfig,
    PreconditionerValueError,
    SpectralDescentPreconditionerConfig,
)
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size
from matrix_functions import (
    matrix_eigendecomposition,
    matrix_inverse_root,
    matrix_orthogonalization,
    stabilize_and_pow_eigenvalues,
)

from matrix_functions_types import EigendecompositionConfig, RootInvConfig
from optimizer_modules import OptimizerModule
from torch import Tensor
from torch.autograd import profiler


logger: logging.Logger = logging.getLogger(__name__)

ADAGRAD = "adagrad"
SHAMPOO = "shampoo"
INVERSE_EXPONENT_OVERRIDE = "inverse_exponent_override"


_MemberFuncReturnType = TypeVar("_MemberFuncReturnType")


def _profile_decorator(
    member_func: Callable[..., _MemberFuncReturnType],
) -> Callable[..., _MemberFuncReturnType]:
    """Decorator that profiles the execution of a class method.

    This decorator wraps a class method with PyTorch's profiler.record_function
    to track its execution time and resource usage. The profiling information
    is recorded with a name that includes the class name and method name.

    Args:
        member_func (Callable[..., _MemberFuncReturnType]): The class method to be profiled.

    Returns:
        wrapper (Callable[..., _MemberFuncReturnType]): A wrapped function that profiles the execution of the original method.
    """

    @wraps(member_func)
    def wrapper(them: object, *args: Any, **kwargs: Any) -> _MemberFuncReturnType:
        with profiler.record_function(
            f"## {them.__class__.__name__}:{member_func.__name__} ##"
        ):
            return member_func(them, *args, **kwargs)

    return wrapper


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


class SpectralDescentPreconditionerList(PreconditionerList):
    """Preconditioner list for spectral descent.

    NOTE: This algorithm can only be used for 2D parameters, or parameters that have been reshaped to 2D.
    Which parameters are reshaped to 2D is determined by the max_preconditioner_dim argument in DistributedShampoo (assuming use_merge_dims=True).

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        preconditioner_config (SpectralDescentPreconditionerConfig): Configuration for spectral descent.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        preconditioner_config: SpectralDescentPreconditionerConfig,
    ) -> None:
        if any(block.dim() != 2 for block in block_list):
            raise ValueError(
                "Spectral descent can only be used for 2D parameters, or parameters that have been reshaped to 2D."
            )
        super().__init__(block_list)
        self._preconditioner_config = preconditioner_config

    @_profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        return

    @_profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return tuple(
            # An error will be raised when grad is not 2D.
            matrix_orthogonalization(
                grad,
                orthogonalization_config=self._preconditioner_config.orthogonalization_config,
            )
            for grad in masked_grad_list
        )

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        return


_SubStateValueType = TypeVar("_SubStateValueType")
_StateValueType = dict[Hashable, _SubStateValueType]


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
        state (Mapping[Tensor, _StateValueType]): Mapping containing optimizer state.
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
        state: Mapping[Tensor, _StateValueType],
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
            assert block_index in state[block_info.param], (
                f"{block_index=} not found in {state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] "
                "within DistributedShampoo._initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                "within Distributor for the correctness of block_index."
            )
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

    @_profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
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

    @_profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions the gradient list using the AdaGrad preconditioner.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A tuple of gradients with None values removed.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        masked_bias_corrected_preconditioner_list = torch._foreach_div(
            self._masked_preconditioner_list,
            self._bias_correction2,
        )
        torch._foreach_sqrt_(masked_bias_corrected_preconditioner_list)
        torch._foreach_add_(masked_bias_corrected_preconditioner_list, self._epsilon)
        return torch._foreach_div(
            masked_grad_list, masked_bias_corrected_preconditioner_list
        )

    @_profile_decorator
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        self._masked_preconditioner_list = compress_list(
            self._local_preconditioner_list, local_grad_selector
        )


@dataclass
class BaseShampooKroneckerFactorsState(OptimizerModule):
    """Base class for Shampoo Kronecker factors (wrapped).

    Attributes:
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "BaseShampooKroneckerFactorsState":
        """
        Creates a BaseShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.

        Returns:
            kronecker_factors_state (BaseShampooKroneckerFactorsState): An instance of BaseShampooKroneckerFactorsState with initialized factor matrices and indices.
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
        super().__init__()  # Add this because the synthesized __init__() does not call super().__init__().
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)


@dataclass
class BaseShampooKroneckerFactorsUnwrapped:
    """Base class for Shampoo Kronecker factors (unwrapped).

    This class represents the unwrapped version of Kronecker factors used in Shampoo optimization.
    Unwrapped tensors are used during the actual computation phase of the optimizer, as opposed
    to the wrapped versions which are stored in the optimizer state.

    Attributes:
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
    """

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]
    roots: tuple[float, ...]

    def __post_init__(self) -> None:
        # NOTE: Due to EigenvalueCorrectedShampooKroneckerFactorsState's roots usage, which is one root only applied on corrected eigenvalues,
        # there is no check of roots with other fields.
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)


@dataclass(kw_only=True)
class RootInvShampooKroneckerFactorsState(BaseShampooKroneckerFactorsState):
    """Shampoo Kronecker factors (wrapped) for storing in the optimizer state.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_block(cls, **kwargs: Any) -> "RootInvShampooKroneckerFactorsState":
        """
        Creates a RootInvShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.
            block_dtype (torch.dtype): Data type for the block.

        Returns:
            kronecker_factors_state (RootInvShampooKroneckerFactorsState): An instance of RootInvShampooKroneckerFactorsState with initialized inverse factor matrices.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        block_dtype: torch.dtype = kwargs["block_dtype"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
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
class RootInvShampooKroneckerFactorsUnwrapped(BaseShampooKroneckerFactorsUnwrapped):
    """Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
    ) -> "RootInvShampooKroneckerFactorsUnwrapped":
        """
        Constructs a RootInvShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor matrices and their indices.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots.

        Returns:
            kronecker_factors_unwrapped (RootInvShampooKroneckerFactorsUnwrapped): An instance of RootInvShampooKroneckerFactorsUnwrapped.
        """
        assert isinstance(kronecker_factors_state, RootInvShampooKroneckerFactorsState)
        return cls(
            inv_factor_matrices=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.inv_factor_matrices,
                )
            ),
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.roots)
            == len(self.factor_matrices)
            == len(self.inv_factor_matrices)
        )


@dataclass(kw_only=True)
class EigendecomposedShampooKroneckerFactorsState(BaseShampooKroneckerFactorsState):
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
            kronecker_factors_state (EigendecomposedShampooKroneckerFactorsState): An instance of EigendecomposedShampooKroneckerFactorsState.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        block_dtype: torch.dtype = kwargs["block_dtype"]

        return cls(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
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
class EigendecomposedShampooKroneckerFactorsUnwrapped(
    BaseShampooKroneckerFactorsUnwrapped
):
    """Eigendecomposed Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        factor_matrices_eigenvalues (tuple[Tensor, ...]): A tuple of tensors representing the eigenvalues of the factor matrices.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    factor_matrices_eigenvalues: tuple[Tensor, ...]

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
    ) -> "EigendecomposedShampooKroneckerFactorsUnwrapped":
        """
        Constructs an EigendecomposedShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor matrices and their indices.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots.

        Returns:
            kronecker_factors_unwrapped (EigendecomposedShampooKroneckerFactorsUnwrapped): An instance of EigendecomposedShampooKroneckerFactorsUnwrapped.
        """
        assert isinstance(
            kronecker_factors_state, EigendecomposedShampooKroneckerFactorsState
        )
        return cls(
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
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert (
            len(self.roots)
            == len(self.factor_matrices)
            == len(self.factor_matrices_eigenvectors)
            == len(self.factor_matrices_eigenvalues)
        )


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooKroneckerFactorsState(BaseShampooKroneckerFactorsState):
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
        """
        Creates an EigenvalueCorrectedShampooKroneckerFactorsState object for a given block.

        Args:
            block_info (BlockInfo): Information about the block, including methods to allocate tensors.
            factor_matrix_dtype (torch.dtype): Data type for the factor matrices.
            preconditioned_dims (tuple[int, ...]): Dimensions for which the factor matrices are preconditioned.
            block_dtype (torch.dtype): Data type for the block.
            dims (tuple[int, ...]): Dimensions of the block.

        Returns:
            kronecker_factors_state (EigenvalueCorrectedShampooKroneckerFactorsState): An instance of EigenvalueCorrectedShampooKroneckerFactorsState.
        """
        block_info: BlockInfo = kwargs["block_info"]
        factor_matrix_dtype: torch.dtype = kwargs["factor_matrix_dtype"]
        preconditioned_dims: tuple[int, ...] = kwargs["preconditioned_dims"]
        block_dtype: torch.dtype = kwargs["block_dtype"]
        dims: tuple[int, ...] = kwargs["dims"]

        return EigenvalueCorrectedShampooKroneckerFactorsState(
            **asdict(
                BaseShampooKroneckerFactorsState.from_block(
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
class EigenvalueCorrectedShampooKroneckerFactorsUnwrapped(
    BaseShampooKroneckerFactorsUnwrapped
):
    """Eigenvalue-corrected Shampoo Kronecker factors (unwrapped) for operations during optimizer computation.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the eigenvectors of the factor matrices.
        corrected_eigenvalues (Tensor): A tensor representing the corrected eigenvalues.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of the factor matrices.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to the corrected eigenvalues during preconditioner computation.
            Note that for eigenvalue-corrected Shampoo, this always contains only a single value.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
    ) -> "EigenvalueCorrectedShampooKroneckerFactorsUnwrapped":
        """
        Constructs an EigenvalueCorrectedShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor matrices and their indices.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots.
                For eigenvalue-corrected Shampoo, this always contains only a single value.

        Returns:
            kronecker_factors_unwrapped (EigenvalueCorrectedShampooKroneckerFactorsUnwrapped): An instance of EigenvalueCorrectedShampooKroneckerFactorsUnwrapped.
        """
        assert isinstance(
            kronecker_factors_state, EigenvalueCorrectedShampooKroneckerFactorsState
        )
        return cls(
            factor_matrices_eigenvectors=tuple(
                map(
                    unwrapped_tensor_getter,
                    kronecker_factors_state.factor_matrices_eigenvectors,
                )
            ),
            corrected_eigenvalues=unwrapped_tensor_getter(
                kronecker_factors_state.corrected_eigenvalues
            ),
            factor_matrices=tuple(
                map(unwrapped_tensor_getter, kronecker_factors_state.factor_matrices)
            ),
            factor_matrix_indices=kronecker_factors_state.factor_matrix_indices,
            roots=roots,
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)
        assert len(self.roots) == 1


_ShampooKroneckerFactorsStateType = TypeVar(
    "_ShampooKroneckerFactorsStateType",
    RootInvShampooKroneckerFactorsState,
    EigendecomposedShampooKroneckerFactorsState,
    EigenvalueCorrectedShampooKroneckerFactorsState,
)
_ShampooKroneckerFactorsUnwrappedType = TypeVar(
    "_ShampooKroneckerFactorsUnwrappedType",
    RootInvShampooKroneckerFactorsUnwrapped,
    EigendecomposedShampooKroneckerFactorsUnwrapped,
    EigenvalueCorrectedShampooKroneckerFactorsUnwrapped,
)


class BaseShampooPreconditionerList(
    PreconditionerList,
    Generic[_ShampooKroneckerFactorsStateType, _ShampooKroneckerFactorsUnwrappedType],
):
    """Base class for Shampoo preconditioners.

    NOTE: Does not support sparse gradients at this time.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        state (Mapping[Tensor, _StateValueType]): Mapping containing optimizer state.
        block_info_list (tuple[BlockInfo, ...]): List containing corresponding BlockInfo for each block/parameter in block_list.
            Note that this should have the same length as block_list.
        preconditioner_config (AmortizedPreconditionerConfig): Configuration for preconditioner computation.
        beta2 (float): Exponential moving average factor for Shampoo factor matrices. If beta2 = 1., will use unweighted sum.
            (Default: 1.0)
        epsilon (float): Epsilon term for regularizing preconditioner to ensure positive definiteness. (Default: 1e-12)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        factor_matrix_dtype (torch.dtype): Data type for accumulating and computing root inverse of preconditioners. (Default: torch.float)

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        preconditioner_config: AmortizedPreconditionerConfig,
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
        kronecker_factors_unwrapped: list[_ShampooKroneckerFactorsUnwrappedType] = (
            self._create_kronecker_factors_state(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                preconditioned_dims_list=preconditioned_dims_list,
                preconditioned_dims_selector_list=preconditioned_dims_selector_list,
            )
        )

        # Initialize state lists.
        self._initialize_state_lists(
            block_list=block_list,
            kronecker_factors_unwrapped=kronecker_factors_unwrapped,
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

    def _create_kronecker_factors_state(
        self,
        block_list: tuple[Tensor, ...],
        state: Mapping[Tensor, _StateValueType],
        block_info_list: tuple[BlockInfo, ...],
        preconditioned_dims_list: tuple[tuple[int, ...], ...],
        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...],
    ) -> list[_ShampooKroneckerFactorsUnwrappedType]:
        # Instantiate (blocked) Kronecker factors and construct list of Kronecker factors.
        # NOTE: We need to instantiate the Kronecker factor states within the optimizer's state dictionary,
        # and do not explicitly store them as RootInvShampooPreconditionerList attributes here.
        # This is because the optimizer state is defined per-parameter, but RootInvShampooPreconditionerList is defined
        # across each parameter group (which includes multiple parameters).
        kronecker_factors_unwrapped = []
        for (
            block,
            block_info,
            dims,
            preconditioned_dims,
            preconditioned_dims_selector,
        ) in zip(
            block_list,
            block_info_list,
            self._dims_list,
            preconditioned_dims_list,
            preconditioned_dims_selector_list,
            strict=True,
        ):
            param_index, block_index = block_info.composable_block_ids
            assert block_index in state[block_info.param], (
                f"{block_index=} not found in {state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] "
                "within DistributedShampoo._initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                "within Distributor for the correctness of block_index."
            )
            block_state = state[block_info.param][block_index]
            # NOTE: Use types.get_original_bases() instead of self.__orig_bases__ when downstream applications are Python 3.12+ available
            kronecker_factors_state_type, kronecker_factors_state_unwrapped_type = (
                get_args(attrgetter("__orig_bases__")(self)[0])
            )
            block_state[SHAMPOO] = kronecker_factors_state_type.from_block(
                block_info=block_info,
                factor_matrix_dtype=self._factor_matrix_dtype,
                preconditioned_dims=preconditioned_dims,
                block_dtype=block.dtype,
                dims=dims,
            )
            kronecker_factors_unwrapped.append(
                kronecker_factors_state_unwrapped_type.from_kronecker_factors_state(
                    kronecker_factors_state=block_state[SHAMPOO],
                    unwrapped_tensor_getter=block_info.get_tensor,
                    roots=self._get_inverse_roots_from_override(
                        preconditioned_dims_selector
                    ),
                )
            )

            logger.info(
                f"Instantiated Shampoo Preconditioner {str(param_index) + '.' + str(block_index)} for Parameter {param_index} ({block_info.param.shape}), Block {block_index} ({block.shape})."
            )

        return kronecker_factors_unwrapped

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

    @abstractmethod
    def _amortized_computation(self) -> None:
        """
        Computes the amortized computation needed for each Shampoo preconditioner implementation.
        This amortized computation is computation heavy work that cannot be done for each step.
        """

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
        last_seen_exception: Exception | None,
        preconditioner_index: int,
        exception_message: str,
    ) -> None:
        """Raises an exception if the number of failed amortized computations exceeds the tolerance.

        Resets the counter at the given index when all amortized computations are successful.

        Args:
            last_seen_exception (Exception | None): The last exception encountered, or None if no exception occurred.
                When None, the counter is reset.
                When an Exception, the counter is incremented and checked against tolerance.
            preconditioner_index (int): The index of the preconditioner in the counter list.
            exception_message (str): The message to include in the raised exception.

        Raises:
            ValueError: If the number of failed computations exceeds the allowed tolerance.
                The original exception is attached as the cause.
        """
        if last_seen_exception is None:
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
                raise ValueError(exception_message) from last_seen_exception

    @_profile_decorator
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
        kronecker_factors_unwrapped: list[_ShampooKroneckerFactorsUnwrappedType],
        preconditioned_dims_list: tuple[tuple[int, ...], ...],
        preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...],
    ) -> None:
        # Initialize local lists.
        self._local_kronecker_factors_unwrapped: tuple[
            _ShampooKroneckerFactorsUnwrappedType,
            ...,
        ] = tuple(kronecker_factors_unwrapped)
        self._local_order_list: tuple[int, ...] = tuple(
            block.dim() for block in block_list
        )
        self._local_failed_amortized_computation_counter_list: list[int] = [0] * len(
            self._local_kronecker_factors_unwrapped
        )
        self._local_preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = (
            preconditioned_dims_selector_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: tuple[int, ...] = self._local_order_list
        self._masked_failed_amortized_computation_counter_list: list[int] = (
            self._local_failed_amortized_computation_counter_list
        )
        self._masked_kronecker_factors_unwrapped: tuple[
            _ShampooKroneckerFactorsUnwrappedType,
            ...,
        ] = self._local_kronecker_factors_unwrapped
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

    @_profile_decorator
    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        self._masked_order_list = compress_list(
            self._local_order_list, local_grad_selector
        )
        self._masked_failed_amortized_computation_counter_list = list(
            compress_list(
                self._local_failed_amortized_computation_counter_list,
                local_grad_selector,
            )
        )
        self._masked_kronecker_factors_unwrapped = compress_list(
            self._local_kronecker_factors_unwrapped, local_grad_selector
        )
        self._masked_preconditioned_dims_selector_list = compress_list(
            self._local_preconditioned_dims_selector_list, local_grad_selector
        )

    @_profile_decorator
    def _update_factor_matrices(self, masked_grad_list: tuple[Tensor, ...]) -> None:
        # NOTE: Unlike AdagradPreconditionerList, we will loop through each gradient individually.
        # We apply foreach operators onto the list of Kronecker factor matrices (as opposed to the
        # full list of gradients/optimizer states).
        for grad, order, preconditioned_dims_selector, kronecker_factors in zip(
            masked_grad_list,
            self._masked_order_list,
            self._masked_preconditioned_dims_selector_list,
            self._masked_kronecker_factors_unwrapped,
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

    @overload
    @staticmethod
    def _save_and_handle_matrix_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        error_handler: Callable[[], NoReturn],
    ) -> NoReturn: ...

    @overload
    @staticmethod
    def _save_and_handle_matrix_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        error_handler: Callable[[], None],
    ) -> None: ...

    @staticmethod
    def _save_and_handle_matrix_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        error_handler: Callable[[], NoReturn | None],
    ) -> NoReturn | None:
        """
        Saves a problematic matrix for debugging and configures detailed tensor printing.

        When numerical issues occur in matrix operations, this method:
        1. Creates a temporary directory and saves the matrix for later analysis
        2. Configures PyTorch's print options to show full tensor details
        3. Executes the provided error handler function

        This approach facilitates debugging of numerical instabilities in preconditioner computations.

        Args:
            factor_matrix_index: Identifier for the factor matrix used in the filename
            source_matrix: The problematic matrix to be saved
            error_handler: Function to execute after saving the matrix. This function may:
                - Raise an exception (NoReturn), which will propagate to the caller
                - Return None to continue execution
                - Implement other error handling logic

        Returns:
            NoReturn: If the error_handler raises an exception
            None: If the error_handler returns normally

        Examples of error_handler:
            - Raise a custom exception with a detailed error message.
            - Log a warning message and continue execution.
            - Trigger a fallback mechanism to use default values.
        """
        # Save the problematic matrix to a file for debugging.
        tmp_dir = Path("/tmp").resolve()
        tmp_dir.mkdir(exist_ok=True)
        file_path = tmp_dir / f"{factor_matrix_index.replace('.', '_')}.pt"
        try:
            torch.save(source_matrix, file_path)
            logger.info(f"Matrix has been saved to {file_path} for debugging.")
        except Exception as e:
            logger.warning(f"Failed to save matrix to {file_path}: {str(e)}")

        torch.set_printoptions(
            precision=16,  # Set the precision for floating point numbers to 16 decimal places.
            linewidth=10000,  # Set the line width to 10000, allowing for long lines without wrapping.
            profile="full",  # Use the 'full' profile to display all elements of tensors.
        )
        error_handler()

    @staticmethod
    def _handle_preconditioner_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        quantity_name: str,
    ) -> NoReturn:
        """
        Handles errors related to preconditioner computation by saving the problematic matrix
        and raising a detailed error message.

        This method is called when NaN or inf values are detected in computed matrices during
        preconditioner operations. It saves the source matrix to a file for debugging purposes
        and raises a PreconditionerValueError with detailed information.

        Args:
            factor_matrix_index (str): The index identifier for the factor matrix.
            source_matrix (Tensor): The source matrix that caused the computation error.
            quantity_name (str): Description of the quantity being computed.

        Raises:
            PreconditionerValueError: Error with details about the problematic matrix.
        """

        def raise_preconditioner_value_error() -> NoReturn:
            raise PreconditionerValueError(
                f"Encountered nan or inf values in {quantity_name} of factor matrix {factor_matrix_index}! "
                f"To mitigate, check factor matrix before the matrix computation: {source_matrix=}"
            )

        BaseShampooPreconditionerList._save_and_handle_matrix_error(
            factor_matrix_index=factor_matrix_index,
            source_matrix=source_matrix,
            error_handler=raise_preconditioner_value_error,
        )

    @staticmethod
    def _handle_amortized_computation_internal_error(
        factor_matrix_index: str,
        source_matrix: Tensor,
        exception: Exception,
        preconditioner_name: str,
    ) -> None:
        """
        Handles internal errors during amortized computation by saving the problematic matrix
        and logging a warning message.

        This method is called when an exception occurs during matrix computations in the amortized
        computation phase. It saves the source matrix to a file for debugging purposes and logs
        a warning with detailed information about the error, allowing the optimization to continue
        using the previous preconditioner.

        Args:
            factor_matrix_index (str): The index identifier for the factor matrix.
            source_matrix (Tensor): The source matrix that caused the computation error.
            exception (Exception): The exception that was raised during computation.
            preconditioner_name (str): Name of the preconditioner being computed.

        Returns:
            None
        """
        BaseShampooPreconditionerList._save_and_handle_matrix_error(
            factor_matrix_index=factor_matrix_index,
            source_matrix=source_matrix,
            error_handler=lambda: logger.warning(
                f"Matrix computation failed for factor matrix {factor_matrix_index} with {exception=}. "
                f"To investigate, check factor matrix before the matrix computation: {source_matrix=}. "
                f"Using previous {preconditioner_name} and continuing..."
            ),
        )

    @abstractmethod
    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: _ShampooKroneckerFactorsUnwrappedType,
    ) -> Tensor:
        """
        Applies the Shampoo preconditioner to a gradient tensor.

        This method is implemented by subclasses to perform the actual preconditioning
        operation using the specific preconditioner implementation (root inverse,
        eigendecomposed, or eigenvalue-corrected).

        Args:
            grad (Tensor): The gradient tensor to be preconditioned.
            preconditioned_dims_selector (tuple[bool, ...]): A boolean tuple indicating which
                dimensions of the gradient should be preconditioned. Dimensions with True
                values will be preconditioned, while dimensions with False values will not.
            kronecker_factors (_ShampooKroneckerFactorsUnwrappedType): The unwrapped Kronecker
                factors containing the necessary matrices for preconditioning.

        Returns:
            preconditioned_grad (Tensor): The preconditioned gradient tensor.
        """

    @_profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """
        Preconditions a list of gradients using the Shampoo preconditioner that rely on ShampooPreconditionerConfig.

        Args:
            masked_grad_list (tuple[Tensor, ...]): A list of gradients with their corresponding masks.

        Returns:
            preconditioned_grads (tuple[Tensor, ...]): A list of preconditioned gradients.
        """
        return tuple(
            self._compute_preconditioned_gradient(
                grad=masked_grad,
                preconditioned_dims_selector=preconditioned_dims_selector,
                kronecker_factors=kronecker_factors,
            )
            for masked_grad, preconditioned_dims_selector, kronecker_factors in zip(
                masked_grad_list,
                self._masked_preconditioned_dims_selector_list,
                self._masked_kronecker_factors_unwrapped,
                strict=True,
            )
        )


_ClassicShampooKroneckerFactorsStateType = TypeVar(
    "_ClassicShampooKroneckerFactorsStateType",
    RootInvShampooKroneckerFactorsState,
    EigendecomposedShampooKroneckerFactorsState,
)

_ClassicShampooKroneckerFactorsUnwrappedType = TypeVar(
    "_ClassicShampooKroneckerFactorsUnwrappedType",
    RootInvShampooKroneckerFactorsUnwrapped,
    EigendecomposedShampooKroneckerFactorsUnwrapped,
)


class ClassicShampooPreconditionerList(
    BaseShampooPreconditionerList[
        _ClassicShampooKroneckerFactorsStateType,
        _ClassicShampooKroneckerFactorsUnwrappedType,
    ]
):
    """Base class for Shampoo preconditioners that rely on ShampooPreconditionerConfig.

    This class factors out common implementations for Shampoo preconditioners that use
    ShampooPreconditionerConfig to determine inverse exponent overrides and preconditioned dimensions.
    It provides methods to retrieve inverse exponent overrides based on dimension and order,
    and to create preconditioned dimension selectors.

    """

    def _get_inverse_exponent(self, dimension: int, order: int) -> float:
        """
        Retrieves the inverse exponent override based on the dimension and order.

        Args:
            dimension (int): The dimension for which the inverse exponent override is needed.
            order (int): The order of the preconditioner.

        Returns:
            float: The inverse exponent override value for the given dimension and order.
        """
        inverse_exponent_override_on_order: dict[int, float] | float = attrgetter(
            INVERSE_EXPONENT_OVERRIDE
        )(self._preconditioner_config).get(order, {})
        if isinstance(inverse_exponent_override_on_order, dict):
            return inverse_exponent_override_on_order.get(
                dimension, 1 / (2 * max(order, 1))
            )
        assert isinstance(
            inverse_exponent_override_on_order, float
        ), f"Expected inverse_exponent_override_on_order to be a float or a dict, but got {type(inverse_exponent_override_on_order)} instead."
        return inverse_exponent_override_on_order

    def _create_preconditioned_dims_selector(
        self, dims: torch.Size
    ) -> tuple[bool, ...]:
        return tuple(
            self._get_inverse_exponent(dimension=d, order=len(dims)) != 0.0
            # Traverse through each dim of a block.
            for d in range(len(dims))
        )

    def _get_inverse_roots_from_override(
        self, preconditioned_dims_selector: tuple[bool, ...]
    ) -> tuple[float, ...]:
        return tuple(
            # Compute the inverse root, 1 / inverse_exponent{_override}, accordingly for each required dim.
            1
            / self._get_inverse_exponent(
                dimension=k, order=len(preconditioned_dims_selector)
            )
            # Traverse through each dim of a block that requires precondition.
            for k, should_precondition in enumerate(preconditioned_dims_selector)
            if should_precondition
        )


class RootInvShampooPreconditionerList(
    ClassicShampooPreconditionerList[
        RootInvShampooKroneckerFactorsState, RootInvShampooKroneckerFactorsUnwrapped
    ]
):
    """Root inverse Shampoo preconditioners for list of parameters."""

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: RootInvShampooKroneckerFactorsUnwrapped,
    ) -> Tensor:
        return self._precondition_grad(
            grad=grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.inv_factor_matrices,
        )

    @torch.compiler.disable
    @_profile_decorator
    def _amortized_computation(self) -> None:
        # NOTE: This function currently only computes the matrix root inverse based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        for idx, kronecker_factors in enumerate(
            self._masked_kronecker_factors_unwrapped
        ):
            last_seen_exception: Exception | None = None
            for (
                factor_matrix,
                inv_factor_matrix,
                factor_matrix_index,
                root,
            ) in zip(
                kronecker_factors.factor_matrices,
                kronecker_factors.inv_factor_matrices,
                kronecker_factors.factor_matrix_indices,
                kronecker_factors.roots,
                strict=True,
            ):
                # Incorporate bias correction.
                bias_corrected_factor_matrix = factor_matrix / self._bias_correction2

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
                except Exception as exception:
                    # Track the last seen exception.
                    last_seen_exception = exception
                    BaseShampooPreconditionerList._handle_amortized_computation_internal_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=bias_corrected_factor_matrix,
                        exception=exception,
                        preconditioner_name="inverted factor matrix",
                    )
                    # Define computed_inv_factor_matrix to prevent undefined local variable error.
                    computed_inv_factor_matrix = inv_factor_matrix

                # Check if we encounter NaN or inf values in computed inverse matrix.
                if not torch.isfinite(computed_inv_factor_matrix).all():
                    BaseShampooPreconditionerList._handle_preconditioner_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=bias_corrected_factor_matrix,
                        quantity_name="inverse",
                    )
                inv_factor_matrix.copy_(computed_inv_factor_matrix)

            # Only reuse previous inverse roots if tolerance is not exceeded.
            self._raise_exception_if_failure_tolerance_exceeded(
                last_seen_exception=last_seen_exception,
                preconditioner_index=idx,
                exception_message=f"The number of failed inverse root computations for factors {kronecker_factors.factor_matrix_indices} exceeded the allowed tolerance."
                f"The last seen exception was {last_seen_exception}.",
            )


class EigendecomposedShampooPreconditionerList(
    ClassicShampooPreconditionerList[
        EigendecomposedShampooKroneckerFactorsState,
        EigendecomposedShampooKroneckerFactorsUnwrapped,
    ]
):
    """Eigendecomposed Shampoo preconditioners for list of parameters."""

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: EigendecomposedShampooKroneckerFactorsUnwrapped,
    ) -> Tensor:
        # TODO: remove assertion when rank_deficient_stability_config is generalized to MatrixFunctionConfig
        assert isinstance(
            self._preconditioner_config.amortized_computation_config,
            EigendecompositionConfig,
        )
        rank_deficient_stability_config = self._preconditioner_config.amortized_computation_config.rank_deficient_stability_config

        return self._precondition_grad(
            grad=grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=tuple(
                eigenvectors
                * stabilize_and_pow_eigenvalues(
                    eigenvalues,
                    root=Fraction(root),
                    epsilon=self._epsilon,
                    rank_deficient_stability_config=rank_deficient_stability_config,
                ).unsqueeze(0)
                @ eigenvectors.T
                for eigenvectors, eigenvalues, root in zip(
                    kronecker_factors.factor_matrices_eigenvectors,
                    kronecker_factors.factor_matrices_eigenvalues,
                    kronecker_factors.roots,
                    strict=True,
                )
            ),
        )

    @torch.compiler.disable
    @_profile_decorator
    def _amortized_computation(self) -> None:
        # NOTE: This function currently only computes the eigendecomposition based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        for idx, kronecker_factors in enumerate(
            self._masked_kronecker_factors_unwrapped
        ):
            last_seen_exception: Exception | None = None
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
                bias_corrected_factor_matrix = factor_matrix / self._bias_correction2

                BaseShampooPreconditionerList._check_factor_matrix_for_nan_and_inf(
                    factor_matrix=bias_corrected_factor_matrix,
                    factor_matrix_index=factor_matrix_index,
                )

                # Compute inverse preconditioner.
                eigendecomposition_config = cast(
                    EigendecompositionConfig,
                    self._preconditioner_config.amortized_computation_config,
                )
                # To estimate the eigenvalues based on the previous eigenvectors, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., bias_corrected_factor_matrix.
                eigenvectors_estimate = factor_matrix_eigenvectors.to(
                    dtype=bias_corrected_factor_matrix.dtype
                )
                try:
                    computed_eigenvalues, computed_eigenvectors = (
                        matrix_eigendecomposition(
                            A=bias_corrected_factor_matrix,
                            eigendecomposition_config=eigendecomposition_config,
                            eigenvectors_estimate=eigenvectors_estimate,
                            is_diagonal=False,
                            epsilon=self._epsilon,
                        )
                    )
                    computed_eigenvalues.to(dtype=factor_matrix_eigenvalues.dtype)
                    computed_eigenvectors.to(dtype=factor_matrix_eigenvectors.dtype)
                except Exception as exception:
                    # Track the last seen exception.
                    last_seen_exception = exception
                    BaseShampooPreconditionerList._handle_amortized_computation_internal_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=bias_corrected_factor_matrix,
                        exception=exception,
                        preconditioner_name="inverted factor matrix",
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
                        BaseShampooPreconditionerList._handle_preconditioner_error(
                            factor_matrix_index=factor_matrix_index,
                            source_matrix=bias_corrected_factor_matrix,
                            quantity_name=f"{computed_quantity=}".split("=")[0].split(
                                "_"
                            )[-1],
                        )
                    target.copy_(computed_quantity)

            # Only reuse previous inverse roots if tolerance is not exceeded.
            self._raise_exception_if_failure_tolerance_exceeded(
                last_seen_exception=last_seen_exception,
                preconditioner_index=idx,
                exception_message=f"The number of failed eigendecompositions for factors {kronecker_factors.factor_matrix_indices} exceeded the allowed tolerance."
                f"The last seen exception was {last_seen_exception}.",
            )


class EigenvalueCorrectedShampooPreconditionerList(
    BaseShampooPreconditionerList[
        EigenvalueCorrectedShampooKroneckerFactorsState,
        EigenvalueCorrectedShampooKroneckerFactorsUnwrapped,
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

    @_profile_decorator
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
        super().update_preconditioners(
            masked_grad_list=masked_grad_list,
            step=step,
            perform_amortized_computation=perform_amortized_computation,
        )

        # Update the eigenvalue corrections of Shampoo's preconditioner.
        for grad, preconditioned_dims_selector, kronecker_factors in zip(
            masked_grad_list,
            self._masked_preconditioned_dims_selector_list,
            self._masked_kronecker_factors_unwrapped,
            strict=True,
        ):
            # Transform the gradient to eigenbasis of Shampoo's factor matrices.
            # Because of preconditioned_dims_selector, this might be a no-op.
            grad = self._precondition_grad(
                grad=grad,
                preconditioned_dims_selector=preconditioned_dims_selector,
                preconditioner_list=kronecker_factors.factor_matrices_eigenvectors,
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

    def _compute_preconditioned_gradient(
        self,
        grad: Tensor,
        preconditioned_dims_selector: tuple[bool, ...],
        kronecker_factors: EigenvalueCorrectedShampooKroneckerFactorsUnwrapped,
    ) -> Tensor:
        # Clone the masked gradient to avoid modifying the original tensor.
        # This is only relevant when _precondition_grad is a no-op.
        preconditioned_grad = grad.clone()
        # Transform the gradient to eigenbasis of Shampoo's factor matrices.
        preconditioned_grad = self._precondition_grad(
            grad=preconditioned_grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.factor_matrices_eigenvectors,
        )

        # Precondition with inverse root of corrected eigenvalues.
        # NOTE: We don't use the stabilize_and_pow_eigenvalues function here because:
        # 1. We have to add epsilon even it has been added to the factor matrices before the eigendecomposition already.
        # 2. We compute the root before adding epsilon to be consistent with the PyTorch Adam(W) implementation.
        # 3. We don't support a pseudo-inverse here.
        preconditioned_grad.div_(
            kronecker_factors.corrected_eigenvalues.div(self._bias_correction2)
            .pow_(1 / kronecker_factors.roots[0])
            .add_(self._epsilon)
        )
        # Convert back to basis of the parameters.
        return self._precondition_grad(
            grad=preconditioned_grad,
            preconditioned_dims_selector=preconditioned_dims_selector,
            preconditioner_list=kronecker_factors.factor_matrices_eigenvectors,
            dims=([0], [1]),
        )

    @torch.compiler.disable
    @_profile_decorator
    def _amortized_computation(self) -> None:
        # NOTE: This function currently only computes the preconditioner eigenvectors based on
        # the masked lists which combines both selection based on the distributor and where
        # grad is not None. Implicitly, this assumes that there are no changes between the
        # selector or masking from iteration-to-iteration within a single precondition_frequency
        # interval.
        for idx, kronecker_factors in enumerate(
            self._masked_kronecker_factors_unwrapped
        ):
            last_seen_exception: Exception | None = None
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
                # To estimate the eigenvalues based on the previous eigenvectors, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., factor_matrix.
                eigenvectors_estimate = factor_matrix_eigenvectors.to(
                    dtype=factor_matrix.dtype
                )
                try:
                    computed_eigenvectors = matrix_eigendecomposition(
                        A=factor_matrix,
                        eigendecomposition_config=eigendecomposition_config,
                        eigenvectors_estimate=eigenvectors_estimate,
                        is_diagonal=False,
                        epsilon=self._epsilon,
                    )[1].to(dtype=factor_matrix_eigenvectors.dtype)
                except Exception as exception:
                    # Track the last seen exception.
                    last_seen_exception = exception
                    BaseShampooPreconditionerList._handle_amortized_computation_internal_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=factor_matrix,
                        exception=exception,
                        preconditioner_name="factor matrix eigenvectors",
                    )
                    # Define computed_eigenvectors to prevent undefined local variable error.
                    computed_eigenvectors = factor_matrix_eigenvectors

                # Check if we encounter NaN or inf values in computed eigenvectors.
                if not torch.isfinite(computed_eigenvectors).all():
                    BaseShampooPreconditionerList._handle_preconditioner_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=factor_matrix,
                        quantity_name="eigenvectors",
                    )
                factor_matrix_eigenvectors.copy_(computed_eigenvectors)

            # Only reuse previous eigenvectors if tolerance is not exceeded.
            self._raise_exception_if_failure_tolerance_exceeded(
                last_seen_exception=last_seen_exception,
                preconditioner_index=idx,
                exception_message=f"The number of failed eigenvector computations for factors {kronecker_factors.factor_matrix_indices} exceeded the allowed tolerance."
                f"The last seen exception was {last_seen_exception}.",
            )
