"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import asdict, dataclass, field, fields
from fractions import Fraction
from functools import partial, reduce, wraps
from itertools import chain
from operator import attrgetter
from pathlib import Path
from typing import Any, Generic, get_args, NoReturn, overload, TypeVar

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.preconditioner.matrix_functions import (
    matrix_eigendecomposition,
    matrix_inverse_root,
    matrix_orthogonalization,
    stabilize_and_pow_eigenvalues,
)

from distributed_shampoo.preconditioner.matrix_functions_types import (
    EigendecompositionConfig,
    MatrixFunctionConfig,
    RootInvConfig,
)
from distributed_shampoo.shampoo_types import (
    AmortizedPreconditionerConfig,
    PreconditionerValueError,
    SpectralDescentPreconditionerConfig,
)
from distributed_shampoo.utils.dict_zip_iterator import DictZipIterator
from distributed_shampoo.utils.optimizer_modules import OptimizerModule
from distributed_shampoo.utils.shampoo_utils import compress_list, get_dtype_size
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
    Which parameters are reshaped to 2D is determined by the max_preconditioner_dim argument in DistributedShampoo.
    If all >2D parameters should be guaranteed to be reshaped to 2D, then max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2 has to be used.

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
                "Spectral descent can only be used for 2D parameters, or parameters that have been reshaped to 2D. "
                "To guarantee that all >2D parameters are reshaped to 2D, set max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2."
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
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
            computation of matrix operations, specifying algorithms and parameters for
            eigendecomposition or matrix inverse computation.
        epsilon (float): Small constant added to matrices to ensure numerical stability
            during matrix operations like inversion or eigendecomposition.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    factor_matrices: tuple[Tensor, ...]
    factor_matrix_indices: tuple[str, ...]
    roots: tuple[float, ...]
    amortized_computation_config: MatrixFunctionConfig
    epsilon: float
    num_tolerated_failed_amortized_computations: int
    _failed_amortized_computation_counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # NOTE: Due to EigenvalueCorrectedShampooKroneckerFactorsState's roots usage, which is one root only applied on corrected eigenvalues,
        # there is no check of roots with other fields.
        assert len(self.factor_matrices) == len(self.factor_matrix_indices)

    def _get_field_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary containing shallow copies of this dataclass's fields.

        This method creates a dictionary where keys are field names and values are
        the corresponding field values from the dataclass. Since this is a shallow copy,
        any modifications to the returned dictionary's values will affect the original
        dataclass fields.

        Returns:
            dict[str, Any]: A dictionary mapping field names to their values
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name
            not in (
                "amortized_computation_config",
                "epsilon",
                "num_tolerated_failed_amortized_computations",
                "_failed_amortized_computation_counter",
            )
        }

    @abstractmethod
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Performs computationally expensive matrix operations for Shampoo preconditioners.

        This method handles the heavy computational work that is too expensive to perform
        at every optimization step. Instead, these operations are "amortized" - performed
        periodically (e.g., every N steps) to update the preconditioner matrices.

        Different Shampoo variants implement this method differently:
        - RootInvShampooPreconditionerList: Computes matrix inverse roots
        - EigendecomposedShampooPreconditionerList: Performs eigendecomposition
        - EigenvalueCorrectedShampooPreconditionerList: Computes eigenvectors

        The method includes error handling to gracefully recover from numerical issues
        that may occur during matrix operations.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied. This is the matrix on which the computationally expensive
                operations (like eigendecomposition or matrix inverse) will be performed.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing factor matrices
                and related data needed for the computation. The exact contents depend on the
                specific Shampoo implementation, but typically include factor matrices,
                their indices, and other relevant tensors.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary mapping from tensor names to computed tensors. The keys and values depend on the specific implementation.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.
        """

    @_profile_decorator
    def amortized_computation(self, bias_correction2: float) -> None:
        """Performs amortized computation for Shampoo preconditioners.

        This method orchestrates the execution of the computationally expensive matrix operations
        that are amortized over multiple optimization steps. It applies bias correction to the
        factor matrices and calls the specialized _amortized_computation method for each factor.

        The method handles exceptions that may occur during computation, keeping track of failures
        and raising an exception if the number of failures exceeds the configured tolerance.

        Args:
            bias_correction2 (float): The bias correction factor to apply to the factor matrices
                before performing the amortized computation.

        Raises:
            PreconditionerValueError: If NaN or infinity values are encountered in the factor matrices.
            ValueError: If the number of failed amortized computations exceeds the configured tolerance.
        """
        last_seen_exception: Exception | None = None
        for kronecker_factors_iter_dict in DictZipIterator(data=self._get_field_dict()):
            bias_corrected_factor_matrix, factor_matrix_index = (
                # Incorporate bias correction.
                kronecker_factors_iter_dict["factor_matrices"] / bias_correction2,
                kronecker_factors_iter_dict["factor_matrix_indices"],
            )

            # Check for nan or inf values.
            if not torch.isfinite(bias_corrected_factor_matrix).all():
                raise PreconditionerValueError(
                    f"Encountered nan/inf values in factor matrix {factor_matrix_index}! "
                    "To mitigate, check if nan inputs are being passed into the network or nan gradients are being passed to the optimizer. "
                    "Otherwise, in some cases, this may be due to divergence of the algorithm. To mitigate, try decreasing the learning rate or increasing grafting epsilon. "
                    f"For debugging purposes, factor_matrix {factor_matrix_index}: "
                    f"{torch.min(bias_corrected_factor_matrix)=}, {torch.max(bias_corrected_factor_matrix)=}, "
                    f"{bias_corrected_factor_matrix.isinf().any()=}, {bias_corrected_factor_matrix.isnan().any()=}."
                )

            computed_quantity_to_result, exception = self._amortized_computation(
                bias_corrected_factor_matrix=bias_corrected_factor_matrix,
                kronecker_factors_iter_dict=kronecker_factors_iter_dict,
            )
            if exception:
                last_seen_exception = exception

                BaseShampooPreconditionerList._save_and_handle_matrix_error(
                    factor_matrix_index=factor_matrix_index,
                    source_matrix=bias_corrected_factor_matrix,
                    error_handler=partial(
                        logger.warning,
                        f"Matrix computation failed for factor matrix {factor_matrix_index} with {exception=}. To investigate, check factor matrix before the matrix computation: {bias_corrected_factor_matrix=} Using previous preconditioner and continuing...",
                    ),
                )

            # Check if we encounter NaN or inf values in computed quantities.
            for (
                computed_quantity_name,
                computed_result,
            ) in computed_quantity_to_result.items():
                if not torch.isfinite(computed_result).all():
                    # Define a closure to handle the error with proper variable capture
                    def raise_preconditioner_value_error(
                        factor_matrix_index: str = factor_matrix_index,
                        bias_corrected_factor_matrix: Tensor = bias_corrected_factor_matrix,
                        computed_quantity_name: str = computed_quantity_name,
                    ) -> NoReturn:
                        quantity_name = f"{computed_quantity_name=}".split("=")[
                            0
                        ].split("_")[-1]
                        raise PreconditionerValueError(
                            f"Encountered nan or inf values in {quantity_name} of factor matrix {factor_matrix_index}! "
                            f"To mitigate, check factor matrix before the matrix computation: {bias_corrected_factor_matrix=}"
                        )

                    BaseShampooPreconditionerList._save_and_handle_matrix_error(
                        factor_matrix_index=factor_matrix_index,
                        source_matrix=bias_corrected_factor_matrix,
                        error_handler=raise_preconditioner_value_error,
                    )

                kronecker_factors_iter_dict[computed_quantity_name].copy_(
                    computed_result
                )

        if last_seen_exception is None:
            # Reset counter for failed amortized computations.
            self._failed_amortized_computation_counter = 0
        else:
            # Increment counter for failed amortized computations.
            self._failed_amortized_computation_counter += 1
            # Raise the exception if the tolerance is exceeded.
            if (
                self._failed_amortized_computation_counter
                > self.num_tolerated_failed_amortized_computations
            ):
                raise ValueError(
                    f"The number of failed amortized computations for factors {self.factor_matrix_indices} exceeded the allowed tolerance. The last seen exception was {last_seen_exception}."
                ) from last_seen_exception


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

    This class implements the Root Inverse variant of Shampoo, which directly computes
    the inverse root of factor matrices for preconditioning.

    Attributes:
        inv_factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the inverse
            of the factor matrices. These are the preconditioners that are applied to gradients.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
            computation of matrix operations, specifying algorithms and parameters for
            matrix inverse computation.
        epsilon (float): Small constant added to matrices to ensure numerical stability
            during matrix operations like inversion.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    inv_factor_matrices: tuple[Tensor, ...]

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: RootInvConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
    ) -> "RootInvShampooKroneckerFactorsUnwrapped":
        """
        Constructs a RootInvShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        This method converts the wrapped Kronecker factors state (which is stored in the optimizer state)
        into an unwrapped version that can be used for computation. It unwraps all tensors using the
        provided unwrapped_tensor_getter function and sets up the configuration for matrix operations.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors,
                typically retrieving them from the optimizer state.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor
                matrices and their indices. Must be an instance of RootInvShampooKroneckerFactorsState.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
                to be applied to each factor matrix during preconditioner computation.
            amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
                computation of matrix operations, specifying algorithms and parameters for
                matrix inverse computation.
            epsilon (float): Small constant added to matrices to ensure numerical stability
                during matrix operations like inversion.
            num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
                failed amortized computations that can be tolerated before raising an error.

        Returns:
            kronecker_factors_unwrapped (RootInvShampooKroneckerFactorsUnwrapped): An instance of
                RootInvShampooKroneckerFactorsUnwrapped with unwrapped tensors and configuration.
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
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Computes matrix inverse roots for Shampoo preconditioners.

        This implementation of the abstract _amortized_computation method specifically handles
        the computation of matrix inverse roots for the RootInvShampoo variant. It applies
        the matrix_inverse_root function to each factor matrix with the appropriate root value.

        The computation is performed on the bias-corrected factor matrices and uses the
        configuration specified in amortized_computation_config. Error handling is included
        to gracefully recover from numerical issues.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing the current
                inv_factor_matrices and roots values for the computation.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary with the computed inverse factor matrices.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.

        Note:
            This function assumes there are no changes in the selector or masking between
            iterations within a single precondition_frequency interval.
        """
        inv_factor_matrix, root = (
            kronecker_factors_iter_dict["inv_factor_matrices"],
            kronecker_factors_iter_dict["roots"],
        )

        try:
            # Compute inverse preconditioners
            return {
                "inv_factor_matrices": matrix_inverse_root(
                    A=bias_corrected_factor_matrix,
                    root=Fraction(root),
                    root_inv_config=self.amortized_computation_config,
                    epsilon=self.epsilon,
                ).to(dtype=inv_factor_matrix.dtype)
            }, None
        except Exception as exception:
            return {"inv_factor_matrices": inv_factor_matrix}, exception

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

    This class implements the Eigendecomposed variant of Shampoo, which computes and stores
    the eigendecomposition of factor matrices for more efficient and numerically stable
    preconditioning operations.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the
            eigenvectors of the factor matrices. These are used to transform gradients into
            the eigenspace and back.
        factor_matrices_eigenvalues (tuple[Tensor, ...]): A tuple of tensors representing the
            eigenvalues of the factor matrices. These are used to scale gradients in the eigenspace.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to each factor matrix during preconditioner computation.
        amortized_computation_config (EigendecompositionConfig): Configuration for the amortized
            computation of eigendecomposition, specifying algorithms and parameters.
        epsilon (float): Small constant added to eigenvalues to ensure numerical stability
            during matrix operations.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    factor_matrices_eigenvalues: tuple[Tensor, ...]

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: EigendecompositionConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
    ) -> "EigendecomposedShampooKroneckerFactorsUnwrapped":
        """
        Constructs an EigendecomposedShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        This method converts the wrapped Kronecker factors state (which is stored in the optimizer state)
        into an unwrapped version that can be used for computation. It unwraps all tensors using the
        provided unwrapped_tensor_getter function and sets up the configuration for eigendecomposition.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors,
                typically retrieving them from the optimizer state.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor
                matrices and their indices. Must be an instance of EigendecomposedShampooKroneckerFactorsState.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
                to be applied to each factor matrix during preconditioner computation.
            amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
                computation of eigendecomposition, specifying algorithms and parameters.
            epsilon (float): Small constant added to eigenvalues to ensure numerical stability
                during matrix operations.
            num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
                failed amortized computations that can be tolerated before raising an error.

        Returns:
            kronecker_factors_unwrapped (EigendecomposedShampooKroneckerFactorsUnwrapped): An instance of
                EigendecomposedShampooKroneckerFactorsUnwrapped with unwrapped tensors and configuration.
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
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Performs eigendecomposition for Shampoo preconditioners.

        This implementation of the abstract _amortized_computation method specifically handles
        the eigendecomposition for the EigendecomposedShampoo variant. It computes both
        eigenvalues and eigenvectors for each factor matrix.

        The computation uses the configuration specified in amortized_computation_config,
        with special handling for QR-based eigendecomposition which requires the previous
        eigenvectors as an initial estimate. Error handling is included to gracefully
        recover from numerical issues.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing the current
                factor_matrices_eigenvalues and factor_matrices_eigenvectors for the computation.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary with the computed eigenvalues and eigenvectors.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.

        Note:
            This function assumes there are no changes in the selector or masking between
            iterations within a single precondition_frequency interval.
        """
        (
            factor_matrix_eigenvectors,
            factor_matrix_eigenvalues,
        ) = (
            kronecker_factors_iter_dict["factor_matrices_eigenvectors"],
            kronecker_factors_iter_dict["factor_matrices_eigenvalues"],
        )

        try:
            # Compute inverse preconditioner.
            computed_eigenvalues, computed_eigenvectors = matrix_eigendecomposition(
                A=bias_corrected_factor_matrix,
                eigendecomposition_config=self.amortized_computation_config,
                # To estimate the eigenvalues based on the previous eigenvectors, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., factor_matrix.
                eigenvectors_estimate=factor_matrix_eigenvectors.to(
                    dtype=bias_corrected_factor_matrix.dtype
                ),
                epsilon=self.epsilon,
            )

            return {
                "factor_matrices_eigenvalues": computed_eigenvalues.to(
                    dtype=factor_matrix_eigenvalues.dtype
                ),
                "factor_matrices_eigenvectors": computed_eigenvectors.to(
                    dtype=factor_matrix_eigenvectors.dtype
                ),
            }, None
        except Exception as exception:
            return {
                "factor_matrices_eigenvalues": factor_matrix_eigenvalues,
                "factor_matrices_eigenvectors": factor_matrix_eigenvectors,
            }, exception

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

    This class implements the Eigenvalue-Corrected variant of Shampoo, which computes eigenvectors
    of factor matrices but maintains a separate tensor of corrected eigenvalues that are updated
    directly from gradients. This approach can provide better conditioning and convergence properties
    in certain optimization scenarios.

    Attributes:
        factor_matrices_eigenvectors (tuple[Tensor, ...]): A tuple of tensors representing the
            eigenvectors of the factor matrices. These are used to transform gradients into
            the eigenspace and back.
        corrected_eigenvalues (Tensor): A tensor representing the corrected eigenvalues that are
            updated directly from squared gradients in the eigenspace. This is a single tensor
            rather than a tuple of tensors per factor matrix.
        factor_matrices (tuple[Tensor, ...]): A tuple of tensors representing the factor matrices.
            These are the Kronecker factors accumulated during optimization.
        factor_matrix_indices (tuple[str, ...]): A tuple of strings representing the indices of
            the factor matrices, used for identification and debugging.
        roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
            to be applied to the corrected eigenvalues during preconditioner computation.
            Note that for eigenvalue-corrected Shampoo, this always contains only a single value
            since all eigenvalues are corrected using the same exponent.
        amortized_computation_config (EigendecompositionConfig): Configuration for the amortized
            computation of eigendecomposition, specifying algorithms and parameters.
        epsilon (float): Small constant added to corrected eigenvalues to ensure numerical stability
            during preconditioning operations.
        num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
            failed amortized computations that can be tolerated before raising an error.
        _failed_amortized_computation_counter (int): Internal counter tracking the number
            of consecutive failed amortized computations.
    """

    factor_matrices_eigenvectors: tuple[Tensor, ...]
    corrected_eigenvalues: Tensor

    @classmethod
    def from_kronecker_factors_state(
        cls,
        unwrapped_tensor_getter: Callable[[Tensor], Tensor],
        kronecker_factors_state: BaseShampooKroneckerFactorsState,
        roots: tuple[float, ...],
        amortized_computation_config: EigendecompositionConfig,
        epsilon: float,
        num_tolerated_failed_amortized_computations: int,
    ) -> "EigenvalueCorrectedShampooKroneckerFactorsUnwrapped":
        """
        Constructs an EigenvalueCorrectedShampooKroneckerFactorsUnwrapped object from the given Kronecker factors state.

        This method converts the wrapped Kronecker factors state (which is stored in the optimizer state)
        into an unwrapped version that can be used for computation. It unwraps all tensors using the
        provided unwrapped_tensor_getter function and sets up the configuration for eigendecomposition
        and eigenvalue correction.

        Args:
            unwrapped_tensor_getter (Callable[[Tensor], Tensor]): A function to unwrap tensors,
                typically retrieving them from the optimizer state.
            kronecker_factors_state (BaseShampooKroneckerFactorsState): The state containing factor
                matrices and their indices. Must be an instance of EigenvalueCorrectedShampooKroneckerFactorsState.
            roots (tuple[float, ...]): A tuple of float values representing the inverse exponent roots
                to be applied to the corrected eigenvalues during preconditioner computation.
                For eigenvalue-corrected Shampoo, this always contains only a single value
                since all eigenvalues are corrected using the same exponent.
            amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized
                computation of eigendecomposition, specifying algorithms and parameters.
            epsilon (float): Small constant added to corrected eigenvalues to ensure numerical stability
                during preconditioning operations.
            num_tolerated_failed_amortized_computations (int): Maximum number of consecutive
                failed amortized computations that can be tolerated before raising an error.

        Returns:
            kronecker_factors_unwrapped (EigenvalueCorrectedShampooKroneckerFactorsUnwrapped): An instance of
                EigenvalueCorrectedShampooKroneckerFactorsUnwrapped with unwrapped tensors and configuration.
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
            amortized_computation_config=amortized_computation_config,
            epsilon=epsilon,
            num_tolerated_failed_amortized_computations=num_tolerated_failed_amortized_computations,
        )

    @torch.compiler.disable
    def _amortized_computation(
        self,
        bias_corrected_factor_matrix: Tensor,
        kronecker_factors_iter_dict: dict[str, Any],
    ) -> tuple[dict[str, Tensor], Exception | None]:
        """Computes eigenvectors for eigenvalue-corrected Shampoo preconditioners.

        This implementation of the abstract _amortized_computation method specifically handles
        the computation of eigenvectors for the EigenvalueCorrectedShampoo variant. Unlike
        the EigendecomposedShampoo variant, this only computes eigenvectors and not eigenvalues,
        as the eigenvalues are corrected separately during the optimization process.

        The computation uses the configuration specified in amortized_computation_config,
        with special handling for QR-based eigendecomposition which requires the previous
        eigenvectors as an initial estimate. Error handling is included to gracefully
        recover from numerical issues.

        Args:
            bias_corrected_factor_matrix (Tensor): The factor matrix after bias correction
                has been applied.
            kronecker_factors_iter_dict (dict[str, Any]): Dictionary containing the current
                factor_matrices_eigenvectors for the computation.

        Returns:
            computed_quantities (dict[str, Tensor]): A dictionary with the computed eigenvectors.
            exception (Exception | None): Any exception that occurred during computation, or None if successful.

        Note:
            This function assumes there are no changes in the selector or masking between
            iterations within a single precondition_frequency interval.
        """
        factor_matrix_eigenvectors = kronecker_factors_iter_dict[
            "factor_matrices_eigenvectors"
        ]

        try:
            # Compute eigenvectors of factor matrix.
            return {
                "factor_matrices_eigenvectors": matrix_eigendecomposition(
                    A=bias_corrected_factor_matrix,
                    eigendecomposition_config=self.amortized_computation_config,
                    # To estimate the eigenvalues based on the previous eigenvectors, we need to pass in the previous eigenvectors with the same dtype as the input matrix, i.e., factor_matrix.
                    eigenvectors_estimate=factor_matrix_eigenvectors.to(
                        dtype=bias_corrected_factor_matrix.dtype
                    ),
                    epsilon=self.epsilon,
                )[1].to(dtype=factor_matrix_eigenvectors.dtype)
            }, None
        except Exception as exception:
            return {
                "factor_matrices_eigenvectors": factor_matrix_eigenvectors
            }, exception

    def __post_init__(self) -> None:
        super().__post_init__()
        assert len(self.factor_matrices) == len(self.factor_matrices_eigenvectors)
        assert len(self.roots) == 1

    def _get_field_dict(self) -> dict[str, Any]:
        """
        Creates a dictionary containing shallow copies of this dataclass's fields, excluding specific fields.

        This method overrides the parent class's _get_field_dict method to exclude fields that don't
        align with the per-factor iteration pattern used in amortized computation:

        1. 'corrected_eigenvalues' is a single tensor that doesn't align with the per-factor iteration pattern
           since it represents eigenvalues across all dimensions rather than per-factor.
        2. 'roots' contains a single value for eigenvalue correction, unlike other fields which have
           one entry per factor matrix.

        Returns:
            dict[str, Any]: A dictionary mapping field names to their values, excluding
                'corrected_eigenvalues' and 'roots'.
        """
        return {
            key: value
            for key, value in super()._get_field_dict().items()
            if key not in ("corrected_eigenvalues", "roots")
        }


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
    ) -> None:
        super().__init__(block_list)

        # Initialize parameters.
        self._preconditioner_config = preconditioner_config
        self._beta2 = beta2
        self._epsilon = epsilon
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
                factor_matrix_dtype=self._preconditioner_config.factor_matrix_dtype,
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
                    amortized_computation_config=self._preconditioner_config.amortized_computation_config,
                    epsilon=self._epsilon,
                    num_tolerated_failed_amortized_computations=self._preconditioner_config.num_tolerated_failed_amortized_computations,
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
            for kronecker_factors_unwrapped in self._masked_kronecker_factors_unwrapped:
                kronecker_factors_unwrapped.amortized_computation(
                    bias_correction2=self._bias_correction2
                )

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
        self._local_preconditioned_dims_selector_list: tuple[tuple[bool, ...], ...] = (
            preconditioned_dims_selector_list
        )

        # Masked lists are the list of active preconditioners or values after filtering out gradients with None.
        self._masked_order_list: tuple[int, ...] = self._local_order_list
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
            * (
                get_dtype_size(self._preconditioner_config.factor_matrix_dtype)
                + get_dtype_size(block.dtype)
            )
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
