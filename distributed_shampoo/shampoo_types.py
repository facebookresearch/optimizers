"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import math
from collections.abc import Callable
from dataclasses import dataclass, field, make_dataclass
from inspect import signature

import torch
from distributed_shampoo.preconditioner.matrix_functions_types import (
    DefaultEigenConfig,
    DefaultEigendecompositionConfig,
    DefaultNewtonSchulzOrthogonalizationConfig,
    EigendecompositionConfig,
    MatrixFunctionConfig,
    OrthogonalizationConfig,
    PerturbationConfig,
    QREigendecompositionConfig,
    RootInvConfig,
)
from distributed_shampoo.utils.abstract_dataclass import AbstractDataclass
from distributed_shampoo.utils.load_balancing_utils import CostModel, DefaultCostModel
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parameter import Parameter

# Keys for optimizer state (always checkpointed)
FILTERED_GRAD = "filtered_grad"
LR_SUM = "lr_sum"
MOMENTUM_BUFFER = "momentum_buffer"
STEP = "step"
TRAIN_MODE = "train_mode"
WEIGHT_BUFFER = "weight_buffer"

# Keys for parameter groups (checkpointed if specified)
BETA3 = "beta3"
BETAS = "betas"
DISTRIBUTED_CONFIG = "distributed_config"
EPSILON = "epsilon"
GRAFTING_CONFIG = "grafting_config"
ITERATE_AVERAGING_CONFIG = "iterate_averaging_config"
LR = "lr"
MAX_PRECONDITIONER_DIM = "max_preconditioner_dim"
PARAMS = "params"  # While this is stored in groups by default, we do not checkpoint this quantity.
PEAK_LR = "peak_lr"
PRECONDITION_FREQUENCY = "precondition_frequency"
PRECONDITIONER_CONFIG = "preconditioner_config"
START_PRECONDITIONING_STEP = "start_preconditioning_step"
USE_BIAS_CORRECTION = "use_bias_correction"
USE_PIN_MEMORY = "use_pin_memory"
WEIGHT_DECAY = "weight_decay"
WEIGHT_DECAY_TYPE = "weight_decay_type"

# Keys for lists of blocked states and metadata (never checkpointed)
CUDA_STREAM = "cuda_stream"
DISTRIBUTOR = "distributor"
# Per-param owner-rank map (never checkpointed; recomputed deterministically in
# split_param_groups on every __init__). Used by the lossless distributors under
# the ROUND_ROBIN assignment strategy.
OWNER_RANKS = "owner_ranks"
EVAL_INTERP_COEFF = "eval_interp_coeff"
FILTERED_GRAD_LIST = "filtered_grad_list"
GRAFTING_PRECONDITIONER_LIST = "grafting_preconditioner_list"
LR_CPU_PINNED = "lr_cpu_pinned"
LR_TENSOR = "lr_tensor"
MASKED_BLOCKED_GRADS = "masked_blocked_grads"
MASKED_BLOCKED_PARAMS = "masked_blocked_params"
MASKED_FILTERED_GRAD_LIST = "masked_filtered_grad_list"
MASKED_MOMENTUM_BUFFER_LIST = "masked_momentum_buffer_list"
MASKED_WEIGHT_BUFFER_LIST = "masked_weight_buffer_list"
MOMENTUM_BUFFER_LIST = "momentum_buffer_list"
PREVIOUS_GRAD_SELECTOR = "previous_grad_selector"
SHAMPOO_PRECONDITIONER_LIST = "shampoo_preconditioner_list"
TRAIN_INTERP_COEFF = "train_interp_coeff"
WEIGHT_BUFFER_LIST = "weight_buffer_list"


###### ERROR CLASSES ######
class PreconditionerValueError(ValueError):
    """ValueError for invalid values encountered during Preconditioner computation."""


###### DATACLASSES ######
@dataclass(init=False)
class PreconditionerConfig(AbstractDataclass):
    """Configuration for preconditioner computation in DistributedShampoo."""


@dataclass
class SGDPreconditionerConfig(PreconditionerConfig):
    """Configuration for SGD preconditioner computation."""


@dataclass(kw_only=True)
class AdaGradPreconditionerConfig(PreconditionerConfig):
    """Configuration for AdaGrad preconditioner computation.

    Attributes:
        epsilon (float): Epsilon term for regularizing square-root of the aggregated second moment to ensure positive definiteness.
            (Default: 1e-10)

    """

    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        if not self.epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. Must be > 0.0.")


@dataclass(kw_only=True)
class RMSpropPreconditionerConfig(AdaGradPreconditionerConfig):
    """Configuration for RMSprop preconditioner computation.

    Attributes:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.99)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-10)
        drop_weighting_factor_on_gsquare (bool): drop the (1 - beta2) weighting factor when computing the updates of the
            preconditioners, i.e. V(t) = beta2 * V(t-1) + G^2. This also disables bias correction (no beta2 bias correction). (default: False)

    """

    beta2: float = 0.99
    drop_weighting_factor_on_gsquare: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.beta2 <= 1.0:
            raise ValueError(
                f"Invalid grafting beta2 parameter: {self.beta2}. Must be in (0.0, 1.0]."
            )


@dataclass(kw_only=True)
class AdamPreconditionerConfig(RMSpropPreconditionerConfig):
    """Configuration for Adam preconditioner computation.

    Attributes:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-10)

    Note:
        The traditional beta1 parameter in Adam is set by betas[0] in DistributedShampoo's hyperparameters.
    """

    beta2: float = 0.999


@dataclass(init=False)
class BaseShampooPreconditionerConfig(PreconditionerConfig):
    """Configuration for amortized preconditioner computation in DistributedShampoo.

    Attributes:
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized computation, e.g., inverse-root computation or eigendecomposition.
        drop_weighting_factor_on_gsquare (bool): If True, drop the (1 - beta2) weighting factor when computing
            the updates of the preconditioners, i.e., V(t) = beta2 * V(t-1) + G^2 instead of
            V(t) = beta2 * V(t-1) + (1 - beta2) * G^2. This keeps _bias_correction2 at 1.0 (no beta2 bias correction). (Default: False)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        use_symmetric_packing (bool): If True, stores symmetric Kronecker factor matrices
            (factor_matrices and inv_factor_matrices) in packed upper triangular format
            (d*(d+1)/2 elements instead of d*d), saving ~50% memory. This is lossless since
            these matrices are symmetric PSD. (Default: True)
        use_trace_scaling (bool): Flag for whether to normalize the factor matrix by its trace raised to trace_scaling_exponent before computing the inverse root.
            Credit to https://arxiv.org/pdf/2506.03595. (Default: False)
        trace_scaling_exponent (float): Exponent applied to the trace when use_trace_scaling is True; the factor matrix is multiplied by trace ** (-trace_scaling_exponent).
            Must be in (0.0, 1.0]. The default 0.5 reproduces 1 / sqrt(trace). (Default: 0.5)

    """

    # repr=False prevents __repr__() from accessing this field to avoid linter complaints
    amortized_computation_config: MatrixFunctionConfig = field(repr=False)
    drop_weighting_factor_on_gsquare: bool = False
    factor_matrix_dtype: torch.dtype = torch.float32
    num_tolerated_failed_amortized_computations: int = 3
    use_symmetric_packing: bool = True
    use_trace_scaling: bool = False
    trace_scaling_exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.num_tolerated_failed_amortized_computations < 0:
            raise ValueError(
                f"Invalid num_tolerated_failed_amortized_computations value: {self.num_tolerated_failed_amortized_computations}. Must be >= 0."
            )
        if self.use_trace_scaling and not 0.0 < self.trace_scaling_exponent <= 1.0:
            raise ValueError(
                f"Invalid trace_scaling_exponent value: {self.trace_scaling_exponent}. Must be in (0.0, 1.0] when use_trace_scaling=True."
            )


@dataclass(init=False)
class ClassicShampooPreconditionerConfig(BaseShampooPreconditionerConfig):
    """Configuration for Shampoo preconditioner computation.

    Attributes:
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized computation, e.g., inverse-root computation or eigendecomposition.
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        inverse_exponent_override (dict[int, dict[int, float] | float]): The inverse_exponent_override attribute is a dictionary that allows for customizing the inverse exponent used in the Shampoo preconditioner computation.
            The keys of the dictionary represent the order of the tensor, and the values are either dictionaries with dimension indices as keys and override values as values, or a single float value for all dimensions. All unspecified dimensions use a default exponent of 1/(2*max(o,1)), where o is the order of the tensor. (Default: {})

            As an example, suppose inverse_exponent_override={2: 0.2, 3: {0: 0.0, 1: 0.25}}. In this case, all 1-D tensors will use the default exponent of 0.5 for preconditioning the first (and only) dimension. All 2-D tensors will be preconditioned with an exponent of 0.2 on all dimensions. All 3-D tensors will have the first dimension be preconditioned with an exponent of 0.5, the second dimension not preconditioned, and the third dimension preconditioned with the default exponent 0.1667.
            A visualization of this example can be seen below:
            1-D:
                            +-------x-------+
                                    |
                                    |
                            (^0.5), the default inverse exponent 1/(2*1) since inverse_exponent_override[1] is not specified
            2-D:
                            +-----------+
                            |           |
                            |           |
                            |           |-----(^0.2), as specified by inverse_exponent_override[2]=0.2
                            |           |
                            |           |
                            +-----------+
                                  |
                                  |
                                (^0.2), as specified by inverse_exponent_override[2]=0.2
            3-D:
                               +---------------+
                              /               /|
                             /               / |
                            +---------------+  |
                            |               |  |
                            |               | -|---(^0.25), as specified by inverse_exponent_override[3][1]=0.25
                            |               |  +
                            |               | /
                            |               |/\
                            +---------------+  \
                                    |          (^0.1667), the default inverse exponent 1/(2*3) since inverse_exponent_override[3][2] is not specified
                                    |
                            no preconditioning since inverse_exponent_override[3][0]=0.0


    """

    inverse_exponent_override: dict[int, dict[int, float] | float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        if non_positive_orders := [
            order for order in self.inverse_exponent_override.keys() if order < 0
        ]:
            raise ValueError(
                f"Invalid orders in {self.inverse_exponent_override=}: {non_positive_orders}. All orders must be >= 0."
            )

        for (
            order,
            dim_override_or_universal_override,
        ) in self.inverse_exponent_override.items():
            if isinstance(dim_override_or_universal_override, dict):
                if illegal_dimensions := [
                    dim
                    for dim in dim_override_or_universal_override
                    if not (0 <= dim <= max(order - 1, 0))
                ]:
                    raise ValueError(
                        f"Invalid dimensions in self.inverse_exponent_override[{order}]={self.inverse_exponent_override[order]}: {illegal_dimensions}. All dimensions must be within [0, {max(order - 1, 0)}]."
                    )
                if non_positive_overrides := [
                    override
                    for override in dim_override_or_universal_override.values()
                    if override < 0
                ]:
                    raise ValueError(
                        f"Invalid override value in self.inverse_exponent_override[{order}]={self.inverse_exponent_override[order]}: {non_positive_overrides}. All overrides must be >= 0."
                    )
            else:
                assert isinstance(dim_override_or_universal_override, float)
                if dim_override_or_universal_override < 0:
                    raise ValueError(
                        f"Invalid override value in self.inverse_exponent_override[{order}]={self.inverse_exponent_override[order]}: {dim_override_or_universal_override}. All overrides must be >= 0."
                    )


@dataclass(kw_only=True)
class RootInvShampooPreconditionerConfig(ClassicShampooPreconditionerConfig):
    """Configuration for Shampoo preconditioner computation with caching of the root inverse factor matrices.

    Note: When using custom amortized_computation_config, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        amortized_computation_config (RootInvConfig): Configuration for the inverse-root computation. (Default: DefaultEigenConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        inverse_exponent_override (dict[int, dict[int, float] | float]): The inverse_exponent_override attribute is a dictionary that allows for customizing the inverse exponent used in the Shampoo preconditioner computation.
            The keys of the dictionary represent the order of the tensor, and the values are either dictionaries with dimension indices as keys and override values as values, or a single float value for all dimensions. All unspecified dimensions use a default exponent of 1/(2*max(o,1)), where o is the order of the tensor. (Default: {})

            As an example, suppose inverse_exponent_override={2: 0.2, 3: {0: 0.0, 1: 0.25}}. In this case, all 1-D tensors will use the default exponent of 0.5 for preconditioning the first (and only) dimension. All 2-D tensors will be preconditioned with an exponent of 0.2 on all dimensions. All 3-D tensors will have the first dimension be preconditioned with an exponent of 0.5, the second dimension not preconditioned, and the third dimension preconditioned with the default exponent 0.1667.
            A visualization of this example can be seen below:
            1-D:
                            +-------x-------+
                                    |
                                    |
                            (^0.5), the default inverse exponent 1/(2*1) since inverse_exponent_override[1] is not specified
            2-D:
                            +-----------+
                            |           |
                            |           |
                            |           |-----(^0.2), as specified by inverse_exponent_override[2]=0.2
                            |           |
                            |           |
                            +-----------+
                                  |
                                  |
                                (^0.2), as specified by inverse_exponent_override[2]=0.2
            3-D:
                               +---------------+
                              /               /|
                             /               / |
                            +---------------+  |
                            |               |  |
                            |               | -|---(^0.25), as specified by inverse_exponent_override[3][1]=0.25
                            |               |  +
                            |               | /
                            |               |/\
                            +---------------+  \
                                    |          (^0.1667), the default inverse exponent 1/(2*3) since inverse_exponent_override[3][2] is not specified
                                    |
                            no preconditioning since inverse_exponent_override[3][0]=0.0
        inv_factor_matrix_dtype (torch.dtype): Data type for inverse factor matrix. (Default: torch.float32)


    """

    @staticmethod
    def _get_default_amortized_computation_config() -> RootInvConfig:
        return DefaultEigenConfig

    # pyrefly: ignore [bad-override-mutable-attribute]
    amortized_computation_config: RootInvConfig = field(
        default_factory=_get_default_amortized_computation_config
    )
    inv_factor_matrix_dtype: torch.dtype = torch.float32


DefaultShampooConfig = RootInvShampooPreconditionerConfig()


@dataclass(kw_only=True)
class EigendecomposedShampooPreconditionerConfig(ClassicShampooPreconditionerConfig):
    """Configuration for Shampoo preconditioner computation with caching of the eigendecomposed factor matrices.

    Note: When using custom amortized_computation_config, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        amortized_computation_config (EigendecompositionConfig): Configuration for the eigendecomposition computation. (Default: DefaultEigendecompositionConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        inverse_exponent_override (dict[int, dict[int, float] | float]): The inverse_exponent_override attribute is a dictionary that allows for customizing the inverse exponent used in the Shampoo preconditioner computation.
            The keys of the dictionary represent the order of the tensor, and the values are either dictionaries with dimension indices as keys and override values as values, or a single float value for all dimensions. All unspecified dimensions use a default exponent of 1/(2*max(o,1)), where o is the order of the tensor. (Default: {})

            As an example, suppose inverse_exponent_override={2: 0.2, 3: {0: 0.0, 1: 0.25}}. In this case, all 1-D tensors will use the default exponent of 0.5 for preconditioning the first (and only) dimension. All 2-D tensors will be preconditioned with an exponent of 0.2 on all dimensions. All 3-D tensors will have the first dimension be preconditioned with an exponent of 0.5, the second dimension not preconditioned, and the third dimension preconditioned with the default exponent 0.1667.
            A visualization of this example can be seen below:
            1-D:
                            +-------x-------+
                                    |
                                    |
                            (^0.5), the default inverse exponent 1/(2*1) since inverse_exponent_override[1] is not specified
            2-D:
                            +-----------+
                            |           |
                            |           |
                            |           |-----(^0.2), as specified by inverse_exponent_override[2]=0.2
                            |           |
                            |           |
                            +-----------+
                                  |
                                  |
                                (^0.2), as specified by inverse_exponent_override[2]=0.2
            3-D:
                               +---------------+
                              /               /|
                             /               / |
                            +---------------+  |
                            |               |  |
                            |               | -|---(^0.25), as specified by inverse_exponent_override[3][1]=0.25
                            |               |  +
                            |               | /
                            |               |/\
                            +---------------+  \
                                    |          (^0.1667), the default inverse exponent 1/(2*3) since inverse_exponent_override[3][2] is not specified
                                    |
                            no preconditioning since inverse_exponent_override[3][0]=0.0
        factor_matrix_eigenvectors_dtype (torch.dtype): Data type for factor matrix eigenvectors. (Default: torch.float32)
        factor_matrix_eigenvalues_dtype (torch.dtype): Data type for factor matrix eigenvalues. (Default: torch.float32)


    """

    @staticmethod
    def _get_default_amortized_computation_config() -> EigendecompositionConfig:
        return DefaultEigendecompositionConfig

    # pyrefly: ignore [bad-override-mutable-attribute]
    amortized_computation_config: EigendecompositionConfig = field(
        default_factory=_get_default_amortized_computation_config
    )
    factor_matrix_eigenvectors_dtype: torch.dtype = torch.float32
    factor_matrix_eigenvalues_dtype: torch.dtype = torch.float32


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooPreconditionerConfig(BaseShampooPreconditionerConfig):
    """Configuration for eigenvalue-corrected Shampoo/SOAP preconditioner computation.

    Recall that in eigenvalue-corrected Shampoo, the eigenvectors and eigenvalues of the factor matrices are computed separately and stored in place of the full inverted preconditioner, as opposed to the single inverse-root computation of the factor matrices in Shampoo.
    In eigenvalue-corrected Shampoo, the eigenvectors are updated periodically like the inverted preconditioners in Shampoo, but the eigenvalues are updated every iteration.

    Note: When using custom amortized_computation_config, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        amortized_computation_config (EigendecompositionConfig): Configuration for the eigenvector computation.
            (Default: DefaultEigendecompositionConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        ignored_basis_change_dims (dict[int, list[int]]): The ignored_basis_change_dims attribute is a dictionary that specifies the dimensions of the gradient to ignore when transforming the basis of the gradient using the corresponding factor matrix's eigenvectors.
            (This is analogous to turning off preconditioning for the specified dimensions in default Shampoo.)
            The keys of the dictionary represent the order of the tensor, and the values are lists of dimension indices to ignore. (Default: {})

            Below is a visualized example of how ignored_basis_change_dims is applied on 1-D, 2-D, and 3-D tensors when given ignored_basis_change_dims={1: [0], 2: [1], 3: [0, 2]}:
            1-D:
                            +-------x-------+
                                    |
                                    |
                             no change basis, as specified by 0 in ignored_basis_change_dims[1]
            2-D:
                            +-----------+
                            |           |
                            |           |
                            |           |-----no change basis, as specified by 1 in ignored_basis_change_dims[2]
                            |           |
                            |           |
                            +-----------+
            3-D:
                               +---------------+
                              /               /|
                             /               / |
                            +---------------+  |
                            |               |  |
                            |               |  |
                            |               |  +
                            |               | /
                            |               |/\
                            +---------------+  \
                                    |        no change basis, as specified by 2 in ignored_basis_change_dims[3]
                                    |
                             no change basis, as specified by 0 in ignored_basis_change_dims[3]

        inverse_exponent_override (dict[int, float]): The inverse_exponent_override attribute is a dictionary that allows for customizing the inverse exponent used in eigenvalue correction.
            The keys of the dictionary represent the order of the tensor, and the values are the exponent override values. For example, if we want to use a custom inverse exponent for 3-D tensors, we can set inverse_exponent_override as inverse_exponent_override={3: 0.25}.
            Note that the inverse_exponent_override dictionary can contain multiple entries for different tensor orders. If the order of the tensor is not specified in the dictionary, the default exponent, 1/2, will be used. (Default: {})
        factor_matrix_eigenvectors_dtype (torch.dtype): Data type for factor matrix eigenvectors. (Default: torch.float32)
        corrected_eigenvalues_dtype (torch.dtype): Data type for corrected eigenvalues. (Default: torch.float32)

    """

    @staticmethod
    def _get_default_amortized_computation_config() -> EigendecompositionConfig:
        return DefaultEigendecompositionConfig

    # pyrefly: ignore [bad-override-mutable-attribute]
    amortized_computation_config: EigendecompositionConfig = field(
        default_factory=_get_default_amortized_computation_config
    )
    ignored_basis_change_dims: dict[int, list[int]] = field(default_factory=dict)
    inverse_exponent_override: dict[int, float] = field(default_factory=dict)
    factor_matrix_eigenvectors_dtype: torch.dtype = torch.float32
    corrected_eigenvalues_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(
            self.amortized_computation_config.rank_deficient_stability_config,
            PerturbationConfig,
        ):
            raise ValueError(
                f"{type(self.amortized_computation_config.rank_deficient_stability_config).__name__} is an invalid rank_deficient_stability_config for {type(self).__name__}."
                f" Please use an instance of {PerturbationConfig.__name__} instead."
            )

        if non_positive_orders := [
            order for order in self.ignored_basis_change_dims.keys() if order < 0
        ]:
            raise ValueError(
                f"Invalid orders in {self.ignored_basis_change_dims=}: {non_positive_orders}. All orders must be >= 0."
            )

        for (
            order,
            ignored_basis_change_dims_in_one_order,
        ) in self.ignored_basis_change_dims.items():
            if illegal_ignored_dimensions := [
                dim
                for dim in ignored_basis_change_dims_in_one_order
                if not (0 <= dim <= max(order - 1, 0))
            ]:
                raise ValueError(
                    f"Invalid dimensions in {self.ignored_basis_change_dims[order]=}: {illegal_ignored_dimensions}. All dimensions must be within [0, {max(order - 1, 0)}]."
                )
            if len(ignored_basis_change_dims_in_one_order) != len(
                set(ignored_basis_change_dims_in_one_order)
            ):
                raise ValueError(
                    f"Invalid ignored dimensions in {self.ignored_basis_change_dims[order]=}. Duplicate dimensions found in {ignored_basis_change_dims_in_one_order}. All dimensions must be unique."
                )

        if non_positive_orders := [
            order for order in self.inverse_exponent_override.keys() if order < 0
        ]:
            raise ValueError(
                f"Invalid orders in {self.inverse_exponent_override=}: {non_positive_orders}. All orders must be >= 0."
            )

        for order, override in self.inverse_exponent_override.items():
            if override <= 0:
                raise ValueError(
                    f"Invalid override value in {self.inverse_exponent_override[order]=}: {override}. All overrides must be > 0."
                )


DefaultEigenvalueCorrectedShampooConfig = (
    EigenvalueCorrectedShampooPreconditionerConfig()
)
DefaultSOAPConfig = EigenvalueCorrectedShampooPreconditionerConfig(
    amortized_computation_config=QREigendecompositionConfig(),
)


@dataclass(kw_only=True)
class RootInvKLShampooPreconditionerConfig(RootInvShampooPreconditionerConfig):
    """Configuration for KL-Shampoo preconditioner computation with caching of the root inverse factor matrices.

    Attributes:
        amortized_computation_config (RootInvConfig): Configuration for the inverse-root computation. (Default: DefaultEigenConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        inverse_exponent_override (dict[int, dict[int, float] | float]): The inverse_exponent_override attribute is a dictionary that allows for customizing the inverse exponent used in the KL-Shampoo preconditioner computation.
            The keys of the dictionary represent the order of the tensor, and the values are either dictionaries with dimension indices as keys and override values as values, or a single float value for all dimensions. All unspecified dimensions use a default exponent of 1/(2*max(o,1)), where o is the order of the tensor. (Default: {})

            As an example, suppose inverse_exponent_override={2: 0.2, 3: {0: 0.0, 1: 0.25}}. In this case, all 1-D tensors will use the default exponent of 0.5 for preconditioning the first (and only) dimension. All 2-D tensors will be preconditioned with an exponent of 0.2 on all dimensions. All 3-D tensors will have the first dimension be preconditioned with an exponent of 0.5, the second dimension not preconditioned, and the third dimension preconditioned with the default exponent 0.1667.
            A visualization of this example can be seen below:
            1-D:
                            +-------x-------+
                                    |
                                    |
                            (^0.5), the default inverse exponent 1/(2*1) since inverse_exponent_override[1] is not specified
            2-D:
                            +-----------+
                            |           |
                            |           |
                            |           |-----(^0.2), as specified by inverse_exponent_override[2]=0.2
                            |           |
                            |           |
                            +-----------+
                                  |
                                  |
                                (^0.2), as specified by inverse_exponent_override[2]=0.2
            3-D:
                               +---------------+
                              /               /|
                             /               / |
                            +---------------+  |
                            |               |  |
                            |               | -|---(^0.25), as specified by inverse_exponent_override[3][1]=0.25
                            |               |  +
                            |               | /
                            |               |/\
                            +---------------+  \
                                    |          (^0.1667), the default inverse exponent 1/(2*3) since inverse_exponent_override[3][2] is not specified
                                    |
                            no preconditioning since inverse_exponent_override[3][0]=0.0


    """


@dataclass(kw_only=True)
class EigendecomposedKLShampooPreconditionerConfig(
    EigendecomposedShampooPreconditionerConfig
):
    """Configuration for KL-Shampoo preconditioner computation with caching of the eigendecomposed factor matrices.

    Attributes:
        amortized_computation_config (EigendecompositionConfig): Configuration for the eigendecomposition computation. (Default: DefaultEigendecompositionConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)
        inverse_exponent_override (dict[int, dict[int, float] | float]): The inverse_exponent_override attribute is a dictionary that allows for customizing the inverse exponent used in the KL-Shampoo preconditioner computation.
            The keys of the dictionary represent the order of the tensor, and the values are either dictionaries with dimension indices as keys and override values as values, or a single float value for all dimensions. All unspecified dimensions use a default exponent of 1/(2*max(o,1)), where o is the order of the tensor. (Default: {})

            As an example, suppose inverse_exponent_override={2: 0.2, 3: {0: 0.0, 1: 0.25}}. In this case, all 1-D tensors will use the default exponent of 0.5 for preconditioning the first (and only) dimension. All 2-D tensors will be preconditioned with an exponent of 0.2 on all dimensions. All 3-D tensors will have the first dimension be preconditioned with an exponent of 0.5, the second dimension not preconditioned, and the third dimension preconditioned with the default exponent 0.1667.
            A visualization of this example can be seen below:
            1-D:
                            +-------x-------+
                                    |
                                    |
                            (^0.5), the default inverse exponent 1/(2*1) since inverse_exponent_override[1] is not specified
            2-D:
                            +-----------+
                            |           |
                            |           |
                            |           |-----(^0.2), as specified by inverse_exponent_override[2]=0.2
                            |           |
                            |           |
                            +-----------+
                                  |
                                  |
                                (^0.2), as specified by inverse_exponent_override[2]=0.2
            3-D:
                               +---------------+
                              /               /|
                             /               / |
                            +---------------+  |
                            |               |  |
                            |               | -|---(^0.25), as specified by inverse_exponent_override[3][1]=0.25
                            |               |  +
                            |               | /
                            |               |/\
                            +---------------+  \
                                    |          (^0.1667), the default inverse exponent 1/(2*3) since inverse_exponent_override[3][2] is not specified
                                    |
                            no preconditioning since inverse_exponent_override[3][0]=0.0


    """


@dataclass(kw_only=True)
class SpectralDescentPreconditionerConfig(PreconditionerConfig):
    """Configuration for spectral descent computation in DistributedShampoo.

    NOTE: This config can only be used for 2D parameters, or parameters that have been reshaped to 2D.
    Which parameters are reshaped to 2D is determined by the max_preconditioner_dim argument in DistributedShampoo.
    If all >2D parameters should be guaranteed to be reshaped to 2D, then max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2 has to be used.


    Note: When using custom orthogonalization config, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        orthogonalization_config (OrthogonalizationConfig): Configuration for orthogonalization of the search direction.
            (Default: DefaultNewtonSchulzOrthogonalizationConfig)

    """

    @staticmethod
    def _default_orthogonalization_config() -> OrthogonalizationConfig:
        return DefaultNewtonSchulzOrthogonalizationConfig

    orthogonalization_config: OrthogonalizationConfig = field(
        default_factory=_default_orthogonalization_config
    )


DefaultSpectralDescentPreconditionerConfig = SpectralDescentPreconditionerConfig()


@dataclass(kw_only=True)
class SignDescentPreconditionerConfig(PreconditionerConfig):
    """Configuration for sign descent in DistributedShampoo.

    Note: When using custom scale functions, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        scale_fn (Callable[[Tensor], float | Tensor]): Function to scale the sign update based on the gradient tensor.
            (Default: _default_scale_fn)

    Example:
        - For scale_fn to implement steepest descent under L-infinity norm:
        def l_infinity_scale_fn(grad: Tensor) -> float:
            return grad.abs().sum()

        scale_fn = l_infinity_scale_fn

    """

    @staticmethod
    def _default_scale_fn(grad: Tensor) -> float:
        """Default scale function that returns 1.0. Implemented as staticmethod for pickleable concerns."""
        return 1.0

    scale_fn: Callable[[Tensor], float | Tensor] = _default_scale_fn


DefaultSignDescentPreconditionerConfig = SignDescentPreconditionerConfig()


@dataclass(init=False)
class IterateAveragingConfig(AbstractDataclass):
    """Configuration for primal or iterate averaging in Shampoo.

    Iterate averaging methods like Generalized Primal Averaging (GPA), Schedule-Free,
    and ClassicMomentumConfig provide different approaches to incorporating momentum
    into the optimizer.

    Learning Rate Adjustment for Momentum -> GPA:
        In SGD heavy-ball momentum (dampening=0), the momentum buffer accumulates
        gradients and amplifies the effective step size by 1/(1-β) at steady state:

            M ← β * M + grad
            W ← W - lr * M

        At steady state with constant gradients, M ≈ grad / (1-β), so the effective
        step size is lr / (1-β).

        In GPA, the gradient step is scaled by (1 - μ_x * μ_y) without any
        accumulation-based amplification:

            search_dir ∝ (1 - μ_x * μ_y) * (-lr * preconditioned_grad) + ...

        For heavy-ball (μ_x=β, μ_y=1.0), this factor is (1-β), so the effective
        step size is lr * (1-β) — a factor of 1/(1-β)^2 smaller than heavy-ball.
        To compensate, set lr_new = lr_old / (1-β).

        ClassicMomentumConfig does NOT require this adjustment because it uses the
        same momentum buffer accumulation as SGD.

        These relationships are validated by `test_gpa_vs_sgd_momentum` and
        `test_classic_momentum_vs_sgd_momentum` in
        `dev/gpu_tests/iterate_averaging_test.py`.

    Concrete Example:
        An optimizer config with `lr=0.04, momentum=0.5` (heavy-ball) can be implemented
        two ways:

        1. ClassicMomentumConfig (no lr change):
            lr = 0.04
            iterate_averaging_config = ClassicMomentumConfig(
                momentum=0.5,
            )

        2. GPA (requires lr change):
            lr = 0.08  # = 0.04 / (1 - 0.5)
            iterate_averaging_config = GeneralizedPrimalAveragingConfig(
                eval_interp_coeff=0.5,   # = momentum
                train_interp_coeff=1.0,  # heavy-ball (non-Nesterov)
            )

    """


@dataclass(kw_only=True)
class ClassicMomentumConfig(IterateAveragingConfig):
    """Configuration for classic SGD-style momentum in Shampoo.

    This configures and enables the momentum/dampening/Nesterov behavior as traditional
    SGD momentum applied to the preconditioned search direction, without requiring
    train/eval mode switching (unlike GPA or Schedule-Free).

    The momentum update applied to the search direction P is:
        M <- momentum * M + (1 - dampening) * P
        P <- (1 - dampening) * P + momentum * M     if use_nesterov
        P <- M                                       otherwise

    Then the parameter update is:
        W <- W - lr * P

    Attributes:
        momentum (float): Momentum factor. Must be in (0.0, 1.0). (Default: 0.5)
            Note: momentum=0.0 is a no-op and is rejected; pass
            `iterate_averaging_config=None` instead to disable iterate averaging.
        dampening (float): Dampening for momentum. Must be in [0.0, 1.0). (Default: 0.0)
        use_nesterov (bool): Enables Nesterov momentum. (Default: False)

    """

    momentum: float = 0.5
    dampening: float = 0.0
    use_nesterov: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.momentum < 1.0:
            raise ValueError(f"Invalid {self.momentum=}. Must be within (0.0, 1.0).")
        if not 0.0 <= self.dampening < 1.0:
            raise ValueError(f"Invalid {self.dampening=}. Must be within [0.0, 1.0).")


@dataclass(kw_only=True)
class GeneralizedPrimalAveragingConfig(IterateAveragingConfig):
    """Configuration for generalized primal averaging in Shampoo.

    Generalized Primal Averaging (GPA) maintains two sequences of iterates:
    - The evaluation sequence (x): Used for model evaluation/inference
    - The training sequence (y): Used for gradient computation

    See https://arxiv.org/pdf/2512.17131 for more details.

    Equivalence to SGD Momentum:
        GPA can reproduce SGD momentum behavior with appropriate coefficient settings:

        Example 1: SGD with Heavy-Ball Momentum (momentum=0.9)
            ```python
            # Original SGD:
            # optimizer = SGD(params, lr=0.01, momentum=0.9)

            # Equivalent Shampoo with GPA:
            optimizer = DistributedShampoo(
                params,
                lr=0.1,  # = 0.01 / (1 - 0.9)
                betas=(0.0, 1.0),
                preconditioner_config=SGDPreconditionerConfig(),
                iterate_averaging_config=GeneralizedPrimalAveragingConfig(
                    eval_interp_coeff=0.9,  # = momentum
                    train_interp_coeff=1.0,
                ),
            )
            ```

        Example 2: SGD with Nesterov Momentum (momentum=0.9, nesterov=True)
            ```python
            # Original SGD:
            # optimizer = SGD(params, lr=0.01, momentum=0.9, nesterov=True)

            # Equivalent Shampoo with GPA:
            optimizer = DistributedShampoo(
                params,
                lr=0.1,  # = 0.01 / (1 - 0.9)
                betas=(0.0, 1.0),
                preconditioner_config=SGDPreconditionerConfig(),
                iterate_averaging_config=GeneralizedPrimalAveragingConfig(
                    eval_interp_coeff=0.9,  # = momentum
                    train_interp_coeff=0.9,  # = momentum (same as eval for Nesterov)
                ),
            )
            ```

    Attributes:
        eval_interp_coeff (float): Interpolation coefficient for the model evaluation sequence (called mu_x).
            Controls the momentum-like behavior of the evaluation iterates.
            Set to the momentum value (β) for SGD momentum equivalence.
            Must be in the interval [0, 1). (Default: 0.9)
        train_interp_coeff (float): Interpolation coefficient for the gradient computation sequence (called mu_y).
            Set to 1.0 for heavy-ball momentum, or to β for Nesterov momentum.
            Must be in the interval (0, 1]. (Default: 0.8)

    """

    eval_interp_coeff: float = 0.9
    train_interp_coeff: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 <= self.eval_interp_coeff < 1.0:
            raise ValueError(
                f"Invalid {self.eval_interp_coeff=}. Must be within [0.0, 1.0)."
            )
        if not 0.0 < self.train_interp_coeff <= 1.0:
            raise ValueError(
                f"Invalid {self.train_interp_coeff=}. Must be within (0.0, 1.0]."
            )


@dataclass(kw_only=True)
class ScheduleFreeConfig(IterateAveragingConfig):
    """Configuration for schedule-free optimization in Shampoo.

    Schedule-Free is an iterate averaging method that eliminates the need for learning rate
    schedules by automatically adapting the effective learning rate during training.

    See https://arxiv.org/abs/2405.15682 for more details.

    Note:
        Schedule-Free is not designed to replicate traditional momentum behavior.
        If you need momentum-equivalent behavior, use GeneralizedPrimalAveragingConfig instead.
        Schedule-Free is intended for cases where you want to avoid manually tuning
        learning rate schedules.

    Attributes:
        train_interp_coeff (float): Interpolation coefficient for the gradient computation sequence (called mu_y).
            Controls the interpolation between the current iterate and the averaged iterate.
            Must be in the interval (0, 1]. (Default: 0.8)
        eval_coeff_lr_power (int): Learning rate power for the evaluation sequence. This heuristic is described
            in Equation (23) in https://arxiv.org/pdf/2405.15682. (Default: 2)

    """

    train_interp_coeff: float = 0.8
    eval_coeff_lr_power: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.train_interp_coeff <= 1.0:
            raise ValueError(
                f"Invalid {self.train_interp_coeff=}. Must be within (0.0, 1.0]."
            )


@enum.unique
class WeightDecayType(enum.Enum):
    """Weight decay strategies for Shampoo.

    L2: Applies weight decay by adding a multiple of the weights to the gradient before preconditioning.

    DECOUPLED: Applies weight decay by adding a multiple of the weights independent of the preconditioned gradient.

    CORRECTED: Applies weight decay by adding a multiple of the weights scaled by the learning rate divided by the maximum learning rate,
        independent of the preconditioned gradient. This was shown to yield nice stability properties for Adam, i.e., a steady-state
        ||g|| / ||w|| ratio for normalized layers in Defazio (2025). This needs to be further studied for AdaGrad-style methods.

    INDEPENDENT: Applies weight decay by adding a multiple of the weights divided by the maximum learning rate,
        independent of the preconditioned gradient. This is a variant of decoupled weight decay that scales by 1 / peak_lr.

    """

    L2 = "L2"
    DECOUPLED = "DECOUPLED"
    CORRECTED = "CORRECTED"
    INDEPENDENT = "INDEPENDENT"


@dataclass
class FSDPParameterMetadata:
    """FSDP Metadata for a parameter.

    Attributes:
        fqn (str): Fully qualified name of the parameter.
        shape (torch.Size): Shape of the parameter.
        numel (int): Number of elements in the parameter.
        start_idx (int): Start index of the local shard in the flattened parameter (inclusive).
        end_idx (int): End index of the local shard in the flattened parameter (exclusive).
        sharding_strategy (ShardingStrategy): Sharding strategy for the parameter.

    """

    fqn: str
    shape: torch.Size
    numel: int
    start_idx: int
    end_idx: int
    sharding_strategy: ShardingStrategy


@enum.unique
class FSDPParamAssignmentStrategy(enum.Enum):
    """Parameter assignment strategy for FSDP2, determining how the parameters are assigned to ranks in shard dimension.

    DEFAULT: By default, parameters are assigned to ranks according to their FSDP shard.
        Shampoo further blocks the parameters for preconditioning, which would likely affect the numerical accuracy
        of the preconditioner. However, this strategy is the most efficient in terms of memory usage.

    REPLICATE: Parameters are all-gathered (replicated) across all ranks in shard dimension.
        This strategy should produce identical results as the default Shampoo implementation, but at the cost of
        significantly increased memory usage and communication.

    ROUND_ROBIN: Parameters are assigned to whole ranks in the shard dimension so each rank owns a subset
        of the full model parameters (use when there are more parameters than shards). Assignment is
        load-balanced: a global owner map is computed by greedy longest-processing-time (LPT) bin-packing on
        per-parameter cost (aligned bytes by default, via ``load_balancing_config``'s cost model), so the
        total gather/update byte load per owner rank is balanced. The map is computed GLOBALLY over the full
        parameter set (before any ``num_sub_groups`` split) so ranks are balanced across the whole model
        rather than per sub-group. Deterministic, hence checkpoint-stable. (The classic positional
        ``i % shard`` assignment is just LPT with a uniform cost model; the default byte cost model makes it
        size-balanced -- there is no separate strategy for it.)

    """

    DEFAULT = enum.auto()
    REPLICATE = enum.auto()
    ROUND_ROBIN = enum.auto()


@dataclass(kw_only=True)
class LoadBalancingConfig:
    """Load balancing configuration for distributing workloads across ranks.

    The `cost_model` defines how the cost of a tensor is computed, and the distributor uses this cost to partition workloads.
    By default, it uses `AlignedMemoryCostModel`, other options include `PolynomialComputationalCostModel`.

    Note: When using custom cost_model, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Args:
        cost_model (CostModel): The cost model used for load balancing. (Default: DefaultCostModel)

    """

    @staticmethod
    def _get_default_cost_model() -> CostModel:
        return DefaultCostModel

    cost_model: CostModel = field(default_factory=_get_default_cost_model)


@dataclass(init=False)
class DistributedConfig(AbstractDataclass):
    """Abstract dataclass for distributed configs in Shampoo.

    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)

    """

    target_parameter_dimensionality: int | float = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_parameter_dimensionality, float)
            and self.target_parameter_dimensionality != math.inf
        ):
            raise ValueError(
                f"Invalid {self.target_parameter_dimensionality=} value. Must be an integer or math.inf."
            )
        elif self.target_parameter_dimensionality < 1:
            raise ValueError(
                f"Invalid {self.target_parameter_dimensionality=} value. Must be >= 1."
            )


@dataclass(kw_only=True)
class SingleDeviceDistributedConfig(DistributedConfig):
    """Configuration for Shampoo without any parallelism.

    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)

    """


DefaultSingleDeviceDistributedConfig = SingleDeviceDistributedConfig()


@dataclass(kw_only=True)
class DDPDistributedConfig(DistributedConfig):
    """Configuration for DDP distributed computation.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo.

    Note: When using custom load_balancing_config, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)
        communication_dtype (torch.dtype): Data type for communication between ranks. (Default: torch.float32)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the LOCAL_WORLD_SIZE. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)
        load_balancing_config (LoadBalancingConfig): Configuration for load balancing. (Default: LoadBalancingConfig(cost_model=AlignedMemoryCostModel()))

    """

    communication_dtype: torch.dtype = torch.float32
    num_trainers_per_group: int = -1
    communicate_params: bool = False

    @staticmethod
    def _get_default_load_balancing_config() -> LoadBalancingConfig:
        return LoadBalancingConfig()

    load_balancing_config: LoadBalancingConfig = field(
        default_factory=_get_default_load_balancing_config
    )


@dataclass(kw_only=True)
class FSDPDistributedConfig(DistributedConfig):
    """Configuration for FSDP distributed computation.

    Passes in additional metadata necessary to run FSDP Shampoo.

    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)
        param_to_metadata (dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from FSDP.

    """

    param_to_metadata: dict[Parameter, FSDPParameterMetadata]


@dataclass(kw_only=True)
class HSDPDistributedConfig(FSDPDistributedConfig, DDPDistributedConfig):
    """Configuration for HSDP distributed computation.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo across ranks with shared
    parameters between different HSDP process groups.

    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)
        device_mesh (torch.distributed.device_mesh.DeviceMesh): A 2D device mesh that specifies the layout of the numbers of
            replicate and shard dimensions.
        param_to_metadata (dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from HSDP.
        communication_dtype (torch.dtype): Data type for communication between ranks. (Default: torch.float32)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the number of workers in each replicated HSDP
            group. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)
        load_balancing_config (LoadBalancingConfig): Configuration for load balancing. (Default: LoadBalancingConfig(cost_model=AlignedMemoryCostModel()))

    """

    device_mesh: DeviceMesh


@dataclass(kw_only=True)
class FullyShardDistributedConfig(DistributedConfig):
    """Configuration for FullyShard (per-parameter FSDP) distributed computation.


    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)
        param_assignment_strategy (FSDPParamAssignmentStrategy): Strategy for assigning model parameters among the FSDP shards.
            (Default: FSDPParamAssignmentStrategy.DEFAULT)
        num_sub_groups (int): Number of sub-groups to split a single param group into. Each sub-group gets its
            own distributor and NCCL communicator, enabling multi-stream overlap of compute and communication.
            Currently only effective with the *lossless* FSDP/HSDP distributors
            (`shampoo_fully_shard_lossless_distributor`,
            `shampoo_hybrid_shard_lossless_distributor`); other distributor
            types ignore this field. (Default: 1)
        load_balancing_config (LoadBalancingConfig): Cost model used by the ROUND_ROBIN assignment
            strategy to balance per-owner-rank load when assigning parameters to ranks. Defaults to
            ``AlignedMemoryCostModel`` (aligned bytes). Only consulted for ROUND_ROBIN. (Default: LoadBalancingConfig())
    """

    param_assignment_strategy: FSDPParamAssignmentStrategy = (
        FSDPParamAssignmentStrategy.DEFAULT
    )
    num_sub_groups: int = 1

    @staticmethod
    def _get_default_load_balancing_config() -> LoadBalancingConfig:
        return LoadBalancingConfig()

    load_balancing_config: LoadBalancingConfig = field(
        default_factory=_get_default_load_balancing_config
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_sub_groups < 1:
            raise ValueError(f"Invalid {self.num_sub_groups=}. Must be >= 1.")
        # Sub-group splitting is only supported by the lossless distributors
        # (selected when param_assignment_strategy is REPLICATE or ROUND_ROBIN).
        if (
            self.num_sub_groups > 1
            and self.param_assignment_strategy == FSDPParamAssignmentStrategy.DEFAULT
        ):
            raise ValueError(
                f"num_sub_groups > 1 requires param_assignment_strategy != DEFAULT "
                f"(REPLICATE or ROUND_ROBIN); only the lossless FSDP/HSDP "
                f"distributors support param-group splitting. Got "
                f"{self.num_sub_groups=}, {self.param_assignment_strategy=}."
            )


@dataclass(kw_only=True)
class HybridShardDistributedConfig(FullyShardDistributedConfig, DDPDistributedConfig):
    """Configuration for HybridShard (per-parameter FSDP) distributed computation.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo across ranks with shared
    parameters between different Hybrid Shard process groups.

    Attributes:
        target_parameter_dimensionality (int | float): The idealized parameter dimensionality for a given algorithm.
            The dimensions of parameters and gradients will be merged (after squeezing dimensions of size 1) while respecting max_preconditioner_dim until the tensor has target_parameter_dimensionality dimensions left.
            If it should be guranteed that the parameters/gradients with > target_parameter_dimensionality dimensions are reshaped to target_parameter_dimensionality-D tensors, then max_preconditioner_dim=math.inf has to be used.
            For example, when target_parameter_dimensionality=1, we ideally want to precondition 1D gradients, e.g. with full-matrix Adagrad.
            If target_parameter_dimensionality=2, we ideally want to consider 2D gradients, e.g. for spectral descent in Muon.
            If target_parameter_dimensionality=math.inf, no dimensions are merged (besides dimensions of size 1).
            (Default: 1)
        param_assignment_strategy (FSDPParamAssignmentStrategy): Strategy for assigning model parameters among the FSDP shards.
            (Default: FSDPParamAssignmentStrategy.DEFAULT)
        device_mesh (torch.distributed.device_mesh.DeviceMesh): Device mesh for Hybrid Shard.
        communication_dtype (torch.dtype): Data type for communication between ranks. (Default: torch.float32)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the number of workers in each replicated HSDP
            group. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)
        load_balancing_config (LoadBalancingConfig): Configuration for load balancing. (Default: LoadBalancingConfig(cost_model=AlignedMemoryCostModel()))

    """

    device_mesh: DeviceMesh


_ShampooPT2CompileConfigImpl: type[object] = make_dataclass(
    "_ShampooPT2CompileConfigImpl",
    [
        (name, param.annotation, param.default)
        for name, param in signature(torch.compile).parameters.items()
        if name != "model"
    ],
    kw_only=True,
)


class ShampooPT2CompileConfig(
    _ShampooPT2CompileConfigImpl  # type: ignore
):
    """Configuration for Shampoo PT2 compilation.

    Enables Shampoo pytorch compilation with configure to speed up model training.
    For more details: https://pytorch.org/get-started/pytorch-2.0/

    The fields under ShampooPT2CompileConfig are the same as the arguments of torch.compile except `model`.
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)


@dataclass(kw_only=True)
class ShampooRuntimeConfig:
    """
    Runtime configuration for Shampoo. Only non-checkpointed options here.

    Attributes:
        eager_nan_check (bool): Flag for checking for NaNs in Shampoo step() eagerly. If enabled, it triggers host-device syncs for each iteration (Default: False)
    """

    eager_nan_check: bool = False


DefaultShampooRuntimeConfig = ShampooRuntimeConfig()


@dataclass(kw_only=True)
class ConcurrencyConfig:
    """
    Controls concurrent execution of param-group steps.

    Attributes:
        enable_cuda_stream_for_param_groups (bool): Run each param group's GPU
            work on its own CUDA stream so independent NCCL communicators can
            overlap on-device. Requires all param groups to be on CUDA;
            ignored on CPU. (Default: True)
    """

    enable_cuda_stream_for_param_groups: bool = True


DefaultConcurrencyConfig = ConcurrencyConfig()
