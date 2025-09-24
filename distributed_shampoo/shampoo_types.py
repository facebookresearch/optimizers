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
MOMENTUM = "momentum"
STEP = "step"

# Keys for parameter groups (checkpointed if specified)
BETA3 = "beta3"
BETAS = "betas"
DAMPENING = "dampening"
DISTRIBUTED_CONFIG = "distributed_config"
EPSILON = "epsilon"
GRAFTING_CONFIG = "grafting_config"
LR = "lr"
MAX_PRECONDITIONER_DIM = "max_preconditioner_dim"
PARAMS = "params"  # While this is stored in groups by default, we do not checkpoint this quantity.
PRECONDITION_FREQUENCY = "precondition_frequency"
PRECONDITIONER_CONFIG = "preconditioner_config"
START_PRECONDITIONING_STEP = "start_preconditioning_step"
USE_EIGENVALUE_CORRECTION = "use_eigenvalue_correction"
USE_BIAS_CORRECTION = "use_bias_correction"
USE_DECOUPLED_WEIGHT_DECAY = "use_decoupled_weight_decay"
USE_NESTEROV = "use_nesterov"
USE_PIN_MEMORY = "use_pin_memory"
WEIGHT_DECAY = "weight_decay"

# Keys for lists of blocked states and metadata (never checkpointed)
DISTRIBUTOR = "distributor"
FILTERED_GRAD_LIST = "filtered_grad_list"
GRAFTING_PRECONDITIONER_LIST = "grafting_preconditioner_list"
MASKED_BLOCKED_GRADS = "masked_blocked_grads"
MASKED_BLOCKED_PARAMS = "masked_blocked_params"
MASKED_FILTERED_GRAD_LIST = "masked_filtered_grad_list"
MASKED_MOMENTUM_LIST = "masked_momentum_list"
MOMENTUM_LIST = "momentum_list"
PREVIOUS_GRAD_SELECTOR = "previous_grad_selector"
SHAMPOO_PRECONDITIONER_LIST = "shampoo_preconditioner_list"


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

    """

    beta2: float = 0.99

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
class AmortizedPreconditionerConfig(PreconditionerConfig):
    """Configuration for amortized preconditioner computation in DistributedShampoo.

    Attributes:
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized computation, e.g., inverse-root computation or eigendecomposition.
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
        factor_matrix_dtype (torch.dtype): Data type for factor matrix. (Default: torch.float32)

    """

    # repr=False prevents __repr__() from accessing this field to avoid linter complaints
    amortized_computation_config: MatrixFunctionConfig = field(repr=False)
    num_tolerated_failed_amortized_computations: int = 3
    factor_matrix_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.num_tolerated_failed_amortized_computations < 0:
            raise ValueError(
                f"Invalid num_tolerated_failed_amortized_computations value: {self.num_tolerated_failed_amortized_computations}. Must be >= 0."
            )


@dataclass(init=False)
class ShampooPreconditionerConfig(AmortizedPreconditionerConfig):
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
class RootInvShampooPreconditionerConfig(ShampooPreconditionerConfig):
    """Configuration for Shampoo preconditioner computation with caching of the root inverse factor matrices.

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

    amortized_computation_config: RootInvConfig = field(
        default_factory=lambda: DefaultEigenConfig
    )
    inv_factor_matrix_dtype: torch.dtype = torch.float32


DefaultShampooConfig = RootInvShampooPreconditionerConfig()


@dataclass(kw_only=True)
class EigendecomposedShampooPreconditionerConfig(ShampooPreconditionerConfig):
    """Configuration for Shampoo preconditioner computation with caching of the eigendecomposed factor matrices.

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

    amortized_computation_config: EigendecompositionConfig = field(
        default_factory=lambda: DefaultEigendecompositionConfig
    )
    factor_matrix_eigenvectors_dtype: torch.dtype = torch.float32
    factor_matrix_eigenvalues_dtype: torch.dtype = torch.float32


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooPreconditionerConfig(AmortizedPreconditionerConfig):
    """Configuration for eigenvalue-corrected Shampoo/SOAP preconditioner computation.

    Recall that in eigenvalue-corrected Shampoo, the eigenvectors and eigenvalues of the factor matrices are computed separately and stored in place of the full inverted preconditioner, as opposed to the single inverse-root computation of the factor matrices in Shampoo.
    In eigenvalue-corrected Shampoo, the eigenvectors are updated periodically like the inverted preconditioners in Shampoo, but the eigenvalues are updated every iteration.

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

    amortized_computation_config: EigendecompositionConfig = field(
        default_factory=lambda: DefaultEigendecompositionConfig
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
class SpectralDescentPreconditionerConfig(PreconditionerConfig):
    """Configuration for spectral descent computation in DistributedShampoo.

    NOTE: This config can only be used for 2D parameters, or parameters that have been reshaped to 2D.
    Which parameters are reshaped to 2D is determined by the max_preconditioner_dim argument in DistributedShampoo.
    If all >2D parameters should be guaranteed to be reshaped to 2D, then max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2 has to be used.

    Attributes:
        orthogonalization_config (OrthogonalizationConfig): Configuration for orthogonalization of the search direction.
            (Default: DefaultNewtonSchulzOrthogonalizationConfig)

    """

    orthogonalization_config: OrthogonalizationConfig = field(
        default_factory=lambda: DefaultNewtonSchulzOrthogonalizationConfig
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

    REPLICATE: parameters are all-gathered (replicated) across all ranks in shard dimension.
        This strategy should produce identical results as the default Shampoo implementation, but at the cost of
        significantly increased memory usage and communication.

    ROUND_ROBIN: whole parameters are assigned to ranks in a round-robin fashion.
        This strategy balances the memory and compute overhead across all ranks, by assigning the whole model
        parameters to the ranks in the shard dimension (and it should be used when there are more parameters than shards).

    """

    DEFAULT = enum.auto()
    REPLICATE = enum.auto()
    ROUND_ROBIN = enum.auto()


@dataclass(kw_only=True)
class LoadBalancingConfig:
    """Load balancing configuration for distributing workloads across ranks.

    The `cost_model` defines how the cost of a tensor is computed, and the distributor uses this cost to partition workloads.
    By default, it uses `AlignedMemoryCostModel`, other options include `PolynomialComputationalCostModel`.

    Args:
        cost_model (CostModel): The cost model used for load balancing. (Default: DefaultCostModel)

    """

    cost_model: CostModel = field(default_factory=lambda: DefaultCostModel)


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
    load_balancing_config: LoadBalancingConfig = field(
        default_factory=lambda: LoadBalancingConfig()
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
    """

    param_assignment_strategy: FSDPParamAssignmentStrategy = (
        FSDPParamAssignmentStrategy.DEFAULT
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
