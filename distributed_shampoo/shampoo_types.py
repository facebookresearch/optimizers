"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from dataclasses import dataclass, field, make_dataclass
from inspect import signature
from typing import Any

import torch

from commons import AbstractDataclass

from matrix_functions_types import (
    DefaultEigenConfig,
    DefaultEigendecompositionConfig,
    EigendecompositionConfig,
    MatrixFunctionConfig,
    QREigendecompositionConfig,
    RootInvConfig,
)
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
EPSILON = "epsilon"
GRAFTING_CONFIG = "grafting_config"
LR = "lr"
MAX_PRECONDITIONER_DIM = "max_preconditioner_dim"
PARAMS = "params"  # While this is stored in groups by default, we do not checkpoint this quantity.
PRECONDITION_FREQUENCY = "precondition_frequency"
PRECONDITIONER_DTYPE = "preconditioner_dtype"
PRECONDITIONER_CONFIG = "preconditioner_config"
START_PRECONDITIONING_STEP = "start_preconditioning_step"
USE_EIGENVALUE_CORRECTION = "use_eigenvalue_correction"
USE_BIAS_CORRECTION = "use_bias_correction"
USE_DECOUPLED_WEIGHT_DECAY = "use_decoupled_weight_decay"
USE_MERGE_DIMS = "use_merge_dims"
USE_NESTEROV = "use_nesterov"
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
    """Configuration for preconditioner computation in DistributedShampoo.

    Attributes:
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized computation, e.g., inverse-root computation or eigendecomposition.
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)

    """

    # repr=False prevents __repr__() from accessing this field to avoid linter complaints
    amortized_computation_config: MatrixFunctionConfig = field(repr=False)
    num_tolerated_failed_amortized_computations: int = 3

    def __post_init__(self) -> None:
        if self.num_tolerated_failed_amortized_computations < 0:
            raise ValueError(
                f"Invalid num_tolerated_failed_amortized_computations value: {self.num_tolerated_failed_amortized_computations}. Must be >= 0."
            )


@dataclass(init=False)
class ShampooPreconditionerConfig(PreconditionerConfig):
    """Configuration for Shampoo preconditioner computation.

    Attributes:
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized computation, e.g., inverse-root computation or eigendecomposition.
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
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

    amortized_computation_config: RootInvConfig = field(
        default_factory=lambda: DefaultEigenConfig
    )


DefaultShampooConfig = RootInvShampooPreconditionerConfig()


@dataclass(kw_only=True)
class EigendecomposedShampooPreconditionerConfig(ShampooPreconditionerConfig):
    """Configuration for Shampoo preconditioner computation with caching of the eigendecomposed factor matrices.

    Attributes:
        amortized_computation_config (EigendecompositionConfig): Configuration for the eigendecomposition computation. (Default: DefaultEigendecompositionConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
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

    amortized_computation_config: EigendecompositionConfig = field(
        default_factory=lambda: DefaultEigendecompositionConfig
    )


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooPreconditionerConfig(PreconditionerConfig):
    """Configuration for eigenvalue-corrected Shampoo/SOAP preconditioner computation.

    Recall that in eigenvalue-corrected Shampoo, the eigenvectors and eigenvalues of the factor matrices are computed separately and stored in place of the full inverted preconditioner, as opposed to the single inverse-root computation of the factor matrices in Shampoo.
    In eigenvalue-corrected Shampoo, the eigenvectors are updated periodically like the inverted preconditioners in Shampoo, but the eigenvalues are updated every iteration.

    Attributes:
        amortized_computation_config (EigendecompositionConfig): Configuration for the eigenvector computation.
            (Default: DefaultEigendecompositionConfig)
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)
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

    """

    amortized_computation_config: EigendecompositionConfig = field(
        default_factory=lambda: DefaultEigendecompositionConfig
    )
    ignored_basis_change_dims: dict[int, list[int]] = field(default_factory=dict)
    inverse_exponent_override: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

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


@dataclass(init=False)
class DistributedConfig(AbstractDataclass):
    """Abstract dataclass for distributed configs in Shampoo."""


@dataclass(kw_only=True)
class DDPShampooConfig(DistributedConfig):
    """Configuration for DDP Shampoo.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo.

    Attributes:
        communication_dtype (torch.dtype): Data type for communication between ranks. (Default: torch.float32)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the LOCAL_WORLD_SIZE. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)

    """

    communication_dtype: torch.dtype = torch.float32
    num_trainers_per_group: int = -1
    communicate_params: bool = False


@dataclass(kw_only=True)
class FSDPShampooConfig(DistributedConfig):
    """Configuration for FSDP Shampoo.

    Passes in additional metadata necessary to run FSDP Shampoo.

    Attributes:
        param_to_metadata (dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from FSDP.

    """

    param_to_metadata: dict[Parameter, FSDPParameterMetadata]


@dataclass
class HSDPShampooConfig(FSDPShampooConfig, DDPShampooConfig):
    """Configuration for HSDP Shampoo.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo across ranks with shared
    parameters between different HSDP process groups.

    Attributes:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): A 2D device mesh that specifies the layout of the numbers of
            replicate and shard dimensions.
        param_to_metadata (dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from HSDP.
        communication_dtype (torch.dtype): Data type for communication between ranks. (Default: torch.float32)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the number of workers in each replicated HSDP
            group. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)

    """

    device_mesh: DeviceMesh


@dataclass(kw_only=True)
class FullyShardShampooConfig(DistributedConfig):
    """Configuration for FullyShard (per-parameter FSDP) Shampoo.

    Currently only a placeholder used for Shampoo optimizer to select FullyShardDistributor.
    """


@dataclass
class HybridShardShampooConfig(FullyShardShampooConfig, DDPShampooConfig):
    """Configuration for HybridShard (per-parameter FSDP) Shampoo.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo across ranks with shared
    parameters between different Hybrid Shard process groups.

    Attributes:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): Device mesh for Hybrid Shard.
        communication_dtype (torch.dtype): Data type for communication between ranks. (Default: torch.float32)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the number of workers in each replicated HSDP
            group. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


@dataclass(init=False)
class GraftingConfig(AbstractDataclass):
    """Abstract dataclass for grafting configurations in Shampoo."""


@dataclass
class SGDGraftingConfig(GraftingConfig):
    """Configuration for grafting from SGD."""


@dataclass(kw_only=True)
class AdaGradGraftingConfig(GraftingConfig):
    """Configuration for grafting from AdaGrad.

    Attributes:
        epsilon (float): Epsilon term for regularizing square-root of the aggregated second moment to ensure positive definiteness.
            (Default: 1e-10)

    """

    epsilon: float = 1e-10

    def __post_init__(self) -> None:
        if not self.epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {self.epsilon}. Must be > 0.0.")


@dataclass(kw_only=True)
class RMSpropGraftingConfig(AdaGradGraftingConfig):
    """Configuration for grafting from RMSprop.

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
class AdamGraftingConfig(RMSpropGraftingConfig):
    """Configuration for grafting from Adam.

    Attributes:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-10)

    Note:
        The traditional beta1 parameter in Adam is set by betas[0] in DistributedShampoo's hyperparameters.
    """

    beta2: float = 0.999
