"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
from dataclasses import dataclass, field

import torch

from commons import AbstractDataclass

from matrix_functions_types import (
    DefaultEigenConfig,
    DefaultEighEigenvectorConfig,
    EigenvectorConfig,
    MatrixFunctionConfig,
    QRConfig,
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
INV_ROOT_OVERRIDE = "inv_root_override"
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


###### ENUM CLASSES ######
class CommunicationDType(enum.Enum):
    DEFAULT = 0
    FP16 = 1
    BF16 = 2
    FP32 = 3


###### ERROR CLASSES ######
class PreconditionerValueError(ValueError):
    """ValueError for invalid values encountered during Preconditioner computation."""


###### DATACLASSES ######
@dataclass(init=False)
class PreconditionerConfig(AbstractDataclass):
    """Configuration for preconditioner computation in DistributedShampoo.

    Args:
        amortized_computation_config (MatrixFunctionConfig): Configuration for the amortized computation, e.g., inverse-root or eigenvector computation.
        num_tolerated_failed_amortized_computations (int): Number of failed amortized computations to tolerate before raising an error. (Default: 3)

    """

    amortized_computation_config: MatrixFunctionConfig  # type: ignore
    num_tolerated_failed_amortized_computations: int = 3

    def __post_init__(self) -> None:
        if self.num_tolerated_failed_amortized_computations < 0:
            raise ValueError(
                f"Invalid num_tolerated_failed_amortized_computations value: {self.num_tolerated_failed_amortized_computations}. Must be >= 0."
            )


@dataclass(kw_only=True)
class ShampooPreconditionerConfig(PreconditionerConfig):
    """Configuration for Shampoo preconditioner computation.

    Args:
        amortized_computation_config (RootInvConfig): Configuration for the inverse-root computation. (Default: DefaultEigenConfig)

    """

    amortized_computation_config: RootInvConfig = field(
        default_factory=lambda: DefaultEigenConfig
    )


DefaultShampooConfig = ShampooPreconditionerConfig()


@dataclass(kw_only=True)
class EigenvalueCorrectedShampooPreconditionerConfig(PreconditionerConfig):
    """Configuration for eigenvalue-corrected Shampoo/SOAP preconditioner computation.

    Args:
        amortized_computation_config (EigenvectorConfig): Configuration for the eigenvector computation.
            (Default: DefaultEighEigenvectorConfig)

    """

    amortized_computation_config: EigenvectorConfig = field(
        default_factory=lambda: DefaultEighEigenvectorConfig
    )


DefaultEigenvalueCorrectedShampooConfig = (
    EigenvalueCorrectedShampooPreconditionerConfig()
)
DefaultSOAPConfig = EigenvalueCorrectedShampooPreconditionerConfig(
    amortized_computation_config=QRConfig(),
)


@dataclass
class FSDPParameterMetadata:
    """FSDP Metadata for a parameter.

    Args:
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

    Args:
        communication_dtype (CommunicationDType): Data type for communication between ranks. (Default: DEFAULT)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the LOCAL_WORLD_SIZE. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)

    """

    communication_dtype: CommunicationDType = CommunicationDType.DEFAULT
    num_trainers_per_group: int = -1
    communicate_params: bool = False


@dataclass(kw_only=True)
class FSDPShampooConfig(DistributedConfig):
    """Configuration for FSDP Shampoo.

    Passes in additional metadata necessary to run FSDP Shampoo.

    Args:
        param_to_metadata (dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from FSDP.

    """

    param_to_metadata: dict[Parameter, FSDPParameterMetadata]


@dataclass
class HSDPShampooConfig(FSDPShampooConfig, DDPShampooConfig):
    """Configuration for HSDP Shampoo.

    Enables distributed computation and optimizer states (like ZeRO-1) via DTensor for Shampoo across ranks with shared
    parameters between different HSDP process groups.

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): A 2D device mesh that specifies the layout of the numbers of
            shard and replicate dimensions.
        param_to_metadata (dict[Parameter, FSDPParameterMetadata]): Dictionary mapping parameter to its metadata from HSDP.
        communication_dtype (CommunicationDType): Data type for communication between ranks. (Default: DEFAULT)
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

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): Device mesh for Hybrid Shard.
        communication_dtype (CommunicationDType): Data type for communication between ranks. (Default: DEFAULT)
        num_trainers_per_group (int): Number of GPUs per distributed process group for distributed computation/memory.
            If num_trainers_per_group = -1 is used, then defaults to using the number of workers in each replicated HSDP
            group. (Default: -1)
        communicate_params (bool): Flag for all-gathering updated params across multiple workers.
            If False, all-gathers parameter updates across multiple workers. (Default: False)

    """

    device_mesh: DeviceMesh


@dataclass
class ShampooPT2CompileConfig:
    """Configuration for Shampoo PT2 compilation.

    Enables Shampoo pytorch compilation with configure to speed up model training.
    For more details: https://pytorch.org/get-started/pytorch-2.0/

    Args:
        pytorch_compile_backend (str): The backend for PT2 compilation. More info about PT2 backends:
            https://pytorch.org/docs/stable/torch.compiler.html (Default: inductor)
        enable_shampoo_pt2_dynamic_shape (bool | None): Compile Shampoo in static, dynamic or auto-dynamic shape mode (Default: False).
            - False: Use 'static' mode. Static mode assumes tensors in Shampoo will NOT change shapes. We recommend using this mode if
                you expect parameters and gradients to change shapes only a very small number of times (e.g. <=5).
            - True: Use 'dynamic' mode.  Dynamic mode assumes all tensors in Shampoo can change shapes during the run. In general, we do
                not recommend using this mode, as it generates kernels that are not specialized to particular tensor shapes, and therefore
                perform much slower.
            - None: Use 'auto-dynamic' mode. Auto-dynamic mode assumes tensors in Shampoo are static, but will switch to dynamic mode if
                some tensors change shapes. If PT2 recompiles excessively during your run, we recommend trying this mode to reduce recompilation overhead.

    """

    pytorch_compile_backend: str = "inductor"
    enable_shampoo_pt2_dynamic_shape: bool | None = False


@dataclass(init=False)
class GraftingConfig(AbstractDataclass):
    """Abstract dataclass for grafting configurations in Shampoo."""


@dataclass
class SGDGraftingConfig(GraftingConfig):
    """Configuration for grafting from SGD."""


@dataclass(kw_only=True)
class AdaGradGraftingConfig(GraftingConfig):
    """Configuration for grafting from AdaGrad.

    Args:
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

    Args:
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

    Args:
        beta2 (float): Exponential moving average factor for second moment. (Default: 0.999)
        epsilon (float): Epsilon term for regularizing square-root of the second moment to ensure positive definiteness.
            (Default: 1e-10)

    """

    beta2: float = 0.999
