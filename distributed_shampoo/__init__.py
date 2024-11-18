from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
    DistributedConfig,
    FSDPShampooConfig,
    FullyShardShampooConfig,
    GraftingConfig,
    HSDPShampooConfig,
    PrecisionConfig,
    RMSpropGraftingConfig,
    SGDGraftingConfig,
    ShampooPT2CompileConfig,
)
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata
from matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEighEigenvalueCorrectionConfig,
    EigenConfig,
    EigenvalueCorrectionConfig,
    EighEigenvalueCorrectionConfig,
    PreconditionerComputationConfig,
    QREigenvalueCorrectionConfig,
    RootInvConfig,
)


__all__ = [
    "DistributedShampoo",
    # `grafting_config` options.
    "GraftingConfig",  # Abstract base class.
    "SGDGraftingConfig",
    "AdaGradGraftingConfig",
    "RMSpropGraftingConfig",
    "AdamGraftingConfig",
    # PT2 compile.
    "ShampooPT2CompileConfig",
    # `distributed_config` options.
    "DistributedConfig",  # Abstract base class.
    "DDPShampooConfig",
    "FSDPShampooConfig",
    "FullyShardShampooConfig",
    "HSDPShampooConfig",
    # `precision_config`.
    "PrecisionConfig",
    # `preconditioner_computation_config` options.
    "PreconditionerComputationConfig",  # Abstract base class.
    "RootInvConfig",  # Abstract base class (based on `PreconditionerComputationConfig`).
    "EigenConfig",
    "DefaultEigenConfig",  # Default `RootInvConfig`.
    "CoupledNewtonConfig",
    "CoupledHigherOrderConfig",
    "EigenvalueCorrectionConfig",  # Abstract base class (based on `PreconditionerComputationConfig`).
    "EighEigenvalueCorrectionConfig",
    "DefaultEighEigenvalueCorrectionConfig",  # Default `EigenvalueCorrectionConfig`.
    "QREigenvalueCorrectionConfig",
    # Other utilities.
    "compile_fsdp_parameter_metadata",  # For `FSDPShampooConfig` and `HSDPShampooConfig`.
    "CommunicationDType",  # For `DDPShampooConfig` and `HSDPShampooConfig`.
]
