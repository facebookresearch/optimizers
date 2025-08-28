"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEigendecompositionConfig,
    DefaultNewtonSchulzOrthogonalizationConfig,
    DefaultPerturbationConfig,
    EigenConfig,
    EigendecompositionConfig,
    EighEigendecompositionConfig,
    MatrixFunctionConfig,
    NewtonSchulzOrthogonalizationConfig,
    OrthogonalizationConfig,
    PerturbationConfig,
    PseudoInverseConfig,
    QREigendecompositionConfig,
    RankDeficientStabilityConfig,
    RootInvConfig,
    SVDOrthogonalizationConfig,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    AdamPreconditionerConfig,
    AmortizedPreconditionerConfig,
    DDPDistributedConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSignDescentPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    DefaultSOAPConfig,
    DefaultSpectralDescentPreconditionerConfig,
    DistributedConfig,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    FSDPDistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    HSDPDistributedConfig,
    HybridShardDistributedConfig,
    PreconditionerConfig,
    RMSpropPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    SGDPreconditionerConfig,
    ShampooPreconditionerConfig,
    ShampooPT2CompileConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
)


__all__ = [
    "DistributedShampoo",
    # PT2 compile.
    "ShampooPT2CompileConfig",
    # `distributed_config` options.
    "DistributedConfig",  # Abstract base class.
    "SingleDeviceDistributedConfig",
    "DefaultSingleDeviceDistributedConfig",  # Default `SingleDeviceDistributedConfig`.
    "DDPDistributedConfig",
    "FSDPParamAssignmentStrategy",
    "FSDPDistributedConfig",
    "FullyShardDistributedConfig",
    "HSDPDistributedConfig",
    "HybridShardDistributedConfig",
    # `precision_config`.
    # `preconditioner_config` options.
    "PreconditionerConfig",  # Abstract base class.
    "AmortizedPreconditionerConfig",  # Abstract base class (based on `PreconditionerConfig`).
    "ShampooPreconditionerConfig",  # Abstract base class (based on `AmortizedPreconditionerConfig`).
    "RootInvShampooPreconditionerConfig",  # Based on `ShampooPreconditionerConfig`.
    "DefaultShampooConfig",  # Default `RootInvShampooPreconditionerConfig` using `EigenConfig`.
    "EigendecomposedShampooPreconditionerConfig",  # Based on `ShampooPreconditionerConfig`.
    "EigenvalueCorrectedShampooPreconditionerConfig",  # Based on `AmortizedPreconditionerConfig`.
    "DefaultEigenvalueCorrectedShampooConfig",  # Default `EigenvalueCorrectedShampooPreconditionerConfig` using `EighEigendecompositionConfig`.
    "DefaultSOAPConfig",  # Default `EigenvalueCorrectedShampooPreconditionerConfig` using `QREigendecompositionConfig`.
    "SpectralDescentPreconditionerConfig",  # Based on `PreconditionerConfig`.
    "SignDescentPreconditionerConfig",  # Based on `PreconditionerConfig`.
    "DefaultSpectralDescentPreconditionerConfig",  # Default `SpectralDescentPreconditionerConfig` using `NewtonSchulzOrthogonalizationConfig`.
    "DefaultSignDescentPreconditionerConfig",  # Default `SignDescentPreconditionerConfig`.
    "SGDPreconditionerConfig",
    "AdaGradPreconditionerConfig",
    "RMSpropPreconditionerConfig",
    "AdamPreconditionerConfig",
    # matrix functions configs.
    "RankDeficientStabilityConfig",  # Abstract base class.
    "PerturbationConfig",  # Based on `RankDeficientStabilityConfig`.
    "DefaultPerturbationConfig",  # Default `PerturbationConfig`.
    "PseudoInverseConfig",  # Based on `RankDeficientStabilityConfig`.
    "MatrixFunctionConfig",  # Abstract base class.
    "EigendecompositionConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "EighEigendecompositionConfig",  # Based on `EigendecompositionConfig`.
    "DefaultEigendecompositionConfig",  # Default `EigendecompositionConfig` using `EighEigendecompositionConfig`.
    "QREigendecompositionConfig",  # Based on `EigendecompositionConfig`.
    "RootInvConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "EigenConfig",  # Based on `RootInvConfig` and `EigendecompositionConfig`.
    "DefaultEigenConfig",  # Default `RootInvConfig` using `EigenConfig`.
    "CoupledNewtonConfig",  # Based on `RootInvConfig`.
    "CoupledHigherOrderConfig",  # Based on `RootInvConfig`.
    "OrthogonalizationConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "SVDOrthogonalizationConfig",  # Based on `OrthogonalizationConfig`.
    "NewtonSchulzOrthogonalizationConfig",  # Based on `OrthogonalizationConfig`.
    "DefaultNewtonSchulzOrthogonalizationConfig",  # Default `OrthogonalizationConfig` using `NewtonSchulzOrthogonalizationConfig`.
    # Other utilities.
    "compile_fsdp_parameter_metadata",  # For `FSDPDistributedConfig` and `HSDPDistributedConfig`.
]
