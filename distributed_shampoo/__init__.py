"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    CommunicationDType,
    DDPShampooConfig,
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    DefaultSOAPConfig,
    DistributedConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    FSDPShampooConfig,
    FullyShardShampooConfig,
    GraftingConfig,
    HSDPShampooConfig,
    PreconditionerConfig,
    RMSpropGraftingConfig,
    SGDGraftingConfig,
    ShampooPreconditionerConfig,
    ShampooPT2CompileConfig,
)
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata
from matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEighEigenvectorConfig,
    EigenConfig,
    EigenvalueDecompositionConfig,
    EigenvectorConfig,
    EighEigenvectorConfig,
    MatrixFunctionConfig,
    QRConfig,
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
    # `preconditioner_config` options.
    "PreconditionerConfig",  # Abstract base class.
    "ShampooPreconditionerConfig",  # Based on `PreconditionerConfig`.
    "DefaultShampooConfig",  # Default `ShampooPreconditionerConfig` using `EigenConfig`.
    "EigenvalueCorrectedShampooPreconditionerConfig",  # Based on `PreconditionerConfig`.
    "DefaultEigenvalueCorrectedShampooConfig",  # Default `EigenvalueCorrectedShampooPreconditionerConfig` using `EighEigenvectorConfig`.
    "DefaultSOAPConfig",  # Default `EigenvalueCorrectedShampooPreconditionerConfig` using `QRConfig`.
    # matrix functions configs.
    "MatrixFunctionConfig",  # Abstract base class.
    "EigenvalueDecompositionConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "RootInvConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "EigenConfig",  # Based on `RootInvConfig` and `EigenvalueDecompositionConfig`.
    "DefaultEigenConfig",  # Default `RootInvConfig` using `EigenConfig`.
    "CoupledNewtonConfig",  # Based on `RootInvConfig`.
    "CoupledHigherOrderConfig",  # Based on `RootInvConfig`.
    "EigenvectorConfig",  # Abstract base class (based on `MatrixFunctionConfig`).
    "EighEigenvectorConfig",  # Based on `EigenvectorConfig` and `EigenvalueDecompositionConfig`.
    "DefaultEighEigenvectorConfig",  # Default `EigenvectorConfig` using `EighEigenvectorConfig`.
    "QRConfig",  # Based on `EigenvectorConfig`.
    # Other utilities.
    "compile_fsdp_parameter_metadata",  # For `FSDPShampooConfig` and `HSDPShampooConfig`.
    "CommunicationDType",  # For `DDPShampooConfig` and `HSDPShampooConfig`.
]
