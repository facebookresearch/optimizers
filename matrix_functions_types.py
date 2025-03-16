"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from dataclasses import dataclass, field

import torch

from commons import AbstractDataclass


@dataclass(init=False)
class MatrixFunctionConfig(AbstractDataclass):
    """Base dataclass for matrix function configurations."""


@dataclass(init=False)
class EigendecompositionConfig(MatrixFunctionConfig):
    """Configuration for eigenvalue decomposition."""


@dataclass(kw_only=True)
class EighEigendecompositionConfig(EigendecompositionConfig):
    """Configuration for eigendecomposition with torch.linalg.eigh.

    Attributes:
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        eigendecomposition_offload_device (torch.device | str): Device to offload eigendecomposition to. If value is empty string, we don't perform offloading. (Default: "")

    """

    retry_double_precision: bool = True
    eigendecomposition_offload_device: torch.device | str = ""

    def __post_init__(self) -> None:
        # Convert an non-empty string to a torch.device; this verifies that the string is a valid device string early.
        if self.eigendecomposition_offload_device != "":
            self.eigendecomposition_offload_device = torch.device(
                self.eigendecomposition_offload_device
            )


DefaultEigendecompositionConfig = EighEigendecompositionConfig()


@dataclass(kw_only=True)
class QREigendecompositionConfig(EigendecompositionConfig):
    """Configuration for eigenvalue decomposition via QR algorithm.

    Determines whether the QR algorithm is converged based on the approximate eigenvalues Q^T A Q =: B, where Q is the last computed eigenvectors and A is the current Kronecker factor.
    The approximate eigenvalues update criterion is then defined as ||B - diag(B)||_F <= tolerance * ||B||_F.  # TODO: Potentially improve the criterion.
    The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

    Attributes:
        max_iterations (int): The maximum number of iterations to perform. (Default: 1)
        tolerance (float): The tolerance for determining convergence in terms of the norm of the off-diagonal elements of the eigenvalue estimate.
            (Default: 0.01)
        eigenvectors_estimate (Tensor): The current estimate of the eigenvectors. Cannot be set at initialization.

    """

    max_iterations: int = 1
    tolerance: float = 0.01
    eigenvectors_estimate: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.tolerance <= 1.0):
            raise ValueError(
                f"Invalid tolerance value: {self.tolerance}. Must be in the interval [0.0, 1.0]."
            )


@dataclass(init=False)
class RootInvConfig(MatrixFunctionConfig):
    """Base dataclass for matrix root inverse method configurations."""


@dataclass(kw_only=True)
class EigenConfig(RootInvConfig, EighEigendecompositionConfig):
    """Configuration for matrix root inverse via an eigendecomposition.

    Attributes:
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        eigendecomposition_offload_device (torch.device | str): Device to offload eigendecomposition to. If value is empty string, we don't perform offloading. (Default: "")
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        enhance_stability (bool): Whether to enhance the stability of the root inverse computation through mathematically identical, but numerically more stable conditioning. (Default: False)

    """

    exponent_multiplier: float = 1.0
    enhance_stability: bool = False


DefaultEigenConfig = EigenConfig()


@dataclass(kw_only=True)
class CoupledNewtonConfig(RootInvConfig):
    """Configuration for matrix root inverse via coupled Newton method.

    Attributes:
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)

    """

    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(kw_only=True)
class CoupledHigherOrderConfig(RootInvConfig):
    """Configuration for matrix root inverse via coupled higher-order method.

    Attributes:
        rel_epsilon (float): Relative epsilon for coupled higher order method. Adds epsilon * lambda_max * I to matrix
            before taking matrix root, where lambda_max is an upper bound on maximum eigenvalue. (Default: 0.0)
        max_iterations (int): Maximum number of iterations for coupled higher order method. (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled higher order method. (Default: 1e-8)
        order (int): Order of the method. Order must be >= 2.  Higher order methods accelerate convergence (fewer iterations),
            but can take more matmuls per iteration. order=2 represents Newton's method. (Default: 3)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True,
            since tf32 is challenging numerically here. (Default: True)

    """

    rel_epsilon: float = 0.0
    max_iterations: int = 100
    tolerance: float = 1e-8
    order: int = 3
    disable_tf32: bool = True
