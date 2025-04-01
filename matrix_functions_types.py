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
class NonInvertibleHandlingConfig(AbstractDataclass):
    """Base data class for configurations for handling non-invertible (i.e. singular/rank-deficient) matrices."""


@dataclass(kw_only=True)
class RegularizationConfig(NonInvertibleHandlingConfig):
    """
    Configuration for perturbing matrix values by a small amount, i.e. epsilon, to guarantee invertibility.

    Attributes:
        add_epsilon_before_computation (bool): Whether to apply epsilon before amortized computation instead of after. Note
            that both options are mathematically equivalent. Recommended to be set to True for numerical stability.
            (Default: True)
    """

    add_epsilon_before_computation: bool = True


DefaultRegularizationConfig = RegularizationConfig()


@dataclass(kw_only=True)
class PseudoInverseConfig(NonInvertibleHandlingConfig):
    """
    Configuration for filtering zero/near-zero singular values (i.e. determining rank) to return a pseudo-inverse when the matrix is non-invertible.
    For more information, refer to https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html.

    Attributes:
        rank_atol: Absolute tolerance for filtering singular values. (Default: 0.0)
        rank_rtol: Relative tolerance for filtering singular values. When None, takes value of max dim of the matrix times the
            epsilon of the dtype of the matrix. (Default: None)
    """

    rank_atol: float = 0.0
    rank_rtol: float | None = None


@dataclass(init=False)
class MatrixFunctionConfig(AbstractDataclass):
    """Base dataclass for matrix function configurations."""


@dataclass(init=False)
class EigendecompositionConfig(MatrixFunctionConfig):
    """Configuration for eigenvalue decomposition.

    Attributes:
        noninvertible_handling_config (NonInvertibleHandlingConfig): Config for handling non-invertible matrices. (Default: DefaultRegularizationConfig) TODO: generalize this to MatrixFunctionConfig
    """

    noninvertible_handling_config: NonInvertibleHandlingConfig = field(
        default_factory=lambda: DefaultRegularizationConfig
    )


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

    Determines whether the QR algorithm has converged based on the estimated eigenvalues Q^T A Q =: B, where Q is the last computed eigenvectors and A is the current Kronecker factor.
    The convergence criterion based on the estimated eigenvalues is then defined as ||B - diag(B)||_F <= tolerance * ||B||_F.
    The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

    Note that if the criterion based on the estimated eigenvalues is already below or equal to the tolerance given the initial eigenvectors_estimate, the QR iterations will be skipped.

    This convergence criterion can be motivated by considering A' = Q diag(B) Q^T as an approximation of A.
    We have ||A - A'||_F = ||A - Q diag(B) Q^T||_F = ||Q^T A Q - diag(B)||_F = ||B - diag(B)||_F.
    Moreover, we have ||B||_F = ||Q^T A Q||_F = ||A||_F.
    Hence, the two relative errors are also equivalent: ||A - A'||_F / ||A||_F = ||B - diag(B)||_F / ||B||_F.

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

    """

    exponent_multiplier: float = 1.0


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
        max_iterations (int): Maximum number of iterations for coupled higher order method. Typically we need < 20 iterations.
            (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled higher order method. In practice, 1e-20
            guarantees a run to convergence. (Default: 1e-8)
        order (int): Order of the method. Order must be >= 2. Higher order methods accelerate convergence (fewer iterations),
            but can take more matmuls per iteration. order=2 represents Newton's method. (Default: 3)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True,
            since tf32 is challenging numerically here. (Default: True)

    """

    rel_epsilon: float = 0.0
    max_iterations: int = 100
    tolerance: float = 1e-8
    order: int = 3
    disable_tf32: bool = True
