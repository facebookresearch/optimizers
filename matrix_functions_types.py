"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from dataclasses import dataclass

from distributed_shampoo.shampoo_types import AbstractDataclass


@dataclass
class PreconditionerComputationConfig(AbstractDataclass):
    """Configuration for preconditioner computation in Shampoo."""

    ...


@dataclass
class RootInvConfig(PreconditionerComputationConfig):
    """Base dataclass for matrix root inverse method configurations in Shampoo."""

    ...


@dataclass(kw_only=True)
class EigenConfig(RootInvConfig):
    """Configuration for eigendecomposition method in Shampoo.

    Args:
        make_positive_semidefinite (bool): Perturbs matrix eigenvalues to ensure it is numerically positive semi-definite. (Default: True)
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher(double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)

    """

    make_positive_semidefinite: bool = True
    retry_double_precision: bool = True
    exponent_multiplier: float = 1.0


DefaultEigenConfig = EigenConfig()


@dataclass(kw_only=True)
class CoupledNewtonConfig(RootInvConfig):
    """Configuration for coupled Newton method in Shampoo.

    Args:
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)

    """

    max_iterations: int = 100
    tolerance: float = 1e-6


@dataclass(kw_only=True)
class CoupledHigherOrderConfig(RootInvConfig):
    """Configuration for coupled higher-order method in Shampoo.

    Args:
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


@dataclass
class EigenvalueCorrectionConfig(PreconditionerComputationConfig):
    """Base dataclass for matrix eigenvector method configurations in Shampoo."""

    ...


@dataclass(kw_only=True)
class EighEigenvalueCorrectionConfig(EigenvalueCorrectionConfig):
    """Configuration for eigendecomposition method used in eigenvalue corrected Shampoo.

    Args:
        retry_double_precision (bool): Whether to re-trying eigendecomposition with higher(double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)

    """

    retry_double_precision: bool = True


DefaultEighEigenvalueCorrectionConfig = EighEigenvalueCorrectionConfig()
