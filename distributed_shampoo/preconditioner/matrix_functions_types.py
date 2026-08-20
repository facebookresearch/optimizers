"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from distributed_shampoo.utils.abstract_dataclass import AbstractDataclass

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(init=False)
class RankDeficientStabilityConfig(AbstractDataclass):
    """Base data class for configurations for handling/stabilizing rank-deficient (i.e. singular/non-invertible) matrices."""


@dataclass(kw_only=True)
class PerturbationConfig(RankDeficientStabilityConfig):
    """
    Configuration for perturbing/damping/regularizing matrix eigenvalues by a small value epsilon (provided in DistributedShampoo arguments) to guarantee invertibility.

    Mathematically, this can be seen as Tikhonov regularization/ridge regression. It can also be viewed as defining a trust region on the optimizer update.

    NOTE: Not to be confused with dampening, which controls the effect of momentum.

    Attributes:
        perturb_before_computation (bool): Whether to apply epsilon before amortized computation instead of after. Note
            that both options are mathematically equivalent, but not necessarily numerically equivalent.
            For eigenvalue-corrected Shampoo this will only affect the stability of the eigenbasis computation and epsilon will always also be added to the corrected eigenvalues.
            Recommended to be set to True for numerical stability.
            TODO: When generalizing to all MatrixFunctionConfigs, this is only applicable to EigendecompositionConfig.
            (Default: True)
    """

    perturb_before_computation: bool = True


DefaultPerturbationConfig = PerturbationConfig()


@dataclass(kw_only=True)
class PseudoInverseConfig(RankDeficientStabilityConfig):
    """
    Configuration for filtering zero/near-zero singular values (i.e., determining rank) to return a pseudo-inverse when the matrix is non-invertible.
    For more information, refer to https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html.

    Attributes:
        rank_atol: Absolute tolerance for filtering singular values.
            TODO: When generalizing to all MatrixFunctionConfigs, this is only applicable to EigendecompositionConfig.
            (Default: 0.0)
        rank_rtol: Relative tolerance for filtering singular values. When None, takes value of max dim of the matrix times the
            epsilon of the dtype of the matrix.
            TODO: When generalizing to all MatrixFunctionConfigs, this is only applicable to EigendecompositionConfig.
            (Default: 0.0)
    """

    rank_atol: float = 0.0
    rank_rtol: float | None = 0.0


@dataclass(init=False)
class MatrixFunctionConfig(AbstractDataclass):
    """Base dataclass for matrix function configurations."""


@dataclass(init=False)
class EigendecompositionConfig(MatrixFunctionConfig):
    """Configuration for eigenvalue decomposition.

    The tolerance hyperparameter is used for a criterion that enables an adaptive eigendecomposition update frequency.
    The criterion uses the estimated eigenvalues Q^T A Q =: B, where Q is the last computed eigenvectors and A is the current Kronecker factor.
    The criterion is then defined as ||B - diag(B)||_F <= tolerance * ||B||_F.
    The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

    If the criterion is already below or equal to the tolerance given the initial eigenvectors estimate, the eigendecomposition will be skipped and the estimated eigenvalues and initial eigenvectors will be returned.
    When the QR algorithm is used, the criterion is also used to determine convergence of the QR algorithm.

    This criterion can be motivated by considering A' = Q diag(B) Q^T as an approximation of A.
    We have ||A - A'||_F = ||A - Q diag(B) Q^T||_F = ||Q^T A Q - diag(B)||_F = ||B - diag(B)||_F.
    Moreover, we have ||B||_F = ||Q^T A Q||_F = ||A||_F.
    Hence, the two relative errors are also equivalent: ||A - A'||_F / ||A||_F = ||B - diag(B)||_F / ||B||_F.

    Note: When using custom rank_deficient_stability_config, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)
            TODO: generalize this to MatrixFunctionConfig
        tolerance (float): The tolerance for the error of the eigendecomposition based on the norm of the off-diagonal elements of the eigenvalue estimate.
            (Default: 0.0)

    """

    @staticmethod
    def _get_default_rank_deficient_stability_config() -> RankDeficientStabilityConfig:
        return DefaultPerturbationConfig

    rank_deficient_stability_config: RankDeficientStabilityConfig = field(
        default_factory=_get_default_rank_deficient_stability_config
    )
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.tolerance <= 1.0):
            raise ValueError(
                f"Invalid tolerance value: {self.tolerance}. Must be in the interval [0.0, 1.0]."
            )


@dataclass(kw_only=True)
class EighEigendecompositionConfig(EigendecompositionConfig):
    """Configuration for eigendecomposition with torch.linalg.eigh.

    The tolerance hyperparameter is used for a criterion that enables an adaptive eigendecomposition update frequency.
    The criterion uses the estimated eigenvalues Q^T A Q =: B, where Q is the last computed eigenvectors and A is the current Kronecker factor.
    The criterion is then defined as ||B - diag(B)||_F <= tolerance * ||B||_F.
    The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

    If the criterion is already below or equal to the tolerance given the initial eigenvectors estimate, the eigendecomposition will be skipped and the estimated eigenvalues and initial eigenvectors will be returned.

    This criterion can be motivated by considering A' = Q diag(B) Q^T as an approximation of A.
    We have ||A - A'||_F = ||A - Q diag(B) Q^T||_F = ||Q^T A Q - diag(B)||_F = ||B - diag(B)||_F.
    Moreover, we have ||B||_F = ||Q^T A Q||_F = ||A||_F.
    Hence, the two relative errors are also equivalent: ||A - A'||_F / ||A||_F = ||B - diag(B)||_F / ||B||_F.

    Attributes:
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)
        retry_double_precision (bool): Whether to retry eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        eigendecomposition_offload_device (str): Device to offload eigendecomposition to. If value is empty string, we don't perform offloading. (Default: "")
        tolerance (float): The tolerance which can lead to skipping of the eigendecomposition based on the norm of the off-diagonal elements of the eigenvalue estimate.
            (Default: 0.0)

    """

    retry_double_precision: bool = True
    eigendecomposition_offload_device: str = ""


DefaultEigendecompositionConfig = EighEigendecompositionConfig()


@dataclass(kw_only=True)
class QREigendecompositionConfig(EigendecompositionConfig):
    """Configuration for eigenvalue decomposition via QR algorithm.

    Determines whether the QR algorithm has converged based on the estimated eigenvalues Q^T A Q =: B, where Q is the last computed eigenvectors and A is the current Kronecker factor.
    The convergence criterion based on the estimated eigenvalues is then defined as ||B - diag(B)||_F <= tolerance * ||B||_F.
    The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

    Note that if the criterion based on the estimated eigenvalues is already below or equal to the tolerance given the initial eigenvectors estimate, the QR iterations will be skipped.

    This convergence criterion can be motivated by considering A' = Q diag(B) Q^T as an approximation of A.
    We have ||A - A'||_F = ||A - Q diag(B) Q^T||_F = ||Q^T A Q - diag(B)||_F = ||B - diag(B)||_F.
    Moreover, we have ||B||_F = ||Q^T A Q||_F = ||A||_F.
    Hence, the two relative errors are also equivalent: ||A - A'||_F / ||A||_F = ||B - diag(B)||_F / ||B||_F.

    Attributes:
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)
        max_iterations (int): The maximum number of iterations to perform. (Default: 1)
        tolerance (float): The tolerance for determining convergence in terms of the norm of the off-diagonal elements of the eigenvalue estimate.
            (Default: 0.0)

    """

    max_iterations: int = 1


@dataclass(init=False)
class RootInvConfig(MatrixFunctionConfig):
    """Base dataclass for matrix root inverse method configurations."""


@dataclass(kw_only=True)
class EigenConfig(RootInvConfig, EighEigendecompositionConfig):
    """Configuration for matrix root inverse via an eigendecomposition.

    Attributes:
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)
        retry_double_precision (bool): Whether to retry eigendecomposition with higher (double) precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        eigendecomposition_offload_device (str): Device to offload eigendecomposition to. If value is empty string, we don't perform offloading. (Default: "")

    """

    def __post_init__(self) -> None:
        EighEigendecompositionConfig.__post_init__(self)
        if self.tolerance != 0.0:
            raise ValueError(
                f"Invalid tolerance value: {self.tolerance}. Must be 0.0 for {type(self).__name__}."
            )


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
class NewtonSchulzRootInvConfig(RootInvConfig):
    """Configuration for matrix root inverse via the coupled Newton-Schulz iteration.

    Unlike CoupledNewtonConfig and CoupledHigherOrderConfig, the iteration runs for a fixed number of
    steps and never evaluates a residual-based stopping criterion, so it incurs no host-device
    synchronization and is expressed entirely as matmuls. Only roots that are powers of two are
    supported, i.e. blocks of order 1, 2, and 4.

    WARNING: On rank-deficient input this regularizes more aggressively than the eigendecomposition
        path does for the same epsilon, so the two do not agree there. See relative_epsilon.

    Attributes:
        relative_epsilon (float): Floors the ridge added before the iteration at
            relative_epsilon * |A|_F. Unlike the eigendecomposition path, this iteration cannot
            stabilize a rank-deficient or indefinite matrix, and Shampoo factor matrices are
            rank-deficient early in training; without this floor the iteration diverges to NaN.
            (Default: 1e-6)
        coefficients (list[list[float]]): Per-iteration schedule of (a, b, c) coefficient triples
            for the odd polynomial p(x) = a x + b x^3 + c x^5 driving the iteration. The number
            of iterations is len(coefficients).
            (Default: Polar Express 10-step schedule from ASGO.)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend
            keeping True. The iteration is built entirely out of matmuls and must resolve
            eigenvalues down to relative_epsilon, which tf32's 10-bit mantissa cannot represent;
            unlike the factor matrix accumulation, Z @ Y is not a Gram product, so tf32 error there
            is unstructured and drives the iteration to NaN. (Default: True)

    """

    @staticmethod
    def _get_default_coefficients() -> list[list[float]]:
        return [
            [8.28721201814563, -23.595886519098837, 17.300387312530933],
            [4.107059111542203, -2.9478499167379106, 0.5448431082926601],
            [3.9486908534822946, -2.9089021159629490, 0.5518191394370137],
            [3.3184196573706015, -2.4884880243148740, 0.5100489401237200],
            [2.300652019954817, -1.6689039845747493, 0.4188073119525673],
            [1.891301407787398, -1.2679958271945868, 0.3768040894852483],
            [1.8750014808534479, -1.2500016453999487, 0.3750001645474248],
            [1.875, -1.25, 0.375],
            [1.875, -1.25, 0.375],
            [1.875, -1.25, 0.375],
        ]

    relative_epsilon: float = 1e-6
    disable_tf32: bool = True
    # TODO: Clean up coefficient definition -- consider using list[tuple[float, float, float]]
    # to enforce 3-tuples, and define defaults from the training pipeline side.
    coefficients: list[list[float]] = field(default_factory=_get_default_coefficients)

    def __post_init__(self) -> None:
        if len(self.coefficients) == 0:
            raise ValueError("coefficients must be non-empty.")
        for index, entry in enumerate(self.coefficients):
            if len(entry) != 3:
                raise ValueError(
                    f"coefficients[{index}] must contain exactly three coefficients (a, b, c) for "
                    f"p(x) = a x + b x^3 + c x^5, but {entry=} has {len(entry)}."
                )
            for coefficient in entry:
                if isinstance(coefficient, bool) or not isinstance(
                    coefficient, (int, float)
                ):
                    raise ValueError(
                        f"coefficients[{index}] must contain real numbers, but {entry=} contains "
                        f"{coefficient!r} of type {type(coefficient).__name__}."
                    )
                if not math.isfinite(coefficient):
                    raise ValueError(
                        f"coefficients[{index}] must be finite, but {entry=} contains {coefficient}."
                    )
        final_coefficients = self.coefficients[-1]
        if not math.isclose(sum(final_coefficients), 1.0):
            logger.warning(
                f"{final_coefficients=} do not sum to 1, so the Newton-Schulz iteration has no fixed "
                "point at 1 and will converge to a band around the inverse root rather than to it."
            )


@dataclass(kw_only=True)
class CoupledHigherOrderConfig(RootInvConfig):
    """Configuration for matrix root inverse via coupled higher-order method.

    Attributes:
        rel_epsilon (float): Relative epsilon for coupled higher order method. Adds epsilon * lambda_max * I to matrix
            before taking matrix root, where lambda_max is an upper bound on maximum eigenvalue.
        abs_epsilon (float): Absolute epsilon for coupled higher order method. Adds epsilon * I to matrix before taking matrix root. When both "abs_epsilon" and "rel_epsilon" are specified, max(rel_epsilon * lambda_max, abs_epsilon) * I is added to the matrix.
        max_iterations (int): Maximum number of iterations for coupled higher order method. Typically we need < 20 iterations.
            (Default: 100)
        tolerance (float): Tolerance for computing root inverse using coupled higher order method. In practice, 1e-20
            guarantees a run to convergence. (Default: 1e-8)
        order (int): Order of the method. Order must be >= 2. Higher order methods accelerate convergence (fewer iterations),
            but can take more matmuls per iteration. order=2 represents Newton's method. (Default: 3)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True,
            since tf32 is challenging numerically here. (Default: True)

    """

    rel_epsilon: float
    abs_epsilon: float
    max_iterations: int = 100
    tolerance: float = 1e-8
    order: int = 3
    disable_tf32: bool = True


@dataclass(init=False)
class OrthogonalizationConfig(MatrixFunctionConfig):
    """Configuration for matrix orthogonalization.

    If the reduced SVD of the matrix A is given by A = U S V^T, then the orthogonalized/closest orthogonal matrix is U V^T.

    Note: When using custom scale_by_dims_fn, avoid lambda functions as they may cause
    pickling issues during serialization/deserialization. Use regular named functions
    instead for better compatibility with distributed training and checkpointing.

    Attributes:
        scale_by_dims_fn (Callable[[int, int], float]): Function to scale the orthogonalized matrix by some function of the dimensions of the matrix.
            (Default: _default_scale_by_dims_fn)

    """

    @staticmethod
    def _default_scale_by_dims_fn(d_in: int, d_out: int) -> float:
        """Default scaling function that returns 1.0 (no scaling)."""
        return 1.0

    @staticmethod
    def _get_default_scale_by_dims_fn() -> Callable[[int, int], float]:
        return OrthogonalizationConfig._default_scale_by_dims_fn

    scale_by_dims_fn: Callable[[int, int], float] = field(
        default_factory=_get_default_scale_by_dims_fn
    )


@dataclass(kw_only=True)
class SVDOrthogonalizationConfig(OrthogonalizationConfig):
    """Configuration for matrix orthogonalization via reduced SVD.

    Attributes:
        scale_by_nuclear_norm (bool): Whether to scale by nuclear norm of the matrix. (Default: False)
        scale_by_dims_fn (Callable[[int, int], float]): Function to scale the orthogonalized matrix by some function of the dimensions of the matrix.
            (Default: lambda d_in, d_out: 1.0)

    """

    scale_by_nuclear_norm: bool = False


@dataclass(kw_only=True)
class NewtonSchulzOrthogonalizationConfig(OrthogonalizationConfig):
    """Configuration for matrix semi-orthogonalization via quintic Newton-Schulz iteration.

    This iteratively performs the iteration:
        X <- p(X) = a * X + b * X * X^T * X + c * (X * X^T)^2 * X.

    NOTE: In order to guarantee convergence, the coefficients must satisfy p(1) = 1.
        This is not true for the Muon coefficients, which only guarantee convergence to [0.7, 1.3].
        Another alternative is to use the coefficients (3., -16./5., 6./5.) as used in Modula.

    References:
        - https://arxiv.org/abs/2409.20325
        - https://kellerjordan.github.io/posts/muon/
        - https://docs.modula.systems/algorithms/newton-schulz/

    Attributes:
        num_iterations (int): Number of iterations for Newton-Schulz iteration. (Default: 5)
        coefficients (tuple[float, float, float]): Coefficients for Newton-Schulz iteration.
            (Default: (3.4445, -4.7750, 2.0315) based on suggestion in Muon.)
        scale_by_dims_fn (Callable[[int, int], float]): Function to scale the orthogonalized matrix by some function of the dimensions of the matrix.
            (Default: lambda d_in, d_out: 1.0)

    """

    num_iterations: int = 5
    coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


DefaultNewtonSchulzOrthogonalizationConfig = NewtonSchulzOrthogonalizationConfig()
