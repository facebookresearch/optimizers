"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import inspect
import logging
import math
import time
from dataclasses import fields
from fractions import Fraction
from functools import wraps
from math import isfinite
from typing import Any, Callable, TypeVar

import torch
from matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEigendecompositionConfig,
    DefaultPerturbationConfig,
    EigenConfig,
    EigendecompositionConfig,
    EighEigendecompositionConfig,
    PerturbationConfig,
    PseudoInverseConfig,
    QREigendecompositionConfig,
    RankDeficientStabilityConfig,
    RootInvConfig,
)

from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


@enum.unique
class NewtonConvergenceFlag(enum.Enum):
    """
    Enum class for the state of the Newton / higher-order iteration method.

    REACHED_MAX_ITERS: Reached maximum iteration count without meeting other exit criteria (rare, unexpected).
    CONVERGED: Met the tolerance criterion (expected).
    EARLY_STOP: Error in residual stopped improving (unexpected).
    """

    REACHED_MAX_ITERS = enum.auto()
    CONVERGED = enum.auto()
    EARLY_STOP = enum.auto()


FuncReturnType = TypeVar("FuncReturnType")
DataclassType = TypeVar("DataclassType")


def _get_function_args_from_config(
    func: Callable[..., FuncReturnType], config: DataclassType
) -> dict[str, Any]:
    """
    Returns a dict of arguments for func that are defined in config. Note that config is not expected to contain all arguments for func, nor are all fields in config expected to be applicable to func.
    """
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)  # type: ignore[arg-type]
        if field.name in inspect.getfullargspec(func).args
    }


def _check_square_matrix(
    func: Callable[..., FuncReturnType],
) -> Callable[..., FuncReturnType]:
    """
    Decorator to check if the input matrix is square.

    This decorator checks if the input matrix `A` is a 2-dimensional square matrix.
    If not, it raises a ValueError. If the matrix is valid, it calls the decorated function.

    Args:
        func (Callable[..., FuncReturnType]): The function to be decorated.

    Returns:
        wrapped_func (Callable[..., FuncReturnType]): The wrapped function that includes the square matrix check.

    """

    @wraps(func)
    def wrapper(A: Tensor, *args: Any, **kwargs: Any) -> FuncReturnType:
        if len(A.shape) != 2:
            raise ValueError(f"Matrix is not 2-dimensional! {A.shape=}")
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix is not square! {A.shape=}")
        return func(A, *args, **kwargs)

    return wrapper


@_check_square_matrix
def check_diagonal(A: Tensor) -> bool:
    """Checks if symmetric matrix is diagonal. Throw if the input is not a square matrix.

    Args:
        A (Tensor): The input matrix.

    Returns:
        is_diagonal (bool): True if the matrix is diagonal, False otherwise.

    Raises:
        ValueError: If the matrix is not 2-dimensional or not square.

    """
    # Check both upper triangular part and lower triangular part are all zeros.
    return not A.triu(diagonal=1).any() and not A.tril(diagonal=-1).any()


def _matrix_perturbation(
    A: Tensor,
    epsilon: float = 0.0,
    is_eigenvalues: bool = True,
) -> Tensor:
    """Add epsilon * I to matrix (if square) or epsilon (if vector).

    Args:
        A (Tensor): Matrix of interest.
        epsilon (float): Value to add to matrix for perturbation/regularization. (Default: 0.0)
        is_eigenvalues (bool): Whether A is a matrix of eigenvalues (true) or a full matrix (false). In the former case (true), add epsilon to all values; in the latter (false), add epsilon along the diagonal. (Default: True)

    Returns:
        A_ridge (Tensor): Matrix with perturbation/regularization.

    """
    return (
        A.add(torch.eye(A.shape[0], dtype=A.dtype, device=A.device), alpha=epsilon)
        if not is_eigenvalues
        else A + epsilon
    )


def stabilize_and_pow_eigenvalues(
    L: Tensor,
    root: Fraction,
    epsilon: float = 0.0,
    rank_deficient_stability_config: RankDeficientStabilityConfig = DefaultPerturbationConfig,
) -> Tensor:
    """
    Stabilize the eigenvalues of a matrix and raise them to a negative fractional power.

    If using epsilon (i.e. rank_deficient_stability_config is a PerturbationConfig), stabilization entails adding epsilon to the eigenvalues, i.e. regularization. See _matrix_perturbation() and PerturbationConfig for details.

    If using pseudo-inverse (i.e. rank_deficient_stability_config is a PseudoInverseConfig), stabilization entails ignoring all eigenvalues sufficiently close to zero as determined by some cutoff. See truncate_eigenvalues_cutoff() and PseudoInverseConfig for details.

    Args:
        L (Tensor): The input matrix.
        root (Fraction): The fractional power to which the eigenvalues should be raised.
        epsilon (float): A small value added to the eigenvalues for stability. (Default: 0.0)
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)

    Returns:
        inv_power_L (Tensor): The resulting matrix with stabilized and powered eigenvalues.

    Raises:
        ValueError: If epsilon is not 0.0 when using pseudo-inverse.
        ValueError: If rank_deficient_stability_config is not a supported config type.

    """

    def truncate_eigenvalues_cutoff(
        L: Tensor,
        rank_rtol: float | None = None,
        rank_atol: float = 0.0,
    ) -> float:
        """Filter the eigenvalues based on the numerical rank of the matrix. The procedure below mimics the steps described in the documentation of https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html.

        Args:
            L (Tensor): Eigenvalues of matrix.
            rank_rtol (float | None): Relative tolerance for determining numerical rank of matrix. (Default: None)
            rank_atol (float): Absolute tolerance for determining numerical rank of matrix. (Default: 0.0)

        Returns:
            spectrum_cutoff (float): Cutoff to filter out eigenvalues.
        """
        if rank_rtol is None:
            rtol = L.numel() * torch.finfo(L.dtype).eps
        else:
            rtol = rank_rtol
        return max(rank_atol, rtol * L.max().relu().item())

    match rank_deficient_stability_config:
        case PseudoInverseConfig():
            if epsilon != 0.0:
                raise ValueError(f"{epsilon=} should be 0.0 when using pseudo-inverse!")

            spectrum_cutoff = truncate_eigenvalues_cutoff(
                L=L,
                rank_rtol=rank_deficient_stability_config.rank_rtol,
                rank_atol=rank_deficient_stability_config.rank_atol,
            )
            inv_power_L = torch.where(
                L <= spectrum_cutoff,
                torch.zeros_like(L),
                L.pow(-1.0 / root),
            )
        case PerturbationConfig():
            lambda_min = torch.min(L).item()

            # make eigenvalues > 0 (if necessary)
            effective_epsilon = (
                -min(lambda_min - epsilon, 0.0)
                if rank_deficient_stability_config.perturb_before_computation
                else (-min(lambda_min, 0.0) + epsilon)
            )
            L = _matrix_perturbation(
                A=L, epsilon=effective_epsilon, is_eigenvalues=True
            )

            inv_power_L = L.pow_(-1.0 / root)
        case _:
            raise NotImplementedError(
                f"{rank_deficient_stability_config=} is not supported."
            )

    return inv_power_L


@_check_square_matrix
def matrix_inverse_root(
    A: Tensor,
    root: Fraction,
    root_inv_config: RootInvConfig = DefaultEigenConfig,
    epsilon: float = 0.0,
    is_diagonal: bool = False,
) -> Tensor:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (Fraction): Root of interest. Any rational number.
        root_inv_config (RootInvConfig): Configuration for root inverse computation. (Default: DefaultEigenConfig)
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        is_diagonal (bool): Flag for whether or not matrix is diagonal. If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)

    Returns:
        X (Tensor): Inverse root of matrix A.

    Raises:
        ValueError: If the matrix is not 2-dimensional or not square, or if the root denominator is not 1 for CoupledNewtonConfig.
        NotImplementedError: If the root inverse config is not implemented.

    """
    if is_diagonal:
        return _matrix_inverse_root_diagonal(
            A=A,
            root=root,
            epsilon=epsilon,
        )

    match root_inv_config:
        case EigenConfig():
            X, _, _ = _matrix_inverse_root_eigen(
                A=A,
                root=root,
                epsilon=epsilon,
                **_get_function_args_from_config(
                    _matrix_inverse_root_eigen, root_inv_config
                ),
            )
        case CoupledNewtonConfig():
            # NOTE: Use Fraction.is_integer() instead when downstream applications are Python 3.12+ available
            if root.denominator != 1:
                raise ValueError(
                    f"{root.denominator=} must be equal to 1 to use coupled inverse Newton iteration!"
                )

            X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
                A=A,
                root=root.numerator,
                epsilon=epsilon,
                **_get_function_args_from_config(
                    _matrix_inverse_root_newton, root_inv_config
                ),
            )
            if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
                logging.warning(
                    "Newton did not converge and reached maximum number of iterations!"
                )
        case CoupledHigherOrderConfig():
            X, _, termination_flag, _, _ = _matrix_inverse_root_higher_order(
                A=A,
                root=root,
                **_get_function_args_from_config(
                    _matrix_inverse_root_higher_order, root_inv_config
                ),
            )
            if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
                logging.warning(
                    "Higher order method did not converge and reached maximum number of iterations!"
                )
        case _:
            raise NotImplementedError(
                f"Root inverse config is not implemented! Specified root inverse config is {root_inv_config=}."
            )

    return X


def _matrix_inverse_root_diagonal(
    A: Tensor,
    root: Fraction,
    epsilon: float = 0.0,
) -> Tensor:
    """Computes matrix inverse root for a diagonal matrix by taking inverse square root of diagonal entries.

    Args:
        A (Tensor): A diagonal matrix.
        root (Fraction): Root of interest. Any rational number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)

    Returns:
        X (Tensor): Inverse root of diagonal entries.

    Raises:
        ValueError: If the root is not a positive integer.

    """
    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    return torch.diag(
        stabilize_and_pow_eigenvalues(torch.diagonal(A), root=root, epsilon=epsilon)
    )


@_check_square_matrix
def matrix_eigendecomposition(
    A: Tensor,
    epsilon: float = 0.0,
    eigendecomposition_config: EigendecompositionConfig = DefaultEigendecompositionConfig,
    is_diagonal: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute the eigendecomposition of a symmetric matrix.

    Args:
        A (Tensor): The input symmetric matrix.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root for numerical stability. (Default: 0.0)
        eigendecomposition_config (EigendecompositionConfig): Determines how eigendecomposition is computed. (Default: DefaultEigendecompositionConfig)
        is_diagonal (bool): Whether A is diagonal. (Default: False)

    Returns:
        eigenvalues (Tensor): The eigenvalues of the input matrix.
        eigenvectors (Tensor): The eigenvectors of the input matrix.

    Raises:
        ValueError: If the matrix is not 2-dimensional or not square.
        ValueError: If epsilon is 0.0 when using pseudo-inverse.
        NotImplementedError: If the eigendecomposition config is not implemented.

    """
    # Return the (sorted) diagonal of A and identity matrix if A is diagonal.
    if is_diagonal:
        return A.diag(), torch.eye(
            A.shape[0],
            dtype=A.dtype,
            device=A.device,
        )

    # TODO: reduce redundant code when rank_deficient_stability_config is generalized to all methods
    # check epsilon is 0 when using pseudo-inverse
    if (
        isinstance(
            eigendecomposition_config.rank_deficient_stability_config,
            PseudoInverseConfig,
        )
        and epsilon != 0.0
    ):
        raise ValueError(f"{epsilon=} should be 0.0 when using pseudo-inverse!")

    # Add epsilon to the diagonal to help with numerical stability of the eigenvalue decomposition
    # Only do it when damp_before_computation is True (root_inv_config must be a DampingConfig)
    if (
        isinstance(
            eigendecomposition_config.rank_deficient_stability_config,
            PerturbationConfig,
        )
        and eigendecomposition_config.rank_deficient_stability_config.perturb_before_computation
    ):
        A_ridge = _matrix_perturbation(A, epsilon=epsilon, is_eigenvalues=False)
    else:
        A_ridge = A

    match eigendecomposition_config:
        case EighEigendecompositionConfig():
            return _eigh_eigenvalue_decomposition(
                A_ridge,
                **_get_function_args_from_config(
                    _eigh_eigenvalue_decomposition, eigendecomposition_config
                ),
            )
        case QREigendecompositionConfig():
            return _qr_algorithm(
                A_ridge,
                **_get_function_args_from_config(
                    _qr_algorithm, eigendecomposition_config
                ),
            )
        case _:
            raise NotImplementedError(
                f"Eigendecomposition config is not implemented! Specified eigendecomposition config is {eigendecomposition_config=}."
            )


def _eigh_eigenvalue_decomposition(
    A: Tensor,
    retry_double_precision: bool = True,
    eigendecomposition_offload_device: torch.device | str = "",
) -> tuple[Tensor, Tensor]:
    """Compute the eigendecomposition of a symmetric matrix using torch.linalg.eigh.

    Args:
        A (Tensor): The input symmetric matrix.
        retry_double_precision (bool): Whether to retry the computation in double precision if it fails in the current precision. (Default: True)
        eigendecomposition_offload_device (torch.device | str): Device to offload eigendecomposition computation. If value is empty string, do not perform offloading. (Default: "")

    Returns:
        eigenvalues (Tensor): The eigenvalues of the input matrix A.
        eigenvectors (Tensor): The eigenvectors of the input matrix A.

    Raises:
        Exception: If the eigendecomposition fails and retry_double_precision is False or fails in double precision.

    """

    current_device = A.device
    if eigendecomposition_offload_device != "":
        A = A.to(device=eigendecomposition_offload_device)

    try:
        # Attempt to compute the eigendecomposition in the current precision
        L, Q = torch.linalg.eigh(A)

    except Exception as exception:
        # If the computation fails and retry_double_precision is True, retry in double precision
        if retry_double_precision and A.dtype != torch.float64:
            logger.warning(
                f"Failed to compute eigendecomposition in {A.dtype} precision with exception {exception}! Retrying in double precision..."
            )
            L, Q = torch.linalg.eigh(A.double())
        else:
            # If retry_double_precision is False or the computation fails in double precision, raise the exception
            raise exception

    return L.to(device=current_device), Q.to(device=current_device, dtype=A.dtype)


def _estimated_eigenvalues_criterion_below_or_equal_tolerance(
    estimated_eigenvalues: Tensor, tolerance: float
) -> bool:
    """Evaluates if a criterion using estimated eigenvalues is below or equal to the tolerance.

    Let Q^T A Q =: B be the estimate of the eigenvalues of the matrix A, where Q is the matrix containing the last computed eigenvectors.
    The criterion based on the estimated eigenvalues is defined as ||B - diag(B)||_F <= tolerance * ||B||_F.
    The tolerance hyperparameter should therefore be in the interval [0.0, 1.0].

    This convergence criterion can be motivated by considering A' = Q diag(B) Q^T as an approximation of A.
    We have ||A - A'||_F = ||A - Q diag(B) Q^T||_F = ||Q^T A Q - diag(B)||_F = ||B - diag(B)||_F.
    Moreover, we have ||B||_F = ||Q^T A Q||_F = ||A||_F.
    Hence, the two relative errors are also equivalent: ||A - A'||_F / ||A||_F = ||B - diag(B)||_F / ||B||_F.

    Args:
        estimated_eigenvalues (Tensor): The estimated eigenvalues.
        tolerance (float): The tolerance for the criterion.

    Returns:
        is_below_tolerance (bool): True if the criterion is below or equal to the tolerance, False otherwise.

    """
    norm = torch.linalg.norm(estimated_eigenvalues)
    diagonal_norm = torch.linalg.norm(estimated_eigenvalues.diag())
    off_diagonal_norm = torch.sqrt(norm**2 - diagonal_norm**2)
    return bool(off_diagonal_norm <= tolerance * norm)


def _qr_algorithm(
    A: Tensor,
    eigenvectors_estimate: Tensor,
    max_iterations: int = 1,
    tolerance: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Approximately compute the eigendecomposition of a symmetric matrix by performing the QR algorithm.

    Given an initial estimate of the eigenvectors Q of matrix A, a power iteration and a QR decomposition is performed each iteration, i.e. Q, _ <- QR(A @ Q).
    When the initial estimate is the zero matrix, the eigendecomposition is computed using _eigh_eigenvalue_decomposition.

    Note that if the criterion based on the estimated eigenvalues is already below or equal to the tolerance given the initial eigenvectors_estimate, the QR iterations will be skipped.

    Args:
        A (Tensor): The symmetric input matrix.
        eigenvectors_estimate (Tensor): The current estimate of the eigenvectors of A.
        max_iterations (int): The maximum number of iterations to perform. (Default: 1)
        tolerance (float): The tolerance for determining convergence in terms of the norm of the off-diagonal elements of the eigenvalue estimate.
            (Default: 0.01)

    Returns:
        estimated_eigenvalues (Tensor): The estimated eigenvalues of the input matrix A.
        estimated_eigenvectors (Tensor): The estimated eigenvectors of the input matrix A.

    Raises:
        AssertionError: If the data types of Q and A do not match.

    """
    if not eigenvectors_estimate.any():
        return _eigh_eigenvalue_decomposition(A)

    # Perform orthogonal/simultaneous iterations (QR algorithm).
    Q = eigenvectors_estimate

    # This assertion provides a more clear error message than the internal error message in `torch.mm`, and assertion makes sure that user-side is unable to catch the error.
    assert (
        Q.dtype == A.dtype
    ), f"Q and A must have the same dtype! {Q.dtype=} {A.dtype=}"

    estimated_eigenvalues = Q.T @ A @ Q
    iteration = 0
    # NOTE: This will skip the QR iterations if the criterion is already below or equal to the tolerance given the initial eigenvectors_estimate.
    while (
        iteration < max_iterations
        and not _estimated_eigenvalues_criterion_below_or_equal_tolerance(
            estimated_eigenvalues, tolerance
        )
    ):
        power_iteration = A @ Q
        Q = torch.linalg.qr(power_iteration).Q
        iteration += 1
        estimated_eigenvalues = Q.T @ A @ Q

    # Ensure consistent ordering of estimated eigenvalues and eigenvectors.
    estimated_eigenvalues, indices = estimated_eigenvalues.diag().sort(stable=True)
    Q = Q[:, indices]

    return estimated_eigenvalues, Q


def _matrix_inverse_root_eigen(
    A: Tensor,
    root: Fraction,
    epsilon: float = 0.0,
    rank_deficient_stability_config: RankDeficientStabilityConfig = DefaultPerturbationConfig,
    retry_double_precision: bool = True,
    eigendecomposition_offload_device: torch.device | str = "",
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute matrix inverse root using eigendecomposition of symmetric positive (semi-)definite matrix.

            A^{-1/r} = Q L^{-1/r} Q^T

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        root (Fraction): Root of interest. Any rational number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        rank_deficient_stability_config (RankDeficientStabilityConfig): Configuration for handling/stabilizing rank-deficient matrices. (Default: DefaultPerturbationConfig)
        retry_double_precision (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        eigendecomposition_offload_device (torch.device | str): Device to offload eigendecomposition computation. If value is empty string, do not perform offloading. (Default: "")

    Returns:
        X (Tensor): (Inverse) root of matrix. Same dimensions as A.
        L (Tensor): Eigenvalues of A.
        Q (Tensor): Orthogonal matrix consisting of eigenvectors of A.

    Raises:
        ValueError: If the root is not a positive integer.
        ValueError: If epsilon is 0.0 when using pseudo-inverse.

    """

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # TODO: reduce redundant code when rank_deficient_stability_config is generalized to all methods
    # check epsilon is 0 when using pseudo-inverse
    if (
        isinstance(
            rank_deficient_stability_config,
            PseudoInverseConfig,
        )
        and epsilon != 0.0
    ):
        raise ValueError(f"{epsilon=} should be 0.0 when using pseudo-inverse!")

    # Add epsilon to the diagonal to help with numerical stability of the eigenvalue decomposition
    # Only do it when damp_before_computation is True (root_inv_config must be a DampingConfig)
    if (
        isinstance(rank_deficient_stability_config, PerturbationConfig)
        and rank_deficient_stability_config.perturb_before_computation
    ):
        A_ridge = _matrix_perturbation(A, epsilon=epsilon, is_eigenvalues=False)
    else:
        A_ridge = A

    # compute eigendecomposition and compute minimum eigenvalue
    L, Q = _eigh_eigenvalue_decomposition(
        A_ridge,
        retry_double_precision=retry_double_precision,
        eigendecomposition_offload_device=eigendecomposition_offload_device,
    )

    inv_power_L = stabilize_and_pow_eigenvalues(
        L,
        root,
        epsilon=epsilon,
        rank_deficient_stability_config=rank_deficient_stability_config,
    )

    # compute the matrix inverse root
    X = Q * inv_power_L.unsqueeze(0) @ Q.T

    return X, L, Q


def _matrix_inverse_root_newton(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
    """Compute matrix inverse root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    NOTE: Exponent multiplier not compatible with coupled inverse Newton iteration!

    Args:
        A (Tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        max_iterations (int): Maximum number of iterations. (Default: 100)
        tolerance (float): Tolerance. (Default: 1e-6)

    Returns:
        A_root (Tensor): Inverse square root of matrix.
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error between M and I.

    """

    # initialize iteration, dimension, and alpha
    iteration = 0
    dim = A.shape[0]
    alpha = -1 / root
    identity = torch.eye(dim, dtype=A.dtype, device=A.device)

    # add regularization
    A_ridge = A.add(identity, alpha=epsilon)

    # initialize matrices
    A_nrm = torch.linalg.norm(A_ridge)
    z = (root + 1) / (2 * A_nrm)
    X = z ** (-alpha) * identity
    M = z * A_ridge
    error = torch.dist(M, identity, p=torch.inf)

    # main for loop
    while error > tolerance and iteration < max_iterations:
        iteration += 1
        M_p = M.mul(alpha).add_(identity, alpha=(1 - alpha))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, root) @ M
        error = torch.dist(M, identity, p=torch.inf)

    # determine convergence flag
    termination_flag = (
        NewtonConvergenceFlag.CONVERGED
        if error <= tolerance
        else NewtonConvergenceFlag.REACHED_MAX_ITERS
    )

    return X, M, termination_flag, iteration, error


def _matrix_inverse_root_higher_order(
    A: Tensor,
    root: Fraction,
    rel_epsilon: float = 0.0,
    abs_epsilon: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-20,
    order: int = 3,  # 2 represents Newton's method
    disable_tf32: bool = True,
) -> tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
    """Compute matrix inverse root using coupled iterations, similar to above but generalized to support higher order.

        Rough sketch (at order = 2, i.e., Newton)

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (k |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (k * |A|_F).
    Here, k > 1, and typically lies in [1, 2]. It is picked internally in this method.

    NOTE: Exponent multiplier not compatible with coupled iterations!

    Args:
        A (Tensor): Matrix of interest.
        root (Fraction): Root of interest. Any rational number. Use small numerator, denominator for best numerics as well as performance.
        rel_epsilon (float): Adds epsilon * lambda_max * I to matrix before taking matrix root, where lambda_max is an upper bound on maximum eigenvalue. (Default: 0.0)
        abs_epsilon (float): Adds epsilon * I to matrix before taking matrix root. When both "abs_epsilon" and "rel_epsilon" are specified, max(rel_epsilon * lambda_max, abs_epsilon) * I is added to the matrix.
            Generally recommend setting according to A.dtype (1e-3 for tf32, 1e-5 for fp32, 1e-9 for fp64) (Default: 0.0)
        max_iterations (int): Maximum number of iterations. Typically we need < 20 iterations. (Default: 100)
        tolerance (float): Tolerance for determining exit criterion from iterations. (Default: 1e-20, which in practice guarantees they run to convergence)
        order (int): Order of the method. Order must be >= 2.  Higher order methods accelerate convergence (fewer iterations), but can take more matmuls per iteration. (Default: 3)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True, since tf32 is challenging numerically here. (Default: True)

    Returns:
        A_root (Tensor): Inverse root of matrix (A^{-1/root}).
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error, measured as |A * A_root^(p/q) - I|_Inf, where root = -q/p.

    Raises:
        ArithmeticError: If the computed result is inaccurate, i.e., error > 1e-1 or if there is an internal error.
        ArithmeticError: If the input matrix has entries close to infinity.
        ArithmeticError: If NaN/Inf is found in the matrix inverse root after powering for fractions.

    """

    tf32_flag = torch.backends.cuda.matmul.allow_tf32
    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
    logger.debug(
        f"Using tf32 precision for fp32 matmul: {torch.backends.cuda.matmul.allow_tf32}"
    )

    try:
        t_iter_begin = time.perf_counter()
        p = root.numerator
        q = root.denominator
        dtype = A.dtype

        if min(abs(p), abs(q)) >= 10:
            logger.warning(
                f"{abs(root.numerator)=} and {abs(root.denominator)=} are probably too big for best performance."
            )

        # develop the b coefficients array first (ref: Lakic's paper)
        b = torch.zeros(order, dtype=A.dtype, device=A.device)
        b[0] = 1
        num = 1
        denom = 1
        for i in range(1, order):
            num *= 1 + (i - 1) * p
            denom *= i * p
            b[i] = num / denom

        # initialize iteration, dimension, and s
        iteration = 0
        n = A.shape[0]
        s = -1 / p

        # We add a diagonal term to condition the matrix better
        # We follow the Google style conditioning (in spirit) and scale by an upper bound on the max eigenvalue
        # NOTE: this is different from other parts of Shampoo for now
        # Simply use the basic upper bound on the spectral radius of A via infinity norm (should not underflow)
        # NOTE: One may wish to use a cheap (|A^4|_inf)**0.25 to get a tighter upper bound, but beware of fp32 underflow!
        lambda_max_approx = torch.linalg.matrix_norm(A, torch.inf)

        # We have not seen lambda_max being Inf in practice, however there is not a whole lot we can do in this pathological case and its good to bail early
        if not isfinite(lambda_max_approx):
            raise ArithmeticError(
                "Input matrix has entries close to inf, exiting root inverse"
            )

        # Now scale and setup our variables
        epsilon = max(rel_epsilon * lambda_max_approx, abs_epsilon)
        identity = torch.eye(n, dtype=dtype, device=A.device)
        A_ridge = torch.add(A, identity, alpha=epsilon)
        lambda_max_approx += epsilon

        # Figure out a constant that gives good starting location
        # We stick to a conservative setting that gives very good accuracy
        # For a ref, see https://github.com/google-research/google-research/blob/master/scalable_shampoo/pytorch/matrix_functions.py#L114
        z = 1.0 / torch.trace(A_ridge).item()
        X = (z ** (-s)) * identity
        M = z * A_ridge
        error = torch.linalg.vector_norm(M - identity, torch.inf)
        t_iter_end = time.perf_counter()
        logger.debug(
            f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
        )

        # Do one iteration of basic Newton first. This is used to mathematically guarantee convergence of higher order method.
        # TODO: we may be able to get rid of this with a more careful analysis of the convergence region
        t_iter_begin = time.perf_counter()
        M_p = M.mul(s).add_(identity, alpha=(1 - s))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, p) @ M
        error = torch.linalg.vector_norm(M - identity, torch.inf)
        n_matmul = math.ceil(math.log2(p)) + 2
        iteration += 1
        t_iter_end = time.perf_counter()
        logger.debug(
            f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
        )

        # main while loop
        while error > tolerance and iteration < max_iterations:
            t_iter_begin = time.perf_counter()
            iteration += 1

            # create M_p via Horner's rule
            base_matrix = identity - M
            M_p = base_matrix.mul(b[order - 1]).add_(
                identity, alpha=float(b[order - 2])
            )
            for i in reversed(range(order - 2)):
                M_p = torch.addmm(identity, M_p, base_matrix, beta=float(b[i]))

            # rest is same as Newton
            X = X @ M_p
            M = torch.linalg.matrix_power(M_p, p) @ M
            new_error = torch.linalg.vector_norm(M - identity, torch.inf)
            n_matmul += math.ceil(math.log2(p)) + order

            # TODO: 1.2 is the value from the Google code, can be tuned
            if new_error > error * 1.2 or (new_error == error and error < 1e-3):
                logger.debug(
                    f"Coupled inverse Newton is stagnating or diverging based on comparing current error {new_error.item()} against last iteration's error {error.item()}."
                    f"(We assume divergence if the new error > 1.2 * previous error, and assume stagnation if they are equal.)"
                )
                termination_flag = NewtonConvergenceFlag.EARLY_STOP
                break
            error = new_error

            t_iter_end = time.perf_counter()
            logger.debug(
                f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
            )
        else:
            # determine convergence flag based on error and tolerance because the main while loop exited with False condition.
            termination_flag = (
                NewtonConvergenceFlag.REACHED_MAX_ITERS
                if error > tolerance
                else NewtonConvergenceFlag.CONVERGED
            )

        # compute a cheap error proxy
        true_error = torch.linalg.vector_norm(
            A_ridge @ torch.linalg.matrix_power(X, p) - identity, torch.inf
        )
        n_matmul += math.ceil(math.log2(p)) + 1

        # If the error is too high, let us log and raise an exception for investigation. This should be relatively infrequent (if epsilon isn't too small)
        if true_error > 1e-1:
            raise ArithmeticError(
                f"Error in matrix inverse root (before powering for fractions) {true_error} exceeds threshold 1e-1, raising an exception!"
            )

        # Now power the root to q
        if q > 1:
            X = torch.linalg.matrix_power(X, q)
            n_matmul += math.ceil(math.log2(q))

        logger.debug(f"Upper bound on maximum eigenvalue: {lambda_max_approx}")
        logger.debug(f"Number of matmuls: {n_matmul}")
        logger.debug(f"Number of iterations: {iteration}")
        logger.debug(f"Error before powering: {true_error}")
        logger.debug(f"Termination Flag: {termination_flag}")

        # If we have inf/nan in our answer also raise an arithmetic exception.
        # Usually, this is due to the powering to q > 1 which can blow up entries.
        # We have not seen this yet for q = 1 in Shampoo.
        if not torch.isfinite(X).all():
            raise ArithmeticError(
                "NaN/Inf in matrix inverse root (after powering for fractions), raising an exception!"
            )

    finally:
        # Make sure we restore tf32 mode correctly before returning
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = tf32_flag

    return X, M, termination_flag, iteration, true_error


@_check_square_matrix
def compute_matrix_root_inverse_residuals(
    A: Tensor,
    X_hat: Tensor,
    root: Fraction,
    epsilon: float,
    root_inv_config: RootInvConfig = DefaultEigenConfig,
) -> tuple[Tensor, Tensor]:
    """Compute residual of matrix root inverse for debugging purposes.

        relative error    = ||X - X_hat||_inf / ||X||_inf
        relative residual = ||A X^r - I||_inf

    Args:
        A (Tensor): Matrix of interest.
        X_hat (Tensor): Computed matrix root inverse.
        root (Fraction): Root of interest. Any rational number.
        epsilon (float): Adds epsilon * I to matrix.
        root_inv_config (RootInvConfig): Configuration for root inverse computation (only supports EigenConfig for now). (Default: DefaultEigenConfig)

    Returns:
        relative_error (Tensor): Relative error of matrix root inverse.
        relative_residual (Tensor): Residual of matrix root inverse.

    Raises:
        AssertionError: If the root_inv_config is not of type EigenConfig.
        ValueError: If the matrix is not 2-dimensional, not square, or if the shapes of A and X_hat do not match.

    """
    # only do root inverse residual computation for EigenConfig
    assert (
        type(root_inv_config) is EigenConfig
    ), f"Only EigenConfig is supported for compute_matrix_root_inverse_residuals; currently {root_inv_config=}."

    # check shape of matrix
    if A.shape != X_hat.shape:
        raise ValueError("Matrix shapes do not match!")

    # compute error by comparing against double precision
    X = matrix_inverse_root(
        A.double(),
        root,
        root_inv_config=root_inv_config,
        epsilon=epsilon,
    )
    relative_error = torch.dist(X, X_hat, p=torch.inf) / torch.norm(X, p=torch.inf)

    # compute residual
    X_invr, _, _ = _matrix_inverse_root_eigen(
        X_hat.double(),
        root=root,
        epsilon=0.0,
        rank_deficient_stability_config=root_inv_config.rank_deficient_stability_config,
        retry_double_precision=root_inv_config.retry_double_precision,
        eigendecomposition_offload_device=root_inv_config.eigendecomposition_offload_device,
    )

    A_reg = A.double() + epsilon * torch.eye(
        A.shape[0], dtype=torch.float64, device=A.device
    )
    relative_residual = torch.dist(X_invr, A_reg, p=torch.inf) / torch.norm(
        A_reg, p=torch.inf
    )

    return relative_error, relative_residual
