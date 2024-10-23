"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
import math
import time
from dataclasses import asdict
from fractions import Fraction
from math import isfinite
from typing import Tuple, Union

import torch
from matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigenConfig,
    DefaultEighEigenvalueCorrectionConfig,
    EigenConfig,
    EigenvalueCorrectionConfig,
    EighEigenvalueCorrectionConfig,
    RootInvConfig,
)

from torch import Tensor

logger: logging.Logger = logging.getLogger(__name__)


class NewtonConvergenceFlag(enum.Enum):
    """
    Enum class for the state of the Newton / higher-order iteration method.

    REACHED_MAX_ITERS: Reached maximum iteration count without meeting other exit criteria (rare, unexpected).
    CONVERGED: Met the tolerance criterion (expected).
    EARLY_STOP: Error in residual stopped improving (unexpected).
    """

    REACHED_MAX_ITERS = 0
    CONVERGED = 1
    EARLY_STOP = 2


def check_diagonal(A: Tensor) -> bool:
    """Checks if symmetric matrix is diagonal. Throw if the input is not a square matrix."""

    A_shape = A.shape
    if len(A_shape) != 2:
        raise ValueError(f"Matrix is not 2-dimensional! {A_shape=}")

    if A_shape[0] != A_shape[1]:
        raise ValueError(f"Matrix is not square! {A_shape=}")

    # Check both upper triangular part and lower triangular part are all zeros.
    return not A.triu(diagonal=1).any() and not A.tril(diagonal=-1).any()


def matrix_inverse_root(
    A: Tensor,
    root: Fraction,
    root_inv_config: RootInvConfig = DefaultEigenConfig,
    epsilon: float = 0.0,
    is_diagonal: Union[Tensor, bool] = False,
) -> Tensor:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (Fraction): Root of interest. Any rational number.
        root_inv_config (RootInvConfig): Configuration for root inverse computation. (Default: DefaultEigenConfig)
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        is_diagonal (Tensor, bool): Flag for whether or not matrix is diagonal. If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)

    Returns:
        X (Tensor): Inverse root of matrix A.

    """

    # check if matrix is scalar
    if torch.numel(A) == 1:
        return (A + epsilon) ** torch.as_tensor(-1.0 / root)

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    if is_diagonal:
        X = _matrix_inverse_root_diagonal(
            A=A,
            root=root,
            epsilon=epsilon,
        )
    elif type(root_inv_config) is EigenConfig:
        X, _, _ = _matrix_inverse_root_eigen(
            A=A,
            root=root,
            epsilon=epsilon,
            make_positive_semidefinite=root_inv_config.make_positive_semidefinite,
            retry_double_precision=root_inv_config.retry_double_precision,
        )
    elif type(root_inv_config) is CoupledNewtonConfig:
        # NOTE: Use Fraction.is_integer() instead when Python 3.12+ is available
        if root.denominator != 1:
            raise ValueError(
                f"{root.denominator=} must be equal to 1 to use coupled inverse Newton iteration!"
            )

        X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
            A=A,
            root=root.numerator,
            epsilon=epsilon,
            **asdict(root_inv_config),
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning(
                "Newton did not converge and reached maximum number of iterations!"
            )
    elif type(root_inv_config) is CoupledHigherOrderConfig:
        X, _, termination_flag, _, _ = _matrix_inverse_root_higher_order(
            A=A,
            root=root,
            abs_epsilon=epsilon,
            **asdict(root_inv_config),
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning(
                "Higher order method did not converge and reached maximum number of iterations!"
            )
    else:
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

    """
    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    return torch.diag((torch.diagonal(A) + epsilon).pow(torch.as_tensor(-1.0 / root)))


def _compute_eigenvalue_decomposition(
    A: Tensor,
    retry_double_precision: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute the eigendecomposition of a symmetric matrix.

    Args:
        A (Tensor): The input symmetric matrix.
        retry_double_precision (bool, optional): Whether to retry the computation in double precision if it fails in the current precision. Defaults to True.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the eigenvalues and eigenvectors of the input matrix.
    """
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

    return L, Q


def _matrix_inverse_root_eigen(
    A: Tensor,
    root: Fraction,
    epsilon: float = 0.0,
    make_positive_semidefinite: bool = True,
    retry_double_precision: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute matrix inverse root using eigendecomposition of symmetric positive (semi-)definite matrix.

            A^{-1/r} = Q L^{-1/r} Q^T

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        root (Fraction): Root of interest. Any rational number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        make_positive_semidefinite (bool): Perturbs matrix eigenvalues to ensure it is numerically positive semi-definite. (Default: True)
        retry_double_precision (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due
            to CuSOLVER failure. (Default: True)

    Returns:
        X (Tensor): (Inverse) root of matrix. Same dimensions as A.
        L (Tensor): Eigenvalues of A.
        Q (Tensor): Orthogonal matrix consisting of eigenvectors of A.

    """

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # compute eigendecomposition and compute minimum eigenvalue
    L, Q = _compute_eigenvalue_decomposition(
        A, retry_double_precision=retry_double_precision
    )

    lambda_min = torch.min(L)

    # make eigenvalues >= 0 (if necessary)
    if make_positive_semidefinite:
        L += -torch.minimum(lambda_min, torch.as_tensor(0.0))

    # add epsilon
    L += epsilon

    # compute inverse preconditioner
    X = Q * L.pow(torch.as_tensor(-1.0 / root)).unsqueeze(0) @ Q.T

    return X, L, Q


def _matrix_inverse_root_newton(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
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
        max_iterations (int): Maximum number of iterations. (Default: 1000)
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
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
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
        rel_epsilon (float): Adds epsilon * lambda_max * I to matrix before taking matrix root, where lambda_max is an upper bound on maximum eigenvalue.
        abs_epsilon (float): Adds epsilon * I to matrix before taking matrix root. When both "abs_epsilon" and "rel_epsilon" are specified, max(rel_epsilon * lambda_max, abs_epsilon) * I is added to the matrix.
        Generally recommend setting according to A.dtype (1e-3 for tf32, 1e-5 for fp32, 1e-9 for fp64) (Default: 0.0)
        max_iterations (int): Maximum number of iterations. Typically we need < 20 iterations. (Default: 100)
        tolerance (float): Tolerance for determining exit criterion from iterations. (Default: 1e-20, which in practice guarantees they run to convergence)
        order (int): Order of the method. Order must be >= 2.  Higher order methods accelerate convergence (fewer iterations), but can take more matmuls per iteration. (Default: 3)
        disable_tf32 (bool): Whether to disable tf32 matmuls or not internally. Highly recommend keeping True, since tf32 is challenging numerically here. (Default: True)

    Returns:
        A_root (Tensor): Inverse root of matrix (A^{-1/root})
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error, measured as |A * A_root^(p/q) - I|_Inf, where root = -q/p.

    Exceptions:
        Method throws an ArithmeticError if the computed result is inaccurate, i.e., error > 1e-1 or if there is an internal error

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
        if torch.isnan(X).any() or torch.isinf(X).any():
            raise ArithmeticError(
                "NaN/Inf in matrix inverse root (after powering for fractions), raising an exception!"
            )

    finally:
        # Make sure we restore tf32 mode correctly before returning
        if disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = tf32_flag

    return X, M, termination_flag, iteration, true_error


def compute_matrix_root_inverse_residuals(
    A: Tensor,
    X_hat: Tensor,
    root: Fraction,
    epsilon: float,
    root_inv_config: RootInvConfig = DefaultEigenConfig,
) -> Tuple[Tensor, Tensor]:
    """Compute residual of matrix root inverse for debugging purposes.

        relative error    = ||X - X_hat||_inf / ||X||_inf
        relative residual = ||A X^r - I||_inf

    Args:
        A (Tensor): Matrix of interest.
        X (Tensor): Computed matrix root inverse.
        root (Fraction): Root of interest. Any rational number.
        epsilon (float): Adds epsilon * I to matrix.
        root_inv_config (RootInvConfig): Configuration for root inverse computation (only supports EigenConfig for now). (Default: DefaultEigenConfig)

    Returns:
        absolute_error (Tensor): absolute error of matrix root inverse
        relative_error (Tensor): relative error of matrix root inverse
        residual (Tensor): residual of matrix root inverse

    """
    # only do root inverse residual computation for EigenConfig
    assert (
        type(root_inv_config) is EigenConfig
    ), f"Only EigenConfig is supported for compute_matrix_root_inverse_residuals; currently {root_inv_config=}."

    # check shape of matrix
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")
    elif A.shape != X_hat.shape:
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
        make_positive_semidefinite=True,
    )

    A_reg = A.double() + epsilon * torch.eye(
        A.shape[0], dtype=torch.float64, device=A.device
    )
    relative_residual = torch.dist(X_invr, A_reg, p=torch.inf) / torch.norm(
        A_reg, p=torch.inf
    )

    return relative_error, relative_residual


def matrix_eigenvectors(
    A: Tensor,
    eigenvector_computation_config: EigenvalueCorrectionConfig = DefaultEighEigenvalueCorrectionConfig,
    is_diagonal: Tensor | bool = False,
) -> Tensor:
    """Compute eigenvectors of matrix using eigendecomposition of symmetric positive (semi-)definite matrix.

            A = Q L Q^T => Q

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        eigenvector_computation_config (EigenvalueCorrectionConfig): Determines how eigenvectors are computed.
            (Default: DefaultEighEigenvalueCorrectionConfig)
        is_diagonal (Tensor | bool): Whether A is diagonal. (Default: False)

    Returns:
        Q (Tensor): Orthogonal matrix containing eigenvectors of A.

    """
    # check if matrix is scalar
    if torch.numel(A) == 1:
        return torch.ones_like(A)

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    # return identity matrix if A is diagonal
    if is_diagonal:
        return torch.eye(
            A.shape[0],
            dtype=A.dtype,
            device=A.device,
        )

    if type(eigenvector_computation_config) is EighEigenvalueCorrectionConfig:
        return _compute_eigenvalue_decomposition(
            A,
            retry_double_precision=eigenvector_computation_config.retry_double_precision,
        )[1]
    else:
        raise NotImplementedError(
            f"Eigenvector computation method is not implemented! Specified eigenvector method is {eigenvector_computation_config=}."
        )
