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
from fractions import Fraction
from math import isfinite
from typing import Tuple, Union

import torch
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


class RootInvMethod(enum.Enum):
    """
    Enum class for supported root inverse methods, i.e., computing M -> M^{-1/root}.

    EIGEN: Uses eigendecomposition followed by diagonal powering.
    NEWTON: Uses coupled inverse Newton iteration (Higham, Functions of Matrices).
    HIGHER_ORDER: Uses higher-order variants of NEWTON (Lakic, 1998: "On the Computation of the Matrix k-th Root").
    """

    EIGEN = 0
    NEWTON = 1
    HIGHER_ORDER = 2


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
    root: Union[Fraction, int],
    epsilon: float = 0.0,
    exponent_multiplier: float = 1.0,
    root_inv_method: RootInvMethod = RootInvMethod.EIGEN,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    is_diagonal: Union[Tensor, bool] = False,
    retry_double_precision: bool = True,
    order: int = 2,
) -> Tensor:
    """Computes matrix root inverse of square symmetric positive definite matrix.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        exponent_multiplier (float): exponent multiplier in the eigen method (Default: 1.0)
        root_inv_method (RootInvMethod): Specifies method to use to compute root inverse. (Default: RootInvMethod.EIGEN)
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 1000)
        tolerance (float): Tolerance for computing root inverse using coupled Newton iteration. (Default: 1e-6)
        is_diagonal (Tensor, bool): Flag for whether or not matrix is diagonal. If so, will compute root inverse by computing
            root inverse of diagonal entries. (Default: False)
        retry_double_precision (bool): Flag for re-trying eigendecomposition with higher precision if lower precision fails due
            to CuSOLVER failure. (Default: True)
        order (int): Order used in the higher-order method. (Default: 2)

    Returns:
        X (Tensor): Inverse root of matrix A.

    """

    # check if matrix is scalar
    if torch.numel(A) == 1:
        alpha = torch.as_tensor(-exponent_multiplier / root)
        return (A + epsilon) ** alpha

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    if is_diagonal:
        X = matrix_root_diagonal(
            A=A,
            root=root,
            epsilon=epsilon,
            inverse=True,
            exponent_multiplier=exponent_multiplier,
            return_full_matrix=True,
        )
    elif root_inv_method == RootInvMethod.EIGEN:
        X, _, _ = _matrix_root_eigen(
            A=A,
            root=root,
            epsilon=epsilon,
            inverse=True,
            exponent_multiplier=exponent_multiplier,
            retry_double_precision=retry_double_precision,
        )
    elif root_inv_method == RootInvMethod.NEWTON:
        if exponent_multiplier != 1.0:
            raise ValueError(
                f"Exponent multiplier {exponent_multiplier} must be equal to 1 to use coupled inverse Newton iteration!"
            )

        if isinstance(root, Fraction):
            raise ValueError(
                f"Root {root} must be an integer to use coupled inverse Newton iteration!"
            )

        X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
            A=A,
            root=root,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning(
                "Newton did not converge and reached maximum number of iterations!"
            )
    elif root_inv_method == RootInvMethod.HIGHER_ORDER:
        if exponent_multiplier != 1.0:
            raise ValueError(
                f"Exponent multiplier {exponent_multiplier} must be equal to 1 to use coupled higher order method!"
            )

        X, _, termination_flag, _, _ = _matrix_inverse_root_higher_order(
            A=A,
            root=Fraction(root),
            rel_epsilon=epsilon,
            order=order,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning(
                "Higher order method did not converge and reached maximum number of iterations!"
            )
    else:
        raise NotImplementedError(
            f"Root inverse method is not implemented! Specified root inverse method is {str(root_inv_method)}."
        )

    return X


def matrix_root_diagonal(
    A: Tensor,
    root: Union[Fraction, int],
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    return_full_matrix: bool = False,
) -> Tensor:
    """Computes matrix inverse root for a diagonal matrix by taking inverse square root of diagonal entries.

    Args:
        A (Tensor): One- or two-dimensional tensor containing either the diagonal entries of the matrix or a diagonal matrix.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        return_full_matrix (bool): Returns full matrix by taking torch.diag of diagonal entries. (bool: False)

    Returns:
        X (Tensor): Inverse root of diagonal entries.

    """

    # check order of tensor
    order = len(A.shape)
    if order == 2:
        A = torch.diag(A)
    elif order > 2:
        raise ValueError("Matrix is not 2-dimensional!")

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # compute matrix power
    alpha = exponent_multiplier / root
    if inverse:
        alpha = -alpha

    X = (A + epsilon).pow(alpha)
    return torch.diag(X) if return_full_matrix else X


def _matrix_root_eigen(
    A: Tensor,
    root: Union[Fraction, int],
    epsilon: float = 0.0,
    inverse: bool = True,
    exponent_multiplier: float = 1.0,
    make_positive_semidefinite: bool = True,
    retry_double_precision: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute matrix (inverse) root using eigendecomposition of symmetric positive (semi-)definite matrix.

            A = Q L Q^T => A^{1/r} = Q L^{1/r} Q^T OR A^{-1/r} = Q L^{-1/r} Q^T

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        exponent_multiplier (float): exponent multiplier in the eigen method (Default: 1.0)
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

    # compute matrix power
    alpha = exponent_multiplier / root
    if inverse:
        alpha = -alpha

    # compute eigendecomposition and compute minimum eigenvalue
    try:
        L, Q = torch.linalg.eigh(A)

    except Exception as exception:
        if retry_double_precision and A.dtype != torch.float64:
            logger.warning(
                f"Failed to compute eigendecomposition in {A.dtype} precision with exception {exception}! Retrying in double precision..."
            )
            L, Q = torch.linalg.eigh(A.double())
        else:
            raise exception

    lambda_min = torch.min(L)

    # make eigenvalues >= 0 (if necessary)
    if make_positive_semidefinite:
        L += -torch.minimum(lambda_min, torch.as_tensor(0.0))

    # add epsilon
    L += epsilon

    # compute inverse preconditioner
    X = Q * L.pow(alpha).unsqueeze(0) @ Q.T

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
    max_iterations: int = 100,
    tolerance: float = 1e-20,
    order: int = 2,  # 2 represents Newton's method
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
        Generally recommend setting according to A.dtype (1e-3 for tf32, 1e-5 for fp32, 1e-9 for fp64) (Default: 0.0)
        max_iterations (int): Maximum number of iterations. Typically we need < 20 iterations. (Default: 100)
        tolerance (float): Tolerance for determining exit criterion from iterations. (Default: 1e-20, which in practice guarantees they run to convergence)
        order (int): Order of the method. Order must be >= 2.  Higher order methods accelerate convergence (fewer iterations), but can take more matmuls per iteration. (Default: 2, ie Newton)
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
        t_iter_begin = time.time()
        p = root.numerator
        q = root.denominator
        dtype = A.dtype

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
        A_fourth = torch.linalg.matrix_power(A, 4)
        n_matmul = 2
        lambda_max_approx = torch.linalg.matrix_norm(A, torch.inf)
        lambda_max_approx = min(
            torch.linalg.matrix_norm(A_fourth, torch.inf) ** 0.25,
            lambda_max_approx,
        )

        # We have not seen lambda_max being Inf in practice, however there is not a whole lot we can do in this pathological case and its good to bail early
        if not isfinite(lambda_max_approx):
            raise ArithmeticError(
                "Input matrix has entries close to inf, exiting root inverse"
            )

        # Now scale and setup our variables
        epsilon = max(rel_epsilon * lambda_max_approx, 1.0e-16)
        identity = torch.eye(n, dtype=dtype, device=A.device)
        A_ridge = torch.add(A, identity, alpha=epsilon)
        lambda_max_approx += epsilon

        # Figure out a constant that gives good starting location
        # We default to 1.001, but adjust depending on epsilon and lambda_max_approx
        # Roughly, this is done to "balance" the eigenvalue dynamics of 1st iteration at the two extreme ends: lambda_max and eps
        # So after the 1st iteration of the method below, lambda_max_approx and eps both get sent to the same value (in the unit disk)
        # TODO: this complex recipe may not actually make much too much of a difference but lets retain it for now
        c = 1.001
        if epsilon > 0:
            cond_term = (lambda_max_approx / epsilon) ** (1 / p)
            c_new = (cond_term * lambda_max_approx - epsilon) / (
                cond_term * lambda_max_approx - lambda_max_approx
            )
            if c_new > c:
                c = c_new
                logger.debug(f"Changed seed factor from 1.001 to: {c_new}")

        # For convergence, c = 1.0 below is enough. Put in at least 1.001 for numerical safety.
        z = (p + 1) / (c * lambda_max_approx)
        X = (z ** (-s)) * identity
        M = z * A_ridge
        error = torch.linalg.vector_norm(M - identity, torch.inf)
        t_iter_end = time.time()
        logger.debug(
            f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
        )

        # Do one iteration of basic Newton first. This is used to mathematically guarantee convergence of higher order method.
        # TODO: we may be able to get rid of this with a more careful analysis of the convergence region
        t_iter_begin = time.time()
        M_p = M.mul(s).add_(identity, alpha=(1 - s))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, p) @ M
        error = torch.linalg.vector_norm(M - identity, torch.inf)
        n_matmul += math.ceil(math.log2(p)) + 2
        iteration += 1
        t_iter_end = time.time()
        logger.debug(
            f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
        )

        # main while loop
        termination_flag = NewtonConvergenceFlag.CONVERGED
        while error > tolerance and iteration < max_iterations:
            t_iter_begin = time.time()
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
            if new_error > error * 1.2 or new_error == error:
                logger.debug(
                    f"Coupled inverse Newton is stagnating or diverging based on comparing current error {new_error.item()} against last iteration's error {error.item()}."
                    f"(We assume divergence if the new error > 1.2 * previous error, and assume stagnation if they are equal.)"
                )
                termination_flag = NewtonConvergenceFlag.EARLY_STOP
                break
            error = new_error

            t_iter_end = time.time()
            logger.debug(
                f"Iteration dur (s): {t_iter_end - t_iter_begin}, Error (|M-I|) at iteration {iteration}: {error.item()}"
            )

        # determine convergence flag
        if termination_flag != NewtonConvergenceFlag.EARLY_STOP and error > tolerance:
            termination_flag = NewtonConvergenceFlag.REACHED_MAX_ITERS

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
    root: int,
    epsilon: float,
    exponent_multiplier: float,
) -> Tuple[Tensor, Tensor]:
    """Compute residual of matrix root inverse for debugging purposes.

        relative error    = ||X - X_hat||_inf / ||X||_inf
        relative residual = ||A X^r - I||_inf

    Args:
        A (Tensor): Matrix of interest.
        X (Tensor): Computed matrix root inverse.
        root (int): Root of interest.
        epsilon (float): Adds epsilon * I to matrix.
        exponent_multiplier (float): Exponent multiplier to be multiplied to the numerator of the inverse root.

    Returns:
        absolute_error (Tensor): absolute error of matrix root inverse
        relative_error (Tensor): relative error of matrix root inverse
        residual (Tensor): residual of matrix root inverse

    """

    # check shape of matrix
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")
    elif A.shape != X_hat.shape:
        raise ValueError("Matrix shapes do not match!")

    # compute error by comparing against double precision
    X = matrix_inverse_root(
        A.double(), root, epsilon=epsilon, exponent_multiplier=exponent_multiplier
    )
    relative_error = torch.dist(X, X_hat, p=torch.inf) / torch.norm(X, p=torch.inf)

    # compute residual
    if exponent_multiplier == 1.0:
        X_invr = torch.linalg.matrix_power(X_hat.double(), n=-root)
    else:
        X_invr, _, _ = _matrix_root_eigen(
            X_hat.double(),
            root=1,
            epsilon=0.0,
            inverse=True,
            make_positive_semidefinite=True,
            exponent_multiplier=root / exponent_multiplier,
        )

    A_reg = A.double() + epsilon * torch.eye(
        A.shape[0], dtype=torch.float64, device=A.device
    )
    relative_residual = torch.dist(X_invr, A_reg, p=torch.inf) / torch.norm(
        A_reg, p=torch.inf
    )

    return relative_error, relative_residual
