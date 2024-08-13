"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import re
import unittest
import unittest.mock as mock
from fractions import Fraction
from functools import partial
from types import ModuleType
from typing import Callable, List, Tuple

import matrix_functions

import numpy as np

import torch
from matrix_functions import (
    _matrix_inverse_root_newton,
    _matrix_root_eigen,
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
    matrix_root_diagonal,
    NewtonConvergenceFlag,
    RootInvMethod,
)
from torch import Tensor


class CheckDiagonalTest(unittest.TestCase):
    def test_check_diagonal_for_not_two_dim_matrix(self) -> None:
        A = torch.zeros((2, 2, 2))
        self.assertRaisesRegex(
            ValueError, re.escape("Matrix is not 2-dimensional!"), check_diagonal, A
        )

    def test_check_diagonal_for_not_square_matrix(self) -> None:
        A = torch.zeros((2, 3))
        self.assertRaisesRegex(
            ValueError, re.escape("Matrix is not square!"), check_diagonal, A
        )

    def test_check_diagonal_for_diagonal_matrix(self) -> None:
        A = torch.eye(2)
        self.assertTrue(check_diagonal(A))


class MatrixInverseRootTest(unittest.TestCase):
    def test_matrix_inverse_root_scalar(self) -> None:
        A = torch.tensor(2.0)
        root = 2
        exponent_multiplier = 1.82
        with self.subTest("Test with scalar case."):
            self.assertEqual(
                A ** torch.tensor(-1.82 / 2),
                matrix_inverse_root(
                    A, root=root, exponent_multiplier=exponent_multiplier
                ),
            )
        with self.subTest("Test with matrix case."):
            self.assertEqual(
                torch.tensor([[A ** torch.tensor(-1.82 / 2)]]),
                matrix_inverse_root(
                    torch.tensor([[A]]),
                    root=root,
                    exponent_multiplier=exponent_multiplier,
                ),
            )

    def test_matrix_inverse_root_with_not_two_dim_matrix(self) -> None:
        A = torch.zeros((1, 2, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            matrix_inverse_root,
            A=A,
            root=root,
            exponent_multiplier=exponent_multiplier,
            is_diagonal=False,
        )

    def test_matrix_inverse_root_not_square(self) -> None:
        A = torch.zeros((2, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            matrix_inverse_root,
            A=A,
            root=root,
            exponent_multiplier=exponent_multiplier,
            is_diagonal=False,
        )

    def test_matrix_inverse_root(self) -> None:
        A_list = [
            torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            torch.tensor(
                [
                    [1195.0, -944.0, -224.0],
                    [-944.0, 746.0, 177.0],
                    [-224.0, 177.0, 42.0],
                ]
            ),
        ]
        root = 2
        exponent_multiplier = 1.0
        actual_root_list = [
            torch.tensor([[1.0, 0.0], [0.0, 0.5]]),
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, -3.0], [1.0, -3.0, 18.0]]),
        ]

        atol = 0.05
        rtol = 1e-2
        with self.subTest("Test with diagonal case."):
            torch.testing.assert_close(
                actual_root_list[0],
                matrix_inverse_root(
                    A_list[0],
                    root=root,
                    exponent_multiplier=exponent_multiplier,
                    is_diagonal=True,
                ),
                atol=atol,
                rtol=rtol,
            )
        with self.subTest("Test with EIGEN."):
            for i in range(len(A_list)):
                torch.testing.assert_close(
                    actual_root_list[i],
                    matrix_inverse_root(
                        A_list[i],
                        root=root,
                        exponent_multiplier=exponent_multiplier,
                        root_inv_method=RootInvMethod.EIGEN,
                        is_diagonal=False,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
        with self.subTest("Test with NEWTON."):
            for i in range(len(A_list)):
                torch.testing.assert_close(
                    actual_root_list[i],
                    matrix_inverse_root(
                        A_list[i],
                        root=root,
                        exponent_multiplier=exponent_multiplier,
                        root_inv_method=RootInvMethod.NEWTON,
                        is_diagonal=False,
                    ),
                    atol=atol,
                    rtol=rtol,
                )
        with self.subTest("Test with HIGHER_ORDER."):
            for i in range(len(A_list)):
                for order in range(2, 7):
                    torch.testing.assert_close(
                        actual_root_list[i],
                        matrix_inverse_root(
                            A_list[i],
                            root=Fraction(root),
                            exponent_multiplier=exponent_multiplier,
                            root_inv_method=RootInvMethod.HIGHER_ORDER,
                            order=order,
                            is_diagonal=False,
                        ),
                        atol=atol,
                        rtol=rtol,
                    )
                    # Also test that powering works
                    exp = 2
                    torch.testing.assert_close(
                        torch.linalg.matrix_power(actual_root_list[i], exp),
                        matrix_inverse_root(
                            A_list[i],
                            root=Fraction(root) / exp,
                            exponent_multiplier=exponent_multiplier,
                            root_inv_method=RootInvMethod.HIGHER_ORDER,
                            order=order,
                            is_diagonal=False,
                        ),
                        atol=atol,
                        rtol=rtol,
                    )

    def test_matrix_inverse_root_higher_order_blowup(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
        root_inv_method = RootInvMethod.HIGHER_ORDER
        self.assertRaisesRegex(
            ArithmeticError,
            re.escape(
                "NaN/Inf in matrix inverse root (after powering for fractions), raising an exception!"
            ),
            matrix_inverse_root,
            A=A,
            root=Fraction(1, 20),
            exponent_multiplier=1.0,
            root_inv_method=root_inv_method,
        )

    def test_matrix_inverse_root_with_no_effect_exponent_multiplier(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root_inv_method_and_msg: List[Tuple[RootInvMethod, str]] = [
            (RootInvMethod.NEWTON, "inverse Newton iteration"),
            (RootInvMethod.HIGHER_ORDER, "higher order method"),
        ]

        for root_inv_method, root_inv_method_msg in root_inv_method_and_msg:
            with self.subTest(
                root_inv_method=root_inv_method, root_inv_method_msg=root_inv_method_msg
            ):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(
                        f"Exponent multiplier 2.0 must be equal to 1 to use coupled {root_inv_method_msg}!"
                    ),
                    matrix_inverse_root,
                    A=A,
                    root=2,
                    exponent_multiplier=2.0,
                    root_inv_method=root_inv_method,
                )

    def test_matrix_inverse_root_newton_fraction(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Root 1/2 must be an integer to use coupled inverse Newton iteration!"
            ),
            matrix_inverse_root,
            A=A,
            root=Fraction(numerator=1, denominator=2),
            root_inv_method=RootInvMethod.NEWTON,
            is_diagonal=False,
        )

    def test_matrix_inverse_root_reach_max_iterations(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = 4
        root_inv_method_and_implementation_and_msg: List[
            Tuple[RootInvMethod, str, str]
        ] = [
            (RootInvMethod.NEWTON, "_matrix_inverse_root_newton", "Newton"),
            (
                RootInvMethod.HIGHER_ORDER,
                "_matrix_inverse_root_higher_order",
                "Higher order method",
            ),
        ]
        for (
            root_inv_method,
            implementation,
            msg,
        ) in root_inv_method_and_implementation_and_msg:
            with mock.patch.object(
                matrix_functions,
                implementation,
                return_value=(
                    None,
                    None,
                    NewtonConvergenceFlag.REACHED_MAX_ITERS,
                    None,
                    None,
                ),
            ), self.subTest(
                root_inv_method=root_inv_method,
                implementation=implementation,
                msg=msg,
            ), self.assertLogs(
                level="WARNING",
            ) as cm:
                matrix_inverse_root(
                    A=A,
                    root=root,
                    root_inv_method=root_inv_method,
                )
                self.assertIn(
                    f"{msg} did not converge and reached maximum number of iterations!",
                    [r.msg for r in cm.records],
                )

    def test_matrix_inverse_root_higher_order_tf32_preservation(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, float("inf")]])
        root = 2
        exponent_multiplier = 1.0
        tf32_flag_before = torch.backends.cuda.matmul.allow_tf32
        self.assertRaisesRegex(
            ArithmeticError,
            re.escape("Input matrix has entries close to inf"),
            matrix_inverse_root,
            A=A,
            root=Fraction(root),
            exponent_multiplier=exponent_multiplier,
            root_inv_method=RootInvMethod.HIGHER_ORDER,
        )
        tf32_flag_after = torch.backends.cuda.matmul.allow_tf32
        assert tf32_flag_before == tf32_flag_after

    def test_matrix_inverse_root_with_invalid_root_inv_method(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = 4
        with mock.patch.object(
            RootInvMethod, "__eq__", return_value=False
        ), self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Root inverse method is not implemented! Specified root inverse method is RootInvMethod.NEWTON."
            ),
        ):
            matrix_inverse_root(
                A=A,
                root=root,
                root_inv_method=RootInvMethod.NEWTON,
                is_diagonal=False,
            )


class MatrixRootDiagonalTest(unittest.TestCase):
    def test_matrix_root_diagonal_with_not_two_dim_matrix(self) -> None:
        A = torch.zeros((1, 2, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            matrix_root_diagonal,
            A=A,
            root=root,
            exponent_multiplier=exponent_multiplier,
            return_full_matrix=True,
        )

    def test_matrix_root_diagonal_nonpositive_root(self) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        root = -1
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Root {root} should be positive!"),
            matrix_root_diagonal,
            A=A,
            root=root,
            return_full_matrix=True,
        )


class EigenRootTest(unittest.TestCase):
    def _test_eigen_root(
        self,
        A: torch.Tensor,
        root: int,
        make_positive_semidefinite: bool,
        inverse: bool,
        epsilon: float,
        tolerance: float,
        eig_sols: Tensor,
    ) -> None:
        X, L, Q = _matrix_root_eigen(
            A=A,
            root=root,
            epsilon=epsilon,
            make_positive_semidefinite=make_positive_semidefinite,
            inverse=inverse,
        )
        if inverse:
            root = -root
        abs_error = torch.dist(torch.linalg.matrix_power(X, root), A, p=torch.inf)
        A_norm = torch.linalg.norm(A, ord=torch.inf)
        rel_error = abs_error / torch.maximum(torch.tensor(1.0), A_norm)
        torch.testing.assert_close(L, eig_sols)
        self.assertTrue(rel_error <= tolerance)

    def _test_eigen_root_multi_dim(
        self,
        A: Callable[[int], Tensor],
        dims: List[int],
        roots: List[int],
        make_positive_semidefinite: bool,
        epsilons: List[float],
        tolerance: float,
        eig_sols: Callable[[int], Tensor],
    ) -> None:
        for n, root, epsilon in itertools.product(dims, roots, epsilons):
            with self.subTest(f"With dim = {n}, root = {root}, epsilon = {epsilon}"):
                self._test_eigen_root(
                    A(n),
                    root,
                    make_positive_semidefinite,
                    False,
                    epsilon,
                    tolerance,
                    eig_sols(n),
                )
                self._test_eigen_root(
                    A(n),
                    root,
                    make_positive_semidefinite,
                    True,
                    epsilon,
                    tolerance,
                    eig_sols(n),
                )

    def test_eigen_root_identity(self) -> None:
        tolerance = 1e-6
        dims = [10, 100]
        roots = [1, 2, 4, 8]
        epsilons = [0.0]
        make_positive_semidefinite = False

        def eig_sols(n: int) -> Tensor:
            return torch.ones(n)

        def A(n: int) -> Tensor:
            return torch.eye(n)

        self._test_eigen_root_multi_dim(
            A, dims, roots, make_positive_semidefinite, epsilons, tolerance, eig_sols
        )

    def test_eigen_root_tridiagonal_1(self) -> None:
        tolerance = 1e-4
        dims = [10, 100]
        roots = [1, 2, 4, 8]
        epsilons = [0.0]
        make_positive_semidefinite = False

        for alpha, beta in itertools.product(
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], repeat=2
        ):
            if 2 * beta > alpha:
                continue

            with self.subTest(f"Test with alpha = {alpha}, beta = {beta}"):

                def eig_sols(n: int, alpha: float, beta: float) -> Tensor:
                    eigs = alpha * torch.ones(n) + 2 * beta * torch.tensor(
                        [np.cos(j * torch.pi / n) for j in range(n)], dtype=torch.float
                    )
                    eigs, _ = torch.sort(eigs)
                    return eigs

                def A(n: int, alpha: float, beta: float) -> Tensor:
                    diag = alpha * torch.ones(n)
                    diag[0] += beta
                    diag[n - 1] += beta
                    off_diag = beta * torch.ones(n - 1)
                    return (
                        torch.diag(diag)
                        + torch.diag(off_diag, diagonal=1)
                        + torch.diag(off_diag, diagonal=-1)
                    )

                self._test_eigen_root_multi_dim(
                    partial(A, alpha=alpha, beta=beta),
                    dims,
                    roots,
                    make_positive_semidefinite,
                    epsilons,
                    tolerance,
                    partial(eig_sols, alpha=alpha, beta=beta),
                )

    def test_eigen_root_tridiagonal_2(self) -> None:
        tolerance = 1e-4
        dims = [10, 100]
        roots = [1, 2, 4, 8]
        epsilons = [0.0]
        make_positive_semidefinite = False

        for alpha, beta in itertools.product(
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], repeat=2
        ):
            if 2 * beta > alpha:
                continue

            with self.subTest(f"Test with alpha = {alpha}, beta = {beta}"):

                def eig_sols(n: int, alpha: float, beta: float) -> Tensor:
                    eigs = alpha * torch.ones(n) + 2 * beta * torch.tensor(
                        [
                            np.cos(2 * j * torch.pi / (2 * n + 1))
                            for j in range(1, n + 1)
                        ],
                        dtype=torch.float,
                    )
                    eigs, _ = torch.sort(eigs)
                    return eigs

                def A(n: int, alpha: float, beta: float) -> Tensor:
                    diag = alpha * torch.ones(n)
                    diag[0] -= beta
                    off_diag = beta * torch.ones(n - 1)
                    return (
                        torch.diag(diag)
                        + torch.diag(off_diag, diagonal=1)
                        + torch.diag(off_diag, diagonal=-1)
                    )

                self._test_eigen_root_multi_dim(
                    partial(A, alpha=alpha, beta=beta),
                    dims,
                    roots,
                    make_positive_semidefinite,
                    epsilons,
                    tolerance,
                    partial(eig_sols, alpha=alpha, beta=beta),
                )

    def test_matrix_root_eigen_nonpositive_root(self) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        root = -1
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Root {root} should be positive!"),
            _matrix_root_eigen,
            A=A,
            root=root,
        )

    torch_lianlg_module: ModuleType = torch.linalg

    @mock.patch.object(
        torch_lianlg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_no_retry_double_precision_raise_exception(
        self, mock_eigh: mock.Mock
    ) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, re.escape("Mock Eigen Error")):
            _matrix_root_eigen(
                A=A,
                root=2,
                epsilon=0.0,
                make_positive_semidefinite=True,
                inverse=False,
                retry_double_precision=False,
            )
        mock_eigh.assert_called_once()

    @mock.patch.object(
        torch_lianlg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_retry_double_precision_raise_exception(self, mock_eigh: mock.Mock) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, re.escape("Mock Eigen Error")):
            _matrix_root_eigen(
                A=A,
                root=2,
                epsilon=0.0,
                make_positive_semidefinite=True,
                inverse=False,
                retry_double_precision=True,
            )
        mock_eigh.assert_called()
        self.assertEqual(mock_eigh.call_count, 2)

    @mock.patch.object(
        torch_lianlg_module,
        "eigh",
        side_effect=[
            RuntimeError("Mock Eigen Error"),
            (torch.ones(2), torch.eye(2)),
        ],
    )
    def test_retry_double_precision_double_precision(
        self, mock_eigh: mock.Mock
    ) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        X, _, _ = _matrix_root_eigen(
            A=A,
            root=2,
            epsilon=0.0,
            make_positive_semidefinite=True,
            inverse=False,
            retry_double_precision=True,
        )
        torch.testing.assert_close(X, torch.eye(2))
        mock_eigh.assert_called()
        self.assertEqual(mock_eigh.call_count, 2)


class NewtonRootInverseTest(unittest.TestCase):
    def _test_newton_root_inverse(
        self,
        A: Tensor,
        root: int,
        epsilon: float,
        max_iterations: int,
        A_tol: float,
        M_tol: float,
    ) -> None:
        X, M, flag, iteration, M_error = _matrix_inverse_root_newton(
            A, root, epsilon, max_iterations, M_tol
        )
        abs_A_error = torch.dist(torch.linalg.matrix_power(X, -root), A, p=torch.inf)
        A_norm = torch.linalg.norm(A, ord=torch.inf)
        rel_A_error = abs_A_error / torch.maximum(torch.tensor(1.0), A_norm)
        self.assertTrue(M_error <= M_tol)
        self.assertTrue(rel_A_error <= A_tol)

    def _test_newton_root_inverse_multi_dim(
        self,
        A: Callable[[int], Tensor],
        dims: List[int],
        roots: List[int],
        epsilons: List[float],
        max_iterations: int,
        A_tol: float,
        M_tol: float,
    ) -> None:

        for n, root, epsilon in itertools.product(dims, roots, epsilons):
            with self.subTest(f"With dim = {n}, root = {root}, epsilon = {epsilon}"):
                self._test_newton_root_inverse(
                    A(n), root, epsilon, max_iterations, A_tol, M_tol
                )

    def test_newton_root_inverse_identity(self) -> None:
        A_tol = 1e-6
        M_tol = 1e-6
        max_iterations = 1000
        dims = [10, 100]
        roots = [2, 4, 8]
        epsilons = [0.0]

        def A(n: int) -> Tensor:
            return torch.eye(n)

        self._test_newton_root_inverse_multi_dim(
            A, dims, roots, epsilons, max_iterations, A_tol, M_tol
        )

    def test_newton_root_inverse_tridiagonal_1(self) -> None:
        A_tol = 1e-4
        M_tol = 1e-6
        max_iterations = 1000
        dims = [10, 100]
        roots = [2, 4, 8]
        epsilons = [0.0]

        for alpha, beta in itertools.product(
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], repeat=2
        ):
            if 2 * beta > alpha:
                continue

            with self.subTest(f"Test with alpha = {alpha}, beta = {beta}"):

                def A(n: int, alpha: float, beta: float) -> Tensor:
                    diag = alpha * torch.ones(n)
                    diag[0] += beta
                    diag[n - 1] += beta
                    off_diag = beta * torch.ones(n - 1)
                    return (
                        torch.diag(diag)
                        + torch.diag(off_diag, diagonal=1)
                        + torch.diag(off_diag, diagonal=-1)
                    )

                self._test_newton_root_inverse_multi_dim(
                    partial(A, alpha=alpha, beta=beta),
                    dims,
                    roots,
                    epsilons,
                    max_iterations,
                    A_tol,
                    M_tol,
                )

    def test_newton_root_inverse_tridiagonal_2(self) -> None:
        A_tol = 1e-4
        M_tol = 1e-6
        max_iterations = 1000
        dims = [10, 100]
        roots = [2, 4, 8]
        epsilons = [0.0]

        for alpha, beta in itertools.product(
            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], repeat=2
        ):
            if 2 * beta > alpha:
                continue

            with self.subTest(f"Test with alpha = {alpha}, beta = {beta}"):

                def A(n: int, alpha: float, beta: float) -> Tensor:
                    diag = alpha * torch.ones(n)
                    diag[0] -= beta
                    off_diag = beta * torch.ones(n - 1)
                    return (
                        torch.diag(diag)
                        + torch.diag(off_diag, diagonal=1)
                        + torch.diag(off_diag, diagonal=-1)
                    )

                self._test_newton_root_inverse_multi_dim(
                    partial(A, alpha=alpha, beta=beta),
                    dims,
                    roots,
                    epsilons,
                    max_iterations,
                    A_tol,
                    M_tol,
                )


class ComputeMatrixRootInverseResidualsTest(unittest.TestCase):
    def test_matrix_root_inverse_residuals_with_not_two_dim_matrix(self) -> None:
        A = torch.zeros((1, 2, 3))
        X_hat = torch.zeros((2, 2))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
            exponent_multiplier=exponent_multiplier,
        )

    def test_matrix_root_inverse_residuals_with_not_square_matrix(self) -> None:
        A = torch.zeros((1, 2))
        X_hat = torch.zeros((2, 2))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
            exponent_multiplier=exponent_multiplier,
        )

    def test_matrix_root_inverse_residuals_with_inconsistent_dims(self) -> None:
        A = torch.zeros((2, 2))
        X_hat = torch.zeros((3, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix shapes do not match!"),
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
            exponent_multiplier=exponent_multiplier,
        )

    def _test_matrix_root_inverse_residuals(
        self,
        A: torch.Tensor,
        X_hat: torch.Tensor,
        root: int,
        exponent_multiplier: float,
        expected_relative_error: torch.Tensor,
        expected_relative_residual: torch.Tensor,
    ) -> None:
        (
            actual_relative_error,
            actual_relative_residual,
        ) = compute_matrix_root_inverse_residuals(
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
            exponent_multiplier=exponent_multiplier,
        )
        torch.testing.assert_close(
            actual_relative_error,
            expected_relative_error,
        )
        torch.testing.assert_close(
            actual_relative_residual,
            expected_relative_residual,
        )

    def test_matrix_root_inverse_residuals(self) -> None:
        A = torch.eye(2)
        X_hat = torch.eye(2)
        expected_relative_error = torch.tensor(0.0, dtype=torch.float64)
        expected_relative_residual = torch.tensor(0.0, dtype=torch.float64)

        with self.subTest("Exponent multiplier = 1."):
            root = 2
            exponent_multiplier = 1.0
            self._test_matrix_root_inverse_residuals(
                A=A,
                X_hat=X_hat,
                root=root,
                exponent_multiplier=exponent_multiplier,
                expected_relative_error=expected_relative_error,
                expected_relative_residual=expected_relative_residual,
            )
        with self.subTest("Exponent multiplier != 1."):
            root = 4
            exponent_multiplier = 2.0
            self._test_matrix_root_inverse_residuals(
                A=A,
                X_hat=X_hat,
                root=root,
                exponent_multiplier=exponent_multiplier,
                expected_relative_error=expected_relative_error,
                expected_relative_residual=expected_relative_residual,
            )
