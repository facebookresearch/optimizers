"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import unittest
import unittest.mock as mock

import numpy as np

import torch

from distributed_shampoo.utils.matrix_functions import (
    _matrix_inverse_root_newton,
    _matrix_root_eigen,
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_inverse_root,
    matrix_root_diagonal,
    RootInvMethod,
)


class CheckDiagonalTest(unittest.TestCase):
    def test_check_diagonal_for_not_two_dim_matrix(self):
        A = torch.zeros((2, 2, 2))
        self.assertRaisesRegex(
            ValueError, "Matrix is not 2-dimensional!", check_diagonal, A
        )

    def test_check_diagonal_for_not_square_matrix(self):
        A = torch.zeros((2, 3))
        self.assertRaisesRegex(ValueError, "Matrix is not square!", check_diagonal, A)

    def test_check_diagonal_for_diagonal_matrix(self):
        A = torch.eye(2)
        self.assertTrue(check_diagonal(A))


class MatrixInverseRootTest(unittest.TestCase):
    def test_matrix_inverse_root_scalar(self):
        A = torch.tensor(2.0)
        root = 2
        exponent_multiplier = 1.82
        with self.subTest("Test with scalar case."):
            self.assertEqual(
                A ** (-1.82 / 2),
                matrix_inverse_root(
                    A, root=root, exponent_multiplier=exponent_multiplier
                ),
            )
        with self.subTest("Test with matrix case."):
            self.assertEqual(
                torch.tensor([[A ** (-1.82 / 2)]]),
                matrix_inverse_root(
                    torch.tensor([[A]]),
                    root=root,
                    exponent_multiplier=exponent_multiplier,
                ),
            )

    def test_matrix_inverse_root_with_not_two_dim_matrix(self):
        A = torch.zeros((1, 2, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            "Matrix is not 2-dimensional!",
            matrix_inverse_root,
            A=A,
            root=root,
            exponent_multiplier=exponent_multiplier,
            is_diagonal=False,
        )

    def test_matrix_inverse_root_not_square(self):
        A = torch.zeros((2, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            "Matrix is not square!",
            matrix_inverse_root,
            A=A,
            root=root,
            exponent_multiplier=exponent_multiplier,
            is_diagonal=False,
        )

    def test_matrix_inverse_root(self):
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = 2
        exponent_multiplier = 1.0
        actual_inverse_root = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
        with self.subTest("Test with diagonal case."):
            torch.testing.assert_close(
                actual_inverse_root,
                matrix_inverse_root(
                    A,
                    root=root,
                    exponent_multiplier=exponent_multiplier,
                    is_diagonal=True,
                ),
            )
        with self.subTest("Test with EIGEN."):
            torch.testing.assert_close(
                actual_inverse_root,
                matrix_inverse_root(
                    A,
                    root=root,
                    exponent_multiplier=exponent_multiplier,
                    root_inv_method=RootInvMethod.EIGEN,
                    is_diagonal=False,
                ),
            )
        with self.subTest("Test with NEWTON."):
            torch.testing.assert_close(
                actual_inverse_root,
                matrix_inverse_root(
                    A,
                    root=root,
                    exponent_multiplier=exponent_multiplier,
                    root_inv_method=RootInvMethod.NEWTON,
                    is_diagonal=False,
                ),
            )

    def test_matrix_inverse_root_newton_with_exponent_multiplier(self):
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        self.assertRaisesRegex(
            ValueError,
            f"Exponent multiplier {2.0} must be equal to 1 to use coupled inverse Newton iteration!",
            matrix_inverse_root,
            A=A,
            root=4,
            exponent_multiplier=2.0,
            root_inv_method=RootInvMethod.NEWTON,
            is_diagonal=False,
        )


class MatrixRootDiagonalTest(unittest.TestCase):
    def test_matrix_root_diagonal_with_not_two_dim_matrix(self):
        A = torch.zeros((1, 2, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            "Matrix is not 2-dimensional!",
            matrix_root_diagonal,
            A=A,
            root=root,
            exponent_multiplier=exponent_multiplier,
            return_full_matrix=True,
        )

    def test_matrix_root_diagonal_nonpositive_root(self):
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        root = -1
        self.assertRaisesRegex(
            ValueError,
            f"Root {root} should be positive!",
            matrix_root_diagonal,
            A=A,
            root=root,
            return_full_matrix=True,
        )


class EigenRootTest(unittest.TestCase):
    def _test_eigen_root(
        self,
        A,
        root,
        make_positive_semidefinite,
        inverse,
        epsilon,
        tolerance,
        eig_sols,
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
        self.assertLessEqual(rel_error, tolerance)

    def _test_eigen_root_multi_dim(
        self,
        A,
        dims,
        roots,
        make_positive_semidefinite,
        epsilons,
        tolerance,
        eig_sols,
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

        def eig_sols(n):
            return torch.ones(n)

        def A(n):
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

                def eig_sols(n):
                    eigs = alpha * torch.ones(n) + 2 * beta * torch.tensor(
                        [np.cos(j * torch.pi / n) for j in range(n)], dtype=torch.float
                    )
                    eigs, _ = torch.sort(eigs)
                    return eigs

                def A(n):
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
                    A,
                    dims,
                    roots,
                    make_positive_semidefinite,
                    epsilons,
                    tolerance,
                    eig_sols,
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

                def eig_sols(n):
                    eigs = alpha * torch.ones(n) + 2 * beta * torch.tensor(
                        [
                            np.cos(2 * j * torch.pi / (2 * n + 1))
                            for j in range(1, n + 1)
                        ],
                        dtype=torch.float,
                    )
                    eigs, _ = torch.sort(eigs)
                    return eigs

                def A(n):
                    diag = alpha * torch.ones(n)
                    diag[0] -= beta
                    off_diag = beta * torch.ones(n - 1)
                    return (
                        torch.diag(diag)
                        + torch.diag(off_diag, diagonal=1)
                        + torch.diag(off_diag, diagonal=-1)
                    )

                self._test_eigen_root_multi_dim(
                    A,
                    dims,
                    roots,
                    make_positive_semidefinite,
                    epsilons,
                    tolerance,
                    eig_sols,
                )

    def test_matrix_root_eigen_nonpositive_root(self):
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        root = -1
        self.assertRaisesRegex(
            ValueError,
            f"Root {root} should be positive!",
            _matrix_root_eigen,
            A=A,
            root=root,
        )

    @mock.patch("torch.linalg.eigh")
    def test_no_retry_double_precision_raise_exception(self, mock_eigh: mock.Mock):
        mock_eigh.side_effect = RuntimeError("Mock Eigen Error")
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, "Mock Eigen Error"):
            _matrix_root_eigen(
                A=A,
                root=2,
                epsilon=0.0,
                make_positive_semidefinite=True,
                inverse=False,
                retry_double_precision=False,
            )
        mock_eigh.assert_called_once()

    @mock.patch("torch.linalg.eigh")
    def test_retry_double_precision_raise_exception(self, mock_eigh: mock.Mock):
        mock_eigh.side_effect = RuntimeError("Mock Eigen Error")
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, "Mock Eigen Error"):
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

    @mock.patch("torch.linalg.eigh")
    def test_retry_double_precision_double_precision(self, mock_eigh: mock.Mock):
        mock_eigh.side_effect = [
            RuntimeError("Mock Eigen Error"),
            (torch.ones(2), torch.eye(2)),
        ]
        A = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        X, _, _ = _matrix_root_eigen(
            A=A,
            root=2,
            epsilon=0.0,
            make_positive_semidefinite=True,
            inverse=False,
            retry_double_precision=True,
        )
        torch.testing.assert_allclose(X, torch.eye(2))
        mock_eigh.assert_called()
        self.assertEqual(mock_eigh.call_count, 2)


class NewtonRootInverseTest(unittest.TestCase):
    def _test_newton_root_inverse(
        self, A, root, epsilon, max_iterations, A_tol, M_tol
    ) -> None:
        X, M, flag, iteration, M_error = _matrix_inverse_root_newton(
            A, root, epsilon, max_iterations, M_tol
        )
        abs_A_error = torch.dist(torch.linalg.matrix_power(X, -root), A, p=torch.inf)
        A_norm = torch.linalg.norm(A, ord=torch.inf)
        rel_A_error = abs_A_error / torch.maximum(torch.tensor(1.0), A_norm)
        self.assertLessEqual(M_error, M_tol)
        self.assertLessEqual(rel_A_error, A_tol)

    def _test_newton_root_inverse_multi_dim(
        self, A, dims, roots, epsilons, max_iterations, A_tol, M_tol
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

        def A(n):
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

                def A(n):
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
                    A, dims, roots, epsilons, max_iterations, A_tol, M_tol
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

                def A(n):
                    diag = alpha * torch.ones(n)
                    diag[0] -= beta
                    off_diag = beta * torch.ones(n - 1)
                    return (
                        torch.diag(diag)
                        + torch.diag(off_diag, diagonal=1)
                        + torch.diag(off_diag, diagonal=-1)
                    )

                self._test_newton_root_inverse_multi_dim(
                    A, dims, roots, epsilons, max_iterations, A_tol, M_tol
                )


class ComputeMatrixRootInverseResidualsTest(unittest.TestCase):
    def test_matrix_root_inverse_residuals_with_not_two_dim_matrix(self):
        A = torch.zeros((1, 2, 3))
        X_hat = torch.zeros((2, 2))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            "Matrix is not 2-dimensional!",
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
            exponent_multiplier=exponent_multiplier,
        )

    def test_matrix_root_inverse_residuals_with_not_square_matrix(self):
        A = torch.zeros((1, 2))
        X_hat = torch.zeros((2, 2))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            "Matrix is not square!",
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
            exponent_multiplier=exponent_multiplier,
        )

    def test_matrix_root_inverse_residuals_with_inconsistent_dims(self):
        A = torch.zeros((2, 2))
        X_hat = torch.zeros((3, 3))
        root = 4
        exponent_multiplier = 1.82
        self.assertRaisesRegex(
            ValueError,
            "Matrix shapes do not match!",
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
    ):
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

    def test_matrix_root_inverse_residuals(self):
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
