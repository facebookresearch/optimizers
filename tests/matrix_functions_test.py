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
    _matrix_inverse_root_eigen,
    _matrix_inverse_root_higher_order,
    _matrix_inverse_root_newton,
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_eigenvectors,
    matrix_inverse_root,
    NewtonConvergenceFlag,
)
from matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    EigenConfig,
    EigenvalueCorrectionConfig,
    RootInvConfig,
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
        root = 2.0
        exp = 1.82
        with self.subTest("Test with scalar case."):
            self.assertEqual(
                A ** torch.tensor(-exp / root),
                matrix_inverse_root(
                    A,
                    root=Fraction(root / exp),
                ),
            )
        with self.subTest("Test with matrix case."):
            self.assertEqual(
                torch.tensor([[A ** torch.tensor(-exp / root)]]),
                matrix_inverse_root(
                    torch.tensor([[A]]),
                    root=Fraction(root / exp),
                ),
            )

    def test_matrix_inverse_root_with_not_two_dim_matrix(self) -> None:
        A = torch.zeros((1, 2, 3))
        root = Fraction(4)
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            matrix_inverse_root,
            A=A,
            root=root,
            is_diagonal=False,
        )

    def test_matrix_inverse_root_not_square(self) -> None:
        A = torch.zeros((2, 3))
        root = Fraction(4)
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            matrix_inverse_root,
            A=A,
            root=root,
            is_diagonal=False,
        )

    def test_matrix_inverse_root(self) -> None:
        A_list = [
            # A diagonal matrix.
            torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            # Non-diagonal matrix.
            torch.tensor(
                [
                    [1195.0, -944.0, -224.0],
                    [-944.0, 746.0, 177.0],
                    [-224.0, 177.0, 42.0],
                ]
            ),
        ]
        root = Fraction(2)
        expected_root_list = [
            torch.tensor([[1.0, 0.0], [0.0, 0.5]]),
            torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, -3.0], [1.0, -3.0, 18.0]]),
        ]

        atol = 0.05
        rtol = 1e-2
        with self.subTest("Test with diagonal case."):
            torch.testing.assert_close(
                expected_root_list[0],
                matrix_inverse_root(
                    A_list[0],
                    root=root,
                    is_diagonal=True,
                ),
                atol=atol,
                rtol=rtol,
            )

        for A, expected_root in zip(A_list, expected_root_list):
            for root_inv_config in (EigenConfig(), CoupledNewtonConfig()):
                with self.subTest(f"Test with {A=}, {root_inv_config=}"):
                    torch.testing.assert_close(
                        expected_root,
                        matrix_inverse_root(
                            A=A,
                            root=root,
                            is_diagonal=False,
                            root_inv_config=root_inv_config,
                        ),
                        atol=atol,
                        rtol=rtol,
                    )

            for order in range(2, 7):
                with self.subTest(f"Test HIGHER_ORDER with {A=}, {order=}"):
                    torch.testing.assert_close(
                        expected_root,
                        matrix_inverse_root(
                            A=A,
                            root=root,
                            is_diagonal=False,
                            root_inv_config=CoupledHigherOrderConfig(order=order),
                        ),
                        atol=atol,
                        rtol=rtol,
                    )
                    # Also test that powering works
                    exp = 2
                    torch.testing.assert_close(
                        torch.linalg.matrix_power(expected_root, exp),
                        matrix_inverse_root(
                            A,
                            root=Fraction(root // exp),
                            root_inv_config=CoupledHigherOrderConfig(order=order),
                            is_diagonal=False,
                        ),
                        atol=atol,
                        rtol=rtol,
                    )

    def test_matrix_inverse_root_higher_order_blowup(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
        self.assertRaisesRegex(
            ArithmeticError,
            re.escape(
                "NaN/Inf in matrix inverse root (after powering for fractions), raising an exception!"
            ),
            _matrix_inverse_root_higher_order,
            A=A,
            root=Fraction(1, 20),
        )

    def test_matrix_inverse_root_with_no_effect_exponent_multiplier(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        exp = 3
        self.assertRaisesRegex(
            ValueError,
            re.escape(
                f"root.denominator={exp} must be equal to 1 to use coupled inverse Newton iteration!"
            ),
            matrix_inverse_root,
            A=A,
            root=Fraction(2, exp),
            root_inv_config=CoupledNewtonConfig(),
        )

    def test_matrix_inverse_root_reach_max_iterations(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(4)
        root_inv_config_and_implementation_and_msg: List[
            Tuple[RootInvConfig, str, str]
        ] = [
            (CoupledNewtonConfig(), "_matrix_inverse_root_newton", "Newton"),
            (
                CoupledHigherOrderConfig(),
                "_matrix_inverse_root_higher_order",
                "Higher order method",
            ),
        ]
        for (
            root_inv_config,
            implementation,
            msg,
        ) in root_inv_config_and_implementation_and_msg:
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
                root_inv_config=root_inv_config,
                implementation=implementation,
                msg=msg,
            ), self.assertLogs(
                level="WARNING",
            ) as cm:
                matrix_inverse_root(
                    A=A,
                    root=root,
                    root_inv_config=root_inv_config,
                )
                self.assertIn(
                    f"{msg} did not converge and reached maximum number of iterations!",
                    [r.msg for r in cm.records],
                )

    def test_matrix_inverse_root_higher_order_tf32_preservation(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, float("inf")]])
        root = Fraction(2)
        tf32_flag_before = torch.backends.cuda.matmul.allow_tf32
        self.assertRaisesRegex(
            ArithmeticError,
            re.escape("Input matrix has entries close to inf"),
            matrix_inverse_root,
            A=A,
            root=root,
            root_inv_config=CoupledHigherOrderConfig(),
        )
        tf32_flag_after = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(tf32_flag_before, tf32_flag_after)

    def test_matrix_inverse_root_higher_order_error_blowup_before_powering(
        self,
    ) -> None:
        # Trigger this error by using an ill-conditioned matrix.
        A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
        root = Fraction(2)
        with self.assertRaisesRegex(
            ArithmeticError,
            "Error in matrix inverse root \\(before powering for fractions\\) [+-]?([0-9]*[.])?[0-9]+ exceeds threshold 1e-1, raising an exception!",
        ):
            matrix_inverse_root(
                A=A,
                root=root,
                # Set max_iterations to 0 to fast forward to the error check before powering.
                root_inv_config=CoupledHigherOrderConfig(max_iterations=0),
            )

    def test_matrix_inverse_root_with_invalid_root_inv_config(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(4)
        with self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Root inverse config is not implemented! Specified root inverse config is root_inv_config=RootInvConfig()."
            ),
        ):
            matrix_inverse_root(
                A=A,
                root=root,
                root_inv_config=RootInvConfig(),
                is_diagonal=False,
            )


class MatrixRootDiagonalTest(unittest.TestCase):
    def test_matrix_root_diagonal_nonpositive_root(self) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        for root in (-1, 0):
            with self.subTest(f"With {root=}"):
                self.assertRaisesRegex(
                    ValueError,
                    re.escape(f"Root {root} should be positive!"),
                    matrix_inverse_root,
                    A=A,
                    root=root,
                    is_diagonal=True,
                )


class EigenRootTest(unittest.TestCase):
    def _test_eigen_root(
        self,
        A: torch.Tensor,
        root: int,
        make_positive_semidefinite: bool,
        epsilon: float,
        tolerance: float,
        eig_sols: Tensor,
    ) -> None:
        X, L, Q = _matrix_inverse_root_eigen(
            A=A,
            root=Fraction(root),
            epsilon=epsilon,
            make_positive_semidefinite=make_positive_semidefinite,
        )
        abs_error = torch.dist(torch.linalg.matrix_power(X, -root), A, p=torch.inf)
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
            matrix_inverse_root,
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
            matrix_inverse_root(
                A=A,
                root=Fraction(2),
                root_inv_config=EigenConfig(retry_double_precision=False),
                epsilon=0.0,
            )
        mock_eigh.assert_called_once()

    @mock.patch.object(
        torch_lianlg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_retry_double_precision_raise_exception(self, mock_eigh: mock.Mock) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        with self.assertRaisesRegex(RuntimeError, re.escape("Mock Eigen Error")):
            matrix_inverse_root(
                A=A,
                root=Fraction(2),
                epsilon=0.0,
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
        X = matrix_inverse_root(
            A=A,
            root=Fraction(2),
            epsilon=0.0,
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
        X, _, _, _, M_error = _matrix_inverse_root_newton(
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


class CoupledHigherOrderRootInverseTest(unittest.TestCase):
    def test_root_with_big_numerator_denominator(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(13, 15)
        with self.assertLogs(
            level="WARNING",
        ) as cm:
            matrix_inverse_root(
                A=A,
                root=root,
                root_inv_config=CoupledHigherOrderConfig(),
            )
            self.assertIn(
                "abs(root.numerator)=13 and abs(root.denominator)=15 are probably too big for best performance.",
                [r.msg for r in cm.records],
            )


class ComputeMatrixRootInverseResidualsTest(unittest.TestCase):
    def test_matrix_root_inverse_residuals_with_not_two_dim_matrix(self) -> None:
        A = torch.zeros((1, 2, 3))
        X_hat = torch.zeros((2, 2))
        root = Fraction(4)
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
        )

    def test_matrix_root_inverse_residuals_with_not_square_matrix(self) -> None:
        A = torch.zeros((1, 2))
        X_hat = torch.zeros((2, 2))
        root = Fraction(4)
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
        )

    def test_matrix_root_inverse_residuals_with_inconsistent_dims(self) -> None:
        A = torch.zeros((2, 2))
        X_hat = torch.zeros((3, 3))
        root = Fraction(4)
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix shapes do not match!"),
            compute_matrix_root_inverse_residuals,
            A=A,
            X_hat=X_hat,
            root=root,
            epsilon=0.0,
        )

    def _test_matrix_root_inverse_residuals(
        self,
        A: torch.Tensor,
        X_hat: torch.Tensor,
        root: Fraction,
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

        with self.subTest("Exponent = 1."):
            exp = 1.0
            root = Fraction(2.0 / exp)
            self._test_matrix_root_inverse_residuals(
                A=A,
                X_hat=X_hat,
                root=root,
                expected_relative_error=expected_relative_error,
                expected_relative_residual=expected_relative_residual,
            )
        with self.subTest("Exponent != 1."):
            exp = 2.0
            root = Fraction(4.0 / exp)
            self._test_matrix_root_inverse_residuals(
                A=A,
                X_hat=X_hat,
                root=root,
                expected_relative_error=expected_relative_error,
                expected_relative_residual=expected_relative_residual,
            )


class MatrixEigenvectorsTest(unittest.TestCase):
    def test_matrix_eigenvectors_scalar(self) -> None:
        A = torch.tensor(2.0)
        with self.subTest("Test with scalar case."):
            self.assertEqual(
                torch.tensor(1),
                matrix_eigenvectors(A),
            )
        with self.subTest("Test with matrix case."):
            self.assertEqual(
                torch.tensor([[1]]),
                matrix_eigenvectors(torch.tensor([[A]])),
            )

    def test_matrix_eigenvectors_with_not_two_dim_matrix(self) -> None:
        A = torch.zeros((1, 2, 3))
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            matrix_eigenvectors,
            A=A,
        )

    def test_matrix_eigenvectors_not_square(self) -> None:
        A = torch.zeros((2, 3))
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            matrix_eigenvectors,
            A=A,
        )

    def test_matrix_eigenvectors(self) -> None:
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
        expected_eigenvectors_list = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor(
                [
                    [0.0460, -0.6287, 0.7763],
                    [-0.1752, -0.7702, -0.6133],
                    [0.9835, -0.1078, -0.1455],
                ]
            ),
        ]

        atol = 0.05
        rtol = 1e-2
        with self.subTest("Test with diagonal case."):
            torch.testing.assert_close(
                expected_eigenvectors_list[0],
                matrix_eigenvectors(
                    A_list[0],
                    is_diagonal=True,
                ),
                atol=atol,
                rtol=rtol,
            )
        with self.subTest("Test with EIGEN."):
            for A, expected_eigenvectors in zip(A_list, expected_eigenvectors_list):
                torch.testing.assert_close(
                    expected_eigenvectors,
                    matrix_eigenvectors(
                        A,
                        is_diagonal=False,
                    ),
                    atol=atol,
                    rtol=rtol,
                )

    def test_invalid_eigenvalue_correction_config(
        self,
    ) -> None:
        with mock.patch.object(
            matrix_functions,
            "type",
            side_effect=lambda object: EigenvalueCorrectionConfig,
        ), self.assertRaisesRegex(
            NotImplementedError,
            re.escape(
                "Eigenvector computation method is not implemented! Specified eigenvector method is eigenvector_computation_config=EighEigenvalueCorrectionConfig(retry_double_precision=True)."
            ),
        ):
            matrix_eigenvectors(
                A=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            )
