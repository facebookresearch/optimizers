"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
import re
import unittest
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from types import ModuleType
from unittest import mock

import matrix_functions

import numpy as np

import torch
from matrix_functions import (
    _check_square_matrix,
    _matrix_inverse_root_eigen,
    _matrix_inverse_root_newton,
    _matrix_perturbation,
    check_diagonal,
    compute_matrix_root_inverse_residuals,
    matrix_eigendecomposition,
    matrix_inverse_root,
    NewtonConvergenceFlag,
    stabilize_and_pow_eigenvalues,
)
from matrix_functions_types import (
    CoupledHigherOrderConfig,
    CoupledNewtonConfig,
    DefaultEigendecompositionConfig,
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
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class CheckSquareMatrixTest(unittest.TestCase):
    @staticmethod
    @_check_square_matrix
    def check_square_matrix_func(A: Tensor) -> Tensor:
        """Helper function decorated with _check_square_matrix for testing."""
        return A

    def test_check_square_matrix_for_not_two_dim_matrix(self) -> None:
        # Test with a 3D tensor
        A = torch.zeros((2, 2, 2))
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not 2-dimensional!"),
            CheckSquareMatrixTest.check_square_matrix_func,
            A=A,
        )

    def test_check_square_matrix_for_not_square_matrix(self) -> None:
        A = torch.zeros((2, 3))
        self.assertRaisesRegex(
            ValueError,
            re.escape("Matrix is not square!"),
            CheckSquareMatrixTest.check_square_matrix_func,
            A=A,
        )

    def test_check_square_matrix_for_square_matrix(self) -> None:
        A = torch.eye(2)
        # Verify the function was called and returned the input
        torch.testing.assert_close(
            A, CheckSquareMatrixTest.check_square_matrix_func(A=A)
        )


class CheckDiagonalTest(unittest.TestCase):
    def test_check_diagonal_for_diagonal_matrix(self) -> None:
        self.assertTrue(check_diagonal(A=torch.eye(2)))


@instantiate_parametrized_tests
class MatrixPerturbationTest(unittest.TestCase):
    def test_matrix_perturbation_not_is_eigenvalues(self) -> None:
        A = torch.eye(2)
        torch.testing.assert_close(
            A * 1.1, _matrix_perturbation(A=A, epsilon=0.1, is_eigenvalues=False)
        )

    @parametrize("A", (torch.ones(5), torch.eye(2)))
    def test_matrix_perturbation_is_eigenvalues(self, A: Tensor) -> None:
        torch.testing.assert_close(
            A + 0.1, _matrix_perturbation(A=A, epsilon=0.1, is_eigenvalues=True)
        )


@instantiate_parametrized_tests
class StabilizeAndPowEigenvaluesTest(unittest.TestCase):
    @parametrize("perturb_before_computation", (True, False))
    def test_stabilize_and_pow_eigenvalues_perturbation(
        self, perturb_before_computation: bool
    ) -> None:
        L = torch.tensor([0.1, 3.1])
        # The stabilized eigenvalues is [1.0, 4.0] and with root = 2,
        # that is why expected output is [1.0, 0.5]
        torch.testing.assert_close(
            torch.tensor([1.0, 0.5]),
            stabilize_and_pow_eigenvalues(
                L=L,
                root=Fraction(2),
                epsilon=1.0 - torch.min(L).item() * (not perturb_before_computation),
                rank_deficient_stability_config=PerturbationConfig(
                    perturb_before_computation=perturb_before_computation
                ),
            ),
        )

    @parametrize("rank_rtol", (None, 1e-6))
    def test_stabilize_and_pow_eigenvalues_pseudoinverse(
        self, rank_rtol: float | None
    ) -> None:
        L = torch.tensor([1.0, 4.0, 0.0])
        torch.testing.assert_close(
            torch.tensor([1.0, 0.5, 0.0]),
            stabilize_and_pow_eigenvalues(
                L=L,
                root=Fraction(2),
                epsilon=0.0,
                rank_deficient_stability_config=PseudoInverseConfig(
                    rank_rtol=rank_rtol
                ),
            ),
        )

    def test_pseudoinverse_with_invalid_epsilon(self) -> None:
        L = torch.tensor([1.0, 4.0, 0.0])
        epsilon = 1e-8
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"{epsilon=} should be 0.0 when using pseudo-inverse!"),
            stabilize_and_pow_eigenvalues,
            L=L,
            root=Fraction(2),
            epsilon=epsilon,
            rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None),
        )

    def test_invalid_rank_deficient_stability_config(self) -> None:
        @dataclass
        class NotSupportedRankDeficientStabilityConfig(RankDeficientStabilityConfig):
            """A dummy rank_deficient_stability_config that is not supported."""

            unsupported_mode: str = ""

        L = torch.tensor([1.0, 4.0, 0.0])
        self.assertRaisesRegex(
            NotImplementedError,
            r"rank_deficient_stability_config=.*\.NotSupportedRankDeficientStabilityConfig\(.*\) is not supported\.",
            stabilize_and_pow_eigenvalues,
            L=L,
            root=Fraction(2),
            rank_deficient_stability_config=NotSupportedRankDeficientStabilityConfig(),
        )


@instantiate_parametrized_tests
class MatrixInverseRootTest(unittest.TestCase):
    @parametrize(
        "root_inv_config",
        (
            EigenConfig(),  # perturb_before_computation=True by default
            CoupledNewtonConfig(),
            EigenConfig(
                rank_deficient_stability_config=PerturbationConfig(
                    perturb_before_computation=False
                )
            ),
            EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None)
            ),  # equivalent behavior when test matrices are full rank
            EigenConfig(eigendecomposition_offload_device="cpu"),
            *(
                CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0, order=order)
                for order in range(2, 7)
            ),
        ),
    )
    @parametrize("exp", (1, 2))
    @parametrize(
        "A, expected_root",
        (
            # A diagonal matrix.
            (
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([[1.0, 0.0], [0.0, 0.5]]),
            ),
            # Non-diagonal matrix.
            (
                torch.tensor(
                    [
                        [1195.0, -944.0, -224.0],
                        [-944.0, 746.0, 177.0],
                        [-224.0, 177.0, 42.0],
                    ]
                ),
                torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, -3.0], [1.0, -3.0, 18.0]]),
            ),
        ),
    )
    def test_matrix_inverse_root(
        self, A: Tensor, expected_root: Tensor, exp: int, root_inv_config: RootInvConfig
    ) -> None:
        atol = 0.05
        rtol = 1e-2

        torch.testing.assert_close(
            torch.linalg.matrix_power(expected_root, exp),
            matrix_inverse_root(
                A=A,
                root=Fraction(2, exp),
                is_diagonal=False,
                root_inv_config=root_inv_config,
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
            matrix_inverse_root,
            A=A,
            root=Fraction(1, 20),
            root_inv_config=CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0),
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

    @parametrize(
        "root_inv_config, implementation, msg",
        [
            (CoupledNewtonConfig(), "_matrix_inverse_root_newton", "Newton"),
            (
                CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0),
                "_matrix_inverse_root_higher_order",
                "Higher order method",
            ),
        ],
    )
    def test_matrix_inverse_root_reach_max_iterations(
        self, root_inv_config: RootInvConfig, implementation: str, msg: str
    ) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(4)
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
            root_inv_config=CoupledHigherOrderConfig(rel_epsilon=0.0, abs_epsilon=0.0),
        )
        tf32_flag_after = torch.backends.cuda.matmul.allow_tf32
        self.assertEqual(tf32_flag_before, tf32_flag_after)

    def test_matrix_inverse_root_higher_order_error_blowup_before_powering(
        self,
    ) -> None:
        # Trigger this error by using an ill-conditioned matrix.
        A = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
        root = Fraction(2)
        self.assertRaisesRegex(
            ArithmeticError,
            r"Error in matrix inverse root \(before powering for fractions\) [+-]?([0-9]*[.])?[0-9]+ exceeds threshold 1e-1, raising an exception!",
            matrix_inverse_root,
            A=A,
            root=root,
            # Set max_iterations to 0 to fast forward to the error check before powering.
            root_inv_config=CoupledHigherOrderConfig(
                rel_epsilon=0.0, abs_epsilon=0.0, max_iterations=0
            ),
        )

    def test_matrix_inverse_root_with_invalid_root_inv_config(self) -> None:
        @dataclass
        class NotSupportedRootInvConfig(RootInvConfig):
            """A dummy root inv config that is not supported."""

            unsupported_root: int = -1

        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(4)
        self.assertRaisesRegex(
            NotImplementedError,
            r"Root inverse config is not implemented! Specified root inverse config is root_inv_config=.*\.NotSupportedRootInvConfig\(.*\)\.",
            matrix_inverse_root,
            A=A,
            root=root,
            root_inv_config=NotSupportedRootInvConfig(),
            is_diagonal=False,
        )


@instantiate_parametrized_tests
class MatrixRootDiagonalTest(unittest.TestCase):
    @parametrize("root", (-1, 0))
    def test_matrix_root_diagonal_nonpositive_root(self, root: int) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Root {root} should be positive!"),
            matrix_inverse_root,
            A=A,
            root=root,
            is_diagonal=True,
        )

    def test_matrix_root(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        root = Fraction(2)
        expected_root_list = torch.tensor([[1.0, 0.0], [0.0, 0.5]])

        torch.testing.assert_close(
            expected_root_list,
            matrix_inverse_root(
                A=A,
                root=root,
                is_diagonal=True,
            ),
        )


feasible_alpha_beta_pairs: tuple[tuple[float, float], ...] = tuple(
    (alpha, beta)
    for alpha, beta in itertools.product((0.001, 0.01, 0.1, 1.0, 10.0, 100.0), repeat=2)
    if 2 * beta <= alpha
)


@instantiate_parametrized_tests
class EigenRootTest(unittest.TestCase):
    def _test_eigen_root_multi_dim(
        self,
        A: Callable[[int], Tensor],
        n: int,
        root: int,
        epsilon: float,
        tolerance: float,
        eig_sols: Callable[[int], Tensor],
    ) -> None:
        X, L, _ = _matrix_inverse_root_eigen(
            A=A(n),
            root=Fraction(root),
            epsilon=epsilon,
        )
        abs_error = torch.dist(torch.linalg.matrix_power(X, -root), A(n), p=torch.inf)
        A_norm = torch.linalg.norm(A(n), ord=torch.inf)
        rel_error = abs_error / torch.maximum(torch.tensor(1.0), A_norm)
        torch.testing.assert_close(L, eig_sols(n))
        self.assertLessEqual(rel_error.item(), tolerance)

    @parametrize("root", [1, 2, 4, 8])
    @parametrize("n", [10, 100])
    def test_eigen_root_identity(self, n: int, root: int) -> None:
        self._test_eigen_root_multi_dim(
            A=torch.eye,
            n=n,
            root=root,
            epsilon=0.0,
            tolerance=1e-6,
            eig_sols=torch.ones,
        )

    @parametrize("alpha, beta", feasible_alpha_beta_pairs)
    @parametrize("root", [1, 2, 4, 8])
    @parametrize("n", [10, 100])
    def test_eigen_root_tridiagonal(
        self, n: int, root: int, alpha: float, beta: float
    ) -> None:
        def eig_sols_tridiagonal_1(n: int, alpha: float, beta: float) -> Tensor:
            eigs = alpha * torch.ones(n) + 2 * beta * torch.tensor(
                [np.cos(j * torch.pi / n) for j in range(n)], dtype=torch.float
            )
            eigs, _ = torch.sort(eigs)
            return eigs

        def A_tridiagonal_1(n: int, alpha: float, beta: float) -> Tensor:
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
            A=partial(A_tridiagonal_1, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            tolerance=1e-4,
            eig_sols=partial(eig_sols_tridiagonal_1, alpha=alpha, beta=beta),
        )

        def eig_sols_tridiagonal_2(n: int, alpha: float, beta: float) -> Tensor:
            eigs = alpha * torch.ones(n) + 2 * beta * torch.tensor(
                [np.cos(2 * j * torch.pi / (2 * n + 1)) for j in range(1, n + 1)],
                dtype=torch.float,
            )
            eigs, _ = torch.sort(eigs)
            return eigs

        def A_tridiagonal_2(n: int, alpha: float, beta: float) -> Tensor:
            diag = alpha * torch.ones(n)
            diag[0] -= beta
            off_diag = beta * torch.ones(n - 1)
            return (
                torch.diag(diag)
                + torch.diag(off_diag, diagonal=1)
                + torch.diag(off_diag, diagonal=-1)
            )

        self._test_eigen_root_multi_dim(
            A=partial(A_tridiagonal_2, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            tolerance=1e-4,
            eig_sols=partial(eig_sols_tridiagonal_2, alpha=alpha, beta=beta),
        )

    def test_eigen_root_nonfull_rank(self) -> None:
        A = torch.tensor([[2.0, 1.0], [2.0, 1.0]])
        root = Fraction(2)
        epsilon = 0.0

        M_default = matrix_inverse_root(
            A=A, root=root, root_inv_config=EigenConfig(), epsilon=epsilon
        )
        self.assertTrue(torch.all(torch.isinf(M_default)))

        M_pseudoinverse = matrix_inverse_root(
            A=A,
            root=root,
            root_inv_config=EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None)
            ),
            epsilon=epsilon,
        )
        self.assertTrue(torch.all(torch.isreal(M_pseudoinverse)))

    def test_matrix_root_eigen_nonpositive_root(self) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        root = Fraction(-1)
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"Root {root} should be positive!"),
            matrix_inverse_root,
            A=A,
            root=root,
        )

    def test_pseudoinverse_with_invalid_epsilon(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        epsilon = 1e-8
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"{epsilon=} should be 0.0 when using pseudo-inverse!"),
            matrix_inverse_root,
            A=A,
            root=Fraction(2),
            epsilon=epsilon,
            root_inv_config=EigenConfig(
                rank_deficient_stability_config=PseudoInverseConfig()
            ),
        )

    torch_linalg_module: ModuleType = torch.linalg

    @mock.patch.object(
        torch_linalg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_no_retry_double_precision_raise_exception(
        self, mock_eigh: mock.Mock
    ) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        self.assertRaisesRegex(
            RuntimeError,
            re.escape("Mock Eigen Error"),
            matrix_inverse_root,
            A=A,
            root=Fraction(2),
            root_inv_config=EigenConfig(retry_double_precision=False),
            epsilon=0.0,
        )
        mock_eigh.assert_called_once()

    @mock.patch.object(
        torch_linalg_module, "eigh", side_effect=RuntimeError("Mock Eigen Error")
    )
    def test_retry_double_precision_raise_exception(self, mock_eigh: mock.Mock) -> None:
        A = torch.tensor([[-1.0, 0.0], [0.0, 2.0]])
        self.assertRaisesRegex(
            RuntimeError,
            re.escape("Mock Eigen Error"),
            matrix_inverse_root,
            A=A,
            root=Fraction(2),
            epsilon=0.0,
        )
        mock_eigh.assert_called()
        self.assertEqual(mock_eigh.call_count, 2)

    @mock.patch.object(
        torch_linalg_module,
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


@instantiate_parametrized_tests
class NewtonRootInverseTest(unittest.TestCase):
    def _test_newton_root_inverse_multi_dim(
        self,
        A: Callable[[int], Tensor],
        n: int,
        root: int,
        epsilon: float,
        max_iterations: int,
        A_tol: float,
        M_tol: float,
    ) -> None:
        X, _, _, _, M_error = _matrix_inverse_root_newton(
            A(n), root, epsilon, max_iterations, M_tol
        )
        abs_A_error = torch.dist(torch.linalg.matrix_power(X, -root), A(n), p=torch.inf)
        A_norm = torch.linalg.norm(A(n), ord=torch.inf)
        rel_A_error = abs_A_error / torch.maximum(torch.tensor(1.0), A_norm)
        self.assertLessEqual(M_error.item(), M_tol)
        self.assertLessEqual(rel_A_error.item(), A_tol)

    @parametrize("root", [2, 4, 8])
    @parametrize("n", [10, 100])
    def test_newton_root_inverse_identity(self, n: int, root: int) -> None:
        max_iterations = 1000

        self._test_newton_root_inverse_multi_dim(
            A=torch.eye,
            n=n,
            root=root,
            epsilon=0.0,
            max_iterations=max_iterations,
            A_tol=1e-6,
            M_tol=1e-6,
        )

    @parametrize("alpha, beta", feasible_alpha_beta_pairs)
    @parametrize("root", [2, 4, 8])
    @parametrize("n", [10, 100])
    def test_newton_root_inverse_tridiagonal(
        self, n: int, root: int, alpha: float, beta: float
    ) -> None:
        max_iterations = 1000

        def A_tridiagonal_1(n: int, alpha: float, beta: float) -> Tensor:
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
            A=partial(A_tridiagonal_1, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            max_iterations=max_iterations,
            A_tol=1e-4,
            M_tol=1e-6,
        )

        def A_tridiagonal_2(n: int, alpha: float, beta: float) -> Tensor:
            diag = alpha * torch.ones(n)
            diag[0] -= beta
            off_diag = beta * torch.ones(n - 1)
            return (
                torch.diag(diag)
                + torch.diag(off_diag, diagonal=1)
                + torch.diag(off_diag, diagonal=-1)
            )

        self._test_newton_root_inverse_multi_dim(
            A=partial(A_tridiagonal_2, alpha=alpha, beta=beta),
            n=n,
            root=root,
            epsilon=0.0,
            max_iterations=max_iterations,
            A_tol=1e-4,
            M_tol=1e-6,
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
                root_inv_config=CoupledHigherOrderConfig(
                    rel_epsilon=0.0, abs_epsilon=0.0
                ),
            )
        self.assertIn(
            "abs(root.numerator)=13 and abs(root.denominator)=15 are probably too big for best performance.",
            [r.msg for r in cm.records],
        )


@instantiate_parametrized_tests
class ComputeMatrixRootInverseResidualsTest(unittest.TestCase):
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

    @parametrize("root", (Fraction(2, 1), Fraction(4, 2)))
    def test_matrix_root_inverse_residuals(self, root: Fraction) -> None:
        A = torch.eye(2)
        X_hat = torch.eye(2)
        expected_relative_error = torch.tensor(0.0, dtype=torch.float64)
        expected_relative_residual = torch.tensor(0.0, dtype=torch.float64)

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


@instantiate_parametrized_tests
class MatrixEigendecompositionTest(unittest.TestCase):
    def test_pseudoinverse_with_invalid_epsilon(self) -> None:
        A = torch.ones((2, 2))
        epsilon = 1e-8
        self.assertRaisesRegex(
            ValueError,
            re.escape(f"{epsilon=} should be 0.0 when using pseudo-inverse!"),
            matrix_eigendecomposition,
            A=A,
            epsilon=epsilon,
            eigendecomposition_config=EighEigendecompositionConfig(
                rank_deficient_stability_config=PseudoInverseConfig(rank_rtol=None)
            ),
        )

    @parametrize(
        "eigendecomposition_config",
        (
            DefaultEigendecompositionConfig,
            EighEigendecompositionConfig(eigendecomposition_offload_device="cpu"),
        ),
    )
    @parametrize(
        "A, expected_eigenvalues, expected_eigenvectors",
        (
            # A diagonal matrix.
            (
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([1.0, 4.0]),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            ),
            # Non-diagonal matrix.
            (
                torch.tensor(
                    [
                        [1195.0, -944.0, -224.0],
                        [-944.0, 746.0, 177.0],
                        [-224.0, 177.0, 42.0],
                    ]
                ),
                torch.tensor([2.9008677229e-03, 1.7424316704e-01, 1.9828229980e03]),
                torch.tensor(
                    [
                        [0.0460073575, -0.6286827326, 0.7762997746],
                        [-0.1751257628, -0.7701635957, -0.6133345366],
                        [0.9834705591, -0.1077321917, -0.1455317289],
                    ]
                ),
            ),
        ),
    )
    def test_matrix_eigendecomposition(
        self,
        A: Tensor,
        expected_eigenvalues: Tensor,
        expected_eigenvectors: Tensor,
        eigendecomposition_config: EigendecompositionConfig,
    ) -> None:
        atol = 1e-4
        rtol = 1e-5

        torch.testing.assert_close(
            (expected_eigenvalues, expected_eigenvectors),
            matrix_eigendecomposition(
                A=A,
                eigendecomposition_config=eigendecomposition_config,
                is_diagonal=False,
            ),
            atol=atol,
            rtol=rtol,
        )

    @parametrize(
        "initialization_fn",
        (
            lambda A: torch.zeros_like(A),
            lambda A: torch.eye(A.shape[0], dtype=A.dtype, device=A.device),
            lambda A: matrix_eigendecomposition(A)[1],
        ),
    )
    @parametrize(
        "A, expected_eigenvalues, expected_eigenvectors",
        (
            # A diagonal matrix.
            (
                torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
                torch.tensor([1.0, 4.0]),
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            ),
            # Non-diagonal matrix.
            (
                torch.tensor(
                    [
                        [1195.0, -944.0, -224.0],
                        [-944.0, 746.0, 177.0],
                        [-224.0, 177.0, 42.0],
                    ]
                ),
                torch.tensor([2.9008677229e-03, 1.7424316704e-01, 1.9828229980e03]),
                torch.tensor(
                    [
                        [0.0460073575, -0.6286827326, 0.7762997746],
                        [-0.1751257628, -0.7701635957, -0.6133345366],
                        [0.9834705591, -0.1077321917, -0.1455317289],
                    ]
                ),
            ),
        ),
    )
    def test_matrix_eigendecomposition_with_qr(
        self,
        A: Tensor,
        expected_eigenvalues: Tensor,
        expected_eigenvectors: Tensor,
        initialization_fn: Callable[[Tensor], Tensor],
    ) -> None:
        atol = 2e-3
        rtol = 1e-5

        qr_config = QREigendecompositionConfig(max_iterations=10_000)
        qr_config.eigenvectors_estimate = initialization_fn(A)
        estimated_eigenvalues, estimated_eigenvectors = matrix_eigendecomposition(
            A=A,
            is_diagonal=False,
            eigendecomposition_config=qr_config,
        )

        # Ensure that the signs of the eigenvectors are consistent.
        estimated_eigenvectors[
            :,
            expected_eigenvectors[0, :] / estimated_eigenvectors[0, :] < 0,
        ] *= -1
        torch.testing.assert_close(
            (expected_eigenvalues, expected_eigenvectors),
            (estimated_eigenvalues, estimated_eigenvectors),
            atol=atol,
            rtol=rtol,
        )

    def test_invalid_eigendecomposition_config(self) -> None:
        @dataclass
        class NotSupportedEigendecompositionConfig(EigendecompositionConfig):
            """A dummy class eigendecomposition config that is not supported."""

            unsupoorted_field: int = 0

        self.assertRaisesRegex(
            NotImplementedError,
            r"Eigendecomposition config is not implemented! Specified eigendecomposition config is eigendecomposition_config=.*\.NotSupportedEigendecompositionConfig\(.*\).",
            matrix_eigendecomposition,
            A=torch.tensor([[1.0, 0.0], [0.0, 4.0]]),
            eigendecomposition_config=NotSupportedEigendecompositionConfig(),
        )


class MatrixEigendecompositionDiagonalTest(unittest.TestCase):
    def test_matrix_eigendecomposition(self) -> None:
        A = torch.tensor([[1.0, 0.0], [0.0, 4.0]])
        expected_eigenvalues, expected_eigenvectors = (
            torch.tensor([1.0, 4.0]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        torch.testing.assert_close(
            (expected_eigenvalues, expected_eigenvectors),
            matrix_eigendecomposition(
                A=A,
                is_diagonal=True,
            ),
        )
