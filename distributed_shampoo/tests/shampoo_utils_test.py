"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import math
import unittest
from typing import cast, Tuple

import torch

from distributed_shampoo.shampoo_utils import (
    AdagradPreconditioner,
    GraftingType,
    merge_small_dims,
    multi_dim_cat,
    multi_dim_split,
    ShampooPreconditioner,
)


class MergeSmallDimsTest(unittest.TestCase):
    def _test_merge_small_dims(
        self,
        dims,
        merged_dims,
        threshold,
    ) -> None:
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

    def test_merge_all_small_dims(self) -> None:
        dims = [1, 2, 5, 1]
        merged_dims = [10]
        threshold = 10
        self._test_merge_small_dims(dims, merged_dims, threshold)

    def test_merge_some_small_dims(self) -> None:
        dims = [1, 2, 5, 1]
        merged_dims = [2, 5]
        threshold = 1
        self._test_merge_small_dims(dims, merged_dims, threshold)

    def test_merge_small_dims_for_single_dim(self) -> None:
        dims = torch.tensor([2])
        merged_dims = [2]
        threshold = 10
        self._test_merge_small_dims(dims, merged_dims, threshold)


class MultiDimSplitTest(unittest.TestCase):
    def _test_multi_dim_split(self, grad, splits, split_grad) -> None:
        for idx, g in enumerate(multi_dim_split(grad, splits)):
            with self.subTest(f"Test with idx = {idx}"):
                torch.testing.assert_close(split_grad[idx], g)

    def test_multi_dim_split_for_one_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [3, 3]
        split_grad = [torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2)]
        self._test_multi_dim_split(grad, num_splits, split_grad)

    def test_multi_dim_split_for_two_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [3, 1]
        split_grad = [
            torch.arange(0, 6, 2).reshape(3, 1),
            torch.arange(0, 6, 2).reshape(3, 1) + 1,
            torch.arange(6, 10, 2).reshape(2, 1),
            torch.arange(6, 10, 2).reshape(2, 1) + 1,
        ]
        self._test_multi_dim_split(grad, num_splits, split_grad)

    def test_multi_dim_split_for_three_dim(self) -> None:
        grad = torch.arange(30).reshape(5, 2, 3)
        num_splits = [3, 1, 2]
        split_grad = [
            torch.tensor([[[0, 1]], [[6, 7]], [[12, 13]]]),
            torch.tensor([[[2]], [[8]], [[14]]]),
            torch.tensor([[[3, 4]], [[9, 10]], [[15, 16]]]),
            torch.tensor([[[5]], [[11]], [[17]]]),
            torch.tensor([[[18, 19]], [[24, 25]]]),
            torch.tensor([[[20]], [[26]]]),
            torch.tensor([[[21, 22]], [[27, 28]]]),
            torch.tensor([[[23]], [[29]]]),
        ]
        self._test_multi_dim_split(grad, num_splits, split_grad)


class MultiDimCatTest(unittest.TestCase):
    def _test_multi_dim_cat(self, grad, num_splits, split_grad) -> None:
        torch.testing.assert_close(multi_dim_cat(split_grad, num_splits), grad)

    def test_multi_dim_cat_for_one_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [2, 0]
        split_grad = [torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2)]
        self._test_multi_dim_cat(grad, num_splits, split_grad)

    def test_multi_dim_cat_for_two_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [2, 2]
        split_grad = [
            torch.arange(0, 6, 2).reshape(3, 1),
            torch.arange(0, 6, 2).reshape(3, 1) + 1,
            torch.arange(6, 10, 2).reshape(2, 1),
            torch.arange(6, 10, 2).reshape(2, 1) + 1,
        ]
        self._test_multi_dim_cat(grad, num_splits, split_grad)

    def test_multi_dim_cat_for_three_dim(self) -> None:
        grad = torch.arange(30).reshape(5, 2, 3)
        num_splits = [2, 2, 2]
        split_grad = [
            torch.tensor([[[0, 1]], [[6, 7]], [[12, 13]]]),
            torch.tensor([[[2]], [[8]], [[14]]]),
            torch.tensor([[[3, 4]], [[9, 10]], [[15, 16]]]),
            torch.tensor([[[5]], [[11]], [[17]]]),
            torch.tensor([[[18, 19]], [[24, 25]]]),
            torch.tensor([[[20]], [[26]]]),
            torch.tensor([[[21, 22]], [[27, 28]]]),
            torch.tensor([[[23]], [[29]]]),
        ]
        self._test_multi_dim_cat(grad, num_splits, split_grad)


class AdagradPreconditionerTest(unittest.TestCase):
    def _setup_test(
        self, beta2, epsilon, use_bias_correction
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, AdagradPreconditioner]:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        adagrad = AdagradPreconditioner(
            param, beta2=beta2, epsilon=epsilon, use_bias_correction=use_bias_correction
        )
        return param, loss, cast(torch.Tensor, param.grad), adagrad

    def _test_update_preconditioners(self, beta2, use_bias_correction) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=beta2, epsilon=0.0, use_bias_correction=use_bias_correction
        )
        precond_sol = grad ** torch.tensor(2)
        precond_sol = precond_sol if beta2 == 1.0 else (1.0 - beta2) * precond_sol
        bias_correction2 = torch.tensor(1.0 if not use_bias_correction else 1.0 - beta2)
        adagrad.update_preconditioners(grad)

        with self.subTest("Test preconditioner"):
            torch.testing.assert_close(adagrad._preconditioner, precond_sol)
        with self.subTest("Test bias correction"):
            torch.testing.assert_close(
                torch.tensor(adagrad._bias_correction2), bias_correction2
            )
        with self.subTest("Test number of updates"):
            self.assertEqual(adagrad._num_updates, 1)

    def test_update_preconditioners_adagrad(self) -> None:
        self._test_update_preconditioners(beta2=1.0, use_bias_correction=False)

    def test_update_preconditioners_rmsprop(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=False)

    def test_update_preconditioners_adam(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=True)

    def test_precondition_without_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        preconditioned_grad = adagrad.precondition(grad)
        torch.testing.assert_close(preconditioned_grad, grad)

    def test_precondition_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad)
        preconditioned_grad = adagrad.precondition(grad)
        torch.testing.assert_close(preconditioned_grad, torch.ones(2))

    def test_compute_norm_without_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        norm = adagrad.compute_norm(grad)
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(20.0)))

    def test_compute_norm_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad)
        norm = adagrad.compute_norm(grad)
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(2.0)))

    def test_to(self) -> None:
        _, _, _, adagrad = self._setup_test(
            beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        try:
            adagrad.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")


class ShampooPreconditionerTest(unittest.TestCase):
    def _setup_test(
        self,
        beta2,
        epsilon,
        use_bias_correction,
        exponent_override=0,
        start_preconditioning_step=0,
        diagonal_threshold=None,
        grafting_type=GraftingType.NONE,
        grafting_epsilon=1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ShampooPreconditioner]:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        shampoo = ShampooPreconditioner(
            param,
            beta2=beta2,
            epsilon=epsilon,
            exponent_override=exponent_override,
            use_bias_correction=use_bias_correction,
            dtype=torch.float,
            start_preconditioning_step=start_preconditioning_step,
            diagonal_threshold=diagonal_threshold,
            grafting_type=grafting_type,
            grafting_epsilon=grafting_epsilon,
        )
        return param, loss, cast(torch.Tensor, param.grad), shampoo

    def _test_update_preconditioners(
        self, beta2, use_bias_correction
    ) -> ShampooPreconditioner:
        param, loss, grad, shampoo = self._setup_test(
            beta2=beta2, epsilon=0.0, use_bias_correction=use_bias_correction
        )
        preconditioner_sols = [
            param @ param.transpose(0, 1),
            param.transpose(0, 1) @ param,
        ]
        shampoo.update_preconditioners(grad)

        for i, (preconditioner, preconditioner_sol) in enumerate(
            zip(shampoo._preconditioners, preconditioner_sols)
        ):
            with self.subTest(f"Test preconditioner {i}"):
                torch.testing.assert_close(
                    preconditioner.factor_matrix,
                    preconditioner_sol
                    if beta2 == 1.0
                    else (1 - beta2) * preconditioner_sol,
                )
        with self.subTest("Test bias correction"):
            torch.testing.assert_close(
                torch.tensor(shampoo._bias_correction2),
                torch.tensor(1.0 if not use_bias_correction else 1.0 - beta2),
            )
        with self.subTest("Test number of updates"):
            self.assertEqual(shampoo._num_updates, 1)

        return shampoo

    def test_update_preconditioners_adagrad(self) -> None:
        self._test_update_preconditioners(beta2=1.0, use_bias_correction=False)

    def test_update_preconditioners_rmsprop(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=False)

    def test_update_preconditioners_adam(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=True)

    def test_precondition_no_root_inverse(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        preconditioned_grad = shampoo.precondition(grad)
        torch.testing.assert_close(preconditioned_grad, torch.zeros((2, 3)))

    def test_precondition_with_root_inverse(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad)
        torch.testing.assert_close(preconditioned_grad, grad)

    def test_precondition_with_diagonal_threshold(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad)
        torch.testing.assert_close(preconditioned_grad, grad)

    def test_precondition_with_grafting(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad)
        torch.testing.assert_close(preconditioned_grad, grad)

    def test_compute_norm(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2
        )
        shampoo.compute_root_inverse()
        norm = shampoo.compute_norm(grad)
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(55.0)))

    def test_to(self) -> None:
        _, _, _, shampoo = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2
        )
        try:
            shampoo.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")

    def test_reset_preconditioners(self) -> None:
        shampoo = self._test_update_preconditioners(
            beta2=1.0, use_bias_correction=False
        )
        shampoo.reset_preconditioners()
        for i, preconditioner in enumerate(shampoo._preconditioners):
            with self.subTest(f"Test preconditioner {i}"):
                torch.testing.assert_allclose(
                    preconditioner.factor_matrix,
                    torch.zeros_like(preconditioner.factor_matrix),
                )

    def test_exponent_override(self) -> None:
        """
        To test, will update preconditioners using two gradients:

           G_1 = [[1, 0, 0], [0, 1, 0]]
           G_2 = [[0, 0, 0], [0, 0, 2]]

        Note that:

           L = G_1 G_1^T + G_2 G_2^T = [[1, 0], [0, 5]]
           R = G_1^T G_1 + G_2^T G_2 = [[1, 0, 0], [0, 1, 0], [0, 0, 4]]

        and we can compute the root inverse of these matrices.
        """

        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_bias_correction=False,
            exponent_override=2,
        )
        with self.subTest("Test stored exponent override int"):
            self.assertEqual(shampoo._exponent_override, 2)

        grad_1 = torch.zeros_like(grad)
        grad_1[0, 0] = 1
        grad_1[1, 1] = 1
        shampoo.update_preconditioners(grad_1)

        grad_2 = torch.zeros_like(grad)
        grad_2[1, 2] = 2
        shampoo.update_preconditioners(grad_2)

        shampoo.compute_root_inverse()
        left_root_inverse = torch.diag(torch.tensor([1.0, 1.0 / math.sqrt(5.0)]))
        right_root_inverse = torch.diag(torch.tensor([1.0, 1.0, 0.5]))

        with self.subTest("Test left root inverse matrix"):
            torch.testing.assert_close(
                shampoo._preconditioners[0].inv_factor_matrix, left_root_inverse
            )
        with self.subTest("Test right root inverse matrix"):
            torch.testing.assert_close(
                shampoo._preconditioners[1].inv_factor_matrix, right_root_inverse
            )

        true_preconditioned_grad = left_root_inverse @ grad @ right_root_inverse
        preconditioned_grad = shampoo.precondition(grad)
        with self.subTest("Test preconditioned grad"):
            torch.testing.assert_close(preconditioned_grad, true_preconditioned_grad)


if __name__ == "__main__":
    unittest.main()
