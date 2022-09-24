"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest
from typing import Optional, Tuple

import torch
from ..shampoo_utils import (
    AdagradPreconditioner,
    GraftingType,
    merge_small_dims,
    multi_dim_cat,
    multi_dim_split,
    RootInvStrategy,
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
                self.assertTrue(torch.equal(split_grad[idx], g))

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
        self.assertTrue(torch.equal(multi_dim_cat(split_grad, num_splits), grad))

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

    def _setup_test(self, beta2, epsilon, use_bias_correction) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], AdagradPreconditioner]:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        adagrad = AdagradPreconditioner(param, beta2=beta2, epsilon=epsilon, use_bias_correction=use_bias_correction)
        return param, loss, param.grad, adagrad

    def _test_update_preconditioners(self, beta2, use_bias_correction) -> None:
        param, loss, grad, adagrad = self._setup_test(beta2=beta2, epsilon=0.0, use_bias_correction=use_bias_correction)
        # pyre-fixme[58]: `**` is not supported for operand types
        #  `Optional[torch._tensor.Tensor]` and `int`.
        precond_sol = grad**2
        precond_sol = precond_sol if beta2 == 1.0 else (1.0 - beta2) * precond_sol
        bias_correction2 = torch.tensor(1.0 if not use_bias_correction else 1.0 - beta2)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        adagrad.update_preconditioners(grad)

        with self.subTest("Test preconditioner"):
            self.assertTrue(torch.allclose(adagrad._preconditioner, precond_sol))
        with self.subTest("Test bias correction"):
            self.assertTrue(
                torch.isclose(torch.tensor(adagrad._bias_correction2), bias_correction2)
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
        param, loss, grad, adagrad = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        preconditioned_grad = adagrad.precondition(grad)
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(preconditioned_grad, grad))

    def test_precondition_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(beta2=1.0, epsilon=0.0, use_bias_correction=False)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        adagrad.update_preconditioners(grad)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        preconditioned_grad = adagrad.precondition(grad)
        self.assertTrue(torch.allclose(preconditioned_grad, torch.ones(2)))

    def test_precondition_and_update_without_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False)
        with torch.no_grad():
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
            adagrad.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.allclose(param, torch.tensor([-1.0, -2.0])))

    def test_precondition_and_update_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(beta2=1.0, epsilon=0.0, use_bias_correction=False)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        adagrad.update_preconditioners(grad)
        with torch.no_grad():
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
            adagrad.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.allclose(param, torch.tensor([0.0, 1.0])))

    def test_compute_norm_without_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        norm = adagrad.compute_norm(grad)
        self.assertEqual(norm, torch.sqrt(torch.tensor(20.0)))

    def test_compute_norm_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(beta2=1.0, epsilon=0.0, use_bias_correction=False)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        adagrad.update_preconditioners(grad)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        norm = adagrad.compute_norm(grad)
        self.assertEqual(norm, torch.sqrt(torch.tensor(2.0)))


class ShampooPreconditionerTest(unittest.TestCase):
    def _setup_test(self, beta2, epsilon, use_bias_correction, start_preconditioning_step=0, diagonal_threshold=None, grafting_type=GraftingType.NONE, grafting_epsilon=1e-3) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], ShampooPreconditioner]:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        shampoo = ShampooPreconditioner(
            param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=use_bias_correction,
            dtype=torch.float,
            root_inv_strategy=RootInvStrategy.NONE,
            start_preconditioning_step=start_preconditioning_step,
            diagonal_threshold=diagonal_threshold,
            grafting_type=grafting_type,
            grafting_epsilon=grafting_epsilon,
        )
        return param, loss, param.grad, shampoo

    def _test_update_preconditioners(self, beta2, use_bias_correction) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=beta2, epsilon=0.0, use_bias_correction=use_bias_correction)
        preconditioner_sols = [
            param @ param.transpose(0, 1),
            param.transpose(0, 1) @ param,
        ]
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        shampoo.update_preconditioners(grad)

        for i, (preconditioner, preconditioner_sol) in enumerate(zip(shampoo._preconditioners, preconditioner_sols)):
            with self.subTest(f"Test preconditioner {i}"):
                self.assertTrue(
                    torch.allclose(preconditioner.factor_matrix, preconditioner_sol if beta2 == 1.0 else (1 - beta2) * preconditioner_sol)
                )
        with self.subTest("Test bias correction"):
            self.assertTrue(
                torch.isclose(torch.tensor(shampoo._bias_correction2), torch.tensor(1.0 if not use_bias_correction else 1.0 - beta2))
            )
        with self.subTest("Test number of updates"):
            self.assertEqual(shampoo._num_updates, 1)

    def test_update_preconditioners_adagrad(self) -> None:
        self._test_update_preconditioners(beta2=1.0, use_bias_correction=False)

    def test_update_preconditioners_rmsprop(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=False)

    def test_update_preconditioners_adam(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=True)

    def test_precondition_no_root_inverse(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, start_preconditioning_step=-1)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        preconditioned_grad = shampoo.precondition(grad)
        self.assertTrue(
            torch.allclose(preconditioned_grad, torch.zeros((2, 3)))
        )

    def test_precondition_with_root_inverse(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, start_preconditioning_step=-1)
        shampoo.compute_root_inverse(rank=-1, group=None)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        preconditioned_grad = shampoo.precondition(grad)
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(preconditioned_grad, grad))

    def test_precondition_with_diagonal_threshold(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2)
        shampoo.compute_root_inverse(rank=-1, group=None)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        preconditioned_grad = shampoo.precondition(grad)
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(preconditioned_grad, grad))

    def test_precondition_with_grafting(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, grafting_type=GraftingType.ADAGRAD, grafting_epsilon=1.0)
        shampoo.compute_root_inverse(rank=-1, group=None)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        preconditioned_grad = shampoo.precondition(grad)
        # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
        self.assertTrue(torch.allclose(preconditioned_grad, grad))

    def test_precondition_and_update(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, start_preconditioning_step=-1)
        shampoo.compute_root_inverse(rank=-1, group=None)
        with torch.no_grad():
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
            shampoo.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.allclose(param, torch.zeros((2, 3))))

    def test_precondition_and_update_with_grafting(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, grafting_type=GraftingType.ADAGRAD, grafting_epsilon=1.0)
        shampoo.compute_root_inverse(rank=-1, group=None)
        with torch.no_grad():
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got `Optional[Tensor]`.
            shampoo.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.allclose(param, torch.zeros((2, 3))))

    def test_compute_norm(self) -> None:
        param, loss, grad, shampoo = self._setup_test(beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2)
        shampoo.compute_root_inverse(rank=-1, group=None)
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Optional[Tensor]`.
        norm = shampoo.compute_norm(grad)
        self.assertTrue(torch.isclose(norm, torch.sqrt(torch.tensor(55.0))))


if __name__ == "__main__":
    unittest.main()
