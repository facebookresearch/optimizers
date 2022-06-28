"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest

import torch
from ..shampoo_utils import (
    merge_small_dims,
    multi_dim_split,
    multi_dim_cat,
    AdagradPreconditioner,
    ShampooPreconditioner,
    GraftingType,
)


class MergeSmallDimsTest(unittest.TestCase):
    def _test_merge_small_dims(
        self,
        dims,
        merged_dims,
        threshold,
    ) -> None:
        self.assertEqual(merge_small_dims(dims, threshold), merged_dims)

    def test_merge_small_dims_1(self) -> None:
        dims = [1, 2, 5, 1]
        merged_dims = [10]
        threshold = 10
        self._test_merge_small_dims(dims, merged_dims, threshold)

    def test_merge_small_dims_2(self) -> None:
        dims = [1, 2, 5, 1]
        merged_dims = [2, 5]
        threshold = 1
        self._test_merge_small_dims(dims, merged_dims, threshold)

    def test_merge_small_dims_3(self) -> None:
        dims = torch.tensor([2])
        merged_dims = [2]
        threshold = 10
        self._test_merge_small_dims(dims, merged_dims, threshold)


class MultiDimSplitTest(unittest.TestCase):
    def _test_multi_dim_split(self, grad, splits, split_grad) -> None:
        for idx, g in enumerate(multi_dim_split(grad, splits)):
            self.assertTrue(torch.equal(split_grad[idx], g))

    def test_multi_dim_split_1(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [[3, 2], [2]]
        split_grad = [torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2)]
        self._test_multi_dim_split(grad, num_splits, split_grad)

    def test_multi_dim_split_2(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [[3, 2], [1, 1]]
        split_grad = [
            torch.arange(0, 6, 2).reshape(3, 1),
            torch.arange(0, 6, 2).reshape(3, 1) + 1,
            torch.arange(6, 10, 2).reshape(2, 1),
            torch.arange(6, 10, 2).reshape(2, 1) + 1,
        ]
        self._test_multi_dim_split(grad, num_splits, split_grad)

    def test_multi_dim_split_3(self) -> None:
        grad = torch.arange(30).reshape(5, 2, 3)
        num_splits = [[3, 2], [1, 1], [2, 1]]
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

    def test_multi_dim_cat_1(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [2, 0]
        split_grad = [torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2)]
        self._test_multi_dim_cat(grad, num_splits, split_grad)

    def test_multi_dim_cat_2(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        num_splits = [2, 2]
        split_grad = [
            torch.arange(0, 6, 2).reshape(3, 1),
            torch.arange(0, 6, 2).reshape(3, 1) + 1,
            torch.arange(6, 10, 2).reshape(2, 1),
            torch.arange(6, 10, 2).reshape(2, 1) + 1,
        ]
        self._test_multi_dim_cat(grad, num_splits, split_grad)

    def test_multi_dim_cat_3(self) -> None:
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
    def test_update_preconditioners_1(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        precond_sol = param.grad ** 2
        adagrad = AdagradPreconditioner(param, beta2=1.0, use_bias_correction=False)
        adagrad.update_preconditioners(param.grad)

        self.assertTrue(torch.all(torch.isclose(adagrad.preconditioner, precond_sol)))
        self.assertTrue(
            torch.isclose(torch.tensor(adagrad.bias_correction2), torch.tensor(1.0))
        )
        self.assertEqual(adagrad.num_updates, 1)

    def test_update_preconditioners_2(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        precond_sol = param.grad ** 2
        adagrad = AdagradPreconditioner(param, beta2=0.999, use_bias_correction=True)
        adagrad.update_preconditioners(param.grad)

        self.assertTrue(
            torch.all(torch.isclose(adagrad.preconditioner, 0.001 * precond_sol))
        )
        self.assertTrue(
            torch.isclose(torch.tensor(adagrad.bias_correction2), torch.tensor(0.001))
        )
        self.assertEqual(adagrad.num_updates, 1)

    def test_precondition_1(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        grad = param.grad
        adagrad = AdagradPreconditioner(
            param, beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        preconditioned_grad = adagrad.precondition(grad)
        self.assertTrue(torch.all(torch.isclose(preconditioned_grad, grad)))

    def test_precondition_2(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        grad = param.grad
        adagrad = AdagradPreconditioner(
            param, beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad)
        preconditioned_grad = adagrad.precondition(grad)
        self.assertTrue(torch.all(torch.isclose(preconditioned_grad, torch.ones(2))))

    def test_precondition_and_update_1(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        grad = param.grad
        adagrad = AdagradPreconditioner(
            param, beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        with torch.no_grad():
            adagrad.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.all(torch.isclose(param, torch.tensor([-1.0, -2.0]))))

    def test_precondition_and_update_2(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        grad = param.grad
        adagrad = AdagradPreconditioner(
            param, beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad)
        with torch.no_grad():
            adagrad.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.all(torch.isclose(param, torch.tensor([0.0, 1.0]))))

    def test_compute_norm_1(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        grad = param.grad
        adagrad = AdagradPreconditioner(
            param, beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        norm = adagrad.compute_norm(grad)
        self.assertEqual(norm, torch.sqrt(torch.tensor(20.0)))

    def test_compute_norm_2(self) -> None:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        grad = param.grad
        adagrad = AdagradPreconditioner(
            param, beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad)
        norm = adagrad.compute_norm(grad)
        self.assertEqual(norm, torch.sqrt(torch.tensor(2.0)))


class ShampooPreconditionerTest(unittest.TestCase):
    def test_update_preconditioners_1(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        preconditioner_sols = [
            param @ param.transpose(0, 1),
            param.transpose(0, 1) @ param,
        ]
        shampoo = ShampooPreconditioner(
            param, beta2=1.0, use_bias_correction=False, dtype=torch.float
        )
        shampoo.update_preconditioners(param.grad)

        for i in range(len(shampoo.preconditioners)):
            self.assertTrue(
                torch.all(
                    torch.isclose(shampoo.preconditioners[i], preconditioner_sols[i])
                )
            )
        self.assertTrue(
            torch.isclose(torch.tensor(shampoo.bias_correction2), torch.tensor(1.0))
        )
        self.assertEqual(shampoo.num_updates, 1)

    def test_update_preconditioners_2(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        preconditioner_sols = [
            param @ param.transpose(0, 1),
            param.transpose(0, 1) @ param,
        ]
        shampoo = ShampooPreconditioner(
            param, beta2=0.999, use_bias_correction=False, dtype=torch.float
        )
        shampoo.update_preconditioners(param.grad)

        for i in range(len(shampoo.preconditioners)):
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        shampoo.preconditioners[i], 0.001 * preconditioner_sols[i]
                    )
                )
            )
        self.assertTrue(
            torch.isclose(torch.tensor(shampoo.bias_correction2), torch.tensor(1.0))
        )
        self.assertEqual(shampoo.num_updates, 1)

    def test_update_preconditioners_3(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        preconditioner_sols = [
            param @ param.transpose(0, 1),
            param.transpose(0, 1) @ param,
        ]
        shampoo = ShampooPreconditioner(
            param, beta2=0.999, use_bias_correction=True, dtype=torch.float
        )
        shampoo.update_preconditioners(param.grad)

        for i in range(len(shampoo.preconditioners)):
            self.assertTrue(
                torch.all(
                    torch.isclose(
                        shampoo.preconditioners[i], 0.001 * preconditioner_sols[i]
                    )
                )
            )
        self.assertTrue(
            torch.isclose(torch.tensor(shampoo.bias_correction2), torch.tensor(0.001))
        )
        self.assertEqual(shampoo.num_updates, 1)

    def test_precondition_1(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param, beta2=1.0, epsilon=1.0, use_bias_correction=False, dtype=torch.float, init_delay=-1,
        )
        preconditioned_grad = shampoo.precondition(grad)
        self.assertTrue(
            torch.all(torch.isclose(preconditioned_grad, torch.zeros((2, 3))))
        )

    def test_precondition_2(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param, beta2=1.0, epsilon=1.0, use_bias_correction=False, dtype=torch.float
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad)
        self.assertTrue(torch.all(torch.isclose(preconditioned_grad, grad)))

    def test_precondition_3(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param,
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            dtype=torch.float,
            diagonal_threshold=2,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad)
        self.assertTrue(torch.all(torch.isclose(preconditioned_grad, grad)))

    def test_precondition_4(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param,
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            dtype=torch.float,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad)
        self.assertTrue(torch.all(torch.isclose(preconditioned_grad, grad)))

    def test_precondition_and_update_1(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param,
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            dtype=torch.float,
            init_delay=-1,
        )
        shampoo.compute_root_inverse()
        with torch.no_grad():
            shampoo.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.all(torch.isclose(param, torch.zeros((2, 3)))))

    def test_precondition_and_update_2(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param,
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            dtype=torch.float,
            init_delay=0,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        shampoo.compute_root_inverse()
        with torch.no_grad():
            shampoo.precondition_and_update(param, grad, 1.0)
        self.assertTrue(torch.all(torch.isclose(param, torch.zeros((2, 3)))))

    def test_compute_norm_1(self) -> None:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        grad = param.grad
        shampoo = ShampooPreconditioner(
            param,
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            dtype=torch.float,
            diagonal_threshold=2,
        )
        shampoo.compute_root_inverse()
        norm = shampoo.compute_norm(grad)
        self.assertEqual(norm, torch.sqrt(torch.tensor(55.0)))


if __name__ == '__main__':
    unittest.main()
