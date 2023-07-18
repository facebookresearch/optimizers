"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import math
import unittest
from typing import cast, Tuple

import numpy as np

import torch
import torch.distributed as dist

try:
    # DTensor requires PyTorch 2.1 nightly build.
    import torch.distributed._tensor as dtensor

    ENABLE_DTENSOR = True
except ImportError:
    ENABLE_DTENSOR = False

import unittest.mock as mock

from torch.testing._internal.common_distributed import spawn_threads_and_init_comms

from distributed_shampoo.shampoo_dist_utils import use_local_tensor
from distributed_shampoo.shampoo_utils import (
    AdagradGrafting,
    AdagradNormalizedGrafting,
    AdagradPreconditioner,
    AdamGrafting,
    AdamNormalizedGrafting,
    BlockShampooPreconditioner,
    convex_split,
    DistributedPreconditioner,
    GraftingType,
    merge_small_dims,
    multi_dim_cat,
    multi_dim_split,
    Preconditioner,
    PreconditionerType,
    RMSPropGrafting,
    RMSPropNormalizedGrafting,
    SGDGrafting,
    ShampooPreconditioner,
)

logger: logging.Logger = logging.getLogger(__name__)

if not ENABLE_DTENSOR:
    logger.warning(
        "DTensor is not available and was not imported. Continuing with Tensor..."
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


class PreconditionerTest(unittest.TestCase):
    def test_preconditioner(self):
        param = torch.tensor([1.0, 2.0])
        preconditioner = Preconditioner(param)
        self.assertEqual(preconditioner._dims, np.array([2]))
        self.assertEqual(preconditioner.parameter_count, 0)


class DistributedPreconditionerTest(unittest.TestCase):
    def _setup_test(
        self, group, group_source_rank, dist_buffer
    ) -> Tuple[torch.Tensor, DistributedPreconditioner]:
        param = torch.tensor([1.0, 2.0])
        preconditioner = DistributedPreconditioner(
            param,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
        )
        return param, preconditioner

    @spawn_threads_and_init_comms(world_size=4)
    def test_combine_and_split_dims(self):
        param, preconditioner = self._setup_test(
            group=None, group_source_rank=0, dist_buffer=None
        )
        expected_tensor_list = [param]
        torch.testing.assert_close(
            preconditioner.combine_and_split_dims(expected_tensor_list[0]),
            expected_tensor_list,
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_get_dist_buffer_size(self):
        param, _ = self._setup_test(group=None, group_source_rank=0, dist_buffer=None)
        actual_buffer_size = DistributedPreconditioner.get_dist_buffer_size(param)
        self.assertEqual(actual_buffer_size, 8)

    @spawn_threads_and_init_comms(world_size=4)
    def test_get_from_dist_buffer(self):
        _, preconditioner = self._setup_test(
            group=None, group_source_rank=0, dist_buffer=None
        )
        self.assertEqual(preconditioner.get_from_dist_buffer(), None)

    @spawn_threads_and_init_comms(world_size=4)
    def test_get_split_dist_buffers(self):
        with self.subTest("Dist buffer is None."):
            _, preconditioner = self._setup_test(
                group=None, group_source_rank=0, dist_buffer=None
            )
            self.assertEqual(preconditioner.get_split_dist_buffers(), [])

        with self.subTest("Dist buffer is Tensor."):
            dist_buffer = torch.zeros(2)
            _, preconditioner = self._setup_test(
                group=None, group_source_rank=0, dist_buffer=dist_buffer
            )
            torch.testing.assert_close(
                preconditioner.get_split_dist_buffers(), [dist_buffer]
            )


class AdagradPreconditionerTest(unittest.TestCase):
    def _setup_test(
        self,
        beta2,
        epsilon,
        use_bias_correction=True,
        group=None,
        group_source_rank=0,
        dist_buffer=None,
        use_dtensor=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, AdagradPreconditioner]:
        param = torch.tensor([1.0, 2.0], requires_grad=True)
        loss = torch.dot(param, param)
        loss.backward()
        adagrad = AdagradPreconditioner(
            param,
            beta2=beta2,
            epsilon=epsilon,
            use_bias_correction=use_bias_correction,
            idx="test",
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_dtensor=use_dtensor,
        )
        return param, loss, cast(torch.Tensor, param.grad), adagrad

    def _test_update_preconditioners(self, beta2, use_bias_correction) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=beta2, epsilon=0.0, use_bias_correction=use_bias_correction
        )
        precond_sol = grad ** torch.tensor(2)
        precond_sol = precond_sol if beta2 == 1.0 else (1.0 - beta2) * precond_sol
        bias_correction2 = torch.tensor(1.0 if not use_bias_correction else 1.0 - beta2)
        adagrad.update_preconditioners(grad, torch.tensor(1))

        with self.subTest("Test preconditioner"):
            torch.testing.assert_close(
                use_local_tensor(adagrad._preconditioner), precond_sol
            )
        with self.subTest("Test bias correction"):
            torch.testing.assert_close(
                torch.tensor(adagrad._bias_correction2), bias_correction2
            )

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners_adagrad(self) -> None:
        self._test_update_preconditioners(beta2=1.0, use_bias_correction=False)

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners_rmsprop(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=False)

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners_adam(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=True)

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_without_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        preconditioned_grad = adagrad.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad, torch.tensor(1))
        preconditioned_grad = adagrad.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, torch.ones(2))

    @spawn_threads_and_init_comms(world_size=4)
    def test_compute_norm_without_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False
        )
        norm = adagrad.compute_norm(grad, torch.tensor(1))
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(20.0)))

    @spawn_threads_and_init_comms(world_size=4)
    def test_compute_norm_with_preconditioner_update(self) -> None:
        param, loss, grad, adagrad = self._setup_test(
            beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        adagrad.update_preconditioners(grad, torch.tensor(1))
        norm = adagrad.compute_norm(grad, torch.tensor(1))
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(2.0)))

    @spawn_threads_and_init_comms(world_size=4)
    def test_to(self) -> None:
        _, _, _, adagrad = self._setup_test(
            beta2=1.0, epsilon=0.0, use_bias_correction=False
        )
        try:
            adagrad.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")

    @spawn_threads_and_init_comms(world_size=4)
    def test_preconditioned_grad_to_dist_buffer(self):
        expected_dist_buffer = torch.zeros(2)
        _, _, grad, preconditioner = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            group=dist.group.WORLD,
            group_source_rank=0,
            dist_buffer=expected_dist_buffer,
        )
        preconditioner.preconditioned_grad_to_dist_buffer(grad, torch.tensor(1))
        torch.testing.assert_close(
            preconditioner._dist_buffer,
            grad if dist.get_rank() == 0 else torch.zeros(2),
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_dtensor_enabled(self):
        _, _, _, adagrad = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            use_dtensor=True,
        )
        self.assertTrue(
            isinstance(
                adagrad._preconditioner,
                dtensor.DTensor if ENABLE_DTENSOR else torch.Tensor,
            )
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_dtensor_disabled(self):
        _, _, _, adagrad = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            use_dtensor=False,
        )
        if ENABLE_DTENSOR:
            self.assertFalse(isinstance(adagrad._preconditioner, dtensor.DTensor))


class ShampooPreconditionerTest(unittest.TestCase):
    def _setup_test(
        self,
        beta2,
        epsilon,
        exponent_override=0,
        exponent_multiplier=1.0,
        use_bias_correction=True,
        diagonal_threshold=None,
        start_preconditioning_step=-1,
        grafting_type=GraftingType.NONE,
        grafting_beta2=1.0,
        grafting_epsilon=1e-3,
        group=None,
        group_source_rank=0,
        dist_buffer=None,
        use_protected_eigh=True,
        use_dtensor=True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ShampooPreconditioner]:
        param = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], requires_grad=True)
        loss = torch.linalg.norm(param, ord="fro") ** 2 / 2.0
        loss.backward()
        idx = "test"

        shampoo = ShampooPreconditioner(
            param,
            beta2=beta2,
            epsilon=epsilon,
            exponent_override=exponent_override,
            exponent_multiplier=exponent_multiplier,
            use_bias_correction=use_bias_correction,
            diagonal_threshold=diagonal_threshold,
            idx=idx,
            dtype=torch.float,
            start_preconditioning_step=start_preconditioning_step,
            grafting_type=grafting_type,
            grafting_beta2=grafting_beta2,
            grafting_epsilon=grafting_epsilon,
            group=group,
            group_source_rank=group_source_rank,
            dist_buffer=dist_buffer,
            use_protected_eigh=use_protected_eigh,
            use_dtensor=use_dtensor,
        )
        return param, loss, cast(torch.Tensor, param.grad), shampoo

    @spawn_threads_and_init_comms(world_size=4)
    def test_init_shampoo(self):
        grafting_types = {
            GraftingType.NONE: None,
            GraftingType.SGD: SGDGrafting,
            GraftingType.ADAGRAD: AdagradGrafting,
            GraftingType.RMSPROP: RMSPropGrafting,
            GraftingType.ADAM: AdamGrafting,
            GraftingType.ADAGRAD_NORMALIZED: AdagradNormalizedGrafting,
            GraftingType.RMSPROP_NORMALIZED: RMSPropNormalizedGrafting,
            GraftingType.ADAM_NORMALIZED: AdamNormalizedGrafting,
        }
        for grafting_type in grafting_types.keys():
            with self.subTest(f"Test grafting type {grafting_type}"):
                _, _, _, shampoo = self._setup_test(
                    beta2=1.0,
                    epsilon=1e-12,
                    exponent_override=4,
                    exponent_multiplier=1.82,
                    use_bias_correction=True,
                    diagonal_threshold=2,
                    start_preconditioning_step=100,
                    grafting_type=grafting_type,
                    grafting_beta2=0.999,
                    grafting_epsilon=1e-3,
                )
                self.assertTrue(
                    shampoo._preconditioners[0].preconditioner_type
                    == PreconditionerType.FULL
                )
                self.assertTrue(
                    shampoo._preconditioners[1].preconditioner_type
                    == PreconditionerType.DIAGONAL
                )
                self.assertEqual(shampoo._order, 2)
                self.assertIsNone(
                    shampoo._grafting
                ) if grafting_type == GraftingType.NONE else self.assertIsInstance(
                    shampoo._grafting, grafting_types[grafting_type]
                )
                self.assertEqual(
                    shampoo._parameter_count,
                    11
                    if grafting_type in [GraftingType.NONE, GraftingType.SGD]
                    else 17,
                )

    def _test_update_preconditioners(
        self, beta2, use_bias_correction
    ) -> ShampooPreconditioner:
        param, loss, grad, shampoo = self._setup_test(
            beta2=beta2,
            epsilon=0.0,
            exponent_override=4,
            exponent_multiplier=1.82,
            use_bias_correction=use_bias_correction,
            diagonal_threshold=2,
            start_preconditioning_step=100,
            grafting_type=GraftingType.NONE,
            grafting_beta2=0.999,
            grafting_epsilon=1e-3,
        )
        preconditioner_sols = [
            param @ param.transpose(0, 1),
            torch.linalg.norm(param, dim=0).pow(2),
        ]
        shampoo.update_preconditioners(grad, torch.tensor(1))

        for i, (preconditioner, preconditioner_sol) in enumerate(
            zip(shampoo._preconditioners, preconditioner_sols)
        ):
            with self.subTest(f"Test preconditioner {i}"):
                torch.testing.assert_close(
                    use_local_tensor(preconditioner.factor_matrix),
                    preconditioner_sol
                    if beta2 == 1.0
                    else (1 - beta2) * preconditioner_sol,
                )
        with self.subTest("Test bias correction"):
            torch.testing.assert_close(
                torch.tensor(shampoo._bias_correction2),
                torch.tensor(1.0 if not use_bias_correction else 1.0 - beta2),
            )

        return shampoo

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners_adagrad(self) -> None:
        self._test_update_preconditioners(beta2=1.0, use_bias_correction=False)

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners_rmsprop(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=False)

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners_adam(self) -> None:
        self._test_update_preconditioners(beta2=0.999, use_bias_correction=True)

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_no_root_inverse(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        preconditioned_grad = shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, torch.zeros((2, 3)))

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_with_root_inverse(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_with_diagonal_threshold(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_with_grafting(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_grafting_precondition(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=5,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        shampoo.compute_root_inverse()
        preconditioned_grad = shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_compute_norm(self) -> None:
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2
        )
        shampoo.compute_root_inverse()
        norm = shampoo.compute_norm(grad, torch.tensor(1))
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(55.0)))

    @spawn_threads_and_init_comms(world_size=4)
    def test_to(self) -> None:
        _, _, _, shampoo = self._setup_test(
            beta2=1.0, epsilon=1.0, use_bias_correction=False, diagonal_threshold=2
        )
        try:
            shampoo.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")

    @spawn_threads_and_init_comms(world_size=4)
    def test_reset_preconditioners(self) -> None:
        shampoo = self._test_update_preconditioners(
            beta2=1.0, use_bias_correction=False
        )
        shampoo.reset_preconditioners()
        for i, preconditioner in enumerate(shampoo._preconditioners):
            with self.subTest(f"Test preconditioner {i}"):
                torch.testing.assert_close(
                    preconditioner.factor_matrix,
                    torch.zeros_like(preconditioner.factor_matrix),
                )

    @spawn_threads_and_init_comms(world_size=4)
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
        shampoo.update_preconditioners(grad_1, torch.tensor(1))

        grad_2 = torch.zeros_like(grad)
        grad_2[1, 2] = 2
        shampoo.update_preconditioners(grad_2, torch.tensor(1))

        shampoo.compute_root_inverse()
        left_root_inverse = torch.diag(torch.tensor([1.0, 1.0 / math.sqrt(5.0)]))
        right_root_inverse = torch.diag(torch.tensor([1.0, 1.0, 0.5]))

        with self.subTest("Test left root inverse matrix"):
            torch.testing.assert_close(
                use_local_tensor(shampoo._preconditioners[0].inv_factor_matrix),
                left_root_inverse,
            )
        with self.subTest("Test right root inverse matrix"):
            torch.testing.assert_close(
                use_local_tensor(shampoo._preconditioners[1].inv_factor_matrix),
                right_root_inverse,
            )

        true_preconditioned_grad = left_root_inverse @ grad @ right_root_inverse
        preconditioned_grad = shampoo.precondition(grad, torch.tensor(1))
        with self.subTest("Test preconditioned grad"):
            torch.testing.assert_close(preconditioned_grad, true_preconditioned_grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_compute_root_inverse_residuals(self):
        param, loss, grad, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_bias_correction=False,
            diagonal_threshold=2,
        )
        shampoo.update_preconditioners(grad, torch.tensor(1))
        shampoo.compute_root_inverse()
        relative_errors, relative_residuals = shampoo.compute_root_inverse_residuals()
        torch.testing.assert_close(
            relative_errors,
            [torch.tensor(0.0, dtype=torch.float64)],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            relative_residuals,
            [torch.tensor(0.0, dtype=torch.float64)],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_preconditioned_grad_to_dist_buffer(self):
        expected_dist_buffer = torch.zeros(2, 3)
        _, _, grad, preconditioner = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            group=None,
            group_source_rank=0,
            dist_buffer=expected_dist_buffer,
            use_protected_eigh=False,
        )
        preconditioner.compute_root_inverse()
        preconditioner.preconditioned_grad_to_dist_buffer(grad, torch.tensor(1))
        torch.testing.assert_close(
            preconditioner._dist_buffer,
            grad,
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_dtensor_enabled(self):
        _, _, _, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_dtensor=True,
        )
        for preconditioner in shampoo._preconditioners:
            self.assertTrue(
                isinstance(
                    preconditioner.factor_matrix,
                    dtensor.DTensor if ENABLE_DTENSOR else torch.Tensor,
                )
            )
            self.assertTrue(
                isinstance(
                    preconditioner.inv_factor_matrix,
                    dtensor.DTensor if ENABLE_DTENSOR else torch.Tensor,
                )
            )

    def test_dtensor_disabled(self):
        _, _, _, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_dtensor=False,
        )
        if ENABLE_DTENSOR:
            for preconditioner in shampoo._preconditioners:
                self.assertFalse(
                    isinstance(
                        preconditioner.factor_matrix,
                        dtensor.DTensor,
                    )
                )
                self.assertFalse(
                    isinstance(
                        preconditioner.inv_factor_matrix,
                        dtensor.DTensor,
                    )
                )

    @mock.patch("distributed_shampoo.shampoo_utils.matrix_inverse_root")
    def test_use_protected_eigh_disabled(self, mock_matrix_root: mock.Mock):
        _, _, _, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_protected_eigh=False,
            use_dtensor=False,
        )
        mock_matrix_root.side_effect = RuntimeError("Mock Matrix Root Eigen Error")
        with self.assertRaisesRegex(RuntimeError, "Mock Matrix Root Eigen Error"):
            shampoo.compute_root_inverse()
        mock_matrix_root.assert_called_once()

    @mock.patch("distributed_shampoo.shampoo_utils.matrix_inverse_root")
    def test_use_protected_eigh_enabled(self, mock_matrix_root: mock.Mock):
        _, _, _, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_protected_eigh=True,
            use_dtensor=False,
        )
        mock_matrix_root.side_effect = RuntimeError("Mock Matrix Root Eigen Error")
        shampoo.compute_root_inverse()
        expected_inv_factors = [torch.zeros((2, 2)), torch.zeros((3, 3))]
        for preconditioner, expected_inv_factor_matrix in zip(
            shampoo._preconditioners, expected_inv_factors
        ):
            torch.testing.assert_close(
                preconditioner.inv_factor_matrix, expected_inv_factor_matrix
            )
        self.assertEqual(mock_matrix_root.call_count, 2)

    @mock.patch("distributed_shampoo.shampoo_utils.matrix_inverse_root")
    def test_raise_inf_in_compute_root_inverse(self, mock_matrix_root: mock.Mock):
        _, _, _, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_dtensor=False,
        )
        mock_matrix_root.side_effect = torch.tensor([torch.inf])
        with self.assertRaisesRegex(
            ValueError, "Encountered inf values in root inv preconditioner"
        ):
            shampoo.compute_root_inverse()
        mock_matrix_root.assert_called_once()

    @mock.patch("distributed_shampoo.shampoo_utils.matrix_inverse_root")
    def test_raise_nan_in_compute_root_inverse(self, mock_matrix_root: mock.Mock):
        _, _, _, shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_dtensor=False,
        )
        mock_matrix_root.side_effect = torch.tensor([torch.nan])
        with self.assertRaisesRegex(
            ValueError, "Encountered nan values in root inv preconditioner"
        ):
            shampoo.compute_root_inverse()
        mock_matrix_root.assert_called_once()


class BlockShampooPreconditionerTest(unittest.TestCase):
    def _setup_test(
        self,
        beta2: float = 1.0,
        epsilon: float = 1e-12,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_bias_correction: bool = True,
        use_merge_dims=True,
        start_preconditioning_step: int = 0,
        grafting_type: GraftingType = GraftingType.NONE,
        grafting_beta2: float = 1.0,
        grafting_epsilon: float = 1e-3,
        use_dtensor: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, BlockShampooPreconditioner]:
        param = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        loss = torch.linalg.norm(param) ** 2 / 2.0
        loss.backward()
        dist_buffer_ranks = [
            (torch.zeros(2), 0),
            (torch.zeros(2), 0),
            (torch.zeros(1), 0),
        ]
        dist_buffer_index = 0

        block_shampoo = BlockShampooPreconditioner(
            param,
            beta2=beta2,
            epsilon=epsilon,
            exponent_override=exponent_override,
            exponent_multiplier=exponent_multiplier,
            use_bias_correction=use_bias_correction,
            block_size=2,
            dtype=torch.float,
            idx=0,
            use_merge_dims=use_merge_dims,
            start_preconditioning_step=start_preconditioning_step,
            grafting_type=grafting_type,
            grafting_beta2=grafting_beta2,
            grafting_epsilon=grafting_epsilon,
            group=None,
            dist_buffer_ranks=dist_buffer_ranks,
            dist_buffer_index=dist_buffer_index,
            use_dtensor=use_dtensor,
        )
        return param, loss, cast(torch.Tensor, param.grad), block_shampoo

    @spawn_threads_and_init_comms(world_size=4)
    def test_block_shampoo_init(self):
        for use_merge_dims in [False, True]:
            with self.subTest("Test use_merge_dims = " + str(use_merge_dims)):
                _, _, _, block_shampoo = self._setup_test(
                    use_merge_dims=use_merge_dims,
                )
                self.assertEqual(block_shampoo.parameter_count, 18)
                self.assertEqual(len(block_shampoo._split_preconditioners), 3)

    @spawn_threads_and_init_comms(world_size=4)
    def test_combine_and_split_dims(self):
        param, _, _, block_shampoo = self._setup_test()
        expected_tensor_list = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
            torch.tensor([5.0]),
        ]
        torch.testing.assert_close(
            block_shampoo.combine_and_split_dims(param), expected_tensor_list
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners(self):
        _, _, grad, block_shampoo = self._setup_test()
        block_shampoo.update_preconditioners(grad, torch.tensor(1))
        actual_factor_matrix_list = [
            use_local_tensor(preconditioner._preconditioners[0].factor_matrix)
            for preconditioner in block_shampoo._split_preconditioners
        ]
        expected_factor_matrix_list = [
            torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
            torch.tensor([[9.0, 12.0], [12.0, 16.0]]),
            torch.tensor([[25.0]]),
        ]
        torch.testing.assert_close(
            actual_factor_matrix_list, expected_factor_matrix_list
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_no_root_inverse(self):
        _, _, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        preconditioned_grad = block_shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, torch.zeros(5))

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_with_root_inverse(self) -> None:
        _, _, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        block_shampoo.compute_root_inverse()
        preconditioned_grad = block_shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition_with_grafting(self) -> None:
        param, loss, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        block_shampoo.compute_root_inverse()
        preconditioned_grad = block_shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_grafting_precondition(self) -> None:
        param, loss, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=5,
            grafting_type=GraftingType.ADAGRAD,
            grafting_epsilon=1.0,
        )
        block_shampoo.compute_root_inverse()
        preconditioned_grad = block_shampoo.precondition(grad, torch.tensor(1))
        torch.testing.assert_close(preconditioned_grad, grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_compute_norm(self) -> None:
        param, loss, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        block_shampoo.compute_root_inverse()
        norm = block_shampoo.compute_norm(grad, torch.tensor(1))
        torch.testing.assert_close(norm, torch.sqrt(torch.tensor(55.0)))

    @spawn_threads_and_init_comms(world_size=4)
    def test_to(self) -> None:
        _, _, _, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        try:
            block_shampoo.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")

    @spawn_threads_and_init_comms(world_size=4)
    def test_reset_preconditioners(self) -> None:
        param, loss, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        block_shampoo.update_preconditioners(grad, torch.tensor(1))
        block_shampoo.reset_preconditioners()
        for shampoo_preconditioner in block_shampoo._split_preconditioners:
            for kronecker_factor in shampoo_preconditioner._preconditioners:
                with self.subTest(
                    f"Test Shampoo preconditioner {shampoo_preconditioner._idx}, factor matrix {kronecker_factor.index}"
                ):
                    torch.testing.assert_close(
                        use_local_tensor(kronecker_factor.factor_matrix),
                        torch.zeros_like(
                            use_local_tensor(kronecker_factor.factor_matrix)
                        ),
                    )

    @spawn_threads_and_init_comms(world_size=4)
    def test_compute_root_inverse_residuals(self):
        param, loss, grad, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=1.0,
            use_bias_correction=False,
            start_preconditioning_step=-1,
        )
        block_shampoo.update_preconditioners(grad, torch.tensor(1))
        block_shampoo.compute_root_inverse()
        (
            relative_errors,
            relative_residuals,
        ) = block_shampoo.compute_root_inverse_residuals()
        torch.testing.assert_close(
            relative_errors,
            [torch.tensor(0.0, dtype=torch.float64)] * 3,
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            relative_residuals,
            [torch.tensor(0.0, dtype=torch.float64)] * 3,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_get_dist_buffer_sizes(self):
        param = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        actual_dist_buffer_sizes = BlockShampooPreconditioner.get_dist_buffer_sizes(
            param, block_size=2, use_merge_dims=True
        )
        expected_dist_buffer_sizes = [8, 8, 4]
        self.assertEqual(actual_dist_buffer_sizes, expected_dist_buffer_sizes)

    @spawn_threads_and_init_comms(world_size=4)
    def test_preconditioned_grad_to_dist_buffer(self):
        _, _, grad, block_shampoo = self._setup_test(epsilon=1.0)
        block_shampoo._on_source_rank = True
        for preconditioner in block_shampoo._split_preconditioners:
            preconditioner._on_source_rank = True
        block_shampoo.compute_root_inverse()
        block_shampoo.preconditioned_grad_to_dist_buffer(grad, torch.tensor(1))
        for split_grad, preconditioner in zip(
            block_shampoo.combine_and_split_dims(grad),
            block_shampoo._split_preconditioners,
        ):
            torch.testing.assert_close(preconditioner._dist_buffer, split_grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_get_from_dist_buffer(self):
        _, _, grad, block_shampoo = self._setup_test()
        torch.testing.assert_close(
            block_shampoo.get_from_dist_buffer(), torch.zeros_like(grad)
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_get_split_dist_buffers(self):
        _, _, grad, block_shampoo = self._setup_test()
        torch.testing.assert_close(
            block_shampoo.get_split_dist_buffers(),
            [torch.zeros(2), torch.zeros(2), torch.zeros(1)],
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_dtensor_enabled(self):
        _, _, _, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_dtensor=True,
        )
        for shampoo in block_shampoo._split_preconditioners:
            for preconditioner in shampoo._preconditioners:
                self.assertTrue(
                    isinstance(
                        preconditioner.factor_matrix,
                        dtensor.DTensor if ENABLE_DTENSOR else torch.Tensor,
                    )
                )
                self.assertTrue(
                    isinstance(
                        preconditioner.inv_factor_matrix,
                        dtensor.DTensor if ENABLE_DTENSOR else torch.Tensor,
                    )
                )

    def test_dtensor_disabled(self):
        _, _, _, block_shampoo = self._setup_test(
            beta2=1.0,
            epsilon=0.0,
            use_dtensor=False,
        )
        if ENABLE_DTENSOR:
            for shampoo in block_shampoo._split_preconditioners:
                for preconditioner in shampoo._preconditioners:
                    self.assertFalse(
                        isinstance(
                            preconditioner.factor_matrix,
                            dtensor.DTensor,
                        )
                    )
                    self.assertFalse(
                        isinstance(
                            preconditioner.inv_factor_matrix,
                            dtensor.DTensor,
                        )
                    )


class SGDGraftingTest(unittest.TestCase):
    def _setup_test(self) -> Tuple[SGDGrafting, torch.Tensor]:
        param = torch.tensor([1.0, 2.0])
        return SGDGrafting(param), param

    def test_init(self):
        grafting, _ = self._setup_test()
        self.assertEqual(grafting.parameter_count, 0)

    def test_precondition(self):
        grafting, grad = self._setup_test()
        torch.testing.assert_close(grafting.precondition(grad, torch.tensor(1)), grad)

    def test_direction_norm(self):
        grafting, grad = self._setup_test()
        torch.testing.assert_close(
            grafting.direction_norm(grad, torch.tensor(1)),
            torch.sqrt(torch.tensor(5.0)),
        )

    def test_to(self) -> None:
        grafting, _ = self._setup_test()
        try:
            grafting.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")


class AdagradGraftingTest(unittest.TestCase):
    def _setup_test(self, use_dtensor=True) -> Tuple[AdagradGrafting, torch.Tensor]:
        param = torch.tensor([1.0, 2.0])
        return AdagradGrafting(param, epsilon=1.0, use_dtensor=use_dtensor), param

    @spawn_threads_and_init_comms(world_size=4)
    def test_init(self):
        grafting, grad = self._setup_test()
        self.assertEqual(grafting.parameter_count, 2)
        self.assertEqual(grafting.normalize_gradient, False)

    @spawn_threads_and_init_comms(world_size=4)
    def test_update_preconditioners(self):
        grafting, grad = self._setup_test()
        try:
            grafting.update_preconditioners(grad, torch.tensor(1))
        except Exception:
            self.fail(".update_preconditioners raised Exception!")

    @spawn_threads_and_init_comms(world_size=4)
    def test_precondition(self):
        grafting, grad = self._setup_test()
        torch.testing.assert_close(grafting.precondition(grad, torch.tensor(1)), grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_direction_norm(self):
        grafting, grad = self._setup_test()
        torch.testing.assert_close(
            grafting.direction_norm(grad, torch.tensor(1)), torch.linalg.norm(grad)
        )

    @spawn_threads_and_init_comms(world_size=4)
    def test_to(self):
        grafting, _ = self._setup_test()
        try:
            grafting.to(torch.device("cpu"))
        except Exception:
            self.fail(".to() raised Exception!")

    @spawn_threads_and_init_comms(world_size=4)
    def test_normalize_grad(self):
        grafting, grad = self._setup_test()
        with self.subTest("Test normalize_gradient = True"):
            grafting.normalize_gradient = True
            torch.testing.assert_close(
                grafting._normalize_grad(grad), grad / torch.linalg.norm(grad)
            )
        with self.subTest("Test normalize_gradient = False"):
            grafting.normalize_gradient = False
            torch.testing.assert_close(grafting._normalize_grad(grad), grad)

    @spawn_threads_and_init_comms(world_size=4)
    def test_dtensor_enabled(self):
        grafting, _ = self._setup_test(use_dtensor=True)
        self.assertTrue(
            isinstance(
                grafting._preconditioner._preconditioner,
                dtensor.DTensor if ENABLE_DTENSOR else torch.Tensor,
            )
        )

    def test_dtensor_disabled(self):
        grafting, _ = self._setup_test(use_dtensor=False)
        if ENABLE_DTENSOR:
            self.assertFalse(
                isinstance(
                    grafting._preconditioner._preconditioner,
                    dtensor.DTensor,
                )
            )


class RMSPropGraftingTest(unittest.TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_init(self):
        param = torch.tensor([1.0, 2.0])
        try:
            RMSPropGrafting(param)
        except Exception:
            self.fail("Instantiating RMSPropGrafting raised Exception!")


class AdamGraftingTest(unittest.TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_init(self):
        param = torch.tensor([1.0, 2.0])
        try:
            AdamGrafting(param)
        except Exception:
            self.fail("Instantiating AdamGrafting raised Exception!")


class AdagradNormalizedGraftingTest(unittest.TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_init(self):
        param = torch.tensor([1.0, 2.0])
        try:
            AdagradNormalizedGrafting(param)
        except Exception:
            self.fail("Instantiating AdagradNormalizedGrafting raised Exception!")


class RMSPropNormalizedGraftingTest(unittest.TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_init(self):
        param = torch.tensor([1.0, 2.0])
        try:
            RMSPropNormalizedGrafting(param)
        except Exception:
            self.fail("Instantiating RMSPropNormalizedGrafting raised Exception!")


class AdamNormalizedGraftingTest(unittest.TestCase):
    @spawn_threads_and_init_comms(world_size=4)
    def test_init(self):
        param = torch.tensor([1.0, 2.0])
        try:
            AdamNormalizedGrafting(param)
        except Exception:
            self.fail("Instantiating AdamNormalizedGrafting raised Exception!")
