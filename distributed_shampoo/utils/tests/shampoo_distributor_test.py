"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest

import torch

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_distributor import Distributor
from torch import nn

PRECONDITIONER_DIM = 5


class DistributorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._model = construct_training_problem(
            (2 * PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            model_dead_layers_dims=None,
            bias=True,
            fill=0.0,
        )[0]
        assert isinstance(self._model, nn.Module)
        self._distributor = Distributor(
            param_group=DistributedShampoo(
                self._model.parameters(),
                lr=0.01,
                betas=(0.9, 1.0),
                epsilon=1e-12,
                momentum=0.0,
                weight_decay=0.0,
                max_preconditioner_dim=PRECONDITIONER_DIM,
                precondition_frequency=1,
                start_preconditioning_step=-1,
            ).param_groups[0]
        )

    def test_update_params(self) -> None:
        # Explicitly disable the gradient of the scalar parameter and call merge_and_block_gradients()
        # to update the local gradient selector for the scalar parameter (i.e., 1st block) and bias layer (i.e., 4th block).
        self._model.scalar.grad = None  # type: ignore[union-attr]
        self._model.linear_layers[0].weight.grad = torch.ones_like(  # type: ignore[index, union-attr]
            (
                self._model.linear_layers[0].weight  # type: ignore[index, union-attr]
            )
        )
        self._model.linear_layers[0].bias.grad = None  # type: ignore[index, union-attr]
        self._distributor.merge_and_block_gradients()

        actual_masked_blocked_params = self._distributor.local_masked_blocked_params

        masked_blocked_search_directions = (
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM, dtype=torch.float
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                2 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                dtype=torch.float,
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
        )
        self._distributor.update_params(
            masked_blocked_search_directions=masked_blocked_search_directions
        )

        expected_masked_blocked_params = (
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM, dtype=torch.float
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
            torch.arange(
                PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                2 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
                dtype=torch.float,
            ).reshape(PRECONDITIONER_DIM, PRECONDITIONER_DIM),
        )
        torch.testing.assert_close(
            actual_masked_blocked_params, expected_masked_blocked_params
        )

    def test_local_grad_selector(self) -> None:
        # Explicitly disable the gradient of the scalar parameter and call merge_and_block_gradients()
        # to update the local gradient selector for the scalar parameter (i.e., 1st block) and bias layer (i.e., 4th block).
        self._model.scalar.grad = None  # type: ignore[union-attr]
        self._model.linear_layers[0].weight.grad = torch.ones_like(  # type: ignore[index, union-attr]
            self._model.linear_layers[0].weight  # type: ignore[index, union-attr]
        )
        self._model.linear_layers[0].bias.grad = None  # type: ignore[index, union-attr]
        self._distributor.merge_and_block_gradients()

        expected_local_grad_selector = (False, True, True, False)
        self.assertEqual(
            self._distributor.local_grad_selector,
            expected_local_grad_selector,
        )

    def test_local_blocked_params(self) -> None:
        # In Distributor, because there is no global vs. local boundary concept,
        # global and local blocked params are always identical.
        expected_local_params = (
            torch.zeros((1,), dtype=torch.float),
            torch.zeros(PRECONDITIONER_DIM, PRECONDITIONER_DIM, dtype=torch.float),
            torch.zeros(PRECONDITIONER_DIM, PRECONDITIONER_DIM, dtype=torch.float),
            torch.zeros(PRECONDITIONER_DIM, dtype=torch.float),
        )
        torch.testing.assert_close(
            self._distributor.local_blocked_params,
            expected_local_params,
        )

    def test_local_block_info_list(self) -> None:
        def block_info_equality(
            a: BlockInfo, b: BlockInfo, msg: str | None = None
        ) -> None:
            # Only comparing param and composable_block_ids fields but not others like get_tensor()
            # because function objects are not comparable in BlockInfo.
            torch.testing.assert_close(a.param, b.param)
            self.assertEqual(a.composable_block_ids, b.composable_block_ids)

        self.addTypeEqualityFunc(BlockInfo, block_info_equality)

        expected_local_block_info_list = (
            BlockInfo(
                param=self._model.scalar,  # type: ignore[arg-type, union-attr]
                composable_block_ids=(0, "block_0"),
            ),
            BlockInfo(
                param=self._model.linear_layers[0].weight,  # type: ignore[index, union-attr]
                composable_block_ids=(1, "block_0"),
            ),
            BlockInfo(
                param=self._model.linear_layers[0].weight,  # type: ignore[index, union-attr]
                composable_block_ids=(1, "block_1"),
            ),
            BlockInfo(
                param=self._model.linear_layers[0].bias,  # type: ignore[index, union-attr]
                composable_block_ids=(2, "block_0"),
            ),
        )
        for index, (a, b) in enumerate(
            zip(
                self._distributor.local_block_info_list,
                expected_local_block_info_list,
                strict=True,
            )
        ):
            self.assertEqual(
                a,
                b,
                f"Difference found at {index=}: {self._distributor.local_block_info_list[index]=} != {expected_local_block_info_list[index]=}",
            )

    def test_merge_and_block_gradients(self) -> None:
        self._model.scalar.grad = torch.ones_like(self._model.scalar)  # type: ignore[arg-type, union-attr]
        self._model.linear_layers[0].weight.grad = torch.ones_like(  # type: ignore[index, union-attr]
            self._model.linear_layers[0].weight  # type: ignore[index, union-attr]
        )
        self._model.linear_layers[0].bias.grad = None  # type: ignore[index, union-attr]
        actual_local_masked_block_grads = self._distributor.merge_and_block_gradients()
        expected_local_masked_block_grads = (
            torch.ones((1,)),  # type: ignore[arg-type, union-attr]
            torch.ones((PRECONDITIONER_DIM, PRECONDITIONER_DIM)),
            torch.ones((PRECONDITIONER_DIM, PRECONDITIONER_DIM)),
        )
        torch.testing.assert_close(
            actual_local_masked_block_grads, expected_local_masked_block_grads
        )
