"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3


import unittest
from typing import Type

import torch

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_distributor import (
    Distributor,
    DistributorInterface,
)
from torch import nn


class DistributorInterfaceTest(unittest.TestCase):
    """DistributorInterfaceTest is the base class for testing all the single process Distributor implementation.

    Note that all the subclasses may not implement setUp() and only need to implement _get_distributor_type()
    to enable the the usage of self._distributor.
    """

    def setUp(self) -> None:
        self._model = nn.Sequential(
            nn.Linear(10, 5, bias=True),
        )
        self._model[0].weight.data.fill_(0.0)
        self._model[0].bias.data.fill_(0.0)
        self._param_group = DistributedShampoo(
            self._model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=5,
            precondition_frequency=1,
            start_preconditioning_step=-1,
        ).param_groups[0]
        self._distributor = self._get_distributor_type()(param_group=self._param_group)

    def _get_distributor_type(self) -> Type[DistributorInterface]:
        # Disable the abstract methods check from the interface so it is possible to instantiate DistributorInterface.
        DistributorInterface.__abstractmethods__ = frozenset()
        return DistributorInterface

    def test_update_params(self) -> None:
        self.assertIsNone(
            self._distributor.update_params(masked_blocked_search_directions=[])
        )

    def test_merge_and_block_gradients(self) -> None:
        self.assertIsNone(self._distributor.merge_and_block_gradients())


class DistributorTest(DistributorInterfaceTest):
    def _get_distributor_type(self) -> Type[DistributorInterface]:
        return Distributor

    def test_update_params(self) -> None:
        # Explicitly disable the gradient of the bias layer and call merge_and_block_gradients()
        # to update the local gradient selector.
        self._model[0].weight.grad = torch.ones((5, 10))
        self._model[0].bias.grad = None
        self._distributor.merge_and_block_gradients()

        actual_masked_blocked_params = self._distributor.local_masked_blocked_params

        masked_blocked_search_directions = (
            torch.arange(5 * 5, dtype=torch.float).reshape(5, 5),
            torch.arange(5 * 5, 2 * 5 * 5, dtype=torch.float).reshape(5, 5),
        )
        self._distributor.update_params(
            masked_blocked_search_directions,
        )

        expected_masked_blocked_params = (
            torch.arange(5 * 5, dtype=torch.float).reshape(5, 5),
            torch.arange(5 * 5, 2 * 5 * 5, dtype=torch.float).reshape(5, 5),
        )
        torch.testing.assert_close(
            actual_masked_blocked_params, expected_masked_blocked_params
        )

    def test_distributor_selector(self) -> None:
        # Two blocks from the linear layer, and one block from the bias layer.
        expected_distributor_selector = (True, True, True)
        self.assertEqual(
            self._distributor.distributor_selector,
            expected_distributor_selector,
        )

    def test_local_grad_selector(self) -> None:
        # Explicitly disable the gradient of the bias layer and call merge_and_block_gradients()
        # to update the local gradient selector for the bias layer (i.e., 3rd block).
        self._model[0].weight.grad = torch.ones((5, 10))
        self._model[0].bias.grad = None
        self._distributor.merge_and_block_gradients()

        expected_local_grad_selector = (True, True, False)
        self.assertEqual(
            self._distributor.local_grad_selector,
            expected_local_grad_selector,
        )

    def test_global_blocked_params(self) -> None:
        expected_global_params = (
            torch.zeros(5, 5, dtype=torch.float),
            torch.zeros(5, 5, dtype=torch.float),
            torch.zeros(5, dtype=torch.float),
        )
        torch.testing.assert_close(
            self._distributor.global_blocked_params,
            expected_global_params,
        )

    def test_local_blocked_params(self) -> None:
        # In Distributor, because there is no global vs. local boundary concept,
        # globl and local blocked params are always identical.
        expected_local_params = (
            torch.zeros(5, 5, dtype=torch.float),
            torch.zeros(5, 5, dtype=torch.float),
            torch.zeros(5, dtype=torch.float),
        )
        torch.testing.assert_close(
            self._distributor.local_blocked_params,
            expected_local_params,
        )

    def test_global_block_info_list(self) -> None:
        expected_global_block_info_list = (
            BlockInfo(
                param=self._model[0].weight,
                composable_block_ids=(0, "block_0"),
            ),
            BlockInfo(
                param=self._model[0].weight,
                composable_block_ids=(0, "block_1"),
            ),
            BlockInfo(
                param=self._model[0].bias,
                composable_block_ids=(1, "block_0"),
            ),
        )
        self.assertEqual(
            self._distributor.global_block_info_list,
            expected_global_block_info_list,
        )

    def test_merge_and_block_gradients(self) -> None:
        self._model[0].weight.grad = torch.ones((5, 10))
        self._model[0].bias.grad = None
        actual_local_masked_block_grads = self._distributor.merge_and_block_gradients()
        expected_local_masked_block_grads = (
            torch.ones((5, 5)),
            torch.ones((5, 5)),
        )
        torch.testing.assert_close(
            actual_local_masked_block_grads, expected_local_masked_block_grads
        )
