"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import abc
import unittest

import torch
from distributed_shampoo.distributor.shampoo_block_info import (
    BlockInfo,
    DTensorBlockInfo,
)
from distributed_shampoo.distributor.shampoo_distributor import DistributorInterface
from torch import nn


class DistributorOnEmptyParamTest:
    class Interface(abc.ABC, unittest.TestCase):
        """
        A test class for validating the behavior of parameter distributors when dealing with empty parameters.

        This class defines an abstract Interface that subclasses must implement to test specific distributor
        implementations. The Interface provides test methods that verify various aspects of distributor behavior.

        Subclasses must implement the following abstract methods:
        - _construct_model_and_distributor(): Creates the model and distributor to test
        - _expected_masked_blocked_params(): Returns expected masked blocked parameters for test_update_params()
        - _expected_local_grad_selector(): Returns expected gradient selector values for test_local_grad_selector()
        - _expected_local_blocked_params(): Returns expected local blocked parameters for test_local_blocked_params()
        - _expected_local_block_info_list(): Returns expected block info list for test_local_block_info_list()
        - _expected_local_masked_block_grads(): Returns expected local masked block gradients for test_merge_and_block_gradients()

        These methods provide the expected values that the tests will compare against the actual values
        from the distributor implementation.
        """

        @abc.abstractmethod
        def _construct_model_and_distributor(
            self,
        ) -> tuple[nn.Module, DistributorInterface]: ...

        @property
        @abc.abstractmethod
        def _expected_masked_blocked_params(self) -> tuple[torch.Tensor, ...]:
            """Returns expected masked blocked parameters used in test_update_params"""

        def test_update_params(self) -> None:
            _, distributor = self._construct_model_and_distributor()

            # Merge and block gradients to prepare for parameter updates
            distributor.merge_and_block_gradients()

            # Get the current masked blocked parameters
            actual_masked_blocked_params = distributor.local_masked_blocked_params

            # Create empty search directions (no updates)
            masked_blocked_search_directions = ()

            # Apply the empty updates to parameters
            distributor.update_params(
                masked_blocked_search_directions=masked_blocked_search_directions
            )

            # Verify that actual and expected parameters match
            torch.testing.assert_close(
                actual_masked_blocked_params, self._expected_masked_blocked_params
            )

        @property
        @abc.abstractmethod
        def _expected_local_grad_selector(self) -> tuple[bool, ...]:
            """Returns expected gradient selector values used in test_local_grad_selector"""

        def test_local_grad_selector(self) -> None:
            _, distributor = self._construct_model_and_distributor()

            # Merge and block gradients to prepare for testing
            distributor.merge_and_block_gradients()

            # Verify that the gradient selector matches expectations
            self.assertEqual(
                distributor.local_grad_selector, self._expected_local_grad_selector
            )

        @property
        @abc.abstractmethod
        def _expected_local_blocked_params(self) -> tuple[torch.Tensor, ...]:
            """Returns expected local blocked parameters used in test_local_blocked_params"""

        def test_local_blocked_params(self) -> None:
            _, distributor = self._construct_model_and_distributor()

            # Merge and block gradients to prepare for testing
            distributor.merge_and_block_gradients()

            # Verify that the local blocked parameters match expectations for the current rank
            torch.testing.assert_close(
                distributor.local_blocked_params,
                self._expected_local_blocked_params,
            )

        @abc.abstractmethod
        def _expected_local_block_info_list(
            self, model: nn.Module
        ) -> tuple[BlockInfo, ...] | tuple[DTensorBlockInfo, ...]:
            """Returns expected block info list used in test_local_block_info_list"""

        def test_local_block_info_list(self) -> None:
            model, distributor = self._construct_model_and_distributor()

            # Note: Manually comparing the block info lists because the order of the lists is not guaranteed to be the same.
            for index, (a, b) in enumerate(
                zip(
                    distributor.local_block_info_list,
                    self._expected_local_block_info_list(model),
                    strict=True,
                )
            ):
                # Only comparing param and composable_block_ids fields but not others like get_tensor()
                # because function objects are not comparable in BlockInfo.
                torch.testing.assert_close(
                    a.param,
                    b.param,
                    msg=f"Difference found at {index=}: {a.param=} != {b.param=}",
                )
                self.assertEqual(
                    a.composable_block_ids,
                    b.composable_block_ids,
                    msg=f"Difference found at {index=}: {a.composable_block_ids=} != {b.composable_block_ids=}",
                )

        @property
        @abc.abstractmethod
        def _expected_local_masked_block_grads(self) -> tuple[torch.Tensor, ...]:
            """Returns expected local masked block gradients used in test_merge_and_block_gradients"""

        def test_merge_and_block_gradients(self) -> None:
            _, distributor = self._construct_model_and_distributor()

            # Process gradients - since layer_weight is empty, it won't produce block gradients
            actual_local_masked_block_grads = distributor.merge_and_block_gradients()

            torch.testing.assert_close(
                actual_local_masked_block_grads, self._expected_local_masked_block_grads
            )
