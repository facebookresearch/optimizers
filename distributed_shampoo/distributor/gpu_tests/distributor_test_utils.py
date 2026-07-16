"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import abc
import unittest
from collections.abc import Callable
from functools import partial

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.shampoo_block_info import (
    BlockInfo,
    DTensorBlockInfo,
)
from distributed_shampoo.distributor.shampoo_distributor import DistributorInterface
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DistributedConfig,
    GeneralizedPrimalAveragingConfig,
    ShampooPT2CompileConfig,
    WeightDecayType,
)
from torch import nn
from torch.optim.optimizer import ParamsT

# Shared constants for fully-shard and hybrid-shard param group tests.
PRECONDITIONER_DIM = 4

FULLY_SHARD_TEST_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = (
    (4 * PRECONDITIONER_DIM, 2 * PRECONDITIONER_DIM, 1),
    (3 * PRECONDITIONER_DIM - 1, PRECONDITIONER_DIM + 1, PRECONDITIONER_DIM - 1),
    (2, 2 * PRECONDITIONER_DIM - 1),
    (PRECONDITIONER_DIM, 1),
)

FULLY_SHARD_DEAD_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = (
    (PRECONDITIONER_DIM, 1),
    (),
)

HYBRID_SHARD_TEST_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = (
    (4 * PRECONDITIONER_DIM, 2 * PRECONDITIONER_DIM, 1, 1, 1),
    (3 * PRECONDITIONER_DIM - 1, PRECONDITIONER_DIM + 1, PRECONDITIONER_DIM - 1, 1, 1),
    (PRECONDITIONER_DIM, 1),
)

HYBRID_SHARD_DEAD_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = (
    (PRECONDITIONER_DIM, 1),
)

# Model dims for param group tests. Uses a single fixed model with enough
# params for num_sub_groups=4. No dims parametrization needed — the model
# size doesn't affect the splitting/threading logic.
# FSDP: 5 params (4 live + 1 dead). No shard_size constraint.
PARAM_GROUP_TEST_MODEL_DIMS: tuple[int, ...] = (
    4 * PRECONDITIONER_DIM,
    2 * PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    1,
)
PARAM_GROUP_TEST_DEAD_DIMS: tuple[int, ...] = (PRECONDITIONER_DIM, 1)
# HSDP with (2,2) mesh: shard_size=2, so each sub-group needs >= 2 params.
# num_sub_groups=4 needs >= 8 params. This model has 8 live + 1 dead = 9.
# Two wider blocks (2*PD) mixed with uniform PD layers give three numel tiers
# (32 / 16 / 4 with PD=4) so greedy LPT bin-packing has to interleave rather
# than trivially round-robin; no single weight dominates enough to strand a bin
# and trip the per-bin shard_size guard in _create_sub_groups.
HYBRID_SHARD_PARAM_GROUP_TEST_MODEL_DIMS: tuple[int, ...] = (
    2 * PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    2 * PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    PRECONDITIONER_DIM,
    1,
)


def shampoo_optim_factory(
    distributed_config: DistributedConfig,
    shampoo_pt2_compile_config: ShampooPT2CompileConfig | None = None,
    start_preconditioning_step: int = 2,
) -> Callable[[ParamsT], torch.optim.Optimizer]:
    """Shared optimizer factory for FSDP/HSDP lossless distributor tests."""
    return partial(
        DistributedShampoo,
        lr=0.001,
        betas=(0.9, 1.0),
        epsilon=1e-8,
        weight_decay=0.0,
        max_preconditioner_dim=PRECONDITIONER_DIM,
        precondition_frequency=1,
        start_preconditioning_step=start_preconditioning_step,
        weight_decay_type=WeightDecayType.DECOUPLED,
        grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
        distributed_config=distributed_config,
        iterate_averaging_config=GeneralizedPrimalAveragingConfig(),
        shampoo_pt2_compile_config=shampoo_pt2_compile_config,
    )


class DistributorOnEmptyParamTest:
    class Interface(abc.ABC, unittest.TestCase):
        """
        A test class for validating the behavior of parameter distributors when dealing with empty parameters.

        This class defines an abstract Interface that subclasses must implement to test specific distributor
        implementations. The Interface provides test methods that verify various aspects of distributor behavior.

        Subclasses must implement the following abstract methods:
        - _construct_model_and_distributor(): Creates the model and distributor to test
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

        def _test_update_params_impl(self, use_masked_tensors: bool) -> None:
            """Implementation of test_update_params - called by concrete test classes.

            This test verifies that:
            - use_masked_tensors=True: Updates only masked blocked params (those with gradients)
            - use_masked_tensors=False: Updates all local blocked params

            Concrete test classes must implement test_update_params with appropriate
            distributed test decorators (e.g., @with_comms, @skip_if_lt_x_gpu) and
            @parametrize("use_masked_tensors", [True, False]), then call this method.
            """
            _, distributor = self._construct_model_and_distributor()

            # Merge and block gradients to prepare for parameter updates
            distributor.merge_and_block_gradients()

            # Get appropriate target params based on use_masked_tensors flag
            target_params = (
                distributor.local_masked_blocked_params
                if use_masked_tensors
                else distributor.local_blocked_params
            )

            # Store original values for comparison
            original_values = tuple(p.clone() for p in target_params)

            # Create search directions matching target params length (all ones)
            blocked_search_directions = tuple(torch.ones_like(p) for p in target_params)

            # Apply the updates to parameters
            distributor.update_params(
                blocked_search_directions=blocked_search_directions,
                use_masked_tensors=use_masked_tensors,
            )

            # Compute expected values (original + 1.0)
            # Handle empty case: torch._foreach_add requires at least one tensor
            expected_params = (
                tuple(torch._foreach_add(original_values, 1.0))
                if len(original_values) > 0
                else ()
            )

            # Verify that parameters were updated correctly
            torch.testing.assert_close(target_params, expected_params)

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
