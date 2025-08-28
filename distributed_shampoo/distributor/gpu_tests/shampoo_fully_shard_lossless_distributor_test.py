"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from collections.abc import Callable
from functools import partial
from typing import cast, overload

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.distributor.gpu_tests.distributor_test_utils import (
    DistributorOnEmptyParamTest,
)
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_fully_shard_lossless_distributor import (
    FullyShardLosslessDistributor,
)
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DefaultSingleDeviceDistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    SingleDeviceDistributedConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_models_devices_on_weight_and_loss,
    construct_training_problem,
    train_model,
)

from torch import distributed as dist, nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

PRECONDITIONER_DIM = 4

# Model layer dimensions were chosen semi-arbitrarily to cover different scenarios:
# - The 1st one has parameters that aligns with the Shampoo preconditioner dim after FSDP sharding;
# - The 2nd and 3rd ones have different numbers of parameters and don't align with the preconditioner dim.
# In these cases, the lossless FSDP Distributor could still guarantee the identicalness with default Shampoo.
TEST_MODEL_LAYER_DIMS: tuple[tuple[int, ...], ...] = (
    (4 * PRECONDITIONER_DIM, 2 * PRECONDITIONER_DIM, 1),
    (3 * PRECONDITIONER_DIM - 1, PRECONDITIONER_DIM + 1, PRECONDITIONER_DIM - 1),
    (2, 2 * PRECONDITIONER_DIM - 1),
)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class ShampooFullyShardLosslessDistributorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @overload
    @staticmethod
    def _construct_model(
        model_linear_layers_dims: tuple[int, ...],
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]: ...

    @overload
    @staticmethod
    def _construct_model(
        model_linear_layers_dims: tuple[int, ...],
        post_model_decoration: Callable[
            [nn.Module], FSDPModule
        ] = lambda x: fully_shard(x),
    ) -> tuple[FSDPModule, nn.Module, torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def _construct_model(
        model_linear_layers_dims: tuple[int, ...],
        post_model_decoration: Callable[
            [nn.Module], nn.Module | FSDPModule
        ] = lambda x: x,
    ) -> tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]:
        # model dead layers won't parpicipate in the training and thus don't have grads.
        model_dead_layers_dims = (PRECONDITIONER_DIM, 1)
        # Using partial here to prevent Pyre complain on incompatible parameter type.
        return partial(
            construct_training_problem, post_model_decoration=post_model_decoration
        )(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.1,
        )

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: FullyShardDistributedConfig | SingleDeviceDistributedConfig,
    ) -> Callable[[ParamsT], torch.optim.Optimizer]:
        return partial(
            DistributedShampoo,
            lr=0.001,
            betas=(0.9, 1.0),
            epsilon=1e-8,
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=PRECONDITIONER_DIM,
            precondition_frequency=1,
            start_preconditioning_step=2,
            use_decoupled_weight_decay=True,
            grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
            distributed_config=distributed_config,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize("model_linear_layers_dims", TEST_MODEL_LAYER_DIMS)
    def test_all_ranks_with_no_grads(
        self,
        model_linear_layers_dims: tuple[int, ...],
    ) -> None:
        fully_shard_config = FullyShardDistributedConfig(
            param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
        )

        steps_without_gradients = 2
        with unittest.mock.patch("torch.Tensor.backward") as mock_backward:
            # By mocking the backward() method, we're intercepting gradient calculation.
            # This effectively simulates running forward passes without computing gradients.
            train_model(
                optim_factory=ShampooFullyShardLosslessDistributorTest._shampoo_optim_factory(
                    distributed_config=fully_shard_config,
                ),
                model_factory=partial(
                    ShampooFullyShardLosslessDistributorTest._construct_model,
                    model_linear_layers_dims=model_linear_layers_dims,
                    post_model_decoration=partial(fully_shard),
                ),
                num_steps=steps_without_gradients,
            )

        # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
        self.assertEqual(mock_backward.call_count, steps_without_gradients)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize("model_linear_layers_dims", TEST_MODEL_LAYER_DIMS)
    def test_fully_shard_shampoo_against_default_shampoo(
        self,
        model_linear_layers_dims: tuple[int, ...],
    ) -> None:
        fully_shard_config = FullyShardDistributedConfig(
            param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
        )
        control_model_factory = partial(
            ShampooFullyShardLosslessDistributorTest._construct_model,
            model_linear_layers_dims=model_linear_layers_dims,
        )
        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFullyShardLosslessDistributorTest._shampoo_optim_factory(
                distributed_config=DefaultSingleDeviceDistributedConfig,
            ),
            control_model_factory=control_model_factory,
            experimental_optim_factory=ShampooFullyShardLosslessDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config,
            ),
            experimental_model_factory=partial(
                control_model_factory,
                post_model_decoration=partial(fully_shard),
            ),
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class FullyShardLosslessDistributorOnEmptyParamTest(
    DTensorTestBase, DistributorOnEmptyParamTest.Interface
):
    @property
    def world_size(self) -> int:
        return 2

    def _construct_model_and_distributor(
        self,
    ) -> tuple[nn.Module, FullyShardLosslessDistributor]:
        # Create a model with specific configuration:
        # - linear_layers are empty params (second dimension is 0)
        # - dead_layers will be replicated on the two ranks. After the merge and block,
        #   each rank will have 3 blocks of torch.size((PRECONDITIONER_DIM, PRECONDITIONER_DIM))
        assert isinstance(
            model := construct_training_problem(
                model_linear_layers_dims=(PRECONDITIONER_DIM, 0),
                model_dead_layers_dims=(PRECONDITIONER_DIM, 3 * PRECONDITIONER_DIM),
                enable_learnable_scalar=False,  # Disable 0D learnable parameter because FSDP doesn't support it.
                device=torch.device("cuda"),
                fill=0.01,
                post_model_decoration=partial(fully_shard),
            )[0],
            nn.Module,
        )
        shampoo_optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 1.0),
            epsilon=1e-8,
            momentum=0.0,
            weight_decay=0.0,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            max_preconditioner_dim=PRECONDITIONER_DIM,
            distributed_config=FullyShardDistributedConfig(
                param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
            ),
        )
        distributor = FullyShardLosslessDistributor(
            param_group=shampoo_optimizer.param_groups[0]
        )

        # Get the weight of the linear layers (which is empty) and set its gradient
        linear_layers: nn.ModuleList = cast(nn.ModuleList, model.linear_layers)
        first_linear_layer_weight: torch.Tensor = cast(
            torch.Tensor, linear_layers[0].weight
        )
        assert first_linear_layer_weight.numel() == 0
        first_linear_layer_weight.grad = torch.ones_like(first_linear_layer_weight)

        # Get the weight of the dead layers and set its gradient to None to make sure this is a dead layer
        dead_layers: nn.ModuleList = cast(nn.ModuleList, model.dead_layers)
        first_dead_layer_weight: torch.Tensor = cast(
            torch.Tensor, dead_layers[0].weight
        )
        first_dead_layer_weight.grad = None

        return model, distributor

    @property
    def _expected_masked_blocked_params(self) -> tuple[torch.Tensor, ...]:
        return ()

    @with_comms
    def test_update_params(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_update_params(self)

    @property
    def _expected_local_grad_selector(self) -> tuple[bool, ...]:
        return (False, False, False)

    @with_comms
    def test_local_grad_selector(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_local_grad_selector(self)

    @property
    def _expected_local_blocked_params(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.zeros(
                (PRECONDITIONER_DIM, PRECONDITIONER_DIM),
                dtype=torch.float,
                device="cuda",
            )
            for _ in range(3)
        )

    @with_comms
    def test_local_blocked_params(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_local_blocked_params(self)

    def _expected_local_block_info_list(
        self, model: nn.Module
    ) -> tuple[BlockInfo, ...]:
        # Get the weight parameter from the first dead layer
        dead_layers: nn.ModuleList = cast(nn.ModuleList, model.dead_layers)
        first_dead_layer_weight: torch.Tensor = cast(
            torch.Tensor, dead_layers[0].weight
        )

        # Define expected BlockInfo objects for each rank
        return tuple(
            BlockInfo(
                param=first_dead_layer_weight,
                composable_block_ids=(0, f"rank_{dist.get_rank()}-block_{i}"),
            )
            for i in range(3)
        )

    @with_comms
    def test_local_block_info_list(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_local_block_info_list(self)

    @property
    def _expected_local_masked_block_grads(self) -> tuple[torch.Tensor, ...]:
        return ()

    @with_comms
    def test_merge_and_block_gradients(self) -> None:  # type: ignore[override]
        DistributorOnEmptyParamTest.Interface.test_merge_and_block_gradients(self)
