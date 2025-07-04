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
from typing import overload

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    FullyShardShampooConfig,
    HybridShardShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_models_devices_on_weight_and_loss,
    construct_training_problem,
    train_model,
)

from torch import nn
from torch.distributed.device_mesh import init_device_mesh
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


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class ShampooFullyShardDistributorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @overload
    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]: ...

    @overload
    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[
            [nn.Module], FSDPModule
        ] = lambda x: fully_shard(x),
    ) -> tuple[FSDPModule, nn.Module, torch.Tensor, torch.Tensor]: ...

    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[
            [nn.Module], nn.Module | FSDPModule
        ] = lambda x: x,
    ) -> tuple[nn.Module | FSDPModule, nn.Module, torch.Tensor, torch.Tensor]:
        # NOTE: We construct the model here specifically in order to ensure that
        #       FullyShard Shampoo and default Shampoo produce equivalent results.
        #       This requires us to construct a model such that FullyShard will split the
        #       parameters such that the preconditioners created between the FullyShard
        #       and default Shampoo are equivalent.
        #      +----------------+
        #      |     [4, 16]    |
        #      |      GPU0      |
        #     --------------------     +------+
        #      |     [4, 16]    |      |[4, 4]|
        #      |      GPU1      |      |      |
        #      +----------------+      +------+
        #      For the first linear layer, each GPU has a [4, 16] parameter. The blocked
        #      parameters are of size [4, 4] and each GPU has four local blocks (eight
        #      blocks in total). In comparison, with default shampoo, the eight blocks
        #      are replicated on two GPUs.
        #      Similarly, the second linear layer has a [1, 8] parameter and is split
        #      into two [4] chunks.

        model_linear_layers_dims = (4 * PRECONDITIONER_DIM, 2 * PRECONDITIONER_DIM, 1)
        # model dead layers won't parpicipate in the training and thus don't have grads.
        model_dead_layers_dims = (PRECONDITIONER_DIM, 1)
        # Using partial here to prevent Pyre complain on incompatible parameter type.
        return partial(
            construct_training_problem, post_model_decoration=post_model_decoration
        )(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            enable_learnable_scalar=False,  # Disable 0D learable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.1,
        )

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: FullyShardShampooConfig | None,
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
            grafting_config=AdaGradGraftingConfig(epsilon=1e-8),
            distributed_config=distributed_config,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_all_ranks_with_no_grads(self) -> None:
        fully_shard_config = FullyShardShampooConfig()  # type: ignore[abstract]

        steps_with_gradients = 2
        model, loss, data, target, optimizer = train_model(
            optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config,
            ),
            model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard),
            ),
            num_steps=steps_with_gradients,
        )

        steps_without_gradients = 3
        for _ in range(steps_without_gradients):
            assert isinstance(model, nn.Module)
            objective = loss(model(data), target)
            objective.backward()

            # Experiment setup: all ranks get no gradients.
            optimizer.zero_grad()

            optimizer.step()

        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, DistributedShampoo)
        # For each rank, no matter getting gradients or not, the step should be updated.
        self.assertEqual(
            optimizer.distributed_state_dict(key_to_param=model.named_parameters())[
                "state"
            ]["linear_layers.0.weight"]['["step"]'].item(),
            steps_with_gradients + steps_without_gradients,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_shampoo_against_default_shampoo(self) -> None:
        fully_shard_config = FullyShardShampooConfig()  # type: ignore[abstract]
        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=None,
            ),
            control_model_factory=ShampooFullyShardDistributorTest._construct_model,
            experimental_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config,
            ),
            experimental_model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard),
            ),
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize("communicate_params", (True, False))
    def test_hybrid_shard_shampoo_config_against_fully_shard_shampoo_config_bitwise_identical(
        self, communicate_params: bool
    ) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (1, self.world_size), mesh_dim_names=("replicate", "shard")
        )
        fully_shard_config = FullyShardShampooConfig()  # type: ignore[abstract]
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=mesh_2d, communicate_params=communicate_params
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config
            ),
            control_model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
            experimental_optim_factory=ShampooFullyShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config
            ),
            experimental_model_factory=partial(
                ShampooFullyShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
            total_steps=100,
            rtol=0.0,
            atol=0.0,
        )
