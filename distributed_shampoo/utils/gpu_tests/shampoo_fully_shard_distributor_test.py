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

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    FullyShardShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_models_devices_on_weight_and_loss,
    construct_training_problem,
    train_model,
)

from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooFullyShardDistributorTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
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

        model_linear_layers_dims = (16, 8, 1)
        # model dead layers won't parpicipate in the training and thus don't have grads.
        model_dead_layers_dims = (4, 1)
        return construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=model_dead_layers_dims,
            device=torch.device("cuda"),
            fill=0.1,
            post_model_decoration=post_model_decoration,
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
            max_preconditioner_dim=4,
            precondition_frequency=1,
            start_preconditioning_step=2,
            use_decoupled_weight_decay=True,
            grafting_config=AdaGradGraftingConfig(epsilon=1e-8),
            distributed_config=distributed_config,
        )

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
                post_model_decoration=fully_shard,
            ),
            num_steps=steps_with_gradients,
        )

        steps_without_gradients = 3
        for _ in range(steps_without_gradients):
            objective = loss(model(data), target)
            objective.backward()

            # Experiment setup: all ranks get no gradients.
            optimizer.zero_grad()

            optimizer.step()

        assert isinstance(optimizer, DistributedShampoo)
        # For each rank, no matter getting gradients or not, the step should be updated.
        self.assertEqual(
            optimizer.distributed_state_dict(key_to_param=model.named_parameters())[
                "state"
            ]["linear_layers.0.weight"]['["step"]'].item(),
            steps_with_gradients + steps_without_gradients,
        )

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
                post_model_decoration=fully_shard,
            ),
        )
