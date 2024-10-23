"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import unittest
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    FullyShardShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem

from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor
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
        device: torch.device,
        distributed_config: Optional[FullyShardShampooConfig],
    ) -> Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, bool]:
        IN_DIM = 16
        data = torch.arange(IN_DIM, dtype=torch.float, device=device)
        data /= torch.norm(data)
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

        model_linear_layers_dims = (IN_DIM, 8, 1)
        # model dead layers won't parpicipate in the training and thus don't have grads.
        model_dead_layer_dims = (4, 1)
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layer_dims=model_dead_layer_dims,
            device=device,
            fill=0.1,
        )

        uses_fully_shard = False
        if distributed_config is not None:
            model = fully_shard(model)  # FSDPv2 (per-param FSDP)
            uses_fully_shard = True
        return model, loss, data, target, uses_fully_shard

    @staticmethod
    def _train_model(
        optim_factory: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        model_factory: Callable[
            [torch.device],
            Tuple[
                nn.Module,
                nn.Module,
                torch.Tensor,
                torch.Tensor,
                bool,
            ],
        ],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        model, loss, data, target, uses_fully_shard = model_factory(device)
        params = model.parameters()
        optimizer = optim_factory(params)
        for _ in range(5):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()

        if uses_fully_shard:
            # When FullyShard is used, model parameters are DTensors. We obtain the full value of
            # parameters from DTensors.
            params = []
            for param in model.parameters():
                # Need this assertion to get pass type-checking test.
                assert isinstance(param, DTensor)
                params.append(param.full_tensor().view(-1).detach().cpu())
        else:
            params = [param.view(-1).detach().cpu() for param in model.parameters()]
        return params, objective.detach().cpu()

    @staticmethod
    def _test_two_configs(
        optim_factory1: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        model_factory1: Callable[
            [torch.device],
            Tuple[
                nn.Module,
                nn.Module,
                torch.Tensor,
                torch.Tensor,
                bool,
            ],
        ],
        optim_factory2: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        model_factory2: Callable[
            [torch.device],
            Tuple[
                nn.Module,
                nn.Module,
                torch.Tensor,
                torch.Tensor,
                bool,
            ],
        ],
        device: torch.device,
    ) -> None:
        params1, loss1 = ShampooFullyShardDistributorTest._train_model(
            optim_factory1,
            model_factory1,
            device=device,
        )
        params2, loss2 = ShampooFullyShardDistributorTest._train_model(
            optim_factory2,
            model_factory2,
            device=device,
        )

        torch.testing.assert_close(loss1, loss2)

        # Check the linear layer parameters. Ignore the dead layer parameters.
        torch.testing.assert_close(params1[0], params2[0])
        torch.testing.assert_close(params1[1], params2[1])

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: Optional[FullyShardShampooConfig],
    ) -> Callable[
        [ParamsT],
        torch.optim.Optimizer,
    ]:
        return lambda parameters: (
            lambda distributed_config: DistributedShampoo(
                parameters,
                lr=0.001,
                betas=(0.9, 1.0),
                epsilon=1e-8,
                momentum=0.0,
                weight_decay=0.0,
                max_preconditioner_dim=4,
                precondition_frequency=1,
                start_preconditioning_step=2,
                use_decoupled_weight_decay=True,
                grafting_config=AdaGradGraftingConfig(
                    epsilon=1e-8,
                ),
                distributed_config=distributed_config,
            )
        )(
            distributed_config,
        )

    @staticmethod
    def _model_factory(
        distributed_config: Optional[FullyShardShampooConfig],
    ) -> Callable[
        [torch.device],
        Tuple[
            nn.Module,
            nn.Module,
            torch.Tensor,
            torch.Tensor,
            bool,
        ],
    ]:
        return partial(
            ShampooFullyShardDistributorTest._construct_model,
            distributed_config=distributed_config,
        )

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_shampoo_against_default_shampoo(self) -> None:
        fully_shard_config = FullyShardShampooConfig()
        ShampooFullyShardDistributorTest._test_two_configs(
            ShampooFullyShardDistributorTest._shampoo_optim_factory(
                None,
            ),
            ShampooFullyShardDistributorTest._model_factory(
                None,
            ),
            ShampooFullyShardDistributorTest._shampoo_optim_factory(
                fully_shard_config,
            ),
            ShampooFullyShardDistributorTest._model_factory(
                fully_shard_config,
            ),
            device=torch.device("cuda"),
        )
