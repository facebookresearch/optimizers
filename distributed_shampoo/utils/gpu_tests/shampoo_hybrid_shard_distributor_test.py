"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
import unittest
from collections.abc import Callable
from functools import partial
from itertools import product
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    CommunicationDType,
    HybridShardShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from distributed_shampoo.utils.shampoo_preconditioner_list import SHAMPOO

from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooHybridShardDistributorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "cpu:gloo,cuda:nccl"

    @staticmethod
    def _construct_model(
        device: torch.device,
        distributed_config: HybridShardShampooConfig | None,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, bool]:
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

        if uses_hybrid_shard := isinstance(
            distributed_config, HybridShardShampooConfig
        ):
            # Need this to get pass type-checking test.
            assert distributed_config is not None
            model = fully_shard(
                model,
                mesh=distributed_config.device_mesh,
            )
        return model, loss, data, target, uses_hybrid_shard

    @staticmethod
    def _train_model(
        optim_factory: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        model_factory: Callable[
            [torch.device],
            tuple[
                nn.Module,
                nn.Module,
                torch.Tensor,
                torch.Tensor,
                bool,
            ],
        ],
        device: torch.device,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        model, loss, data, target, uses_hybrid_shard = model_factory(device)
        params = model.parameters()
        optimizer = optim_factory(params)
        for _ in range(5):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()

        linear_layers = model.get_submodule("linear_layers")
        # Need this assertion to get pass type-checking test.
        assert linear_layers is not None
        if uses_hybrid_shard:
            # When Hybrid Shard is used, model parameters are DTensors. We obtain the full value of
            # parameters from DTensors.
            params_list = []
            # We only care model_linear_layers_dim params, not model_dead_layer params.
            for param in linear_layers.parameters():
                # Need this assertion to get pass type-checking test.
                assert isinstance(param, DTensor)
                params_list.append(param.full_tensor().view(-1).detach().cpu())
        else:
            params_list = [
                param.view(-1).detach().cpu() for param in linear_layers.parameters()
            ]

        return params_list, objective.detach().cpu()

    @staticmethod
    def _test_two_configs(
        optim_factory1: Callable[
            [ParamsT],
            torch.optim.Optimizer,
        ],
        model_factory1: Callable[
            [torch.device],
            tuple[
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
            tuple[
                nn.Module,
                nn.Module,
                torch.Tensor,
                torch.Tensor,
                bool,
            ],
        ],
        device: torch.device,
    ) -> None:
        params1, loss1 = ShampooHybridShardDistributorTest._train_model(
            optim_factory1,
            model_factory1,
            device=device,
        )
        params2, loss2 = ShampooHybridShardDistributorTest._train_model(
            optim_factory2,
            model_factory2,
            device=device,
        )
        torch.testing.assert_close(loss1, loss2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(params1, params2)

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: HybridShardShampooConfig | None,
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
        distributed_config: HybridShardShampooConfig | None,
    ) -> Callable[
        [torch.device],
        tuple[
            nn.Module,
            nn.Module,
            torch.Tensor,
            torch.Tensor,
            bool,
        ],
    ]:
        return partial(
            ShampooHybridShardDistributorTest._construct_model,
            distributed_config=distributed_config,
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_hybrid_shard_shampoo_against_default_shampoo(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        for num_trainers_per_group, (
            communication_dtype,
            communicate_params,
        ) in product(
            (-1, 1, 2),
            (
                (CommunicationDType.DEFAULT, False),
                (CommunicationDType.DEFAULT, True),
                (CommunicationDType.FP16, False),
                (CommunicationDType.BF16, False),
            ),
        ):
            hybrid_shard_config = HybridShardShampooConfig(
                device_mesh=mesh_2d,
                communication_dtype=communication_dtype,
                num_trainers_per_group=num_trainers_per_group,
                communicate_params=communicate_params,
            )

            with self.subTest(
                communication_dtype=communication_dtype,
                num_trainers_per_group=num_trainers_per_group,
                communicate_params=communicate_params,
            ):
                ShampooHybridShardDistributorTest._test_two_configs(
                    ShampooHybridShardDistributorTest._shampoo_optim_factory(
                        None,
                    ),
                    ShampooHybridShardDistributorTest._model_factory(
                        None,
                    ),
                    ShampooHybridShardDistributorTest._shampoo_optim_factory(
                        distributed_config=hybrid_shard_config,
                    ),
                    ShampooHybridShardDistributorTest._model_factory(
                        hybrid_shard_config,
                    ),
                    device=torch.device("cuda"),
                )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_hybrid_shard_shampoo_block_index(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=mesh_2d,
        )
        model_factory = ShampooHybridShardDistributorTest._model_factory(
            hybrid_shard_config
        )
        optim_factory = ShampooHybridShardDistributorTest._shampoo_optim_factory(
            hybrid_shard_config
        )
        model, loss, data, target, _ = model_factory(torch.device("cuda"))
        params = model.parameters()
        optimizer = optim_factory(params)
        assert isinstance(optimizer, DistributedShampoo)
        state_dict = optimizer.distributed_state_dict(
            key_to_param=model.named_parameters()
        )
        flattened_state_dict = flatten_state_dict(state_dict)[0]

        # Note that we get the local rank corresponding to the second mesh dimension
        # because the first mesh dimension corresponds to replication and the second
        # mesh dimension corresponds to the sharding dimension.
        #
        # We expect that the rank should correspond to the rank in the shard dimension
        # in order to avoid having the same key.
        rank = mesh_2d.get_local_rank(mesh_dim=1)
        matches = 0
        for key in flattened_state_dict.keys():
            if SHAMPOO in key:
                with self.subTest(key=key):
                    self.assertIn(f"rank_{rank}-block_", key)
                    matches += 1
        self.assertGreater(matches, 0)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_number_of_trainers_per_group_out_of_range(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=mesh_2d,
            num_trainers_per_group=3,
        )

        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Invalid number of trainers per group: 3. Must be between [1, 2] or set to -1."
            ),
            ShampooHybridShardDistributorTest._train_model,
            optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config,
            ),
            model_factory=ShampooHybridShardDistributorTest._model_factory(
                hybrid_shard_config
            ),
            device=torch.device("cuda"),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_dist_is_initialized(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=mesh_2d,
        )

        with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
            self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "HybridShardDistributor needs torch.distributed to be initialized!"
                ),
                ShampooHybridShardDistributorTest._train_model,
                optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                    distributed_config=hybrid_shard_config,
                ),
                model_factory=ShampooHybridShardDistributorTest._model_factory(
                    hybrid_shard_config
                ),
                device=torch.device("cuda"),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_incompatible_replicated_group_size_and_num_trainers_per_group(
        self,
    ) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=mesh_2d,
            num_trainers_per_group=3,
        )

        # Hijack the DeviceMesh.size() method to return 4 instead of 2 to bypass the check of num_trainers_per_group.
        with mock.patch.object(
            torch.distributed.device_mesh.DeviceMesh, "size", return_value=4
        ):
            self.assertRaisesRegex(
                ValueError,
                re.escape(
                    "distributed_config.num_trainers_per_group=3 must divide self._replicated_group_size=4!"
                ),
                ShampooHybridShardDistributorTest._train_model,
                optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                    distributed_config=hybrid_shard_config,
                ),
                model_factory=ShampooHybridShardDistributorTest._model_factory(
                    hybrid_shard_config
                ),
                device=torch.device("cuda"),
            )
