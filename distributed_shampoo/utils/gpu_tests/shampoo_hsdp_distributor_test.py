"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import re
from functools import partial
from itertools import pairwise, product
from typing import Callable, List, Optional, Tuple
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    CommunicationDType,
    HSDPShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata
from distributed_shampoo.utils.shampoo_preconditioner_list import SHAMPOO

from torch import nn
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest


class ShampooHSDPDistributorTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @staticmethod
    def _construct_model(
        device: torch.device,
        distributed_config: Optional[HSDPShampooConfig],
    ) -> Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        # NOTE: We construct the model here specifically in order to ensure that
        #       HSDP Shampoo and default Shampoo produce equivalent results.
        #       This requires us to construct a model such that FSDP will split the
        #       parameters such that the resulting blocks from tensor block recovery
        #       and merging/blocking are equivalent to what would be obtained by
        #       merging/blocking on the original parameters.
        #
        #       An additional constraint imposed by FSDP is from PT2; the split must be
        #       16-byte aligned. With FP32 elements, this corresponds to 4 elements.
        #
        #       Based on the design of the model below, the model has 512 + 72 + 576 + 64 =
        #       1224 elements, which means that the model will be split at index 612 across
        #       the flattened param in FSDP.
        #       This corresponds to index 612 - 512 - 72 = 28 in the third parameter. Note
        #       that splitting at this index is equivalent to the standard blocking with a
        #       block size of 4.
        model_linear_layers_dims = (16 * 4, 8, 9, 16 * 4, 1)
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layer_dims=None,
            device=device,
            fill=0.01,
        )
        if isinstance(distributed_config, HSDPShampooConfig):
            model = FSDP(
                model,
                device_mesh=distributed_config.device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                use_orig_params=True,
            )
            distributed_config.param_to_metadata = compile_fsdp_parameter_metadata(
                model
            )
            assert (
                sum(param.numel() for param in model.parameters())
                == sum(a * b for a, b in pairwise(model_linear_layers_dims)) // 2
            ), f"{sum(param.numel() for param in model.parameters())=}, {sum(a * b for a, b in pairwise(model_linear_layers_dims)) // 2=}"
        return model, loss, data, target

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
            ],
        ],
        device: torch.device,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        model, loss, data, target = model_factory(device)
        params = model.parameters()
        optimizer = optim_factory(params)
        for _ in range(5):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()

        with FSDP.summon_full_params(model):
            return [
                param.view(-1).detach().cpu() for param in model.parameters()
            ], objective.detach().cpu()

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
            ],
        ],
        device: torch.device,
    ) -> None:
        params1, loss1 = ShampooHSDPDistributorTest._train_model(
            optim_factory1,
            model_factory1,
            device=device,
        )
        params2, loss2 = ShampooHSDPDistributorTest._train_model(
            optim_factory2,
            model_factory2,
            device=device,
        )
        torch.testing.assert_close(loss1, loss2, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(params1, params2)

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: Optional[HSDPShampooConfig],
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
        distributed_config: Optional[HSDPShampooConfig],
    ) -> Callable[
        [torch.device],
        Tuple[
            nn.Module,
            nn.Module,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        return partial(
            ShampooHSDPDistributorTest._construct_model,
            distributed_config=distributed_config,
        )

    @skip_if_lt_x_gpu(4)
    def test_hsdp_shampoo_against_default_shampoo(self) -> None:
        mesh_2d = init_device_mesh("cuda", (2, 2))
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
            hsdp_config = HSDPShampooConfig(
                param_to_metadata={},
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
                ShampooHSDPDistributorTest._test_two_configs(
                    ShampooHSDPDistributorTest._shampoo_optim_factory(
                        None,
                    ),
                    ShampooHSDPDistributorTest._model_factory(
                        None,
                    ),
                    ShampooHSDPDistributorTest._shampoo_optim_factory(
                        distributed_config=hsdp_config,
                    ),
                    ShampooHSDPDistributorTest._model_factory(
                        hsdp_config,
                    ),
                    device=torch.device("cuda"),
                )

    @skip_if_lt_x_gpu(4)
    def test_hsdp_shampoo_block_index(self) -> None:
        mesh_2d = init_device_mesh("cuda", (2, 2))
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=mesh_2d,
        )
        model_factory = ShampooHSDPDistributorTest._model_factory(hsdp_config)
        optim_factory = ShampooHSDPDistributorTest._shampoo_optim_factory(hsdp_config)
        model, loss, data, target = model_factory(torch.device("cuda"))
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

    @skip_if_lt_x_gpu(4)
    def test_number_of_trainers_per_group_out_of_range(self) -> None:
        mesh_2d = init_device_mesh("cuda", (2, 2))
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=mesh_2d,
            num_trainers_per_group=3,
        )

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Invalid number of trainers per group: 3. Must be between [1, 2] or set to -1."
            ),
        ):
            ShampooHSDPDistributorTest._train_model(
                optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=hsdp_config,
                ),
                model_factory=ShampooHSDPDistributorTest._model_factory(hsdp_config),
                device=torch.device("cuda"),
            )

    @skip_if_lt_x_gpu(4)
    def test_dist_is_initialized(self) -> None:
        mesh_2d = init_device_mesh("cuda", (2, 2))
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=mesh_2d,
        )

        with mock.patch.object(
            torch.distributed, "is_initialized", return_value=False
        ), self.assertRaisesRegex(
            RuntimeError,
            re.escape("HSDPDistributor needs torch.distributed to be initialized!"),
        ):
            ShampooHSDPDistributorTest._train_model(
                optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=hsdp_config,
                ),
                model_factory=ShampooHSDPDistributorTest._model_factory(hsdp_config),
                device=torch.device("cuda"),
            )

    @skip_if_lt_x_gpu(4)
    def test_incompatible_replicated_group_size_and_num_trainers_per_group(
        self,
    ) -> None:
        mesh_2d = init_device_mesh("cuda", (2, 2))
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=mesh_2d,
            num_trainers_per_group=3,
        )

        # Hijack the DeviceMesh.size() method to return 4 instead of 2 to bypass the check of num_trainers_per_group.
        with mock.patch.object(
            torch.distributed.device_mesh.DeviceMesh, "size", return_value=4
        ), self.assertRaisesRegex(
            ValueError,
            re.escape(
                "distributed_config.num_trainers_per_group=3 must divide self._replicated_group_size=4!"
            ),
        ):
            ShampooHSDPDistributorTest._train_model(
                optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=hsdp_config,
                ),
                model_factory=ShampooHSDPDistributorTest._model_factory(hsdp_config),
                device=torch.device("cuda"),
            )
