"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import math
import re
import unittest
from collections.abc import Callable
from functools import partial
from itertools import filterfalse
from typing import overload
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    DDPShampooConfig,
    FullyShardShampooConfig,
    HybridShardShampooConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_models_devices_on_weight_and_loss,
    construct_training_problem,
    train_model,
)
from distributed_shampoo.utils.shampoo_preconditioner_list import SHAMPOO

from torch import nn
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
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
class ShampooHybridShardDistributorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

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
        # NOTE: We construct the model here specifically in order to ensure that HybridShard
        #       Shampoo, and default Shampoo produce equivalent results.
        #       This requires us to construct a model such that FullyShard will split the
        #       parameters such that the preconditioners created between the HybridShard Shampoo,
        #       and default Shampoo are equivalent.
        #
        #       In a (2, 2) mesh, we have the following parameter distribution:
        #
        #      +----------------+                       +----------------+
        #      |     [4, 16]    |                       |     [4, 16]    |
        #      |      GPU0      |                       |      GPU1      |
        #     --------------------     +------+        --------------------     +------+
        #      |     [4, 16]    |      |[4, 4]|         |     [4, 16]    |      |[4, 4]|
        #      |      GPU2      |      |      |         |      GPU3      |      |      |
        #      +----------------+      +------+         +----------------+      +------+
        #
        #      Each FSDP group has the complete model. (GPU0, GPU2) and (GPU1, GPU3) are
        #      2 FDSP groups.
        #
        #      For the first linear layer, each GPU has a [4, 16] parameter. The blocked
        #      parameters are of size [4, 4] and each GPU has four local blocks. In comparison,
        #      with default shampoo, the eight blocks are replicated on four GPUs.
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
        distributed_config: DDPShampooConfig
        | FullyShardShampooConfig
        | HybridShardShampooConfig
        | None,
        start_preconditioning_step: int = 2,
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
            start_preconditioning_step=start_preconditioning_step,
            use_decoupled_weight_decay=True,
            grafting_config=AdaGradGraftingConfig(epsilon=1e-8),
            distributed_config=distributed_config,
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "communication_dtype, communicate_params",
        (
            (torch.float32, False),
            (torch.float32, True),
            (torch.float16, False),
            (torch.bfloat16, False),
        ),
    )
    @parametrize("num_trainers_per_group", (-1, 1, 2))
    def test_hybrid_shard_shampoo_against_default_shampoo(
        self,
        num_trainers_per_group: int,
        communication_dtype: torch.dtype,
        communicate_params: bool,
    ) -> None:
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
            ),
            communication_dtype=communication_dtype,
            num_trainers_per_group=num_trainers_per_group,
            communicate_params=communicate_params,
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=None,
            ),
            control_model_factory=ShampooHybridShardDistributorTest._construct_model,
            experimental_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config,
            ),
            experimental_model_factory=partial(
                ShampooHybridShardDistributorTest._construct_model,
                post_model_decoration=partial(
                    fully_shard, mesh=hybrid_shard_config.device_mesh
                ),
            ),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "communication_dtype, communicate_params",
        (
            (torch.float32, False),
            (torch.float32, True),
            (torch.float16, False),
            (torch.bfloat16, False),
        ),
    )
    @parametrize("num_trainers_per_group", (-1, 1, 2, 4))
    def test_hybrid_shampoo_n_by_one_mesh_against_default_shampoo(
        self,
        num_trainers_per_group: int,
        communication_dtype: torch.dtype,
        communicate_params: bool,
    ) -> None:
        """
        Testing the correctness of hybrid shard shampoo distributor of (n, 1) mesh
        by comparing it with default shampoo. (n, 1) mesh is a special case of hybrid shard.
        The shard size is 1 so it is equivalent to default or DDP Shampoo.
        """
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (4, 1), mesh_dim_names=("replicate", "shard")
            ),
            communication_dtype=communication_dtype,
            num_trainers_per_group=num_trainers_per_group,
            communicate_params=communicate_params,
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=None,
            ),
            control_model_factory=ShampooHybridShardDistributorTest._construct_model,
            experimental_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config,
            ),
            experimental_model_factory=partial(
                ShampooHybridShardDistributorTest._construct_model,
                post_model_decoration=partial(
                    fully_shard, mesh=hybrid_shard_config.device_mesh
                ),
            ),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "start_preconditioning_step", (2, math.inf)
    )  # math.inf here is to test the grafting similarities between HSDP2 and DDP
    @parametrize(
        "communication_dtype, communicate_params",
        (
            (torch.float32, False),
            (torch.float32, True),
            (torch.float16, False),
            (torch.bfloat16, False),
        ),
    )
    @parametrize("num_trainers_per_group", (-1, 1, 2, 4))
    def test_hybrid_shampoo_n_by_one_mesh_against_ddp_shampoo(
        self,
        num_trainers_per_group: int,
        communication_dtype: torch.dtype,
        communicate_params: bool,
        start_preconditioning_step: int,
    ) -> None:
        """
        Testing the correctness of hybrid shard Shampoo distributor of (n, 1) mesh
        by comparing it with DDP Shampoo. (n, 1) mesh is a special case of hybrid shard.
        The shard size is 1 so it is equivalent to DDP Shampoo.
        """
        ddp_config = DDPShampooConfig(
            communication_dtype=communication_dtype,
            num_trainers_per_group=num_trainers_per_group,
        )
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (4, 1), mesh_dim_names=("replicate", "shard")
            ),
            communication_dtype=communication_dtype,
            num_trainers_per_group=num_trainers_per_group,
            communicate_params=communicate_params,
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=ddp_config,
                start_preconditioning_step=start_preconditioning_step,
            ),
            control_model_factory=ShampooHybridShardDistributorTest._construct_model,
            experimental_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config,
                start_preconditioning_step=start_preconditioning_step,
            ),
            experimental_model_factory=partial(
                ShampooHybridShardDistributorTest._construct_model,
                post_model_decoration=partial(
                    fully_shard, mesh=hybrid_shard_config.device_mesh
                ),
            ),
            rtol=0.0,
            atol=0.0,
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "communication_dtype, communicate_params",
        (
            (torch.float32, False),
            (torch.float32, True),
            (torch.float16, False),
            (torch.bfloat16, False),
        ),
    )
    @parametrize("num_trainers_per_group", (-1, 1, 2))
    def test_hybrid_shard_shampoo_config_against_fully_shard_shampoo_config(
        self,
        num_trainers_per_group: int,
        communication_dtype: torch.dtype,
        communicate_params: bool,
    ) -> None:
        """
        Testing the correctness of hybrid shard shampoo distributor by comparing it with
        fully shard distributor.
        """
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        fully_shard_config = FullyShardShampooConfig()  # type: ignore[abstract]
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=mesh_2d,
            communication_dtype=communication_dtype,
            num_trainers_per_group=num_trainers_per_group,
            communicate_params=communicate_params,
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=fully_shard_config
            ),
            control_model_factory=partial(
                ShampooHybridShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
            experimental_optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config
            ),
            experimental_model_factory=partial(
                ShampooHybridShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("communicate_params", (False, True))
    def test_all_ranks_with_no_grads(self, communicate_params: bool) -> None:
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
            ),
            communicate_params=communicate_params,
        )

        steps_without_gradients = 2
        with unittest.mock.patch("torch.Tensor.backward") as mock_backward:
            # By mocking the backward() method, we're intercepting gradient calculation.
            # This effectively simulates running forward passes without computing gradients.
            train_model(
                optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                    distributed_config=hybrid_shard_config
                ),
                model_factory=partial(
                    ShampooHybridShardDistributorTest._construct_model,
                    post_model_decoration=partial(
                        fully_shard, mesh=hybrid_shard_config.device_mesh
                    ),
                ),
                num_steps=steps_without_gradients,
            )

        # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
        self.assertEqual(mock_backward.call_count, steps_without_gradients)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_hybrid_shard_shampoo_block_index(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
        )
        model, _, _, _, optimizer = train_model(
            optim_factory=ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=HybridShardShampooConfig(device_mesh=mesh_2d)
            ),
            model_factory=partial(
                ShampooHybridShardDistributorTest._construct_model,
                post_model_decoration=partial(fully_shard, mesh=mesh_2d),
            ),
        )
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, DistributedShampoo)
        state_dict = optimizer.distributed_state_dict(
            key_to_param=model.named_parameters()
        )
        flattened_state_dict = flatten_state_dict(state_dict["state"])[0]

        # Note that we get the local rank corresponding to the second mesh dimension
        # because the first mesh dimension corresponds to replication and the second
        # mesh dimension corresponds to the sharding dimension.
        #
        # We expect that the rank should correspond to the rank in the shard dimension
        # in order to avoid having the same key.
        rank: int = mesh_2d.get_local_rank(mesh_dim=1)

        def expected_key_criterion(key: str) -> bool:
            return f"rank_{rank}-block_" in key

        keys_with_shampoo = filter(
            lambda key: SHAMPOO in key, flattened_state_dict.keys()
        )
        keys_with_expected_key, keys_without_expected_key = (
            list(filter(expected_key_criterion, keys_with_shampoo)),
            list(filterfalse(expected_key_criterion, keys_with_shampoo)),
        )
        self.assertFalse(
            keys_without_expected_key, msg=f"{keys_without_expected_key=} is not empty."
        )
        self.assertTrue(keys_with_expected_key)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_number_of_trainers_per_group_out_of_range(self) -> None:
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
            ),
            num_trainers_per_group=3,
        )
        model = ShampooHybridShardDistributorTest._construct_model(
            post_model_decoration=partial(
                fully_shard, mesh=hybrid_shard_config.device_mesh
            ),
        )[0]
        assert isinstance(model, nn.Module)

        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Invalid number of trainers per group: 3. Must be between [1, 2] or set to -1."
            ),
            ShampooHybridShardDistributorTest._shampoo_optim_factory(
                distributed_config=hybrid_shard_config,
            ),
            model.parameters(),
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_dist_is_initialized(self) -> None:
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
            )
        )
        model = ShampooHybridShardDistributorTest._construct_model(
            post_model_decoration=partial(
                fully_shard, mesh=hybrid_shard_config.device_mesh
            ),
        )[0]
        assert isinstance(model, nn.Module)

        with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
            self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    "HybridShardDistributor needs torch.distributed to be initialized!"
                ),
                ShampooHybridShardDistributorTest._shampoo_optim_factory(
                    distributed_config=hybrid_shard_config,
                ),
                model.parameters(),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_incompatible_replicated_group_size_and_num_trainers_per_group(
        self,
    ) -> None:
        hybrid_shard_config = HybridShardShampooConfig(
            device_mesh=init_device_mesh(
                "cuda", (2, 2), mesh_dim_names=("replicate", "shard")
            ),
            num_trainers_per_group=3,
        )
        model = ShampooHybridShardDistributorTest._construct_model(
            post_model_decoration=partial(
                fully_shard, mesh=hybrid_shard_config.device_mesh
            ),
        )[0]
        assert isinstance(model, nn.Module)

        # Hijack the DeviceMesh.size() method to return 4 instead of 2 to bypass the check of num_trainers_per_group.
        with mock.patch.object(
            torch.distributed.device_mesh.DeviceMesh, "size", return_value=4
        ):
            self.assertRaisesRegex(
                ValueError,
                re.escape(
                    "distributed_config.num_trainers_per_group=3 must divide self._replicated_group_size=4!"
                ),
                ShampooHybridShardDistributorTest._shampoo_optim_factory(
                    distributed_config=hybrid_shard_config,
                ),
                model.parameters(),
            )
