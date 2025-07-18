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
from itertools import filterfalse, pairwise
from unittest import mock

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdaGradGraftingConfig, HSDPShampooConfig
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_models_devices_on_weight_and_loss,
    construct_training_problem,
    train_model,
)
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata
from distributed_shampoo.utils.shampoo_preconditioner_list import SHAMPOO

from torch import nn
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP1, ShardingStrategy
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


PRECONDITIONER_DIM = 4


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
@instantiate_parametrized_tests
class ShampooHSDPDistributorTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @staticmethod
    def _construct_model(
        post_model_decoration: Callable[[nn.Module], nn.Module] = lambda x: x,
        distributed_config: HSDPShampooConfig | None = None,
    ) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
        # NOTE: We construct the model here specifically in order to ensure that
        #       HSDP Shampoo and default Shampoo produce equivalent results.
        #       This requires us to construct a model such that FSDP1 will split the
        #       parameters such that the resulting blocks from tensor block recovery
        #       and merging/blocking are equivalent to what would be obtained by
        #       merging/blocking on the original parameters.
        #
        #       An additional constraint imposed by FSDP1 is from PT2; the split must be
        #       16-byte aligned. With FP32 elements, this corresponds to 4 elements.
        #
        #       Based on the design of the model below, the model has 512 + 72 + 576 + 64 =
        #       1224 elements, which means that the model will be split at index 612 across
        #       the flattened param in FSDP1.
        #       This corresponds to index 612 - 512 - 72 = 28 in the third parameter. Note
        #       that splitting at this index is equivalent to the standard blocking with a
        #       block size of 4.
        model_linear_layers_dims = (
            4 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
            2 * PRECONDITIONER_DIM,
            9,
            4 * PRECONDITIONER_DIM * PRECONDITIONER_DIM,
            1,
        )
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layers_dims=None,
            enable_learnable_scalar=False,  # Disable 0D learable parameter because FSDP doesn't support it.
            device=torch.device("cuda"),
            fill=0.01,
            post_model_decoration=post_model_decoration,
        )
        if isinstance(distributed_config, HSDPShampooConfig):
            assert (
                sum(param.numel() for param in model.parameters())
                == sum(a * b for a, b in pairwise(model_linear_layers_dims)) // 2
            ), f"{sum(param.numel() for param in model.parameters())=}, {sum(a * b for a, b in pairwise(model_linear_layers_dims)) // 2=}"
            distributed_config.param_to_metadata = compile_fsdp_parameter_metadata(
                model
            )
        return model, loss, data, target

    @staticmethod
    def _shampoo_optim_factory(
        distributed_config: HSDPShampooConfig | None,
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
    def test_hsdp_shampoo_against_default_shampoo(
        self,
        num_trainers_per_group: int,
        communication_dtype: torch.dtype,
        communicate_params: bool,
    ) -> None:
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=init_device_mesh("cuda", (2, 2)),
            communication_dtype=communication_dtype,
            num_trainers_per_group=num_trainers_per_group,
            communicate_params=communicate_params,
        )

        compare_two_optimizers_models_devices_on_weight_and_loss(
            control_optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                distributed_config=None,
            ),
            control_model_factory=ShampooHSDPDistributorTest._construct_model,
            experimental_optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                distributed_config=hsdp_config,
            ),
            experimental_model_factory=partial(
                ShampooHSDPDistributorTest._construct_model,
                post_model_decoration=partial(
                    FSDP1,
                    device_mesh=hsdp_config.device_mesh,
                    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    use_orig_params=True,
                ),
                distributed_config=hsdp_config,
            ),
        )

    @skip_if_lt_x_gpu(4)
    def test_hsdp_shampoo_block_index(self) -> None:
        mesh_2d = init_device_mesh("cuda", (2, 2))
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=mesh_2d,
        )

        model, _, _, _, optimizer = train_model(
            optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                hsdp_config
            ),
            model_factory=partial(
                ShampooHSDPDistributorTest._construct_model,
                post_model_decoration=partial(
                    FSDP1,
                    device_mesh=hsdp_config.device_mesh,
                    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    use_orig_params=True,
                ),
                distributed_config=hsdp_config,
            ),
        )
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

    @skip_if_lt_x_gpu(4)
    @parametrize("communicate_params", (False, True))
    def test_all_ranks_with_no_grads(self, communicate_params: bool) -> None:
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=init_device_mesh("cuda", (2, 2)),
            communicate_params=communicate_params,
        )

        steps_without_gradients = 2
        with unittest.mock.patch("torch.Tensor.backward") as mock_backward:
            # By mocking the backward() method, we're intercepting gradient calculation.
            # This effectively simulates running forward passes without computing gradients.
            train_model(
                optim_factory=ShampooHSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=hsdp_config
                ),
                model_factory=partial(
                    ShampooHSDPDistributorTest._construct_model,
                    post_model_decoration=partial(
                        FSDP1,
                        device_mesh=hsdp_config.device_mesh,
                        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                        use_orig_params=True,
                    ),
                    distributed_config=hsdp_config,
                ),
                num_steps=steps_without_gradients,
            )

        # Verify that the backward() method was called the expected number of times and the training loop completed successfully.
        self.assertEqual(mock_backward.call_count, steps_without_gradients)

    @skip_if_lt_x_gpu(4)
    def test_number_of_trainers_per_group_out_of_range(self) -> None:
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=init_device_mesh("cuda", (2, 2)),
            num_trainers_per_group=3,
        )
        model = ShampooHSDPDistributorTest._construct_model(
            post_model_decoration=partial(
                FSDP1,
                device_mesh=hsdp_config.device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                use_orig_params=True,
            ),
            distributed_config=hsdp_config,
        )[0]

        self.assertRaisesRegex(
            ValueError,
            re.escape(
                "Invalid number of trainers per group: 3. Must be between [1, 2] or set to -1."
            ),
            ShampooHSDPDistributorTest._shampoo_optim_factory(
                distributed_config=hsdp_config,
            ),
            model.parameters(),
        )

    @skip_if_lt_x_gpu(4)
    def test_dist_is_initialized(self) -> None:
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=init_device_mesh("cuda", (2, 2)),
        )
        model = ShampooHSDPDistributorTest._construct_model(
            post_model_decoration=partial(
                FSDP1,
                device_mesh=hsdp_config.device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                use_orig_params=True,
            ),
            distributed_config=hsdp_config,
        )[0]

        with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
            self.assertRaisesRegex(
                RuntimeError,
                re.escape("HSDPDistributor needs torch.distributed to be initialized!"),
                ShampooHSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=hsdp_config
                ),
                model.parameters(),
            )

    @skip_if_lt_x_gpu(4)
    def test_incompatible_replicated_group_size_and_num_trainers_per_group(
        self,
    ) -> None:
        hsdp_config = HSDPShampooConfig(
            param_to_metadata={},
            device_mesh=init_device_mesh("cuda", (2, 2)),
            num_trainers_per_group=3,
        )
        model = ShampooHSDPDistributorTest._construct_model(
            post_model_decoration=partial(
                FSDP1,
                device_mesh=hsdp_config.device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                use_orig_params=True,
            ),
            distributed_config=hsdp_config,
        )[0]

        # Hijack the DeviceMesh.size() method to return 4 instead of 2 to bypass the check of num_trainers_per_group.
        with mock.patch.object(
            torch.distributed.device_mesh.DeviceMesh, "size", return_value=4
        ):
            self.assertRaisesRegex(
                ValueError,
                re.escape(
                    "distributed_config.num_trainers_per_group=3 must divide self._replicated_group_size=4!"
                ),
                ShampooHSDPDistributorTest._shampoo_optim_factory(
                    distributed_config=hsdp_config
                ),
                model.parameters(),
            )
