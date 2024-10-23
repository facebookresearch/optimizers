"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest
from typing import List, Tuple

import torch
from distributed_shampoo.shampoo_types import FSDPParameterMetadata
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from distributed_shampoo.utils.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
    parse_fsdp_params,
    parse_fully_shard_params,
)
from torch import distributed as dist, nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parameter import Parameter
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest


# Note: Ideally this function should be resided inside Test as part of setUp() but FSDPTest
#       only calls setUp() on one device; as a result, every device has to call this function
#       separately.
def _create_model_and_params(
    model_linear_layers_dims: tuple[int, ...] = (2, 5, 3),
) -> Tuple[nn.Module, List[Parameter]]:
    model, _, _, _ = construct_training_problem(
        model_linear_layers_dims, model_dead_layer_dims=None, fill=(1.0, 2.0)
    )
    return model, list(model.parameters())


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class CompileFSDPParameterMetadataTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_compile_fsdp_parameter_metadata(self) -> None:
        model, params = _create_model_and_params()
        fsdp_model = FSDP(model, use_orig_params=True)
        actual_fsdp_parameter_metadata = compile_fsdp_parameter_metadata(fsdp_model)

        expected_fsdp_parameter_metadata = (
            {
                params[0]: FSDPParameterMetadata(
                    fqn="linear_layers.0.weight",
                    shape=torch.Size([5, 2]),
                    numel=10,
                    start_idx=0,
                    end_idx=10,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
                params[1]: FSDPParameterMetadata(
                    fqn="linear_layers.1.weight",
                    shape=torch.Size([3, 5]),
                    numel=15,
                    start_idx=0,
                    end_idx=2,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
            }
            if dist.get_rank() == 0
            else {
                params[0]: FSDPParameterMetadata(
                    fqn="linear_layers.0.weight",
                    shape=torch.Size([5, 2]),
                    numel=10,
                    start_idx=0,
                    end_idx=0,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
                params[1]: FSDPParameterMetadata(
                    fqn="linear_layers.1.weight",
                    shape=torch.Size([3, 5]),
                    numel=15,
                    start_idx=2,
                    end_idx=15,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                ),
            }
        )

        self.assertEqual(
            actual_fsdp_parameter_metadata, expected_fsdp_parameter_metadata
        )

    @skip_if_lt_x_gpu(2)
    def test_compile_fsdp_parameter_metadata_with_no_flat_param(self) -> None:
        model, params = _create_model_and_params()
        # Ignored all params in FSDP so there is no flat_param field in FSDP module.
        fsdp_model = FSDP(model, use_orig_params=True, ignored_states=params)
        actual_fsdp_parameter_metadata = compile_fsdp_parameter_metadata(fsdp_model)

        expected_fsdp_parameter_metadata = {}

        self.assertEqual(
            actual_fsdp_parameter_metadata, expected_fsdp_parameter_metadata
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ParseFSDPParamsTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_parse_fsdp_params(self) -> None:
        HYBRID_SHARDING_STRATEGIES_TO_EXPECTED_KEYS = {
            ShardingStrategy.HYBRID_SHARD: (
                [],
                [
                    "0._fsdp_wrapped_module.linear_layers.0.weight",
                    "0._fsdp_wrapped_module.linear_layers.1.weight",
                ],
                ["1.weight"],
            ),
            ShardingStrategy._HYBRID_SHARD_ZERO2: (
                [],
                [
                    "0._fsdp_wrapped_module.linear_layers.0.weight",
                    "0._fsdp_wrapped_module.linear_layers.1.weight",
                ],
                ["1.weight"],
            ),
        }
        SHARDING_STRATEGIES_TO_EXPECTED_KEYS = {
            ShardingStrategy.NO_SHARD: (
                [],
                [],
                [
                    "0._fsdp_wrapped_module.linear_layers.0.weight",
                    "0._fsdp_wrapped_module.linear_layers.1.weight",
                    "1.weight",
                ],
            ),
            ShardingStrategy.SHARD_GRAD_OP: (
                [
                    "0._fsdp_wrapped_module.linear_layers.0.weight",
                    "0._fsdp_wrapped_module.linear_layers.1.weight",
                ],
                [],
                ["1.weight"],
            ),
            ShardingStrategy.FULL_SHARD: (
                [
                    "0._fsdp_wrapped_module.linear_layers.0.weight",
                    "0._fsdp_wrapped_module.linear_layers.1.weight",
                ],
                [],
                ["1.weight"],
            ),
        } | HYBRID_SHARDING_STRATEGIES_TO_EXPECTED_KEYS

        for sharding_strategy, (
            expected_fsdp_keys,
            expected_hsdp_keys,
            expected_other_keys,
        ) in SHARDING_STRATEGIES_TO_EXPECTED_KEYS.items():
            with self.subTest(sharding_strategy=sharding_strategy):
                fsdp_module = FSDP(
                    _create_model_and_params()[0],
                    sharding_strategy=sharding_strategy,
                    device_mesh=(
                        init_device_mesh("cuda", (2, 2))
                        if sharding_strategy
                        in HYBRID_SHARDING_STRATEGIES_TO_EXPECTED_KEYS
                        else None
                    ),
                    use_orig_params=True,
                )

                model = nn.Sequential(
                    fsdp_module,
                    nn.Linear(3, 2, bias=False),
                )
                model[1].weight.data.fill_(3.0)

                fsdp_parameter_metadata = compile_fsdp_parameter_metadata(model)
                named_params = dict(model.named_parameters())
                actual_fsdp_params, actual_hsdp_params, actual_other_params = (
                    parse_fsdp_params(
                        named_params,
                        fsdp_parameter_metadata,
                    )
                )

                actual_fsdp_keys = list(actual_fsdp_params.keys())
                actual_hsdp_keys = list(actual_hsdp_params.keys())
                actual_other_keys = list(actual_other_params.keys())

                self.assertEqual(actual_fsdp_keys, expected_fsdp_keys)
                self.assertEqual(actual_hsdp_keys, expected_hsdp_keys)
                self.assertEqual(actual_other_keys, expected_other_keys)

    @skip_if_lt_x_gpu(4)
    def test_parse_fully_shard_params(self) -> None:
        mesh_1d = init_device_mesh("cuda", (self.world_size,))
        fully_shard_module, _ = _create_model_and_params((16, 8, 1))
        fully_shard(fully_shard_module, mesh=mesh_1d)
        mesh_2d = init_device_mesh(
            "cuda",
            (2, self.world_size // 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        hybrid_shard_module, _ = _create_model_and_params((16, 8, 1))
        fully_shard(hybrid_shard_module, mesh=mesh_2d)

        model = nn.Sequential(
            fully_shard_module, hybrid_shard_module, nn.Linear(3, 2, bias=False)
        )

        fully_shard_params = {f"0.{k}": v for k, v in model[0].named_parameters()}
        hybrid_shard_params = {f"1.{k}": v for k, v in model[1].named_parameters()}
        other_params = {f"2.{k}": v for k, v in model[2].named_parameters()}
        parsed_fully_shard_params, parsed_hybrid_shard_params, parsed_other_params = (
            parse_fully_shard_params(dict(model.named_parameters()))
        )

        self.assertEqual(fully_shard_params, parsed_fully_shard_params)
        self.assertEqual(hybrid_shard_params, parsed_hybrid_shard_params)
        self.assertEqual(other_params, parsed_other_params)
