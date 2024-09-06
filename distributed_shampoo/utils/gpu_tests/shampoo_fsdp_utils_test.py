"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from typing import List, Tuple

import torch
from distributed_shampoo.shampoo_types import FSDPParameterMetadata

from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from distributed_shampoo.utils.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
    parse_fsdp_params,
)
from torch import distributed as dist, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.nn.parameter import Parameter
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest


# Note: Ideally this function should be resided inside Test as part of setUp() but FSDPTest
#       only calls setUp() on one device; as a result, every device has to call this function
#       separately.
def _create_model_and_params() -> Tuple[nn.Module, List[Parameter]]:
    model, _, _, _ = construct_training_problem(
        (2, 5, 3), model_dead_layer_dims=None, fill=(1.0, 2.0)
    )
    return model, list(model.parameters())


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
