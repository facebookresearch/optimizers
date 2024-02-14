"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from typing import List, Tuple

import torch
from distributed_shampoo.shampoo_types import FSDPParameterMetadata

from distributed_shampoo.utils.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
    parse_fsdp_params,
)
from torch import distributed as dist, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parameter import Parameter
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest


# Note: Ideally this function should be resided inside Test as part of setUp() but FSDPTest
#       only calls setUp() on one device; as a result, every device has to call this function
#       separately.
def _create_model_and_params() -> Tuple[nn.Module, List[Parameter]]:
    model = nn.Sequential(
        nn.Linear(2, 5, bias=False),
        nn.Linear(5, 3, bias=False),
    )
    model[0].weight.data.fill_(1.0)
    model[1].weight.data.fill_(2.0)
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
                    fqn="0.weight",
                    shape=torch.Size([5, 2]),
                    numel=10,
                    start_idx=0,
                    end_idx=10,
                ),
                params[1]: FSDPParameterMetadata(
                    fqn="1.weight",
                    shape=torch.Size([3, 5]),
                    numel=15,
                    start_idx=0,
                    end_idx=2,
                ),
            }
            if dist.get_rank() == 0
            else {
                params[0]: FSDPParameterMetadata(
                    fqn="0.weight",
                    shape=torch.Size([5, 2]),
                    numel=10,
                    start_idx=0,
                    end_idx=0,
                ),
                params[1]: FSDPParameterMetadata(
                    fqn="1.weight",
                    shape=torch.Size([3, 5]),
                    numel=15,
                    start_idx=2,
                    end_idx=15,
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
        return 2

    @skip_if_lt_x_gpu(2)
    def test_parse_fsdp_params(self) -> None:
        fsdp_module = FSDP(
            _create_model_and_params()[0],
            use_orig_params=True,
        )

        model = nn.Sequential(
            fsdp_module,
            nn.Linear(3, 2, bias=False),
        )
        model[1].weight.data.fill_(3.0)

        fsdp_parameter_metadata = compile_fsdp_parameter_metadata(model)
        named_params = dict(model.named_parameters())
        actual_fsdp_params, actual_other_params = parse_fsdp_params(
            named_params,
            fsdp_parameter_metadata,
        )

        actual_fsdp_keys = list(actual_fsdp_params.keys())
        actual_other_keys = list(actual_other_params.keys())

        expected_fsdp_keys = [
            "0._fsdp_wrapped_module.0.weight",
            "0._fsdp_wrapped_module.1.weight",
        ]
        expected_other_keys = ["1.weight"]

        self.assertEqual(actual_fsdp_keys, expected_fsdp_keys)
        self.assertEqual(actual_other_keys, expected_other_keys)
