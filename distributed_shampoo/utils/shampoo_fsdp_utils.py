"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from typing import Dict, Tuple

import torch
from distributed_shampoo.shampoo_types import FSDPParameterMetadata

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Parameter


def compile_fsdp_parameter_metadata(
    module: torch.nn.Module,
) -> Dict[Parameter, FSDPParameterMetadata]:
    """Compiles parameter metadata necessary for FSDP Shampoo.

    Args:
        module (nn.Module): Module to compile metadata for.

    Returns:
        param_metadata (Dict[Parameter, FSDPParameterMetadata]): Dictionary mapping each parameter to its FSDP metadata.

    """
    param_metadata: Dict[Parameter, FSDPParameterMetadata] = {}

    for fsdp_module in FSDP.fsdp_modules(module):
        if (flat_param := fsdp_module._flat_param) is None:
            continue

        fqns = flat_param._fqns
        shapes = flat_param._shapes
        numels = flat_param._numels
        shard_param_infos = flat_param._shard_param_infos
        assert (
            flat_param._params is not None
        ), "flat_param._params should not be None! Set the value of `use_orig_params` in FSDP module to True "
        "would populate flat_param._params."
        params = flat_param._params

        param_metadata |= {
            param: FSDPParameterMetadata(
                fqn=fqn,
                shape=shape,
                numel=numel,
                start_idx=shard_param_info.intra_param_start_idx or 0,
                end_idx=(
                    shard_param_info.intra_param_end_idx + 1
                    if shard_param_info.intra_param_end_idx is not None
                    else 0
                ),
            )
            for param, fqn, shape, numel, shard_param_info in zip(
                params,
                fqns,
                shapes,
                numels,
                shard_param_infos,
                strict=True,
            )
        }

    return param_metadata


def parse_fsdp_params(
    named_params: Dict[str, Parameter],
    param_metadata: Dict[Parameter, FSDPParameterMetadata],
) -> Tuple[Dict[str, Parameter], Dict[str, Parameter]]:
    """Splits parameters into FSDP and non-FSDP parameters.

    This is useful for parsing the parameters when FSDP is only wrapping a subset of modules within a model.

    Args:
        named_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding parameter.
        param_metadata (Dict[Parameter, FSDPParameterMetadata]): Dictionary mapping each parameter to its FSDP metadata.

    Returns:
        fsdp_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding FSDP parameter.
        other_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding non-FSDP parameter.

    """
    fsdp_params = {
        fqn: param for fqn, param in named_params.items() if param in param_metadata
    }
    other_params = {
        fqn: param for fqn, param in named_params.items() if param not in param_metadata
    }

    return fsdp_params, other_params
