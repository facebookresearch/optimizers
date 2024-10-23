"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Callable
from typing import Dict, Tuple

import torch
from distributed_shampoo.shampoo_types import FSDPParameterMetadata

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.tensor import DTensor
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
        sharding_strategy = fsdp_module.sharding_strategy

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
                sharding_strategy=sharding_strategy,
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


def _partition_params(
    named_params: Dict[str, Parameter],
    fsdp_criteria: Callable[[Parameter], bool],
    hsdp_criteria: Callable[[Parameter], bool],
    other_criteria: Callable[[Parameter], bool],
) -> Tuple[Dict[str, Parameter], Dict[str, Parameter], Dict[str, Parameter]]:
    """Partitions parameters into FSDP, HSDP, and the rest of parameters.

    NOTE: The output dictionaries are guaranteed to be the partitions of the input `named_params`.

    Args:
        named_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding parameter.
        fsdp_criteria (Callable[[Parameter], bool]): Criteria for determining if a parameter is FSDP.
        hsdp_criteria (Callable[[Parameter], bool]): Criteria for determining if a parameter is HSDP.
        other_criteria (Callable[[Parameter], bool]): Criteria for determining if a parameter is not FSDP or HSDP.

    Returns:
        fsdp_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding FSDP parameter.
        hsdp_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding HSDP parameter.
        other_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding non-FSDP parameter.

    """
    fsdp_params = {
        fqn: param for fqn, param in named_params.items() if fsdp_criteria(param)
    }
    hsdp_params = {
        fqn: param for fqn, param in named_params.items() if hsdp_criteria(param)
    }
    other_params = {
        fqn: param for fqn, param in named_params.items() if other_criteria(param)
    }

    assert (
        (unioned_keys := fsdp_params.keys() | hsdp_params.keys() | other_params.keys())
        == named_params.keys()
    ), f"{unioned_keys - named_params.keys()=} {named_params.keys() - unioned_keys=}"
    assert not (
        fsdp_and_other := fsdp_params.keys() & other_params.keys()
    ), f"{fsdp_and_other} exist in both fsdp_params and other_params!"
    assert not (
        hsdp_and_other := hsdp_params.keys() & other_params.keys()
    ), f"{hsdp_and_other} exist in both hsdp_params and other_params!"
    assert not (
        fsdp_and_hsdp := fsdp_params.keys() & hsdp_params.keys()
    ), f"{fsdp_and_hsdp} exist in both fsdp_params and hsdp_params!"

    return fsdp_params, hsdp_params, other_params


def parse_fsdp_params(
    named_params: Dict[str, Parameter],
    param_metadata: Dict[Parameter, FSDPParameterMetadata],
) -> Tuple[Dict[str, Parameter], Dict[str, Parameter], Dict[str, Parameter]]:
    """Splits parameters into FSDP, HSDP, and the rest of parameters.

    This is useful for parsing the parameters when FSDP and HSDP are wrapping a subset of modules within a model.

    NOTE: The output dictionaries are guaranteed to be the partitions of the input `named_params`.

    Args:
        named_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding parameter.
        param_metadata (Dict[Parameter, FSDPParameterMetadata]): Dictionary mapping each parameter to its FSDP metadata.

    Returns:
        fsdp_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding FSDP parameter.
        hsdp_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding HSDP parameter.
        other_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding non-FSDP parameter.

    """
    return _partition_params(
        named_params=named_params,
        fsdp_criteria=lambda param: param in param_metadata
        and param_metadata[param].sharding_strategy
        in [ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP],
        hsdp_criteria=lambda param: param in param_metadata
        and param_metadata[param].sharding_strategy
        in [ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2],
        other_criteria=lambda param: param not in param_metadata
        or param_metadata[param].sharding_strategy == ShardingStrategy.NO_SHARD,
    )


def parse_fully_shard_params(
    named_params: Dict[str, Parameter],
) -> Tuple[Dict[str, Parameter], Dict[str, Parameter], Dict[str, Parameter]]:
    """Splits parameters into fully shard(per parameter FSDP), hybrid shard parameters(per parameter FSDP) and the rest of parameters.

    This is useful for parsing the parameters when fully shard or hybrid shard are wrapping a subset of modules within a model.

    NOTE: The output dictionaries are guaranteed to be the partitions of the input `named_params`.

    Args:
        named_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding parameter.

    Returns:
        fully_shard_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding fully shard parameter.
        hybrid_shard_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding hybrid shard parameter.
        other_params (Dict[str, Parameter]): Dictionary mapping each parameter name to its corresponding non fully shard parameter.
    """
    return _partition_params(
        named_params=named_params,
        fsdp_criteria=lambda param: isinstance(param, DTensor)
        and len(param.device_mesh.shape) == 1,
        hsdp_criteria=lambda param: isinstance(param, DTensor)
        and len(param.device_mesh.shape) == 2,
        other_criteria=lambda param: not isinstance(param, DTensor),
    )
