"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Iterable
from typing import Any, Literal, overload

import torch
from distributed_shampoo.distributor.shampoo_block_info import BlockInfo
from distributed_shampoo.distributor.shampoo_distributor import Distributor

from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    PARAMS,
)
from torch import distributed as dist, Tensor

logger: logging.Logger = logging.getLogger(__name__)


class FullyShardLosslessDistributor(Distributor):
    """FullyShard Lossless Distributor class.

    On top of FullyShardDistributor, this distributor handles the parameter assignment to exchange the gradients
    and parameter updates across the shards to achieve lossless numerical results comapred to default Shampoo.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.

    """

    def __init__(self, param_group: dict[str, Any]) -> None:
        distributed_config: FullyShardDistributedConfig = param_group[
            DISTRIBUTED_CONFIG
        ]
        self._param_assignment_strategy: FSDPParamAssignmentStrategy = (
            distributed_config.param_assignment_strategy
        )
        logger.info(
            f"Shampoo FullyShardLosslessDistributor {self._param_assignment_strategy=}",
        )
        # Stores full parameters (as opposed to DTensors) for the model parameters assigned to this rank.
        # For example, when the strategy is REPLICATE, it stores the full parameters on all ranks.
        # Note that we explicitly disable the unnecessary gradient tracking for the all-gather collectives
        # used to initialize the full parameters.
        with torch.no_grad():
            self._assigned_full_params: tuple[torch.Tensor, ...] = tuple(
                p.full_tensor() for p in param_group[PARAMS]
            )

        super().__init__(param_group)

    @overload
    @torch.no_grad()
    def _get_params_or_grads(
        self, get_grad: Literal[True]
    ) -> Iterable[Tensor | None]: ...

    @overload
    @torch.no_grad()
    def _get_params_or_grads(
        self, get_grad: Literal[False] = False
    ) -> Iterable[Tensor]: ...

    @torch.no_grad()
    def _get_params_or_grads(self, get_grad: bool = False) -> Iterable[Tensor | None]:
        """Helper function to get the assigned full params (or gradients) from the param_group.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param.
        Returns:
            local (Iterable[Tensor | None]): assigned full params (or grad) from the param_group.
        """

        if get_grad:
            # Getting grads at every optimizer step triggers implicit all-gather. Note that p.numel()
            # returns total number of elements in the tensor (as opposed to local shard of DTensor).
            return (
                None if p.grad is None else p.grad.full_tensor()
                for p in self._param_group[PARAMS]
                if p.numel() > 0
            )
        else:
            return filter(
                lambda p: isinstance(p, Tensor) and p.numel() > 0,
                self._assigned_full_params,
            )

    @torch.no_grad()
    def update_params(
        self,
        masked_blocked_search_directions: tuple[Tensor, ...],
    ) -> None:
        """Update params stored inside this distributor according to the input search directions argument.

        Args:
            masked_blocked_search_directions (tuple[Tensor, ...]): Search directions for each local blocked parameter.
            This tuple might be empty if the parameters are not receiving gradients.

        """
        super().update_params(masked_blocked_search_directions)

        # Copy the updated full parameters to the original parameters in the param group.
        # For example, when the strategy is REPLICATE, we need to take each updated full parameter `full_param`,
        # redistribute it according to the device mesh to get the locally assigned slice, and copy the slice to the
        # corresponding local parameter `local_param` in the param group.
        if self._param_assignment_strategy == FSDPParamAssignmentStrategy.REPLICATE:
            local_params = list(
                filter(lambda p: p.numel() > 0, self._param_group[PARAMS])
            )
            full_param_slices = [
                # When param assignment strategy is REPLICATE, explicitly set `src_data_rank` to None to avoid
                # triggering communication and simply use the local copy of replicated parameters.
                dist.tensor.distribute_tensor(
                    full_param,
                    local_param.device_mesh,
                    local_param.placements,
                    src_data_rank=None,
                ).to_local()
                for local_param, full_param in zip(
                    local_params,
                    self._get_params_or_grads(),
                    strict=True,
                )
            ]
            # torch._foreach_copy_ requires both lists of tensors to be local tensors.
            torch._foreach_copy_(
                [p.to_local() for p in local_params], full_param_slices
            )

    @torch.no_grad()
    def _construct_local_block_info_list(self) -> tuple[BlockInfo, ...]:
        """Construct local block info list from param_group."""
        return self._construct_local_block_info_list_with_params(
            params=filter(lambda p: p.numel() > 0, self._param_group[PARAMS])
        )
