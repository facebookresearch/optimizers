"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from collections.abc import Iterable

import torch

from distributed_shampoo.shampoo_types import PARAMS
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_distributor import Distributor
from torch import distributed as dist, Tensor
from torch.distributed.tensor import DTensor


class FullyShardDistributor(Distributor):
    """FullyShard Distributor class.

    Handles merging and blocking of the tensor blocks at instantiation, and the gradients at each iteration.
    Note that parameters for module wrapped by `fully_shard` are represented as DTensors, sharded at dim-0:
    https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor.
    No communication is performed in FullyShard Distributor.

    """

    @torch.no_grad()
    def _get_params_or_grads(self, get_grad: bool = False) -> Iterable[Tensor | None]:
        """Helper function to get the local params (or grad) from the param_group, where params are represented as DTensors.

        Args:
            get_grad (bool): Whether to return the param or the grad of the param.
        Returns:
            local (Iterable[Tensor | None]): Local params (or grad) from the param_group.
        """
        # If a parameter is in a "dead layer", it won't have any gradient. In this case, we
        # should return `None` for the gradient.
        return (
            (None if p.grad is None else p.grad.to_local()) if get_grad else local_p
            for p in self._param_group[PARAMS]
            if (local_p := p.to_local()).numel() > 0
        )

    @torch.no_grad()
    def _construct_global_block_info_list(
        self,
    ) -> None:
        """Construct global block info list from param_group and num_blocks_within_param."""
        rank = dist.get_rank()

        # Call `super()` instead of `self` as a performance optimization.
        # This leads to O(1) instead of O(N) complexity to retrieve the parameters.
        non_empty_params: Iterable[DTensor] = filter(
            lambda p: p.to_local().numel() > 0,  # type: ignore[arg-type]
            super()._get_params_or_grads(),
        )
        self._global_block_info_list = tuple(
            BlockInfo(
                param=param,
                composable_block_ids=(param_index, f"rank_{rank}-block_{block_index}"),
            )
            # Block index that is accumulated across all parameters within a parameter group.
            for ((param_index, param), num_blocks_within_param) in zip(
                enumerate(non_empty_params),
                self._global_num_blocks_per_param,
                strict=True,
            )
            for block_index in range(num_blocks_within_param)
        )
