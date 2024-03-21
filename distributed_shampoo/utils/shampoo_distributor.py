"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from distributed_shampoo.shampoo_types import (
    MAX_PRECONDITIONER_DIM,
    PARAMS,
    USE_MERGE_DIMS,
)
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    generate_pairwise_indices,
    merge_small_dims,
    multi_dim_split,
)
from torch import Tensor


###### DISTRIBUTOR CLASSES ######
class DistributorInterface(ABC):
    """Distributor interface.

    Functionally specifies the API for Distributor classes.

    Args:
        param_group (Dict[str, Any]): Parameter group containing parameters.

    """

    def __init__(self, param_group: Dict[str, Any]) -> None:
        self._param_group = param_group
        # Merge and block parameters creates self._global_blocked_params, self._global_num_blocks_per_param,
        # and self._global_merged_dims_list.
        # Global blocked params are all the blocked parameters after merging and blocking.
        # Global num blocks per param stores the number of blocks for each global parameter.
        # Global merged dims list stores the merged dimensions for each global parameter.
        self._merge_and_block_parameters()
        # Global grad selector masks all global gradients that are None.
        self._global_grad_selector: Tuple[bool, ...] = (True,) * len(
            self._global_blocked_params
        )
        # In order to avoid redundant computation, we store the previous global grad selector.
        self._previous_global_grad_selector: Optional[Tuple[bool, ...]] = None

        # Declare properties that will be populated by subclasses.
        # Distributor selector masks all global parameter blocks that are NOT assigned to the local device.
        self._distributor_selector: Tuple[bool, ...]
        # Local grad selector masks all local gradients (i.e., already masked by distributor selector) that are None.
        self._local_grad_selector: Tuple[bool, ...]
        # Local blocked params are the parameters masked by the distributor selector.
        self._local_blocked_params: Tuple[Tensor, ...]
        # Local masked blocked params are the parameters masked by the distributor selector AND the local grad selector.
        self._local_masked_blocked_params: Tuple[Tensor, ...]
        # Global block info list contains information about each global block.
        self._global_block_info_list: Tuple[BlockInfo, ...]

    @abstractmethod
    @torch.no_grad()
    def update_params(
        self,
        masked_blocked_search_directions: Tuple[Tensor, ...],
    ) -> None: ...

    @property
    def global_blocked_params(self) -> Tuple[Tensor, ...]:
        return self._global_blocked_params

    @property
    def distributor_selector(self) -> Tuple[bool, ...]:
        return self._distributor_selector

    @property
    def local_grad_selector(self) -> Tuple[bool, ...]:
        return self._local_grad_selector

    @property
    def local_blocked_params(self) -> Tuple[Tensor, ...]:
        return self._local_blocked_params

    @property
    def local_masked_blocked_params(self) -> Tuple[Tensor, ...]:
        return self._local_masked_blocked_params

    @property
    def global_block_info_list(self) -> Tuple[BlockInfo, ...]:
        return self._global_block_info_list

    def _merge_and_block_parameters(
        self,
    ) -> None:
        """Merge and block parameters.

        NOTE: FSDP may modify this function.

        """

        # Merge dimensions for each parameter.
        self._global_merged_dims_list: Tuple[Tuple[int, ...], ...] = tuple(
            (
                merge_small_dims(
                    param.size(), self._param_group[MAX_PRECONDITIONER_DIM]
                )
                if self._param_group[USE_MERGE_DIMS]
                else param.size()
            )
            for param in self._param_group[PARAMS]
        )

        # Generate blocked parameters list and number of blocks per parameter.
        global_blocked_params = []
        global_num_blocks_per_param = []

        for param, merged_dims in zip(
            self._param_group[PARAMS], self._global_merged_dims_list
        ):
            # Obtain blocks for each parameter after merging.
            blocks_within_param = multi_dim_split(
                param.view(merged_dims), self._param_group[MAX_PRECONDITIONER_DIM]
            )

            # Generate and extend blocked parameters list.
            global_blocked_params.extend(
                # Note: We are using tensor.detach() here to explicitly set block_param (a view of the original
                # parameter) to requires_grad = False in order to prevent errors with print and PT2 compile.
                # Remove this tensor.detach() once https://github.com/pytorch/pytorch/issues/113793 is fixed.
                block_param.detach()
                for block_param in blocks_within_param
            )
            global_num_blocks_per_param.append(len(blocks_within_param))

        self._global_blocked_params: Tuple[Tensor, ...] = tuple(global_blocked_params)
        self._global_num_blocks_per_param: Tuple[int, ...] = tuple(
            global_num_blocks_per_param
        )

    @abstractmethod
    def merge_and_block_gradients(
        self,
    ) -> Tuple[Tensor, ...]: ...

    def _merge_and_block_gradients(
        self,
    ) -> Tuple[Tensor, ...]:
        """Merge and block gradients.

        NOTE: FSDP Distributor may modify this function.

        Returns:
            local_masked_blocked_grads (Tuple[Tensor, ...]): Local gradients with grad not None.

        """

        local_masked_blocked_grads = []
        global_grad_selector = []

        for param, merged_dims, num_blocks, (block_index, next_block_index) in zip(
            self._param_group[PARAMS],
            self._global_merged_dims_list,
            self._global_num_blocks_per_param,
            generate_pairwise_indices(self._global_num_blocks_per_param),
            strict=True,
        ):
            grad = param.grad
            param_distributor_selector = self._distributor_selector[
                block_index:next_block_index
            ]

            # Update the selector
            global_grad_selector.extend([grad is not None] * num_blocks)

            if grad is None or not any(param_distributor_selector):
                # Skip multi_dim_split if this blocked grad will not be used locally.
                continue

            # Obtain blocks for each gradient after merging.
            blocks_within_grad = multi_dim_split(
                grad.view(merged_dims), self._param_group[MAX_PRECONDITIONER_DIM]
            )
            # Generate block-to-parameter metadata and extend blocked parameters list.
            local_masked_blocked_grads.extend(
                compress_list(blocks_within_grad, param_distributor_selector)
            )

        # Set global grad selector as tuple.
        self._global_grad_selector = tuple(global_grad_selector)

        return tuple(local_masked_blocked_grads)


class Distributor(DistributorInterface):
    """Default Distributor class.

    Handles merging and blocking of the parameters at instantiation, and the gradients
    at each iteration. Note that no communication is performed since it assumes only
    single-GPU training.

    Args:
        param_group (Dict[str, Any]): Parameter group containing parameters.
    """

    def __init__(
        self,
        param_group: Dict[str, Any],
    ) -> None:
        super().__init__(param_group)
        self._construct_global_block_info_list()

        # Initialize selectors and local blocked (masked) parameters.
        self._local_grad_selector: Tuple[bool, ...] = (True,) * len(
            self._global_blocked_params
        )
        self._distributor_selector: Tuple[bool, ...] = self._local_grad_selector
        self._local_masked_blocked_params: Tuple[Tensor, ...] = (
            self._global_blocked_params
        )
        self._local_blocked_params: Tuple[Tensor, ...] = (
            self._local_masked_blocked_params
        )

    @torch.no_grad()
    def update_params(
        self,
        masked_blocked_search_directions: Tuple[Tensor, ...],
    ) -> None:
        """Update params stored inside this distributor according to the input search directions argument.

        Args:
            masked_blocked_search_directions (Tuple[Tensor, ...]): Search directions for each local blocked parameter.

        """
        torch._foreach_add_(
            self._local_masked_blocked_params,
            masked_blocked_search_directions,
        )

    def _construct_global_block_info_list(
        self,
    ) -> None:
        """Construct global block info list from param_group and num_blocks_within_param."""
        self._global_block_info_list = tuple(
            BlockInfo(
                param=param,
                composable_block_ids=(param_index, f"block_{block_index}"),
            )
            # Block index that is accumulated across all parameters within a parameter group.
            for ((param_index, param), num_blocks_within_param) in zip(
                enumerate(self._param_group[PARAMS]),
                self._global_num_blocks_per_param,
                strict=True,
            )
            for block_index in range(num_blocks_within_param)
        )

    def merge_and_block_gradients(
        self,
    ) -> Tuple[Tensor, ...]:
        """Merge and block gradients.

        NOTE: This function MUST be called in the step function of the optimizer after the
        gradient has been updated.

        Returns:
            local_masked_blocked_grads (Tuple[Tensor, ...]): Local blocked gradients masked with grad existence.

        """
        local_masked_blocked_grads = self._merge_and_block_gradients()

        if self._previous_global_grad_selector != self._global_grad_selector:
            self._previous_global_grad_selector = self._global_grad_selector

            # Update _local_grad_selector and _local_masked_blocked_params only when global_grad_selector is changed.
            self._local_grad_selector = compress_list(
                self._global_grad_selector,
                self._distributor_selector,
            )
            self._local_masked_blocked_params = compress_list(
                self._local_blocked_params, self._local_grad_selector
            )

        return local_masked_blocked_grads
