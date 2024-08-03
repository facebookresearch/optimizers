# pyre-strict
import logging
from math import prod
from typing import Any, Dict, List, Tuple

import torch
from distributed_shampoo.shampoo_types import (
    FSDPParameterMetadata,
    FSDPShampooConfig,
    LR,
    MAX_PRECONDITIONER_DIM,
    PARAMS,
    USE_MERGE_DIMS,
)
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_distributor import (
    DistributorInterface,
)
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    generate_pairwise_indices,
    merge_small_dims,
    multi_dim_split,
)
from torch import distributed as dist, Tensor
from torch.nn import Parameter


class FSDP2Distributor(DistributorInterface):
    """FSDP Distributor class.

    Handles split tensor block recovery of different parameters, then merging and blocking of
    the tensor blocks at instantiation, and the gradients at each iteration.
    Note that no communication is performed in FSDP Distributor.

    Args:
        param_group (Dict[str, Any]): Parameter group containing parameters.
        distributed_config (FSDPShampooConfig): Configuration for FSDP Shampoo.

    """

    def __init__(
        self,
        param_group: Dict[str, Any],
        distributed_config: FSDPShampooConfig,
    ) -> None:
        self._param_to_metadata: Dict[Parameter, FSDPParameterMetadata] = (
            distributed_config.param_to_metadata
        )

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
        self._local_blocked_params: Tuple[Tensor, ...] = self._global_blocked_params

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
        rank = dist.get_rank()
        local_params = [local for p in self._param_group[PARAMS] if (local := p.to_local()).numel() > 0]
        self._global_block_info_list = tuple(
            BlockInfo(
                param=param,
                composable_block_ids=(param_index, f"rank_{rank}-block_{block_index}"),
            )
            # Block index that is accumulated across all parameters within a parameter group.
            for ((param_index, param), num_blocks_within_param) in zip(
                enumerate(local_params),
                self._global_num_blocks_per_param,
                strict=True,
            )
            for block_index in range(num_blocks_within_param)
        )

    def _merge_and_block_parameters(
        self,
    ) -> None:
        """Split, merge, and block parameters."""
        # Generate blocked parameters list and number of blocks per parameter.
        global_blocked_params = []
        global_num_blocks_per_param = []

        # Convert params from DTensor to local tensor
        # local_params = [p.to_local() for p in self._param_group[PARAMS]]
        local_params = [local for p in self._param_group[PARAMS] if (local := p.to_local()).numel() > 0]

        # Merge dimensions for each parameter.
        self._global_merged_dims_list: Tuple[Tuple[int, ...], ...] = tuple(
            (
                merge_small_dims(
                    param.size(), self._param_group[MAX_PRECONDITIONER_DIM]
                )
                if self._param_group[USE_MERGE_DIMS]
                else param.size()
            )
            for param in local_params
        )

        # ====================== czhuge logging: ======================
        shape_str = "["
        local_shape_str = "["
        for param in self._param_group[PARAMS]:
            shape_str += str(param.shape) + " type: " + str(type(param)) + ", "
        shape_str += "]"

        for local_param in local_params:
            local_shape_str += (
                str(local_param.shape) + " type: " + str(type(local_param)) + " numel: " + str(local_param.numel()) + ", "
            )
        local_shape_str += "]"
        logging.info(
            "czhuge: GPU:%s, _merge_and_block_parameters:\nshape_str: %s\nlocal_shape_str: %s",
            torch.cuda.current_device(),
            shape_str,
            local_shape_str,
        )
        # ====================== czhuge logging: ======================

        for param, merged_dims in zip(local_params, self._global_merged_dims_list):
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

        logging.info(
            "czhuge: GPU:%s, global_num_blocks_per_param: %s",
            torch.cuda.current_device(),
            global_num_blocks_per_param,
        )

        # Set lists as tuples.
        self._global_blocked_params: Tuple[Tensor, ...] = tuple(global_blocked_params)
        self._global_num_blocks_per_param: Tuple[int, ...] = tuple(
            global_num_blocks_per_param
        )

    def _merge_and_block_gradients(
        self,
    ) -> Tuple[Tensor, ...]:
        """Split, merge, and block gradients.

        Returns:
            local_masked_blocked_grads (Tuple[Tensor, ...]): Local gradients with grad not None.

        """
        local_masked_blocked_grads = []
        global_grad_selector = []

        # local_grads = [p.grad.to_local() for p in self._param_group[PARAMS]]
        local_grads = [local for p in self._param_group[PARAMS] if (local := p.grad.to_local()).numel() > 0]

        for grad, merged_dims, num_blocks, (block_index, next_block_index) in zip(
            local_grads,
            self._global_merged_dims_list,
            self._global_num_blocks_per_param,
            generate_pairwise_indices(self._global_num_blocks_per_param),
        ):
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

        # # ====================== czhuge logging: ======================
        # shape_str = "["
        # local_shape_str = "["
        # for param in self._param_group[PARAMS]:
        #     shape_str += str(param.shape) + " type: " + str(type(param)) + ", "
        # shape_str += "]"

        # local_shape_str = "["
        # for lg in local_grads:
        #     local_shape_str += str(lg.shape) + " type: " + str(type(lg)) + ", "
        # local_shape_str += "]"

        # local_grad_shape_str = "["
        # for grad in local_masked_blocked_grads:
        #     local_grad_shape_str += str(grad.shape) + " type: " + str(type(grad)) + ", "
        # local_grad_shape_str += "]"
        # logging.info(
        #     "czhuge: GPU:%s, _merge_and_block_gradients: len params: %s, shape_str: %s,\nlen local_params: %s, local_shape_str: %s,\n"
        #     "len local_masked_blocked_grads: %s, local_grad_shape_str: %s",
        #     torch.cuda.current_device(),
        #     len(self._param_group[PARAMS]),
        #     shape_str,
        #     len(local_grads),
        #     local_shape_str,
        #     len(local_masked_blocked_grads),
        #     local_grad_shape_str,
        # )
        # # ====================== czhuge logging: ======================

        return tuple(local_masked_blocked_grads)

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
