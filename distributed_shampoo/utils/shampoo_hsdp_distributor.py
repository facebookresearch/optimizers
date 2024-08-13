"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import heapq
import logging
from math import prod
from typing import Any, Dict, List, Tuple

import torch
from distributed_shampoo.shampoo_types import (
    CommunicationDType,
    FSDPParameterMetadata,
    HSDPShampooConfig,
    MAX_PRECONDITIONER_DIM,
    PARAMS,
    USE_MERGE_DIMS,
)
from distributed_shampoo.utils.shampoo_block_info import DDPBlockInfo
from distributed_shampoo.utils.shampoo_dist_utils import get_device_mesh
from distributed_shampoo.utils.shampoo_distributor import DistributorInterface
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    generate_pairwise_indices,
    get_dtype_size,
    merge_small_dims,
    multi_dim_split,
)
from torch import distributed as dist, Tensor
from torch.distributed import _tensor as dtensor
from torch.distributed._tensor import zeros as dtensor_zeros
from torch.nn import Parameter

logger: logging.Logger = logging.getLogger(__name__)


class HSDPDistributor(DistributorInterface):
    """HSDP Distributor class.

    Handles split tensor block recovery of different parameters, then merging and blocking of
    the tensor blocks, as well as distributing of the parameters at instantiation.
    The gradients are also recovered at each iteration.

    The constructor internally sets up process groups as well, so torch.distributed must be
    initialized in advance.

    Args:
        param_group (Dict[str, Any]): Parameter group containing parameters.
        distributed_config (HSDPShampooConfig): Configuration for HSDP Shampoo.

    """

    def __init__(
        self,
        param_group: Dict[str, Any],
        distributed_config: HSDPShampooConfig,
    ) -> None:
        self._param_to_metadata: Dict[Parameter, FSDPParameterMetadata] = (
            distributed_config.param_to_metadata
        )
        self._hsdp_device_mesh: torch.distributed.device_mesh.DeviceMesh = (
            distributed_config.device_mesh
        )
        self._global_num_splits_per_param: Tuple[int, ...] = ()
        self._global_num_blocks_per_split_param: Tuple[int, ...] = ()

        super().__init__(param_group)
        if not dist.is_initialized():
            raise RuntimeError(
                "HSDPDistributor needs torch.distributed to be initialized!"
            )

        # Construct global masked blocked parameters (which is DDP-specific).
        self._global_masked_blocked_params: Tuple[Tensor, ...] = (
            self._global_blocked_params
        )

        # Check num_trainers_per_group and replicated group size.
        # NOTE: If num_trainers_per_group = -1, then we use the replicated group size.
        self._replicated_group_size: int = self._hsdp_device_mesh.size(0)

        if not (
            1
            <= distributed_config.num_trainers_per_group
            <= self._replicated_group_size
            or distributed_config.num_trainers_per_group == -1
        ):
            raise ValueError(
                f"Invalid number of trainers per group: {distributed_config.num_trainers_per_group}. "
                f"Must be between [1, {self._replicated_group_size}] or set to -1."
            )
        if distributed_config.num_trainers_per_group == -1:
            logger.info(
                f"Note that {distributed_config.num_trainers_per_group=}! Defaulting to replicated group size {self._replicated_group_size}."
            )
        elif (
            not self._replicated_group_size % distributed_config.num_trainers_per_group
            == 0
        ):
            raise ValueError(
                f"{distributed_config.num_trainers_per_group=} must divide {self._replicated_group_size=}!"
            )

        # Group size for distributing computation / memory requirements.
        self._dist_group_size: int = (
            distributed_config.num_trainers_per_group
            if distributed_config.num_trainers_per_group != -1
            else self._replicated_group_size
        )

        # Create flag for distributing parameters instead of search directions.
        self._communicate_params: bool = distributed_config.communicate_params

        # Determine communication type.
        if distributed_config.communication_dtype == CommunicationDType.BF16:
            self._communication_dtype: torch.dtype = torch.bfloat16
        elif distributed_config.communication_dtype == CommunicationDType.FP16:
            self._communication_dtype: torch.dtype = torch.float16
        else:
            assert distributed_config.communication_dtype in [
                CommunicationDType.FP32,
                CommunicationDType.DEFAULT,
            ]
            self._communication_dtype = torch.float32

        # Initialize _dist_group and _group_rank.
        # Note that this requires initializing all process groups.
        if self._dist_group_size == self._replicated_group_size:
            self._dist_group: dist.ProcessGroup = self._hsdp_device_mesh.get_group(0)
        else:
            # Splits replicated ranks group into smaller groups of size self._dist_group_size.
            # Instantiates this by using DeviceMesh.
            ranks_in_all_replicated_groups = self._hsdp_device_mesh.mesh.T
            for ranks_in_replicated_group in ranks_in_all_replicated_groups:
                device_mesh = get_device_mesh(
                    device_type=self._hsdp_device_mesh.device_type,
                    mesh=ranks_in_replicated_group.view(-1, self._dist_group_size),
                    mesh_dim_names=("replicate", "shard"),
                )
                if dist.get_rank() in ranks_in_replicated_group:
                    self._dist_group = device_mesh.get_group("shard")

        self._group_rank: int = dist.get_rank(self._dist_group)

        # Assign ranks to blocks with their respective buffer size.
        buffer_size_ranks = self._distribute_buffer_sizes(
            buffer_sizes=tuple(
                blocked_param.numel() * get_dtype_size(self._communication_dtype)
                for blocked_param in self._global_blocked_params
            )
        )

        self._construct_global_block_info_list(buffer_size_ranks)

        # Initialize selectors and local blocked (masked) parameters.
        self._distributor_selector: Tuple[bool, ...] = tuple(
            block_info.group_source_rank == self._group_rank
            for block_info in self._global_block_info_list
        )
        self._local_blocked_params: Tuple[Tensor, ...] = compress_list(
            self._global_blocked_params, self._distributor_selector
        )
        self._local_masked_blocked_params: Tuple[Tensor, ...] = (
            self._local_blocked_params
        )
        self._local_grad_selector: Tuple[bool, ...] = (True,) * len(
            self._local_blocked_params
        )

        self._construct_distributed_buffers(buffer_size_ranks)

    # NOTE: Remove this function once PT2 supports all_gather with functional collective
    @torch.no_grad()
    @torch.compiler.disable
    def all_gather_into_tensor(self) -> None:
        dist.all_gather_into_tensor(
            self._global_dist_buffer,
            self._local_dist_buffer,
            group=self._dist_group,
        )

    @torch.no_grad()
    def update_params(
        self,
        masked_blocked_search_directions: Tuple[Tensor, ...],
    ) -> None:
        """Update params stored inside this distributor according to the input search directions argument.

        Args:
            masked_blocked_search_directions (Tuple[Tensor, ...]): Search directions for each local blocked parameter.

        See the comment in the parent class for details.

        """
        if self._communicate_params:
            # Perform your update to your local masked parameters and copy into buffers.
            torch._foreach_add_(
                self._local_masked_blocked_params,
                masked_blocked_search_directions,
            )
            torch._foreach_copy_(
                self._local_masked_dist_blocked_buffers,
                self._local_masked_blocked_params,
            )

            self.all_gather_into_tensor()

            # Copy updated blocked params in global_masked_dist_blocked_buffers
            # into global_masked_blocked_params.
            torch._foreach_copy_(
                self._global_masked_blocked_params,
                self._global_masked_dist_blocked_buffers,
            )

        else:
            # Search directions multiplied by alpha are distributed.
            # Copy the local search directions to the communication buffer.
            torch._foreach_copy_(
                self._local_masked_dist_blocked_buffers,
                masked_blocked_search_directions,
            )

            self.all_gather_into_tensor()

            # Add search directions in global_masked_dist_blocked_buffers
            # to global_masked_blocked_params.
            torch._foreach_add_(
                self._global_masked_blocked_params,
                self._global_masked_dist_blocked_buffers,
            )

    def _distribute_buffer_sizes(
        self,
        buffer_sizes: Tuple[int, ...],
    ) -> Tuple[Tuple[int, int], ...]:
        """Distribute given buffer sizes across ranks in a group.

        Buffer sizes will be rounded up for memory allocation. Buffers are distributed such that
        total buffer sizes of each rank are as even as possible. This is currently performed
        using a greedy algorithm. We do not currently consider computational cost
        or kernel launching overheads.

        Note: A better distribution strategy should try to minimize the delta of buffer sizes
        between the most and the least allocated groups.

        Args:
            buffer_sizes (Tuple[int, ...]): Buffer sizes of blocks to be distributed.

        Returns:
            buffer_size_ranks (Tuple[Tuple[int, int], ...]): A list of tuples containing the
                buffer size for each block and its assigned rank.

        Example:
            Assuming ALIGNMENT_BYTES = 64, given buffer_sizes = [128, 64, 500, 256], group_size = 2
            -> buffer_size_ranks = [(128, 1), (64, 1), (512, 0), (256, 1)]

        """
        ALIGNMENT_BYTES = (
            64  # necessary for determining buffer size, possibly hardware-dependent
        )

        # Convert each of buffer_sizes into smallest multiple of ALIGNMENT_BYTES that is >= buffer size.
        aligned_buffer_sizes = [
            (buffer_size + ALIGNMENT_BYTES - 1) // ALIGNMENT_BYTES * ALIGNMENT_BYTES
            for buffer_size in buffer_sizes
        ]
        buffer_size_ranks = [(-1, -1)] * len(buffer_sizes)
        allocated_buffer_sizes = [
            (0, group_index) for group_index in range(self._dist_group_size)
        ]
        heapq.heapify(allocated_buffer_sizes)

        for index, aligned_buffer_size in sorted(
            enumerate(aligned_buffer_sizes),
            key=lambda t: t[1],
            reverse=True,
        ):
            # Greedily find the group with the least allocated buffer size and its group index
            # in order to allocate buffers on that group.
            (
                min_allocated_buffer_size,
                min_allocated_buffer_size_group_index,
            ) = heapq.heappop(allocated_buffer_sizes)

            heapq.heappush(
                allocated_buffer_sizes,
                (
                    min_allocated_buffer_size + aligned_buffer_size,
                    min_allocated_buffer_size_group_index,
                ),
            )
            buffer_size_ranks[index] = (
                aligned_buffer_size,
                min_allocated_buffer_size_group_index,
            )

        return tuple(buffer_size_ranks)

    def _construct_global_block_info_list(
        self, buffer_size_ranks: Tuple[Tuple[int, int], ...]
    ) -> None:
        """Construct global block info list from param_group and num_blocks_within_param."""
        # Note that for HSDP, we want to get the rank within each sharded group for the block id.
        # When using a device mesh, 0 corresponds to the replicated group and 1 corresponds to the sharded group.
        sharded_group_rank = self._hsdp_device_mesh.get_local_rank(1)
        self._global_block_info_list = tuple(
            DDPBlockInfo(
                param=param,
                composable_block_ids=(
                    param_index,
                    f"sharded_group_rank_{sharded_group_rank}-block_{block_index}",
                ),
                allocate_zeros_tensor=self._allocate_zeros_distributed_tensor,
                get_tensor=lambda input_tensor: (
                    input_tensor.to_local()
                    if isinstance(input_tensor, dtensor.DTensor)
                    else input_tensor
                ),
                group_source_rank=group_source_rank,
            )
            for (
                (param_index, param),
                num_blocks_within_param,
                (buffer_size_ranks_start, buffer_size_ranks_end),
            ) in zip(
                enumerate(self._param_group[PARAMS]),
                self._global_num_blocks_per_param,
                generate_pairwise_indices(self._global_num_blocks_per_param),
                strict=True,
            )
            for block_index, (_, group_source_rank) in zip(
                range(num_blocks_within_param),
                buffer_size_ranks[buffer_size_ranks_start:buffer_size_ranks_end],
                strict=True,
            )
        )

    def _merge_and_block_parameters(
        self,
    ) -> None:
        """Split, merge, and block parameters."""
        global_blocked_params = []
        # self._global_num_splits_per_param refers to the total number of splits within each
        # flattened parameter (obtained by split tensor block recovery).
        # This has the same length as the number of flattened parameters contained in
        # self._param_group[PARAMS].
        global_num_splits_per_param = []
        # self._global_num_blocks_per_split refers to the total number of blocks within each
        # split parameter.
        # This has the same length as the number of split parameters.
        global_num_blocks_per_split_param = []
        # self._global_merged_dims_list has the same length as the total number of split tensor
        # blocks within all flattened parameters obtained from split tensor block recovery.
        global_merged_dims_list = []

        for flattened_param in self._param_group[PARAMS]:
            # Split flattened parameters into valid tensor blocks of the parameter.
            split_params = HSDPDistributor._split_tensor_block_recovery(
                flattened_param,
                self._param_to_metadata[flattened_param].shape,
                self._param_to_metadata[flattened_param].start_idx,
                self._param_to_metadata[flattened_param].end_idx,
            )
            global_num_splits_per_param.append(len(split_params))

            for split_param in split_params:
                # Obtain blocks for each parameter after merging.
                merged_dims = (
                    merge_small_dims(
                        split_param.size(), self._param_group[MAX_PRECONDITIONER_DIM]
                    )
                    if self._param_group[USE_MERGE_DIMS]
                    else split_param.size()
                )
                blocks_within_split_param = multi_dim_split(
                    split_param.view(merged_dims),
                    self._param_group[MAX_PRECONDITIONER_DIM],
                )

                # Generate and extend block info list and extend blocked parameters list.
                # Note that the block info list should have the same length as the blocked parameters list.
                global_blocked_params.extend(
                    # Note: We are using tensor.detach() here to explicitly set block_param (a view of the original
                    # parameter) to requires_grad = False in order to prevent errors with print and PT2 compile.
                    # Remove this tensor.detach() once https://github.com/pytorch/pytorch/issues/113793 is fixed.
                    block_param.detach()
                    for block_param in blocks_within_split_param
                )

                # Stores the merged dimensions for each parameter and the number of blocks for each param so
                # we could use this later for constructing the mask on filtering blocks when grad is None.
                global_merged_dims_list.append(merged_dims)
                global_num_blocks_per_split_param.append(len(blocks_within_split_param))

        # Check that the number of blocks for each parameter equals to the summation of the number of blocks
        # from each split parameter.
        self._global_num_blocks_per_param = tuple(
            sum(global_num_blocks_per_split_param[block_index:next_block_index])
            for (block_index, next_block_index) in generate_pairwise_indices(
                global_num_splits_per_param
            )
        )

        # Set lists as tuples.
        self._global_blocked_params = tuple(global_blocked_params)
        self._global_num_splits_per_param = tuple(global_num_splits_per_param)
        self._global_num_blocks_per_split_param = tuple(
            global_num_blocks_per_split_param
        )
        self._global_merged_dims_list = tuple(global_merged_dims_list)

    @staticmethod
    def _split_local_dist_buffers(
        buffer_size_ranks: Tuple[Tuple[int, int], ...],
        local_dist_buffers: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Split distributed buffers for each local rank into views for each assigned block.

        Args:
            buffer_size_ranks (Tuple[Tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.
            local_dist_buffers (Tuple[torch.Tensor, ...]): A list of local distributed buffers that
                correspond to each rank. Each distributed buffer will be split according to the
                assigned tensor blocks.

        Returns:
            splitted_local_dist_buffers (Tuple[torch.Tensor, ...]): A list of tuples containing a view of the
                local distributed buffer for each tensor block.

        Example:
            tensor0 = tensor(1024)
            tensor1 = tensor(1024)
            buffer_size_ranks = [(128, 0), (64, 0), (512, 1), (256, 0)]
            local_dist_buffers = [tensor0, tensor1]
            -> splitted_local_dist_buffers = [
                tensor0's view(  0-128 bytes),
                tensor0's view(128-192 bytes),
                tensor1's view(  0-512 bytes),
                tensor0's view(192-448 bytes),
            ]

        """

        # Create list of lists containing local views of each split tensor for each rank.
        split_tensors_list = []
        for rank, local_dist_buffer in enumerate(local_dist_buffers):
            required_buffer_sizes = [s for s, r in buffer_size_ranks if r == rank]
            remainder_size = local_dist_buffer.size(0) - sum(required_buffer_sizes)
            assert (
                remainder_size >= 0
            ), f"Local distributed buffer size {local_dist_buffer.size(0)} is "
            "not larger than or equal to the sum of buffer sizes {sum(required_buffer_sizes)}!"
            split_tensors = torch.split(
                local_dist_buffer, required_buffer_sizes + [remainder_size]
            )
            split_tensors_list.append(split_tensors)

        # Obtain ordered buffer ranks containing (view of local buffer, rank).
        splitted_local_dist_buffers = []
        buffer_indices = [0] * len(
            local_dist_buffers
        )  # index counter for each rank for obtaining right buffer
        for _, rank in buffer_size_ranks:
            splitted_local_dist_buffers.append(
                split_tensors_list[rank][buffer_indices[rank]]
            )
            buffer_indices[rank] += 1

        return tuple(splitted_local_dist_buffers)

    def _construct_distributed_buffers(
        self, buffer_size_ranks: Tuple[Tuple[int, int], ...]
    ) -> None:
        """Construct the distributed buffers for AllGather communications.

        Note that this function will construct the distributed buffer for the AllGather
        communication. In addition, it massages the distributed buffer to obtain views
        of the buffer corresponding to each block assigned to the current rank.

        Args:
            buffer_size_ranks (Tuple[Tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.

        """

        # Calculate buffer size each rank needs.
        local_buffer_sizes = tuple(
            sum(buffer_size for buffer_size, rank in buffer_size_ranks if rank == i)
            for i in range(self._dist_group_size)
        )

        # Calculate the whole buffer size and obtain buffers for every rank.
        max_buffer_size_sum = max(local_buffer_sizes)
        total_buffer_size = max_buffer_size_sum * self._dist_group_size
        self._global_dist_buffer = torch.zeros(
            total_buffer_size,
            dtype=torch.int8,
            device=self._global_block_info_list[0].param.device,
        )
        local_dist_buffers = torch.split(self._global_dist_buffer, max_buffer_size_sum)
        splitted_local_dist_buffers = HSDPDistributor._split_local_dist_buffers(
            buffer_size_ranks, local_dist_buffers
        )

        # Get local buffer for specific group rank.
        self._local_dist_buffer = local_dist_buffers[self._group_rank]

        # Obtain the list of buffers corresponding to each block (ignoring padding).
        # Note that each buffer is reshaped into the block's shape and viewed in terms
        # of the communication data type.
        self._global_dist_blocked_buffers = tuple(
            buffer.split(
                blocked_param.numel() * get_dtype_size(self._communication_dtype)
            )[0]
            .view(self._communication_dtype)
            .view(blocked_param.shape)
            for buffer, blocked_param in zip(
                splitted_local_dist_buffers, self._global_blocked_params, strict=True
            )
        )
        self._local_dist_blocked_buffers = compress_list(
            self._global_dist_blocked_buffers, self._distributor_selector
        )
        self._global_masked_dist_blocked_buffers = self._global_dist_blocked_buffers
        self._local_masked_dist_blocked_buffers = self._local_dist_blocked_buffers

    def _merge_and_block_gradients(
        self,
    ) -> Tuple[Tensor, ...]:
        """Split, merge, and block gradients.

        Returns:
            local_masked_blocked_grads (Tuple[Tensor, ...]): Local gradients with grad not None.

        """
        local_masked_blocked_grads = []
        global_grad_selector = []

        for (
            flattened_param,
            num_blocks,
            (block_index, next_block_index),
            (split_index, next_split_index),
        ) in zip(
            self._param_group[PARAMS],
            self._global_num_blocks_per_param,
            generate_pairwise_indices(self._global_num_blocks_per_param),
            generate_pairwise_indices(self._global_num_splits_per_param),
            strict=True,
        ):
            flattened_grad = flattened_param.grad
            param_distributor_selector = self._distributor_selector[
                block_index:next_block_index
            ]

            # Update the selector.
            global_grad_selector.extend([flattened_grad is not None] * num_blocks)

            if flattened_grad is None or not any(param_distributor_selector):
                # Skip split_tensor_block_recovery and multi_dim_split if this blocked grad will not be used locally.
                continue

            # Split flattened gradients into valid tensor blocks of the gradient.
            split_grads = HSDPDistributor._split_tensor_block_recovery(
                flattened_grad,
                self._param_to_metadata[flattened_param].shape,
                self._param_to_metadata[flattened_param].start_idx,
                self._param_to_metadata[flattened_param].end_idx,
            )

            # Get the merged dimensions and the number of blocks for each split gradient.
            merged_dims_within_flattened_param = self._global_merged_dims_list[
                split_index:next_split_index
            ]
            num_blocks_within_split_grads = self._global_num_blocks_per_split_param[
                split_index:next_split_index
            ]

            for (
                grad,
                merged_dims,
                (blocks_within_split_index, next_blocks_within_split_index),
            ) in zip(
                split_grads,
                merged_dims_within_flattened_param,
                generate_pairwise_indices(num_blocks_within_split_grads),
                strict=True,
            ):
                # Obtain blocks for each split gradient after merging.
                blocks_within_grad = multi_dim_split(
                    grad.view(merged_dims), self._param_group[MAX_PRECONDITIONER_DIM]
                )
                # Generate block-to-parameter metadata and extend blocked parameters list.
                local_masked_blocked_grads.extend(
                    compress_list(
                        blocks_within_grad,
                        param_distributor_selector[
                            blocks_within_split_index:next_blocks_within_split_index
                        ],
                    )
                )

        # Set global grad selector as tuple.
        self._global_grad_selector = tuple(global_grad_selector)

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

            # Re-compress DDP-specific tensor lists using the updated selector.
            self._global_masked_blocked_params = compress_list(
                self._global_blocked_params, self._global_grad_selector
            )
            self._global_masked_dist_blocked_buffers = compress_list(
                self._global_dist_blocked_buffers, self._global_grad_selector
            )
            self._local_masked_dist_blocked_buffers = compress_list(
                self._local_dist_blocked_buffers, self._local_grad_selector
            )

        return local_masked_blocked_grads

    @staticmethod
    def _split_tensor_block_recovery(
        tensor_shard: Tensor,
        original_shape: torch.Size,
        start_idx: int,
        end_idx: int,
    ) -> List[Tensor]:
        """Chunks flattened tensor in order to re-construct valid blocks with respect to the original
        multi-dimensional tensor shape and parameter boundaries.

        Starting from the first dimension, the largest possible slices in each dimension
        (with the remaining dimensions on the right retaining the original shape) are split off.

        The following is an example of how the function works for a 2-D tensor shard:

        Given an original tensor with shape (7, 14) in Fig. 1, we receive a flattened tensor shard from FSDP
        corresponding to Fig. 4. Note that this flattened tensor shard corresponds to the shard of the tensor
        in Fig. 2. In order to respect the tensor shape, we need to split the tensor into up to three blocks
        (as in Fig. 5). This requires splitting the tensor in Fig. 2 (see flattened tensor shard in Fig. 4)
        then reshaping each flattened split tensor into its original shape (see reshaped split tensors in Fig.
        3 and 6).

          ______________
         |       _______|                        _______                         _______
         |______|       |                 ______|       |                 ______|_______|
         |              |       ->       |              |       ->       |              |
         |           ___|                |           ___|                |______________|
         |__________|   |                |__________|                    |__________|
         |______________|

          original tensor                  tensor_shard                    split tensors

              Fig. 1                           Fig. 2                          Fig. 3

        Flattened original tensor in Fig. 1:
         ________________________________________________________________
        |____________________|_________________________|_________________|
                             ^       tensor_shard      ^
                          start_idx                 end_idx

                                    Fig. 4

         ________________________________________________________________
        |____________________|______|_______________|__|_________________|
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    ^^^ denoted the flattened split tensors in Fig. 3.

                                    Fig. 5

        Reshaped split tensors (i.e., the tensors in Fig. 3):
                                     _______
                                    |_______|  <- left split
                             ______________
                            |              |  <- center split
                            |______________|
                             __________
                            |__________|      <- right split

                                    Fig. 6

        Args:
            tensor_shard (Tensor): A shard of the flattened version of original tensor to split.
            original_shape (torch.Size): Shape of original tensor that tensor_shard is a slice of.
            start_idx (int): Flattened index in the original tensor where tensor starts (inclusive).
            end_idx (int): Flattened index in the original tensor where tensor ends (exclusive).

        Returns:
            split_tensors (List[Tensor]): List of tensors.

        """
        if len(tensor_shard.size()) != 1:
            raise ValueError(
                f"Input tensor is not flat, has shape {tensor_shard.size()=}."
            )

        def block_within_tensor_shard_recovery(
            block_within_tensor_shard: Tensor,
            dimension: int,
            block_start_idx: int,
            block_end_idx: int,
        ) -> List[Tensor]:
            assert (
                block_end_idx - block_start_idx == block_within_tensor_shard.numel()
            ), f"Start/end indices do not match tensor size: {block_start_idx=}, "
            f"{block_end_idx=}, {block_within_tensor_shard.numel()=}!"

            if block_end_idx == block_start_idx:
                return []

            # Handle case where shape is one-dimensional.
            # Because it reached the last dimension, we can simply return the flattened tensor.
            if dimension == len(original_shape) - 1:
                return [block_within_tensor_shard]

            # Instantiate list of tensor blocks.
            center_split_tensor_blocks = []

            # Instantiates flattened indices for recursion.
            remaining_size = prod(original_shape[dimension + 1 :])

            """
             ________________________________________________________________
            |____________________|______|_______________|__|_________________|
                                 ^      ^               ^  ^
                       block_start_idx  |               | block_end_idx
                                        |               |
                            center_split_start_idx      |
                                                center_split_end_idx

            This came from Fig. 4 above.

            """
            # Get starting index of the center split of the tensor shard. (See figure above.)
            # This is equal to ceil(block_start_idx / remaining_size) * remaining_size.
            center_split_start_idx = (
                (block_start_idx + remaining_size - 1) // remaining_size
            ) * remaining_size
            # Similarly, get end index of the center split of the tensor shard.
            # This is equal to floor(block_end_idx / remaining_size) * remaining_size.
            center_split_end_idx = block_end_idx // remaining_size * remaining_size

            # Handles largest convex partition in the center.
            if center_split_start_idx < center_split_end_idx:
                center_split_start_idx_in_block = (
                    center_split_start_idx - block_start_idx
                )
                length_of_center_split = center_split_end_idx - center_split_start_idx
                new_shape = [-1] + list(original_shape[dimension + 1 :])
                # NOTE: We use Tensor.narrow() instead of slicing in order to guarantee
                # there is no copy of the tensor.
                center_split_tensor_blocks.append(
                    block_within_tensor_shard.narrow(
                        0,
                        center_split_start_idx_in_block,
                        length_of_center_split,
                    ).view(new_shape)
                )
            elif center_split_start_idx > center_split_end_idx:
                # Recursively call split tensor block recovery on the full
                # flattened tensor ignoring the first dimension of the original
                # tensor shape.
                return block_within_tensor_shard_recovery(
                    block_within_tensor_shard=block_within_tensor_shard,
                    dimension=dimension + 1,
                    block_start_idx=block_start_idx,
                    block_end_idx=block_end_idx,
                )

            # Recursively call split tensor block recovery on the left and right
            # splits of the flattened tensor.
            left_split_start_idx_in_block = 0
            left_split_tensor_size = center_split_start_idx - block_start_idx
            left_split_tensor_blocks = block_within_tensor_shard_recovery(
                block_within_tensor_shard=block_within_tensor_shard.narrow(
                    0,
                    start=left_split_start_idx_in_block,
                    length=left_split_tensor_size,
                ),
                dimension=dimension + 1,
                block_start_idx=block_start_idx,
                block_end_idx=center_split_start_idx,
            )

            center_split_end_idx_in_block = center_split_end_idx - block_start_idx
            right_split_tensor_size = block_end_idx - center_split_end_idx
            right_split_tensor_blocks = block_within_tensor_shard_recovery(
                block_within_tensor_shard=block_within_tensor_shard.narrow(
                    0,
                    start=center_split_end_idx_in_block,
                    length=right_split_tensor_size,
                ),
                dimension=dimension + 1,
                block_start_idx=center_split_end_idx,
                block_end_idx=block_end_idx,
            )

            return (
                left_split_tensor_blocks
                + center_split_tensor_blocks
                + right_split_tensor_blocks
            )

        return block_within_tensor_shard_recovery(
            block_within_tensor_shard=tensor_shard,
            dimension=0,
            block_start_idx=start_idx,
            block_end_idx=end_idx,
        )

    def _allocate_zeros_distributed_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Instantiates distributed tensor using DTensor.

        Args:
            shape (shape type accepted by torch.zeros() including Tuple[int, ...]):
                Shape of desired tensor.
            dtype (dtype type accepted by torch.zeros() including torch.dtype):
                DType of desired tensor.
            device (device type accepted by torch.zeros() including torch.device):
                Device of desired tensor.

        Returns:
            out (Tensor): Desired Tensor.

        """
        ranks_in_replicated_group = torch.tensor(
            dist.get_process_group_ranks(self._hsdp_device_mesh.get_group(0))
        )
        device_mesh_2d = get_device_mesh(
            device_type=device.type,
            mesh=ranks_in_replicated_group.view(-1, self._dist_group_size),
            mesh_dim_names=("replicate", "shard"),
        )

        return dtensor_zeros(
            shape,
            dtype=dtype,
            device_mesh=device_mesh_2d["replicate"],
            placements=[dtensor.Replicate()],
        )
