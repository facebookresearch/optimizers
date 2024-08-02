"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import heapq
import logging
from functools import cache, partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from distributed_shampoo.shampoo_types import (
    CommunicationDType,
    DDPShampooConfig,
    PARAMS,
)
from distributed_shampoo.utils.shampoo_block_info import DDPBlockInfo
from distributed_shampoo.utils.shampoo_distributor import DistributorInterface
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    generate_pairwise_indices,
    get_dtype_size,
)
from torch import Tensor
from torch.distributed import _tensor as dtensor

from torch.distributed._tensor import zeros as dtensor_zeros

logger: logging.Logger = logging.getLogger(__name__)


class DDPDistributor(DistributorInterface):
    """DDP Distributor class.

    Handles merging, blocking, and distributing of the parameters at instantiation.
    The constructor internally sets up process groups, so torch.distributed must be initialized in advance.

    Args:
        param_group (Dict[str, Any]): Parameter group containing parameters.
        distributed_config (DDPShampooConfig): Configuration for DDP Shampoo.

    """

    def __init__(
        self,
        param_group: Dict[str, Any],
        distributed_config: DDPShampooConfig,
    ) -> None:
        super().__init__(param_group)

        # Construct global masked blocked parameters (which is DDP-specific).
        self._global_masked_blocked_params: Tuple[Tensor, ...] = (
            self._global_blocked_params
        )

        # Check num_trainers_per_group and get global and group sizes.
        # NOTE: If num_trainers_per_group = -1, then we use the global world size.
        self._global_size: int = dist.get_world_size()

        if distributed_config.num_trainers_per_group == -1:
            logger.info(
                f"Note that {distributed_config.num_trainers_per_group=}! Defaulting to world size {self._global_size}."
            )
        self._group_size: int = (
            distributed_config.num_trainers_per_group
            if distributed_config.num_trainers_per_group != -1
            else self._global_size
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
        self._dist_group: Optional[dist.ProcessGroup] = (
            dist.distributed_c10d.GroupMember.WORLD
            if self._group_size == self._global_size
            else dist.new_subgroups(group_size=self._group_size)[0]
        )
        self._group_rank: int = dist.get_rank(group=self._dist_group)

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
            (0, group_index) for group_index in range(self._group_size)
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
        """Construct the global block info list.

        Args:
            buffer_size_ranks (Tuple[Tuple[int, int], ...]): A list of tuples containing the buffer size
                and an assigned rank for each block.

        """
        # Construct global block info list.
        self._global_block_info_list = tuple(
            DDPBlockInfo(
                param=param,
                composable_block_ids=(param_index, f"block_{block_index}"),
                # Curry a function to capture a local variable "group_source_rank".
                allocate_zeros_tensor=partial(
                    self._allocate_zeros_distributed_tensor,
                    group_source_rank=group_source_rank,
                ),
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
            for i in range(self._group_size)
        )

        # Calculate the whole buffer size and obtain buffers for every rank.
        max_buffer_size_sum = max(local_buffer_sizes)
        total_buffer_size = max_buffer_size_sum * self._group_size
        self._global_dist_buffer = torch.zeros(
            total_buffer_size,
            dtype=torch.int8,
            device=self._global_block_info_list[0].param.device,
        )
        local_dist_buffers = torch.split(self._global_dist_buffer, max_buffer_size_sum)
        splitted_local_dist_buffers = DDPDistributor._split_local_dist_buffers(
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

    def _allocate_zeros_distributed_tensor(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group_source_rank: int,
    ) -> torch.Tensor:
        """Instantiates distributed tensor using DTensor.

        Args:
            shape (shape type accepted by torch.zeros() including Tuple[int, ...]):
                Shape of desired tensor.
            dtype (dtype type accepted by torch.zeros() including torch.dtype):
                DType of desired tensor.
            device (device type accepted by torch.zeros() including torch.device):
                Device of desired tensor.
            group_source_rank (int): Desired source rank of allocated zeros tensor within the process group.

        Returns:
            out (Tensor): Desired DTensor.

        """
        device_mesh_ranks = tuple(
            range(
                group_source_rank % self._group_size,
                self._global_size,
                self._group_size,
            )
        )

        @cache
        def get_device_mesh(device_mesh_ranks: Tuple[int, ...]) -> dtensor.DeviceMesh:
            """Returns device mesh from provided ranks. This function will cache previous meshes according to the input ranks.

            Args:
                device_mesh_ranks ([Tuple[int, ...]): Ranks to use in device mesh of desired tensor.

            """
            return dtensor.DeviceMesh(device_type=device.type, mesh=device_mesh_ranks)

        return dtensor_zeros(
            shape,
            dtype=dtype,
            device_mesh=get_device_mesh(device_mesh_ranks),
            placements=[dtensor.Replicate()],
        )
