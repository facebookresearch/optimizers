"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections.abc import Iterable
from functools import partial
from itertools import islice
from typing import Any, Literal, overload

import torch
from distributed_shampoo.shampoo_types import (
    CommunicationDType,
    HybridShardShampooConfig,
    PARAMS,
)
from distributed_shampoo.utils.shampoo_block_info import DTensorBlockInfo
from distributed_shampoo.utils.shampoo_dist_utils import get_device_mesh
from distributed_shampoo.utils.shampoo_distributor import DistributorInterface
from distributed_shampoo.utils.shampoo_utils import (
    compress_list,
    distribute_buffer_sizes,
    generate_pairwise_indices,
    get_dtype_size,
)
from torch import distributed as dist, Tensor
from torch.distributed import tensor as dtensor
from torch.distributed.device_mesh import _mesh_resources
from torch.distributed.tensor import DTensor, zeros as dtensor_zeros

logger: logging.Logger = logging.getLogger(__name__)


class HybridShardDistributor(DistributorInterface):
    """HybridShard Distributor class.

    The constructor internally sets up `DeviceMesh` objects as necessary for distributing memory
    and computation, so torch.distributed must be initialized in advance.

    Unlike FullyShardDistributor, HybridShardDistributor requires the user to pass in a device mesh used for
    Hybrid Shard. For example, suppose we have 48 GPUs and the Hybrid Shard group size is 8. Then:

    Hybrid Shard Device Mesh with (Replicate, Shard) = (6, 8):

        device_mesh = [[ 0,  1,  2,  3,  4,  5,  6,  7]
                       [ 8,  9, 10, 11, 12, 13, 14, 15]
                       [16, 17, 18, 19, 20, 21, 22, 23]
                       [24, 25, 26, 27, 28, 29, 30, 31]
                       [32, 33, 34, 35, 36, 37, 38, 39]
                       [40, 41, 42, 43, 44, 45, 46, 47]]

    For example, if my device is rank 11, then:
        device_mesh["replicate"] = [3, 11, 19, 27, 35, 43]
        device_mesh["shard"] = [8, 9, 10, 11, 12, 13, 14, 15]

    Since the parameters are sharded along the "shard" dimension, we would normally replicate the
    computation along the "replicate" dimension. With Hybrid Shard Shampoo, we instead want to
    distribute the computation and memory requirements across the "replicate" dimension of the original
    Hybrid Shard device mesh.

    For example, suppose that the num_trainers_per_group = 3. We want to form a (2, 3)-submesh on
    the ranks [3, 11, 19, 27, 35, 43] (and similar).

    HybridShardDistributor 2D Sub-Mesh Example with (Replicate, Shard) = (2, 3):

        submesh = [[ 3, 11, 19]
                   [27, 35, 43]]

    In this case, optimizer states will live on different "replicate" meshes: {[3, 27], [11, 35],
    [19, 43]}. In order to synchronize the optimizer step, we will communicate along the "shard"
    mesh {[3, 11, 19], [27, 35, 43]}.

    Args:
        param_group (dict[str, Any]): Parameter group containing parameters.
        distributed_config (HybridShardShampooConfig): Configuration for HybridShard Shampoo.

    """

    def __init__(
        self,
        param_group: dict[str, Any],
        distributed_config: HybridShardShampooConfig,
    ) -> None:
        self._hybrid_shard_device_mesh: torch.distributed.device_mesh.DeviceMesh = (
            distributed_config.device_mesh
        )

        super().__init__(param_group)
        if not dist.is_initialized():
            raise RuntimeError(
                "HybridShardDistributor needs torch.distributed to be initialized!"
            )

        # Construct global masked blocked parameters (which is DDP-specific).
        self._global_masked_blocked_params: tuple[Tensor, ...] = (
            self._global_blocked_params
        )

        # Check num_trainers_per_group and replicated group size.
        # NOTE: If num_trainers_per_group = -1, then we use the replicated group size.
        self._replicated_group_size: int = self._hybrid_shard_device_mesh.size(0)

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
        match distributed_config.communication_dtype:
            case CommunicationDType.BF16:
                communication_dtype = torch.bfloat16
            case CommunicationDType.FP16:
                communication_dtype = torch.float16
            case CommunicationDType.FP32 | CommunicationDType.DEFAULT:
                communication_dtype = torch.float32
            case _:
                raise NotImplementedError(
                    f"Unsupported communication dtype: {distributed_config.communication_dtype}"
                )

        # Initialize _dist_group and _group_rank.
        # Note that this requires initializing all process groups.
        # Splits replicated ranks group into smaller groups of size self._dist_group_size.
        # Instantiates this by using DeviceMesh.
        ranks_in_all_replicated_groups = self._hybrid_shard_device_mesh.mesh.T
        for ranks_in_replicated_group in ranks_in_all_replicated_groups:
            device_mesh = get_device_mesh(
                device_type=self._hybrid_shard_device_mesh.device_type,
                mesh=tuple(
                    map(
                        partial(tuple),
                        ranks_in_replicated_group.view(
                            -1, self._dist_group_size
                        ).tolist(),
                    )
                ),
                mesh_dim_names=("replicate", "shard"),
            )
            if dist.get_rank() in ranks_in_replicated_group:
                # NOTE: We want the process group in the device mesh that the current rank
                # belongs to but solely along the "shard" dimension for communications.
                #
                # For example, if the current rank is 11, then I want the process group
                # that contains the ranks [3, 11, 19].
                self._comms_dist_group: dist.ProcessGroup = device_mesh.get_group(
                    "shard"
                )

        comms_group_rank: int = dist.get_rank(self._comms_dist_group)

        # Assign ranks to blocks with their respective buffer size.
        buffer_size_ranks = distribute_buffer_sizes(
            buffer_sizes=tuple(
                blocked_param.numel() * get_dtype_size(communication_dtype)
                for blocked_param in self._global_blocked_params
            ),
            group_size=self._dist_group_size,
        )

        self._local_block_info_list: tuple[DTensorBlockInfo, ...] = (
            self._construct_local_block_info_list(
                group_source_ranks=tuple(
                    group_source_rank for _, group_source_rank in buffer_size_ranks
                ),
                group_rank=comms_group_rank,
            )
        )

        # Initialize selectors and local blocked (masked) parameters.
        self._distributor_selector: tuple[bool, ...] = tuple(
            group_source_rank == comms_group_rank
            for _, group_source_rank in buffer_size_ranks
        )
        self._local_blocked_params: tuple[Tensor, ...] = compress_list(
            self._global_blocked_params, self._distributor_selector
        )
        self._local_masked_blocked_params: tuple[Tensor, ...] = (
            self._local_blocked_params
        )
        self._local_grad_selector: tuple[bool, ...] = (True,) * len(
            self._local_blocked_params
        )

        self._construct_distributed_buffers(
            buffer_size_ranks=buffer_size_ranks,
            communication_dtype=communication_dtype,
            comms_group_rank=comms_group_rank,
        )

    @overload
    @torch.no_grad()
    def _get_params_or_grads(self) -> Iterable[Tensor]: ...

    @overload
    @torch.no_grad()
    def _get_params_or_grads(
        self, get_grad: Literal[True]
    ) -> Iterable[Tensor | None]: ...

    @overload
    @torch.no_grad()
    def _get_params_or_grads(self, get_grad: Literal[False]) -> Iterable[Tensor]: ...

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

    # NOTE: Remove this function once PT2 supports all_gather with functional collective
    @torch.no_grad()
    @torch.compiler.disable
    def all_gather_into_tensor(self) -> None:
        dist.all_gather_into_tensor(
            self._global_dist_buffer,
            self._local_dist_buffer,
            group=self._comms_dist_group,
        )

    @torch.no_grad()
    def update_params(
        self,
        masked_blocked_search_directions: tuple[Tensor, ...],
    ) -> None:
        """Update params stored inside this distributor according to the input search directions argument.

        Args:
            masked_blocked_search_directions (tuple[Tensor, ...]): Search directions for each local blocked parameter.

        """
        if self._communicate_params:
            assert (
                len(self._local_masked_blocked_params)
                == len(masked_blocked_search_directions)
            ), f"Expected {len(self._local_masked_blocked_params)=} to be equal to {len(masked_blocked_search_directions)=}."

            # torch._foreach only accepts non-empty list
            if masked_blocked_search_directions:
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

            # torch._foreach only accepts non-empty list
            if self._global_masked_blocked_params:
                # Copy updated blocked params in global_masked_dist_blocked_buffers into global_masked_blocked_params.
                torch._foreach_copy_(
                    self._global_masked_blocked_params,
                    self._global_masked_dist_blocked_buffers,
                )

        else:
            assert (
                len(self._local_masked_dist_blocked_buffers)
                == len(masked_blocked_search_directions)
            ), f"Expected {len(self._local_masked_dist_blocked_buffers)=} to be equal to {len(masked_blocked_search_directions)=}."

            # torch._foreach only accepts non-empty list
            if masked_blocked_search_directions:
                # Search directions multiplied by alpha are distributed.
                # Copy the local search directions to the communication buffer.
                torch._foreach_copy_(
                    self._local_masked_dist_blocked_buffers,
                    masked_blocked_search_directions,
                )

            self.all_gather_into_tensor()

            # torch._foreach only accepts non-empty list
            if self._global_masked_blocked_params:
                # Add search directions in global_masked_dist_blocked_buffers to global_masked_blocked_params.
                torch._foreach_add_(
                    self._global_masked_blocked_params,
                    self._global_masked_dist_blocked_buffers,
                )

    def _construct_composable_block_ids(
        self,
        param_index: int,
        block_index: int,
        rank: int | None = None,
    ) -> tuple[int, str]:
        """Construct composable block ids for each parameter.

        Args:
            param_index (int): Index of the current parameter within self._param_group[PARAMS].
            block_index (int): Block index that is accumulated across all parameters within a parameter group.
            rank (int | None): Rank of this process group; should be non None in FullyShard/HybridShard setting. (Default: None)

        Returns:
            tuple[int, str]: Composable block id tuple containing global block index and local block name.
                The latter will be used to identify blocks in the masked tensor.
        """
        assert rank is not None, "Rank must be provided in hybrid shard setting."
        return (param_index, f"rank_{rank}-block_{block_index}")

    @torch.no_grad()
    def _construct_local_block_info_list(
        self, group_source_ranks: tuple[int, ...], group_rank: int
    ) -> tuple[DTensorBlockInfo, ...]:
        """Construct the local block info list.

        This method creates a list of DTensorBlockInfo objects, which contain information about each parameter block,
        including its composable block IDs, functions to allocate tensors, and a method to retrieve tensors.

        Args:
            group_source_ranks (tuple[int, ...]): A list of assigned ranks for each block.
            group_rank (int): Rank of the current process group.

        Returns:
            tuple[DTensorBlockInfo, ...]: A tuple of DTensorBlockInfo objects for each parameter block.
        """
        # Call `super()` instead of `self` as a performance optimization.
        # This leads to O(1) instead of O(N) complexity to retrieve the parameters.
        non_empty_params: Iterable[Tensor] = filter(
            lambda p: isinstance(p, DTensor) and p.to_local().numel() > 0,
            super()._get_params_or_grads(),
        )

        # Note that for HybridShard, we want to get the rank within each sharded group for the block id.
        # When using a device mesh, 0 corresponds to the replicated group and 1 corresponds to the sharded group.
        sharded_group_rank = self._hybrid_shard_device_mesh.get_local_rank(1)
        return tuple(
            DTensorBlockInfo(
                param=param,
                composable_block_ids=self._construct_composable_block_ids(
                    param_index=param_index,
                    block_index=block_index,
                    rank=sharded_group_rank,
                ),
                allocate_zeros_tensor=partial(
                    self._allocate_zeros_distributed_tensor,
                    group_source_rank=group_source_rank,
                ),
            )
            for (
                (param_index, param),
                (buffer_size_ranks_start, buffer_size_ranks_end),
            ) in zip(
                enumerate(non_empty_params),
                generate_pairwise_indices(self._global_num_blocks_per_param),
                strict=True,
            )
            for block_index, group_source_rank in enumerate(
                islice(
                    group_source_ranks, buffer_size_ranks_start, buffer_size_ranks_end
                )
            )
            if group_source_rank == group_rank
        )

    @staticmethod
    def _split_local_dist_buffers(
        buffer_size_ranks: tuple[tuple[int, int], ...],
        local_dist_buffers: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Split distributed buffers for each local rank into views for each assigned block.

        Args:
            buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.
            local_dist_buffers (tuple[torch.Tensor, ...]): A list of local distributed buffers that
                correspond to each rank. Each distributed buffer will be split according to the
                assigned tensor blocks.

        Returns:
            splitted_local_dist_buffers (tuple[torch.Tensor, ...]): A list of tuples containing a view of the
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
            f"not larger than or equal to the sum of buffer sizes {sum(required_buffer_sizes)}!"
            split_tensors = torch.split(
                local_dist_buffer, required_buffer_sizes + [remainder_size]
            )
            split_tensors_list.append(split_tensors)

        split_tensors_iterators = list(map(iter, split_tensors_list))
        return tuple(
            next(split_tensors_iterators[rank]) for _, rank in buffer_size_ranks
        )

    def _construct_distributed_buffers(
        self,
        buffer_size_ranks: tuple[tuple[int, int], ...],
        communication_dtype: torch.dtype,
        comms_group_rank: int,
    ) -> None:
        """Construct the distributed buffers for AllGather communications.

        Note that this function will construct the distributed buffer for the AllGather
        communication. In addition, it massages the distributed buffer to obtain views
        of the buffer corresponding to each block assigned to the current rank.

        Args:
            buffer_size_ranks (tuple[tuple[int, int], ...]): A list of tuples containing the
                buffer size and an assigned rank for each block.
            communication_dtype (torch.dtype): The data type used for communication.
            comms_group_rank (int): The rank of the current group within the comms group.

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
            device=self._global_blocked_params[0].device,
        )
        local_dist_buffers = torch.split(self._global_dist_buffer, max_buffer_size_sum)
        splitted_local_dist_buffers = HybridShardDistributor._split_local_dist_buffers(
            buffer_size_ranks, local_dist_buffers
        )

        # Get local buffer for specific group rank.
        self._local_dist_buffer = local_dist_buffers[comms_group_rank]

        # Obtain the list of buffers corresponding to each block (ignoring padding).
        # Note that each buffer is reshaped into the block's shape and viewed in terms
        # of the communication data type.
        self._global_dist_blocked_buffers = tuple(
            buffer.split(blocked_param.numel() * get_dtype_size(communication_dtype))[0]
            .view(communication_dtype)
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
    ) -> tuple[Tensor, ...]:
        """Merge and block gradients.

        NOTE: This function MUST be called in the step function of the optimizer after the
        gradient has been updated.

        Returns:
            local_masked_blocked_grads (tuple[Tensor, ...]): Local blocked gradients masked with grad existence.

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
        size: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        group_source_rank: int,
    ) -> torch.Tensor:
        """Instantiates distributed tensor using DTensor.

        Args:
            size (tuple[int, ...]): Shape of desired tensor.
            dtype (torch.dtype): DType of desired tensor.
            device (torch.device): Device of desired tensor.
            group_source_rank (int): Group rank (with respect to the sharded group of
                the 2D submesh) that determines which ranks the DTensor is allocated on.

        Returns:
            out (Tensor): Desired Tensor.

        """
        ranks_in_replicated_group = dist.get_process_group_ranks(
            self._hybrid_shard_device_mesh.get_group(0)
        )
        device_mesh_2d = get_device_mesh(
            device_type=device.type,
            # NOTE: Use itertools.batched(ranks_in_replicated_group, self._dist_group_size) when downstream applications are Python 3.12+ available
            mesh=tuple(
                map(
                    partial(tuple),
                    torch.tensor(ranks_in_replicated_group)
                    .view(-1, self._dist_group_size)
                    .tolist(),
                )
            ),
            mesh_dim_names=("replicate", "shard"),
        )
        # NOTE: We get all submeshes along the "replicate" dimension, then pick out
        # the sub-mesh that the optimizer state is assigned to.
        #
        # For the example above, this would give me submeshes [[3, 27], [11, 35], [19, 43]].
        # Note that the group source rank must belong to {0, 1, 2} in this case.
        # Suppose the group_source_rank = 1, then this would get the submesh [11, 35].
        replicate_submesh = _mesh_resources._get_all_submeshes(
            device_mesh_2d, "replicate"
        )[group_source_rank]

        return dtensor_zeros(
            size,
            dtype=dtype,
            device_mesh=replicate_submesh,
            placements=[dtensor.Replicate()],
        )
