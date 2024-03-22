"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor


@dataclass
class BlockInfo:
    """Utilies and metadata for each parameter block.

    Args:
        param (Tensor): The original parameter that contains the block.
        composable_block_ids (Tuple[int, str]): Tuple containing the per-parameter, per-block index tuple.
            In the DDP case, this will contain (param_index, block_index), where the param_index corresponds to
            the index of the parameter in the parameter group, and the block_index is the index of the block within
            the parameter.

            Example: If we have a model with two parameters, p1 and p2, with 2 and 3 blocks respectively, then the
                possible values of composable_block_ids are (0, "block_0"), (0, "block_1"), (1, "block_0"), (1, "block_1"),
                (1, "block_2").

                For FSDP, the block index is constructed as a string containing rank information. For example, block 0 of
                parameter p1 on rank 0 will have the composable_block_ids being (0, "rank_0-block_0"), while block 0 of parameter p1
                on rank 1 will have composable_block_ids being (0, "rank_1-block_0").

        allocate_zeros_tensor (Callable): A function that returns a zero-initialized tensor.
            This tensor must be saved in the state dictionary for checkpointing.
            This tensor might be DTensor.  get_tensor() must be used to access the value.
            Its function signature is (shape, dtype, device) -> Tensor.
            (Default: lambda shape, dtype, device: torch.zeros(shape, dtype=dtype, device=device))
        get_tensor (Callable): A function that takes a tensor allocated by allocator and returns its local tensor.
            Its function signature is (tensor: Tensor) -> Tensor.
            (Default: lambda tensor: tensor)
    """

    param: Tensor
    composable_block_ids: Tuple[int, str]
    allocate_zeros_tensor: Callable[..., Tensor] = (
        lambda shape, dtype, device: torch.zeros(size=shape, dtype=dtype, device=device)
    )
    get_tensor: Callable[..., Tensor] = lambda tensor_obj: tensor_obj


@dataclass
class DDPBlockInfo(BlockInfo):
    """Utilies and metadata for each parameter block specific to DDP Distributor.

    Args:
        group_source_rank (int): Group rank of the owner of this block. (Default: 0)

    """

    group_source_rank: int = 0
