# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_api import _ReduceOp
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultReduceScatter


class BaseVarianceCaptureHook:
    """
    Base class for functionalities for capturing gradient variance statistics
    during training. It maintains running sums of squared gradients across data parallel ranks.

    Attributes:
        _capture_enabled (bool): Flag indicating whether gradient capture is currently active.
        _sum_g_sq (torch.Tensor | None): Accumulated sum of squared gradient samples, used for
            variance computation.

    Note:
        This is an abstract base class and should not be instantiated directly. Use
        FSDPVarianceCaptureHook or DDPVarianceCaptureHook depending on your distributed
        training strategy which impacts the capture mechanism and the statistics stored.
    """

    def __init__(self) -> None:
        self._capture_enabled: bool = False

        # Gradient sample statistics
        self._sum_g_sq: torch.Tensor | None = None

    def clear_statistics(self):
        # Clear the accumulated sum of squared gradient samples
        self._sum_g_sq = None

    def set_capture_status(self, enable: bool):
        # Set gradient capture status
        self._capture_enabled = enable

    def clone_sum_g_sq(self) -> torch.Tensor:
        # Return a clone of the sum of squared gradient samples
        return self._sum_g_sq.clone()


class FSDPVarianceCaptureHook(DefaultReduceScatter, BaseVarianceCaptureHook):
    """
    This hook captures gradient variance statistics for FSDP (Fully Sharded Data Parallel) modules
    by intercepting gradients before the reduce-scatter operation by overloading the reduce-scatter operation.

    Reference implementation:
    https://github.com/pytorch/pytorch/blob/e273ff028a8cf197a47b863a589882c00959b502/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L116C7-L131

    It captures gradients for an entire FSDP module, meaning it operates at the module level and
    collects statistics for all parameters managed by the FSDP wrapper.

    Key behaviors:
    - Captures local batch gradients on the FSDP module level before the reduce-scatter collective operation
    - Performs reduce-scatter on squared gradient samples for memory-efficient storage
    - Operates only across FSDP shard groups (not across the replicate groups for when HSDP enabled)
    - Stores statistics in the shape of the sharded tensor (per-rank shard)

    Attributes:
        _sum_g_sq (torch.Tensor): Accumulated sum of squared gradient samples for the parameter
            shard stored on this rank, summed across data samples from all sharded dimensions for the data parallel ranks.

    """

    def __init__(self) -> None:
        DefaultReduceScatter.__init__(self)
        BaseVarianceCaptureHook.__init__(self)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: _ReduceOp,
        async_op: bool = False,
    ) -> dist.Work:
        if self._capture_enabled:
            # Flatten gradient for simplicity
            g_sample = input_tensor.detach()

            # Accumulate squared gradient samples
            if self._sum_g_sq is None:
                self._sum_g_sq = torch.zeros_like(output_tensor)
            scattered_sample = torch.zeros_like(self._sum_g_sq)
            DefaultReduceScatter.__call__(
                self,
                output_tensor=scattered_sample,
                input_tensor=g_sample.pow(2),
                group=group,
                op=dist.ReduceOp.SUM,
                async_op=async_op,
            )
            self._sum_g_sq += scattered_sample

        # Call the original reduce-scatter operation
        return DefaultReduceScatter.__call__(
            self,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            op=op,
            async_op=async_op,
        )


class ReplicateVarianceCaptureHook(DefaultReduceScatter, BaseVarianceCaptureHook):
    """
    This hook captures gradient variance statistics for replicate modules (replicate_with_fsdp).

    Since replicate modules have world_size == 1 for the reduce-scatter group,
    the custom reduce_scatter ``__call__`` is never invoked by ``foreach_reduce``.
    Instead, this hook captures gradient statistics by overriding ``allocate``.

    In ``foreach_reduce``, ``allocate`` is called twice:
    1. First call: allocates the input buffer (gradient data is copied in afterwards).
    2. Second call: allocates the output buffer. At this point, the input buffer
       contains the gradient data (post copy-in, pre-divide), so we capture g^2 here.

    The captured statistics have the same format as FSDPVarianceCaptureHook (flattened
    per-module gradient), enabling reuse of FSDPVarianceMetricsCalculator for
    variance computation.

    Attributes:
        _sum_g_sq (torch.Tensor): Accumulated sum of squared gradient samples for all
            parameters in the module on this rank, summed across data samples.
    """

    def __init__(self) -> None:
        DefaultReduceScatter.__init__(self)
        BaseVarianceCaptureHook.__init__(self)
        self._input_buffer: torch.Tensor | None = None
        self._is_first_allocate: bool = True
        # Spectral path: lazily set via set_param_infos().
        self._param_infos: list[tuple[int, tuple[int, ...]]] | None = None
        self._nuclear_gram_matrices: list[torch.Tensor | None] | None = None

    def set_param_infos(self, param_infos: list[tuple[int, tuple[int, ...]]]) -> None:
        """Switch this hook to the spectral capture mode.

        Args:
            param_infos: ``(numel, shape)`` for each parameter in the module's
                flattened gradient buffer, in buffer order. Non-2D entries
                produce ``None`` gram matrices.
        """
        self._param_infos = param_infos
        self._nuclear_gram_matrices = [None] * len(param_infos)

    def clone_nuclear_gram_matrices(self) -> list[torch.Tensor | None]:
        """Clone per-parameter gram matrices (None for non-2D parameters)."""
        if self._nuclear_gram_matrices is None:
            raise ValueError(
                "Nuclear gram matrices not available. "
                "Ensure set_param_infos() was called before capture."
            )
        return [
            g.clone() if g is not None else None for g in self._nuclear_gram_matrices
        ]

    def clear_statistics(self):
        super().clear_statistics()
        self._input_buffer = None
        self._is_first_allocate = True
        # Reset gram matrices but preserve param_infos (set once during setup).
        if self._param_infos is not None:
            self._nuclear_gram_matrices = [None] * len(self._param_infos)

    def allocate(
        self,
        size: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = DefaultReduceScatter.allocate(self, size, dtype=dtype, device=device)

        if self._is_first_allocate:
            # First call: input buffer allocation.
            # Gradient data will be copied into this buffer by foreach_reduce_scatter_copy_in.
            self._input_buffer = buffer
            self._is_first_allocate = False
        else:
            # Second call: output buffer allocation.
            # The input buffer now contains the per-rank gradient data.
            if self._capture_enabled and self._input_buffer is not None:
                g_sample = self._input_buffer.detach()

                if self._param_infos is not None:
                    # Spectral path: per-parameter gram matrices for 2D params.
                    assert self._nuclear_gram_matrices is not None
                    offset = 0
                    for i, (numel, shape) in enumerate(self._param_infos):
                        if len(shape) == 2:
                            g_param = g_sample[offset : offset + numel].reshape(shape)
                            k = min(shape)
                            if self._nuclear_gram_matrices[i] is None:
                                self._nuclear_gram_matrices[i] = torch.zeros(
                                    k, k, device=g_param.device, dtype=g_param.dtype
                                )
                            if shape[0] >= shape[1]:
                                self._nuclear_gram_matrices[i] += g_param.T @ g_param
                            else:
                                self._nuclear_gram_matrices[i] += g_param @ g_param.T
                        offset += numel
                else:
                    # Default path: elementwise sum_g_sq.
                    if self._sum_g_sq is None:
                        self._sum_g_sq = torch.zeros_like(buffer)
                    self._sum_g_sq += g_sample.pow(2)

            # Reset for the next backward pass
            self._input_buffer = None
            self._is_first_allocate = True

        return buffer
