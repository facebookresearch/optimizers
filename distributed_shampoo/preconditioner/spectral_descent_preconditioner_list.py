"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import torch
from collections.abc import Callable
from dataclasses import asdict
from distributed_shampoo.preconditioner.matrix_functions import matrix_orthogonalization
from distributed_shampoo.preconditioner.matrix_functions_types import (
    NewtonSchulzOrthogonalizationConfig,
)
from distributed_shampoo.preconditioner.preconditioner_list import (
    PreconditionerList,
    profile_decorator,
)
from distributed_shampoo.shampoo_types import (
    ShampooPT2CompileConfig,
    SpectralDescentPreconditionerConfig,
)
from torch._higher_order_ops import foreach_map
from torch import Tensor


def _newton_schulz(
    A: Tensor,
    a: float,
    b: float,
    c: float,
    num_iterations: int,
) -> Tensor:
    transpose = A.shape[0] > A.shape[1]
    X = A.T if transpose else A
    X = X / X.norm().clamp(min=1e-8)
    for _ in range(num_iterations):
        gram = X @ X.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        X = torch.addmm(X, gram_update, X, beta=a)
    return X.T if transpose else X


def _foreach_newton_schulz(
    grads: tuple[Tensor, ...],
    coefficients: tuple[float, float, float],
    num_iterations: int,
    scales: tuple[float, ...],
) -> tuple[Tensor, ...]:
    a, b, c = coefficients
    orthogonalized = foreach_map(
        _newton_schulz,
        grads,
        a,
        b,
        c,
        num_iterations,
    )
    return tuple(result.mul(scale) for result, scale in zip(orthogonalized, scales))


class SpectralDescentPreconditionerList(PreconditionerList):
    """Preconditioner list for spectral descent.

    NOTE: This algorithm can only be used for 2D parameters, or parameters that have been reshaped to 2D.
    Which parameters are reshaped to 2D is determined by the max_preconditioner_dim argument in DistributedShampoo.
    If all >2D parameters should be guaranteed to be reshaped to 2D, then max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2 has to be used.

    Args:
        block_list (tuple[Tensor, ...]): List of (blocks of) parameters.
        preconditioner_config (SpectralDescentPreconditionerConfig): Configuration for spectral descent.

    """

    def __init__(
        self,
        block_list: tuple[Tensor, ...],
        preconditioner_config: SpectralDescentPreconditionerConfig,
        shampoo_pt2_compile_config: ShampooPT2CompileConfig | None = None,
    ) -> None:
        if any(block.dim() != 2 for block in block_list):
            raise ValueError(
                "Spectral descent can only be used for 2D parameters, or parameters that have been reshaped to 2D. "
                "To guarantee that all >2D parameters are reshaped to 2D, set max_preconditioner_dim=math.inf and distributed_config.target_parameter_dimensionality=2."
            )
        super().__init__(block_list)
        self._preconditioner_config = preconditioner_config
        self._foreach_newton_schulz: Callable[..., tuple[Tensor, ...]] = (
            torch.compile(
                _foreach_newton_schulz,
                **asdict(shampoo_pt2_compile_config),
            )
            if shampoo_pt2_compile_config is not None
            else _foreach_newton_schulz
        )

    @profile_decorator
    def update_preconditioners(
        self,
        masked_grad_list: tuple[Tensor, ...],
        step: Tensor,
        perform_amortized_computation: bool = False,
    ) -> None:
        return

    @profile_decorator
    def precondition(self, masked_grad_list: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        config = self._preconditioner_config.orthogonalization_config
        if (
            masked_grad_list
            and isinstance(config, NewtonSchulzOrthogonalizationConfig)
            and all(grad.dtype is torch.bfloat16 for grad in masked_grad_list)
        ):
            return self._foreach_newton_schulz(
                masked_grad_list,
                config.coefficients,
                config.num_iterations,
                tuple(
                    config.scale_by_dims_fn(grad.shape[1], grad.shape[0])
                    for grad in masked_grad_list
                ),
            )
        return tuple(
            # An error will be raised when grad is not 2D.
            matrix_orthogonalization(
                grad,
                orthogonalization_config=self._preconditioner_config.orthogonalization_config,
            )
            for grad in masked_grad_list
        )

    def compress_preconditioner_list(
        self, local_grad_selector: tuple[bool, ...]
    ) -> None:
        return
