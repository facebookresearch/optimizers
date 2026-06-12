# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

__all__ = ["Muon"]


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int) -> torch.Tensor:
    """Quintic Newton-Schulz iteration that drives the singular values of
    ``g`` to ~1. See https://kellerjordan.github.io/posts/muon/ for the
    coefficients (a, b, c) = (3.4445, -4.7750, 2.0315), tuned so that the
    iteration converges quickly for random matrices.
    """
    assert g.ndim == 2, "Newton-Schulz expects a matrix"
    a, b, c = 3.4445, -4.7750, 2.0315
    # Run in bfloat16 for speed; the iteration is numerically stable here.
    x = g.to(torch.bfloat16)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: MomentUm Orthogonalized by Newton-schulz.

    Update rule per 2D parameter:
        m_t  = momentum * m_{t-1} + g_t
        u_t  = m_t + momentum * (m_t - m_{t-1})       # if nesterov
             = m_t                                     # otherwise
        o_t  = newton_schulz(u_t)                     # orthogonalized update
        p    = (1 - lr * weight_decay) * p - lr * o_t

    Only matrix-shaped parameters (ndim == 2) are orthogonalized; everything
    else falls back to plain SGD with momentum on the same buffer.
    Embedding and final-projection layers should typically be optimized with
    AdamW in a separate param group.

    Reference: Keller Jordan et al.,
    https://kellerjordan.github.io/posts/muon/

    Args:
        params: Iterable of parameters / param groups.
        lr: Learning rate.
        momentum: Momentum coefficient.
        nesterov: Use Nesterov-style lookahead momentum.
        ns_steps: Number of Newton-Schulz iterations (5 is the standard).
        weight_decay: Decoupled weight decay (AdamW-style).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(p.grad)
                update = p.grad.add(buf, alpha=momentum) if nesterov else buf

                if p.ndim == 2:
                    # Orthogonalize the update; preserves spectral scale.
                    update = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Scale so the per-parameter step size matches the matrix
                    # shape (shorter side of the matrix). Standard in Muon.
                    update = update * max(1, update.size(0) / update.size(1)) ** 0.5

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss

    def get_hyperparams(self) -> dict[str, float]:
        group = self.param_groups[0]
        return {
            "lr": group["lr"],
            "momentum": group["momentum"],
            "weight_decay": group["weight_decay"],
        }
