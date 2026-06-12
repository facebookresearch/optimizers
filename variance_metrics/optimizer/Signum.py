# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

__all__ = ["Signum"]


class Signum(torch.optim.Optimizer):
    """Implements the Signum algorithm (SignSGD with momentum).

    The update rule is:
        When beta > 0 (Signum with momentum):
            m_t = beta * m_{t-1} + (1-beta) * gradient
            param = param - lr * sign(m_t)
        When beta = 0 (vanilla SignSGD):
            param = param - lr * sign(gradient)

    Weight decay is applied as decoupled weight decay (similar to AdamW):
        param = (1 - lr * weight_decay) * param - lr * sign(m)

    Reference: "SIGNSGD: Compressed Gradients for SGD with Majority Vote"
               Bernstein et al., https://arxiv.org/abs/1802.04434

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-1)
        weight_decay: Weight decay coefficient for decoupled weight decay (default: 0.1)
        beta: Momentum coefficient for exponential moving average (default: 0.9)
    """

    def __init__(
        self, params, lr: float = 1e-1, weight_decay: float = 0.1, beta: float = 0.9
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "beta": beta}
        self.original_beta = beta
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients")

                # Signum momentum
                if beta > 0:
                    state = self.state[p]
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p.grad)

                    m = beta * state["m"] + (1 - beta) * p.grad
                    state["m"] = m
                else:
                    m = p.grad

                # Weight Decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # Update rule: p_t+1 = p_t - lr * sign(m_t)
                p.add_(torch.sign(m), alpha=-lr)

        return loss

    def get_hyperparams(self) -> dict[str, float]:
        """Return a dictionary of optimizer hyperparameters.

        Returns:
            Dictionary containing lr, beta, and weight_decay.
        """
        group = self.param_groups[0]
        return {
            "lr": group["lr"],
            "beta": group["beta"],
            "weight_decay": group["weight_decay"],
        }
