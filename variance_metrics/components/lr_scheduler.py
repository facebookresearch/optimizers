# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Token-budget-based warmup-stable-cosine-decay learning-rate scheduler.

The schedule is parameterised by *tokens seen* rather than step count, so
it works uniformly for fixed and adaptive batch sizes: when the batch
size grows, more tokens are processed per step and ``tokens_seen /
total_tokens`` progresses correspondingly faster.
"""

import math

from components.configs import LRSchedulerConfig
from components.configurable import Configurable
from optimizer import OptimizersContainer

__all__ = ["LRSchedulersContainer"]


class LRSchedulersContainer(Configurable):
    """Token-budget-based LR scheduler shared across all optimizers.

    Three phases over the token budget:
    - Warmup: LR linearly ramps from 0 to base_lr over
        ``warmup_ratio * total_tokens`` tokens.
    - Stable: LR == base_lr for ``(1 - warmup_ratio - decay_ratio) *
        total_tokens`` tokens (skipped when ``decay_ratio`` is None — decay
        then starts immediately after warmup).
    - Decay: cosine decay from base_lr to ``min_lr_factor * base_lr``
        over ``decay_ratio * total_tokens`` tokens.

    When ``scale_with_batch_size`` is set, the LR is additionally scaled
    by ``sqrt(current_global_batch / initial_global_batch)`` — useful for
    the adaptive batch-size flow.
    """

    Config = LRSchedulerConfig

    def __init__(
        self,
        *,
        optimizers: OptimizersContainer,
        config: LRSchedulerConfig,
        total_tokens: int,
    ) -> None:
        assert len(optimizers) > 0, "at least one optimizer required"
        assert total_tokens > 0, "total_tokens must be > 0"
        self.optimizers = optimizers
        self.config = config
        self.total_tokens = total_tokens

        # Capture the base LR for every param group exactly once.
        self._base_lrs: list[float] = [
            pg["lr"] for opt in optimizers for pg in opt.param_groups
        ]
        self._last_lrs: list[float] = list(self._base_lrs)

    def update(self, *, tokens_seen: int, batch_scale: float = 1.0) -> None:
        """Set the LR for every optimizer param group based on tokens seen.

        Call this once per training step before ``optimizer.step()``.

        Args:
            tokens_seen: Cumulative global tokens processed so far.
            batch_scale: ``current_global_batch / initial_global_batch`` —
                used only when ``scale_with_batch_size`` is True.
        """
        decay_factor = self._decay_factor(tokens_seen)
        scale = math.sqrt(batch_scale) if self.config.scale_with_batch_size else 1.0
        i = 0
        for opt in self.optimizers:
            for pg in opt.param_groups:
                lr = self._base_lrs[i] * scale * decay_factor
                pg["lr"] = lr
                self._last_lrs[i] = lr
                i += 1

    def get_last_lr(self) -> list[float]:
        return list(self._last_lrs)

    def _decay_factor(self, tokens_seen: int) -> float:
        """Warmup-stable-decay factor as a function of tokens seen."""
        cfg = self.config
        total = self.total_tokens
        warmup = round(cfg.warmup_ratio * total)
        decay = (
            round(cfg.decay_ratio * total)
            if cfg.decay_ratio is not None
            else max(total - warmup, 0)
        )
        if warmup + decay > total:
            decay = max(total - warmup, 0)
        stable = total - warmup - decay

        if tokens_seen < warmup:
            return tokens_seen / warmup if warmup > 0 else 1.0
        if tokens_seen < warmup + stable:
            return 1.0
        if decay <= 0:
            return cfg.min_lr_factor
        progress = min((tokens_seen - warmup - stable) / decay, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_factor + (1.0 - cfg.min_lr_factor) * cosine_factor
