# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-head causal attention with SDPA + complex Rotary Position Embedding.

RoPE (``RoPE`` + ``apply_rotary_emb_complex``) lives here because it is only
consumed by ``Attention``. ``_LocalMapAttention`` / ``_SDPAWrapper`` are
private helpers that wrap ``F.scaled_dot_product_attention`` so it accepts
DTensor inputs under tensor parallelism.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from model.module import Linear, Module
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import local_map
from torch.nn.attention import sdpa_kernel, SDPBackend


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------


class RoPE(Module):
    """Complex-exponential Rotary Position Embedding (Llama3 style)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_seq_len: int
        theta: float = 10000.0
        scaling: Literal["none", "llama"] = "none"
        # llama scaling params
        scaling_factor: float = 8.0
        low_freq_factor: float = 1.0
        high_freq_factor: float = 4.0
        original_max_position_embeddings: int = 8192

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.cache: torch.Tensor = self._precompute()

    def _precompute(self) -> torch.Tensor:
        cfg = self.config
        dim, end, theta = cfg.dim, cfg.max_seq_len, cfg.theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        if cfg.scaling == "llama":
            wavelen = 2 * math.pi / freqs
            high_freq_wavelen = (
                cfg.original_max_position_embeddings / cfg.high_freq_factor
            )
            low_freq_wavelen = (
                cfg.original_max_position_embeddings / cfg.low_freq_factor
            )
            freqs = torch.where(
                wavelen > low_freq_wavelen, freqs / cfg.scaling_factor, freqs
            )
            smooth_factor = (
                cfg.original_max_position_embeddings / wavelen - cfg.low_freq_factor
            ) / (cfg.high_freq_factor - cfg.low_freq_factor)
            smoothed_freqs = (
                1 - smooth_factor
            ) * freqs / cfg.scaling_factor + smooth_factor * freqs
            is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64

    def init_weights(self, **kwargs) -> None:
        buffer_device = kwargs.get("buffer_device")
        if buffer_device is not None:
            with torch.device(buffer_device):
                self.cache = self._precompute()
        else:
            self.cache = self._precompute()


def _reshape_for_broadcast_complex(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    if positions.size(0) == 1:
        freqs_cis = freqs_cis[positions.squeeze(0)]
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    freqs_cis_expanded = freqs_cis[None, :, None, :].expand(x.shape[0], -1, -1, -1)
    return torch.gather(
        freqs_cis_expanded,
        dim=1,
        index=positions.view(x.shape[0], seqlen, 1, 1).expand(
            x.shape[0], seqlen, 1, freqs_cis_expanded.shape[-1]
        ),
    )


def _maybe_wrap_positions(
    positions: torch.Tensor | None, x: torch.Tensor
) -> torch.Tensor | None:
    """Wrap ``positions`` as a DTensor matching ``x`` for TP gather ops."""
    if (
        positions is not None
        and isinstance(x, DTensor)
        and not isinstance(positions, DTensor)
    ):
        ndim = positions.ndim
        placements = tuple(
            p if not isinstance(p, Shard) or p.dim < ndim else Replicate()
            for p in x.placements
        )
        positions = DTensor.from_local(
            positions, x.device_mesh, placements, run_check=False
        )
    return positions


def apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply complex-format RoPE to query and key tensors."""
    positions = _maybe_wrap_positions(positions, xq)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast_complex(freqs_cis, xq_, positions)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class _LocalMapAttention(Module):
    """Inner-attention base supporting DTensor inputs (TP).

    When ``q``, ``k``, ``v`` arrive as DTensors (e.g. with TP and
    ``use_local_output=False``), wraps the call with ``local_map`` so the
    forward sees plain tensors. Plain tensors fall through unmodified.
    """

    def __init__(self) -> None:
        super().__init__()
        self._local_map_fn: Callable | None = None

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(q, DTensor):
            assert isinstance(k, DTensor) and isinstance(v, DTensor)
            assert q.placements == k.placements == v.placements
            for p in q.placements:
                assert p == Shard(1), (
                    f"_LocalMapAttention requires Shard(1) placements, got {p}"
                )
            if self._local_map_fn is None:
                self._local_map_fn = local_map(
                    super().__call__,
                    in_placements=(q.placements, k.placements, v.placements),
                    out_placements=(q.placements,),
                    in_grad_placements=(q.placements, k.placements, v.placements),
                    device_mesh=q.device_mesh,
                )
            return self._local_map_fn(q, k, v, **kwargs)
        return super().__call__(q, k, v, **kwargs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class _SDPAWrapper(_LocalMapAttention):
    """``F.scaled_dot_product_attention`` wrapped as an ``nn.Module``."""

    sdpa_backends: ClassVar[list[SDPBackend]] = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class Attention(Module):
    """Multi-head causal attention with SDPA + complex RoPE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = dim // config.n_heads

        self.wq = Linear(config=Linear.Config(), in_features=dim, out_features=dim)
        self.wk = Linear(config=Linear.Config(), in_features=dim, out_features=dim)
        self.wv = Linear(config=Linear.Config(), in_features=dim, out_features=dim)
        self.wo = Linear(config=Linear.Config(), in_features=dim, out_features=dim)

        self.inner_attention = _SDPAWrapper()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 to infer local heads (TP may have sharded after the linear ops).
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        xq, xk = apply_rotary_emb_complex(
            xq, xk, freqs_cis=rope_cache, positions=positions
        )

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = self.inner_attention(xq, xk, xv).transpose(1, 2).contiguous()
        return self.wo(output.view(bs, seqlen, -1))

    def init_weights(self, init_std: float = 0.02, **kwargs) -> None:
        for linear in (self.wq, self.wk, self.wv):
            linear.init_weights()
        self.wo.init_weights(init_std=init_std)
