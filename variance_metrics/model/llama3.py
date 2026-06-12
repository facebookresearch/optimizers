# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Llama3 model: abstract bases (``TransformerBlock`` / ``Decoder``) followed
by the SwiGLU feed-forward block and the concrete Llama3 specializations
(``Llama3TransformerBlock`` / ``Llama3Model``)."""

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from components.utils import logger
from model.attention import Attention, RoPE
from model.module import Embedding, Linear, Module, ModuleDict, RMSNorm


# ---------------------------------------------------------------------------
# Feed-forward block
# ---------------------------------------------------------------------------


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Llama3 SwiGLU hidden-dim calculation: 8*dim/3 (× optional multiplier),
    rounded up to ``multiple_of``."""
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.w1 = Linear(
            config=Linear.Config(), in_features=dim, out_features=config.hidden_dim
        )
        self.w2 = Linear(
            config=Linear.Config(), in_features=config.hidden_dim, out_features=dim
        )
        self.w3 = Linear(
            config=Linear.Config(), in_features=dim, out_features=config.hidden_dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float = 0.02, **kwargs):
        self.w1.init_weights()
        self.w2.init_weights(init_std=init_std)
        self.w3.init_weights(init_std=init_std)


# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------


class TransformerBlock(Module):
    """Base class for language-model transformer blocks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Attention.Config  # required, no default
        feed_forward: FeedForward.Config
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config


class Decoder(Module):
    """Base class for autoregressive decoder-only language models."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        n_layers: int
        vocab_size: int
        output: Linear.Config
        tok_embeddings: Embedding.Config
        norm: RMSNorm.Config
        rope: RoPE.Config
        layer: TransformerBlock.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = config.tok_embeddings.build(
            num_embeddings=config.vocab_size, embedding_dim=config.dim
        )
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = ModuleDict()
        for layer_id in range(config.n_layers):
            self.layers[str(layer_id)] = config.layer.build(
                layer_id=layer_id, dim=config.dim, n_layers=config.n_layers
            )

        self.norm = config.norm.build(normalized_shape=config.dim)
        self.output = config.output.build(
            in_features=config.dim, out_features=config.vocab_size
        )

    def init_weights(self, **kwargs):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        buffer_device = buffer_device or self.freqs_cis.device
        self.rope.init_weights(buffer_device=buffer_device)
        self.freqs_cis = self.rope.cache
        self.tok_embeddings.init_weights()
        for layer in self.layers.values():
            # pyrefly: ignore [not-callable]
            layer.init_weights(buffer_device=buffer_device)
        self.norm.init_weights()

        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, positions)
        return self.output(self.norm(h))


# ---------------------------------------------------------------------------
# Llama3 specializations
# ---------------------------------------------------------------------------


class Llama3TransformerBlock(TransformerBlock):
    """Llama3 transformer block (RMSNorm + MHA + SwiGLU FFN)."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        depth_init: bool = True

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.attention = config.attention.build(dim=dim)
        self.feed_forward = config.feed_forward.build(dim=dim)
        self.attention_norm = config.attention_norm.build(normalized_shape=dim)
        self.ffn_norm = config.ffn_norm.build(normalized_shape=dim)

        if config.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        h = x + self.attention(self.attention_norm(x), freqs_cis, positions)
        return h + self.feed_forward(self.ffn_norm(h))

    def init_weights(self, **kwargs):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.init_weights()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Llama3Model(Decoder):
    """Llama3 model."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 4096
        n_layers: int = 32
        vocab_size: int = 128256
        layer: TransformerBlock.Config

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum "
                    f"{self.rope.max_seq_len}."
                )
            self.rope = dataclasses.replace(self.rope, max_seq_len=seq_len)

            tp = parallelism.tensor_parallel_degree
            if tp > 1 and self.layer.attention.n_heads % tp != 0:
                raise ValueError(
                    f"tensor_parallel_degree ({tp}) must divide "
                    f"n_heads ({self.layer.attention.n_heads})."
                )
