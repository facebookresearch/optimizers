# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from components.configs import ModelSpec
from model.attention import Attention, RoPE
from model.llama3 import (
    compute_ffn_hidden_dim,
    FeedForward,
    Llama3Model,
    Llama3TransformerBlock,
)
from model.module import Embedding, Linear, RMSNorm
from model.parallelize import parallelize_llama


def _llama3_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int = 128256,
    multiple_of: int = 256,
) -> "Llama3Model.Config":
    return Llama3Model.Config(
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(dim, multiple_of=multiple_of),
            ),
            attention=Attention.Config(n_heads=n_heads),
        ),
        rope=RoPE.Config(
            dim=dim // n_heads,
            max_seq_len=131072,
            theta=500000,
            scaling="llama",
        ),
    )


llama3_configs = {
    "debugmodel": _llama3_config(dim=256, n_layers=6, n_heads=16, vocab_size=2048),
    "160M": _llama3_config(dim=768, n_layers=18, n_heads=12),
    "300M": _llama3_config(dim=1024, n_layers=20, n_heads=16),
    "660M": _llama3_config(dim=1408, n_layers=24, n_heads=11),
    "1B": _llama3_config(dim=1792, n_layers=23, n_heads=14),
    "3B": _llama3_config(dim=3072, n_layers=28, n_heads=16),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="llama3",
        flavor=flavor,
        model=llama3_configs[flavor],
        parallelize_fn=parallelize_llama,
    )
