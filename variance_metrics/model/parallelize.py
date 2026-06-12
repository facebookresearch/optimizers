# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply DDP / FSDP / TP to a Llama3 model."""

import torch
import torch.nn as nn
from components.configs import ParallelismConfig, TrainingConfig
from components.parallel_dims import ParallelDims
from components.utils import logger
from model.llama3 import Llama3Model
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

# Mixed-precision policy used by FSDP / replicate: bfloat16 for forward,
# float32 for gradient reductions.
_PARAM_DTYPE = torch.bfloat16
_REDUCE_DTYPE = torch.float32


def parallelize_llama(
    model: Llama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
):
    """Apply TP and DDP/FSDP to the model.

    The model preferably should be on meta device.
    """
    del parallelism  # currently unused; kept for API parity with other models
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"seq_len {training.seq_len} must be divisible by TP degree "
        f"({parallel_dims.tp})"
    )

    if parallel_dims.tp_enabled:
        apply_tp(model, parallel_dims.get_mesh("tp"))

    if parallel_dims.fsdp_enabled:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        apply_fsdp(model, parallel_dims.get_mesh(names))
        logger.info(
            "Applied %s", "HSDP" if parallel_dims.dp_replicate_enabled else "FSDP"
        )
    elif parallel_dims.dp_replicate_enabled:
        apply_replicate(model, parallel_dims.get_mesh("dp_replicate"))

    return model


def apply_tp(model: nn.Module, tp_mesh: DeviceMesh) -> None:
    """Apply tensor + sequence parallelism (loss parallel is always on)."""
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
        },
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }
        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism")


def _disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division.

    The training loop scales the loss by global token count itself.
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)


def apply_fsdp(model: nn.Module, dp_mesh: DeviceMesh) -> None:
    """Apply FSDP2 (or HSDP if `dp_mesh` has a replicate dim) to the model.

    Reshards parameters after forward for ``tok_embeddings`` + every
    transformer block (memory-optimal: free, re-all-gather in backward).
    Skips resharding for the final ``norm`` / ``output`` layers — FSDP
    prefetches them in backward anyway, so resharding would only buy an
    extra all-gather.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=_PARAM_DTYPE, reduce_dtype=_REDUCE_DTYPE
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # pyrefly: ignore [no-matching-overload]
    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    # pyrefly: ignore [missing-attribute]
    for _, transformer_block in model.layers.items():
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=True)
    # pyrefly: ignore [no-matching-overload]
    fully_shard([model.norm, model.output], **fsdp_config, reshard_after_forward=False)
    fully_shard(model, **fsdp_config)
    _disable_fsdp_gradient_division(model)


def apply_replicate(model: nn.Module, dp_mesh: DeviceMesh) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=_PARAM_DTYPE, reduce_dtype=_REDUCE_DTYPE
    )
    replicate_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    # pyrefly: ignore [no-matching-overload, invalid-param-spec]
    replicate(model.tok_embeddings, **replicate_config)
    # pyrefly: ignore [missing-attribute]
    for _, transformer_block in model.layers.items():
        # pyrefly: ignore [invalid-param-spec]
        replicate(transformer_block, **replicate_config)
    # pyrefly: ignore [no-matching-overload, invalid-param-spec]
    replicate([model.norm, model.output], **replicate_config)
    # pyrefly: ignore [invalid-param-spec]
    replicate(model, **replicate_config)
    _disable_fsdp_gradient_division(model)
    logger.info("Applied DDP / replicate")
