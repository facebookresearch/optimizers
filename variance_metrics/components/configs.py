# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""All Trainer-facing configuration dataclasses.

Each Configurable component binds its `.Config` attribute to one of these
classes (e.g., `class HuggingFaceTokenizer(Configurable): Config = TokenizerConfig`),
so existing access patterns like `Trainer.Config(...)` continue to work.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from components.configurable import Configurable
from torch.optim import Optimizer


@dataclass
class ModelSpec:
    """Per-model bundle. Architecture config + callables."""

    name: str
    flavor: str
    model: Configurable.Config
    parallelize_fn: Callable


# ---------------------------------------------------------------------------
# Plain (non-Configurable) configs
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class TrainingConfig:
    local_batch_size: int = 8
    """Local batch size (i.e., per-device batch size)."""

    global_batch_size: int = 64
    """Global batch size (must be > 0)."""

    seq_len: int = 2048
    """Sequence length."""

    max_norm: float | int = 1.0
    """Max norm for gradient clipping."""

    steps: int = 10000
    """How many train steps to run."""

    n_token_limit: int = -1
    """Limit the number of tokens to be processed; -1 means no token-based limit.
    Only consulted when adaptive batch sizing is enabled."""


@dataclass(kw_only=True, slots=True)
class ParallelismConfig:
    data_parallel_replicate_degree: int = 1
    """Degree of data parallelism for weight replication.
    >1 → DDP (or HSDP if also sharding). 1 means disabled."""

    data_parallel_shard_degree: int = -1
    """Degree of data parallelism for weight sharding.
    >1 → FSDP (or HSDP if also replicating). -1 means use leftover ranks
    after DP_REPLICATE / TP. 1 means disabled."""

    tensor_parallel_degree: int = 1
    """Tensor Parallelism degree. 1 means disabled."""


@dataclass(kw_only=True, slots=True)
class DebugConfig:
    seed: int | None = None
    """Choose the base RNG seed used for training."""

    deterministic: bool = False
    """Use deterministic algorithms wherever possible, may be slower."""


@dataclass
class VarianceMetricsConfig:
    """Configuration for gradient variance metrics tracking."""

    enable: bool = False
    """Enable gradient variance metric tracking."""

    freq: int = 100
    """Frequency of computing variance metrics (in training steps).

    """

    spectral_variance: bool = False
    """Use the nuclear-norm spectral GNS path instead of the elementwise
    L1/L2 path. Only 2D parameters contribute (1D parameters are skipped).
    Pure DDP only — errors out under FSDP / HSDP / TP.
    """


class BatchSizeMethods(Enum):
    """GNS-based strategies for changing the batch size."""

    L1_GNS_BATCH_SIZE = "l1_gns_batch_size"
    L2_GNS_BATCH_SIZE = "l2_gns_batch_size"
    NUCLEAR_GNS_BATCH_SIZE = "nuclear_gns_batch_size"


@dataclass
class AdaptiveBatchSizeConfig:
    """Adaptive Batch Size: Controls the batch size during the training loop.

    The batch size grows monotonically; the local batch size is always
    increased first (up to ``largest_local_batch_size``), then gradient
    accumulation is used to reach larger global batch sizes.
    """

    enable: bool = False
    """Enable adaptive changes in batch size"""

    largest_global_batch_size: int | None = None
    """Largest global batch size to use. No limit if set to None."""

    batch_size_method: str | None = None
    """GNS method to change the batch size. Must be one of:
    l1_gns_batch_size, l2_gns_batch_size."""

    largest_local_batch_size: int = 16
    """Largest local batch size to use"""

    gns_batch_size_constant: float = 0.6
    """Constant to scale GNS to batch size"""

    var_ema_constant: float = 0.9
    """EMA constant for the variance component of GNS."""

    gradient_ema_constant: float = 0.9
    """EMA constant for the gradient component of GNS."""

    batch_size_update_freq_gns: int = 100
    """Frequency of updating the batch size when using GNS"""

    initial_constant_batch_steps: int | None = None
    """Number of steps before starting to change the batch size
    (lr warmup length if None), must be > 0"""


# ---------------------------------------------------------------------------
# Configurable.Config subclasses (one per swappable component)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class TokenizerConfig(Configurable.Config):
    pass  # tokenizer_path is passed at build time


@dataclass(kw_only=True, slots=True)
class LoggerConfig(Configurable.Config):
    log_freq: int = 10
    """How often to log metrics, in steps."""

    enable_wandb: bool = False
    """Whether to log to Weights & Biases."""


@dataclass(kw_only=True, slots=True)
class OptimizerConfig(Configurable.Config):
    optimizer_cls: type[Optimizer] | None = None
    """The optimizer class to instantiate (e.g., ``torch.optim.AdamW``)."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to ``optimizer_cls`` (besides ``params``)."""

    def to_dict(self) -> dict[str, Any]:
        assert self.optimizer_cls is not None, "optimizer.optimizer_cls must be set"
        return {
            "optimizer_cls": f"{self.optimizer_cls.__module__}.{self.optimizer_cls.__qualname__}",
            **self.optimizer_kwargs,
        }


@dataclass(kw_only=True, slots=True)
class DataLoaderConfig(Configurable.Config):
    dataset: str = ""
    dataset_path: str | None = None


@dataclass(kw_only=True, slots=True)
class HFTextDataLoaderConfig(DataLoaderConfig):
    dataset: str = "c4_test"
    """Dataset to use"""

    infinite: bool = True
    """Whether to loop the dataset infinitely"""

    num_workers: int = 0
    """Number of worker processes for data loading."""

    persistent_workers: bool = False
    """Keep workers alive between epochs. Only valid when num_workers > 0."""

    pin_memory: bool = False
    """Copy tensors to CUDA pinned memory before returning them."""

    prefetch_factor: int | None = None
    """Number of batches loaded in advance by each worker. Only valid when
    num_workers > 0. Default is 2 when num_workers > 0, otherwise None."""


@dataclass(kw_only=True, slots=True)
class LRSchedulerConfig(Configurable.Config):
    warmup_ratio: float = 0.15
    """Fraction of the token budget used for warmup. Must be in [0, 1]."""

    decay_ratio: float | None = None
    """Fraction of the token budget used for decay. ``None`` means
    decay starts immediately after warmup (no stable plateau)."""

    min_lr_factor: float = 0.0
    """Minimum LR as a multiple of base LR. The decay range is
    ``[1, min_lr_factor]`` rather than ``[1, 0]``."""

    scale_with_batch_size: bool = False
    """If True, multiply the LR by ``sqrt(batch_scale)`` where
    ``batch_scale = current_global_batch / initial_global_batch``."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")
        decay_ratio = self.decay_ratio
        if decay_ratio is not None and not 0.0 <= decay_ratio <= 1.0:
            raise ValueError(f"decay_ratio must be in [0, 1], got {decay_ratio}")


@dataclass(kw_only=True, slots=True)
class ValidatorConfig(Configurable.Config):
    enable: bool = False
    """Enable validation."""

    freq: int = 10
    """Run validation every ``freq`` train steps (and at step 1)."""

    steps: int = 100
    """Number of validation batches per run. Must be > 0."""

    dataloader: DataLoaderConfig = field(
        default_factory=lambda: HFTextDataLoaderConfig(
            dataset="c4_validation",
            infinite=True,
        )
    )
    """DataLoader configuration for validation."""


@dataclass(kw_only=True, slots=True)
class TrainerConfig(Configurable.Config):
    model_spec: ModelSpec | None = None
    """Set programmatically by the model registry before Trainer construction."""

    hf_assets_path: str = "./tests/assets/tokenizer"
    """Path to local HF tokenizer assets."""

    metrics: LoggerConfig = field(default_factory=LoggerConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    variance_metrics: VarianceMetricsConfig = field(
        default_factory=VarianceMetricsConfig
    )
    adaptive_batch_size: AdaptiveBatchSizeConfig = field(
        default_factory=AdaptiveBatchSizeConfig
    )

    def __post_init__(self):
        if self.adaptive_batch_size.enable and not self.variance_metrics.enable:
            raise ValueError(
                "adaptive_batch_size.enable requires "
                "variance_metrics.enable=True (driven by GNS)."
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            if f.name == "model_spec":
                assert self.model_spec is not None
                d["model_spec"] = {
                    "name": self.model_spec.name,
                    "flavor": self.model_spec.flavor,
                }
                continue
            val = getattr(self, f.name)
            if hasattr(val, "to_dict"):
                d[f.name] = val.to_dict()
            elif dataclasses.is_dataclass(val):
                d[f.name] = asdict(val)
            else:
                d[f.name] = val
        return d
