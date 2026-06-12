# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

try:
    import pkg_resources

    os.environ["_WANDB_CORE_PATH"] = pkg_resources.resource_filename(
        "wandb", "wandb-core"
    )
except Exception:
    pass

# Resolve asset paths relative to this file so the run works regardless of
# the worker's cwd (torchrun spawns workers from the caller's cwd).
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
from components.configs import (
    AdaptiveBatchSizeConfig,
    DebugConfig,
    HFTextDataLoaderConfig,
    LoggerConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    ParallelismConfig,
    TrainerConfig,
    TrainingConfig,
    ValidatorConfig,
    VarianceMetricsConfig,
)
from components.utils import init_logger, logger
from model import model_registry
from optimizer.Muon import Muon  # noqa: F401
from optimizer.Signum import Signum
from torch.optim import AdamW  # noqa: F401


def main() -> None:
    init_logger()

    # Default: bundled 2009-vocab debug tokenizer + c4_test fixture; quickstart
    # works without any downloads. Bump vocab_size and swap paths below to use
    # a real tokenizer (e.g. Llama-3.1-8B — see README "Real-model runs").
    model_spec = model_registry("debugmodel")

    # Edit this config to change the run.
    config = TrainerConfig(
        hf_assets_path=os.path.join(REPO_DIR, "tests/assets/tokenizer/debug"),
        model_spec=model_spec,
        optimizer=OptimizerConfig(
            optimizer_cls=Signum,
            optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.1, "beta": 0.9},
        ),
        dataloader=HFTextDataLoaderConfig(
            dataset="c4_test",
            dataset_path=os.path.join(REPO_DIR, "tests/assets/c4_test"),
        ),
        metrics=LoggerConfig(log_freq=1, enable_wandb=True),
        validator=ValidatorConfig(freq=100, steps=500),
        debug=DebugConfig(deterministic=True, seed=1024),
        variance_metrics=VarianceMetricsConfig(
            enable=True,
            freq=1,
            spectral_variance=False,
        ),
        adaptive_batch_size=AdaptiveBatchSizeConfig(
            enable=True,
            batch_size_method="l1_gns_batch_size",
        ),
        lr_scheduler=LRSchedulerConfig(
            warmup_ratio=0.15,
            min_lr_factor=0.0,
            scale_with_batch_size=True,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            global_batch_size=64,
            seq_len=2048,
            steps=100,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=8,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
        ),
    )

    if not os.path.exists(config.hf_assets_path):
        logger.warning(
            f"HF assets path {config.hf_assets_path} does not exist! "
            "See the 'Tokenizer' section of README.md."
        )

    from trainer import Trainer

    trainer: Trainer | None = None

    try:
        trainer = config.build()
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
