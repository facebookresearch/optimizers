# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import datetime
from typing import Any

import torch
from components.configs import LoggerConfig
from components.configurable import Configurable
from components.utils import logger


class _NoOpLogger:
    """No-op logger used on non-rank-0 processes."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        pass

    def close(self) -> None:
        pass


class WandBLogger:
    """Weights & Biases backend."""

    def __init__(
        self,
        log_dir: str,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        import wandb

        self.wandb = wandb
        self.tag = tag
        os.makedirs(log_dir, exist_ok=True)
        self.wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "variance_metrics_oss"),
            name=os.getenv("WANDB_RUN_NAME", None),
            id=os.getenv("WANDB_RUN_ID", None),
            notes=os.getenv("WANDB_RUN_NOTES", None),
            tags=os.getenv("WANDB_RUN_TAGS", None),
            group=os.getenv("WANDB_RUN_GROUP", None),
            job_type=os.getenv("WANDB_RUN_JOB_TYPE", None),
            dir=log_dir,
            config=config_dict,
            # pyre-ignore[16]: wandb.Settings exists at runtime but is missing from the type stubs we ship with
            settings=wandb.Settings(x_disable_stats=True),
        )
        logger.info("WandB logging enabled")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.tag is not None:
            metrics = {f"{self.tag}/{k}": v for k, v in metrics.items()}
        self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.wandb.run is not None:
            self.wandb.finish()


class MetricsProcessor(Configurable):
    """Logs training metrics to stdout and (optionally) Weights & Biases.

    The focus is variance metrics + adaptive batch size; system / hardware
    metrics (memory, MFU, throughput) are intentionally not tracked.
    """

    Config = LoggerConfig

    config: LoggerConfig
    _backend: WandBLogger | _NoOpLogger

    def __init__(
        self,
        config: LoggerConfig,
        *,
        config_dict: dict[str, Any] | None = None,
        tag: str | None = None,
    ):
        self.config = config
        self._backend = self._build_backend(config_dict, tag)

    def _build_backend(
        self,
        config_dict: dict[str, Any] | None,
        tag: str | None,
    ) -> WandBLogger | _NoOpLogger:
        cfg = self.config
        if not cfg.enable_wandb:
            return _NoOpLogger()
        if torch.distributed.get_rank() != 0:
            return _NoOpLogger()
        log_dir = os.path.join(
            "./outputs",
            "wandb",
            datetime.now().strftime("%Y%m%d-%H%M"),
        )
        try:
            return WandBLogger(log_dir, config_dict=config_dict, tag=tag)
        except Exception as e:
            logger.error(f"Failed to create WandB logger: {e}")
            return _NoOpLogger()

    def should_log(self, step: int) -> bool:
        return step == 1 or step % self.config.log_freq == 0

    def log(self, step: int, metrics: dict[str, Any], verbose: bool = False) -> None:
        """Log a flat metric dict to WandB. Caller prepares the dict.

        When ``verbose=True``, also prints the metrics to stdout.
        """
        self._backend.log(metrics, step)
        if verbose:
            tokens = metrics.get("n_tokens_seen")
            prefix = f"step {step}"
            if tokens is not None:
                prefix += f"  tokens {tokens:,}"
            summary = "  ".join(
                f"{k}={v}" for k, v in metrics.items() if k != "n_tokens_seen"
            )
            logger.info(f"{prefix}  {summary}")

    def close(self) -> None:
        self._backend.close()
