# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from components.configs import AdaptiveBatchSizeConfig, BatchSizeMethods
from components.utils import logger
from variance_metrics.types import AggregatedVarianceMetrics, MeanMetrics


class BatchSizeManager:
    """Changes the global batch size during training, driven by GNS.

    The global batch size grows by first increasing the local batch size
    (up to ``largest_local_batch_size``) and then gradient accumulation.
    The batch size is monotonically non-decreasing.
    """

    def __init__(
        self,
        adaptive_batch_config: AdaptiveBatchSizeConfig,
        local_batch_size: int,
        dp_degree: int,
    ) -> None:
        self.batch_size_change_enabled = adaptive_batch_config.enable
        if not self.batch_size_change_enabled:
            return

        self.largest_global_batch_size: int | None = (
            adaptive_batch_config.largest_global_batch_size
        )
        self.largest_local_batch_size: int = (
            adaptive_batch_config.largest_local_batch_size
        )

        method = adaptive_batch_config.batch_size_method
        if method is None:
            raise ValueError(
                "batch_size_method must be specified when adaptive batch is enabled"
            )
        available = [m.value for m in BatchSizeMethods]
        if method not in available:
            raise ValueError(
                f"Invalid batch_size_method: {method}. Must be one of {available}"
            )
        self.batch_size_method = BatchSizeMethods(method)

        self.gns_batch_size_constant = adaptive_batch_config.gns_batch_size_constant
        self.var_ema_constant = adaptive_batch_config.var_ema_constant
        self.gradient_ema_constant = adaptive_batch_config.gradient_ema_constant
        self.batch_size_update_freq_gns: int = (
            adaptive_batch_config.batch_size_update_freq_gns
        )
        self.initial_constant_batch_steps: int = (
            adaptive_batch_config.initial_constant_batch_steps
        )

        logger.info(
            f"Adaptive batch size enabled: method={self.batch_size_method.value}, "
            f"largest_global_batch_size={self.largest_global_batch_size}, "
            f"largest_local_batch_size={self.largest_local_batch_size}, "
            f"gns_batch_size_constant={self.gns_batch_size_constant}, "
            f"var_ema_constant={self.var_ema_constant}, "
            f"gradient_ema_constant={self.gradient_ema_constant}, "
            f"batch_size_update_freq_gns={self.batch_size_update_freq_gns}, "
            f"initial_constant_batch_steps={self.initial_constant_batch_steps}"
        )

        # EMAs of variance and gradient magnitude.
        self.var_ema: float = 0.0
        self.mean_gradient_ema: float = 0.0

        # Fallback batch size used while the GNS EMAs haven't warmed up.
        self.initial_global_batch_size: int = local_batch_size * dp_degree

    def update_parameters(
        self,
        global_variance_metrics: AggregatedVarianceMetrics,
        global_mean_metrics: MeanMetrics,
    ) -> None | dict[str, float]:
        """Update GNS EMAs and report current adaptive-batch metrics."""
        if not self.batch_size_change_enabled:
            return None

        match self.batch_size_method:
            case BatchSizeMethods.L2_GNS_BATCH_SIZE:
                self.var_ema = (
                    self.var_ema_constant * self.var_ema
                    + (1 - self.var_ema_constant)
                    * global_variance_metrics.variance_metrics_example.variance
                )
                self.mean_gradient_ema = (
                    self.gradient_ema_constant * self.mean_gradient_ema
                    + (1 - self.gradient_ema_constant) * global_mean_metrics.mean_l2_sq
                )
            case BatchSizeMethods.L1_GNS_BATCH_SIZE:
                self.var_ema = (
                    self.var_ema_constant * self.var_ema
                    + (1 - self.var_ema_constant)
                    * global_variance_metrics.variance_metrics_example.std_l1**2
                )
                self.mean_gradient_ema = (
                    self.gradient_ema_constant * self.mean_gradient_ema
                    + (1 - self.gradient_ema_constant) * global_mean_metrics.mean_l1**2
                )
            case BatchSizeMethods.NUCLEAR_GNS_BATCH_SIZE:
                self.var_ema = (
                    self.var_ema_constant * self.var_ema
                    + (1 - self.var_ema_constant)
                    * global_variance_metrics.variance_metrics_example.nuclear_norm**2
                )
                self.mean_gradient_ema = (
                    self.gradient_ema_constant * self.mean_gradient_ema
                    + (1 - self.gradient_ema_constant) * global_mean_metrics.nuc_norm**2
                )
            case _:
                raise ValueError(
                    f"Invalid batch_size_method: {self.batch_size_method}. "
                    f"Must be one of {[m.value for m in BatchSizeMethods]}"
                )

        return {
            "adaptive_batch_metrics/var_ema": self.var_ema,
            "adaptive_batch_metrics/mean_gradient_ema": self.mean_gradient_ema,
            "adaptive_batch_metrics/gns_ema": (
                self.var_ema / self.mean_gradient_ema
                if self.mean_gradient_ema > 0
                else 0.0
            ),
            "adaptive_batch_metrics/suggested_batch_size": self._gns_based_global_batch_size(),
        }

    def update_global_batch_size(
        self,
        train_step: int,
        global_batch_size: int,
        local_batch_size: int,
        gradient_accumulation_steps: int,
        dp_degree: int,
    ) -> tuple[int, int, int]:
        """Returns (new_accumulation_steps, new_local_batch_size, new_global_batch_size)."""
        if not self.batch_size_change_enabled:
            return gradient_accumulation_steps, local_batch_size, global_batch_size

        new_global_batch_size = global_batch_size
        if (
            train_step > self.initial_constant_batch_steps
            and train_step % self.batch_size_update_freq_gns == 0
        ):
            suggested_batch_size = self._gns_based_global_batch_size()
            # Batch size is monotonically non-decreasing.
            new_global_batch_size = max(suggested_batch_size, global_batch_size)
            if self.largest_global_batch_size is not None:
                new_global_batch_size = min(
                    new_global_batch_size, self.largest_global_batch_size
                )
            if new_global_batch_size < local_batch_size * dp_degree:
                new_global_batch_size = local_batch_size * dp_degree

        new_accumulation_steps, new_local_batch_size = (
            self._determine_local_batch_size_and_accumulation_steps(
                new_global_batch_size=new_global_batch_size,
                local_batch_size=local_batch_size,
                dp_degree=dp_degree,
            )
        )
        return new_accumulation_steps, new_local_batch_size, new_global_batch_size

    def _determine_local_batch_size_and_accumulation_steps(
        self,
        new_global_batch_size: int,
        local_batch_size: int,
        dp_degree: int,
    ) -> tuple[int, int]:
        """Grow local batch size up to ``largest_local_batch_size``, then
        use gradient accumulation to reach the target global batch size."""
        new_local_batch_size = min(
            self.largest_local_batch_size,
            int(new_global_batch_size / dp_degree),
        )
        new_local_batch_size = max(new_local_batch_size, local_batch_size)
        if new_local_batch_size % local_batch_size != 0:
            new_local_batch_size = (
                new_local_batch_size // local_batch_size
            ) * local_batch_size
        new_accumulation_steps = new_global_batch_size // (
            new_local_batch_size * dp_degree
        )
        return new_accumulation_steps, new_local_batch_size

    def _gns_based_global_batch_size(self) -> int:
        """Returns the new global batch size based on GNS."""
        denominator = self.mean_gradient_ema * self.gns_batch_size_constant**2
        if denominator <= 0:
            # EMAs not warmed up yet — fall back to the initial global batch
            # size so logged metrics stay sensible instead of showing 0.
            return self.initial_global_batch_size
        return int(self.var_ema / denominator)
