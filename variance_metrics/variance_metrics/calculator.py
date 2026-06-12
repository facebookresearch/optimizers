# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single concrete variance-metrics calculator.

Supports DDP, FSDP, HSDP, and (optionally) Tensor Parallelism for the default
elementwise (L1/L2/L_inf) path, plus a spectral / nuclear-norm path for 2D
parameters under pure DDP.
"""

import dataclasses
from dataclasses import field

import torch
import torch.distributed as dist
from components.configs import VarianceMetricsConfig
from components.utils import logger

from .hooks import (
    BaseVarianceCaptureHook,
    FSDPVarianceCaptureHook,
    ReplicateVarianceCaptureHook,
)
from .types import AggregatedVarianceMetrics, MeanMetrics, VarianceMetrics


@dataclasses.dataclass
class VarianceMetricsCalculatorResult:
    """Aggregated global mean + variance metrics."""

    global_mean_metrics: MeanMetrics = field(default_factory=MeanMetrics)
    global_variance_metrics: AggregatedVarianceMetrics = field(
        default_factory=AggregatedVarianceMetrics
    )

    def clear(self) -> None:
        self.global_mean_metrics = MeanMetrics()
        self.global_variance_metrics = AggregatedVarianceMetrics()

    def metrics_to_logging_dict(self) -> dict[str, float]:
        """Flatten the global gradient variance + mean metrics for logging.

        Tabs:
            - ``variance_global_gradient``   variance of full-dataset gradient
            - ``variance_minibatch_gradient`` variance of a minibatch-sample gradient
            - ``variance_example_gradient``  variance of a per-example gradient
            - ``mean_gradient_norms``        mean gradient norms (L2^2, L1, L_inf)
        """
        metrics: dict[str, float] = {}
        agg = self.global_variance_metrics
        tabs = (
            ("variance_global_gradient", agg.variance_metrics_global),
            ("variance_minibatch_gradient", agg.variance_metrics_sample),
            ("variance_example_gradient", agg.variance_metrics_example),
        )
        for tab, m in tabs:
            metrics[f"{tab}/variance"] = m.variance
            metrics[f"{tab}/std_l1"] = m.std_l1
            metrics[f"{tab}/std_linf"] = m.std_linf
            metrics[f"{tab}/nuclear_norm"] = m.nuclear_norm
        mean = self.global_mean_metrics
        metrics["mean_gradient_norms/mean_l2_sq"] = mean.mean_l2_sq
        metrics["mean_gradient_norms/mean_l1"] = mean.mean_l1
        metrics["mean_gradient_norms/mean_linf"] = mean.mean_linf
        metrics["mean_gradient_norms/nuc_norm"] = mean.nuc_norm
        return metrics


@dataclasses.dataclass
class VarianceMetricsParallelismConfig:
    """Parallelism context for variance computation.

    Attributes:
        dp_degree: Data-parallel degree (replicate * shard).
        replicate_degree: Replication degree (DDP / HSDP outer dim).
        ddp_process_group: Process group for DDP / HSDP replicate all-reduce.
        fsdp_process_group: Process group for FSDP shard all-reduce.
        tp_fsdp_process_group: Combined TP + FSDP process group (TP enabled
            only when this is not None).
    """

    dp_degree: int = 1
    replicate_degree: int = 1

    ddp_process_group: dist.ProcessGroup | None = None
    fsdp_process_group: dist.ProcessGroup | None = None
    tp_fsdp_process_group: dist.ProcessGroup | None = None

    @property
    def ddp_enabled(self) -> bool:
        return self.replicate_degree > 1

    @property
    def fsdp_enabled(self) -> bool:
        return self.dp_degree // self.replicate_degree > 1

    @property
    def tp_enabled(self) -> bool:
        return self.tp_fsdp_process_group is not None


class VarianceMetricsCalculator:
    """Gradient-variance calculator across DDP / FSDP / HSDP / TP.

    Registers a per-module hook to capture per-sample gradient statistics,
    then aggregates them across distributed ranks to compute variance metrics
    at a configurable frequency.

    Two paths:
      * **Elementwise (default)**: per-parameter ``sum_g_sq`` across all
        parallelism dims.
      * **Spectral** (``spectral_variance=True``): nuclear-norm GNS for 2D
        parameters under pure DDP. 1D parameters are skipped.
    """

    def __init__(
        self,
        variance_config: VarianceMetricsConfig,
        parallelism_parameters: VarianceMetricsParallelismConfig,
        spectral_variance: bool = False,
    ) -> None:
        logger.info("Initializing variance metrics calculator.")
        self.freq: int = variance_config.freq
        self.parallelism_parameters = parallelism_parameters
        self.variance_metrics_result = VarianceMetricsCalculatorResult()
        self._hooks: dict[torch.nn.Module, BaseVarianceCaptureHook] = {}
        self._spectral_variance = spectral_variance

        # The FSDP wrapper combines ``norm`` and ``output`` into a single
        # module, so compute_variance_metrics needs both to extract the
        # right gradient slices.
        self._norm_module: torch.nn.Module | None = None
        self._output_module: torch.nn.Module | None = None
        self._spectral_param_infos_initialized: bool = False

    # ------------------------------------------------------------------ hooks

    def register_hook(self, module: torch.nn.Module, module_name: str) -> None:
        """Attach a variance-capture hook to ``module``.

        Uses FSDPVarianceCaptureHook for FSDP modules and
        ReplicateVarianceCaptureHook for replicate modules (and always for
        the spectral path, which is pure DDP only).
        """
        if self._spectral_variance or not self.parallelism_parameters.fsdp_enabled:
            hook: BaseVarianceCaptureHook = ReplicateVarianceCaptureHook()
        else:
            hook = FSDPVarianceCaptureHook()
        module.set_custom_reduce_scatter(hook)
        self._hooks[module] = hook
        if module_name == "norm":
            self._norm_module = module
        elif module_name == "output":
            self._output_module = module
        logger.info(f"Registered {type(hook).__name__} for {module_name}")

    # -------------------------------------------------------- capture control

    def should_capture_variance(self, train_step: int) -> None:
        """Reset stored stats and toggle capture for this train step."""
        if self._spectral_variance and not self._spectral_param_infos_initialized:
            self._setup_spectral_param_infos()
        self._clear_statistics()
        capture = self._compute_variance_this_step(train_step=train_step)
        for hook in self._hooks.values():
            hook.set_capture_status(enable=capture)

    def _clear_statistics(self) -> None:
        for hook in self._hooks.values():
            hook.clear_statistics()
        self.variance_metrics_result.clear()

    def _compute_variance_this_step(self, train_step: int) -> bool:
        return train_step == 1 or train_step % self.freq == 0

    def _setup_spectral_param_infos(self) -> None:
        """Lazily push ``(numel, shape)`` info into each replicate hook so the
        spectral path can build per-parameter gram matrices.

        The output module's hook captures both ``norm`` and ``output`` params
        because they share an FSDP module (see model/parallelize.py).
        """
        norm_module = self._norm_module
        output_module = self._output_module

        for module, hook in self._hooks.items():
            if module is norm_module:
                continue
            param_infos: list[tuple[int, tuple[int, ...]]] = []
            if module is output_module and norm_module is not None:
                for _, p in norm_module.named_parameters():
                    local_p = p.to_local()
                    param_infos.append((local_p.numel(), tuple(local_p.shape)))
            for _, p in module.named_parameters():
                local_p = p.to_local()
                param_infos.append((local_p.numel(), tuple(local_p.shape)))
            assert isinstance(hook, ReplicateVarianceCaptureHook)
            hook.set_param_infos(param_infos)

        self._spectral_param_infos_initialized = True

    # ----------------------------------------------------------- entry point

    @torch.no_grad()
    def compute_variance_metrics(
        self,
        accumulation_steps: int,
        local_batch_size: int,
        train_step: int,
    ) -> None:
        """Compute variance metrics from per-module hooks and accumulate them
        into ``self.variance_metrics_result``. No-op when this step is not a
        sampling step."""
        if not self._compute_variance_this_step(train_step=train_step):
            return

        sample_size = local_batch_size
        n_samples = accumulation_steps * self.parallelism_parameters.dp_degree

        if self._spectral_variance:
            self._accumulate_nuclear_metrics(
                accumulation_steps=accumulation_steps,
                sample_size=sample_size,
                n_samples=n_samples,
            )
        else:
            self._accumulate_elementwise_metrics(
                accumulation_steps=accumulation_steps,
                sample_size=sample_size,
                n_samples=n_samples,
            )

    def metrics_to_log(self, train_step: int) -> dict[str, float] | None:
        """Return the flattened logging dict, or ``None`` on non-sampling steps."""
        if self._compute_variance_this_step(train_step=train_step):
            return self.variance_metrics_result.metrics_to_logging_dict()
        return None

    # ----------------------------------------------------- elementwise path

    def _clone_rescaled_sum_g_sq(
        self,
        hook: BaseVarianceCaptureHook,
        accumulation_steps: int,
    ) -> torch.Tensor:
        """Undo the loss-by-token-count rescaling baked into captured stats."""
        return (
            hook.clone_sum_g_sq()
            * accumulation_steps**2
            * self.parallelism_parameters.dp_degree**2
        )

    def _accumulate_elementwise_metrics(
        self,
        accumulation_steps: int,
        sample_size: int,
        n_samples: int,
    ) -> None:
        """Default L1 / L2 / L_inf variance accumulation."""
        norm_module = self._norm_module
        output_module = self._output_module
        for module, hook in self._hooks.items():
            # Stats for the norm layer live inside the output-layer FSDP
            # module's hook (norm and output are combined into one FSDP
            # module; see components/model/parallelize.py).
            hook_to_clone = (
                self._hooks[output_module]
                if module is norm_module and output_module is not None
                else hook
            )
            sum_g_sq = self._clone_rescaled_sum_g_sq(
                hook=hook_to_clone,
                accumulation_steps=accumulation_steps,
            )
            if module is norm_module:
                total_norm_params = sum(
                    p.to_local().numel() for p in module.parameters()
                )
                sum_g_sq = sum_g_sq[:total_norm_params]
            elif module is output_module and norm_module is not None:
                total_norm_params = sum(
                    p.to_local().numel() for p in norm_module.parameters()
                )
                sum_g_sq = sum_g_sq[total_norm_params:]

            offset = 0
            for param_name, param in module.named_parameters():
                numel = param.to_local().numel()
                # When TP is enabled, norm layers are parallelized via
                # Sequence Parallel; their weights are duplicated along TP,
                # so their metrics must NOT be all-reduced across TP.
                reduce_along_tp = module is not norm_module and "norm" not in param_name
                agg, mean = self._variance_from_sum_g_sq(
                    sum_g_sq=sum_g_sq[offset : offset + numel],
                    mean_g=param.grad.to_local().view(-1),
                    n_samples=n_samples,
                    sample_size=sample_size,
                    reduce_along_tp=reduce_along_tp,
                )
                self.variance_metrics_result.global_variance_metrics += agg
                self.variance_metrics_result.global_mean_metrics += mean
                offset += numel

    @torch.no_grad()
    def _variance_from_sum_g_sq(
        self,
        mean_g: torch.Tensor,
        sum_g_sq: torch.Tensor,
        n_samples: int,
        sample_size: int,
        reduce_along_tp: bool = False,
    ) -> tuple[AggregatedVarianceMetrics, MeanMetrics]:
        """Aggregate elementwise variance for one parameter across DP / TP."""
        # Population variance: E_hat[g^2] / N - (E_hat[g])^2.
        # The /replicate_degree compensates for the all-reduce sum across
        # replicate groups that follows when DDP is enabled.
        var_sample: torch.Tensor = (
            sum_g_sq / n_samples
            - mean_g.pow(2) / self.parallelism_parameters.replicate_degree
        )
        # Population -> sample variance.
        var_sample = var_sample * (n_samples / (n_samples - 1))

        if self.parallelism_parameters.ddp_enabled:
            dist.all_reduce(
                var_sample,
                op=dist.ReduceOp.SUM,
                group=self.parallelism_parameters.ddp_process_group,
            )

        var_sample = torch.clamp(var_sample, min=0)

        var_sum = torch.sum(var_sample)
        std_l1_sum = torch.sum(torch.sqrt(var_sample))
        std_linf_max = torch.max(torch.sqrt(var_sample))
        mean_l2_sq = torch.linalg.norm(mean_g, ord=2) ** 2
        mean_l1 = torch.linalg.norm(mean_g, ord=1)
        mean_linf = torch.linalg.norm(mean_g, ord=torch.inf)

        sum_bundle = torch.stack([var_sum, std_l1_sum, mean_l2_sq, mean_l1])
        max_bundle = torch.stack([std_linf_max, mean_linf])

        if self.parallelism_parameters.fsdp_enabled:
            # TP cannot be enabled with only DDP in the current framework.
            group: dist.ProcessGroup = (
                self.parallelism_parameters.tp_fsdp_process_group
                if self.parallelism_parameters.tp_enabled and reduce_along_tp
                else self.parallelism_parameters.fsdp_process_group
            )
            dist.all_reduce(sum_bundle, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(max_bundle, op=dist.ReduceOp.MAX, group=group)

        sample_metrics = VarianceMetrics(
            variance=sum_bundle[0].item(),
            std_l1=sum_bundle[1].item(),
            std_linf=max_bundle[0].item(),
        )
        aggregated = AggregatedVarianceMetrics(
            variance_metrics_sample=sample_metrics,
            variance_metrics_example=sample_metrics * sample_size,
            variance_metrics_global=sample_metrics / n_samples,
        )
        mean_metrics = MeanMetrics(
            mean_l2_sq=sum_bundle[2].item(),
            mean_l1=sum_bundle[3].item(),
            mean_linf=max_bundle[1].item(),
        )
        return aggregated, mean_metrics

    # ----------------------------------------------------------- spectral path

    def _accumulate_nuclear_metrics(
        self,
        accumulation_steps: int,
        sample_size: int,
        n_samples: int,
    ) -> None:
        """Spectral / nuclear-norm variance accumulation (pure DDP, 2D params)."""
        norm_module = self._norm_module
        output_module = self._output_module
        rescale = accumulation_steps**2 * self.parallelism_parameters.dp_degree**2

        for module, hook in self._hooks.items():
            if module is norm_module:
                # Norm params are 1D - nothing to contribute under nuclear norm.
                continue

            assert isinstance(hook, ReplicateVarianceCaptureHook)
            gram_matrices = hook.clone_nuclear_gram_matrices()
            # The output hook's buffer starts with norm params; offset past them.
            gram_offset = (
                sum(1 for _ in norm_module.parameters())
                if module is output_module and norm_module is not None
                else 0
            )

            for param_idx, (_, param) in enumerate(module.named_parameters()):
                local_param = param.to_local()
                if local_param.ndim != 2:
                    continue
                gram = gram_matrices[gram_offset + param_idx]
                if gram is None:
                    continue
                agg, mean = self._nuclear_variance(
                    sum_g_sq=gram * rescale,
                    mean_g=param.grad.to_local().view(-1),
                    n_samples=n_samples,
                    sample_size=sample_size,
                    shape=tuple(local_param.shape),
                )
                self.variance_metrics_result.global_variance_metrics += agg
                self.variance_metrics_result.global_mean_metrics += mean

    @torch.no_grad()
    def _nuclear_variance(
        self,
        mean_g: torch.Tensor,
        sum_g_sq: torch.Tensor,
        n_samples: int,
        sample_size: int,
        shape: tuple[int, ...],
    ) -> tuple[AggregatedVarianceMetrics, MeanMetrics]:
        """Nuclear-norm variance metrics for a single 2D parameter.

        Args:
            mean_g: Flattened mean gradient for this parameter on this rank.
            sum_g_sq: Per-sample gram matrix on this rank, shape ``(k, k)``
                where ``k = min(shape)``, already aggregated across accumulation
                steps and rescaled for the loss-by-token-count normalization.
            n_samples: ``accumulation_steps * dp_degree``.
            sample_size: Examples per gradient sample (local batch).
            shape: Original 2D shape of the parameter gradient.
        """
        mat = mean_g.reshape(shape)
        mean_g_nuclear = torch.linalg.matrix_norm(mat, ord="nuc")

        if self.parallelism_parameters.ddp_enabled:
            dist.all_reduce(
                sum_g_sq,
                op=dist.ReduceOp.SUM,
                group=self.parallelism_parameters.ddp_process_group,
            )

        mean_g_gram = mat.T @ mat if mat.shape[0] >= mat.shape[1] else mat @ mat.T

        cov_gram = sample_size * (sum_g_sq - n_samples * mean_g_gram) / (n_samples - 1)
        eigenvalues = torch.clamp(torch.linalg.eigvalsh(cov_gram), min=0)
        nuclear_norm_std = torch.sum(torch.sqrt(eigenvalues)).item()

        sample_metrics = VarianceMetrics(nuclear_norm=nuclear_norm_std)
        aggregated = AggregatedVarianceMetrics(
            variance_metrics_sample=sample_metrics,
            variance_metrics_example=sample_metrics * sample_size,
            variance_metrics_global=sample_metrics / n_samples,
        )
        mean_metrics = MeanMetrics(nuc_norm=mean_g_nuclear.item())
        return aggregated, mean_metrics
