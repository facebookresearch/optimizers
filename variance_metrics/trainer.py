# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single Trainer class with optional gradient-variance tracking and
adaptive batch size, supporting DDP / FSDP / TP via PyTorch native APIs."""

import dataclasses
import json
import os
import time
from collections.abc import Iterable, Iterator

import torch
from components import utils
from components.adaptive_batch_size_manager import BatchSizeManager
from components.configs import TrainerConfig
from components.configurable import Configurable
from components.dataloader import BaseDataLoader, DataloaderExhaustedError
from components.logger import MetricsProcessor
from components.lr_scheduler import LRSchedulersContainer
from components.parallel_dims import ParallelDims
from components.tokenizer import HuggingFaceTokenizer
from components.utils import logger
from components.validate import Validator
from optimizer import OptimizersContainer
from torch.distributed.elastic.multiprocessing.errors import record
from variance_metrics.calculator import (
    VarianceMetricsCalculator,
    VarianceMetricsParallelismConfig,
)

# PyTorch's default ignore index for cross-entropy loss.
IGNORE_INDEX = -100


class Trainer(Configurable):
    """Llama3 trainer with DDP/FSDP/TP, optional variance metrics, and
    optional adaptive batch sizing driven by GNS."""

    Config = TrainerConfig

    # core configs
    config: TrainerConfig
    parallel_dims: ParallelDims

    # swappable training components
    tokenizer: HuggingFaceTokenizer
    dataloader: BaseDataLoader
    model_config: Configurable.Config
    model: torch.nn.Module
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer
    validator: Validator
    metrics_processor: MetricsProcessor

    # runtime utilities
    device: torch.device
    train_context: utils.TrainContext
    gradient_accumulation_steps: int

    # additional training states
    step: int
    global_tokens_seen: int

    @record
    def __init__(self, config: TrainerConfig):
        torch._C._log_api_usage_once("components.train")

        # If a token budget is set with adaptive batch sizing, derive total
        # steps + warmup steps before anything reads them.
        if config.training.n_token_limit > 0:
            config.training.steps = 1 + config.training.n_token_limit // (
                config.training.seq_len * config.training.global_batch_size
            )
        if (
            config.adaptive_batch_size.enable
            and config.adaptive_batch_size.initial_constant_batch_steps is None
        ):
            config.adaptive_batch_size.initial_constant_batch_steps = round(
                config.lr_scheduler.warmup_ratio * config.training.steps
            )

        self.config = config
        assert config.model_spec is not None, "model_spec must be set"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        # Init distributed and build meshes.
        self.parallel_dims = parallel_dims = self._init_distributed()

        self._validate_optimizer_parallelism()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0
        self.dp_degree = batch_degree

        utils.set_determinism(parallel_dims, self.device, config.debug)

        # Tokenizer + dataloader.
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
        )

        # Build model on meta device.
        model_config = model_spec.model
        model_config.update_from_config(trainer_config=config)
        self.model_config = model_config
        logger.info(
            f"Building {model_spec.name} {model_spec.flavor} with "
            f"{json.dumps(dataclasses.asdict(model_config), indent=2, ensure_ascii=False)}"
        )
        with torch.device("meta"):
            model = model_config.build()

        # Metrics processor.
        self.metrics_processor = config.metrics.build(
            config_dict=config.to_dict(),
        )
        model_param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model {model_spec.name} {model_spec.flavor} "
            f"size: {model_param_count:,} parameters"
        )

        # Verify and resolve global / accumulation batch sizes.
        global_batch_size = config.training.global_batch_size
        assert global_batch_size > 0, "training.global_batch_size must be > 0"
        assert (
            global_batch_size % (config.training.local_batch_size * batch_degree) == 0
        ), (
            f"global batch size ({global_batch_size}) must be divisible by "
            f"local_batch_size * dp ({config.training.local_batch_size} * {batch_degree})"
        )
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * batch_degree
        )
        assert self.gradient_accumulation_steps > 0

        # Apply DDP / FSDP / TP.
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
        )
        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights()
        model.train()
        self.model = model

        # Optimizer + token-budget-driven LR scheduler.
        self.optimizers = config.optimizer.build(model=self.model)
        # Token budget: explicit limit if set, otherwise the count implied
        # by training.steps * global_batch_size * seq_len.
        if config.training.n_token_limit > 0:
            self.total_tokens = config.training.n_token_limit
        else:
            self.total_tokens = (
                config.training.steps * global_batch_size * config.training.seq_len
            )
        self.initial_global_batch_size = global_batch_size
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            total_tokens=self.total_tokens,
        )

        self.step = 0
        self.global_tokens_seen = 0

        self.train_context = utils.get_train_context(parallel_dims.tp_enabled)
        self.maybe_enable_amp = utils.maybe_enable_amp(parallel_dims, device_type)

        if config.validator.enable:
            self.validator = config.validator.build(
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
            )

        logger.info(
            "Trainer initialized: "
            f"local batch {config.training.local_batch_size}, "
            f"global batch {global_batch_size}, "
            f"grad accum steps {self.gradient_accumulation_steps}, "
            f"seq_len {config.training.seq_len}, "
            f"steps {config.training.steps} "
            f"(warmup ratio {config.lr_scheduler.warmup_ratio})"
        )

        self.variance_metrics_calculator = None
        if config.variance_metrics.enable:
            self._init_variance_metrics()

        # Adaptive batch size.
        self.batch_size_manager: BatchSizeManager | None = None
        self.local_batch_size = config.training.local_batch_size
        self.local_batch_size_from_config = config.training.local_batch_size
        self.global_batch_size = (
            self.local_batch_size * self.gradient_accumulation_steps * self.dp_degree
        )
        if config.adaptive_batch_size.enable:
            self._init_adaptive_batch_size()

    # ---------- init helpers ----------

    def _init_distributed(self) -> ParallelDims:
        world_size = utils.init_distributed()
        p = self.config.parallelism
        return ParallelDims(
            dp_shard=p.data_parallel_shard_degree,
            dp_replicate=p.data_parallel_replicate_degree,
            tp=p.tensor_parallel_degree,
            world_size=world_size,
        )

    def _validate_optimizer_parallelism(self) -> None:
        """Muon orthogonalizes each 2D parameter's update via Newton-Schulz,
        which needs the full (replicated) gradient matrix. Under FSDP / HSDP /
        TP the parameters are sharded ``DTensor``s, so the iteration would run
        on shards (or silently rely on implicit redistribution) rather than the
        intended full matrix. Restrict Muon to the replicated (DDP) setting.
        """
        optimizer_cls = self.config.optimizer.optimizer_cls
        if optimizer_cls is None or optimizer_cls.__name__ != "Muon":
            return
        if self.parallel_dims.fsdp_enabled or self.parallel_dims.tp_enabled:
            raise RuntimeError(
                "Muon requires replicated (DDP) parameters: it orthogonalizes "
                "each 2D parameter via Newton-Schulz, which needs the full "
                "gradient matrix. Use data_parallel_replicate_degree > 1 with "
                "data_parallel_shard_degree=1 and tensor_parallel_degree=1. "
                f"Got dp_replicate={self.parallel_dims.dp_replicate}, "
                f"dp_shard={self.parallel_dims.dp_shard}, "
                f"tp={self.parallel_dims.tp}."
            )

    def _init_variance_metrics(self) -> None:
        logger.info("Variance tracking enabled")
        params = self._variance_parallelism_config()
        if self.config.variance_metrics.spectral_variance:
            # Spectral / nuclear-norm GNS only supports pure DDP (replicate).
            if (
                self.parallel_dims.fsdp_enabled
                or self.parallel_dims.tp_enabled
                or not self.parallel_dims.dp_replicate_enabled
            ):
                raise RuntimeError(
                    "spectral_variance=True requires pure DDP "
                    "(data_parallel_replicate_degree > 1, "
                    "data_parallel_shard_degree=1, tensor_parallel_degree=1). "
                    f"Got dp_replicate={self.parallel_dims.dp_replicate}, "
                    f"dp_shard={self.parallel_dims.dp_shard}, "
                    f"tp={self.parallel_dims.tp}."
                )
        self.variance_metrics_calculator = VarianceMetricsCalculator(
            variance_config=self.config.variance_metrics,
            parallelism_parameters=params,
            spectral_variance=self.config.variance_metrics.spectral_variance,
        )
        m = self.model
        self.variance_metrics_calculator.register_hook(
            module=m.tok_embeddings, module_name="tok_embeddings"
        )
        self.variance_metrics_calculator.register_hook(
            module=m.norm, module_name="norm"
        )
        self.variance_metrics_calculator.register_hook(
            module=m.output, module_name="output"
        )
        for layer_id, transformer_block in m.layers.items():
            self.variance_metrics_calculator.register_hook(
                module=transformer_block, module_name=f"layer_{layer_id}"
            )

    def _variance_parallelism_config(self) -> VarianceMetricsParallelismConfig:
        if not self.parallel_dims.dp_enabled:
            raise RuntimeError(
                "Variance cannot be computed with data-parallel degree = 1"
            )
        cfg = VarianceMetricsParallelismConfig(
            dp_degree=self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard,
        )
        if self.parallel_dims.dp_replicate_enabled:
            mesh = self.parallel_dims.get_mesh(["dp_replicate"])
            cfg.ddp_process_group = mesh.get_group()
            cfg.replicate_degree = mesh.size()
        if self.parallel_dims.fsdp_enabled:
            cfg.fsdp_process_group = self.parallel_dims.get_mesh(["fsdp"]).get_group()
            if self.parallel_dims.tp_enabled:
                cfg.tp_fsdp_process_group = (
                    self.parallel_dims.get_mesh(["fsdp", "tp"])
                    ._flatten(mesh_dim_name="dp_shard_tp")
                    .get_group()
                )
        return cfg

    def _init_adaptive_batch_size(self) -> None:
        logger.info("Adaptive batch size enabled")
        config = self.config
        assert self.local_batch_size > 0 and self.global_batch_size > 0
        self.batch_size_manager = BatchSizeManager(
            adaptive_batch_config=config.adaptive_batch_size,
            local_batch_size=self.local_batch_size_from_config,
            dp_degree=self.dp_degree,
        )

    def _update_lr(self) -> None:
        """Set the LR for this step from token progress + batch scale."""
        batch_scale = (
            self.global_batch_size / self.initial_global_batch_size
            if self.initial_global_batch_size > 0
            else 1.0
        )
        self.lr_schedulers.update(
            tokens_seen=self.global_tokens_seen, batch_scale=batch_scale
        )

    # ---------- per-batch helpers ----------

    def _batch_generator(
        self, data_iterable: Iterable
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        data_iterator = iter(data_iterable)
        while True:
            try:
                yield next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex

    def _forward_backward(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        inputs = input_dict["input"]
        with self.train_context():
            with self.maybe_enable_amp:
                pred = self.model(inputs)
                loss_sum = torch.nn.functional.cross_entropy(
                    pred.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                loss = loss_sum / global_valid_tokens
            del pred
            loss.backward()
        return loss

    # ---------- training loop ----------

    def train_step(self, data_iterator: Iterator) -> None:
        data_iterator, suggested_batch_size = self._apply_adaptive_batch_size(
            data_iterator
        )

        self.optimizers.zero_grad()
        # Update LR for this step from token progress + batch scale.
        self._update_lr()
        lr = self.lr_schedulers.get_last_lr()[0]

        if self.variance_metrics_calculator is not None:
            self.variance_metrics_calculator.should_capture_variance(
                train_step=self.step
            )

        microbatches, global_valid_tokens, local_valid_tokens = (
            self._collect_microbatches(data_iterator)
        )

        # Forward / backward over microbatches.
        accumulated_losses = []
        for input_dict, labels in microbatches:
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)
            loss = self._forward_backward(
                input_dict,
                labels,
                global_valid_tokens,  # pyrefly: ignore
            )
            accumulated_losses.append(loss.detach())

        # Variance metrics are computed before clipping (clipping mutates grads).
        self._compute_and_log_variance_metrics()

        grad_norm = utils.clip_grad_norm_(
            list(self.model.parameters()),
            self.config.training.max_norm,
            foreach=True,
        )
        self.optimizers.step()

        if self.batch_size_manager is not None:
            self._log_adaptive_batch_metrics(suggested_batch_size)

        if not self.metrics_processor.should_log(self.step):
            return

        self._log_step_metrics(
            losses=accumulated_losses,
            global_valid_tokens=global_valid_tokens,
            local_valid_tokens=local_valid_tokens,
            grad_norm=grad_norm,
            lr=lr,
        )

    def _apply_adaptive_batch_size(
        self, data_iterator: Iterator
    ) -> tuple[Iterator, int]:
        suggested_batch_size = self.global_batch_size
        if self.batch_size_manager is None:
            return data_iterator, suggested_batch_size
        suggested_batch_size = self._adjust_batch_size()
        if self.local_batch_size % self.local_batch_size_from_config != 0:
            raise ValueError(
                f"Local batch size can only be increased in multiples. "
                f"Got {self.local_batch_size} from base "
                f"{self.local_batch_size_from_config}."
            )
        data_iterator = utils.BatchAggregator(
            data_iterable=data_iterator,
            number_of_aggregations=self.local_batch_size
            // self.local_batch_size_from_config,
        )
        return data_iterator, suggested_batch_size

    def _collect_microbatches(
        self, data_iterator: Iterator
    ) -> tuple[list, torch.Tensor, torch.Tensor]:
        """Pull ``gradient_accumulation_steps`` microbatches, count tokens, and
        all-reduce raw+valid token counts across the data-parallel mesh."""
        parallel_dims = self.parallel_dims
        microbatches = []
        local_raw_tokens = 0
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_raw_tokens += labels.numel()
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = utils.dist_sum(local_valid_tokens, batch_mesh)
            global_raw_tokens = int(
                utils.dist_sum(
                    torch.tensor(
                        local_raw_tokens, dtype=torch.int64, device=self.device
                    ),
                    batch_mesh,
                )
            )
        else:
            global_valid_tokens = local_valid_tokens.float()
            global_raw_tokens = local_raw_tokens
        self.global_tokens_seen += global_raw_tokens
        return microbatches, global_valid_tokens, local_valid_tokens

    def _compute_and_log_variance_metrics(self) -> None:
        calc = self.variance_metrics_calculator
        if calc is None:
            return
        calc.compute_variance_metrics(
            accumulation_steps=self.gradient_accumulation_steps,
            local_batch_size=self.local_batch_size,
            train_step=self.step,
        )
        variance_metrics = calc.metrics_to_log(train_step=self.step)
        if variance_metrics:
            self.metrics_processor.log(self.step, variance_metrics)

    def _log_step_metrics(
        self,
        losses: list[torch.Tensor],
        global_valid_tokens: torch.Tensor,
        local_valid_tokens: torch.Tensor,
        grad_norm: torch.Tensor,
        lr: float,
    ) -> None:
        parallel_dims = self.parallel_dims
        loss = torch.sum(torch.stack(losses))
        if parallel_dims.dp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_avg_loss = utils.dist_sum(loss, loss_mesh)
            global_max_loss = utils.dist_max(local_avg_loss, loss_mesh)
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())

        self.metrics_processor.log(
            self.step,
            {
                "loss_metrics/global_avg_loss": global_avg_loss,
                "loss_metrics/global_max_loss": global_max_loss,
                "grad_norm": float(grad_norm.item()),
                "n_tokens_seen": self.global_tokens_seen,
                "lr": lr,
            },
            verbose=True,
        )

    def _adjust_batch_size(self) -> int:
        """Apply the adaptive-batch-size update for this step.

        The LR is recomputed inline from ``global_tokens_seen`` + the new
        batch size on the next ``_update_lr`` call, so no scheduler rebuild
        is needed here.
        """
        assert self.batch_size_manager is not None
        if self.batch_size_manager.batch_size_change_enabled:
            (
                new_grad_accum,
                new_local_bs,
                suggested_bs,
            ) = self.batch_size_manager.update_global_batch_size(
                train_step=self.step,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                global_batch_size=self.global_batch_size,
                local_batch_size=self.local_batch_size_from_config,
                dp_degree=self.dp_degree,
            )
        else:
            new_grad_accum = self.gradient_accumulation_steps
            new_local_bs = self.local_batch_size
            suggested_bs = self.global_batch_size

        self.local_batch_size = new_local_bs
        self.gradient_accumulation_steps = new_grad_accum
        self.global_batch_size = (
            self.local_batch_size * self.gradient_accumulation_steps * self.dp_degree
        )
        return suggested_bs

    def _log_adaptive_batch_metrics(self, suggested_batch_size: int) -> None:
        assert self.batch_size_manager is not None
        metrics: dict[str, float] = {
            "batch_size/global_batch_size": self.global_batch_size,
            "batch_size/accumulation_steps": self.gradient_accumulation_steps,
            "batch_size/local_batch_size": self.local_batch_size,
            "batch_size/suggested_batch_size": suggested_batch_size,
        }

        calc = self.variance_metrics_calculator
        if calc is not None and calc._compute_variance_this_step(train_step=self.step):
            result = calc.variance_metrics_result
            extra = self.batch_size_manager.update_parameters(
                global_variance_metrics=result.global_variance_metrics,
                global_mean_metrics=result.global_mean_metrics,
            )
            if extra:
                metrics.update(extra)

        if metrics:
            self.metrics_processor.log(self.step, metrics)

    @record
    def train(self) -> None:
        logger.info(f"Training starts at step {self.step + 1}")

        data_iterator = self._batch_generator(self.dataloader)
        while self._should_continue_training():
            self.step += 1
            try:
                self.train_step(data_iterator)
            except DataloaderExhaustedError:
                logger.warning("Ran out of data; last step was canceled.")
                break

            if self.config.validator.enable and self.validator.should_validate(
                self.step
            ):
                self.validator.validate(self.model, self.step)

        if torch.distributed.get_rank() == 0:
            time.sleep(2)
        logger.info(f"Training completed; processed {self.global_tokens_seen:,} tokens")

    def _should_continue_training(self) -> bool:
        # When adaptive batch sizing is on, per-step token count grows during
        # training, so `training.steps` (derived from the *initial* global
        # batch) is no longer a reliable stopping criterion. Cap on
        # `global_tokens_seen` instead, and run a final validation pass.
        token_limit = self.config.training.n_token_limit
        if self.batch_size_manager is not None and token_limit > 0:
            if self.global_tokens_seen > token_limit:
                logger.info(
                    f"Stopping: token limit {token_limit} reached "
                    f"({self.global_tokens_seen} seen)."
                )
                if self.config.validator.enable:
                    self.validator.validate(self.model, self.step)
                return False
        return self.step < self.config.training.steps

    def close(self) -> None:
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
