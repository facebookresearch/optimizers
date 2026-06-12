# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TypeAlias

import torch
import torch.nn as nn
from components import utils
from components.configs import ValidatorConfig
from components.configurable import Configurable
from components.dataloader import BaseDataLoader
from components.logger import MetricsProcessor
from components.parallel_dims import ParallelDims
from components.tokenizer import HuggingFaceTokenizer

# PyTorch's default ignore index for cross-entropy loss.
IGNORE_INDEX = -100

ValidationContext: TypeAlias = Callable[[], AbstractContextManager[None]]


class Validator(Configurable):
    """Runs cross-entropy validation against a held-out dataloader."""

    Config = ValidatorConfig

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: HuggingFaceTokenizer,
        parallel_dims: ParallelDims,
        validation_context: ValidationContext,
        maybe_enable_amp: AbstractContextManager[None],
        metrics_processor: MetricsProcessor,
        seq_len: int,
        local_batch_size: int,
    ):
        self.config = config
        self.parallel_dims = parallel_dims
        # pyrefly: ignore [unexpected-keyword]
        dl_config = replace(config.dataloader, infinite=True)
        self.validation_dataloader = dl_config.build(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor

    def should_validate(self, step: int) -> bool:
        return step == 1 or step % self.config.freq == 0

    def _post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Splits the input dict into the main ``inputs`` tensor and any
        auxiliary fields. SDPA handles causal masking internally, so no
        attention-mask kwargs are produced."""
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        extra_inputs.pop("positions", None)
        return inputs, labels, extra_inputs

    @torch.no_grad()
    def validate(self, model: nn.Module, step: int) -> None:
        model.eval()

        parallel_dims = self.parallel_dims
        device_type = utils.device_type
        accumulated_losses = []

        for num_steps, (input_dict, labels) in enumerate(self.validation_dataloader):
            if num_steps >= self.config.steps:
                break

            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            inputs, labels, extra_inputs = self._post_dataloading_process(
                input_dict, labels
            )

            local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=device_type)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()

            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = utils.dist_sum(local_valid_tokens, batch_mesh)
            else:
                global_valid_tokens = local_valid_tokens.float()

            with self.validation_context():
                with self.maybe_enable_amp:
                    predictions = model(inputs, **extra_inputs)
                    loss_sum = torch.nn.functional.cross_entropy(
                        predictions.flatten(0, 1).float(),
                        labels.flatten(0, 1),
                        reduction="sum",
                        ignore_index=IGNORE_INDEX,
                    )

            accumulated_losses.append(loss_sum.detach() / global_valid_tokens)

        loss = torch.sum(torch.stack(accumulated_losses)) / self.config.steps
        if parallel_dims.dp_enabled:
            global_avg_loss = utils.dist_sum(
                loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_avg_loss = float(loss.item())

        self.metrics_processor.log(
            step, {"validation/loss": global_avg_loss}, verbose=True
        )
        model.train()
