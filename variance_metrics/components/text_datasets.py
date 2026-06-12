# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from components.configs import HFTextDataLoaderConfig
from components.dataloader import BaseDataLoader
from components.tokenizer import HuggingFaceTokenizer
from components.utils import logger
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader


def _check_hf_auth() -> None:
    """Verify HuggingFace credentials are available before streaming gated
    datasets like ``allenai/c4``.

    Streaming requires an authenticated HF account. Raises a clear error
    pointing the user at the README setup steps if no token is found.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return
    token_path = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        / "token"
    )
    if token_path.is_file():
        return
    raise RuntimeError(
        "HuggingFace authentication not configured but a HF-streamed dataset "
        "(c4 / c4_validation) was requested.\n"
        "Set HF_TOKEN, or run `huggingface-cli login` once. "
        "See the 'Hugging Face authentication' section of README.md for setup."
    )


@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration.

    Streams from HuggingFace (``allenai/c4``); requires an authenticated
    HF account. The check fails fast with a pointer to the README if no
    token is configured.
    """
    _check_hf_auth()
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


# Add your dataset here - see the "Adding a new dataset" section of README.md
#
# Behavior notes:
#   c4 / c4_validation - streamed from HuggingFace; needs HF authentication.
#                        No local download step.
#   c4_test            - local fixture under tests/assets/c4_test; no network
#                        required. See README for how to populate it.
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(
            "json", data_files=os.path.join(path, "data.jsonl"), split="train"
        ),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: HuggingFaceTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self._text_processor = text_processor

        # In-iteration buffers used to chunk documents into seq_len-sized
        # segments; positions reset at document boundaries.
        self._token_buffer: list[int] = []
        self._position_buffer: list[int] = []

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in iter(self._data):
                sample_text = self._text_processor(sample)
                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=True, add_eos=True
                )
                self._token_buffer.extend(sample_tokens)
                # Per-document positions reset at document boundaries,
                # matching inference frameworks (e.g. vLLM) that start
                # positions at 0 per request. Positions wrap at seq_len
                # to stay within the RoPE cache, effectively chunking
                # long documents into seq_len-sized segments.
                self._position_buffer.extend(
                    i % self.seq_len for i in range(len(sample_tokens))
                )

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    pos = torch.LongTensor(self._position_buffer[:max_buffer_token_len])
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    self._position_buffer = self._position_buffer[max_buffer_token_len:]
                    yield {"input": x[:-1], "positions": pos[:-1]}, x[1:]

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break
            logger.warning(f"Dataset {self.dataset_name} is being re-looped")
            if hasattr(self._data, "set_epoch") and hasattr(self._data, "epoch"):
                self._data.set_epoch(self._data.epoch + 1)


class HuggingFaceTextDataLoader(StatefulDataLoader, BaseDataLoader):
    """Configurable text dataloader that wraps HuggingFaceTextDataset.

    This dataloader can be used for both training and validation by
    configuring the appropriate dataset, seq_len, batch_size, etc.
    """

    Config = HFTextDataLoaderConfig

    def __init__(
        self,
        config: HFTextDataLoaderConfig,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: HuggingFaceTokenizer,
        seq_len: int,
        local_batch_size: int,
    ):
        hf_ds = HuggingFaceTextDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )

        dataloader_kwargs = {
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "batch_size": local_batch_size,
        }
        if config.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = config.persistent_workers
            dataloader_kwargs["prefetch_factor"] = config.prefetch_factor

        super().__init__(hf_ds, **dataloader_kwargs)
