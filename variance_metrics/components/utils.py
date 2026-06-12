# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""General utilities: logging, device discovery, distributed helpers."""

import contextlib
import logging
import os
import sys
from collections.abc import Iterable
from types import ModuleType
from typing import Protocol

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

# ---------- Logging ----------

logger = logging.getLogger()


def init_logger() -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"


# ---------- Device discovery ----------


def get_device_info() -> tuple[str, ModuleType]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


# ---------- Batch aggregation ----------


class BatchAggregator:
    """Aggregates ``number_of_aggregations`` consecutive batches into one.

    Used by the adaptive batch-size flow to grow the local batch size by
    concatenating several batches from the underlying iterator.
    """

    def __init__(self, data_iterable: Iterable, number_of_aggregations: int = 1):
        if number_of_aggregations < 1:
            raise ValueError(
                f"number_of_aggregations must be a positive integer, got: "
                f"{number_of_aggregations}"
            )
        self._data_iter = iter(data_iterable)
        self.number_of_aggregations = number_of_aggregations

    def __iter__(self):
        return iter(self._data_iter)

    def __next__(self):
        input_dict, labels = next(self._data_iter)
        for _ in range(self.number_of_aggregations - 1):
            input_more, label_more = next(self._data_iter)
            input_dict = {
                k: torch.cat([input_dict[k], input_more[k]], dim=0) for k in input_dict
            }
            labels = torch.cat([labels, label_more], dim=0)
        return input_dict, labels


# ---------- Distributed helpers ----------


def _dist_reduce(x: torch.Tensor, reduceOp: str, mesh: DeviceMesh | None) -> float:
    if isinstance(x, DTensor):
        x = x.full_tensor()
    if mesh is None:
        return float(x.item())
    return float(funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item())


def dist_max(x: torch.Tensor, mesh: DeviceMesh | None = None) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh)


def dist_sum(x: torch.Tensor, mesh: DeviceMesh | None = None) -> float:
    return _dist_reduce(x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh)


def set_determinism(parallel_dims, device, debug_config) -> None:
    """Set the same DTensor manual seed across all ranks for reproducibility."""
    if debug_config.deterministic:
        logger.info("Deterministic mode enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    seed = debug_config.seed
    if parallel_dims.world_size == 1:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
        return

    if seed is None:
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()
    assert isinstance(seed, int)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    torch.distributed.tensor._random.manual_seed(seed, parallel_dims.world_mesh)


class TrainContext(Protocol):
    def __call__(self) -> contextlib.AbstractContextManager[None]: ...


def get_train_context(enable_loss_parallel: bool) -> TrainContext:
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())
            yield

    return context


def maybe_enable_amp(
    parallel_dims, device_type: str
) -> contextlib.AbstractContextManager[None] | torch.autocast:
    """bfloat16 autocast when neither FSDP nor TP is taking care of mixed precision."""
    if parallel_dims.fsdp_enabled or parallel_dims.dp_replicate_enabled:
        # FSDP / replicate handle mixed precision internally.
        return contextlib.nullcontext()
    if parallel_dims.tp_enabled:
        # TP-only without FSDP/DDP: AMP can't safely wrap loss_parallel; skip.
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=torch.bfloat16)


def init_distributed() -> int:
    """Initialize the default process group with NCCL."""
    if torch.distributed.is_initialized():
        logger.warning("torch.distributed already initialized; skipping init.")
        return torch.distributed.get_world_size()

    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "3"
    backend = torch.distributed.Backend.default_device_backend_map.get(
        device_type, "nccl"
    )
    torch.distributed.init_process_group(backend=backend)
    return torch.distributed.get_world_size()


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    """Standard grad clipping that converts a DTensor total-norm to local."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )
    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()
    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
