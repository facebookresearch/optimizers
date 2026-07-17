#!/usr/bin/env python3


"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

Regression tests asserting that DistributedShampoo's torch.compile recompilation count
is bounded.

With ShampooPT2CompileConfig enabled, the only Python-bool input to the per-group step
function that varies across steps AND enters a compiled region is
``perform_amortized_computation``. (``use_grafting_method`` also varies, but it is
consumed entirely inside ``_precondition_and_grafting`` which is decorated with
``@torch.compiler.disable`` — see distributed_shampoo.py — so it cannot specialize a
compiled graph.) Across a run with start_preconditioning_step=S and
precondition_frequency=2, dynamo should specialize on exactly two values of
``perform_amortized_computation`` and therefore trigger compilation at exactly two
steps:
  - step 1 (perform_amortized=False; first call, fresh compile)
  - step S (perform_amortized=True; first amortized inverse-root recomputation, fresh
    compile)
Subsequent steps reuse the cached graph for whichever value of
``perform_amortized_computation`` they hit and must not trigger any further compilation.

A new compilation at any other step indicates a regression where new dynamic state has
leaked into the compiled region (e.g., a new Python-int input that varies across steps,
a new control-flow branch keyed on a Python int, a new dynamic-shape guard).

Scope: covers the actively used distributors (single-device, DDP, FullyShard family,
HybridShard family). Each multi-proc distributor lives in its own test class so that
MultiProcessTestCase spawns fresh subprocesses per variant; sharing one parametrized
class across all five was observed to flake on NCCL/dynamo state cleanup. The legacy
FSDPDistributor (FSDP1) and HSDPDistributor (FSDP1-based HSDP) are out of scope here
since they require the FSDPTest harness plus compile_fsdp_parameter_metadata; HSDP
topology with the modern fully_shard backend IS covered via the hybrid_shard classes.

Numerical equivalence between the compiled and eager paths is covered separately in
dev/gpu_tests/shampoo_pt2_test.py; this file owns the orthogonal compile-count invariant.
"""

import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial

import torch
import torch._dynamo
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    DDPDistributedConfig,
    DefaultSingleDeviceDistributedConfig,
    DistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    HybridShardDistributedConfig,
    ShampooPT2CompileConfig,
    WeightDecayType,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    construct_training_problem,
    generate_global_train_data,
)
from torch import distributed as dist, nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import ParamsT
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


PRECONDITIONER_DIM = 4
PRECONDITION_FREQUENCY = 2
START_PRECONDITIONING_STEP = 2
NUM_STEPS = 6  # Run past S+1 to verify steady state has no further recompiles.

# 1-indexed step numbers at which dynamo is expected to compile a new graph.
# Assumes START_PRECONDITIONING_STEP >= 2 so the two compile points are distinct.
# NOTE: dynamo emits MULTIPLE sub-graphs per top-level _per_group_step_impl call (due to
# graph breaks at @torch.compiler.disable boundaries and similar). This test only asserts
# WHICH steps trigger compilation, not the per-step graph COUNT — so a regression that
# inflates the number of sub-graphs within an already-expected step (e.g., an
# accidentally-introduced graph break) will not be caught here. Pinning per-distributor
# graph-count baselines is a follow-up.
EXPECTED_COMPILE_STEPS: frozenset[int] = frozenset({1, START_PRECONDITIONING_STEP})


@contextmanager
def _fresh_dynamo_state() -> Iterator[None]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        yield
    finally:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()


def _shampoo_optim_factory(
    distributed_config: DistributedConfig,
) -> Callable[[ParamsT], torch.optim.Optimizer]:
    return partial(
        DistributedShampoo,
        lr=0.001,
        betas=(0.9, 1.0),
        epsilon=1e-8,
        weight_decay=0.0,
        max_preconditioner_dim=PRECONDITIONER_DIM,
        precondition_frequency=PRECONDITION_FREQUENCY,
        start_preconditioning_step=START_PRECONDITIONING_STEP,
        weight_decay_type=WeightDecayType.DECOUPLED,
        grafting_config=AdaGradPreconditionerConfig(epsilon=1e-8),
        distributed_config=distributed_config,
        shampoo_pt2_compile_config=ShampooPT2CompileConfig(),
    )


def _run_and_assert_recompiles(
    test_case: unittest.TestCase,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    loss_fn: nn.Module,
    train_data: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Run NUM_STEPS optimizer.step() calls; assert the set of steps that triggered any
    dynamo compilation equals EXPECTED_COMPILE_STEPS.

    Mechanism: snapshot ``torch._dynamo.utils.counters["stats"]["unique_graphs"]`` before
    each step; any positive delta after step k means dynamo emitted at least one new
    graph at step k. Catches regressions that compile at unexpected steps OR fail to
    compile at expected steps.
    """

    def unique_graphs() -> int:
        return torch._dynamo.utils.counters["stats"]["unique_graphs"]

    per_step_deltas: dict[int, int] = {}
    last_count = unique_graphs()

    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()
        objective = loss_fn(model(train_data[step - 1]), target)
        objective.backward()
        optimizer.step()

        current_count = unique_graphs()
        per_step_deltas[step] = current_count - last_count
        last_count = current_count

    actual_compile_steps = {step for step, d in per_step_deltas.items() if d > 0}

    test_case.assertEqual(
        actual_compile_steps,
        set(EXPECTED_COMPILE_STEPS),
        msg=(
            "Dynamo compilation steps mismatch.\n"
            f"  Expected (compile only at): {sorted(EXPECTED_COMPILE_STEPS)}\n"
            f"  Actual   (compiled at):     {sorted(actual_compile_steps)}\n"
            f"  Per-step deltas:            {per_step_deltas}\n"
            "Each unexpected step represents a recompilation regression — most often "
            "caused by a new Python-int/bool input to _per_group_step_impl that varies "
            "across steps, a new dynamic-shape guard, or new control flow in the "
            "Shampoo per-group step path."
        ),
    )


def _run_distributed_recompilation_test(
    test_case: DTensorTestBase,
    distributed_config: DistributedConfig,
    post_model_decoration: Callable[[nn.Module], nn.Module],
) -> None:
    """Shared body for every distributed-distributor recompilation test class."""
    with _fresh_dynamo_state():
        model, loss_fn, _, target = construct_training_problem(
            model_linear_layers_dims=(
                4 * PRECONDITIONER_DIM,
                2 * PRECONDITIONER_DIM,
                1,
            ),
            model_dead_layers_dims=None,
            enable_learnable_scalar=False,
            device=torch.device("cuda"),
            fill=0.1,
            post_model_decoration=post_model_decoration,
        )
        assert isinstance(model, nn.Module)
        train_data = generate_global_train_data(
            num_steps=NUM_STEPS,
            world_size=test_case.world_size,
            data_shape=(4 * PRECONDITIONER_DIM,),
            device=torch.device("cuda"),
        )[:, dist.get_rank()]

        optimizer = _shampoo_optim_factory(
            distributed_config=distributed_config,
        )(model.parameters())

        _run_and_assert_recompiles(
            test_case, optimizer, model, loss_fn, train_data, target
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooSingleDeviceRecompilationTest(unittest.TestCase):
    def test_recompilation_count(self) -> None:
        with _fresh_dynamo_state():
            model, loss_fn, _, target = construct_training_problem(
                model_linear_layers_dims=(
                    4 * PRECONDITIONER_DIM,
                    2 * PRECONDITIONER_DIM,
                    1,
                ),
                model_dead_layers_dims=None,
                enable_learnable_scalar=False,
                device=torch.device("cuda"),
                fill=0.1,
            )
            assert isinstance(model, nn.Module)
            train_data = generate_global_train_data(
                num_steps=NUM_STEPS,
                world_size=1,
                data_shape=(4 * PRECONDITIONER_DIM,),
                device=torch.device("cuda"),
            ).squeeze(1)

            optimizer = _shampoo_optim_factory(
                distributed_config=DefaultSingleDeviceDistributedConfig,
            )(model.parameters())

            _run_and_assert_recompiles(
                self, optimizer, model, loss_fn, train_data, target
            )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooDDPRecompilationTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_recompilation_count(self) -> None:
        _run_distributed_recompilation_test(
            self,
            distributed_config=DDPDistributedConfig(),
            post_model_decoration=partial(
                DDP, device_ids=[self.rank], find_unused_parameters=False
            ),
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooFullyShardRecompilationTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_recompilation_count(self) -> None:
        _run_distributed_recompilation_test(
            self,
            distributed_config=FullyShardDistributedConfig(),
            post_model_decoration=partial(fully_shard),  # type: ignore
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooFullyShardLosslessRecompilationTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_recompilation_count(self) -> None:
        _run_distributed_recompilation_test(
            self,
            distributed_config=FullyShardDistributedConfig(
                param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE,
            ),
            post_model_decoration=partial(fully_shard),  # type: ignore
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooHybridShardRecompilationTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_recompilation_count(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda",
            (1, self.world_size),
            mesh_dim_names=("replicate", "shard"),
        )
        _run_distributed_recompilation_test(
            self,
            distributed_config=HybridShardDistributedConfig(device_mesh=mesh_2d),
            post_model_decoration=partial(fully_shard, mesh=mesh_2d),  # type: ignore
        )


@unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
class ShampooHybridShardLosslessRecompilationTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_recompilation_count(self) -> None:
        mesh_2d = init_device_mesh(
            "cuda",
            (1, self.world_size),
            mesh_dim_names=("replicate", "shard"),
        )
        _run_distributed_recompilation_test(
            self,
            distributed_config=HybridShardDistributedConfig(
                device_mesh=mesh_2d,
                param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE,
            ),
            post_model_decoration=partial(fully_shard, mesh=mesh_2d),  # type: ignore
        )
