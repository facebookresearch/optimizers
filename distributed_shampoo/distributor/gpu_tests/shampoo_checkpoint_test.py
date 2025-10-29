"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import copy
from collections.abc import Callable
from operator import attrgetter
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from distributed_shampoo.distributed_shampoo import (
    DDPDistributedConfig,
    DistributedShampoo,
)
from distributed_shampoo.tests.shampoo_test_utils import construct_training_problem
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


PRECONDITIONER_DIM = 4


@instantiate_parametrized_tests
class DistributedShampooDistributedCheckpointTest(DTensorTestBase):
    def _compare_optimizer_state_dict(
        self,
        ref_optim_state: dict[str, Any],
        optim_state: dict[str, Any],
        assert_fn: Callable[..., None],
    ) -> None:
        """
        Recursively compares two optimizer state dicts, skipping 'param_groups' keys.

        Args:
            ref_optim_state (dict[str, Any]): The reference optimizer state dict.
            optim_state (dict[str, Any]): The optimizer state dict to compare.
            assert_fn (Callable[..., None]): Assertion function to use for comparing values (e.g., self.assertEqual or self.assertNotEqual).

        This function is useful for verifying that optimizer state dicts are equal (or not equal) after checkpoint save/load operations,
        while ignoring 'param_groups', as 'param_groups' is unchanged.
        """

        def recursive_compare(d1: dict[str, Any], d2: dict[str, Any]) -> None:
            self.assertEqual(d1.keys(), d2.keys())
            for k in d1:
                if isinstance(k, str) and "param_groups" in k:
                    # skip param_groups comparison
                    continue
                v1 = d1[k]
                self.assertIn(k, d2.keys())
                v2 = d2[k]
                if isinstance(v1, dict) and isinstance(v2, dict):
                    recursive_compare(v1, v2)
                else:
                    if isinstance(v1, DTensor):
                        assert_fn(v1.to_local(), v2.to_local())
                    else:
                        assert_fn(v1, v2)

        recursive_compare(ref_optim_state, optim_state)

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    @parametrize(
        "flatten_optimizer_state_dict",
        [
            True,
            False,
        ],
    )
    def test_ddp_shampoo_checkpoint(self, flatten_optimizer_state_dict: bool) -> None:
        """
        This test is intended to make sure that Shampoo `state_dict` and `load_state_dict`
        are compatible with PyTorch's API, including `torch.distributed.checkpoint.get_optimizer_state_dict`,
        `torch.distributed.checkpoint.set_optimizer_state_dict`, `torch.distributed.checkpoint.save`,
        and `torch.distributed.checkpoint.load`. `get_optimizer_state_dict` calls `state_dict` and
        `set_optimizer_state_dict` calls `load_state_dict` in Shampoo under the hood.

        Note: It tests both scenarios where `flatten_optimizer_state_dict` is True and False.
        """

        CHECKPOINT_DIR = attrgetter("temp_dir")(self)
        device = torch.device(self.device_type)

        # create a DDP model
        model, _, _, _ = construct_training_problem(
            model_linear_layers_dims=(PRECONDITIONER_DIM * 2, PRECONDITIONER_DIM * 4),
            model_dead_layers_dims=None,
            device=device,
            post_model_decoration=torch.nn.parallel.DistributedDataParallel,
        )
        optim = DistributedShampoo(
            model.parameters(),
            max_preconditioner_dim=PRECONDITIONER_DIM,
            distributed_config=DDPDistributedConfig(num_trainers_per_group=-1),
        )

        # step ahead to initialize the optimizer
        model(torch.rand(8, PRECONDITIONER_DIM * 2, device=device)).sum().backward()
        optim.step()

        # deep copy step 1's optimizer state for comparison later.
        ref_optim_state_dict: dict[str, Any] = copy.deepcopy(
            get_optimizer_state_dict(
                model,
                optim,
                options=StateDictOptions(
                    flatten_optimizer_state_dict=flatten_optimizer_state_dict
                ),
            )
        )

        # get the current model and optimizer state at step 1
        # `get_model_state_dict` and `get_optimizer_state_dict` call `model.state_dict()`
        # and `optim.state_dict()` under the hood.
        ref_state_dict = {
            "model": get_model_state_dict(model),
            "optim": ref_optim_state_dict,
        }

        # save model's state and optimizer's state to disk
        dcp.save(
            state_dict=ref_state_dict,
            storage_writer=dcp.FileSystemWriter(CHECKPOINT_DIR),
        )

        # step forward to step 2
        # so both the model and optimizer are different from previous step.
        model(torch.rand(8, PRECONDITIONER_DIM * 2, device=device)).sum().backward()
        optim.step()

        # get the current model and optimizer state at step 2 for dcp to load into.
        model_state_dict: dict[str, Any] = get_model_state_dict(model)
        optim_state_dict: dict[str, Any] = get_optimizer_state_dict(
            model,
            optim,
            options=StateDictOptions(
                flatten_optimizer_state_dict=flatten_optimizer_state_dict
            ),
        )

        # compare to make sure optimizer state is different between step 1 and 2.
        self._compare_optimizer_state_dict(
            ref_optim_state=ref_optim_state_dict,
            optim_state=optim_state_dict,
            assert_fn=self.assertNotEqual,
        )

        state_dict = {
            "model": model_state_dict,
            "optim": optim_state_dict,
        }

        # load from disk to memory
        dcp.load(
            state_dict=state_dict,
            storage_reader=dcp.FileSystemReader(CHECKPOINT_DIR),
        )
        # load from memory to model and optimizer
        # `set_model_state_dict` and `set_optimizer_state_dict` call `model.load_state_dict()`
        # and `optim.load_state_dict()` under the hood.
        set_model_state_dict(
            model=model,
            model_state_dict=state_dict["model"],
        )
        set_optimizer_state_dict(
            model=model,
            optimizers=optim,
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(
                flatten_optimizer_state_dict=flatten_optimizer_state_dict
            ),
        )
        osd_after_load: dict[str, Any] = get_optimizer_state_dict(
            model,
            optim,
            options=StateDictOptions(
                flatten_optimizer_state_dict=flatten_optimizer_state_dict
            ),
        )

        # compare to make sure the current optimizer state is the same as step 1.
        self._compare_optimizer_state_dict(
            ref_optim_state_dict,
            osd_after_load,
            self.assertEqual,
        )

        # step ahead to make sure training can continue.
        model(torch.rand(8, PRECONDITIONER_DIM * 2, device=device)).sum().backward()
        optim.step()
