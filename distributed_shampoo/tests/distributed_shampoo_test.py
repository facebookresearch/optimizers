"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import unittest
from typing import Dict, Tuple

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.utils.shampoo_utils import CommunicationDType
from torch import nn


# DType mapping for quantized communications.
dtype_mapping: Dict[int, torch.dtype] = {
    1: torch.float16,
    2: torch.bfloat16,
    3: torch.float32,
}


class DistributedShampooTest(unittest.TestCase):
    def _train_quadratic_with_comms_dtype(
        self,
        communication_dtype: CommunicationDType = CommunicationDType.DEFAULT,
    ) -> Tuple[nn.Module, DistributedShampoo]:
        data = torch.arange(10, dtype=torch.float)
        model = nn.Sequential(
            nn.Linear(10, 1, bias=False),
        )
        model[0].weight.data.fill_(1.0)
        loss = nn.MSELoss()
        optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            num_trainers_per_group=1,
            communication_dtype=communication_dtype,
        )
        loss(model(data), torch.tensor([0.0])).backward()
        optimizer.step()
        return model, optimizer

    def test_distributed_state_dict(self) -> None:
        model, optimizer = self._train_quadratic_with_comms_dtype()
        state_dict = optimizer.distributed_state_dict(model.named_parameters())
        true_keys = {
            "step",
            "preconditioners._split_preconditioners.0._dist_buffer",
            "preconditioners._split_preconditioners.0._preconditioners.0.factor_matrix",
            "preconditioners._split_preconditioners.0._preconditioners.0.inv_factor_matrix",
            "preconditioners._split_preconditioners.0._grafting._preconditioner._dist_buffer",
            "preconditioners._split_preconditioners.0._grafting._preconditioner._preconditioner",
            "preconditioners._split_sizes.0",
            "preconditioners._split_preconditioners.0._filtered_grad",
        }
        assert len(state_dict["state"]["0.weight"].keys()) == len(true_keys)
        self.assertEqual(set(state_dict["state"]["0.weight"].keys()), true_keys)

    def test_load_distributed_state_dict(self) -> None:
        model, old_optimizer = self._train_quadratic_with_comms_dtype()
        old_state_dict = old_optimizer.distributed_state_dict(model.named_parameters())
        new_optimizer = DistributedShampoo(
            model.parameters(),
            lr=0.01,
            betas=(0.9, 1.0),
            epsilon=1e-12,
            momentum=0.0,
            weight_decay=0.0,
            max_preconditioner_dim=10,
            precondition_frequency=1,
            start_preconditioning_step=-1,
            num_trainers_per_group=1,
            communication_dtype=CommunicationDType.DEFAULT,
        )
        new_optimizer.load_distributed_state_dict(
            old_state_dict, model.named_parameters()
        )
        new_state_dict = new_optimizer.distributed_state_dict(model.named_parameters())
        self.assertEqual(
            set(old_state_dict["state"]["0.weight"].keys()),
            set(new_state_dict["state"]["0.weight"].keys()),
        )
        for key in new_state_dict["state"]["0.weight"].keys():
            torch.testing.assert_allclose(
                old_state_dict["state"]["0.weight"][key],
                new_state_dict["state"]["0.weight"][key],
            )

    def test_quantized_comms_on_small_model(self) -> None:
        # compute baseline
        baseline_model, _ = self._train_quadratic_with_comms_dtype(
            communication_dtype=CommunicationDType.DEFAULT
        )
        baseline_params = baseline_model.parameters()

        communication_dtypes = [
            CommunicationDType.FP32,
            CommunicationDType.FP16,
            CommunicationDType.BF16,
        ]

        # loop through all supported communication_dtypes
        for communication_dtype in communication_dtypes:
            with self.subTest(f"Compare against {communication_dtype} communications:"):
                machine_epsilon = torch.finfo(
                    dtype_mapping[communication_dtype.value]
                ).eps
                low_precision_model, _ = self._train_quadratic_with_comms_dtype(
                    communication_dtype=communication_dtype,
                )
                low_precision_params = low_precision_model.parameters()
                for baseline_param, low_precision_param in zip(
                    baseline_params, low_precision_params
                ):
                    torch.testing.assert_close(
                        baseline_param,
                        low_precision_param,
                        rtol=machine_epsilon,
                        atol=machine_epsilon,
                    )
