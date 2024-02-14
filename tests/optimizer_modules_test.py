"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from optimizer_modules import OptimizerModule


@dataclass
class OptimizerTestModule(OptimizerModule):
    attribute: Union[torch.Tensor, int]
    list_of_values: Optional[List[Union[OptimizerModule, torch.Tensor]]] = None
    tuple_of_values: Optional[Tuple[float, ...]] = None
    dictionary_of_values: Optional[Dict[str, torch.Tensor]] = None
    other_module: Optional[OptimizerModule] = None


class OptimizerModulesTest(unittest.TestCase):
    def init_optimizer_module(self) -> OptimizerModule:
        other_module = OptimizerTestModule(
            attribute=42,
        )
        test_module = OptimizerTestModule(
            attribute=torch.tensor(42),
            list_of_values=[OptimizerModule(), torch.tensor(1.0)],
            tuple_of_values=(1.0, 2.0, 3.0),
            dictionary_of_values={"tensor": torch.tensor(2.0)},
            other_module=other_module,
        )
        return test_module

    def test_state_dict(self) -> None:
        test_module = self.init_optimizer_module()

        self.assertEqual(
            test_module.state_dict(store_non_tensors=True),
            {
                "attribute": torch.tensor(42),
                "list_of_values": {0: {}, 1: torch.tensor(1.0)},
                "tuple_of_values": {0: 1.0, 1: 2.0, 2: 3.0},
                "dictionary_of_values": {"tensor": torch.tensor(2.0)},
                "other_module": {
                    "attribute": 42,
                    "list_of_values": None,
                    "tuple_of_values": None,
                    "dictionary_of_values": None,
                    "other_module": None,
                },
            },
        )

    def test_state_dict_without_non_tensor_objects(self) -> None:
        test_module = self.init_optimizer_module()

        self.assertEqual(
            test_module.state_dict(store_non_tensors=False),
            {
                "attribute": torch.tensor(42),
                "list_of_values": {0: {}, 1: torch.tensor(1.0)},
                "tuple_of_values": {},
                "dictionary_of_values": {"tensor": torch.tensor(2.0)},
                "other_module": {},
            },
        )

    def test_load_state_dict_with_non_tensor_objects(self) -> None:
        test_module = self.init_optimizer_module()

        # state dict to load
        state_dict = {
            "attribute": torch.tensor(24),
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {0: 4.0, 1: 5.0, 2: 6.0},
            "dictionary_of_values": {"tensor": torch.tensor(4.0)},
            "other_module": {
                "attribute": 24,
                "list_of_values": None,
                "tuple_of_values": None,
                "dictionary_of_values": None,
                "other_module": None,
            },
        }
        test_module.load_state_dict(state_dict=state_dict, store_non_tensors=True)

        self.assertEqual(test_module.state_dict(store_non_tensors=True), state_dict)

    def test_load_state_dict_without_non_tensor_objects(self) -> None:
        test_module = self.init_optimizer_module()

        # state dict to load
        state_dict = {
            "attribute": torch.tensor(24),
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {},
            "dictionary_of_values": {"tensor": torch.tensor(4.0)},
            "other_module": {},
        }
        test_module.load_state_dict(state_dict=state_dict, store_non_tensors=False)

        self.assertEqual(test_module.state_dict(store_non_tensors=False), state_dict)

    def test_load_state_dict_with_non_matching_objects(self) -> None:
        test_module = self.init_optimizer_module()

        # state dict to load
        state_dict = {
            "attribute": 24,
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {0: "hello", 1: 5.0, 2: 6.0},
            "dictionary_of_values": torch.tensor(4.0),
            "other_module": {
                "attribute": 24,
                "list_of_values": None,
                "tuple_of_values": None,
                "dictionary_of_values": None,
                "other_module": None,
            },
        }
        expected_state_dict = {
            "attribute": torch.tensor(42),
            "list_of_values": {0: {}, 1: torch.tensor(3.0)},
            "tuple_of_values": {0: 1.0, 1: 5.0, 2: 6.0},
            "dictionary_of_values": {"tensor": torch.tensor(2.0)},
            "other_module": {
                "attribute": 24,
                "list_of_values": None,
                "tuple_of_values": None,
                "dictionary_of_values": None,
                "other_module": None,
            },
        }
        test_module.load_state_dict(state_dict=state_dict, store_non_tensors=True)

        self.assertEqual(
            test_module.state_dict(store_non_tensors=True), expected_state_dict
        )
