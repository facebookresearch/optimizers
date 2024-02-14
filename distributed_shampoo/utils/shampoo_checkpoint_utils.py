"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import json
import logging
from copy import deepcopy
from functools import reduce
from operator import or_
from typing import Any, Dict, List, Union

import torch
from optimizer_modules import OptimizerModule


logger: logging.Logger = logging.getLogger(__name__)


def flatten(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive flattening function for checkpointing support.

    Args:
        input_dict (Dict[str, Any]): Input dictionary to flatten.

    Returns:
        output_dict (Dict[str, Any]): Flattened dictionary of the input dictionary.

    """

    def flatten_with_parent_keys(
        input_dict: Dict[str, Any], parent_keys: List[str]
    ) -> Dict[str, Any]:
        # Given a dict to flatten containing child key and child value pairs where
        # each child value is either a dict or a tensor, this function recursively
        # flattens each child value and combines those child key and flattened child values
        # pair into a single dictionary with respect to parent keys.
        #
        # Symbolically, if the input_dict is a dict with the following structure:
        # {
        #   child_key1: child_value1, (child_value1 is a dict)
        #   child_key2: child_value2, (child_value2 is a tensor)
        #   ...,
        #   child_keyN: child_valueN, (child_valueN is a dict)
        # },
        # and parent_keys, then the output_dict is the union of flattened dictionaries as follows:
        #   flattened_with_parent_keys(child_value1, parent_keys=parent_keys + [child_key1]) |
        #   { json.dumps(parent_keys + [child_key2]): child_value2 }                         |
        #   ...                                                                              |
        #   flattened_with_parent_keys(child_valueN, parent_keys=parent_keys + [child_keyN])
        #
        # The process above is a reduce bitwise OR operation over all flattened dicts of child values.
        def parse_key_value(
            key: str, value: Union[Dict[str, Any], torch.Tensor]
        ) -> Dict[str, Any]:
            # If the value is a dict, then recursively flatten it.
            # If the value is a tensor, then return a dict with the key being the
            # flattened parent keys and the value being the tensor.
            return (
                flatten_with_parent_keys(
                    input_dict=value,
                    parent_keys=parent_keys + [key],
                )
                if isinstance(value, dict)
                else {
                    json.dumps(parent_keys + [key]): value,
                }
            )

        return reduce(
            or_,
            (
                parse_key_value(key=child_key, value=child_value)
                for child_key, child_value in input_dict.items()
            ),
            {},
        )

    return flatten_with_parent_keys(input_dict=input_dict, parent_keys=[])


def unflatten(
    flattened_dict: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for flattened_key, value in flattened_dict.items():
        *parent_keys, key = json.loads(flattened_key)
        immediate_input_dict = reduce(
            lambda result_iter, parent_key: result_iter.setdefault(parent_key, {}),
            parent_keys,
            result,
        )
        immediate_input_dict[key] = value
    return result


def update_param_state_dict_object(
    current_param_state_dict: Dict[str, Any],
    param_state_dict_to_load: Dict[str, Any],
    enable_missing_key_check: bool = True,
) -> None:

    for k, v in current_param_state_dict.items():
        if k not in param_state_dict_to_load:
            if enable_missing_key_check:
                raise KeyError(f"Key {k} not found in state dict to load.")
            else:
                logger.warning(f"Key {k} not found in state dict to load.")
                continue

        if isinstance(v, dict):
            update_param_state_dict_object(
                v,
                param_state_dict_to_load[k],
                enable_missing_key_check,
            )
        elif hasattr(v, "load_state_dict") and callable(v.load_state_dict):
            v.load_state_dict(param_state_dict_to_load[k])
        elif isinstance(v, torch.Tensor):
            v.detach().copy_(param_state_dict_to_load[k])
        else:
            current_param_state_dict[k] = deepcopy(param_state_dict_to_load[k])


def extract_state_dict_content(
    input_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Converts nested dictionary with objects with state dict functionality.

    Args:
        input_dict (Dict[str, Any]): Nested dictionary containing objects with
            state dict functionality.

    Output:
        output_dict (Dict[str, Any]): Nested dictionary where the terminal values
            cannot have state dict functionality.

    """

    def parse_value(
        value: Union[Dict[str, Any], torch.Tensor, OptimizerModule]
    ) -> Union[Dict[str, Any], torch.Tensor]:
        if isinstance(value, dict):
            return extract_state_dict_content(value)
        elif isinstance(value, OptimizerModule):
            return value.state_dict()
        else:
            return value

    return {k: parse_value(v) for k, v in input_dict.items()}
