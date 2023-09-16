"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

from typing import Any, Dict, Optional


def _build_full_key(key: str, key_prefix: Optional[str] = None) -> str:
    return f"{key_prefix}.{key}" if key_prefix is not None else key


def _flatten(
    input_dict: Dict[str, Any],
    output_dict: Dict[str, Any],
    key_prefix: Optional[str] = None,
) -> None:
    """Recursive flattening function for checkpointing support.

    Args:
        input_dict (Dict[str, Any]): Input dictionary to flatten.
        output_dict (Dict[str, Any]): Flattened dictionary.
        key_prefix (str): Optional prefix for flattening. (Default: None)

    """
    for k, v in input_dict.items():
        key = _build_full_key(k, key_prefix)
        if key in output_dict:
            raise KeyError(
                f"{key} already exists in output. Overwriting is not allowed. "
                f"If {k} is desired at this level, please consider updating "
                f"parent level keys if possible as a workaround."
            )
        if isinstance(v, dict):
            _flatten(v, output_dict, key_prefix=key)
            continue
        output_dict[key] = v


def flatten_state_dict(
    input_dict: Dict[str, Any],
    prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Flattens state dictionary.

    Used for supporting distributed checkpointing solution.

    Args:
        input_dict (Dict[str, Any]): Input dictionary to flatten.
        prefix (str): Optional prefix for dictionary. (Default: None)

    Returns:
        output_dict (Dict[str, Any]): Flattened dictionary.

    """
    output_dict: Dict[str, Any] = {}
    _flatten(input_dict, output_dict, prefix)
    return output_dict
