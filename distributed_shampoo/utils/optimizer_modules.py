"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

logger = logging.getLogger(__name__)

# TODO: Support additional data structures
COMPATIBLE_DATA_STRUCTURES = (list, tuple, set)
ALL_CLASSES = (torch.Tensor, dict) + COMPATIBLE_DATA_STRUCTURES


def are_states_equal(prev_state: Any, new_state: Any) -> bool:
    r"""
    Comparison function that checks whether or not two nested state dictionaries containing tensors
    or other custom data types are equal.

    Useful for debugging purposes.

    Args:
        prev_state (Any): State to compare.
        new_state (Any): State to compare.

    """

    if type(prev_state) != type(new_state):
        return False

    if isinstance(prev_state, torch.Tensor):
        return torch.equal(prev_state, new_state)
    elif isinstance(prev_state, dict):
        prev_keys = prev_state.keys()
        if prev_keys != new_state.keys():
            return False
        return all(
            [are_states_equal(prev_state[key], new_state[key]) for key in prev_keys]
        )
    else:
        return prev_state == new_state


class OptimizerModule:
    r"""
    Optimizer module that supports state_dict and load_state_dict functions that recursively
    constructs the state dictionary by examining other OptimizerModule objects. Similar to
    nn.Module but "trims the fat" by removing unnecessary functions for more general optimizer
    modules.

    When generating the state_dict, looks at the internal dictionary and recursively calls state_dict
    on other optimizer modules.

    """

    def _save_to_state_dict(
        self,
        states: Iterable,
        destination: Dict,
        keep_vars: bool,
        store_non_tensors: bool,
    ):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~OptimizerModule.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Args:
            states (Iterable): iterable that gives tuples of values to be stored
                in destination dict
            destination (dict): a dict where state will be stored
            keep_vars (bool): keep variables for tensor
            store_non_tensors (bool): flag for storing non-tensor objects

        """

        for key, value in states:
            # TODO: Add case for ShardedTensor
            if isinstance(value, torch.Tensor):
                destination[key] = value if keep_vars else value.detach()
            elif isinstance(value, OptimizerModule):
                destination[key] = {}
                value.state_dict(
                    destination=destination[key],
                    keep_vars=keep_vars,
                    store_non_tensors=store_non_tensors,
                )
            elif isinstance(value, dict):
                destination[key] = {}
                self._save_to_state_dict(
                    states=value.items(),
                    destination=destination[key],
                    keep_vars=keep_vars,
                    store_non_tensors=store_non_tensors,
                )
            elif isinstance(value, COMPATIBLE_DATA_STRUCTURES):
                destination[key] = {}
                self._save_to_state_dict(
                    states=enumerate(value),
                    destination=destination[key],
                    keep_vars=keep_vars,
                    store_non_tensors=store_non_tensors,
                )
            elif store_non_tensors:
                destination[key] = value

    def state_dict(
        self,
        destination: Optional[Dict] = None,
        keep_vars: bool = False,
        store_non_tensors: bool = False,
    ) -> Dict[str, Any]:
        r"""Returns a nested state dictionary containing a whole internal
        dict of the module. OptimizerModules and other common data structures
        are represented by a dictionary within the dict.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (dict, optional): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            store_non_tensors (bool, optional): flag for storing non-tensor
                objects. Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        """

        if destination is None:
            destination = {}

        self._save_to_state_dict(
            self.__dict__.items(), destination, keep_vars, store_non_tensors
        )

        return destination

    def _load_from_state_dict(
        self, old_state: Any, new_state: Any, store_non_tensors: bool
    ) -> Any:
        if isinstance(old_state, torch.Tensor):
            if not isinstance(new_state, torch.Tensor):
                logger.warning(
                    f"Both old state {old_state} and new state {new_state} must be tensors! Continuing..."
                )
                return old_state
            old_state.detach().copy_(new_state)
        elif isinstance(old_state, OptimizerModule):
            old_state.load_state_dict(new_state, store_non_tensors)
        elif isinstance(old_state, dict):
            if not isinstance(new_state, dict):
                logger.warning(
                    f"Both old state {old_state} and new_state {new_state} must be dicts! Continuing..."
                )
                return old_state
            for key, old_value in old_state.items():
                if key in new_state:
                    old_state[key] = self._load_from_state_dict(
                        old_state=old_value,
                        new_state=new_state[key],
                        store_non_tensors=store_non_tensors,
                    )
        elif isinstance(old_state, COMPATIBLE_DATA_STRUCTURES):
            old_state = type(old_state)(
                self._load_from_state_dict(
                    old_state=old_value,
                    new_state=new_state[i],
                    store_non_tensors=store_non_tensors,
                )
                if store_non_tensors
                or isinstance(old_value, ALL_CLASSES + (OptimizerModule,))
                else old_value
                for i, old_value in enumerate(old_state)
            )
        elif store_non_tensors:
            if type(old_state) != type(new_state):
                logger.warning(
                    f"Types of old value {type(old_state)} and new value {type(new_state)} do not match! Continuing..."
                )
                return old_state
            old_state = deepcopy(new_state)

        return old_state

    def load_state_dict(
        self, state_dict: Mapping[str, Any], store_non_tensors: bool = False
    ) -> None:
        """
        This implementation requires the stored and loaded states to be fully initialized.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves usability
            * avoid state duplication by directly copying into state tensors, e.g.
              optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard if needed
              optimizer.load_state_dict(sd)  # replace param_groups

        Args:
            state_dict (dict): State dictionary to load
            store_non_tensors (bool, optional): Load non-tensor objects

        """

        # load state
        self._load_from_state_dict(self.__dict__, state_dict, store_non_tensors)
