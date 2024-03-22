"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from copy import deepcopy
from typing import Any, Iterable, Optional

import torch
from torch.optim.optimizer import StateDict

logger: logging.Logger = logging.getLogger(__name__)


class OptimizerModule:
    r"""
    Optimizer module that supports state_dict and load_state_dict functions that recursively
    constructs the state dictionary by examining other OptimizerModule objects. Similar to
    nn.Module but "trims the fat" by removing unnecessary functions for more general optimizer
    modules.

    When generating the state_dict, looks at the internal dictionary and recursively calls state_dict
    on other optimizer modules.

    """

    def state_dict(
        self,
        destination: Optional[StateDict] = None,
        keep_vars: bool = False,
        store_non_tensors: bool = False,
    ) -> StateDict:
        r"""Returns a nested state dictionary containing a whole internal
        dict of the module. OptimizerModules and other common data structures
        are represented by a dictionary within the dict.

        .. warning::
            Please avoid the use of argument ``destination`` as it is not
            designed for end-users.

        Args:
            destination (Optional[StateDict]): If provided, the state of module will
                be updated into the dict and the same object is returned.
                Otherwise, an ``OrderedDict`` will be created and returned.
                Default: ``None``.
            keep_vars (bool): by default the :class:`~torch.Tensor` s
                returned in the state dict are detached from autograd. If it's
                set to ``True``, detaching will not be performed.
                Default: ``False``.
            store_non_tensors (bool): flag for storing non-tensor
                objects. Default: ``False``.

        Returns:
            dict:
                a dictionary containing a whole state of the module

        """

        def save_to_state_dict(
            states: Iterable[Any],
            destination: StateDict,
        ) -> None:
            r"""Saves module state to `destination` dictionary, containing a state
            of the module, but not its descendants. This is called on every
            submodule in :meth:`~OptimizerModule.state_dict`.

            In rare cases, subclasses can achieve class-specific behavior by
            overriding this method with custom logic.

            Args:
                states (Iterable[Any]): iterable that gives tuples of values to be stored
                    in destination dict
                destination (StateDict): a dict where state will be stored

            """

            for key, value in states:
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
                    save_to_state_dict(
                        states=value.items(),
                        destination=destination[key],
                    )
                elif isinstance(value, (list, tuple, set)):
                    destination[key] = {}
                    save_to_state_dict(
                        states=enumerate(value),
                        destination=destination[key],
                    )
                elif store_non_tensors:
                    destination[key] = value

        if destination is None:
            destination = {}

        save_to_state_dict(self.__dict__.items(), destination)
        return destination

    def load_state_dict(
        self, state_dict: StateDict, store_non_tensors: bool = False
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
            state_dict (StateDict): State dictionary to load
            store_non_tensors (bool): Load non-tensor objects

        """

        def load_from_new_state_to_old_state(
            old_state: StateDict, new_state: StateDict
        ) -> StateDict:
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
                old_state |= {
                    key: load_from_new_state_to_old_state(
                        old_state=old_value,
                        new_state=new_state[key],
                    )
                    for key, old_value in old_state.items()
                    if key in new_state
                }
            elif isinstance(old_state, (list, tuple, set)):
                old_state = type(old_state)(
                    (
                        load_from_new_state_to_old_state(
                            old_state=old_value,
                            new_state=new_state[i],
                        )
                        if store_non_tensors
                        or isinstance(
                            old_value,
                            (torch.Tensor, dict, list, tuple, set, OptimizerModule),
                        )
                        else old_value
                    )
                    for i, old_value in enumerate(old_state)
                )
            elif store_non_tensors:
                if type(old_state) is not type(new_state):
                    logger.warning(
                        f"Types of old value {type(old_state)} and new value {type(new_state)} do not match! Continuing..."
                    )
                    return old_state
                old_state = deepcopy(new_state)

            return old_state

        # load state
        load_from_new_state_to_old_state(old_state=self.__dict__, new_state=state_dict)
