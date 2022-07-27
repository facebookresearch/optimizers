"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
from typing import Tuple

import torch
import torch.distributed as dist

from shampoo_utils import (
    AdagradPreconditioner,
    BlockShampooPreconditioner,
    Grafting,
    GraftingType,
    LargeDimMethod,
    Preconditioner,
    RootInvStrategy,
    ShampooPreconditioner,
)
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)

BETAS = "betas"
EXP_AVG = "exp_avg"
EPSILON = "epsilon"
GRAFTING_BETA2 = "grafting_beta2"
GRAFTING_EPSILON = "grafting_epsilon"
LR = "lr"
MOMENTUM = "momentum"
PARAMS = "params"
PRECONDITIONERS = "preconditioners"
STEP = "step"
WEIGHT_DECAY = "weight_decay"


class DistributedShampoo(Optimizer):
    """Implements distributed Shampoo algorithm.

    See details in:
    - https://arxiv.org/pdf/1802.09568.pdf
    - https://arxiv.org/pdf/2002.09018.pdf

    Uses infinity norm to evaluate residuals and errors.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (Default: 1e-2)
        betas (Tuple[float, float]): coefficients used for computing running averages
            of gradient and its square (Default: (0.9, 1.0))
        epsilon (float): term added to the denominator to improve numerical stability (Default: 1e-12)
        momentum (float): momentum parameter (default: 0.9)
        use_nesterov (bool): uses Nesterov momentum (default: True)
        use_bias_correction (bool): flag for using bias correction (Default: True)
        adam_w_mode (bool): Flag for using AdamW-style weight decay (Default: True)
        weight_decay (float): weight decay (L2 penalty) (Default: 0)
        update_freq (int): frequency for updating inverse preconditioner (Default: 100)
        init_delay (int): initial delay before starting to compute root inverse (Default: 1000)
        threshold (int): threshold for switching to diagonal preconditioner (Default: 1024)
        preconditioner_dtype (torch.dtype): data type for preconditioner (Default: torch.float)
        large_dim_method (LargeDimMethod): method for handling large scale tensors. (Default: LargeDimMethod.BLOCKING)
        root_inv_strategy (RootInvStrategy): distributes root inverse computation across multiple GPU workers using
            specified strategy. (Default: RootInvStrategy.PRECOND)
        use_merge_dims (bool): merge dimensions if possible while respecting threshold. (Default: True)
        grafting_type (GraftingType): Selects grafting method. (Default: GraftingType.ADAGRAD)
        grafting_epsilon (float): Epsilon for grafting method. (Default: 1e-3)
        grafting_beta2 (float): Exponential moving average factor for grafting method. (Default: 1.0)

    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        epsilon: float = 1e-12,
        momentum: float = 0.9,
        use_nesterov: bool = True,
        use_bias_correction: bool = True,
        adam_w_mode: bool = True,
        weight_decay: float = 0.0,
        update_freq: int = 100,
        init_delay: int = 1000,
        threshold: int = 1024,
        preconditioner_dtype: torch.dtype = torch.float,
        large_dim_method: LargeDimMethod = LargeDimMethod.BLOCKING,
        root_inv_strategy: RootInvStrategy = RootInvStrategy.PRECOND,
        use_merge_dims: bool = True,
        grafting_type: GraftingType = GraftingType.ADAGRAD,
        grafting_epsilon: float = 1e-3,
        grafting_beta2: float = 1.0,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 < betas[1] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 < grafting_beta2 <= 1.0:
            raise ValueError(f"Invalid grafting beta parameter: {grafting_beta2}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not grafting_epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {grafting_epsilon}")

        super(DistributedShampoo, self).__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                MOMENTUM: momentum,
                WEIGHT_DECAY: weight_decay,
                EPSILON: epsilon,
                GRAFTING_EPSILON: grafting_epsilon,
                GRAFTING_BETA2: grafting_beta2,
            },
        )

        self.threshold = threshold
        self.update_freq = update_freq
        self.init_delay = init_delay
        self.root_inv_strategy = root_inv_strategy
        self.use_merge_dims = use_merge_dims
        self.large_dim_method = large_dim_method
        self.adam_w_mode = adam_w_mode
        self.preconditioner_dtype = preconditioner_dtype
        self.use_bias_correction = use_bias_correction
        self.grafting_type = grafting_type
        self.grafting_epsilon = grafting_epsilon
        self.grafting_beta2 = grafting_beta2
        self.parameter_count = 0
        self.use_nesterov = use_nesterov
        self.world_size = (
            dist.get_world_size()
            if self.root_inv_strategy != RootInvStrategy.NONE
            else 0
        )

        self._initialize_momentum_states()
        self._initialize_preconditioners()
        self._assign_preconditioners_to_ranks()

    @torch.no_grad()
    def _initialize_momentum_states(self):
        """Initialize exponential average and momentum."""
        for group in self.param_groups:
            beta1, _ = group[BETAS]
            momentum = group[MOMENTUM]
            for p in group[PARAMS]:
                state = self.state[p]
                state[STEP] = 0
                if beta1 != 0:
                    state[EXP_AVG] = None
                if momentum != 0:
                    state[MOMENTUM] = None

    @torch.no_grad()
    def _initialize_preconditioners(self):
        """Initialize Shampoo preconditioners and inverse preconditioners."""

        for group in self.param_groups:
            for idx, p in enumerate(group[PARAMS]):
                state = self.state[p]
                dims = torch.tensor(p.shape)

                # Uses Adagrad if larger than threshold
                if self.large_dim_method == LargeDimMethod.ADAGRAD:
                    if torch.any(dims > self.threshold):
                        state[PRECONDITIONERS] = AdagradPreconditioner(
                            p,
                            beta2=group[BETAS][1],
                            epsilon=group[EPSILON],
                            use_bias_correction=self.use_bias_correction,
                            idx=idx,
                        )
                    else:
                        state[PRECONDITIONERS] = ShampooPreconditioner(
                            p,
                            beta2=group[BETAS][1],
                            epsilon=group[EPSILON],
                            use_bias_correction=self.use_bias_correction,
                            diagonal_threshold=self.threshold,
                            dtype=self.preconditioner_dtype,
                            root_inv_strategy=self.root_inv_strategy,
                            idx=idx,
                            init_delay=self.init_delay,
                            grafting_type=self.grafting_type,
                            grafting_beta2=self.grafting_beta2,
                            grafting_epsilon=self.grafting_epsilon,
                        )

                # Uses diagonal preconditioners if larger than threshold
                elif self.large_dim_method == LargeDimMethod.DIAGONAL:
                    state[PRECONDITIONERS] = ShampooPreconditioner(
                        p,
                        beta2=group[BETAS][1],
                        epsilon=group[EPSILON],
                        use_bias_correction=self.use_bias_correction,
                        diagonal_threshold=self.threshold,
                        dtype=self.preconditioner_dtype,
                        idx=idx,
                        init_delay=self.init_delay,
                        grafting_type=self.grafting_type,
                        grafting_beta2=self.grafting_beta2,
                        grafting_epsilon=self.grafting_epsilon,
                    )

                # Uses blocking if larger than threshold
                elif self.large_dim_method == LargeDimMethod.BLOCKING:
                    state[PRECONDITIONERS] = BlockShampooPreconditioner(
                        p,
                        beta2=group[BETAS][1],
                        epsilon=group[EPSILON],
                        use_bias_correction=self.use_bias_correction,
                        block_size=self.threshold,
                        dtype=self.preconditioner_dtype,
                        root_inv_strategy=self.root_inv_strategy,
                        idx=idx,
                        use_merge_dims=self.use_merge_dims,
                        init_delay=self.init_delay,
                        grafting_type=self.grafting_type,
                        grafting_beta2=self.grafting_beta2,
                        grafting_epsilon=self.grafting_epsilon,
                    )

                else:
                    raise ValueError(
                        "Large dim method "
                        + self.large_dim_method
                        + " is not implemented!"
                    )

                # increase parameter count
                self.parameter_count += state[PRECONDITIONERS].parameter_count

        # log total number of parameters for optimizer
        logger.info(f"Total Parameter Count: {self.parameter_count}")

    @torch.no_grad()
    def _assign_preconditioners_to_ranks(self):
        """Assign each preconditioner to a rank depending on strategy.

        This method uses the following strategy:
            RootInvStrategy.NONE: All workers are independently responsible for all preconditioners.
            RootInvStrategy.PARAM: Preconditioners are distributed based on their parameter in a round-robin fashion.
            RootInvStrategy.BLOCK: Preconditioners are distributed based on their block in a round-robin fashion.
            RootInvStrategy.PRECOND: Preconditioners are distributed in a round-robin fashion.

        """

        if self.root_inv_strategy == RootInvStrategy.NONE:
            return
        elif self.root_inv_strategy in (
            RootInvStrategy.PARAM,
            RootInvStrategy.BLOCK,
            RootInvStrategy.PRECOND,
        ):
            rank = 0
            for group in self.param_groups:
                for p in group[PARAMS]:
                    state = self.state[p]
                    if isinstance(
                        state[PRECONDITIONERS],
                        (ShampooPreconditioner, BlockShampooPreconditioner),
                    ):
                        rank = state[PRECONDITIONERS].assign_preconditioners_rank(
                            rank, self.world_size
                        )
                    if self.root_inv_strategy == RootInvStrategy.PARAM:
                        rank += 1
        else:
            raise NotImplementedError(
                "Root inverse strategy is not implemented! Specified root inverse strategy is "
                + str(self.root_inv_strategy)
                + "."
            )

    @torch.no_grad()
    def _compute_root_inverse(self):
        """Root inverse computation across all preconditioners/parameters."""
        # loop through parameters
        for group in self.param_groups:
            for p in group[PARAMS]:
                if p.grad is None:
                    continue

                # Initialize state
                state = self.state[p]

                # compute Shampoo preconditioner
                if isinstance(
                    state[PRECONDITIONERS],
                    (ShampooPreconditioner, BlockShampooPreconditioner),
                ):
                    state[PRECONDITIONERS].compute_root_inverse()

    @torch.no_grad()
    def _broadcast_inv_preconditioners(self):
        """Broadcasts inverse preconditioners."""
        for group in self.param_groups:
            for p in group[PARAMS]:
                state = self.state[p]
                if PRECONDITIONERS in state:
                    state[PRECONDITIONERS].broadcast()

    @torch.no_grad()
    def _update_preconditioners(self):
        """Updates preconditioners.

        Note: If using L2-regularization/weight decay, it is computed within this function and therefore should not be
        recomputed elsewhere.

        """

        for group in self.param_groups:
            for p in group[PARAMS]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                weight_decay = group[WEIGHT_DECAY]

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception(
                        "Sparse parameters are not currently supported by Shampoo."
                    )

                # Dense case
                else:
                    # incorporate L2 regularization / weight decay
                    if not self.adam_w_mode and weight_decay != 0:
                        grad.add_(p, alpha=weight_decay)

                    state[PRECONDITIONERS].update_preconditioners(grad)

    @torch.no_grad()
    def _iterate_step(self):
        for group in self.param_groups:
            for p in group[PARAMS]:
                state = self.state[p]
                state[STEP] += 1
        return state[STEP]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        iteration = self._iterate_step()

        # update preconditioners
        self._update_preconditioners()

        # compute root inverse if delay is 0
        if iteration % self.update_freq == 0 and iteration >= self.init_delay:
            self._compute_root_inverse()
            if self.root_inv_strategy != RootInvStrategy.NONE:
                self._broadcast_inv_preconditioners()

        # perform update
        for group in self.param_groups:
            beta1, _ = group[BETAS]
            momentum = group[MOMENTUM]
            weight_decay = group[WEIGHT_DECAY]
            lr = group[LR]

            for p in group[PARAMS]:
                if p.grad is None:
                    continue

                # Initialize gradient, states, and dim for parameter
                grad = p.grad
                state = self.state[p]

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception(
                        "Sparse parameters are not currently supported by Shampoo."
                    )

                # Dense case
                else:
                    # incorporate first-moment estimation
                    if beta1 != 0:
                        # compute bias corrections if necessary
                        bias_correction1 = 1.0
                        if self.use_bias_correction and beta1 < 1:
                            bias_correction1 -= beta1**iteration

                        # modify grad with momentum term
                        if state[EXP_AVG] is None:
                            state[EXP_AVG] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        buf = state[EXP_AVG]
                        buf.mul_(beta1).add_(grad, alpha=1 - beta1)
                        grad.copy_(buf / bias_correction1)

                    # compute preconditioned gradient and update parameters
                    if momentum == 0.0:
                        # perform AdamW weight decay
                        if self.adam_w_mode and weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)

                        state[PRECONDITIONERS].precondition_and_update(p, grad, lr)

                    else:
                        # compute preconditioned gradient
                        search_direction = state[PRECONDITIONERS].precondition(grad)

                        # add AdamW weight decay
                        if self.adam_w_mode and weight_decay != 0:
                            search_direction.add_(p, alpha=weight_decay)

                        # initialize momentum term if necessary
                        if state[MOMENTUM] is None:
                            state[MOMENTUM] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )

                        # apply momentum / primal iterate averaging
                        state[MOMENTUM].mul_(group[MOMENTUM]).add_(search_direction)

                        # incorporate Nesterov momentum
                        if self.use_nesterov:
                            search_direction.add_(
                                state[MOMENTUM], alpha=group[MOMENTUM]
                            )
                        else:
                            search_direction = state[MOMENTUM]

                        # update parameters
                        p.add_(search_direction, alpha=-lr)

        return loss

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.
        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict["param_groups"]

        if len(groups) != len(saved_groups):
            raise ValueError(
                "loaded state dict has a different number of " "parameter groups"
            )
        param_lens = (len(g[PARAMS]) for g in groups)
        saved_lens = (len(g[PARAMS]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable((g[PARAMS] for g in saved_groups)),
                chain.from_iterable((g[PARAMS] for g in groups)),
            )
        }

        def cast(param, value, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                # Floating-point types are a bit special here. They are the only ones
                # that are assumed to always match the type of params.
                # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
                if key != STEP:
                    if param.is_floating_point():
                        value = value.to(param.dtype)
                    value = value.to(param.device)
                return value
            elif isinstance(value, dict):
                return {k: cast(param, v, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v) for v in value)
            elif isinstance(value, Preconditioner):
                value.to(param.device)
                return value
            elif isinstance(value, Grafting):
                value.to(param.device)
                return value
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v)
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group[PARAMS] = group[PARAMS]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})
