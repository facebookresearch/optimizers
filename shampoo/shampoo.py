"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging

import torch
import torch.distributed as dist
from hpc.trainer.shampoo_utils import (
    BlockShampooPreconditioner,
    AdagradPreconditioner,
    ShampooPreconditioner,
    LargeDimMethod,
    RootInvMethod,
    GraftingType,
)
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class Shampoo(Optimizer):
    """Implements Shampoo algorithm.

    See details in:
    - https://arxiv.org/pdf/1802.09568.pdf
    - https://arxiv.org/pdf/2002.09018.pdf

    If root_inv_dist = True, assigns each parameter's preconditioners to different GPUs in a
    round-robin fashion.

    Uses infinity norm to evaluate residuals and errors.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        betas (Tuple[float, float], optional): coefficients used for computing running averages
            of gradient and its square (default: (0.9, 1.0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-12)
        bias_correction (bool, optional): flag for using bias correction (default: False)
        adam_w_mode (bool, optional): Flag for using AdamW-style weight decay (default: True)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        update_freq (int, optional): frequency for updating inverse preconditioner (default: 1)
        init_delay (int, optional): initial delay before starting to compute root inverse (default: 0)
        threshold (int, optional): threshold for switching to diagonal preconditioner (default: 1024)
        preconditioner_dtype (torch.dtype, optional): data type for preconditioner (default: torch.double)
        large_dim_method (LargeDimMethod, optional): method for handling large scale tensors. (default: LargeDimMethod.ADAGRAD)
        root_inv_method (RootInvMethod, optional): method for computing root inverse (default: RootInvMethod.EIGEN)
        root_inv_dist (bool, optional): distributes root inverse computation across multiple GPU workers (default: True)
        merge_dims (bool, optional): merge dimensions if possible while respecting threshold. (default: True)
        grafting_type (GraftingType, optional): Selects grafting method. (Default: GraftingType.ADAGRAD)
        grafting_epsilon (float, optional): Epsilon for grafting method. (Default: 1e-3)
        grafting_beta2 (float, optional): Exponential moving average factor for grafting method. (Default: 1.0)
        debug_mode (bool, optional): flag for debug mode (default: True)

    """

    def __init__(
        self,
        params,
        lr=1e-2,
        betas=(0.9, 1.0),
        epsilon=1e-12,
        bias_correction=False,
        adam_w_mode=True,
        weight_decay=0.0,
        update_freq=100,
        init_delay=1000,
        threshold=1024,
        preconditioner_dtype=torch.float,
        large_dim_method=LargeDimMethod.BLOCKING,
        root_inv_method=RootInvMethod.EIGEN,
        root_inv_dist=True,
        merge_dims=True,
        grafting_type=GraftingType.ADAGRAD,
        grafting_epsilon=1e-3,
        grafting_beta2=1.0,
        debug_mode=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if betas[0] < 0.0 or betas[0] >= 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if betas[1] <= 0.0 or betas[1] > 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if grafting_beta2 <= 0.0 or grafting_beta2 > 1.0:
            raise ValueError(f"Invalid grafting beta parameter: {grafting_beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if grafting_epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon value: {grafting_epsilon}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
            "epsilon": epsilon,
            "grafting_epsilon": grafting_epsilon,
            "grafting_beta2": grafting_beta2,
        }
        super(Shampoo, self).__init__(params, defaults)

        self.threshold = threshold
        self.update_freq = update_freq
        self.iter = 0
        self.init_delay = init_delay
        self.debug_mode = debug_mode
        self.root_inv_method = root_inv_method
        self.root_inv_dist = root_inv_dist
        self.merge_dims = merge_dims
        self.large_dim_method = large_dim_method
        self.adam_w_mode = adam_w_mode
        self.preconditioner_dtype = preconditioner_dtype
        self.bias_correction = bias_correction
        self.grafting_type = grafting_type
        self.grafting_epsilon = grafting_epsilon
        self.grafting_beta2 = grafting_beta2
        self.parameter_count = 0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                beta1, _ = group["betas"]
                if beta1 != 0:
                    state["exp_avg"] = None

        self._initialize_preconditioners()

    @torch.no_grad()
    def _initialize_preconditioners(self):
        """Initialize Shampoo preconditioners and inverse preconditioners."""

        # iterate through each parameter (and parameter group)
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                # extract state
                state = self.state[p]
                dims = torch.tensor(p.shape)

                # Uses Adagrad if larger than threshold
                if self.large_dim_method == LargeDimMethod.ADAGRAD:
                    if torch.any(dims > self.threshold):
                        state["preconditioners"] = AdagradPreconditioner(
                            p,
                            beta2=group["betas"][1],
                            epsilon=group["epsilon"],
                            bias_correction=self.bias_correction,
                            idx=idx,
                        )
                    else:
                        state["preconditioners"] = ShampooPreconditioner(
                            p,
                            beta2=group["betas"][1],
                            epsilon=group["epsilon"],
                            bias_correction=self.bias_correction,
                            diagonal_threshold=self.threshold,
                            dtype=self.preconditioner_dtype,
                            root_inv_method=self.root_inv_method,
                            idx=idx,
                            init_delay=self.init_delay,
                            grafting_type=self.grafting_type,
                            grafting_beta2=self.grafting_beta2,
                            grafting_epsilon=self.grafting_epsilon,
                        )

                # Uses diagonal preconditioners if larger than threshold
                elif self.large_dim_method == LargeDimMethod.DIAGONAL:
                    state["preconditioners"] = ShampooPreconditioner(
                        p,
                        beta2=group["betas"][1],
                        epsilon=group["epsilon"],
                        bias_correction=self.bias_correction,
                        diagonal_threshold=self.threshold,
                        dtype=self.preconditioner_dtype,
                        root_inv_method=self.root_inv_method,
                        idx=idx,
                        init_delay=self.init_delay,
                        grafting_type=self.grafting_type,
                        grafting_beta2=self.grafting_beta2,
                        grafting_epsilon=self.grafting_epsilon,
                    )

                # Uses blocking if larger than threshold
                elif self.large_dim_method == LargeDimMethod.BLOCKING:
                    state["preconditioners"] = BlockShampooPreconditioner(
                        p,
                        beta2=group["betas"][1],
                        epsilon=group["epsilon"],
                        bias_correction=self.bias_correction,
                        block_size=self.threshold,
                        dtype=self.preconditioner_dtype,
                        root_inv_method=self.root_inv_method,
                        idx=idx,
                        merge_dims=self.merge_dims,
                        init_delay=self.init_delay,
                        grafting_type=self.grafting_type,
                        grafting_beta2=self.grafting_beta2,
                        grafting_epsilon=self.grafting_epsilon,
                    )

                # increase parameter count
                self.parameter_count += state["preconditioners"].parameter_count

        # log total number of parameters for optimizer
        logger.info(f"Total Parameter Count: {self.parameter_count}")

    @torch.no_grad()
    def _compute_root_inverse(self):
        """Preprocesses and computes root inverse of each preconditioner. Syncs root inverse across different
        workers."""

        # if debugging, generate metrics to track
        if self.debug_mode:
            max_cond_number = torch.tensor(1.0)
            max_residual = torch.tensor(0.0)
            min_eigenvalue_gap = torch.tensor(float('inf'))

        # loop through parameters
        for group in self.param_groups:

            # get world size
            if self.root_inv_dist:
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size = 0
                rank = None

            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # distribute computation between workers if world_size > 0
                # NOTE: This can be further optimized by better load-balancing in the future.
                if world_size == 0 or (world_size > 0 and idx % world_size == rank):

                    # Initialize state
                    state = self.state[p]

                    # compute Shampoo preconditioner
                    if isinstance(
                        state["preconditioners"], ShampooPreconditioner
                    ) or isinstance(
                        state["preconditioners"], BlockShampooPreconditioner
                    ):
                        residuals, cond_numbers, eigenvalue_gaps = state[
                            "preconditioners"
                        ].compute_root_inverse(debug=self.debug_mode)

                        # update maximum residual and condition number
                        if self.debug_mode:
                            if (
                                self.root_inv_method == RootInvMethod.EIGEN
                                and cond_numbers is not None
                                and cond_numbers.nelement() > 0
                            ):
                                max_cond_number = torch.maximum(
                                    max_cond_number, torch.max(cond_numbers)
                                )
                            if (
                                self.root_inv_method == RootInvMethod.EIGEN
                                and eigenvalue_gaps is not None
                                and eigenvalue_gaps.nelement() > 0
                            ):
                                min_eigenvalue_gap = torch.minimum(
                                    min_eigenvalue_gap, torch.min(eigenvalue_gaps)
                                )
                            if residuals is not None and residuals.nelement() > 0:
                                max_residual = torch.maximum(
                                    max_residual, torch.max(residuals)
                                )

        if self.debug_mode:
            # print statistics for debugging
            logger.info(f"Iteration: {self.iter}")
            logger.info(f"Max Condition Number: {max_cond_number}")
            logger.info(f"Min Eigenvalue Gap: {min_eigenvalue_gap}")
            logger.info(f"Max Residual (|X^-r - A| / max(1, |A|)): {max_residual}")

    @torch.no_grad()
    def _broadcast_inv_preconditioners(self):
        """Broadcasts inverse preconditioners."""

        for group in self.param_groups:
            # get world size
            world_size = dist.get_world_size()

            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                # get state, dimensions, and order
                state = self.state[p]
                src_rank = idx % world_size
                state["preconditioners"].broadcast(src_rank)

    @torch.no_grad()
    def _update_preconditioners(self):
        """Updates preconditioners.

        Note: If using L2-regularization/weight decay, it is computed within this function and therefore should not be
        recomputed elsewhere.

        """

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                weight_decay = group["weight_decay"]

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

                    state["preconditioners"].update_preconditioners(grad)

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

        self.iter += 1

        # update preconditioners
        self._update_preconditioners()

        # compute root inverse if delay is 0
        if self.iter % self.update_freq == 0 and self.iter >= self.init_delay:
            self._compute_root_inverse()
            if self.root_inv_dist:
                self._broadcast_inv_preconditioners()

        # perform update
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Initialize gradient, states, and dim for parameter
                grad = p.grad
                state = self.state[p]
                beta1, _ = group["betas"]
                weight_decay = group["weight_decay"]
                lr = group["lr"]

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception(
                        "Sparse parameters are not currently supported by Shampoo."
                    )

                # Dense case
                else:

                    # incorporate momentum
                    if beta1 != 0:
                        # compute bias corrections if necessary
                        bias_correction1 = 1.0
                        if self.bias_correction and beta1 < 1:
                            bias_correction1 -= beta1 ** self.iter

                        # modify grad with momentum term
                        if state["exp_avg"] is None:
                            state["exp_avg"] = torch.zeros_like(
                                grad, memory_format=torch.preserve_format
                            )
                        buf = state["exp_avg"]
                        buf.mul_(beta1).add_(grad, alpha=1 - beta1)
                        grad.copy_(buf / bias_correction1)

                    # perform AdamW weight decay
                    if self.adam_w_mode and weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # compute preconditioned gradient and update parameters
                    state["preconditioners"].precondition_and_update(p, grad, lr)

        return loss
