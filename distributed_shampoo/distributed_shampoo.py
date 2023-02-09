"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import bisect
import itertools
import logging
import os
from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from distributed_shampoo.shampoo_utils import (
    AdagradPreconditioner,
    BlockShampooPreconditioner,
    GraftingType,
    LargeDimMethod,
    Preconditioner,
    DistStrategy,
    ShampooPreconditioner,
)

logger = logging.getLogger(__name__)

BETAS = "betas"
EXP_AVG = "exp_avg"
EPSILON = "epsilon"
GRAFTING_BETA2 = "grafting_beta2"
GRAFTING_EPSILON = "grafting_epsilon"
GRAFTING_MOMENTUM = "grafting_momentum"
LR = "lr"
MOMENTUM = "momentum"
PARAMS = "params"
PRECONDITIONERS = "preconditioners"
STEP = "step"
WEIGHT_DECAY = "weight_decay"


class DistributedShampoo(torch.optim.Optimizer):
    """Implements distributed Shampoo algorithm.

    Implemented by:
        Hao-Jun Michael Shi (Meta Platforms, Inc.)
        Tsung-Hsien Lee (Cruise)

    with support from:
        Rohan Anil (Google), Vineet Gupta (Google), Shintaro Iwasaki (Meta), Zhijing Li (Meta), Dheevatsa Mudigere (Nvidia),
        Mike Rabbat (Meta), and Kaushik Rangadurai (Meta).

    Partly based on the work in:
    - https://arxiv.org/pdf/1802.09568.pdf
    - https://arxiv.org/pdf/2002.09018.pdf

    Uses infinity norm to evaluate residuals and errors. By default, grafts from Adagrad.

    --------
    Features
    --------

    1. Layerwise Grafting: In order to tune Shampoo, we can "graft" a layer-wise learning rate schedule from a previous method
        and apply it to Shampoo. This is performed by taking the norm of the layer-wise step of the grafted method, normalizing
        the Shampoo step, and re-scaling the normalized Shampoo step by the product of the norm of the grafted step + learning rate.

        This may be interpreted as an additional block re-scaling of the entire Shampoo preconditioner.
        This is the key ingredient to making Shampoo work in practice.

        We support the following methods:
            - GraftingType.NONE: Performs no grafting.
            - GraftingType.SGD: Grafts the stochastic gradient method.
            - GraftingType.ADAGRAD: Grafts the Adagrad method.
            - GraftingType.RMSPROP: Grafts the RMSProp method.
            - GraftingType.ADAM: Grafts the Adam method.
            - GraftingType.ADAGRAD_NORMALIZED: Grafts the Adagrad method with normalized gradients.
            - GraftingType.RMSPROP_NORMALIZED: Grafts the RMSProp method with normalized gradients.
            - GraftingType.ADAM_NORMALIZED: Grafts the Adam method with normalized gradients.

        NOTE: These methods do not graft the first-moment component - it is entirely based upon grafting using the diagonal preconditioner.
        If using an exponential moving average of the gradient (or gradient filtering), we can set beta1 as the same value from before, and
        both Shampoo and the grafted method will use the filtered gradient.

    2. Large-Dimensional Tensors: Supports multiple approaches for scaling Shampoo to tensors with large dimensions.
        For simplicity, we explain using a linear layer/matrix parameter, although this is generalizable to higher-order tensors.
        Suppose that W is a m x n matrix, i.e.,

            [[w_11 w_12 ... w_1n]
             [w_21 w_22 ... w_2n]
        W =           :
             [w_m1 w_m2 ... w_mn]]

        - LargeDimMethod.BLOCKING (Default): Given a max_preconditioner_dim tau > 0, blocks W and applies Shampoo to each block, i.e.,
            if tau divides both m, n, then:

                [[W_11 W_12 ... W_1k]
                 [W_21 W_22 ... W_2k]
            W =           :
                 [W_l1 W_l2 ... W_lk]]

            and apply Shampoo to W_ij which is a tau x tau matrix. This can be viewed as further blocking each block of the
            block-diagonal preconditioner.

            Computational cost = O(tau^3)
            Memory cost = 4mn (including root inverse preconditioners)

        - LargeDimMethod.ADAGRAD: Given a max_preconditioner_dim tau > 0, checks if any dimensions of the tensor is greater than tau. If so,
            uses Adagrad preconditioner in place of Shampoo. Corresponds to a diagonal preconditioner.

            Computational cost = O(mn)
            Memory cost = mn

        - LargeDimMethod.DIAGONAL: Given a max_preconditioner_dim tau > 0, uses a diagonal Shampoo preconditioner in place of the full
            Shampoo preconditioner. Corresponds to a (crude) diagonal preconditioner.

            Computational cost = O(mn)
            Memory cost = m + n

    3. Distributed Root Inverse Computation: Supports multi-GPU data-parallel training via torch.distributed by setting dist_strategy.
        Enables multiple strategies for distributing root inverse computation over multiple GPUs:
        - DistStrategy.CROSS_NODE (Default): Distributes preconditioner root inverse computation in
            a per-preconditioner round-robin fashion across all nodes. Performs inter- and intra-node communication
            to synchronize root inverse preconditioners.
        - DistStrategy.INTRA_NODE_ONLY: Distributes preconditioner root inverse computation in
            a per-preconditioner round-robin fashion across GPUs within each node. Performs intra-node communication
            only to synchronize root inverse preconditioners.
        - DistStrategy.NONE: No distributing of the preconditioner root inverse computation.

        Requirements:
        - torch.distributed must be initialized in advance.
        - Currently only supports homogeneous hardware architectures.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate (Default: 1e-2)
        betas (Tuple[float, float]): coefficients used for computing running averages
            of gradient and its square (Default: (0.9, 1.0))
        epsilon (float): term added to the denominator to improve numerical stability (Default: 1e-12)
        momentum (float): momentum parameter (default: 0.)
        weight_decay (float): weight decay (L2 penalty) (Default: 0)
        max_preconditioner_dim (int): maximum preconditioner dimension (Default: 1024)
        precondition_frequency (int): frequency for computing root inverse preconditioner (Default: 1)
        start_preconditioning_step (int): iteration to start computing inverse preconditioner. If -1, uses
            the same value as precondition_frequency. (Default: -1)
        exponent_override (int): exponent to use in Shampoo. (Default: 0)
        exponent_multiplier (float): number to be multiplied to the numerator of the inverse root. (Default: 1.0)
        use_nesterov (bool): uses Nesterov momentum (default: False)
        use_bias_correction (bool): flag for using bias correction (Default: True)
        use_decoupled_weight_decay (bool): Flag for using AdamW-style decoupled weight decay (Default: True)
        use_separate_momentum (bool): Flag for using separate momentum terms between Shampoo and the grafted method.
            (Default: False)
        preconditioner_dtype (torch.dtype): data type for preconditioner (Default: torch.float)
        large_dim_method (LargeDimMethod): method for handling large scale tensors. (Default: LargeDimMethod.BLOCKING)
        dist_strategy (DistStrategy): distributes root inverse computation across multiple GPU workers using
            specified strategy. (Default: DistStrategy.INTRA_NODE_ONLY)
        use_merge_dims (bool): merge dimensions if possible while respecting max_preconditioner_dim. (Default: True)
        grafting_type (GraftingType): selects grafting method. (Default: GraftingType.ADAGRAD)
        grafting_epsilon (float): epsilon for grafting method. (Default: 1e-3)
        grafting_beta2 (float): exponential moving average factor for grafting method. (Default: 1.0)
        debug_mode (bool): debugging mode. Uses more memory to compute error to fp64 case. Must enable logging level to DEBUG. (Default: False)

    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        epsilon: float = 1e-12,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        exponent_override: int = 0,
        exponent_multiplier: float = 1.0,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        use_separate_momentum: bool = False,
        preconditioner_dtype: torch.dtype = torch.float,
        large_dim_method: LargeDimMethod = LargeDimMethod.BLOCKING,
        dist_strategy: DistStrategy = DistStrategy.CROSS_NODE,
        use_merge_dims: bool = True,
        grafting_type: GraftingType = GraftingType.ADAGRAD,
        grafting_epsilon: float = 1e-3,
        grafting_beta2: float = 1.0,
        debug_mode: bool = False,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 < betas[1] <= 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum parameter: {momentum}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not max_preconditioner_dim >= 1:
            raise ValueError(
                f"Invalid max preconditioner dimension: {max_preconditioner_dim}"
            )
        if not precondition_frequency >= 1:
            raise ValueError(
                f"Invalid precondition frequency: {precondition_frequency}"
            )
        if not start_preconditioning_step >= -1:
            raise ValueError(
                f"Invalid start preconditioning step: {start_preconditioning_step}"
            )
        if not exponent_override >= 0:
            raise ValueError(f"Invalid exponent override: {exponent_override}")
        if not 0.0 < grafting_beta2 <= 1.0:
            raise ValueError(f"Invalid grafting beta parameter: {grafting_beta2}")
        if not grafting_epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {grafting_epsilon}")
        if dist_strategy != DistStrategy.NONE:
            if not torch.cuda.is_available():
                raise ValueError("Using distributed version of Shampoo without GPUs!")
            if not dist.is_initialized():
                raise ValueError(
                    "Using distributed version of Shampoo without initializing distributed process group!"
                )

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

        # Initialize algorithm-related fields.
        self._max_preconditioner_dim = max_preconditioner_dim
        self._precondition_frequency = precondition_frequency
        self._exponent_override = exponent_override
        self._exponent_multiplier = exponent_multiplier
        self._dist_strategy = dist_strategy
        self._use_merge_dims = use_merge_dims
        self._large_dim_method = large_dim_method
        self._use_decoupled_weight_decay = use_decoupled_weight_decay
        self._preconditioner_dtype = preconditioner_dtype
        self._use_bias_correction = use_bias_correction
        self._grafting_type = grafting_type
        self._grafting_epsilon = grafting_epsilon
        self._grafting_beta2 = grafting_beta2
        self._parameter_count = 0
        self._use_nesterov = use_nesterov
        self._use_separate_momentum = use_separate_momentum
        self._debug_mode = debug_mode
        if self._use_nesterov and momentum == 0.0:
            logger.warning(
                "Nesterov flag is enabled but momentum parameter is zero! Continuing without using momentum or Nesterov acceleration..."
            )

        if start_preconditioning_step == -1:
            self._start_preconditioning_step = precondition_frequency
            logger.warning(
                f"start_preconditioning_step set to -1. Setting start_preconditioning_step equal to precondition frequency {precondition_frequency} by default."
            )
        else:
            self._start_preconditioning_step = start_preconditioning_step

        # Initialize comms-related fields.
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Currently supports only homogeneous architectures and assumes that the environmental
        # variable LOCAL_WORLD_SIZE is set (for example, through torchrun / torch.distributed.launch).
        #
        # TODO: Need to find way to obtain the number of GPUs / node to support heterogeneous architectures
        # and not rely on an environmental variable.
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self._number_of_gpus_per_node = (
            [local_world_size] * (self._world_size // local_world_size)
            if torch.cuda.is_available()
            and self._dist_strategy == DistStrategy.INTRA_NODE_ONLY
            else [self._world_size]
        )
        if sum(self._number_of_gpus_per_node) != self._world_size:
            raise ValueError(
                f"Sum of number of GPUs per node {self._number_of_gpus_per_node} does not equal world size {self._world_size}!"
            )
        self._root_inv_dist_groups = self._create_root_inv_distributed_groups(
            self._number_of_gpus_per_node
        )

        # Initialize Shampoo preconditioners.
        self._initialize_preconditioners_and_steps()
        self._assign_preconditioners_to_ranks()

    @torch.no_grad()
    def _use_distributed(self) -> bool:
        return self._dist_strategy != DistStrategy.NONE and dist.is_initialized()

    @torch.no_grad()
    def _create_root_inv_distributed_groups(
        self, number_of_gpus_per_node: List[int]
    ) -> Optional[Dict[int, Optional[dist.ProcessGroup]]]:
        """Creates different distributed groups for GPUs within each node. Used for distributing root inverse computation."""
        assert (
            min(number_of_gpus_per_node) > 0
        ), f"Number of GPUs per node {number_of_gpus_per_node} must be greater than 0 on all nodes!"

        if self._dist_strategy == DistStrategy.CROSS_NODE:
            logger.info(
                "Using default (global) distributed process group for distributing root inverse computation with cross-node communication..."
            )
            root_inv_dist_groups = {rank: None for rank in range(self._world_size)}

        elif self._dist_strategy == DistStrategy.INTRA_NODE_ONLY:
            logger.info(
                "Setting up distributed process groups for each node for distributing root inverse computation with intra-node-only communication..."
            )
            prev_rank = 0
            root_inv_dist_groups = {}
            for i, number_of_gpus_current_node in enumerate(number_of_gpus_per_node):
                split = list(range(prev_rank, prev_rank + number_of_gpus_current_node))
                dist_group = dist.new_group(ranks=split)
                root_inv_dist_groups.update({rank: dist_group for rank in split})
                prev_rank = prev_rank + number_of_gpus_current_node
                logger.info(
                    f"Ranks {split} from node {i} assigned to same process group for distributing root inverse computation!"
                )

        else:
            # We use -1 for a designated rank for non-distributed ranks.
            # None corresponds to the group, which will lead to the default (global) process group.
            root_inv_dist_groups = {-1: None}

        return root_inv_dist_groups

    @torch.no_grad()
    def _initialize_preconditioners_and_steps(self):
        """Initialize Shampoo preconditioners and inverse preconditioners."""

        for group in self.param_groups:
            for idx, p in enumerate(group[PARAMS]):
                state = self.state[p]
                dims = torch.as_tensor(p.shape)
                state[STEP] = torch.as_tensor(0)

                # Blocks the tensor and applies Shampoo to each block, with block
                # size equal to the max_preconditioner_dim; see feature above.
                if self._large_dim_method == LargeDimMethod.BLOCKING:
                    state[PRECONDITIONERS] = BlockShampooPreconditioner(
                        p,
                        beta2=group[BETAS][1],
                        epsilon=group[EPSILON],
                        exponent_override=self._exponent_override,
                        exponent_multiplier=self._exponent_multiplier,
                        use_bias_correction=self._use_bias_correction,
                        block_size=self._max_preconditioner_dim,
                        dtype=self._preconditioner_dtype,
                        dist_strategy=self._dist_strategy,
                        idx=idx,
                        use_merge_dims=self._use_merge_dims,
                        start_preconditioning_step=self._start_preconditioning_step,
                        grafting_type=self._grafting_type,
                        grafting_beta2=self._grafting_beta2,
                        grafting_epsilon=self._grafting_epsilon,
                    )

                # Uses Adagrad preconditioner if any dimension is larger than
                # the max_preconditioner_dim; see features above.
                elif self._large_dim_method == LargeDimMethod.ADAGRAD:
                    state[PRECONDITIONERS] = (
                        AdagradPreconditioner(
                            p,
                            beta2=group[BETAS][1],
                            epsilon=group[EPSILON],
                            use_bias_correction=self._use_bias_correction,
                            idx=idx,
                        )
                        if torch.any(dims > self._max_preconditioner_dim)
                        else ShampooPreconditioner(
                            p,
                            beta2=group[BETAS][1],
                            epsilon=group[EPSILON],
                            exponent_override=self._exponent_override,
                            exponent_multiplier=self._exponent_multiplier,
                            use_bias_correction=self._use_bias_correction,
                            diagonal_threshold=self._max_preconditioner_dim,
                            dtype=self._preconditioner_dtype,
                            dist_strategy=self._dist_strategy,
                            idx=idx,
                            start_preconditioning_step=self._start_preconditioning_step,
                            grafting_type=self._grafting_type,
                            grafting_beta2=self._grafting_beta2,
                            grafting_epsilon=self._grafting_epsilon,
                        )
                    )

                # Uses diagonal Shampoo preconditioner in place of full Shampoo
                # preconditioner if dimension is larger than max_preconditioner_dim; see feature
                # above.
                elif self._large_dim_method == LargeDimMethod.DIAGONAL:
                    state[PRECONDITIONERS] = ShampooPreconditioner(
                        p,
                        beta2=group[BETAS][1],
                        epsilon=group[EPSILON],
                        exponent_override=self._exponent_override,
                        exponent_multiplier=self._exponent_multiplier,
                        use_bias_correction=self._use_bias_correction,
                        diagonal_threshold=self._max_preconditioner_dim,
                        dtype=self._preconditioner_dtype,
                        idx=idx,
                        start_preconditioning_step=self._start_preconditioning_step,
                        grafting_type=self._grafting_type,
                        grafting_beta2=self._grafting_beta2,
                        grafting_epsilon=self._grafting_epsilon,
                    )

                else:
                    raise ValueError(
                        "Large dim method "
                        + str(self._large_dim_method)
                        + " is not implemented!"
                    )

                # Count parameters from preconditioners for logging purposes.
                self._parameter_count += state[PRECONDITIONERS].parameter_count

        # Logs total number of parameters for optimizer.
        logger.info(f"Total Parameter Count: {self._parameter_count}")

    @torch.no_grad()
    def _assign_preconditioners_to_ranks(self):
        """Assign each preconditioner to a rank depending on strategy.

        This method uses the following strategy:
            DistStrategy.NONE: All workers are independently responsible for all preconditioners.
            DistStrategy.CROSS_NODE: Preconditioners are distributed in a round-robin fashion across all nodes.
            DistStrategy.INTRA_NODE_ONLY: Preconditioners are distributed in a round-robin fashion within each node.

        """
        if not self._use_distributed():
            return
        elif self._dist_strategy in (
            DistStrategy.CROSS_NODE,
            DistStrategy.INTRA_NODE_ONLY,
        ):
            # Obtain number of GPUs on current node based on number of GPUs per node and the current rank.
            # Example: Suppose the number of GPUs per node is [8, 12, 8] (a heterogeneous GPU architecture).
            #   If the current rank is 8, then the node index is 1 and the number of GPUs on the current node is 12.
            #   If the current rank is 7, then the node index is 0 and the number of GPUs on the current node is 8.
            my_rank = dist.get_rank()
            assert (
                0 <= my_rank < sum(self._number_of_gpus_per_node)
            ), f"Rank is not within the range {self._world_size}"
            node_index = bisect.bisect_right(
                list(itertools.accumulate(self._number_of_gpus_per_node)), my_rank
            )
            number_of_gpus_current_node = self._number_of_gpus_per_node[node_index]

            preconditioner_count = 0
            for group in self.param_groups:
                for p in group[PARAMS]:
                    state = self.state[p]
                    if isinstance(
                        state[PRECONDITIONERS],
                        (ShampooPreconditioner, BlockShampooPreconditioner),
                    ):
                        preconditioner_count = state[
                            PRECONDITIONERS
                        ].assign_preconditioners_rank(
                            preconditioner_count, number_of_gpus_current_node
                        )
        else:
            raise NotImplementedError(
                "Root inverse strategy is not implemented! Specified root inverse strategy is "
                + str(self._dist_strategy)
                + "."
            )

    @torch.no_grad()
    def _compute_root_inverse(self):
        """Root inverse computation across all preconditioners/parameters."""
        for group in self.param_groups:
            for p in group[PARAMS]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if isinstance(
                    state[PRECONDITIONERS],
                    (ShampooPreconditioner, BlockShampooPreconditioner),
                ):
                    my_rank = dist.get_rank() if self._use_distributed() else -1
                    my_group = self._root_inv_dist_groups[my_rank]

                    state[PRECONDITIONERS].compute_root_inverse(
                        rank=my_rank, group=my_group
                    )

    @torch.no_grad()
    def _compute_and_log_root_inverse_residuals(
        self,
    ):
        """Compute root inverse residuals over all preconditioners."""

        # Compute expected relative errors/residuals for debugging purposes
        if self._preconditioner_dtype == torch.float64:
            expected_relative_error = 1e-7
        elif self._preconditioner_dtype == torch.float:
            expected_relative_error = 1e-3
        else:
            logger.warning(
                "Expected relative error/residual not supported for precision lower than float32."
            )

        # Accumulate relative errors/residuals
        relative_errors = []
        relative_residuals = []

        for group in self.param_groups:
            for p in group[PARAMS]:
                state = self.state[p]

                if isinstance(
                    state[PRECONDITIONERS],
                    (ShampooPreconditioner, BlockShampooPreconditioner),
                ):
                    relative_error, relative_residual = state[
                        PRECONDITIONERS
                    ].compute_root_inverse_residuals()

                    relative_errors += relative_error
                    relative_residuals += relative_residual

        relative_errors = torch.stack(relative_errors)
        relative_residuals = torch.stack(relative_residuals)

        quantiles = torch.as_tensor(
            [0, 0.25, 0.5, 0.75, 1],
            device=relative_errors.device,
            dtype=relative_errors.dtype,
        )
        logger.debug(f"Expect Relative Error <= {expected_relative_error}")
        logger.debug(
            f"Relative Error (||X - X_hat||_inf / ||X||_inf)       Average: {torch.mean(relative_errors)}, Quantiles [0, 25, 50, 75, 100]: {torch.quantile(relative_errors, quantiles, interpolation='nearest')}"
        )
        logger.debug(
            f"Relative Residual (||X_hat^-r - A||_inf / ||A||_inf) Average: {torch.mean(relative_residuals)}, Quantiles [0, 25, 50, 75, 100]: {torch.quantile(relative_residuals, quantiles, interpolation='nearest')}"
        )

    @torch.no_grad()
    def _broadcast_inv_preconditioners(self):
        """Broadcasts inverse preconditioners."""
        if not self._use_distributed():
            return

        for group in self.param_groups:
            for p in group[PARAMS]:
                state = self.state[p]
                if PRECONDITIONERS in state:
                    my_rank = dist.get_rank()
                    my_group = self._root_inv_dist_groups[my_rank]

                    state[PRECONDITIONERS].broadcast(group=my_group)

    @torch.no_grad()
    def _update_preconditioners(self):
        """Updates preconditioners.

        Note: If using L2-regularization/weight decay, it is computed within this function and
        therefore should not be recomputed elsewhere.

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

                else:
                    # Incorporate weight decay into the gradient
                    # if we are not using decoupled weight decay.
                    #
                    # Equivalent to adding an L2-regularization term:
                    #   F(w) + lambda * ||w||^2.
                    if not self._use_decoupled_weight_decay and weight_decay != 0:
                        grad.add_(p, alpha=weight_decay)

                    # Update each preconditioner using the gradient.
                    state[PRECONDITIONERS].update_preconditioners(grad)

    @torch.no_grad()
    def _iterate_step(self):
        for group in self.param_groups:
            for p in group[PARAMS]:
                self.state[p][STEP] += 1
        return self.state[p][STEP]

    @torch.no_grad()
    def reset_preconditioners(self):
        for group in self.param_groups:
            for p in group[PARAMS]:
                self.state[p][PRECONDITIONERS].reset_preconditioners()

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
        self._update_preconditioners()

        # Computes root inverse of all preconditioners every self._precondition_frequency
        # after the self._start_preconditioning_step iteration.
        if (
            iteration % self._precondition_frequency == 0
            and iteration >= self._start_preconditioning_step
        ):
            self._compute_root_inverse()
            self._broadcast_inv_preconditioners()

            if self._debug_mode:
                self._compute_and_log_root_inverse_residuals()

        # Loops over all parameter groups and parameters to perform update.
        for group in self.param_groups:
            beta1, _ = group[BETAS]
            momentum_param = group[MOMENTUM]
            weight_decay = group[WEIGHT_DECAY]
            lr = group[LR]

            for p in group[PARAMS]:
                if p.grad is None:
                    continue

                # Initialize gradient, states, and dim for each parameter.
                grad = p.grad
                state = self.state[p]

                # TODO: Sparse case still not supported.
                if p.grad.is_sparse:
                    raise Exception(
                        "Sparse parameters are not currently supported by Shampoo."
                    )

                # Dense case
                else:
                    # Incorporate first-moment or filtered gradient estimation.
                    if beta1 != 0:
                        # Compute bias corrections if necessary.
                        if self._use_bias_correction:
                            bias_correction1 = 1.0 - beta1**iteration
                        else:
                            bias_correction1 = torch.as_tensor(1.0)

                        # Compute exponential moving average of the gradient (with
                        # potential bias correction).
                        filtered_grad = state.setdefault(
                            EXP_AVG,
                            torch.zeros_like(grad, memory_format=torch.preserve_format),
                        )
                        filtered_grad.mul_(beta1).add_(grad, alpha=1 - beta1)
                        grad.copy_(filtered_grad / bias_correction1)

                    # Compute preconditioned gradient and update parameters.
                    #
                    # If we are not applying momentum, uses the precondition_and_update
                    # function for improved performance.
                    if momentum_param == 0.0:
                        # Adds decoupled weight decay term.
                        if self._use_decoupled_weight_decay and weight_decay != 0:
                            p.mul_(1 - lr * weight_decay)

                        state[PRECONDITIONERS].precondition_and_update(p, grad, lr)

                    # Otherwise, uses the precondition function in order to apply
                    # momentum to the entire update to the parameters.
                    else:
                        # Uses two separate momentum terms for the grafted and Shampoo methods.
                        if self._use_separate_momentum:

                            # Compute search direction / preconditioned gradient for both Shampoo and
                            # grafted methods separately.
                            grafted_direction = state[
                                PRECONDITIONERS
                            ].graft_precondition(grad)
                            shampoo_direction = state[
                                PRECONDITIONERS
                            ].shampoo_precondition(grad)

                            # Adds decoupled weight decay term to both directions.
                            if self._use_decoupled_weight_decay and weight_decay != 0:
                                grafted_direction.add_(p, alpha=weight_decay)
                                shampoo_direction.add_(p, alpha=weight_decay)

                            # Generates momentum term and applies momentum or stochastic
                            # primal iterate averaging.
                            grafted_momentum_direction = state.setdefault(
                                GRAFTING_MOMENTUM,
                                torch.zeros_like(
                                    grad, memory_format=torch.preserve_format
                                ),
                            )
                            shampoo_momentum_direction = state.setdefault(
                                MOMENTUM,
                                torch.zeros_like(
                                    grad, memory_format=torch.preserve_format
                                ),
                            )
                            grafted_momentum_direction.mul_(momentum_param).add_(
                                grafted_direction
                            )
                            shampoo_momentum_direction.mul_(momentum_param).add_(
                                shampoo_direction
                            )

                            # Select which direction to use (with possible Nesterov acceleration).
                            search_direction, momentum_direction = (
                                (grafted_direction, grafted_momentum_direction)
                                if iteration < self._start_preconditioning_step
                                else (shampoo_direction, shampoo_momentum_direction)
                            )

                        else:
                            # Compute search direction / preconditioned gradient.
                            search_direction = state[PRECONDITIONERS].precondition(grad)

                            # Adds decoupled weight decay term.
                            if self._use_decoupled_weight_decay and weight_decay != 0:
                                search_direction.add_(p, alpha=weight_decay)

                            # Generates momentum term and applies momentum or stochastic
                            # primal iterate averaging.
                            momentum_direction = state.setdefault(
                                MOMENTUM,
                                torch.zeros_like(
                                    grad, memory_format=torch.preserve_format
                                ),
                            )
                            momentum_direction.mul_(momentum_param).add_(
                                search_direction
                            )

                        # Incorporates Nesterov momentum.
                        if self._use_nesterov:
                            search_direction.add_(
                                momentum_direction, alpha=momentum_param
                            )
                        else:
                            search_direction = momentum_direction

                        # Performs update by taking step along search direction.
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
                "loaded state dict has a different number of parameter groups"
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
            # Modified load_state_dict in order to ensure that the preconditioner objects
            # are casted correctly. This enables us to use generic map_locations when
            # checkpointing.
            elif isinstance(value, Preconditioner):
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
