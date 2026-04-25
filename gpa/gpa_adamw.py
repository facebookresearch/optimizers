"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

# pyre-unsafe
from logging import getLogger
from typing import Callable, Optional, Union

import torch
import torch.optim
from gpa.gpa_types import (
    BETA1,
    BETA2,
    EPS,
    EVAL_INTERP_COEFF,
    EXP_AVG,
    EXP_AVG_SQ,
    ITERATE_AVERAGING_TYPE,
    IterateAveragingType,
    LR,
    LR_MAX,
    PARAMS,
    STEP,
    TRAIN_INTERP_COEFF,
    TRAIN_MODE,
    WEIGHT_DECAY,
    WEIGHT_LR_POWER,
    WEIGHT_POW_COEFF,
    WEIGHT_SUM,
    Z_BUFFER,
)
from torch.optim.optimizer import ParamsT

logger = getLogger()


class GPAAdamW(torch.optim.Optimizer):
    r"""AdamW with Generalized Primal Averaging (GPA-AdamW).

    Incorporates two generalizations of Nesterov (Schedule-Free and GPA) in its
    primal averaging formulation to AdamW.

    The updates in GPAAdamW are as follows:
        y^{(t)} = mu_y x^{(t)} + (1 - mu_y) z^{(t)},
        g^{(t)} = \nabla f(y^{(t)}),
        m^{(t)} = beta_1 m^{(t-1)} + (1 - beta_1) g^{(t)},
        v^{(t)} = beta_2 v^{(t-1)} + (1 - beta_2) (g^{(t)})^2,
        z^{(t+1)} = z^{(t)} - alpha^{(t)} \frac{m^{(t)}}{\sqrt{v^{(t)}} + epsilon},
        x^{(t+1)} = mu_x x^{(t)} + (1 - mu_x) z^{(t+1)},

    Here, the sequences (iterates) x, y and z have the following meanings:
        - z is the primary sequence where the update step is applied. Updates to
          z are performed by the underlying base optimizer (e.g. SGD, AdamW,
          Shampoo etc.).
        - y is the sequence where the gradient is computed.
        - x is the sequence where the test/val loss should be evaluated at.
        - alpha^{(t)} denotes the learning rate scaling term which includes both
          warmup and bias-correction.
        - mu_y is the train interpolation coefficient (weight for x in y-update).
          This corresponds to `train_interp_coeff` in the code.
        - mu_x is the eval interpolation coefficient (weight for x in x-update).
          This corresponds to `eval_interp_coeff` in the code.
        - beta_1 and beta_2 are the coefficients used for computing running
          average of the gradient term and the gradient squared term respectively.

    Some other notes about the GPA Optimizers:
        - Since GPAAdamW uses two different points for gradient calls and test/val
          calculations, it is necessary to switch the parameter buffer between
          the two during training and evaluation. This is done by calling
          optimizer.train() and optimizer.eval() at the same place where you
          would call model.train() and model.eval().
        - If using offline evaluation of the model, the optimizer should be in
          eval mode while storing checkpoints.
        - For Schedule-Free (iterate_averaging_type=SCHEDULE_FREE), no scheduler
          is typically needed. Warmup is recommended. To add warmup only, use
          the learning rate schedule `constant` and set the warmup as usual as
          an argument.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    References:
    - The Road Less Scheduled
        (https://arxiv.org/abs/2405.15682)
    - Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs
        (https://arxiv.org/abs/2512.17131)

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate parameter. (default: 1.0)
        eps (float): Term added to the denominator outside of the root operation
            to improve numerical stability. (default: 1e-8)
        beta1 (float): Coefficient used for computing running average of the
            gradient term. (default: 0.9)
        beta2 (float): Coefficient used for computing running average of the
            gradient squared term. (default: 0.999)
        weight_decay (float): Weight decay. Note that weight_decay can be either
            applied to the y or x sequence. In this implementation, it is applied
            to the y sequence. (default: 0)
        iterate_averaging_type (IterateAveragingType): Controls which averaging
            mode is used. GPA uses a fixed eval_interp_coeff (mu_x).
            SCHEDULE_FREE uses polynomial weighting (no fixed mu_x).
            (default: IterateAveragingType.GPA)
        train_interp_coeff (float): The mu_y coefficient in the GPA paper. This
            is the weight for x in the y computation: y = mu_y * x + (1 - mu_y) * z.
            Higher values mean y is closer to x, lower values mean y is closer
            to z. This value must be between 0 and 1 exclusive. (default: 0.7)
        eval_interp_coeff (float): The mu_x coefficient in the GPA paper. This
            is the weight for x in the x-update: x_{new} = mu_x * x + (1 - mu_x) * z.
            Higher values mean x retains more of its previous value, lower values
            mean x moves closer to z. Only used in GPA mode. (default: 0.9967)
        weight_pow_coeff (float): Use polynomial weighting in the average with
            power r. Only used in Schedule-Free mode. (default: 0)
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting.
            Only used in Schedule-Free mode. (default: 2.0)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1.0,
        eps: float = 1e-8,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0,
        iterate_averaging_type: IterateAveragingType = IterateAveragingType.GPA,
        train_interp_coeff: float = 0.7,
        eval_interp_coeff: float = 0.9967,
        weight_pow_coeff: float = 0.0,
        weight_lr_power: float = 2,
    ):
        # Hyper-parameter checks
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not eps >= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        if not weight_decay >= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not isinstance(iterate_averaging_type, IterateAveragingType):
            raise ValueError(
                "iterate_averaging_type must be an IterateAveragingType enum, "
                "got {}".format(type(iterate_averaging_type))
            )
        if not (train_interp_coeff > 0.0 and train_interp_coeff <= 1.0):
            raise ValueError(
                "train_interp_coeff must be in range (0, 1]: {}".format(
                    train_interp_coeff
                )
            )
        if not (eval_interp_coeff >= 0.0 and eval_interp_coeff <= 1.0):
            raise ValueError(
                "eval_interp_coeff must be in range [0, 1]: {}".format(
                    eval_interp_coeff
                )
            )
        if not weight_pow_coeff >= 0.0:
            raise ValueError(
                "Invalid weight_pow_coeff value: {}".format(weight_pow_coeff)
            )
        if not weight_lr_power >= 0.0:
            raise ValueError(
                "Invalid weight_lr_power value: {}".format(weight_lr_power)
            )

        logger.debug(
            f"GPAAdamW.__init__(), lr={lr}, train_interp_coeff={train_interp_coeff}, weight_decay={weight_decay}, "
            f"r={weight_pow_coeff}, weight_lr_power={weight_lr_power}, eval_interp_coeff={eval_interp_coeff}, "
            f"iterate_averaging_type={iterate_averaging_type}"
        )

        super().__init__(
            params,
            {
                LR: lr,
                EPS: eps,
                TRAIN_INTERP_COEFF: train_interp_coeff,
                BETA1: beta1,
                BETA2: beta2,
                WEIGHT_POW_COEFF: weight_pow_coeff,
                WEIGHT_LR_POWER: weight_lr_power,
                WEIGHT_DECAY: weight_decay,
                EVAL_INTERP_COEFF: eval_interp_coeff,
                ITERATE_AVERAGING_TYPE: iterate_averaging_type,
            },
        )

    def _ensure_shared_state(self, group) -> Optional[torch.nn.Parameter]:
        """Ensure shared optimizer state is initialized for the group.

        This method initializes TRAIN_MODE, LR_MAX, WEIGHT_SUM, and STEP in
        the first parameter's state if they don't already exist. These are
        shared across all parameters in the group.

        Args:
            group: The parameter group to initialize shared state for.

        Returns:
            The first parameter in the group, or None if the group is empty.
        """
        if not group[PARAMS]:
            return None
        first_param = group[PARAMS][0]
        if first_param not in self.state:
            self.state[first_param] = {}
        state = self.state[first_param]
        if TRAIN_MODE not in state:
            state[TRAIN_MODE] = torch.tensor(False, dtype=torch.bool)
        if LR_MAX not in state:
            state[LR_MAX] = torch.tensor(-1.0, dtype=torch.float32)
        if WEIGHT_SUM not in state:
            state[WEIGHT_SUM] = torch.tensor(0.0, dtype=torch.float32)
        if STEP not in state:
            state[STEP] = torch.tensor(0, dtype=torch.int64)
        return first_param

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        z_buffer_list,
    ):
        """Initialize per-parameter state and collect buffers for the optimization step.

        This method handles per-parameter buffer initialization (z_buffer, exp_avg,
        exp_avg_sq) and collects them into the provided lists for use in the
        step() loop. It only processes parameters that have gradients.

        Note: This method calls _ensure_shared_state() internally to initialize
        shared state (TRAIN_MODE, LR_MAX, WEIGHT_SUM, STEP). The separation of
        concerns is:
            - _ensure_shared_state(): Initializes shared optimizer state that is
              common to all parameters in a group (stored in first_param's state).
            - _init_group(): Initializes per-parameter buffers and collects them
              into lists for the optimization loop.

        Args:
            group: The parameter group to initialize.
            params_with_grad: List to append parameters with gradients.
            grads: List to append gradients.
            exp_avgs: List to append exponential moving averages of gradients.
            exp_avg_sqs: List to append exponential moving averages of squared
                gradients.
            z_buffer_list: List to append z-sequence buffers.

        Returns:
            bool: True if any parameter has a sparse gradient, False otherwise.
        """
        has_sparse_grad = False

        # Ensure shared state is initialized.
        self._ensure_shared_state(group)

        for p in group[PARAMS]:
            if p.grad is None:
                continue

            # Append parameters and gradients into list.
            params_with_grad.append(p)
            grads.append(p.grad)
            if p.grad.is_sparse:
                has_sparse_grad = True

            # Initialize optimizer states.
            state = self.state[p]

            # z buffer (weight) in GPA.
            if Z_BUFFER not in self.state[p]:
                state[Z_BUFFER] = torch.clone(p, memory_format=torch.preserve_format)
            z_buffer_list.append(state[Z_BUFFER])

            # Exponential moving average of gradient values.
            if EXP_AVG not in state:
                state[EXP_AVG] = torch.zeros_like(
                    p, dtype=torch.float32, memory_format=torch.preserve_format
                )
            exp_avgs.append(state[EXP_AVG])

            # Exponential moving average of squared gradient values.
            if EXP_AVG_SQ not in state:
                state[EXP_AVG_SQ] = torch.zeros_like(
                    p, dtype=torch.float32, memory_format=torch.preserve_format
                )
            exp_avg_sqs.append(state[EXP_AVG_SQ])

        return has_sparse_grad

    @torch.no_grad()
    def eval(self):
        """Switch optimizer to eval mode.

        Converts parameters from y-sequence (gradient computation) to x-sequence
        (model evaluation).
        """
        logger.info("Toggling GPAAdamW eval mode!")
        for group in self.param_groups:
            first_param = self._ensure_shared_state(group)
            if first_param is None:
                continue
            train_mode = self.state[first_param][TRAIN_MODE].item()
            train_interp_coeff = group[TRAIN_INTERP_COEFF]
            if train_mode:
                for p in group[PARAMS]:
                    if p in self.state:
                        state = self.state[p]
                        if Z_BUFFER in state:
                            # Set p to x
                            p.lerp_(
                                end=state[Z_BUFFER].to(p.device),
                                weight=1 - 1 / train_interp_coeff,
                            )
                self.state[first_param][TRAIN_MODE].fill_(False)

    @torch.no_grad()
    def train(self):
        """Switch optimizer to train mode.

        Converts parameters from x-sequence (model evaluation) to y-sequence
        (gradient computation).
        """
        logger.info("Toggling GPAAdamW train mode!")
        for group in self.param_groups:
            first_param = self._ensure_shared_state(group)
            if first_param is None:
                continue
            train_mode = self.state[first_param][TRAIN_MODE].item()
            train_interp_coeff = group[TRAIN_INTERP_COEFF]
            if not train_mode:
                for p in group[PARAMS]:
                    if p in self.state:
                        state = self.state[p]
                        if Z_BUFFER in state:
                            # Set p to y
                            p.lerp_(
                                end=state[Z_BUFFER].to(p.device),
                                weight=1 - train_interp_coeff,
                            )
                self.state[first_param][TRAIN_MODE].fill_(True)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Check train mode before proceeding
        for group in self.param_groups:
            first_param = self._ensure_shared_state(group)
            if first_param is not None:
                if not self.state[first_param][TRAIN_MODE]:
                    raise RuntimeError(
                        "Optimizer was not in train mode when step is called. "
                        "Please insert .train() and .eval() calls on the "
                        "optimizer. See documentation for details."
                    )
                break

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            z_buffer_list = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                z_buffer_list,
            )

            # Get first_param for accessing shared state.
            first_param = group[PARAMS][0] if group[PARAMS] else None

            # Increment step counter and use it as group step.
            self.state[first_param][STEP] += 1
            k = self.state[first_param][STEP].item()

            # Get all group variables.
            eps = group[EPS]
            train_interp_coeff = group[TRAIN_INTERP_COEFF]
            beta1 = group[BETA1]
            beta2 = group[BETA2]
            weight_decay = group[WEIGHT_DECAY]
            weight_lr_power = group[WEIGHT_LR_POWER]
            lr = group[LR]
            weight_pow_coeff = group[WEIGHT_POW_COEFF]
            eval_interp_coeff = group[EVAL_INTERP_COEFF]
            iterate_averaging_type = group[ITERATE_AVERAGING_TYPE]

            # Compute bias correction.
            bias_correction1 = 1 - beta1**k
            bias_correction2 = 1 - beta2**k

            # Update LR_MAX in first parameter's state.
            lr_max = max(lr, self.state[first_param][LR_MAX].item())
            self.state[first_param][LR_MAX].fill_(lr_max)

            assert (
                lr_max > 0
            ), f"lr_max must be positive, got lr_max={lr_max}. Check that lr={lr} is positive."

            # Compute avg_coeff ONCE per step (before the parameter loop).
            # This is important for Schedule-Free: the coefficient should be the same
            # for all parameters within a single optimization step.
            avg_coeff = self.compute_avg_coeff(
                iterate_averaging_type=iterate_averaging_type,
                eval_interp_coeff=eval_interp_coeff,
                k=k,
                weight_pow_coeff=weight_pow_coeff,
                lr_max=lr_max,
                weight_lr_power=weight_lr_power,
                weight_sum_ref=self.state[first_param][WEIGHT_SUM],
            )

            for y, grad, exp_avg, exp_avg_sq, z in zip(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                z_buffer_list,
                strict=True,
            ):
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                # Reuse grad buffer for memory efficiency
                # grad_normalized = grad.div_(denom)
                # grad_normalized = exp_avg.div_(denom)
                grad_normalized = exp_avg.div(bias_correction1).div_(denom)

                # Weight decay calculated at y
                if weight_decay != 0:
                    grad_normalized.add_(y, alpha=weight_decay)

                # Memory-efficient y-update without explicitly computing x:
                # The standard updates are:
                #   z_new = z - lr * grad_normalized
                #   x_new = mu_x * x + (1 - mu_x) * z_new
                #   y_new = mu_y * x_new + (1 - mu_y) * z_new
                #
                # We can show that (where avg_coeff = 1 - mu_x for GPA or
                # the polynomial weight for Schedule-Free):
                #   y_new = (1 - avg_coeff) * y + avg_coeff * z
                #           + lr * (mu_y * (1 - avg_coeff) - 1) * grad_normalized
                #
                # This allows us to update y in-place without computing x.
                y.lerp_(end=z, weight=avg_coeff)
                y.add_(
                    grad_normalized,
                    alpha=lr * (train_interp_coeff * (1 - avg_coeff) - 1),
                )

                z.sub_(grad_normalized, alpha=lr)

        return loss

    @staticmethod
    def compute_avg_coeff(
        iterate_averaging_type: IterateAveragingType,
        eval_interp_coeff: float,
        k: int,
        weight_pow_coeff: float,
        lr_max: float,
        weight_lr_power: float,
        weight_sum_ref: torch.Tensor,
    ) -> float:
        """Compute the averaging coefficient (z-weight) for primal averaging.

        For Schedule-Free mode (SCHEDULE_FREE), computes the weight based on the
        polynomial formula from the paper and updates weight_sum_ref in-place.
        For GPA mode (GPA), computes (1 - mu_x) since eval_interp_coeff
        represents mu_x (weight for x) and we need the z-weight.

        Args:
            iterate_averaging_type: The type of iterate averaging to use.
            eval_interp_coeff: The mu_x coefficient (weight for x in x-update).
                Only used in GPA mode.
            k: The current step (1-indexed).
            weight_pow_coeff: Power coefficient for polynomial weighting.
                Only used in Schedule-Free mode.
            lr_max: Maximum learning rate seen so far.
                Only used in Schedule-Free mode.
            weight_lr_power: Power to raise lr_max to for weighting.
                Only used in Schedule-Free mode.
            weight_sum_ref: Reference to the cumulative weight sum tensor
                (updated in-place). Only used in Schedule-Free mode.

        Returns:
            The z-weight (1 - mu_x) to use in the averaging step.
        """
        if iterate_averaging_type == IterateAveragingType.SCHEDULE_FREE:
            weight = (k**weight_pow_coeff) * (lr_max**weight_lr_power)
            weight_sum_ref += weight
            return weight / weight_sum_ref
        else:  # GPA
            return 1 - eval_interp_coeff
