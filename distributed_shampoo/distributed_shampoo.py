"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
import math
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import asdict
from typing import Any, overload

import torch
from distributed_shampoo.distributor.shampoo_ddp_distributor import DDPDistributor
from distributed_shampoo.distributor.shampoo_distributor import (
    Distributor,
    DistributorInterface,
)
from distributed_shampoo.distributor.shampoo_fsdp_distributor import FSDPDistributor
from distributed_shampoo.distributor.shampoo_fully_shard_distributor import (
    FullyShardDistributor,
)
from distributed_shampoo.distributor.shampoo_fully_shard_lossless_distributor import (
    FullyShardLosslessDistributor,
)
from distributed_shampoo.distributor.shampoo_hsdp_distributor import HSDPDistributor
from distributed_shampoo.distributor.shampoo_hybrid_shard_distributor import (
    HybridShardDistributor,
)
from distributed_shampoo.preconditioner.adagrad_preconditioner_list import (
    AdagradPreconditionerList,
)
from distributed_shampoo.preconditioner.matrix_functions_types import (
    EigendecompositionConfig,
    PseudoInverseConfig,
)
from distributed_shampoo.preconditioner.preconditioner_list import PreconditionerList
from distributed_shampoo.preconditioner.sgd_preconditioner_list import (
    SGDPreconditionerList,
)
from distributed_shampoo.preconditioner.shampoo_preconditioner_list import (
    EigendecomposedShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    RootInvShampooPreconditionerList,
)
from distributed_shampoo.preconditioner.sign_descent_preconditioner_list import (
    SignDescentPreconditionerList,
)
from distributed_shampoo.preconditioner.spectral_descent_preconditioner_list import (
    SpectralDescentPreconditionerList,
)

from distributed_shampoo.shampoo_types import (
    AdaGradPreconditionerConfig,
    AdamPreconditionerConfig,
    AmortizedPreconditionerConfig,
    BETA3,
    BETAS,
    DAMPENING,
    DDPDistributedConfig,
    DefaultShampooConfig,
    DefaultSingleDeviceDistributedConfig,
    DISTRIBUTED_CONFIG,
    DistributedConfig,
    DISTRIBUTOR,
    EigendecomposedShampooPreconditionerConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    EPSILON,
    FILTERED_GRAD,
    FILTERED_GRAD_LIST,
    FSDPDistributedConfig,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    GRAFTING_CONFIG,
    GRAFTING_PRECONDITIONER_LIST,
    HSDPDistributedConfig,
    HybridShardDistributedConfig,
    LR,
    MASKED_BLOCKED_GRADS,
    MASKED_BLOCKED_PARAMS,
    MASKED_FILTERED_GRAD_LIST,
    MASKED_MOMENTUM_LIST,
    MAX_PRECONDITIONER_DIM,
    MOMENTUM,
    MOMENTUM_LIST,
    PARAMS,
    PRECONDITION_FREQUENCY,
    PRECONDITIONER_CONFIG,
    PreconditionerConfig,
    PREVIOUS_GRAD_SELECTOR,
    RMSpropPreconditionerConfig,
    RootInvShampooPreconditionerConfig,
    SGDPreconditionerConfig,
    SHAMPOO_PRECONDITIONER_LIST,
    ShampooPT2CompileConfig,
    SignDescentPreconditionerConfig,
    SingleDeviceDistributedConfig,
    SpectralDescentPreconditionerConfig,
    START_PRECONDITIONING_STEP,
    STEP,
    USE_BIAS_CORRECTION,
    USE_DECOUPLED_WEIGHT_DECAY,
    USE_NESTEROV,
    USE_PIN_MEMORY,
    WEIGHT_DECAY,
)

from distributed_shampoo.utils.shampoo_checkpoint_utils import (
    extract_state_dict_content,
    flatten,
    unflatten,
    update_param_state_dict_object,
)
from distributed_shampoo.utils.shampoo_utils import compress_list

from torch.optim.optimizer import ParamsT, StateDict

logger: logging.Logger = logging.getLogger(__name__)


class DistributedShampoo(torch.optim.Optimizer):
    """Implements distributed Shampoo algorithm.

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
            - GraftingType.RMSPROP: Grafts the RMSprop method.
            - GraftingType.ADAM: Grafts the Adam method.

        NOTE: These methods do not graft the first-moment component - it is entirely based upon grafting using the
        diagonal preconditioner. If using an exponential moving average of the gradient (or gradient filtering), we
        can set beta1 as the same value from before, and both Shampoo and the grafted method will use the filtered
        gradient.

    2. Blocking for Large-Dimensional Tensors: In order to scale Shampoo to large-dimensional tensors, we block the tensor
        and apply Shampoo to each block. For simplicity, suppose we have a linear layer/matrix parameter, W is a m x n matrix:

                [[w_11 w_12 ... w_1n]
                 [w_21 w_22 ... w_2n]
            W =           :
                 [w_m1 w_m2 ... w_mn]]

        Given a max_preconditioner_dim b > 0, blocks W and applies Shampoo to each block, i.e., if b divides both m, n, then:

                [[W_11 W_12 ... W_1k]
                 [W_21 W_22 ... W_2k]
            W =           :
                 [W_l1 W_l2 ... W_lk]]

        where l = m / b, k = n / b, and apply Shampoo to W_ij which is a b x b matrix. This can be viewed as further blocking
        each block of the Shampoo block-diagonal preconditioner.

        Computational cost = O(b^3)
        Memory cost = 4mn (including root inverse preconditioners)

    3. Distributed Memory and Computation: We support different distributed training setups through the distributed_config option,
        which specifies a configuration specific to that setting.

        - None: Performs serial single-GPU training. Replicates all computation and optimizer states across all
            devices.

        - DDPDistributedConfig: Supports multi-GPU distributed data-parallel training via torch.distributed. Assigns optimizer states
            and computation for each block in a greedy fashion to different workers. Leverages DTensor in order to distribute the
            per-block optimizer states from Shampoo. An AllGather communication is performed in order to synchronize the parameter
            updates to applied to all parameter blocks.

            Distributed Training Specific Fields:
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - Only supports homogeneous hardware architectures.

        - FSDPDistributedConfig: Supports multi-GPU fully-sharded data-parallel training via torch.distributed. This option uses
            additional metadata in order to reconstruct valid tensor blocks of the original parameter from the flattened parameter
            representation.

            Distributed Training Specific Fields:
                - param_to_metadata: One must create a dictionary containing the metadata for each parameter in the FSDP model. This
                    includes the shape of the original parameter as well as the start and end indices of the tensor shard with
                    respect to the unsharded flattened parameter.

            Requirements:
                - torch.distributed must be initialized in advance.
                - One must enable the option use_orig_params = True in FSDP.

        - HSDPDistributedConfig: Supports hierarchical parallelism approach that combines DDP and FSDP to scale up training on large models.
            It works by dividing the model into smaller sub-models, each of which is trained in parallel using data parallelism.
            The gradients from each sub-model are then aggregated and used to update the full model.

            Distributed Training Specific Fields:
                - device_mesh: A 2D device mesh that specifies the layout of the model parallelism and data parallelism.
                - param_to_metadata: One must create a dictionary containing the metadata for each parameter in the FSDP model. This
                    includes the shape of the original parameter as well as the start and end indices of the tensor shard with
                    respect to the unsharded flattened parameter.
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - One must enable the option use_orig_params = True in HSDP.
                - Only works with the option sharding_strategy=ShardingStrategy.HYBRID_SHARD.
                - Within data parallelism process groups, only supports homogeneous hardware architectures.

        - FullyShardDistributedConfig: Supports per-parameter FSDP training, a.k.a. FSDP2, or "fully_shard" api in torch.distributed. Please see
            README for more detailed introduction on Shampoo FSDP2.

            Requirements:
                - torch.distributed must be initialized in advance.

        - HybridShardDistributedConfig: Supports hierarchical parallelism approach that combines DDP and FSDP to scale up training on large models
            for FSDP2. Please see README for more detailed introduction.

            Distributed Training Specific Fields:
                - device_mesh: A 2D device mesh that specifies the layout of the model parallelism and data parallelism.
                - communication_dtype: We can specify the communication dtype used for the AllGather communication in order to
                    reduce communication overhead per-iteration.
                - num_trainers_per_group: Specifies the number of GPUs used per distributed group. This enables us to only
                    distribute computation across a subset of GPUs, and replicate the same computation across different distributed
                    groups. This is useful for performance by trading off communication costs vs. computational costs.
                - communicate_params: We offer the option to communicate the parameter updates or the updated parameters. Enabling
                    this option specifically communicates the updated parameters. Note that using a lower-precision
                    communication_dtype is more amenable to the case where this option is disabled (i.e., we are communicating the
                    parameter updates).

            Requirements:
                - torch.distributed must be initialized in advance.
                - Within data parallelism process groups, only supports homogeneous hardware architectures.

    4. PyTorch 2.0 Compile Support: Shampoo supports PyTorch 2.0's compilation feature to speed up model training. This is enabled by
        setting up the shampoo_pt2_compile_config arg for Shampoo PyTorch 2.0 compilation.

        - If shampoo_pt2_compile_config = None, ignores compilation, and Shampoo will run in eager mode.
            Shampoo PT2 eager mode means the optimizer runs on plain python code, there is no graphs and lower level kernels used
            to speed up the optimizer stage; and typically you would expect lower QPS for model training as a result.
            For more details regarding PT2 compilation: https://pytorch.org/get-started/pytorch-2.0/

        - If shampoo_pt2_compile_config is set to ShampooPT2CompileConfig class, Shampoo will run in PT2 mode. Shampoo PT2 mode typically gives
            on par numerics and model quality, plus higher QPS. But due to differences in lower level kernel implementation, model quality on par
            is not always guaranteed. If you see model quality gap, please switch back to Shampoo PT2 eager mode by setting
            shampoo_pt2_compile_config = None.

        Shampoo PT2 compilation can also be customized for the backend and options via ShampooPT2CompileConfig.

    5. [EXPERIMENTAL] Eigenvalue correction (SOAP): We can (approximately) correct the eigenvalues of Shampoo's preconditioner by accumulating a running
        average of the squared gradient in the eigenbasis of Shampoo's preconditioner. This running average (with hyperparameter `betas[1]`) is
        updated every iteration while the eigenbasis of Shampoo's preconditioner is only computed every `precondition_frequency` steps.
        Alternatively, this can be seen as running Adam in the eigenbasis of Shampoo's preconditioner, also known as SOAP.

        When setting `preconditioner_config` as an instance of `EigenvalueCorrectedShampooPreconditionerConfig`, there is typically no need to use learning
        rate grafting from Adam (`grafting_config=None`) and, when they are available, Adam's optimal `lr`, `betas`, and `weight_decay` should be
        a good starting point for further tuning. However, the case of `beta2=1.0`, i.e. an AdaGrad-like accumulation, has not been explored yet.
        Also, in settings where Shampoo would usually graft its learning rate from SGD, grafting might still be beneficial.

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate. (Default: 1e-2)
        betas (tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            (Default: (0.9, 1.0))
        beta3 (float): Coefficient used for computing running average of gradient only for the current iteration.
            This can be used to replicate a version of NAdam if set appropriately. For example, if beta1 = 0.9, then applying
            beta1 interpolation a second time is equivalent to setting beta3 = 0.9 * 0.9 = 0.81.
            If set to -1.0, will set equal to beta1. (Default: -1.0)
        epsilon (float): Term added to the denominator to improve numerical stability, also known as the damping term. (Default: 1e-12)
        momentum (float): Momentum parameter. (Default: 0.)
        dampening (float): Dampening parameter for momentum. (Default: 0.)
        weight_decay (float): Weight decay (L2 penalty). (Default: 0.)
        max_preconditioner_dim (int | float): Maximum preconditioner dimension. (Default: 1024)
        precondition_frequency (int): Frequency of updating all components of the preconditioner.
            If this field is an instance ShampooPreconditionerConfig, this is the update frequency of the root inverse of the preconditioner.
            If this field is an instance EigenvalueCorrectedShampooPreconditionerConfig, this is the update frequency of the eigenbasis of preconditioner.
            (Default: 1)
        start_preconditioning_step (int): Iteration to start computing inverse preconditioner. If -1, uses
            the same value as precondition_frequency. (Default: -1)
        use_nesterov (bool): Flag for using Nesterov momentum. (Default: False)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        use_decoupled_weight_decay (bool): Flag for using AdamW-style decoupled weight decay. (Default: True)
        grafting_config (PreconditionerConfig | None): Configuration for grafting method. If None, ignores grafting.
            (Default: None)
        use_pin_memory (bool): Whether to use pin memory to remove sync point in memory copy. (Default: False)
        shampoo_pt2_compile_config (ShampooPT2CompileConfig | None): Configuration for Shampoo PT2 compilation. If None,
            ignores compilation, and Shampoo will run in eager mode. (Default: None)
        distributed_config (DistributedConfig): Configuration for applying Shampoo to different distributed training frameworks, such as distributed-data parallel (DDP) training.
            (Default: DefaultSingleDeviceDistributedConfig)
        preconditioner_config (PreconditionerConfig): Configuration for preconditioner computation.
            If this field is an instance ShampooPreconditionerConfig, Shampoo uses the root inverse of the preconditioner.
            If this field is an instance EigenvalueCorrectedShampooPreconditionerConfig Shampoo uses corrected the eigenvalues/running Adam in the eigenbasis of preconditioner.
            (Default: DefaultShampooConfig)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 1.0),
        beta3: float = -1.0,
        epsilon: float = 1e-12,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int | float = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        grafting_config: PreconditionerConfig | None = None,
        use_pin_memory: bool = False,
        shampoo_pt2_compile_config: ShampooPT2CompileConfig | None = None,
        distributed_config: DistributedConfig = DefaultSingleDeviceDistributedConfig,
        preconditioner_config: PreconditionerConfig = DefaultShampooConfig,
    ) -> None:
        super().__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                BETA3: beta3,
                EPSILON: epsilon,
                MOMENTUM: momentum,
                DAMPENING: dampening,
                WEIGHT_DECAY: weight_decay,
                MAX_PRECONDITIONER_DIM: max_preconditioner_dim,
                PRECONDITION_FREQUENCY: precondition_frequency,
                START_PRECONDITIONING_STEP: start_preconditioning_step,
                USE_NESTEROV: use_nesterov,
                USE_BIAS_CORRECTION: use_bias_correction,
                USE_DECOUPLED_WEIGHT_DECAY: use_decoupled_weight_decay,
                GRAFTING_CONFIG: grafting_config,
                USE_PIN_MEMORY: use_pin_memory,
                DISTRIBUTED_CONFIG: distributed_config,
                PRECONDITIONER_CONFIG: preconditioner_config,
            },
        )

        def param_group_hyperparameter_check(param_group: dict[str, Any]) -> None:
            if not param_group[LR] >= 0.0:
                raise ValueError(f"Invalid {param_group[LR]=}. Must be >= 0.0.")
            if not 0.0 <= param_group[BETAS][0] < 1.0:
                raise ValueError(
                    f"Invalid {param_group[BETAS][0]=}. Must be in [0.0, 1.0)."
                )
            if not 0.0 <= param_group[BETAS][1] <= 1.0:
                raise ValueError(
                    f"Invalid {param_group[BETAS][1]=}. Must be in [0.0, 1.0]."
                )
            if param_group[BETA3] == -1.0:
                param_group[BETA3] = param_group[BETAS][0]
            elif not 0.0 <= param_group[BETA3] < 1.0:
                raise ValueError(
                    f"Invalid {param_group[BETA3]=}. Must be in [0.0, 1.0)."
                )
            if (
                isinstance(
                    param_group[PRECONDITIONER_CONFIG],
                    AmortizedPreconditionerConfig,
                )
                and isinstance(
                    param_group[PRECONDITIONER_CONFIG].amortized_computation_config,
                    EigendecompositionConfig,
                )
                and isinstance(
                    param_group[
                        PRECONDITIONER_CONFIG
                    ].amortized_computation_config.rank_deficient_stability_config,
                    PseudoInverseConfig,
                )
            ):
                if param_group[EPSILON] != 0.0:
                    raise ValueError(
                        f"Invalid {param_group[EPSILON]=}. Must be == 0.0 when PseudoInverseConfig is used."
                    )
            elif not param_group[EPSILON] > 0.0:
                raise ValueError(f"Invalid {param_group[EPSILON]=}. Must be > 0.0.")
            if not 0.0 <= param_group[MOMENTUM] < 1.0:
                raise ValueError(
                    f"Invalid {param_group[MOMENTUM]=}. Must be [0.0, 1.0)."
                )
            if not 0.0 <= param_group[DAMPENING] < 1.0:
                raise ValueError(
                    f"Invalid {param_group[DAMPENING]=}. Must be [0.0, 1.0)."
                )
            if not param_group[WEIGHT_DECAY] >= 0.0:
                raise ValueError(
                    f"Invalid {param_group[WEIGHT_DECAY]=}. Must be >= 0.0."
                )
            if (
                isinstance(param_group[MAX_PRECONDITIONER_DIM], float)
                and param_group[MAX_PRECONDITIONER_DIM] != math.inf
            ):
                raise ValueError(
                    f"Invalid {param_group[MAX_PRECONDITIONER_DIM]=}. Must be an integer or math.inf."
                )
            if not param_group[MAX_PRECONDITIONER_DIM] >= 1:
                raise ValueError(
                    f"Invalid {param_group[MAX_PRECONDITIONER_DIM]=}. Must be >= 1."
                )
            if not param_group[PRECONDITION_FREQUENCY] >= 1:
                raise ValueError(
                    f"Invalid {param_group[PRECONDITION_FREQUENCY]=}. Must be >= 1."
                )
            if not param_group[START_PRECONDITIONING_STEP] >= -1:
                raise ValueError(
                    f"Invalid {param_group[START_PRECONDITIONING_STEP]=}. Must be >= -1."
                )

            if isinstance(
                param_group[PRECONDITIONER_CONFIG],
                (SignDescentPreconditionerConfig, SpectralDescentPreconditionerConfig),
            ):
                preconditioner_config_name = param_group[
                    PRECONDITIONER_CONFIG
                ].__class__.__name__
                # Warn about hyperparameters that won't have any effect.
                logger.warning(
                    f"{param_group[BETAS][1]=} does not have any effect when {preconditioner_config_name} is used."
                )
                logger.warning(
                    f"{param_group[EPSILON]=} does not have any effect when {preconditioner_config_name} is used."
                )
                logger.warning(
                    f"{param_group[PRECONDITION_FREQUENCY]=} does not have any effect when {preconditioner_config_name} is used. Setting precondition_frequency to 1..."
                )
                param_group[PRECONDITION_FREQUENCY] = 1

            if (
                isinstance(
                    param_group[PRECONDITIONER_CONFIG],
                    SpectralDescentPreconditionerConfig,
                )
                and param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality != 2
            ):
                logger.warning(
                    f"{param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality=} is not equal to 2. Setting target_parameter_dimensionality to 2..."
                )
                param_group[DISTRIBUTED_CONFIG].target_parameter_dimensionality = 2

            # Provide warning/error for start_preconditioning_step.
            if param_group[START_PRECONDITIONING_STEP] == -1:
                param_group[START_PRECONDITIONING_STEP] = param_group[
                    PRECONDITION_FREQUENCY
                ]
                logger.warning(
                    f"start_preconditioning_step set to -1. Setting start_preconditioning_step equal to {param_group[PRECONDITION_FREQUENCY]=} by default."
                )
            if (
                param_group[START_PRECONDITIONING_STEP]
                < param_group[PRECONDITION_FREQUENCY]
            ):
                raise ValueError(
                    f"Invalid {param_group[START_PRECONDITIONING_STEP]=}. Must be >= {param_group[PRECONDITION_FREQUENCY]=}."
                )

            # Warn when Nesterov is used but momentum is 0.
            if param_group[USE_NESTEROV] and param_group[MOMENTUM] == 0.0:
                logger.warning(
                    "Nesterov flag is enabled but momentum parameter is zero! Continuing without using momentum or Nesterov acceleration..."
                )

        # Perform per param_group hyperparameter checks.
        for i, param_group in enumerate(self.param_groups):
            logger.info(f"Checking param_group {i} hyperparameters...")
            param_group_hyperparameter_check(param_group=param_group)

        # Initialize non-group-related fields.
        self._shampoo_pt2_compile_config: ShampooPT2CompileConfig | None = (
            shampoo_pt2_compile_config
        )

        # Initialize list containing group state dictionaries.
        self._per_group_state_lists: list[dict[str, Any]] = [
            {} for _ in self.param_groups
        ]

        # Block parameters and instantiate optimizer states.
        # NOTE: _instantiate_distributor() has to be called first and _initialize_blocked_parameters_state() second.
        self._instantiate_distributor()
        self._initialize_blocked_parameters_state()
        self._instantiate_shampoo_preconditioner_list()
        self._instantiate_grafting()
        self._instantiate_steps()
        self._instantiate_momentum()
        self._instantiate_filtered_grads()
        self._instantiate_per_group_step(
            shampoo_pt2_compile_config=shampoo_pt2_compile_config
        )

    @torch.no_grad()
    def _instantiate_distributor(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            match group[DISTRIBUTED_CONFIG]:
                case SingleDeviceDistributedConfig():
                    distributor_cls: type[DistributorInterface] = Distributor
                case HSDPDistributedConfig():
                    distributor_cls = HSDPDistributor
                case HybridShardDistributedConfig():
                    distributor_cls = HybridShardDistributor
                case DDPDistributedConfig():
                    distributor_cls = DDPDistributor
                case FSDPDistributedConfig():
                    distributor_cls = FSDPDistributor
                case FullyShardDistributedConfig(
                    param_assignment_strategy=FSDPParamAssignmentStrategy.DEFAULT
                ):
                    distributor_cls = FullyShardDistributor
                case FullyShardDistributedConfig(
                    param_assignment_strategy=FSDPParamAssignmentStrategy.REPLICATE
                ):
                    distributor_cls = FullyShardLosslessDistributor
                case _:
                    raise NotImplementedError(
                        f"{group[DISTRIBUTED_CONFIG]=} not supported!"
                    )

            # Instantiate distributors for each group.
            state_lists[DISTRIBUTOR] = distributor_cls(group)

            # If the number of trainers is more than the number of blocks,
            # some workers might not get any parameters which cause wasting resources because
            # those trainers are working on nothing.
            assert state_lists[
                DISTRIBUTOR
            ].local_blocked_params, f"Some workers have no parameters to work on. This mostly happens when the value of num_trainers_per_group field in {group[DISTRIBUTED_CONFIG]=} is more than the number of local blocked params on a single device. Please check the num_trainers_per_group setting and consider reducing it."

            # Compile blocked parameters and block-to-parameter metadata into group lists.
            state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
                DISTRIBUTOR
            ].local_blocked_params
            # First PREVIOUS_GRAD_SELECTOR is set to None.
            state_lists[PREVIOUS_GRAD_SELECTOR] = None

    @torch.no_grad()
    def _initialize_blocked_parameters_state(self) -> None:
        for state_lists in self._per_group_state_lists:
            # NOTE: We need to initialize the optimizer states within the optimizer's state dictionary.
            for block_info in state_lists[DISTRIBUTOR].local_block_info_list:
                param_state = self.state[block_info.param]
                assert (
                    (block_index := block_info.composable_block_ids[1])
                    not in param_state
                ), "There should not exist any optimizer state yet. Maybe verify that _instantiate_distributor was called before all other instantiation functions."
                param_state[block_index] = {}

    @torch.no_grad()
    def _preconditioner_config_to_list_cls(
        self,
        state_lists: dict[str, Any],
        group: dict[str, Any],
        preconditioner_config: PreconditionerConfig,
    ) -> PreconditionerList | None:
        match preconditioner_config:
            case None:
                return None
            case SGDPreconditionerConfig():
                return SGDPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                )
            case (
                AdaGradPreconditionerConfig()
                | RMSpropPreconditionerConfig()
                | AdamPreconditionerConfig()
            ):
                return AdagradPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].local_block_info_list,
                    beta2=(
                        1.0
                        if type(group[GRAFTING_CONFIG]) is AdaGradPreconditionerConfig
                        else group[GRAFTING_CONFIG].beta2
                    ),
                    epsilon=group[GRAFTING_CONFIG].epsilon,
                    use_bias_correction=type(group[GRAFTING_CONFIG])
                    is AdamPreconditionerConfig,
                )
            case (
                RootInvShampooPreconditionerConfig()
                | EigendecomposedShampooPreconditionerConfig()
                | EigenvalueCorrectedShampooPreconditionerConfig()
            ):
                preconditioner_config_to_list_cls: dict[
                    type[PreconditionerConfig], Callable[..., PreconditionerList]
                ] = {
                    RootInvShampooPreconditionerConfig: RootInvShampooPreconditionerList,
                    EigendecomposedShampooPreconditionerConfig: EigendecomposedShampooPreconditionerList,
                    EigenvalueCorrectedShampooPreconditionerConfig: EigenvalueCorrectedShampooPreconditionerList,
                }
                return preconditioner_config_to_list_cls[type(preconditioner_config)](
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    preconditioner_config=group[PRECONDITIONER_CONFIG],
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].local_block_info_list,
                    beta2=group[BETAS][1],
                    epsilon=group[EPSILON],
                    use_bias_correction=group[USE_BIAS_CORRECTION],
                )
            case SignDescentPreconditionerConfig():
                return SignDescentPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    preconditioner_config=group[PRECONDITIONER_CONFIG],
                )
            case SpectralDescentPreconditionerConfig():
                assert (
                    group[DISTRIBUTED_CONFIG].target_parameter_dimensionality == 2
                ), f"{group[DISTRIBUTED_CONFIG].target_parameter_dimensionality=} must be 2 when using SpectralDescentPreconditionerConfig."
                return SpectralDescentPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].local_blocked_params,
                    preconditioner_config=group[PRECONDITIONER_CONFIG],
                )
            case _:
                raise NotImplementedError(f"{preconditioner_config=} not supported!")

    @torch.no_grad()
    def _instantiate_shampoo_preconditioner_list(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            assert (
                group[PRECONDITIONER_CONFIG] is not None
            ), f"{group[PRECONDITIONER_CONFIG]=} is None. Please check the instantiation of DistributedShampoo."
            state_lists[SHAMPOO_PRECONDITIONER_LIST] = (
                self._preconditioner_config_to_list_cls(
                    state_lists=state_lists,
                    group=group,
                    preconditioner_config=group[PRECONDITIONER_CONFIG],
                )
            )

    @torch.no_grad()
    def _instantiate_grafting(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            state_lists[GRAFTING_PRECONDITIONER_LIST] = (
                self._preconditioner_config_to_list_cls(
                    state_lists=state_lists,
                    group=group,
                    preconditioner_config=group[GRAFTING_CONFIG],
                )
            )

    @torch.no_grad()
    def _instantiate_steps(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            assert (
                len(state_lists[DISTRIBUTOR].local_block_info_list) > 0
            ), "There is no params in your param_group. Please check the instantiation of DistributedShampoo "
            'with param_group containing no params. For example, DistributedShampoo(params=[{"params": []}])'
            # NOTE: We instantiate a single step tensor on CPU for each group in order
            #       to track the number of steps taken by all parameters within the group.
            #       Instantiating on CPU avoids GPU synchronization.
            state_lists[STEP] = torch.tensor(0, dtype=torch.int64, device="cpu")

            # In order to ensure that the step counter is checkpointed correctly, we store it
            # as a tensor (which is replicated across all devices) under the first parameter's state.
            self.state[group[PARAMS][0]][STEP] = state_lists[STEP]

    @torch.no_grad()
    def _instantiate_momentum(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[MOMENTUM] == 0.0:
                continue

            # Construct local momentum list.
            local_momentum_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].local_blocked_params,
                state_lists[DISTRIBUTOR].local_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[block_info.param], (
                    f"{block_index=} not found in {self.state[block_info.param]=}. "
                    "Please check the initialization of self.state[block_info.param][block_index] "
                    "within _initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                    "within Distributor for the correctness of block_index."
                )
                block_state = self.state[block_info.param][block_index]

                block_state[MOMENTUM] = block_info.allocate_zeros_tensor(
                    size=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                local_momentum_list.append(block_info.get_tensor(block_state[MOMENTUM]))

            state_lists[MOMENTUM_LIST] = local_momentum_list
            # Here, we set masked momentum list to momentum list because we assume
            # all parameters are active.
            state_lists[MASKED_MOMENTUM_LIST] = state_lists[MOMENTUM_LIST]

    @torch.no_grad()
    def _instantiate_filtered_grads(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[BETAS][0] == 0.0:
                continue

            # Construct local filtered gradient list.
            local_filtered_grad_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].local_blocked_params,
                state_lists[DISTRIBUTOR].local_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[block_info.param], (
                    f"{block_index=} not found in {self.state[block_info.param]=}. "
                    "Please check the initialization of self.state[block_info.param][block_index] "
                    "within _initialize_blocked_parameters_state, and check the initialization of BlockInfo "
                    "within Distributor for the correctness of block_index."
                )
                block_state = self.state[block_info.param][block_index]

                block_state[FILTERED_GRAD] = block_info.allocate_zeros_tensor(
                    size=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                local_filtered_grad_list.append(
                    block_info.get_tensor(block_state[FILTERED_GRAD])
                )

            state_lists[FILTERED_GRAD_LIST] = local_filtered_grad_list
            # Here, we set masked filtered grad list to filtered grad list because we assume
            # all parameters are active.
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[FILTERED_GRAD_LIST]

    @torch.no_grad()
    def _instantiate_per_group_step(
        self, shampoo_pt2_compile_config: ShampooPT2CompileConfig | None
    ) -> None:
        # Use PT2 to compile the step function for each parameter group.
        self._per_group_step: Callable[..., None] = (
            torch.compile(
                self._per_group_step_impl, **asdict(shampoo_pt2_compile_config)
            )
            if shampoo_pt2_compile_config is not None
            else self._per_group_step_impl
        )
        if shampoo_pt2_compile_config is not None:
            logger.info(
                f"DistributedShampoo optimizer initialization is using {shampoo_pt2_compile_config=}"
            )

    @staticmethod
    @torch.no_grad()
    def _mask_state_lists(
        state_lists: dict[str, Any],
        group: dict[str, Any],
        shampoo_pt2_enabled: bool = False,
    ) -> None:
        if (
            state_lists[DISTRIBUTOR].local_grad_selector
            == state_lists[PREVIOUS_GRAD_SELECTOR]
        ):
            return

        # Warning for potential PT2 recompile due to gradient selector change.
        # This warning is expected in either training from scratch or reloading from a checkpoint, as state_lists[PREVIOUS_GRAD_SELECTOR] is initialized to `None`, triggering this warning.
        if state_lists[PREVIOUS_GRAD_SELECTOR] is not None and shampoo_pt2_enabled:
            grad_selector_different = [
                a ^ b
                for a, b in zip(
                    state_lists[DISTRIBUTOR].local_grad_selector,
                    state_lists[PREVIOUS_GRAD_SELECTOR],
                    strict=True,
                )
            ]
            mismatch_grad_selector_indices = [
                i
                for i, is_grad_selector_different in enumerate(grad_selector_different)
                if is_grad_selector_different
            ]
            logger.warning(
                f"""PT2 will recompile because the gradient selction of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model.
                Details:
                - Current step: {state_lists[STEP].item()}
                - Changed gradient selector indices: {mismatch_grad_selector_indices}"""
            )

        # Updates masked state lists if previous block selector disagrees with current selector.
        # State list compression is necessary in order to avoid handling gradients with None.
        state_lists[PREVIOUS_GRAD_SELECTOR] = state_lists[
            DISTRIBUTOR
        ].local_grad_selector
        state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
            DISTRIBUTOR
        ].local_masked_blocked_params
        state_lists[SHAMPOO_PRECONDITIONER_LIST].compress_preconditioner_list(
            local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
        )
        if group[GRAFTING_CONFIG] is not None:
            state_lists[GRAFTING_PRECONDITIONER_LIST].compress_preconditioner_list(
                local_grad_selector=state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[BETAS][0] != 0.0:
            state_lists[MASKED_FILTERED_GRAD_LIST] = compress_list(
                state_lists[FILTERED_GRAD_LIST],
                state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[MOMENTUM] != 0.0:
            state_lists[MASKED_MOMENTUM_LIST] = compress_list(
                state_lists[MOMENTUM_LIST],
                state_lists[DISTRIBUTOR].local_grad_selector,
            )

    @torch.no_grad()
    @torch.compiler.disable
    def _precondition_and_grafting(
        self,
        state_lists: dict[str, Any],
        masked_filtered_grad_list: tuple[torch.Tensor, ...],
        use_grafting_method: bool,
        grafting_config_not_none: bool,
    ) -> tuple[torch.Tensor, ...]:
        # Precondition gradients.
        # If the step count is less than start_preconditioning_step, then we use the grafting method.
        # Assumes that the step state is consistent across all parameters.
        if use_grafting_method:
            masked_blocked_search_directions = state_lists[
                GRAFTING_PRECONDITIONER_LIST
            ].precondition(
                masked_grad_list=masked_filtered_grad_list,
            )

        # Otherwise, we use Shampoo.
        else:
            masked_blocked_search_directions = state_lists[
                SHAMPOO_PRECONDITIONER_LIST
            ].precondition(
                masked_grad_list=masked_filtered_grad_list,
            )

            # Apply grafting.
            if grafting_config_not_none:
                grafting_norm_list = torch._foreach_norm(
                    state_lists[GRAFTING_PRECONDITIONER_LIST].precondition(
                        masked_grad_list=masked_filtered_grad_list,
                    )
                )
                shampoo_norm_list = torch._foreach_norm(
                    masked_blocked_search_directions
                )
                torch._foreach_add_(shampoo_norm_list, 1e-16)
                torch._foreach_div_(grafting_norm_list, shampoo_norm_list)
                torch._foreach_mul_(
                    masked_blocked_search_directions, grafting_norm_list
                )

        return masked_blocked_search_directions

    @torch.no_grad()
    def _add_l2_regularization(
        self,
        state_lists: dict[str, Any],
        weight_decay: float,
        use_decoupled_weight_decay: bool,
    ) -> None:
        # Add L2 regularization / weight decay to the gradient if weight decay is not decoupled.
        if weight_decay != 0.0 and not use_decoupled_weight_decay:
            torch._foreach_add_(
                state_lists[MASKED_BLOCKED_GRADS],
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

    @torch.no_grad()
    def _update_preconditioners(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        perform_amortized_computation: bool,
        grafting_config_not_none: bool,
    ) -> None:
        # Update Shampoo and grafting preconditioners.
        state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
            perform_amortized_computation=perform_amortized_computation,
        )
        if grafting_config_not_none:
            state_lists[GRAFTING_PRECONDITIONER_LIST].update_preconditioners(
                masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )

    @torch.no_grad()
    def _compute_filtered_grad_list(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        beta1: float,
        beta3: float,
        use_bias_correction: bool,
    ) -> tuple[torch.Tensor, ...]:
        if beta1 != 0.0:
            # Computes filtered gradient or EMA of the gradients with respect to beta3 if beta3 != beta1.
            masked_filtered_grad_list = (
                torch._foreach_lerp(
                    state_lists[MASKED_FILTERED_GRAD_LIST],
                    state_lists[MASKED_BLOCKED_GRADS],
                    weight=1 - beta3,
                )
                if beta3 != beta1
                else state_lists[MASKED_FILTERED_GRAD_LIST]
            )

            # Update EMA of the gradients (with respect to beta1).
            torch._foreach_lerp_(
                state_lists[MASKED_FILTERED_GRAD_LIST],
                state_lists[MASKED_BLOCKED_GRADS],
                weight=1 - beta1,
            )

            # Apply bias correction if necessary.
            if use_bias_correction:
                bias_correction1 = 1.0 - beta3 * beta1 ** (step - 1)
                masked_filtered_grad_list = torch._foreach_div(
                    masked_filtered_grad_list,
                    bias_correction1,
                )
        else:
            masked_filtered_grad_list = state_lists[MASKED_BLOCKED_GRADS]

        return masked_filtered_grad_list

    @torch.no_grad()
    def _apply_decoupled_weight_decay(
        self,
        state_lists: dict[str, Any],
        masked_blocked_search_directions: tuple[torch.Tensor, ...],
        weight_decay: float,
        use_decoupled_weight_decay: bool,
    ) -> None:
        # Apply decoupled weight decay.
        if weight_decay != 0.0 and use_decoupled_weight_decay:
            torch._foreach_add_(
                masked_blocked_search_directions,
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

    @torch.no_grad()
    def _update_momentum(
        self,
        state_lists: dict[str, Any],
        masked_blocked_search_directions: tuple[torch.Tensor, ...],
        momentum_param: float,
        dampening: float,
        use_nesterov: bool,
    ) -> None:
        # Update momentum optimizer state and use momentum / Nesterov if enabled.
        if momentum_param != 0.0:
            torch._foreach_mul_(state_lists[MASKED_MOMENTUM_LIST], momentum_param)
            torch._foreach_add_(
                state_lists[MASKED_MOMENTUM_LIST],
                masked_blocked_search_directions,
                alpha=1 - dampening,
            )

            # Incorporates Nesterov momentum.
            if use_nesterov:
                torch._foreach_mul_(
                    masked_blocked_search_directions,
                    1 - dampening,
                )
                torch._foreach_add_(
                    masked_blocked_search_directions,
                    state_lists[MASKED_MOMENTUM_LIST],
                    alpha=momentum_param,
                )
            else:
                torch._foreach_copy_(
                    masked_blocked_search_directions,
                    state_lists[MASKED_MOMENTUM_LIST],
                )

    @torch.no_grad()
    def _compute_search_directions(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        beta3: float,
        weight_decay: float,
        momentum_param: float,
        dampening: float,
        grafting_config_not_none: bool,
        perform_amortized_computation: bool,
        use_decoupled_weight_decay: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        use_nesterov: bool,
    ) -> tuple[torch.Tensor, ...]:
        # Incorporate L2-regularization or (coupled) weight decay if enabled.
        #   G <- G + lr * weight_decay * W
        self._add_l2_regularization(
            state_lists,
            weight_decay,
            use_decoupled_weight_decay,
        )

        # Update Shampoo and grafting preconditioners.
        # Example for AdaGrad accumulation:
        # 1. Update factor matrices/grafting preconditioners.
        #   L <- L + G * G^T
        #   R <- R + G^T * G
        #   V <- V + G^2    (element-wise)
        #   (and similar)
        # 2. Compute root inverse if necessary.
        #   L_inv <- L ** (-1/4)
        #   R_inv <- R ** (-1/4)
        #   (and similar);
        self._update_preconditioners(
            state_lists=state_lists,
            step=step,
            perform_amortized_computation=perform_amortized_computation,
            grafting_config_not_none=grafting_config_not_none,
        )

        # Compute filtered gradient or EMA of the gradients if beta1 > 0 and beta3 > 0.
        # Note that we use two beta factors here akin to Lion.
        #   G_bar <- beta3 * G_tilde + (1 - beta3) * G
        #   G_tilde <- beta1 * G_tilde + (1 - beta1) * G
        masked_filtered_grad_list = self._compute_filtered_grad_list(
            state_lists,
            step,
            beta1,
            beta3,
            use_bias_correction,
        )

        # Precondition and graft filtered gradients.
        # PT2 compile is currently disabled for preconditioning and grafting.
        # NOTE: Preconditioning and grafting is not compatible with PT2 compile.
        #
        #   P_shampoo <- L_inv * G_bar * R_inv (and similar)
        #   P_grafting <- G_bar / (sqrt(V) + epsilon)
        #   P <- P_grafting                                     if step < start_preconditioning_step
        #   P <- ||P_grafting|| / ||P_shampoo|| * P_shampoo     otherwise
        masked_blocked_search_directions = self._precondition_and_grafting(
            state_lists,
            masked_filtered_grad_list,
            use_grafting_method,
            grafting_config_not_none,
        )

        # Incorporate decoupled weight decay into search direction if enabled.
        #   P <- P + weight_decay * W
        self._apply_decoupled_weight_decay(
            state_lists,
            masked_blocked_search_directions,
            weight_decay,
            use_decoupled_weight_decay,
        )

        # Update momentum optimizer state and use momentum / Nesterov if enabled.
        #   M <- momentum_param * M + (1 - dampening) * P
        #   P <- (1 - dampening) * P + momentum_param * M     if use_nesterov
        #   P <- M                                            otherwise.
        self._update_momentum(
            state_lists,
            masked_blocked_search_directions,
            momentum_param,
            dampening,
            use_nesterov,
        )

        torch._foreach_mul_(masked_blocked_search_directions, -lr)

        return masked_blocked_search_directions

    @torch.no_grad()
    def _per_group_step_impl(
        self,
        state_lists: dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        beta3: float,
        weight_decay: float,
        momentum_param: float,
        dampening: float,
        grafting_config_not_none: bool,
        perform_amortized_computation: bool,
        use_decoupled_weight_decay: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        use_nesterov: bool,
    ) -> None:
        # This method computes search directions and updates parameters in one step
        # It's designed to be compiled with PyTorch 2.0 for performance optimization

        # Call update_params on the distributor with the computed search directions
        # The distributor is responsible for applying updates to the actual parameters
        state_lists[DISTRIBUTOR].update_params(
            # Compute search directions based on current state and optimization parameters
            # This returns the directions in which parameters should be updated
            masked_blocked_search_directions=self._compute_search_directions(
                state_lists=state_lists,
                step=step,
                lr=lr,
                beta1=beta1,
                beta3=beta3,
                weight_decay=weight_decay,
                momentum_param=momentum_param,
                dampening=dampening,
                grafting_config_not_none=grafting_config_not_none,
                perform_amortized_computation=perform_amortized_computation,
                use_decoupled_weight_decay=use_decoupled_weight_decay,
                use_bias_correction=use_bias_correction,
                use_grafting_method=use_grafting_method,
                use_nesterov=use_nesterov,
            )
            # Only update parameters if there are gradients to use
            # Otherwise, return an empty tuple to avoid unnecessary computation
            if state_lists[MASKED_BLOCKED_GRADS]
            else ()
        )

    @overload
    @torch.no_grad()
    def step(self, closure: None = None) -> None: ...

    @overload
    @torch.no_grad()
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Performs a single optimization step.

        Args:
            closure (Callable[[], float] | Nnoe): A closure that reevaluates the model and returns the loss. (Default: None)

        Returns:
            loss (float | None): The loss value returned by the closure if provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Construct blocked gradient list.
            state_lists[MASKED_BLOCKED_GRADS] = state_lists[
                DISTRIBUTOR
            ].merge_and_block_gradients()

            # Based on the current block selector, mask lists of parameters and optimizer states.
            DistributedShampoo._mask_state_lists(
                state_lists=state_lists,
                group=group,
                shampoo_pt2_enabled=self._shampoo_pt2_compile_config is not None,
            )

            # Iterate group step counter and define Python scalar step.
            step = state_lists[STEP].add_(1)
            # NOTE: Wrap scalar of group[LR] into a 0D tensor to avoid PT2 recompilation;
            # Send 0D tensor to GPU in `non_blocking` to avoid QPS regression. Remove the gpu
            # tensor impl once PT2 supports cpu 0D tensor properly.
            lr = torch.tensor(
                group[LR], dtype=torch.float, pin_memory=group[USE_PIN_MEMORY]
            ).to(
                # NOTE: Assume all parameter groups consistently exist on the same rank.
                state_lists[DISTRIBUTOR].local_blocked_params[0].device,
                non_blocking=True,
            )
            beta1 = group[BETAS][0]
            beta3 = group[BETA3]
            weight_decay = group[WEIGHT_DECAY]
            momentum_param = group[MOMENTUM]
            dampening = group[DAMPENING]
            grafting_config_not_none = group[GRAFTING_CONFIG] is not None
            perform_amortized_computation = (
                step.item() % group[PRECONDITION_FREQUENCY] == 0
                and step.item() > group[START_PRECONDITIONING_STEP]
                or step.item() == group[START_PRECONDITIONING_STEP]
            )
            use_decoupled_weight_decay = group[USE_DECOUPLED_WEIGHT_DECAY]
            use_bias_correction = group[USE_BIAS_CORRECTION]
            # Check applying grafting method or not
            use_grafting_method = (
                step.item() < group[START_PRECONDITIONING_STEP]
                and grafting_config_not_none
            )
            use_nesterov = group[USE_NESTEROV]

            self._per_group_step(
                state_lists,
                step,
                lr,
                beta1,
                beta3,
                weight_decay,
                momentum_param,
                dampening,
                grafting_config_not_none,
                perform_amortized_computation,
                use_decoupled_weight_decay,
                use_bias_correction,
                use_grafting_method,
                use_nesterov,
            )

            # Explicitly set masked blocked gradients to None to save memory so the original param.grad has no pointer to it.
            state_lists[MASKED_BLOCKED_GRADS] = None

        return loss

    def state_dict(self) -> StateDict:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard state_dict() method for checkpointing!"
        )

    def load_state_dict(self, state_dict: StateDict) -> None:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard load_state_dict() method for checkpointing!"
        )

    @staticmethod
    def _construct_param_group_key(
        group: dict[str, Any], param_to_key: dict[torch.Tensor, str]
    ) -> str:
        return "/".join(sorted(param_to_key[param] for param in group[PARAMS]))

    def distributed_state_dict(
        self,
        key_to_param: Iterator[tuple[str, torch.Tensor]],
        save_param_groups: bool = True,
    ) -> StateDict:
        """Distributed state dict simplified from TorchRec's KeyedOptimizer.
        Compatible with torch.distributed.checkpoint with DTensor.

        Returned state and param_groups will contain parameter keys
        instead of parameter indices in torch.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.

        Can also handle classes and supported data structures that follow the PyTorch stateful
        protocol.

        Args:
            key_to_param (Iterator[tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
                maps a FQN to the parameters in the model.
            save_param_groups (bool): Flag for saving parameter groups. (Default: True)

        Returns:
            state_dict (StateDict): Dictionary containing the optimizer state and potentially parameter
                groups.

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        param_to_key = {param: key for key, param in key_to_param}
        ret: StateDict = {
            "state": {
                param_to_key[param]: flatten(extract_state_dict_content(param_state))
                for param, param_state in self.state.items()
            }
        }
        if not save_param_groups:
            return ret

        # Store parameter groups with unique parameter group identifier.
        # NOTE: The parameters are ignored since they are assumed to be checkpointed separately.
        ret["param_groups"] = {
            DistributedShampoo._construct_param_group_key(group, param_to_key): {
                k: deepcopy(v) for k, v in group.items() if k != PARAMS
            }
            for group in self.param_groups
        }

        return ret

    def load_distributed_state_dict(
        self,
        state_dict: StateDict,
        key_to_param: Iterator[tuple[str, torch.Tensor]],
        save_param_groups: bool = True,
        enable_missing_key_check: bool = True,
    ) -> None:
        """Load state dict simplified from TorchRec's KeyedOptimizer.
        Compatible with torch.distributed.checkpoint.

        This implementation is much stricter than the one in torch.Optimizer:
        it requires implementations to fully initialize their state during first optimization iteration,
        and it prohibits loading an empty state into already initialized KeyedOptimizer and vise versa.

        Because of introduced strictness it allows us to:
            * do compatibility checks for state and param_groups, which improves usability
            * avoid state duplication by directly copying into state tensors, e.g.
              optimizer.step()  # make sure optimizer is initialized
              sd = optimizer.state_dict()
              load_checkpoint(sd)  # copy state directly into tensors, re-shard if needed
              optimizer.load_state_dict(sd)  # replace param_groups

        Args:
            state_dict (StateDict): State dictionary to load containing the optimizer state and
                parameter groups.
            key_to_param (Iterator[tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
                maps a FQN to the parameters in the model.
            save_param_groups (bool): Flag for saving parameter groups. (Default: True)
            enable_missing_key_check (bool): Flag for enabling missing key check. (Default: True)

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        state_to_load = state_dict["state"]
        key_to_param_mapping = dict(key_to_param)

        # Load state
        for param_key, param_state in state_to_load.items():
            # Check if parameter exists in current parameter state dict.
            if param_key not in key_to_param_mapping:
                if enable_missing_key_check:
                    raise KeyError(
                        f"Parameter key {param_key} not found in key_to_param mapping!"
                    )
                else:
                    logger.warning(
                        f"Parameter key {param_key} not found in key_to_param mapping!"
                    )
                    continue

            param = key_to_param_mapping[param_key]

            if param not in self.state:
                if enable_missing_key_check:
                    raise KeyError(f"Parameter {param} not found in state!")
                else:
                    logger.warning(f"Parameter {param} not found in state!")
                    continue

            # Update parameter state.
            update_param_state_dict_object(
                self.state[param],
                unflatten(param_state),
            )

        # Load param_groups.
        if save_param_groups:
            param_groups_to_load = state_dict["param_groups"]

            if len(self.param_groups) != len(param_groups_to_load):
                raise ValueError(
                    f"Different param_groups count: {len(self.param_groups)} vs {len(param_groups_to_load)}"
                )
            param_to_key = {param: key for key, param in key_to_param_mapping.items()}

            # Loading the parameter group based on the unique parameter group key.
            for group in self.param_groups:
                param_group_key = DistributedShampoo._construct_param_group_key(
                    group, param_to_key
                )
                if param_group_key not in param_groups_to_load:
                    raise ValueError(
                        f"Param group {param_group_key} not found in param_groups_to_load!"
                    )
                group |= {
                    key: deepcopy(value)
                    for key, value in param_groups_to_load[param_group_key].items()
                }
