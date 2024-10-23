"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import contextlib
import dataclasses
import logging
from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    BETA3,
    BETAS,
    DAMPENING,
    DDPShampooConfig,
    DistributedConfig,
    DISTRIBUTOR,
    EPSILON,
    FILTERED_GRAD,
    FILTERED_GRAD_LIST,
    FSDPShampooConfig,
    FullyShardShampooConfig,
    GRAFTING_CONFIG,
    GRAFTING_PRECONDITIONER_LIST,
    GraftingConfig,
    HSDPShampooConfig,
    INV_ROOT_OVERRIDE,
    LR,
    MASKED_BLOCKED_GRADS,
    MASKED_BLOCKED_PARAMS,
    MASKED_FILTERED_GRAD_LIST,
    MASKED_MOMENTUM_LIST,
    MAX_PRECONDITIONER_DIM,
    MOMENTUM,
    MOMENTUM_LIST,
    PARAMS,
    PRECISION_CONFIG,
    PrecisionConfig,
    PRECONDITION_FREQUENCY,
    PREVIOUS_GRAD_SELECTOR,
    RMSpropGraftingConfig,
    ROOT_INV_CONFIG,
    SGDGraftingConfig,
    SHAMPOO_PRECONDITIONER_LIST,
    ShampooPT2CompileConfig,
    START_PRECONDITIONING_STEP,
    STEP,
    USE_BIAS_CORRECTION,
    USE_DECOUPLED_WEIGHT_DECAY,
    USE_MERGE_DIMS,
    USE_NESTEROV,
    WEIGHT_DECAY,
)

from distributed_shampoo.utils.shampoo_checkpoint_utils import (
    extract_state_dict_content,
    flatten,
    unflatten,
    update_param_state_dict_object,
)
from distributed_shampoo.utils.shampoo_ddp_distributor import DDPDistributor
from distributed_shampoo.utils.shampoo_distributor import Distributor
from distributed_shampoo.utils.shampoo_fsdp_distributor import FSDPDistributor
from distributed_shampoo.utils.shampoo_fully_shard_distributor import (
    FullyShardDistributor,
)
from distributed_shampoo.utils.shampoo_hsdp_distributor import HSDPDistributor

from distributed_shampoo.utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    DequantizePreconditionersContext,
    SGDPreconditionerList,
    ShampooPreconditionerList,
)
from distributed_shampoo.utils.shampoo_quantization import (
    DequantizeQuantizedTensorListContext,
    QuantizedTensor,
    QuantizedTensorList,
)
from distributed_shampoo.utils.shampoo_utils import compress_list

from matrix_functions_types import DefaultEigenConfig, EigenConfig, RootInvConfig
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
            - GraftingType.RMSPROP: Grafts the RMSProp method.
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

        - DDPShampooConfig: Supports multi-GPU distributed data-parallel training via torch.distributed. Assigns optimizer states
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

        - FSDPShampooConfig: Supports multi-GPU fully-sharded data-parallel training via torch.distributed. This option uses
            additional metadata in order to reconstruct valid tensor blocks of the original parameter from the flattened parameter
            representation.

            Distributed Training Specific Fields:
                - param_to_metadata: One must create a dictionary containing the metadata for each parameter in the FSDP model. This
                    includes the shape of the original parameter as well as the start and end indices of the tensor shard with
                    respect to the unsharded flattened parameter.

            Requirements:
                - torch.distributed must be initialized in advance.
                - One must enable the option use_orig_params = True in FSDP.

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
            ShampooPT2CompileConfig
                - pytorch_compile_backend: PT2 backend to use. All available backends in pytorch 2.0 is available for Shampoo. Typical backends to use
                    include 'inductor', 'aot_eager'. For more details: https://pytorch.org/docs/stable/torch.compiler.html
                - enable_shampoo_pt2_dynamic_shape: if true, PT2 will compile Shampoo data/tensors with `dynamic shape` mode. Default is False and use
                    `static` mode. `dynamic shape` means the tensor shapes can change from run to run, and PT2 will generate kernels not specialized to
                    particular tensor shape. Recommended to use `static` mode here for Shampoo.
                    More about dynamic shape: https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html

    Args:
        params (ParamsT): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate. (Default: 1e-2)
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            (Default: (0.9, 1.0))
        beta3 (float): Coefficient used for computing running average of gradient only for the current iteration.
            This can be used to replicate a version of NAdam if set appropriately. For example, if beta1 = 0.9, then applying
            beta1 interpolation a second time is equivalent to setting beta3 = 0.9 * 0.9 = 0.81.
            If set to -1.0, will set equal to beta1. (Default: -1.0)
        epsilon (float): Term added to the denominator to improve numerical stability. (Default: 1e-12)
        momentum (float): Momentum parameter. (Default: 0.)
        dampening (float): Dampening parameter for momentum. (Default: 0.)
        weight_decay (float): Weight decay (L2 penalty). (Default: 0.)
        max_preconditioner_dim (int): Maximum preconditioner dimensio. (Default: 1024)
        precondition_frequency (int): Frequency for computing root inverse preconditioner. (Default: 1)
        start_preconditioning_step (int): Iteration to start computing inverse preconditioner. If -1, uses
            the same value as precondition_frequency. (Default: -1)
        inv_root_override (int, Sequence[int]): Inverse root to use in Shampoo. If a list [l1, l2, ..., lp], then we will
            use -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on. If the order of the
            tensor exceeds the order of the tensor, reverts to the default value. If 0 is used, uses the default inverse
            root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float | None): **DEPRECATING** Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: None)
        use_nesterov (bool): Flag for using Nesterov momentum. (default: False)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        use_decoupled_weight_decay (bool): Flag for using AdamW-style decoupled weight decay. (Default: True)
        grafting_config (Optional[GraftingConfig]): Configuration for grafting method. If None, ignores grafting.
            (Default: None)
        use_merge_dims (bool): Merge dimensions if possible while respecting max_preconditioner_dim. (Default: True)
        use_pytorch_compile (Optional[bool]): Use PyTorch 2.0 compiler feature to speed up training. Deprecating, please use
            shampoo_pt2_compile_config instead; when this field is None, the use of PyTorch 2.0 compiler is decided by
            shampoo_pt2_compile_config. (Default: None)
        shampoo_pt2_compile_config (Optional[ShampooPT2CompileConfig]): Configuration for Shampoo PT2 compilation. If None,
            ignores compilation, and Shampoo will run in eager mode. (Default: None)
        distributed_config (Optional[DistributedConfig]): Configuration for applying Shampoo
            to different distributed training frameworks, such as distributed-data parallel (DDP) training.
            Based on the configuration, determines which version of Shampoo to use. (Default: None)
        preconditioner_dtype (Optional[torch.dtype]): **DEPRECATING** Data type for preconditioner. (Default: None)
        precision_config (PrecisionConfig): Data types for optimizer states. (Default: all fields torch.float)
        use_protected_eigh (bool): **DEPRECATED** Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition in higher precision if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        track_root_inv_residuals (bool): Track errors and residuals of root inverse. For debugging purposes.
            (Default: False)
        root_inv_config (RootInvConfig): Configuration for root inverse computation. (Default: DefaultEigenConfig)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        beta3: float = -1.0,
        epsilon: float = 1e-12,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        inv_root_override: Union[int, Sequence[int]] = 0,
        exponent_multiplier: float | None = None,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        grafting_config: Optional[GraftingConfig] = None,
        use_merge_dims: bool = True,
        use_pytorch_compile: Optional[bool] = None,
        shampoo_pt2_compile_config: Optional[ShampooPT2CompileConfig] = None,
        distributed_config: Optional[DistributedConfig] = None,
        preconditioner_dtype: Optional[torch.dtype] = None,
        precision_config: Optional[PrecisionConfig] = None,
        use_protected_eigh: bool = True,
        track_root_inv_residuals: bool = False,
        root_inv_config: RootInvConfig = DefaultEigenConfig,
    ) -> None:
        # Hyperparameter checks.
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}. Must be >= 0.0.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}. Must be in [0.0, 1.0)."
            )
        if not 0.0 < betas[1] <= 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}. Must be in (0.0, 1.0]."
            )
        if beta3 == -1.0:
            beta3 = betas[0]
        elif not 0.0 <= beta3 < 1.0:
            raise ValueError(
                f"Invalid beta3 parameter: {beta3}. Must be in [0.0, 1.0)."
            )
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Must be > 0.0.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Invalid momentum parameter: {momentum}. Must be [0.0, 1.0)."
            )
        if not 0.0 <= dampening < 1.0:
            raise ValueError(
                f"Invalid damping parameter: {dampening}. Must be [0.0, 1.0)."
            )
        if not weight_decay >= 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}. Must be >= 0.0."
            )
        if not max_preconditioner_dim >= 1:
            raise ValueError(
                f"Invalid max preconditioner dimension: {max_preconditioner_dim}. Must be >= 1."
            )
        if not precondition_frequency >= 1:
            raise ValueError(
                f"Invalid precondition frequency: {precondition_frequency}. Must be >= 1."
            )
        if not start_preconditioning_step >= -1:
            raise ValueError(
                f"Invalid start preconditioning step: {start_preconditioning_step}. Must be >= -1."
            )
        if isinstance(inv_root_override, Sequence):
            if not all(e >= 0 for e in inv_root_override):
                raise ValueError(
                    f"Invalid exponent override list: {inv_root_override}. All values must be >= 0."
                )
        else:
            if not inv_root_override >= 0:
                raise ValueError(
                    f"Invalid exponent override: {inv_root_override}. Must be >= 0."
                )
        if track_root_inv_residuals:
            logger.setLevel(logging.DEBUG)

        # Provide warning/error for start_preconditioning_step.
        if start_preconditioning_step == -1:
            start_preconditioning_step = precondition_frequency
            logger.warning(
                "start_preconditioning_step set to -1. Setting start_preconditioning_step equal to "
                f"precondition frequency {precondition_frequency} by default."
            )
        if start_preconditioning_step < precondition_frequency:
            raise ValueError(
                f"Invalid start_preconditioning_step value: {start_preconditioning_step}. Must be >= {precondition_frequency=}."
            )

        # Warn when Nesterov is used but momentum is 0.
        if use_nesterov and momentum == 0.0:
            logger.warning(
                "Nesterov flag is enabled but momentum parameter is zero! "
                "Continuing without using momentum or Nesterov acceleration..."
            )

        # Deprecation warning for use_pytorch_compile
        if use_pytorch_compile is not None:
            if use_pytorch_compile and shampoo_pt2_compile_config is None:
                shampoo_pt2_compile_config = ShampooPT2CompileConfig()
                logger.warning(
                    "use_pytorch_compile is deprecating. Please use shampoo_pt2_compile_config instead."
                )
            elif use_pytorch_compile and shampoo_pt2_compile_config is not None:
                raise ValueError(
                    "Both use_pytorch_compile and shampoo_pt2_compile_config are provided. Please use only shampoo_pt2_compile_config as use_pytorch_compile is deprecating."
                )
            elif not use_pytorch_compile and shampoo_pt2_compile_config is not None:
                raise ValueError(
                    "use_pytorch_compile=False conflicts with non-None shampoo_pt2_compile_config arg. Please use only shampoo_pt2_compile_config as use_pytorch_compile is deprecating."
                )

        # Provide error for system Pytorch compile availability
        if shampoo_pt2_compile_config is not None and not torch.cuda.is_available():
            raise ValueError(
                "Backend does NOT support Pytorch 2.0 compile. Switch to use_pytorch_compile in (False, None) and shampoo_pt2_compile_config=None."
            )

        # Deprecation warning for preconditioner_dtype
        if preconditioner_dtype is not None:
            if precision_config is None:
                precision_config = PrecisionConfig(
                    factor_matrix_dtype=preconditioner_dtype,
                    factor_matrix_computation_dtype=preconditioner_dtype,
                )
                logger.warning(
                    "preconditioner_dtype is deprecated. Please use precision_config instead."
                )
            else:
                raise ValueError(
                    "Both preconditioner_dtype and precision_config are provided. Please use only precision_config as preconditioner_dtype is deprecated."
                )

        # Create default precision config if it is not provided.
        if precision_config is None:
            precision_config = PrecisionConfig()

        # Set exponent multiplier if this is not provided.
        if isinstance(root_inv_config, EigenConfig) and exponent_multiplier is not None:
            logger.warning(
                f"{exponent_multiplier=} is deprecating. Please consider using EigenConfig.exponent_multiplier directly and setting exponent_multipler=None instead in the future."
            )
            root_inv_config = dataclasses.replace(
                root_inv_config, exponent_multiplier=exponent_multiplier
            )

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
                INV_ROOT_OVERRIDE: inv_root_override,
                USE_NESTEROV: use_nesterov,
                USE_BIAS_CORRECTION: use_bias_correction,
                USE_DECOUPLED_WEIGHT_DECAY: use_decoupled_weight_decay,
                GRAFTING_CONFIG: grafting_config,
                USE_MERGE_DIMS: use_merge_dims,
                PRECISION_CONFIG: precision_config,
                ROOT_INV_CONFIG: root_inv_config,
            },
        )

        # Initialize non-group-related fields.
        self._track_root_inv_residuals = track_root_inv_residuals

        # Initialize list containing group state dictionaries.
        self._per_group_state_lists: List[Dict[str, Any]] = [
            {} for _ in self.param_groups
        ]

        # Block parameters and instantiate optimizer states.
        self._instantiate_distributor(distributed_config)
        self._instantiate_shampoo_preconditioner_list(use_protected_eigh)
        self._instantiate_grafting()
        self._instantiate_steps()
        self._instantiate_momentum()
        self._instantiate_filtered_grads()
        self._instantiate_device()
        self._instantiate_per_group_step(shampoo_pt2_compile_config)

    @torch.no_grad()
    def _instantiate_distributor(
        self, distributed_config: DistributedConfig | None
    ) -> None:
        if distributed_config is None:
            distributor = Distributor
        elif type(distributed_config) is DDPShampooConfig:
            distributor = partial(DDPDistributor, distributed_config=distributed_config)
        elif type(distributed_config) is FSDPShampooConfig:
            distributor = partial(
                FSDPDistributor, distributed_config=distributed_config
            )
        elif type(distributed_config) is FullyShardShampooConfig:
            distributor = FullyShardDistributor
        elif type(distributed_config) is HSDPShampooConfig:
            distributor = partial(
                HSDPDistributor,
                distributed_config=distributed_config,
            )
        else:
            raise NotImplementedError(f"{distributed_config=} not supported!")

        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Instantiate distributors for each group.
            state_lists[DISTRIBUTOR] = distributor(group)

            # If the number of trainers is more than the number of blocks,
            # some workers might not get any parameters which cause wasting resources because
            # those trainers are working on nothing.
            assert state_lists[
                DISTRIBUTOR
            ].local_blocked_params, f"Some workers have no parameters to work on. This mostly happens when the value of num_trainers_per_group field in {distributed_config=} is more than the number of local blocked params on a single device. Please check the num_trainers_per_group setting and consider reducing it."

            # Compile blocked parameters and block-to-parameter metadata into group lists.
            state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
                DISTRIBUTOR
            ].local_blocked_params
            # First PREVIOUS_GRAD_SELECTOR is set to None.
            state_lists[PREVIOUS_GRAD_SELECTOR] = None

    @torch.no_grad()
    def _instantiate_shampoo_preconditioner_list(
        self, use_protected_eigh: bool
    ) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            state_lists[SHAMPOO_PRECONDITIONER_LIST] = ShampooPreconditionerList(
                block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                state=self.state,
                block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                root_inv_config=group[ROOT_INV_CONFIG],
                beta2=group[BETAS][1],
                epsilon=group[EPSILON],
                inv_root_override=group[INV_ROOT_OVERRIDE],
                use_bias_correction=group[USE_BIAS_CORRECTION],
                precision_config=group[PRECISION_CONFIG],
                use_protected_eigh=use_protected_eigh,
            )

    @torch.no_grad()
    def _instantiate_grafting(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[GRAFTING_CONFIG] is None:
                state_lists[GRAFTING_PRECONDITIONER_LIST] = None
            elif type(group[GRAFTING_CONFIG]) is SGDGraftingConfig:
                state_lists[GRAFTING_PRECONDITIONER_LIST] = SGDPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                )
            elif type(group[GRAFTING_CONFIG]) in (
                AdaGradGraftingConfig,
                RMSpropGraftingConfig,
                AdamGraftingConfig,
            ):
                state_lists[GRAFTING_PRECONDITIONER_LIST] = AdagradPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                    distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                    precision_config=group[PRECISION_CONFIG],
                    beta2=(
                        1.0
                        if type(group[GRAFTING_CONFIG]) is AdaGradGraftingConfig
                        else group[GRAFTING_CONFIG].beta2
                    ),
                    epsilon=group[GRAFTING_CONFIG].epsilon,
                    use_bias_correction=type(group[GRAFTING_CONFIG])
                    is AdamGraftingConfig,
                )
            else:
                raise NotImplementedError(
                    f"Unsupported grafting config: {group[GRAFTING_CONFIG]=}."
                )

    @torch.no_grad()
    def _instantiate_steps(self) -> None:
        for state_lists in self._per_group_state_lists:
            assert (
                len(state_lists[DISTRIBUTOR].global_block_info_list) > 0
            ), "There is no params in your param_group. Please check the instantiation of DistributedShampoo "
            'with param_group containing no params. For example, DistributedShampoo(params=[{"params": []}])'
            # NOTE: We instantiate a single step tensor on CPU for each group in order
            #       to track the number of steps taken by all parameters within the group.
            #       Instantiating on CPU avoids GPU synchronization.
            state_lists[STEP] = torch.tensor(0, dtype=torch.int64, device="cpu")

            # In order to ensure that the step counter is checkpointed correctly, we store it
            # as a tensor (which is replicated across all devices) under the first parameter's state.
            block_info = state_lists[DISTRIBUTOR].global_block_info_list[0]
            self.state[block_info.param][STEP] = state_lists[STEP]

    @torch.no_grad()
    def _instantiate_momentum(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[MOMENTUM] == 0.0:
                continue

            # Construct global momentum list.
            global_momentum_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].global_blocked_params,
                state_lists[DISTRIBUTOR].global_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[
                    block_info.param
                ], f"{block_index=} not found in {self.state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] within "
                "PreconditionerList, and check the initialization of BlockInfo within Distributor "
                "for the correctness of block_index."
                block_state = self.state[block_info.param][block_index]

                block_state[MOMENTUM] = QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        shape=block.size(),
                        dtype=group[PRECISION_CONFIG].momentum_dtype,
                        device=block.device,
                    ),
                    block_info,
                )
                global_momentum_list.append(block_state[MOMENTUM])

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[MOMENTUM_LIST] = QuantizedTensorList(
                compress_list(
                    global_momentum_list,
                    state_lists[DISTRIBUTOR].distributor_selector,
                ),
                group[PRECISION_CONFIG].momentum_dtype,
                group[PRECISION_CONFIG].computation_dtype,
            )
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

            # Construct global filtered gradient list.
            global_filtered_grad_list = []
            for block, block_info in zip(
                state_lists[DISTRIBUTOR].global_blocked_params,
                state_lists[DISTRIBUTOR].global_block_info_list,
                strict=True,
            ):
                assert (
                    block_index := block_info.composable_block_ids[1]
                ) in self.state[
                    block_info.param
                ], f"{block_index=} not found in {self.state[block_info.param]=}. "
                "Please check the initialization of self.state[block_info.param][block_index] "
                "within PreconditionerList, and check the initialization of BlockInfo within "
                "Distributor for the correctness of block_index."
                block_state = self.state[block_info.param][block_index]

                block_state[FILTERED_GRAD] = QuantizedTensor(
                    block_info.allocate_zeros_tensor(
                        shape=block.size(),
                        dtype=group[PRECISION_CONFIG].filtered_grad_dtype,
                        device=block.device,
                    ),
                    block_info,
                )
                global_filtered_grad_list.append(block_state[FILTERED_GRAD])

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[FILTERED_GRAD_LIST] = QuantizedTensorList(
                compress_list(
                    global_filtered_grad_list,
                    state_lists[DISTRIBUTOR].distributor_selector,
                ),
                group[PRECISION_CONFIG].filtered_grad_dtype,
                group[PRECISION_CONFIG].computation_dtype,
            )
            # Here, we set masked filtered grad list to filtered grad list because we assume
            # all parameters are active.
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[FILTERED_GRAD_LIST]

    @torch.no_grad()
    def _instantiate_device(self) -> None:
        # NOTE: Assume all parameter groups consistently exist on the same rank.
        self._device = self._per_group_state_lists[0][MASKED_BLOCKED_PARAMS][0].device

    @torch.no_grad()
    def _instantiate_per_group_step(
        self, shampoo_pt2_compile_config: ShampooPT2CompileConfig | None
    ) -> None:
        # Use PT2 to compile the step function for each parameter group.
        self._per_group_step: Callable[
            [
                Dict[str, Any],
                torch.Tensor,
                torch.Tensor,
                float,
                float,
                float,
                float,
                float,
                bool,
                bool,
                bool,
                bool,
                bool,
                bool,
            ],
            None,
        ] = (
            torch.compile(
                self._per_group_step_impl,
                backend=shampoo_pt2_compile_config.pytorch_compile_backend,
                dynamic=shampoo_pt2_compile_config.enable_shampoo_pt2_dynamic_shape,
            )
            if shampoo_pt2_compile_config is not None
            else self._per_group_step_impl
        )
        if shampoo_pt2_compile_config is not None:
            logger.info(
                f"DistributedShampoo optimizer initialization is using {shampoo_pt2_compile_config.pytorch_compile_backend} backend and enable_shampoo_pt2_dynamic_shape={shampoo_pt2_compile_config.enable_shampoo_pt2_dynamic_shape}"
            )

    @staticmethod
    @torch.no_grad()
    def _mask_state_lists(state_lists: Dict[str, Any], group: Dict[str, Any]) -> None:
        if (
            state_lists[DISTRIBUTOR].local_grad_selector
            == state_lists[PREVIOUS_GRAD_SELECTOR]
        ):
            return

        if state_lists[STEP].item() >= 1:
            logger.warn(
                "PT2 will recompile because the gradient selction of model parameters have changed from the previous step. Possible reasons include some gradients are None. If this is not intended, please check the data and/or model."
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
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[
                MASKED_FILTERED_GRAD_LIST
            ].compress(
                state_lists[DISTRIBUTOR].local_grad_selector,
            )
        if group[MOMENTUM] != 0.0:
            state_lists[MASKED_MOMENTUM_LIST] = state_lists[
                MASKED_MOMENTUM_LIST
            ].compress(
                state_lists[DISTRIBUTOR].local_grad_selector,
            )

    @torch.no_grad()
    def _compute_and_log_root_inverse_residuals(
        self,
    ) -> None:
        """Compute root inverse residuals over all preconditioners.

        Uses infinity norm to evaluate residuals and errors.
        """

        # Accumulate relative errors/residuals
        relative_errors = []
        relative_residuals = []

        for (group_index, group), state_lists in zip(
            enumerate(self.param_groups), self._per_group_state_lists, strict=True
        ):
            # TODO: update values depending on both factor_matrix_dtype and inv_factor_matrix_dtype
            # Get expected relative errors/residuals for debugging purposes
            if group[PRECISION_CONFIG].inv_factor_matrix_dtype == torch.float64:
                expected_relative_error = 1e-7
            elif group[PRECISION_CONFIG].inv_factor_matrix_dtype == torch.float32:
                expected_relative_error = 1e-3
            else:
                logger.warning(
                    "Expected relative error/residual not supported for precision lower than float32."
                )
                continue

            relative_errors, relative_residuals = state_lists[
                SHAMPOO_PRECONDITIONER_LIST
            ].compute_root_inverse_residuals()

            relative_errors = torch.stack(relative_errors)
            relative_residuals = torch.stack(relative_residuals)

            quantiles = torch.as_tensor(
                [0, 0.25, 0.5, 0.75, 1],
                device=relative_errors.device,
                dtype=relative_errors.dtype,
            )
            logger.debug(f"Group Index: {group_index}")
            logger.debug(f"Expect Relative Error <= {expected_relative_error}")
            logger.debug(
                f"Relative Error (||X - X_hat||_inf / ||X||_inf)       Average: {torch.mean(relative_errors)}, "
                f"Quantiles [0, 25, 50, 75, 100]: {torch.quantile(relative_errors, quantiles, interpolation='nearest')}"
            )
            logger.debug(
                f"Relative Residual (||X_hat^-r - A||_inf / ||A||_inf) Average: {torch.mean(relative_residuals)}, "
                "Quantiles [0, 25, 50, 75, 100]: "
                f"{torch.quantile(relative_residuals, quantiles, interpolation='nearest')}"
            )

    @torch.no_grad()
    @torch.compiler.disable
    def _compute_root_inverse(
        self, state_lists: Dict[str, Any], compute_root_inverse: bool
    ) -> None:
        if compute_root_inverse:
            state_lists[SHAMPOO_PRECONDITIONER_LIST].compute_root_inverse()
            if self._track_root_inv_residuals:
                self._compute_and_log_root_inverse_residuals()

    @torch.no_grad()
    @torch.compiler.disable
    def _precondition_and_grafting(
        self,
        state_lists: Dict[str, Any],
        masked_filtered_grad_list: Tuple[torch.Tensor, ...],
        use_grafting_method: bool,
        grafting_config_not_none: bool,
    ) -> Tuple[torch.Tensor, ...]:
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
        state_lists: Dict[str, Any],
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
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        grafting_config_not_none: bool,
    ) -> None:
        # Update Shampoo and grafting preconditioners / factor matrices.
        state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
        )
        if grafting_config_not_none:
            state_lists[GRAFTING_PRECONDITIONER_LIST].update_preconditioners(
                masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )

    @torch.no_grad()
    def _compute_filtered_grad_list(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        beta1: float,
        beta3: float,
        use_bias_correction: bool,
    ) -> Tuple[torch.Tensor, ...]:
        if beta1 != 0.0:
            with DequantizeQuantizedTensorListContext(
                quantized_tensor_list=state_lists[MASKED_FILTERED_GRAD_LIST]
            ):
                # Computes filtered gradient or EMA of the gradients with respect to beta3 if beta3 != beta1.
                masked_filtered_grad_list = (
                    torch._foreach_lerp(
                        state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
                        state_lists[MASKED_BLOCKED_GRADS],
                        weight=1 - beta3,
                    )
                    if beta3 != beta1
                    else state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value
                )

                # Update EMA of the gradients (with respect to beta1).
                torch._foreach_lerp_(
                    state_lists[MASKED_FILTERED_GRAD_LIST].dequantized_value,
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
        state_lists: Dict[str, Any],
        masked_blocked_search_directions: Tuple[torch.Tensor, ...],
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
        state_lists: Dict[str, Any],
        masked_blocked_search_directions: Tuple[torch.Tensor, ...],
        momentum_param: float,
        dampening: float,
        use_nesterov: bool,
    ) -> None:
        # Update momentum optimizer state and use momentum / Nesterov if enabled.
        if momentum_param != 0.0:
            with DequantizeQuantizedTensorListContext(
                quantized_tensor_list=state_lists[MASKED_MOMENTUM_LIST]
            ):
                torch._foreach_mul_(
                    state_lists[MASKED_MOMENTUM_LIST].dequantized_value, momentum_param
                )
                torch._foreach_add_(
                    state_lists[MASKED_MOMENTUM_LIST].dequantized_value,
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
                        state_lists[MASKED_MOMENTUM_LIST].dequantized_value,
                        alpha=momentum_param,
                    )
                else:
                    torch._foreach_copy_(
                        masked_blocked_search_directions,
                        state_lists[MASKED_MOMENTUM_LIST].dequantized_value,
                    )

    @torch.no_grad()
    def _per_group_step_impl(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        beta3: float,
        weight_decay: float,
        momentum_param: float,
        dampening: float,
        grafting_config_not_none: bool,
        compute_root_inverse: bool,
        use_decoupled_weight_decay: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        use_nesterov: bool,
    ) -> None:
        # Incorporate L2-regularization or (coupled) weight decay if enabled.
        #   G <- G + lr * weight_decay * W
        self._add_l2_regularization(
            state_lists,
            weight_decay,
            use_decoupled_weight_decay,
        )

        with DequantizePreconditionersContext(
            preconditioner_list=state_lists[SHAMPOO_PRECONDITIONER_LIST]
        ), (
            DequantizePreconditionersContext(
                preconditioner_list=state_lists[GRAFTING_PRECONDITIONER_LIST]
            )
            if grafting_config_not_none
            else contextlib.nullcontext()
        ):
            # Update Shampoo and grafting preconditioners / factor matrices.
            #   Example for AdaGrad accumulation:
            #   L <- L + G * G^T
            #   R <- R + G^T * G
            #   V <- V + G^2    (element-wise)
            #   (and similar)
            self._update_preconditioners(
                state_lists,
                step,
                grafting_config_not_none,
            )

            # Compute matrix root inverse.
            #   L_inv <- L ** (-1/4)
            #   R_inv <- R ** (-1/4)
            #   (and similar)
            self._compute_root_inverse(state_lists, compute_root_inverse)

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
            # TODO: Resolve preconditioning and grafting PT2 NEX issue and enable them.
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

        # Updates parameters in distributed fashion.
        # If DDP, executes AllGather communication to ensure all parameters are updated after local updates.
        torch._foreach_mul_(masked_blocked_search_directions, -lr)
        state_lists[DISTRIBUTOR].update_params(
            masked_blocked_search_directions=masked_blocked_search_directions
        )

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

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
            DistributedShampoo._mask_state_lists(state_lists, group)

            # Check if gradient list is empty. If so, continue.
            if not state_lists[MASKED_BLOCKED_GRADS]:
                continue

            # Convert the gradient dtype to the computation dtype set in the precision_config if
            # necessary.
            #
            # This conversion is needed because the blocked gradient list has float32 dtype, and we
            # need to convert it to the desired precision before precondition computation.
            if (
                computation_dtype := group[PRECISION_CONFIG].computation_dtype
            ) != state_lists[MASKED_BLOCKED_GRADS][0].dtype:
                state_lists[MASKED_BLOCKED_GRADS] = tuple(
                    tensor.to(dtype=computation_dtype)
                    for tensor in state_lists[MASKED_BLOCKED_GRADS]
                )

            # Iterate group step counter and define Python scalar step.
            step = state_lists[STEP].add_(1)
            # NOTE: Wrap scalar of group[LR] into a 0D tensor to avoid PT2 recompilation;
            # Send 0D tensor to GPU in `non_blocking` to avoid QPS regression. Remove the gpu
            # tensor impl once PT2 supports cpu 0D tensor properly.
            lr = torch.tensor(group[LR], dtype=torch.float).to(
                self._device, non_blocking=True
            )
            beta1 = group[BETAS][0]
            beta3 = group[BETA3]
            weight_decay = group[WEIGHT_DECAY]
            momentum_param = group[MOMENTUM]
            dampening = group[DAMPENING]
            grafting_config_not_none = group[GRAFTING_CONFIG] is not None
            # Check compute root inverse or not for preconditioner
            compute_root_inverse = (
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
                compute_root_inverse,
                use_decoupled_weight_decay,
                use_bias_correction,
                use_grafting_method,
                use_nesterov,
            )

        return loss

    def state_dict(self) -> NoReturn:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard state_dict() method for checkpointing!"
        )

    def load_state_dict(self, state_dict: StateDict) -> NoReturn:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard load_state_dict() method for checkpointing!"
        )

    @staticmethod
    def _construct_param_group_key(
        group: Dict[str, Any], param_to_key: Dict[torch.Tensor, str]
    ) -> str:
        return "/".join(sorted(param_to_key[param] for param in group[PARAMS]))

    def distributed_state_dict(
        self,
        key_to_param: Iterator[Tuple[str, torch.Tensor]],
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
            key_to_param (Iterator[Tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
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
        key_to_param: Iterator[Tuple[str, torch.Tensor]],
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
            key_to_param (Iterator[Tuple[str, Tensor]]): Iterator (like model.named_parameters()) that
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
            param_groups = self.param_groups

            if len(param_groups) != len(param_groups_to_load):
                raise ValueError(
                    f"Different param_groups count: {len(param_groups)} vs {len(param_groups_to_load)}"
                )
            param_to_key = {param: key for key, param in key_to_param_mapping.items()}

            # Loading the parameter group based on the unique parameter group key.
            for group in param_groups:
                param_group_key = DistributedShampoo._construct_param_group_key(
                    group, param_to_key
                )
                if param_group_key not in param_groups_to_load:
                    raise ValueError(
                        f"Param group {param_group_key} not found in param_groups_to_load!"
                    )
                param_group_to_load = param_groups_to_load[param_group_key]
                for key, value in param_group_to_load.items():
                    group[key] = deepcopy(value)
