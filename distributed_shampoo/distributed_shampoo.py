"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import logging
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch

from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    BETAS,
    DDPShampooConfig,
    DistributedConfig,
    DISTRIBUTOR,
    EPSILON,
    EXPONENT_MULTIPLIER,
    FILTERED_GRAD,
    FILTERED_GRAD_LIST,
    FSDPShampooConfig,
    GRAFTING_CONFIG,
    GRAFTING_PRECONDITIONER_LIST,
    GraftingConfig,
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
    PRECONDITION_FREQUENCY,
    PRECONDITIONER_DTYPE,
    PREVIOUS_GRAD_SELECTOR,
    RMSpropGraftingConfig,
    SGDGraftingConfig,
    SHAMPOO_PRECONDITIONER_LIST,
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
from distributed_shampoo.utils.shampoo_ddp_distributor import (
    DDPDistributor,
)
from distributed_shampoo.utils.shampoo_distributor import Distributor
from distributed_shampoo.utils.shampoo_fsdp_distributor import (
    FSDPDistributor,
)

from distributed_shampoo.utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    SGDPreconditionerList,
    ShampooPreconditionerList,
)
from distributed_shampoo.utils.shampoo_utils import compress_list
from torch.optim.optimizer import ParamsT

logger: logging.Logger = logging.getLogger(__name__)


class DistributedShampoo(torch.optim.Optimizer):
    """Implements distributed Shampoo algorithm.

    Developers:
        Hao-Jun Michael Shi (Meta Platforms, Inc.)
        Tsung-Hsien Lee
        Anna Cai (Meta Platforms, Inc.)
        Shintaro Iwasaki (Meta Platforms, Inc.)
        Ke Sang (Meta Platforms, Inc.)
        Wang Zhou (Meta Platforms, Inc.)

    with contributions and support from:

    Rohan Anil (Google), Adnan Aziz (Meta), Pavan Balaji (Meta), Shuo Chang (Meta), Weiwei Chu (Meta), Assaf Eisenman (Meta),
    Will Feng (Meta), Zhuobo Feng (Meta), Jose Gallego-Posada (Mila / Meta Platforms, Inc.), Avirup Ghosh (Meta), Yizi Gu (Meta),
    Vineet Gupta (Google), Yuchen Hao (Meta), Brian Hirsh (Meta), Yusuo Hu (Meta), Yuxi Hu (Meta), Minhui Huang (Meta),
    Guna Lakshminarayanan (Meta), Michael Lazos (Meta), Zhijing Li (Meta), Ming Liang (Meta), Wanchao Liang (Meta), Ying Liu
    (Meta), Wenguang Mao (Meta), Dheevatsa Mudigere (NVIDIA), Maxim Naumov (Meta), Jongsoo Park (Meta), Mike Rabbat (Meta),
    Kaushik Rangadurai (Meta), Dennis van der Staay (Meta), Fei Tian (Meta), Sanjay Vishwakarma (Meta), Xunnan (Shawn) Xu (Meta),
    Jiyan Yang (Meta), Chunxing Yin (Meta), and Iris Zhang (Meta).

    Details in: https://arxiv.org/pdf/2309.06497.pdf.

    Partly based on the work in:
    - https://arxiv.org/pdf/1802.09568.pdf
    - https://arxiv.org/pdf/2002.09018.pdf

    ------------
    Requirements
    ------------

    1. PyTorch >= 2.0
    2. Python >= 3.8
    3. CUDA 11.3, 11.4, 12.2+

    In order to support checkpointing, one must use torch.distributed.checkpoint and pass the named parameters into state_dict.
    Note that the standard checkpointing solution by PyTorch is not supported!

    Note: We have observed known instabilities with the torch.linalg.eigh operator on CUDA 11.6-12.1, specifically for low-rank
    matrices, which may appear with using a small start_preconditioning_step. Please avoid these versions of CUDA if possible.
    See: https://github.com/pytorch/pytorch/issues/94772.

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

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): Learning rate. (Default: 1e-2)
        betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            (Default: (0.9, 1.0))
        epsilon (float): Term added to the denominator to improve numerical stability. (Default: 1e-12)
        momentum (float): Momentum parameter. (default: 0.)
        weight_decay (float): Weight decay (L2 penalty). (Default: 0.)
        max_preconditioner_dim (int): Maximum preconditioner dimensio. (Default: 1024)
        precondition_frequency (int): Frequency for computing root inverse preconditioner. (Default: 1)
        start_preconditioning_step (int): Iteration to start computing inverse preconditioner. If -1, uses
            the same value as precondition_frequency. (Default: -1)
        inv_root_override (int, Sequence[int]): Inverse root to use in Shampoo. If a list [l1, l2, ..., lp], then we will
            use -1 / l1 for 1-D tensor (vectors), -1 / l2 for 2-D tensors (matrices), and so on. If the order of the
            tensor exceeds the order of the tensor, reverts to the default value. If 0 is used, uses the default inverse
            root -1 / (2 * o), where o is the order of the tensor. (Default: 0)
        exponent_multiplier (float): Number to be multiplied to the numerator of the inverse root, i.e., eta where the
            exponent is -eta / (2 * p). (Default: 1.0)
        use_nesterov (bool): Flag for using Nesterov momentum. (default: False)
        use_bias_correction (bool): Flag for using bias correction. (Default: True)
        use_decoupled_weight_decay (bool): Flag for using AdamW-style decoupled weight decay. (Default: True)
        grafting_config (Optional[GraftingConfig]): Configuration for grafting method. If None, ignores grafting.
            (Default: None)
        use_merge_dims (bool): Merge dimensions if possible while respecting max_preconditioner_dim. (Default: True)
        use_pytorch_compile (bool): Use PyTorch 2.0 compiler feature to speed up training. (Default: False)
        distributed_config (Optional[DistributedConfig]): Configuration for applying Shampoo
            to different distributed training frameworks, such as distributed-data parallel (DDP) training.
            Based on the configuration, determines which version of Shampoo to use. (Default: None)
        preconditioner_dtype (torch.dtype): Data type for preconditioner. (Default: torch.float)
        use_protected_eigh (bool): Flag for using two guards to prevent failures of torch.linalg.eigh. (Default: True)
            1. Attempts to compute root inverse in preconditioner_dtype precision.
            2. Attempts to recompute the eigendecomposition in higher precision if using lower-precision fails.
            3. Otherwise, re-uses previous inverse factor matrix when both root inverse computations fail.
        track_root_inv_residuals (bool): Track errors and residuals of root inverse. For debugging purposes.
            (Default: False)

    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 1.0),
        epsilon: float = 1e-12,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        max_preconditioner_dim: int = 1024,
        precondition_frequency: int = 1,
        start_preconditioning_step: int = -1,
        inv_root_override: Union[int, Sequence[int]] = 0,
        exponent_multiplier: float = 1.0,
        use_nesterov: bool = False,
        use_bias_correction: bool = True,
        use_decoupled_weight_decay: bool = True,
        grafting_config: Optional[GraftingConfig] = None,
        use_merge_dims: bool = True,
        use_pytorch_compile: bool = False,
        distributed_config: Optional[DistributedConfig] = None,
        preconditioner_dtype: torch.dtype = torch.float,
        use_protected_eigh: bool = True,
        track_root_inv_residuals: bool = False,
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
        if not epsilon > 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}. Must be > 0.0.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(
                f"Invalid momentum parameter: {momentum}. Must be [0.0, 1.0)."
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

        # Provide error for system Pytorch compile availability
        if use_pytorch_compile and not torch.cuda.is_available():
            raise ValueError(
                "Backend does NOT support Pytorch 2.0 compile. Switch to use_pytorch_compile=False."
            )

        super().__init__(
            params,
            {
                LR: lr,
                BETAS: betas,
                EPSILON: epsilon,
                MOMENTUM: momentum,
                WEIGHT_DECAY: weight_decay,
                MAX_PRECONDITIONER_DIM: max_preconditioner_dim,
                PRECONDITION_FREQUENCY: precondition_frequency,
                START_PRECONDITIONING_STEP: start_preconditioning_step,
                INV_ROOT_OVERRIDE: inv_root_override,
                EXPONENT_MULTIPLIER: exponent_multiplier,
                USE_NESTEROV: use_nesterov,
                USE_BIAS_CORRECTION: use_bias_correction,
                USE_DECOUPLED_WEIGHT_DECAY: use_decoupled_weight_decay,
                GRAFTING_CONFIG: grafting_config,
                USE_MERGE_DIMS: use_merge_dims,
                PRECONDITIONER_DTYPE: preconditioner_dtype,
            },
        )

        # Initialize non-group-related fields.
        self._distributed_config = distributed_config
        self._use_protected_eigh = use_protected_eigh
        self._track_root_inv_residuals = track_root_inv_residuals
        self._use_pytorch_compile = use_pytorch_compile

        # Initialize dictionary containing lists of .
        self._per_group_state_lists: List[Dict[str, Any]] = [
            {} for _ in self.param_groups
        ]

        # Block parameters and instantiate optimizer states.
        self._instantiate_distributor()
        self._instantiate_shampoo_preconditioner_list()
        self._instantiate_grafting()
        self._instantiate_steps()
        self._instantiate_momentum()
        self._instantiate_filtered_grads()
        self._instantiate_device()

        # Use PT2 to compile the step function for each parameter group
        self._per_group_step: Callable[
            [
                Dict[str, Any],
                torch.Tensor,
                torch.Tensor,
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
            torch.compile(self._per_group_step_impl, backend="inductor")
            if self._use_pytorch_compile
            else self._per_group_step_impl
        )

    @torch.no_grad()
    def _instantiate_distributor(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            # Instantiate distributors for each group.
            if self._distributed_config is None:
                state_lists[DISTRIBUTOR] = Distributor(
                    param_group=group,
                )
            elif isinstance(self._distributed_config, DDPShampooConfig):
                state_lists[DISTRIBUTOR] = DDPDistributor(
                    param_group=group,
                    distributed_config=self._distributed_config,
                )
            elif isinstance(self._distributed_config, FSDPShampooConfig):
                state_lists[DISTRIBUTOR] = FSDPDistributor(
                    param_group=group,
                    distributed_config=self._distributed_config,
                )
            else:
                raise NotImplementedError(f"{self._distributed_config=} not supported!")

            # Compile blocked parameters and block-to-parameter metadata into group lists.
            state_lists[MASKED_BLOCKED_PARAMS] = state_lists[
                DISTRIBUTOR
            ].local_blocked_params
            # First PREVIOUS_GRAD_SELECTOR is set to None.
            state_lists[PREVIOUS_GRAD_SELECTOR] = None

    @torch.no_grad()
    def _instantiate_shampoo_preconditioner_list(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            state_lists[SHAMPOO_PRECONDITIONER_LIST] = ShampooPreconditionerList(
                block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                state=self.state,
                block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                beta2=group[BETAS][1],
                epsilon=group[EPSILON],
                inv_root_override=group[INV_ROOT_OVERRIDE],
                exponent_multiplier=group[EXPONENT_MULTIPLIER],
                use_bias_correction=group[USE_BIAS_CORRECTION],
                factor_matrix_dtype=group[PRECONDITIONER_DTYPE],
                use_protected_eigh=self._use_protected_eigh,
            )

    @torch.no_grad()
    def _instantiate_grafting(self) -> None:
        for state_lists, group in zip(
            self._per_group_state_lists, self.param_groups, strict=True
        ):
            if group[GRAFTING_CONFIG] is None:
                state_lists[GRAFTING_PRECONDITIONER_LIST] = None
            elif isinstance(group[GRAFTING_CONFIG], SGDGraftingConfig):
                state_lists[GRAFTING_PRECONDITIONER_LIST] = SGDPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                )
            elif isinstance(
                group[GRAFTING_CONFIG],
                (AdaGradGraftingConfig, RMSpropGraftingConfig, AdamGraftingConfig),
            ):
                state_lists[GRAFTING_PRECONDITIONER_LIST] = AdagradPreconditionerList(
                    block_list=state_lists[DISTRIBUTOR].global_blocked_params,
                    state=self.state,
                    block_info_list=state_lists[DISTRIBUTOR].global_block_info_list,
                    distributor_selector=state_lists[DISTRIBUTOR].distributor_selector,
                    beta2=1.0
                    if isinstance(group[GRAFTING_CONFIG], AdaGradGraftingConfig)
                    else group[GRAFTING_CONFIG].beta2,
                    epsilon=group[GRAFTING_CONFIG].epsilon,
                    use_bias_correction=isinstance(
                        group[GRAFTING_CONFIG], AdamGraftingConfig
                    ),
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

                block_state[MOMENTUM] = block_info.allocate_zeros_tensor(
                    shape=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                global_momentum_list.append(
                    block_info.get_tensor(block_state[MOMENTUM])
                )

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[MOMENTUM_LIST] = compress_list(
                global_momentum_list,
                state_lists[DISTRIBUTOR].distributor_selector,
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

                block_state[FILTERED_GRAD] = block_info.allocate_zeros_tensor(
                    shape=block.size(),
                    dtype=block.dtype,
                    device=block.device,
                )
                global_filtered_grad_list.append(
                    block_info.get_tensor(block_state[FILTERED_GRAD])
                )

            # We compress the momentum list to only the locally-owned parameter states.
            state_lists[FILTERED_GRAD_LIST] = compress_list(
                global_filtered_grad_list,
                state_lists[DISTRIBUTOR].distributor_selector,
            )
            # Here, we set masked filtered grad list to filtered grad list because we assume
            # all parameters are active.
            state_lists[MASKED_FILTERED_GRAD_LIST] = state_lists[FILTERED_GRAD_LIST]

    @torch.no_grad()
    def _instantiate_device(self) -> None:
        # NOTE: Assume all parameter groups consistently exist on the same rank
        self._device = self._per_group_state_lists[0][MASKED_BLOCKED_PARAMS][0].device

    @staticmethod
    @torch.no_grad()
    def _mask_state_lists(state_lists: Dict[str, Any], group: Dict[str, Any]) -> None:
        if (
            state_lists[DISTRIBUTOR].local_grad_selector
            == state_lists[PREVIOUS_GRAD_SELECTOR]
        ):
            return

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

            # Get expected relative errors/residuals for debugging purposes
            if group[PRECONDITIONER_DTYPE] == torch.float64:
                expected_relative_error = 1e-7
            elif group[PRECONDITIONER_DTYPE] == torch.float32:
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
    def _per_group_step_impl(
        self,
        state_lists: Dict[str, Any],
        step: torch.Tensor,
        lr: torch.Tensor,
        beta1: float,
        weight_decay: float,
        momentum_param: float,
        grafting_config_not_none: bool,
        compute_root_inverse: bool,
        use_decoupled_weight_decay: bool,
        use_bias_correction: bool,
        use_grafting_method: bool,
        use_nesterov: bool,
    ) -> None:
        # Incorporate L2-regularization or decoupled weight decay.
        if weight_decay != 0.0 and not use_decoupled_weight_decay:
            torch._foreach_add_(
                state_lists[MASKED_BLOCKED_GRADS],
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

        # Update Shampoo and grafting preconditioners.
        state_lists[SHAMPOO_PRECONDITIONER_LIST].update_preconditioners(
            masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
            step=step,
        )
        if grafting_config_not_none:
            state_lists[GRAFTING_PRECONDITIONER_LIST].update_preconditioners(
                masked_grad_list=state_lists[MASKED_BLOCKED_GRADS],
                step=step,
            )

        # Compute matrix root inverse.
        self._compute_root_inverse(state_lists, compute_root_inverse)

        # Compute filtered gradient or EMA of the gradients.
        if beta1 != 0.0:
            torch._foreach_mul_(state_lists[MASKED_FILTERED_GRAD_LIST], beta1)
            torch._foreach_add_(
                state_lists[MASKED_FILTERED_GRAD_LIST],
                state_lists[MASKED_BLOCKED_GRADS],
                alpha=1 - beta1,
            )
            if use_bias_correction:
                bias_correction1 = 1.0 - beta1**step
                masked_filtered_grad_list = torch._foreach_div(
                    state_lists[MASKED_FILTERED_GRAD_LIST], bias_correction1
                )
            else:
                masked_filtered_grad_list = state_lists[MASKED_FILTERED_GRAD_LIST]
        else:
            masked_filtered_grad_list = state_lists[MASKED_BLOCKED_GRADS]

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

        # Incorporate decoupled weight decay.
        if weight_decay != 0.0 and use_decoupled_weight_decay:
            torch._foreach_add_(
                masked_blocked_search_directions,
                state_lists[MASKED_BLOCKED_PARAMS],
                alpha=weight_decay,
            )

        # Update momentum.
        if momentum_param != 0.0:
            torch._foreach_mul_(state_lists[MASKED_MOMENTUM_LIST], momentum_param)
            torch._foreach_add_(
                state_lists[MASKED_MOMENTUM_LIST],
                masked_blocked_search_directions,
            )

            # Incorporates Nesterov momentum.
            if use_nesterov:
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

            # Iterate group step counter and define Python scalar step.
            step = state_lists[STEP].add_(1)
            # NOTE: Wrap scalar of group[LR] into a 0D tensor to avoid PT2 recompilation;
            # Send 0D tensor to GPU in `non_blocking` to avoid QPS regression. Remove the gpu
            # tensor impl once PT2 supports cpu 0D tensor properly.
            lr = torch.tensor(group[LR], dtype=torch.float).to(
                self._device, non_blocking=True
            )
            beta1 = group[BETAS][0]
            weight_decay = group[WEIGHT_DECAY]
            momentum_param = group[MOMENTUM]
            grafting_config_not_none = group[GRAFTING_CONFIG] is not None
            # Check compute root inverse or not for preconditioner
            compute_root_inverse = (
                step % group[PRECONDITION_FREQUENCY] == 0
                and step > group[START_PRECONDITIONING_STEP]
                or step == group[START_PRECONDITIONING_STEP]
            )
            use_decoupled_weight_decay = group[USE_DECOUPLED_WEIGHT_DECAY]
            use_bias_correction = group[USE_BIAS_CORRECTION]
            # Check applying grafting method or not
            use_grafting_method = (
                step < group[START_PRECONDITIONING_STEP] and grafting_config_not_none
            )
            use_nesterov = group[USE_NESTEROV]

            self._per_group_step(
                state_lists,
                step,
                lr,
                beta1,
                weight_decay,
                momentum_param,
                grafting_config_not_none,
                compute_root_inverse,
                use_decoupled_weight_decay,
                use_bias_correction,
                use_grafting_method,
                use_nesterov,
            )

        return loss

    def state_dict(self) -> None:
        raise NotImplementedError(
            "Distributed Shampoo does not support the standard state_dict() method for checkpointing!"
        )

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
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
    ) -> Dict[str, Any]:
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
            state_dict (Dict[str, Any]): Dictionary containing the optimizer state and potentially parameter
                groups.

        """

        # Create mapping from parameter to its name. Generate flattened state dictionary for state.
        param_to_key = {param: key for key, param in key_to_param}
        ret: Dict[str, Any] = {
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
            self._construct_param_group_key(group, param_to_key): {
                k: deepcopy(v) for k, v in group.items() if k != PARAMS
            }
            for group in self.param_groups
        }

        return ret

    def load_distributed_state_dict(
        self,
        state_dict: Mapping[str, Any],
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
            state_dict (Dict[str, Any]): State dictionary to load containing the optimizer state and
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
                param_group_key = self._construct_param_group_key(group, param_to_key)
                if param_group_key not in param_groups_to_load:
                    raise ValueError(
                        f"Param group {param_group_key} not found in param_groups_to_load!"
                    )
                param_group_to_load = param_groups_to_load[param_group_key]
                for key, value in param_group_to_load.items():
                    group[key] = deepcopy(value)
