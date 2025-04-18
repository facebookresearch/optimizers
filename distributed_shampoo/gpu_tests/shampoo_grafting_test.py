"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3

import math
import unittest
from functools import partial
from itertools import product
from typing import Any, Type

import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import (
    AdaGradGraftingConfig,
    AdamGraftingConfig,
    DefaultShampooConfig,
    RMSpropGraftingConfig,
    SGDGraftingConfig,
    ShampooPreconditionerConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_optimizer_on_cpu_and_device,
    compare_two_optimizers_on_weight_and_loss,
)
from matrix_functions_types import DefaultEigendecompositionConfig
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import ParamsT
from torch.optim.rmsprop import RMSprop
from torch.optim.sgd import SGD


class DistributedShampooGraftingTest(unittest.TestCase):
    @staticmethod
    def _optim_factory(
        parameters: ParamsT,
        optim_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        return optim_cls(parameters, **kwargs)

    def test_adagrad_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),)
            + ((torch.device("cuda"),) if torch.cuda.is_available() else ()),
            (
                DefaultShampooConfig,
                ShampooPreconditionerConfig(
                    amortized_computation_config=DefaultEigendecompositionConfig,
                ),
            ),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.01,
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                experimental_optim_factory = partial(
                    optim_factory,
                    optim_cls=DistributedShampoo,
                    betas=(0.0, 1.0),
                    epsilon=1e-10,
                    momentum=0.0,
                    max_preconditioner_dim=10,
                    precondition_frequency=1,
                    start_preconditioning_step=math.inf,
                    use_decoupled_weight_decay=False,
                    grafting_config=AdaGradGraftingConfig(
                        epsilon=1e-10,
                    ),
                    preconditioner_config=preconditioner_config,
                )

                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory, optim_cls=Adagrad, eps=1e-10
                    ),
                    experimental_optim_factory=experimental_optim_factory,
                    device=device,
                )

                compare_optimizer_on_cpu_and_device(
                    optim_factory=experimental_optim_factory,
                    device=device,
                )

    def test_adam_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),)
            + ((torch.device("cuda"),) if torch.cuda.is_available() else ()),
            (
                DefaultShampooConfig,
                ShampooPreconditionerConfig(
                    amortized_computation_config=DefaultEigendecompositionConfig,
                ),
            ),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                experimental_optim_factory = partial(
                    optim_factory,
                    optim_cls=DistributedShampoo,
                    epsilon=1e-8,
                    momentum=0.0,
                    max_preconditioner_dim=10,
                    precondition_frequency=1,
                    start_preconditioning_step=math.inf,
                    use_decoupled_weight_decay=False,
                    grafting_config=AdamGraftingConfig(
                        beta2=0.999,
                        epsilon=1e-8,
                    ),
                    preconditioner_config=preconditioner_config,
                )

                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory, optim_cls=Adam, eps=1e-8
                    ),
                    experimental_optim_factory=experimental_optim_factory,
                    device=device,
                )

                compare_optimizer_on_cpu_and_device(
                    optim_factory=experimental_optim_factory,
                    device=device,
                )

    def test_adamw_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),)
            + ((torch.device("cuda"),) if torch.cuda.is_available() else ()),
            (
                DefaultShampooConfig,
                ShampooPreconditionerConfig(
                    amortized_computation_config=DefaultEigendecompositionConfig,
                ),
            ),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                experimental_optim_factory = partial(
                    optim_factory,
                    optim_cls=DistributedShampoo,
                    epsilon=1e-8,
                    momentum=0.0,
                    max_preconditioner_dim=10,
                    precondition_frequency=1,
                    start_preconditioning_step=math.inf,
                    use_decoupled_weight_decay=True,
                    grafting_config=AdamGraftingConfig(
                        beta2=0.999,
                        epsilon=1e-8,
                    ),
                    preconditioner_config=preconditioner_config,
                )

                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory, optim_cls=AdamW, eps=1e-8
                    ),
                    experimental_optim_factory=experimental_optim_factory,
                    device=device,
                )

                compare_optimizer_on_cpu_and_device(
                    optim_factory=experimental_optim_factory,
                    device=device,
                )

    def test_rmsprop_grafting_on_quadratic(self) -> None:
        # Test with and without weight decay, and with CPU or GPU
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),)
            + ((torch.device("cuda"),) if torch.cuda.is_available() else ()),
            (
                DefaultShampooConfig,
                ShampooPreconditionerConfig(
                    amortized_computation_config=DefaultEigendecompositionConfig,
                ),
            ),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.01,
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                experimental_optim_factory = partial(
                    optim_factory,
                    optim_cls=DistributedShampoo,
                    betas=(0.0, 0.99),
                    epsilon=1e-8,
                    momentum=0.0,
                    max_preconditioner_dim=10,
                    precondition_frequency=1,
                    start_preconditioning_step=math.inf,
                    use_bias_correction=False,
                    use_decoupled_weight_decay=False,
                    grafting_config=RMSpropGraftingConfig(
                        beta2=0.99,
                        epsilon=1e-8,
                    ),
                    preconditioner_config=preconditioner_config,
                )

                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory,
                        optim_cls=RMSprop,
                        alpha=0.99,
                        eps=1e-8,
                    ),
                    experimental_optim_factory=experimental_optim_factory,
                    device=device,
                )

                compare_optimizer_on_cpu_and_device(
                    optim_factory=experimental_optim_factory, device=device
                )

    def test_sgd_grafting_on_quadratic(self) -> None:
        # Test all the combinations of with and without weight decay, with and without nesterov, and with CPU or GPU.
        for weight_decay, use_nesterov, device, preconditioner_config in product(
            (0.0, 0.3),
            (True, False),
            (torch.device("cpu"),)
            + ((torch.device("cuda"),) if torch.cuda.is_available() else ()),
            (
                DefaultShampooConfig,
                ShampooPreconditionerConfig(
                    amortized_computation_config=DefaultEigendecompositionConfig,
                ),
            ),
        ):
            optim_factory = partial(
                DistributedShampooGraftingTest._optim_factory,
                lr=0.1,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                use_nesterov=use_nesterov,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                experimental_optim_factory = partial(
                    optim_factory,
                    optim_cls=DistributedShampoo,
                    betas=(0.0, 0.9),
                    epsilon=1e-10,
                    max_preconditioner_dim=10,
                    precondition_frequency=1,
                    start_preconditioning_step=math.inf,
                    use_nesterov=use_nesterov,
                    use_decoupled_weight_decay=False,
                    grafting_config=SGDGraftingConfig(),  # type: ignore[abstract]
                    preconditioner_config=preconditioner_config,
                )

                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory,
                        optim_cls=SGD,
                        nesterov=use_nesterov,
                    ),
                    experimental_optim_factory=experimental_optim_factory,
                    device=device,
                )

                compare_optimizer_on_cpu_and_device(
                    optim_factory=experimental_optim_factory, device=device
                )
