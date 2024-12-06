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
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultSOAPConfig,
)
from distributed_shampoo.tests.shampoo_test_utils import (
    compare_two_optimizers_on_weight_and_loss,
)
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.optimizer import ParamsT
from torch.optim.rmsprop import RMSprop


# Note: We have to set the epsilon to a very small value (i.e., 1e-15) due to the
# the place epsilon is added in the PyTorch optimizers (i.e., AdaGrad, RMSProp, Adam, AdamW)
# and Distributed Shampoo.
# The PyTorch optimizers add epsilon outside of the square root, and Distributed Shampoo
# adds epsilon inside of the square root.


class DistributedShampooEigenvalueCorrectionTest(unittest.TestCase):
    @staticmethod
    def _optim_factory(
        parameters: ParamsT,
        optim_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        return optim_cls(parameters, **kwargs)

    def test_adagrad_eigenvalue_correction_on_quadratic(self) -> None:
        # Test with and without weight decay, with CPU or GPU, and using eigendecomposition or QR algorithm.
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
            (DefaultEigenvalueCorrectedShampooConfig, DefaultSOAPConfig),
        ):
            optim_factory = partial(
                DistributedShampooEigenvalueCorrectionTest._optim_factory,
                lr=0.01,
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory, optim_cls=Adagrad, eps=1e-15
                    ),
                    experimental_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        betas=(0.0, 1.0),
                        epsilon=1e-15,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=math.inf,
                        use_decoupled_weight_decay=False,
                        grafting_config=None,
                        preconditioner_config=preconditioner_config,
                    ),
                    device=device,
                )

    def test_adam_eigenvalue_correction_on_quadratic(self) -> None:
        # Test with and without weight decay, with CPU or GPU, and using eigendecomposition or QR algorithm.
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
            (DefaultEigenvalueCorrectedShampooConfig, DefaultSOAPConfig),
        ):
            optim_factory = partial(
                DistributedShampooEigenvalueCorrectionTest._optim_factory,
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory,
                        optim_cls=Adam,
                        eps=1e-15,
                    ),
                    experimental_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        epsilon=1e-15,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=math.inf,
                        use_decoupled_weight_decay=False,
                        grafting_config=None,
                        preconditioner_config=preconditioner_config,
                    ),
                    device=device,
                )

    def test_adamw_eigenvalue_correction_on_quadratic(self) -> None:
        # Test with and without weight decay, with CPU or GPU, and using eigendecomposition or QR algorithm.
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
            (DefaultEigenvalueCorrectedShampooConfig, DefaultSOAPConfig),
        ):
            optim_factory = partial(
                DistributedShampooEigenvalueCorrectionTest._optim_factory,
                lr=0.001,
                betas=(0.9, 0.999),
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory,
                        optim_cls=AdamW,
                        eps=1e-15,
                    ),
                    experimental_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        epsilon=1e-15,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=math.inf,
                        use_decoupled_weight_decay=True,
                        grafting_config=None,
                        preconditioner_config=preconditioner_config,
                    ),
                    device=device,
                )

    def test_rmsprop_eigenvalue_correction_on_quadratic(self) -> None:
        # Test with and without weight decay, with CPU or GPU, and using eigendecomposition or QR algorithm.
        for weight_decay, device, preconditioner_config in product(
            (0.0, 0.3),
            (torch.device("cpu"),) + (torch.device("cuda"),)
            if torch.cuda.is_available()
            else (),
            (DefaultEigenvalueCorrectedShampooConfig, DefaultSOAPConfig),
        ):
            optim_factory = partial(
                DistributedShampooEigenvalueCorrectionTest._optim_factory,
                lr=0.01,
                weight_decay=weight_decay,
            )
            with self.subTest(
                weight_decay=weight_decay,
                device=device,
                preconditioner_config=preconditioner_config,
            ):
                compare_two_optimizers_on_weight_and_loss(
                    control_optim_factory=partial(
                        optim_factory,
                        optim_cls=RMSprop,
                        alpha=0.99,
                        eps=1e-15,
                    ),
                    experimental_optim_factory=partial(
                        optim_factory,
                        optim_cls=DistributedShampoo,
                        betas=(0.0, 0.99),
                        epsilon=1e-15,
                        momentum=0.0,
                        max_preconditioner_dim=10,
                        precondition_frequency=1,
                        start_preconditioning_step=math.inf,
                        use_decoupled_weight_decay=False,
                        grafting_config=None,
                        use_bias_correction=False,
                        preconditioner_config=preconditioner_config,
                    ),
                    device=device,
                )
