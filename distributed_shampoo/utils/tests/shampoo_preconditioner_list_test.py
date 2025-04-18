"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import abc
import math
import re
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any
from unittest import mock

import torch

from commons import AbstractDataclass
from distributed_shampoo.shampoo_types import (
    AdaptiveAmortizedComputationFrequencyConfig,
    DefaultShampooConfig,
    DefaultSOAPConfig,
    EigenvalueCorrectedShampooPreconditionerConfig,
    PreconditionerConfig,
    PreconditionerValueError,
    ShampooPreconditionerConfig,
)
from distributed_shampoo.utils import shampoo_preconditioner_list
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    BaseShampooPreconditionerList,
    EigendecomposedShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    PreconditionerList,
    SGDPreconditionerList,
    ShampooPreconditionerList,
)
from distributed_shampoo.utils.shampoo_utils import compress_list
from matrix_functions_types import QREigendecompositionConfig
from torch import Tensor


class PreconditionerListTest(unittest.TestCase):
    """PreconditionerListTest is the base class for testing all Preconditioner implementations.

    Note that all the subclasses may not implement setUp() and only need to implement _instantiate_block_list() and
    _instantiate_preconditioner_list() to enable the the usages of self._block_list and self._preconditioner_list.

    Subclasses could override the following test cases for their specific needs:
        1. test_update_preconditioners_and_precondition()
            - consider using _test_update_preconditioners_and_precondition() as a helper function to test this.
        2. test_numel_list()
        3. test_dims_list()
        4. test_num_bytes_list()
        5. test_numel()
        6. test_num_bytes()
        7. test_compress_preconditioner_list()
    """

    def setUp(self) -> None:
        self._block_list = self._instantiate_block_list()
        self._preconditioner_list = self._instantiate_preconditioner_list()

    def _instantiate_block_list(self) -> tuple[Tensor, ...]:
        return (
            torch.tensor([1.0, 2.0]),
            torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        )

    def _instantiate_preconditioner_list(self, **kwargs: Any) -> PreconditionerList:
        # Disable the abstract methods check from the interface so it is possible to instantiate PreconditionerList.
        PreconditionerList.__abstractmethods__ = frozenset()
        return PreconditionerList(block_list=self._block_list)  # type: ignore[abstract]

    def _test_update_preconditioners_and_precondition(
        self,
        preconditioner_list: PreconditionerList,
        masked_grad_lists: list[tuple[Tensor, ...]],
        masked_expected_preconditioned_grad_list: tuple[Tensor, ...] | None,
    ) -> None:
        for step, masked_grad_list in enumerate(masked_grad_lists, start=1):
            preconditioner_list.update_preconditioners(
                masked_grad_list=masked_grad_list,
                step=torch.tensor(step),
                # Only update the complete preconditioner during the last call to update_preconditioners().
                perform_amortized_computation=isinstance(
                    preconditioner_list, BaseShampooPreconditionerList
                )
                and step == len(masked_grad_lists),
            )
        masked_preconditioned_grad_list = preconditioner_list.precondition(
            masked_grad_list=masked_grad_lists[-1]
        )
        if masked_expected_preconditioned_grad_list is not None:
            torch.testing.assert_close(
                masked_preconditioned_grad_list,
                masked_expected_preconditioned_grad_list,
            )
        else:
            self.assertIsNone(masked_preconditioned_grad_list)

    def test_update_preconditioners_and_precondition(self) -> None:
        masked_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._preconditioner_list,
            masked_grad_lists=[masked_grad_list],
            masked_expected_preconditioned_grad_list=None,
        )

    def test_numel_list(self) -> None:
        self.assertEqual(self._preconditioner_list.numel_list, (0, 0))

    def test_dims_list(self) -> None:
        self.assertEqual(
            self._preconditioner_list.dims_list, (torch.Size([2]), torch.Size([2, 2]))
        )

    def test_num_bytes_list(self) -> None:
        self.assertEqual(self._preconditioner_list.num_bytes_list, (0, 0))

    def test_numel(self) -> None:
        self.assertEqual(self._preconditioner_list.numel(), 0)

    def test_num_bytes(self) -> None:
        self.assertEqual(self._preconditioner_list.num_bytes(), 0)

    def _test_compress_preconditioner_list(
        self,
        expected_compress_list_call_count: int,
    ) -> None:
        with mock.patch.object(
            shampoo_preconditioner_list,
            "compress_list",
        ) as mock_compress_list:
            self.assertIsNone(
                self._preconditioner_list.compress_preconditioner_list(
                    local_grad_selector=(True,) * len(self._block_list)
                )
            )
            self.assertEqual(
                mock_compress_list.call_count,
                expected_compress_list_call_count,
            )

    def test_compress_preconditioner_list(self) -> None:
        self._test_compress_preconditioner_list(expected_compress_list_call_count=0)


class SGDPreconditionerListTest(PreconditionerListTest):
    def _instantiate_preconditioner_list(self, **kwargs: Any) -> SGDPreconditionerList:
        return SGDPreconditionerList(block_list=self._block_list, **kwargs)

    def test_update_preconditioners_and_precondition(self) -> None:
        masked_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(),
            masked_grad_lists=[masked_grad_list],
            masked_expected_preconditioned_grad_list=masked_grad_list,
        )

    def test_compress_preconditioner_list(self) -> None:
        self._test_compress_preconditioner_list(expected_compress_list_call_count=0)


class AdagradPreconditionerListTest(PreconditionerListTest):
    def _instantiate_block_list(self) -> tuple[Tensor, ...]:
        # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
        # and self._params[1] are split into two blocks.
        return (
            self._params[0],
            *torch.split(self._params[1], 2, dim=0),
        )

    def _instantiate_preconditioner_list(
        self, **kwargs: Any
    ) -> AdagradPreconditionerList:  # type: ignore[override]
        kwargs = {"beta2": 1.0, "epsilon": 0.0, "use_bias_correction": True} | kwargs
        return AdagradPreconditionerList(
            block_list=self._block_list,
            state=self._state,
            block_info_list=self._block_info_list,
            **kwargs,
        )

    def setUp(self) -> None:
        self._params = (
            torch.tensor([1.0, 2.0]),
            torch.arange(6, dtype=torch.float).reshape(3, 2),
        )
        self._state = {  # type: ignore[var-annotated]
            self._params[0]: {},
            self._params[1]: {},
        }
        # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
        # and self._params[1] are split into two blocks.
        self._block_info_list = (
            BlockInfo(
                param=self._params[0],
                composable_block_ids=(0, "block_0"),
            ),
            BlockInfo(
                param=self._params[1],
                composable_block_ids=(1, "block_0"),
            ),
            BlockInfo(
                param=self._params[1],
                composable_block_ids=(1, "block_1"),
            ),
        )
        super().setUp()

    def test_update_preconditioners_and_precondition(self) -> None:
        grad_list = (
            torch.tensor([1.0, 1.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        )

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(beta2=1.0),
            masked_grad_lists=[grad_list],
            masked_expected_preconditioned_grad_list=torch._foreach_sign(grad_list),
        )
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(beta2=0.9),
            masked_grad_lists=[grad_list],
            masked_expected_preconditioned_grad_list=torch._foreach_sign(grad_list),
        )
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=0.99,
                use_bias_correction=False,
            ),
            masked_grad_lists=[grad_list],
            masked_expected_preconditioned_grad_list=torch._foreach_mul(
                [torch.tensor(10.0), torch.tensor(10.0), torch.tensor(10.0)],
                torch._foreach_sign(grad_list),
            ),
        )

    def test_numel_list(self) -> None:
        self.assertEqual(self._preconditioner_list.numel_list, (2, 4, 2))

    def test_dims_list(self) -> None:
        self.assertEqual(
            self._preconditioner_list.dims_list,
            (torch.Size([2]), torch.Size([2, 2]), torch.Size([1, 2])),
        )

    def test_num_bytes_list(self) -> None:
        self.assertEqual(self._preconditioner_list.num_bytes_list, (8, 16, 8))

    def test_numel(self) -> None:
        self.assertEqual(self._preconditioner_list.numel(), 8)

    def test_num_bytes(self) -> None:
        self.assertEqual(self._preconditioner_list.num_bytes(), 32)

    def test_compress_preconditioner_list(self) -> None:
        self._test_compress_preconditioner_list(expected_compress_list_call_count=1)


class BaseShampooPreconditionerListTest(unittest.TestCase):
    def test_abstract_methods(self) -> None:
        # Basic setup for instantiating BaseShampooPreconditionerList.
        param = torch.tensor([1.0, 2.0])

        # Disable the abstract methods check from the interface so it is possible to instantiate BaseShampooPreconditionerList.
        BaseShampooPreconditionerList.__abstractmethods__ = frozenset()

        with (
            mock.patch.object(
                # Mock compress_list() to enable the instantiation of BaseShampooPreconditionerList.
                shampoo_preconditioner_list,
                "compress_list",
                return_value=(True,) * max(param.dim(), 1),
            ) as mock_compress_list,
            mock.patch.object(
                # Mock _update_factor_matrices() otherwise the access of factor_matrices will throw errors.
                BaseShampooPreconditionerList,
                "_update_factor_matrices",
            ) as mock_update_factor_matrices,
        ):
            # Test the abstract methods _create_preconditioned_dims_selector(), _create_kronecker_factors_state_for_block(), _create_kronecker_factors_list(), and _get_inverse_roots_from_override().
            preconditioner_list = BaseShampooPreconditionerList(  # type: ignore
                block_list=(param,),
                state={param: {}},
                block_info_list=(
                    BlockInfo(
                        param=param,
                        composable_block_ids=(0, "block_0"),
                    ),
                ),
                preconditioner_config=DefaultShampooConfig,
                beta2=1.0,
            )

            # Test the abstract_method _amortized_computation().
            preconditioner_list.update_preconditioners(
                masked_grad_list=(torch.tensor([1.0, 1.0]),),
                step=torch.tensor(1),
                perform_amortized_computation=True,
            )

            mock_compress_list.assert_called_once()
            mock_update_factor_matrices.assert_called_once()


@dataclass(init=False)
class AmortizedComputationProperties(AbstractDataclass):
    """Dataclass for properties of amortized computation functions."""

    amortized_computation_function_name: str = field(init=False)
    invalid_amortized_computation_return_values: (
        tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
    ) = field(init=False)
    valid_amortized_computation_return_value: Tensor | tuple[Tensor, Tensor] = field(
        init=False
    )


@dataclass
class InverseRootProperties(AmortizedComputationProperties):
    """Dataclass for properties of matrix_inverse_root function."""

    amortized_computation_function_name: str = "matrix_inverse_root"
    invalid_amortized_computation_return_values: tuple[Tensor, Tensor] = (
        torch.tensor([torch.nan]),
        torch.tensor([torch.inf]),
    )
    valid_amortized_computation_return_value: Tensor = torch.tensor([1.0])


@dataclass
class EigendecompositionProperties(AmortizedComputationProperties):
    """Dataclass for properties of matrix_eigendecomposition function."""

    amortized_computation_function_name: str = "matrix_eigendecomposition"
    invalid_amortized_computation_return_values: tuple[
        tuple[Tensor, Tensor], tuple[Tensor, Tensor]
    ] = (
        (torch.tensor([torch.nan]), torch.tensor([torch.nan])),
        (torch.tensor([torch.inf]), torch.tensor([torch.inf])),
    )
    valid_amortized_computation_return_value: tuple[Tensor, Tensor] = (
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )


# Use outer class as wrapper to avoid running the abstract test.
class AbstractTest:
    class BaseShampooPreconditionerListTest(abc.ABC, AdagradPreconditionerListTest):
        @property
        @abc.abstractmethod
        def _amortized_computation_properties(
            self,
        ) -> AmortizedComputationProperties: ...

        @property
        @abc.abstractmethod
        def _default_preconditioner_config(self) -> PreconditionerConfig: ...

        @property
        @abc.abstractmethod
        def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]: ...

        def _instantiate_preconditioner_list(self, **kwargs: Any) -> PreconditionerList:  # type: ignore[override]
            kwargs = {
                "beta2": 1.0,
                "epsilon": 1e-12,
                "use_bias_correction": True,
                "preconditioner_config": self._default_preconditioner_config,
            } | kwargs
            return self._preconditioner_list_factory(
                block_list=self._block_list,
                state=self._state,
                block_info_list=self._block_info_list,
                factor_matrix_dtype=torch.float64,
                **kwargs,
            )

        def _test_raise_invalid_value_in_factor_matrix(
            self, invalid_value: float
        ) -> None:
            self.assertRaisesRegex(
                PreconditionerValueError,
                re.escape("Encountered nan/inf values in factor matrix"),
                self._preconditioner_list.update_preconditioners,
                masked_grad_list=(
                    torch.tensor([invalid_value, invalid_value]),
                    torch.eye(2) / torch.tensor(2.0).sqrt(),
                    torch.tensor([[invalid_value, invalid_value]]),
                ),
                step=torch.tensor(1),
                perform_amortized_computation=True,
            )

        # Because nan as the input of self._preconditioner_list.update_preconditioners() would change the internal state to nan (and stayed as nan even after other updates),
        # we need to test the cases of nan and inf separately.
        def test_raise_inf_in_factor_matrix(self) -> None:
            self._test_raise_invalid_value_in_factor_matrix(invalid_value=torch.inf)

        def test_raise_nan_in_factor_matrix(self) -> None:
            self._test_raise_invalid_value_in_factor_matrix(invalid_value=torch.nan)

        def test_raise_nan_and_inf_in_inv_factor_matrix_amortized_computation(
            self,
        ) -> None:
            for invalid_value in self._amortized_computation_properties.invalid_amortized_computation_return_values:
                with (
                    self.subTest(invalid_value=invalid_value),
                    mock.patch.object(
                        shampoo_preconditioner_list,
                        self._amortized_computation_properties.amortized_computation_function_name,
                        side_effect=(invalid_value,),
                    ) as mock_amortized_computation,
                ):
                    self.assertRaisesRegex(
                        PreconditionerValueError,
                        re.escape("Encountered nan or inf values in"),
                        self._preconditioner_list.update_preconditioners,
                        masked_grad_list=(
                            torch.tensor([1.0, 0.0]),
                            torch.eye(2) / torch.tensor(2.0).sqrt(),
                            torch.tensor([[1.0, 0.0]]),
                        ),
                        step=torch.tensor(1),
                        perform_amortized_computation=True,
                    )
                mock_amortized_computation.assert_called_once()

        def test_amortized_computation_internal_failure(self) -> None:
            with mock.patch.object(
                shampoo_preconditioner_list,
                self._amortized_computation_properties.amortized_computation_function_name,
                # Simulate the situation throws an exception (not nan and inf) to test the warning
                side_effect=ZeroDivisionError,
            ) as mock_amortized_computation:
                with self.assertLogs(level="WARNING") as cm:
                    self._preconditioner_list.update_preconditioners(
                        masked_grad_list=(
                            torch.tensor([1.0, 0.0]),
                            torch.eye(2) / torch.tensor(2.0).sqrt(),
                            torch.tensor([[1.0, 0.0]]),
                        ),
                        step=torch.tensor(1),
                        perform_amortized_computation=True,
                    )
                self.assertCountEqual(
                    # Only extracts the first sentence in the warning message for simple comparison.
                    [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                    [
                        "Matrix computation failed for factor matrix 0.block_0.0 with exception=ZeroDivisionError()",
                        "Matrix computation failed for factor matrix 1.block_0.0 with exception=ZeroDivisionError()",
                        "Matrix computation failed for factor matrix 1.block_0.1 with exception=ZeroDivisionError()",
                        "Matrix computation failed for factor matrix 1.block_1.0 with exception=ZeroDivisionError()",
                        "Matrix computation failed for factor matrix 1.block_1.1 with exception=ZeroDivisionError()",
                    ],
                )
                mock_amortized_computation.assert_called()
                mock_amortized_computation.reset_mock()

        def test_amortized_computation_failure_tolerance(self) -> None:
            self._preconditioner_list = self._instantiate_preconditioner_list()
            masked_grad_list0 = (
                torch.tensor([1.0, 0.0]),
                torch.eye(2) / torch.tensor(2.0).sqrt(),
                torch.tensor([[1.0, 0.0]]),
            )
            masked_grad_list = (
                torch.tensor([0.0, 1.0]),
                torch.eye(2) / torch.tensor(2.0).sqrt(),
                torch.tensor([[0.0, 1.0]]),
            )

            # Number of calls to the amortized computation function per update.
            NUM_AMORTIZED_COMPUTATION_CALLS = 5

            # Initialize step counter.
            step = 1
            # Define the side effect for each call of the amortized computation function.
            fail = ValueError
            success = self._amortized_computation_properties.valid_amortized_computation_return_value
            all_but_one_fail = (fail,) * (NUM_AMORTIZED_COMPUTATION_CALLS - 1) + (
                success,
            )
            all_fail = (fail,) * NUM_AMORTIZED_COMPUTATION_CALLS
            all_success = (success,) * NUM_AMORTIZED_COMPUTATION_CALLS
            with (
                mock.patch.object(
                    shampoo_preconditioner_list,
                    self._amortized_computation_properties.amortized_computation_function_name,
                    # Note that the cases causally depend on each other.
                    side_effect=[
                        # Case 1: amortized computation fails less often than tolerance.
                        *all_but_one_fail,  # Success for a single Kronecker factor is not enough to reset counter.
                        # Case 2: amortized computation fails exactly as often as tolerance (3).
                        *all_fail,
                        *all_fail,
                        # Case 3: amortized computation succeeds after tolerance hit (counter is reset).
                        *all_success,
                        # Case 4: amortized computation fails more often than tolerance.
                        *all_fail,
                        *all_fail,
                        *all_fail,
                        fail,  # One failure is enough to raise an exception in this case.
                    ],
                ) as mock_amortized_computation
            ):
                # Accumulate factor matrices for valid amortized computation.
                self._preconditioner_list.update_preconditioners(
                    masked_grad_list=masked_grad_list0,
                    step=torch.tensor(step),
                    perform_amortized_computation=False,
                )
                self.assertEqual(mock_amortized_computation.call_count, 0)
                step += 1

                # Case 1: amortized computation fails less often than tolerance -> no error.
                with self.assertLogs(level="WARNING") as cm:
                    self._preconditioner_list.update_preconditioners(
                        masked_grad_list=masked_grad_list,
                        step=torch.tensor(step),
                        perform_amortized_computation=True,
                    )
                # Check that warnings are logged for four failed amortized computations.
                # The fifth one doesn't raise an exception (see the definition of the side effect), so no warning is logged.
                self.assertCountEqual(
                    # Only extracts the first sentence in the warning message for simple comparison.
                    [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                    [
                        "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                        "Matrix computation failed for factor matrix 1.block_0.0 with exception=ValueError()",
                        "Matrix computation failed for factor matrix 1.block_0.1 with exception=ValueError()",
                        "Matrix computation failed for factor matrix 1.block_1.0 with exception=ValueError()",
                    ],
                )
                step += 1

                # Case 2: amortized computation fails exactly as often as tolerance (3) -> no error.
                for _ in range(2):
                    with self.assertLogs(level="WARNING") as cm:
                        self._preconditioner_list.update_preconditioners(
                            masked_grad_list=masked_grad_list,
                            step=torch.tensor(step),
                            perform_amortized_computation=True,
                        )
                    # Check that warnings are logged for all failed amortized computations.
                    self.assertCountEqual(
                        # Only extracts the first sentence in the warning message for simple comparison.
                        [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                        [
                            "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.1 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.1 with exception=ValueError()",
                        ],
                    )
                    step += 1

                # Case 3: amortized computation succeeds after tolerance hit (test reset) -> no error.
                with self.assertNoLogs(level="WARNING"):
                    self._preconditioner_list.update_preconditioners(
                        masked_grad_list=masked_grad_list,
                        step=torch.tensor(step),
                        perform_amortized_computation=True,
                    )
                step += 1

                # Case 4: amortized computation fails more often than tolerance -> error.
                for _ in range(3):
                    with self.assertLogs(level="WARNING") as cm:
                        self._preconditioner_list.update_preconditioners(
                            masked_grad_list=masked_grad_list,
                            step=torch.tensor(step),
                            perform_amortized_computation=True,
                        )
                    # Check that warnings are logged for four failed amortized computations.
                    self.assertCountEqual(
                        # Only extracts the first sentence in the warning message for simple comparison.
                        [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                        [
                            "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_0.1 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.0 with exception=ValueError()",
                            "Matrix computation failed for factor matrix 1.block_1.1 with exception=ValueError()",
                        ],
                    )
                    step += 1
                # Exactly at failure tolerance now.
                with self.assertLogs(level="WARNING") as cm:
                    expected_error_message = r"The number of failed .* for factors \('0.block_0.0',\) exceeded the allowed tolerance\."
                    self.assertRaisesRegex(
                        ValueError,
                        expected_error_message,
                        self._preconditioner_list.update_preconditioners,
                        masked_grad_list=masked_grad_list,
                        step=torch.tensor(step),
                        perform_amortized_computation=True,
                    )
                    # Check that the warning is logged for the failed amortized computation of the first matrix.
                    self.assertCountEqual(
                        # Only extracts the first sentence in the warning message for simple comparison.
                        [r.msg.split(". ", maxsplit=1)[0] for r in cm.records],
                        [
                            "Matrix computation failed for factor matrix 0.block_0.0 with exception=ValueError()",
                        ],
                    )

        def test_precondition_grad(self) -> None:
            # Generate a random gradient tensor with shape (2, 3, 4, 5, 6, 7).
            grad = torch.randn((2, 3, 4, 5, 6, 7))

            # Define selectors for which dimensions to precondition in the experimental setup.
            # Note that in the control setup, we will precondtion all dimensions normally except for the `False` ones with identity matrices.
            experimental_preconditioned_dims_selector = (
                True,
                False,
                False,
                True,
                True,
                False,
            )
            # Define selectors for which dimensions to precondition in the control setup.
            control_preconditioned_dims_selector = (True,) * grad.ndim

            # Create a list of random preconditioner matrices for each dimension of the gradient.
            preconditioner_list = [torch.randn((d, d)) for d in grad.shape]

            # Compress the preconditioner list based on experimental_preconditioned_dims_selector.
            experimental_preconditioner_list = compress_list(
                preconditioner_list,
                experimental_preconditioned_dims_selector,
            )

            # Create a control preconditioner list, using identity matrices where not preconditioning.
            control_preconditioner_list = [
                preconditioner
                if should_precondition
                else torch.eye(preconditioner.shape[0])
                for preconditioner, should_precondition in zip(
                    preconditioner_list,
                    experimental_preconditioned_dims_selector,
                    strict=True,
                )
            ]

            # Compare the results of preconditioning the gradient with both setups for different contract dimensions.
            for dims in (([0], [0]), ([0], [1])):
                with self.subTest(dims=dims):
                    torch.testing.assert_close(
                        self._preconditioner_list._precondition_grad(  # type: ignore[attr-defined]
                            grad=grad,
                            preconditioned_dims_selector=experimental_preconditioned_dims_selector,
                            preconditioner_list=experimental_preconditioner_list,
                            dims=dims,
                        ),
                        self._preconditioner_list._precondition_grad(  # type: ignore[attr-defined]
                            grad=grad,
                            preconditioned_dims_selector=control_preconditioned_dims_selector,
                            preconditioner_list=control_preconditioner_list,
                            dims=dims,
                        ),
                    )

        def test_numel_list(self) -> None:
            self.assertEqual(self._preconditioner_list.numel_list, (8, 16, 10))

        def test_dims_list(self) -> None:
            self.assertEqual(
                self._preconditioner_list.dims_list,
                (torch.Size([2]), torch.Size([2, 2]), torch.Size([1, 2])),
            )

        def test_num_bytes_list(self) -> None:
            self.assertEqual(self._preconditioner_list.num_bytes_list, (48, 96, 60))

        def test_numel(self) -> None:
            self.assertEqual(self._preconditioner_list.numel(), 34)

        def test_num_bytes(self) -> None:
            self.assertEqual(self._preconditioner_list.num_bytes(), 204)

        def test_compress_preconditioner_list(self) -> None:
            self._test_compress_preconditioner_list(expected_compress_list_call_count=5)


class ShampooPreconditionerListTest(AbstractTest.BaseShampooPreconditionerListTest):
    @property
    def _amortized_computation_properties(self) -> InverseRootProperties:
        return InverseRootProperties()

    @property
    def _default_preconditioner_config(self) -> ShampooPreconditionerConfig:
        return DefaultShampooConfig

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return ShampooPreconditionerList

    def test_update_preconditioners_and_precondition(self) -> None:
        """
        We provide examples where we update the preconditioners twice using specially
        chosen gradients such that we get a scalar * identity matrix for both Kronecker
        factor matrices for all parameters of interest.

        Specifically, for the beta2 = 1 case, we have 3 parameters and define their gradients
        as the following in order to get the expected preconditioned gradient list:

        (1) Tensor of Size 2
            G1 = [1, 0]^T
            G2 = [0, 1]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            P = L^{-1/2} G2 = [0, 1]^T = G2

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            P = L^{-1/4} G2 R^{-1/4} = [[1, 0], [0, 1]] / sqrt(2) = G2

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 1]]

            L = G1 * G1^T + G2 * G2^T = 2
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            P = L^{-1/4} G2 R^{-1/4} = 2^{-1/4} * [[0, 1]] = 2^{-1/4} G2

        """
        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[1.0, 0.0]]),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 1.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 1.0]]),
        )

        masked_expected_preconditioned_grad_list = [
            preconditioned_grad.clone() for preconditioned_grad in masked_grad_list2
        ]
        masked_expected_preconditioned_grad_list[2] /= torch.tensor(2.0 ** (1 / 4))

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                use_bias_correction=True,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

        """
        For the other two cases (beta2 < 1), note:

            L = beta2 * (1 - beta2) * G1 * G1^T + (1 - beta2) * G2 * G2^T
            R = beta2 * (1 - beta2) * G1^T * G1 + (1 - beta2) * G2^T * G2

        Therefore, in order to retain the identity matrix, we simply need to scale each gradient by:

            G1 -> G1 / sqrt(beta2 * (1 - beta2))
            G2 -> G2 / sqrt(1 - beta2).

        """
        beta2 = 0.9

        beta2_compensated_grad_list1 = torch._foreach_div(
            masked_grad_list1,
            torch.tensor(beta2 * (1 - beta2)).sqrt(),
        )
        beta2_compensated_grad_list2 = torch._foreach_div(
            masked_grad_list2,
            torch.tensor(1 - beta2).sqrt(),
        )

        masked_expected_preconditioned_grad_list = [
            preconditioned_grad.clone()
            for preconditioned_grad in beta2_compensated_grad_list2
        ]
        masked_expected_preconditioned_grad_list[2] /= torch.tensor(2.0 ** (1 / 4))

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=beta2,
                use_bias_correction=False,
            ),
            masked_grad_lists=[
                beta2_compensated_grad_list1,
                beta2_compensated_grad_list2,
            ],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

        """
        For the last case of including bias correction, we re-scale the entire matrix by the
        bias correction at iteration 2.

            L -> L / (1 - beta2^2)
            R -> R / (1 - beta2^2).

        Therefore, it is sufficient to additionally scale by this value:

            G1 -> sqrt(1 - beta2^2) * G1
            G2 -> sqrt(1 - beta2^2) * G2.

        """
        bias_compensated_grad_list1 = torch._foreach_mul(
            beta2_compensated_grad_list1,
            torch.tensor(1 - beta2**2).sqrt(),
        )
        bias_compensated_grad_list2 = torch._foreach_mul(
            beta2_compensated_grad_list2,
            torch.tensor(1 - beta2**2).sqrt(),
        )

        masked_expected_preconditioned_grad_list = [
            preconditioned_grad.clone()
            for preconditioned_grad in bias_compensated_grad_list2
        ]
        masked_expected_preconditioned_grad_list[2] /= torch.tensor(2.0 ** (1 / 4))

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=beta2,
                use_bias_correction=True,
            ),
            masked_grad_lists=[
                bias_compensated_grad_list1,
                bias_compensated_grad_list2,
            ],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

    def test_update_preconditioners_and_precondition_with_dims_ignored(self) -> None:
        """

        (1) Tensor of Size 2
            G1 = [4, 0]^T
            G2 = [0, 4]^T

            L = G1 * G1^T + G2 * G2^T = [[4*4, 0], [0, 4*4]]
            P = L^{-1/2} G2 = [[1/4, 0], [0, 1/4]] G2 = [0, 1]^T

        (2) Tensor of Size 2 x 2
            G1 = [[3, 0], [0, 3]]
            G2 = [[4, 0], [0, 4]]

            L = G1 * G1^T + G2 * G2^T = [[3*3+4*4, 0], [0, 3*3+4*4]]
            R = G1^T * G1 + G2^T * G2 = [[3*3+4*4, 0], [0, 3*3+4*4]]
            P = L^{-1/4} G2 R^{-1/4} = [[1/sqrt(5), 0], [0, 1/sqrt(5)]] G2 [[1/sqrt(5), 0], [0, 1/sqrt(5)]] = G2 / 5

        (3) Tensor of Size 1 x 2
            G1 = [[2, 0]]
            G2 = [[0, 2]]

            L = G1 * G1^T + G2 * G2^T = 2*2+2*2 = 8
            R = G1^T * G1 + G2^T * G2 = [[4, 0], [0, 4]]
            P = L^{-1/4} G2 R^{-1/4} = 8^{-1/4} G2 [[1/sqrt(2), 0], [0, 1/sqrt(2)]] = G2 / (sqrt(2 * sqrt(8)))

        """
        masked_grad_list1 = (
            torch.tensor([4.0, 0.0]),
            torch.eye(2) * 3,
            torch.tensor([[2.0, 0.0]]),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 4.0]),
            torch.eye(2) * 4,
            torch.tensor([[0.0, 2.0]]),
        )

        masked_expected_preconditioned_grad_list = [
            torch.tensor([0.0, 1.0]),
            masked_grad_list2[1] / 5,
            masked_grad_list2[2] / math.sqrt(2 * math.sqrt(8)),
        ]

        # The default case where we do not ignore any dimensions.
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

        # When ignoring all the dimensions by setting all inverse exponent override values to 0.0, the preconditioner should be the identity matrix, and the expected preconditioned gradient should be the same as the input gradient.
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                preconditioner_config=replace(
                    self._default_preconditioner_config,
                    inverse_exponent_override={
                        0: {0: 0.0},
                        1: {0: 0.0},
                        2: {0: 0.0, 1: 0.0},
                    },
                ),
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=masked_grad_list2,
        )

    def test_inverse_exponent_override(self) -> None:
        """
        For this example, we modify the one given above such that the inverse_exponent_override = {0: 1.0, 1: 1.0, 2: 1.0}.
        This effectively means all tensors in this test setting will use the inverse root of 1 rather than the default.
        This should result in the following behavior:

        (1) Tensor of Size 2
            G1 = [1, 0]^T
            G2 = [0, 2]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 4]]
            P = L^{-1} G2 = [0, 0.5]^T

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            P = L^{-1} G2 R^{-1} = [[1, 0], [0, 1]] / sqrt(2) = G2

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 2]]

            L = G1 * G1^T + G2 * G2^T = 5
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            P = L^{-1} G2 R^{-1} =  [[0, 0.1]]

        """

        preconditioner_config = replace(
            self._default_preconditioner_config,
            inverse_exponent_override={
                0: {0: 1.0},
                1: {0: 1.0},
                2: {0: 1.0, 1: 1.0},
            },
        )

        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[1.0, 0.0]]),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 2.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 2.0]]),
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0, 0.5]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0, 0.1]]),
        )

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                use_bias_correction=True,
                preconditioner_config=preconditioner_config,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
        )


class EigendecomposedShampooPreconditionerListTest(ShampooPreconditionerListTest):
    @property
    def _amortized_computation_properties(self) -> EigendecompositionProperties:  # type: ignore[override]
        return EigendecompositionProperties()

    @property
    def _default_preconditioner_config(self) -> ShampooPreconditionerConfig:
        return ShampooPreconditionerConfig(
            amortized_computation_config=QREigendecompositionConfig()
        )

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return EigendecomposedShampooPreconditionerList


class EigenvalueCorrectedShampooPreconditionerListTest(
    AbstractTest.BaseShampooPreconditionerListTest
):
    @property
    def _amortized_computation_properties(self) -> EigendecompositionProperties:
        return EigendecompositionProperties()

    @property
    def _default_preconditioner_config(
        self,
    ) -> EigenvalueCorrectedShampooPreconditionerConfig:
        return DefaultSOAPConfig

    @property
    def _preconditioner_list_factory(self) -> Callable[..., PreconditionerList]:
        return EigenvalueCorrectedShampooPreconditionerList

    def test_update_preconditioners_and_precondition(self) -> None:
        """
        We provide examples where we update the preconditioners twice using specially
        chosen gradients such that we get a scalar * identity matrix for both Kronecker
        factor matrices for all parameters of interest.

        Specifically, for the beta2 = 1 case, we have 3 parameters and define their gradients
        as the following in order to get the expected preconditioned gradient list:

        (1) Tensor of Size 2
            G1 = [1, 0]^T
            G2 = [0, 2]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 4]]
            B = [[1, 0], [0, 1]]  # eigenvectors of L
            E = G1^2 + (B G2)^2   # corrected eigenvalues
            P = B ((B G2) / sqrt(E + eps)) = G2 / sqrt(E + eps)

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            B_L = [[1, 0], [0, 1]]     # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / sqrt(E + eps) B_R = G2 / sqrt(E + eps)

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 2]]

            L = G1 * G1^T + G2 * G2^T = 5
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            B_L = 1                    # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / sqrt(E + eps) B_R = G2 / sqrt(E + eps)

        """
        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[1.0, 0.0]]),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 2.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 2.0]]),
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 1.0]]),
        )
        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                use_bias_correction=True,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
        )

        """
        For the other two cases (beta2 < 1), note:

            E = beta2 * (1 - beta2) G1^2 + (1 - beta2) G2^2

        Therefore, in order to retain the identity matrix, we simply need to scale each gradient by:

            G1 -> G1 / sqrt(beta2 * (1 - beta2))
            G2 -> G2 / sqrt(1 - beta2).

        """
        beta2 = 0.9

        beta2_compensated_grad_list1 = torch._foreach_div(
            masked_grad_list1,
            torch.tensor(beta2 * (1 - beta2)).sqrt(),
        )
        beta2_compensated_grad_list2 = torch._foreach_div(
            masked_grad_list2,
            torch.tensor(1 - beta2).sqrt(),
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 1.0]]),
        )
        # Fix scaling due to EMA.
        torch._foreach_div_(
            masked_expected_preconditioned_grad_list,
            torch.tensor(1 - beta2).sqrt(),
        )

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=beta2,
                use_bias_correction=False,
            ),
            masked_grad_lists=[
                beta2_compensated_grad_list1,
                beta2_compensated_grad_list2,
            ],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

        """
        For the last case of including bias correction, we re-scale the entire matrix by the
        bias correction at iteration 2.

            E -> E / (1 - beta2^2).

        Therefore, it is sufficient to additionally scale by this value:

            G1 -> sqrt(1 - beta2^2) * G1
            G2 -> sqrt(1 - beta2^2) * G2.

        """
        bias_compensated_grad_list1 = torch._foreach_mul(
            beta2_compensated_grad_list1,
            torch.tensor(1 - beta2**2).sqrt(),
        )
        bias_compensated_grad_list2 = torch._foreach_mul(
            beta2_compensated_grad_list2,
            torch.tensor(1 - beta2**2).sqrt(),
        )

        # Fix scaling due to bias correction.
        torch._foreach_mul_(
            masked_expected_preconditioned_grad_list,
            torch.tensor(1 - beta2**2).sqrt(),
        )

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=beta2,
                use_bias_correction=True,
            ),
            masked_grad_lists=[
                bias_compensated_grad_list1,
                bias_compensated_grad_list2,
            ],
            masked_expected_preconditioned_grad_list=tuple(
                masked_expected_preconditioned_grad_list
            ),
        )

    def test_inverse_exponent_override(self) -> None:
        """
        For this example, we modify the one given above such that the inverse_exponent_override = {0: 1.0, 1: 1.0, 2: 1.0}.
        This effectively means all tensors in this test setting will use the inverse root of 1 rather than the default, 1/2.
        This should result in the following behavior:

        (1) Tensor of Size 2
            G1 = [1, 0]^T
            G2 = [0, 2]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 4]]
            B = [[1, 0], [0, 1]]  # eigenvectors of L
            E = G1^2 + (B G2)^2   # corrected eigenvalues
            P = B ((B G2) / (E + eps) = G2 / (E + eps) ≈ [0, 0.5]^T

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            B_L = [[1, 0], [0, 1]]     # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / (E + eps) B_R = G2 / (E + eps) ≈ G2

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 2]]

            L = G1 * G1^T + G2 * G2^T = 5
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            B_L = 1                    # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / (E + eps)) B_R = G2 / (E + eps) ≈ [[0, 0.5]]

        """

        preconditioner_config = EigenvalueCorrectedShampooPreconditionerConfig(
            inverse_exponent_override={0: 1.0, 1: 1.0, 2: 1.0},
        )

        masked_grad_list1 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[1.0, 0.0]]),
        )
        masked_grad_list2 = (
            torch.tensor([0.0, 2.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 2.0]]),
        )

        masked_expected_preconditioned_grad_list = (
            torch.tensor([0, 0.5]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0, 0.5]]),
        )

        self._test_update_preconditioners_and_precondition(
            preconditioner_list=self._instantiate_preconditioner_list(
                beta2=1.0,
                use_bias_correction=True,
                preconditioner_config=preconditioner_config,
            ),
            masked_grad_lists=[masked_grad_list1, masked_grad_list2],
            masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
        )

    def test_adaptive_amortized_computation_frequency(self):
        # Setup the preconditioner list with the adaptive amortized computation frequency.
        self._preconditioner_list = self._instantiate_preconditioner_list(
            preconditioner_config=EigenvalueCorrectedShampooPreconditionerConfig(
                amortized_computation_frequency_config=AdaptiveAmortizedComputationFrequencyConfig(
                    tolerance=1e-4,  # The value does not matter here.
                ),
            ),
        )
        # Create the masked gradients for the test.
        masked_grad_list0 = (
            torch.tensor([1.0, 0.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[1.0, 0.0]]),
        )
        masked_grad_list = (
            torch.tensor([0.0, 1.0]),
            torch.eye(2) / torch.tensor(2.0).sqrt(),
            torch.tensor([[0.0, 1.0]]),
        )

        # Setup the constants for the mock functions.
        NUM_FACTOR_MATRICES = sum(grad.dim() for grad in masked_grad_list0)
        # If criterion is False, amortized computation is performed.
        # If criterion is True, amortized computation is not performed.
        # We want to simulate two update steps.
        CRITERION_RESULTS_STEP_ONE = [False, True, False, False, True]
        assert len(CRITERION_RESULTS_STEP_ONE) == NUM_FACTOR_MATRICES
        CRITERION_RESULTS_STEP_TWO = [True, True, False, True, False]
        assert len(CRITERION_RESULTS_STEP_TWO) == NUM_FACTOR_MATRICES
        # We will perform two update steps. If we performed the amortized computation
        # for all factor matrices in both steps, we would have 2 * NUM_FACTOR_MATRICES
        # calls. However, we will skip the update when the criterion result is mocked
        # to be True, i.e. sum(CRITERION_RESULTS_STEP_ONE/TWO) times.
        # Hence, the number of calls of the amortized computation function should be:
        NUM_AMORTIZED_COMPUTATION_CALLS_STEP_ONE = NUM_FACTOR_MATRICES - sum(
            CRITERION_RESULTS_STEP_ONE
        )
        NUM_AMORTIZED_COMPUTATION_CALLS_STEP_TWO = NUM_FACTOR_MATRICES - sum(
            CRITERION_RESULTS_STEP_TWO
        )
        NUM_AMORTIZED_COMPUTATION_CALLS = (
            NUM_AMORTIZED_COMPUTATION_CALLS_STEP_ONE
            + NUM_AMORTIZED_COMPUTATION_CALLS_STEP_TWO
        )

        # Mock the amortized computation function and the criterion function.
        with (
            mock.patch.object(
                shampoo_preconditioner_list,
                self._amortized_computation_properties.amortized_computation_function_name,
                side_effect=[*(torch.tensor([1.0]),) * NUM_AMORTIZED_COMPUTATION_CALLS],
            ) as mock_amortized_computation,
            mock.patch.object(
                shampoo_preconditioner_list,
                "estimated_eigenvalues_criterion_below_or_equal_tolerance",
                side_effect=CRITERION_RESULTS_STEP_ONE + CRITERION_RESULTS_STEP_TWO,
            ) as mock_criterion,
        ):
            # First update step.
            self._preconditioner_list.update_preconditioners(
                masked_grad_list=masked_grad_list0,
                step=torch.tensor(1),
                perform_amortized_computation=True,
            )
            # Amortized computation is performed for all factor matrices in the first step.
            self.assertEqual(mock_criterion.call_count, NUM_FACTOR_MATRICES)
            self.assertEqual(
                mock_amortized_computation.call_count,
                NUM_AMORTIZED_COMPUTATION_CALLS_STEP_ONE,
            )
            # Second update step.
            self._preconditioner_list.update_preconditioners(
                masked_grad_list=masked_grad_list,
                step=torch.tensor(2),
                perform_amortized_computation=True,
            )
            # The criterion is evaluated for all factor matrices in the second step.
            self.assertEqual(mock_criterion.call_count, 2 * NUM_FACTOR_MATRICES)
            # Amortized computation is performed for the first, third, and fourth factor matrices in the second step.
            self.assertEqual(
                mock_amortized_computation.call_count,
                NUM_AMORTIZED_COMPUTATION_CALLS,
            )
