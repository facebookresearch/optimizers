"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import abc
import re
import unittest
from types import ModuleType
from typing import Any
from unittest import mock

import torch
from distributed_shampoo.shampoo_types import (
    DefaultEigenvalueCorrectedShampooConfig,
    DefaultShampooConfig,
    PreconditionerValueError,
    ShampooPreconditionerConfig,
)

from distributed_shampoo.utils import shampoo_preconditioner_list
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    BaseShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    PreconditionerList,
    SGDPreconditionerList,
    ShampooPreconditionerList,
)
from matrix_functions_types import EigenConfig
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
            # Count the number of list compressions at the preconditioner list level, including compressions of QuantizedTensorList.
            # Each call to compress() under QuantizedTensorList counts once, though note that it calls compress_list() three times inside.
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
        # self._params[1] are split into two blocks, and self._params[2] forms a block by itself.
        return (
            self._params[0],
            *torch.split(self._params[1], 2, dim=0),
            self._params[2],
        )

    def _instantiate_preconditioner_list(
        self, **kwargs: Any
    ) -> AdagradPreconditionerList:  # type: ignore[override]
        kwargs = {"beta2": 1.0, "epsilon": 0.0, "use_bias_correction": True} | kwargs
        return AdagradPreconditionerList(
            block_list=self._block_list,
            state=self._state,
            block_info_list=self._block_info_list,
            distributor_selector=self._distributor_selector,
            **kwargs,
        )

    def setUp(self) -> None:
        self._params = (
            torch.tensor([1.0, 2.0]),
            torch.arange(6, dtype=torch.float).reshape(3, 2),
            # Following param will not be used due to the distributor selector below.
            torch.tensor([torch.nan, torch.nan]),
        )
        self._state = {  # type: ignore[var-annotated]
            self._params[0]: {},
            self._params[1]: {},
            self._params[2]: {},
        }
        # Because maximum_preconditioner_dim = 2, self._params[0] forms a block by itself,
        # self._params[1] are split into two blocks, and self._params[2] forms a block by itself.
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
            BlockInfo(
                param=self._params[2],
                composable_block_ids=(2, "block_0"),
            ),
        )
        # Ignores the last block, which is self._params[2] itself.
        self._distributor_selector = (True, True, True, False)
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

        # Mock _update_factor_matrices() otherwise the access of factor_matrices will throw errors.
        with mock.patch.object(
            BaseShampooPreconditionerList, "_update_factor_matrices"
        ) as mock_update_factor_matrices:
            # Test the abstract methods _create_kronecker_factors_state_for_block(), _create_kronecker_factors_list(), and _get_inverse_roots_from_override().
            preconditioner_list = BaseShampooPreconditionerList(  # type: ignore
                block_list=(param,),
                state={param: {}},
                block_info_list=(
                    BlockInfo(
                        param=param,
                        composable_block_ids=(0, "block_0"),
                    ),
                ),
                distributor_selector=(True,),
                preconditioner_config=DefaultShampooConfig,
                beta2=1.0,
            )

            # Test the abstract_method _amortized_computation().
            preconditioner_list.update_preconditioners(
                masked_grad_list=(torch.tensor([1.0, 1.0]),),
                step=torch.tensor(1),
                perform_amortized_computation=True,
            )

            mock_update_factor_matrices.assert_called_once()


# Use outer class as wrapper to avoid running the abstract test.
class AbstractTest:
    class BaseShampooPreconditionerListTest(abc.ABC, AdagradPreconditionerListTest):
        @abc.abstractmethod
        def _amortized_computation_function(self) -> str: ...

        @abc.abstractmethod
        def _instantiate_preconditioner_list(  # type: ignore[override]
            self, **kwargs: Any
        ) -> PreconditionerList: ...

        def _test_raise_invalid_value_in_factor_matrix(
            self, invalid_value: float
        ) -> None:
            with self.assertRaisesRegex(
                PreconditionerValueError,
                re.escape(f"Encountered {str(invalid_value)} values in factor matrix"),
            ):
                self._preconditioner_list.update_preconditioners(
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
            for invalid_value in (torch.nan, torch.inf):
                with self.subTest(invalid_value=invalid_value), self.assertRaisesRegex(
                    PreconditionerValueError,
                    re.escape("Encountered nan or inf values in"),
                ), mock.patch.object(
                    shampoo_preconditioner_list,
                    self._amortized_computation_function(),
                    side_effect=(torch.tensor([invalid_value]),),
                ) as mock_amortized_computation:
                    self._preconditioner_list.update_preconditioners(
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
                self._amortized_computation_function(),
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

        # Note: This is needed for type checking to infer the type of argument into mock.patch.object.
        shampoo_preconditioner_list_module: ModuleType = shampoo_preconditioner_list

        @mock.patch.object(
            shampoo_preconditioner_list_module,
            "check_diagonal",
            return_value=False,
        )
        def test_amortized_computation_factor_matrix_non_diagonal(
            self, mock_check_diagonal: mock.Mock
        ) -> None:
            self._preconditioner_list = self._instantiate_preconditioner_list(
                epsilon=1.0
            )
            with self.assertLogs(
                level="DEBUG",
            ) as cm:
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
                [r.msg for r in cm.records],
                [
                    "Factor matrix 0.block_0.0 is not diagonal.",
                    "Factor matrix 1.block_0.0 is not diagonal.",
                    "Factor matrix 1.block_0.1 is not diagonal.",
                    "Factor matrix 1.block_1.0 is not diagonal.",
                    "Factor matrix 1.block_1.1 is not diagonal.",
                ],
            )
            mock_check_diagonal.assert_called()

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
            self._test_compress_preconditioner_list(expected_compress_list_call_count=3)


class ShampooPreconditionerListTest(AbstractTest.BaseShampooPreconditionerListTest):
    def _amortized_computation_function(self) -> str:
        return "matrix_inverse_root"

    def _instantiate_preconditioner_list(  # type: ignore[override]
        self, **kwargs: Any
    ) -> ShampooPreconditionerList:
        kwargs = {
            "beta2": 1.0,
            "epsilon": 0.0,
            "inv_root_override": 0,
            "use_bias_correction": True,
            "preconditioner_config": DefaultShampooConfig,
        } | kwargs
        return ShampooPreconditionerList(
            block_list=self._block_list,
            state=self._state,
            block_info_list=self._block_info_list,
            distributor_selector=self._distributor_selector,
            factor_matrix_dtype=torch.float64,
            **kwargs,  # type: ignore[arg-type]
        )

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
            P = L^{-1/4} G2 = [0, 1]^T = G2

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

    def test_inv_root_override_and_exponent_multiplier(self) -> None:
        """
        For this example, we modify the one given above such that the inv_root_override = 2
        and exponent_multiplier = 2.0. Therefore, in all cases, the exponent is -2 / 2 = -1.
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

            L = G1 * G1^T + G2 * G2^T = 2
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            P = L^{-1} G2 R^{-1} =  [[0, 0.1]]

        """

        def test_inverse_roots_from_override(
            inv_root_override: int | list[int],
        ) -> None:
            """
            Tests that the inverse roots are computed correctly from inv_root_override.
            """
            preconditioner_config = ShampooPreconditionerConfig(
                amortized_computation_config=EigenConfig(exponent_multiplier=2.0),
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

            with self.subTest(inv_root_override=inv_root_override):
                self._test_update_preconditioners_and_precondition(
                    preconditioner_list=self._instantiate_preconditioner_list(
                        beta2=1.0,
                        use_bias_correction=True,
                        inv_root_override=inv_root_override,
                        preconditioner_config=preconditioner_config,
                    ),
                    masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                    masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
                )

        test_inverse_roots_from_override(inv_root_override=2)
        test_inverse_roots_from_override(inv_root_override=[2, 2, 2])

    def test_compute_root_inverse_residuals(self) -> None:
        """
        Create a factor matrix of size 2x2 by updating preconditioners in two steps:
            Step 1. G1 = [1, 0]^T
            Step 2. G2 = [0, 1]^T

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
        """
        preconditioner_list = ShampooPreconditionerList(
            block_list=(self._params[0],),
            state=self._state,
            block_info_list=(self._block_info_list[0],),
            distributor_selector=(self._distributor_selector[0],),
            preconditioner_config=DefaultShampooConfig,
            epsilon=0.0,
        )

        masked_grad_list1 = (torch.tensor([1.0, 0.0]),)
        masked_grad_list2 = (torch.tensor([0.0, 1.0]),)
        preconditioner_list.update_preconditioners(
            masked_grad_list=masked_grad_list1,
            step=torch.tensor(1),
            perform_amortized_computation=False,
        )
        preconditioner_list.update_preconditioners(
            masked_grad_list=masked_grad_list2,
            step=torch.tensor(2),
            perform_amortized_computation=True,
        )

        # Expect no relative errors and residuals because L is a diagonal matrix.
        (
            relative_errors,
            relative_residuals,
        ) = preconditioner_list.compute_root_inverse_residuals()

        expected_relative_errors = (torch.tensor(0.0),)
        expected_relative_residuals = (torch.tensor(0.0),)

        self.assertTupleEqual(relative_errors, expected_relative_errors)
        self.assertTupleEqual(relative_residuals, expected_relative_residuals)


class EigenvalueCorrectedShampooPreconditionerListTest(
    AbstractTest.BaseShampooPreconditionerListTest
):
    def _amortized_computation_function(self) -> str:
        return "matrix_eigenvectors"

    def _instantiate_preconditioner_list(  # type: ignore[override]
        self, **kwargs: Any
    ) -> EigenvalueCorrectedShampooPreconditionerList:
        kwargs = {
            "beta2": 1.0,
            "epsilon": 1e-12,
            "inv_root_override": 0,
            "use_bias_correction": True,
            "preconditioner_config": DefaultEigenvalueCorrectedShampooConfig,
        } | kwargs
        return EigenvalueCorrectedShampooPreconditionerList(
            block_list=self._block_list,
            state=self._state,
            block_info_list=self._block_info_list,
            distributor_selector=self._distributor_selector,
            factor_matrix_dtype=torch.float64,
            **kwargs,  # type: ignore[arg-type]
        )

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
            B = [[1, 0], [0, 1]]  # eigenvectors of L
            E = G1^2 + (B G2)^2   # corrected eigenvalues
            P = B ((B G2) / sqrt(E + eps)) = G2 / sqrt(E + eps) ≈ G2

        (2) Tensor of Size 2 x 2
            G1 = [[1, 0], [0, 1]] / sqrt(2)
            G2 = [[1, 0], [0, 1]] / sqrt(2)

            L = G1 * G1^T + G2 * G2^T = [[1, 0], [0, 1]]
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            B_L = [[1, 0], [0, 1]]     # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / sqrt(E + eps) B_R = G2 / sqrt(E + eps) ≈ G2

        (3) Tensor of Size 1 x 2
            G1 = [[1, 0]]
            G2 = [[0, 1]]

            L = G1 * G1^T + G2 * G2^T = 2
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 1]]
            B_L = 1                    # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / sqrt(E + eps) B_R = G2 / sqrt(E + eps) ≈ G2

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

    def test_inv_root_override(self) -> None:
        """
        For this example, we modify the one given above such that the inv_root_override = 1.
        Therefore, in all cases, the exponent is -2 / 2 = -1.
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

            L = G1 * G1^T + G2 * G2^T = 2
            R = G1^T * G1 + G2^T * G2 = [[1, 0], [0, 4]]
            B_L = 1                    # eigenvectors of L
            B_R = [[1, 0], [0, 1]]     # eigenvectors of R
            E = G1^2 + (B_L G2 B_R)^2  # corrected eigenvalues
            P = B_L ((B_L G2 B_R) / (E + eps)) B_R = G2 / (E + eps) ≈ [[0, 0.5]]

        """

        def test_inverse_roots_from_override(
            inv_root_override: int | list[int],
        ) -> None:
            """
            Tests that the inverse roots are computed correctly from inv_root_override.
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
                torch.tensor([0, 0.5]),
                torch.eye(2) / torch.tensor(2.0).sqrt(),
                torch.tensor([[0, 0.5]]),
            )

            with self.subTest(inv_root_override=inv_root_override):
                self._test_update_preconditioners_and_precondition(
                    preconditioner_list=self._instantiate_preconditioner_list(
                        beta2=1.0,
                        use_bias_correction=True,
                        inv_root_override=inv_root_override,
                        preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
                    ),
                    masked_grad_lists=[masked_grad_list1, masked_grad_list2],
                    masked_expected_preconditioned_grad_list=masked_expected_preconditioned_grad_list,
                )

        test_inverse_roots_from_override(inv_root_override=1)
        test_inverse_roots_from_override(inv_root_override=[1, 1, 1])
