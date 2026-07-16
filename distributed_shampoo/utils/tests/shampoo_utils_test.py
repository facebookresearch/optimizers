"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import math
import re
import unittest
from operator import methodcaller
from unittest.mock import MagicMock

import torch
from distributed_shampoo.shampoo_types import (
    DISTRIBUTED_CONFIG,
    FSDPParamAssignmentStrategy,
    FullyShardDistributedConfig,
    HybridShardDistributedConfig,
    LoadBalancingConfig,
    PARAMS,
    SingleDeviceDistributedConfig,
)
from distributed_shampoo.utils.load_balancing_utils import (
    PolynomialComputationalCostModel,
)
from distributed_shampoo.utils.shampoo_fully_shard_utils import _compute_chunk_sizes
from distributed_shampoo.utils.shampoo_utils import (
    _device_key,
    _get_triu_indices,
    compress_list,
    distribute_buffer_sizes,
    generate_pairwise_indices,
    get_dtype_size,
    greedy_bin_pack,
    merge_small_dims,
    multi_dim_split,
    pack_upper_triangular,
    ParameterizeEnterExitContext,
    split_param_groups,
    unpack_upper_triangular,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)

# Hoisted as an annotated module constant so pyre can infer the type of the
# argument passed to @unittest.skipUnless (it cannot infer from a bare call).
_HAS_CUDA: bool = torch.cuda.is_available()


@instantiate_parametrized_tests
class MergeSmallDimsTest(unittest.TestCase):
    @parametrize("threshold, expected_new_tensor_shape", ((10, (10,)), (1, (2, 5))))
    def test_merge_all_small_dims(
        self, threshold: int, expected_new_tensor_shape: tuple[int, ...]
    ) -> None:
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(1, 2, 5, 1),
                threshold=threshold,
                target_tensor_dimensionality=1,
            ),
            expected_new_tensor_shape,
        )

    def test_merge_small_dims_for_single_dim(self) -> None:
        expected_new_tensor_shape = (2,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=torch.Size([2]),
                threshold=10,
                target_tensor_dimensionality=1,
            ),
            expected_new_tensor_shape,
        )

    @parametrize("threshold", (10, 1))
    def test_merge_small_dims_all_ones(self, threshold: int) -> None:
        expected_new_tensor_shape = (1,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(1, 1, 1, 1),
                threshold=threshold,
                target_tensor_dimensionality=1,
            ),
            expected_new_tensor_shape,
        )

    @parametrize(
        "tensor_shape", ((0,), (0, 1, 5, 10, 20), (1, 5, 0, 10, 20), (1, 5, 10, 20, 0))
    )
    def test_merge_small_dims_empty(self, tensor_shape: tuple[int, ...]) -> None:
        expected_new_tensor_shape = (0,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=tensor_shape, threshold=10, target_tensor_dimensionality=1
            ),
            expected_new_tensor_shape,
        )

    @parametrize("threshold", (10, 1))
    def test_empty_dims(self, threshold: int) -> None:
        expected_new_tensor_shape = (1,)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(), threshold=threshold, target_tensor_dimensionality=1
            ),
            expected_new_tensor_shape,
        )

    def test_target_tensor_dimensionality_is_inf(self) -> None:
        expected_new_tensor_shape = (1, 2, 5, 1)
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(1, 2, 5, 1),
                threshold=10,
                target_tensor_dimensionality=math.inf,
            ),
            expected_new_tensor_shape,
        )

    @parametrize(
        "threshold, target_tensor_dimensionality, expected_new_tensor_shape",
        [
            (10, 1, (32, 3, 64, 64)),
            (200, 1, (32, 192, 64)),
            (8192, 1, (96, 4096)),
            (1_000_000, 1, (96 * 4096,)),
            (10, 2, (32, 3, 64, 64)),
            (200, 2, (32, 192, 64)),
            (8192, 2, (96, 4096)),
            (
                1_000_000,
                2,
                (32, 3 * 4096),
            ),
            (8192, 1, (96, 4096)),
            (8192, 2, (96, 4096)),
            (8192, 3, (32, 3, 4096)),
            (8192, 4, (32, 3, 64, 64)),
            (8192, math.inf, (32, 3, 64, 64)),
        ],
    )
    def test_convolution_like_dims(
        self,
        threshold: int,
        target_tensor_dimensionality: int,
        expected_new_tensor_shape: tuple[int, ...],
    ) -> None:
        self.assertEqual(
            merge_small_dims(
                tensor_shape=(32, 3, 64, 64),
                threshold=threshold,
                target_tensor_dimensionality=target_tensor_dimensionality,
            ),
            expected_new_tensor_shape,
        )


class MultiDimSplitTest(unittest.TestCase):
    def test_multi_dim_split_for_one_dim(self) -> None:
        grad = torch.arange(10).reshape(5, 2)
        expected_split_grad = (
            torch.arange(6).reshape(3, 2),
            torch.arange(6, 10).reshape(2, 2),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=3), expected_split_grad
        )

    def test_multi_dim_split_for_two_dim(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (
            torch.tensor([[0, 1], [3, 4]]),
            torch.tensor([[2], [5]]),
            torch.tensor([[6, 7], [9, 10]]),
            torch.tensor([[8], [11]]),
            torch.tensor([[12, 13]]),
            torch.tensor([[14]]),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=2), expected_split_grad
        )

    def test_multi_dim_split_without_spliting(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (
            torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]),
        )
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=5), expected_split_grad
        )

    def test_split_size_is_inf(self) -> None:
        grad = torch.arange(15).reshape(5, 3)
        expected_split_grad = (grad,)
        torch.testing.assert_close(
            multi_dim_split(grad, split_size=math.inf), expected_split_grad
        )


@instantiate_parametrized_tests
class CompressListTest(unittest.TestCase):
    @parametrize(
        "selector, compressed_tuple",
        (
            ((True, True, False), (1, 2)),
            ((False, True, True), (2, 3)),
            ((True, False, True), (1, 3)),
        ),
    )
    def test_compress_list(
        self, selector: tuple[bool, ...], compressed_tuple: tuple[int,]
    ) -> None:
        self.assertEqual(
            compress_list(complete_list=[1, 2, 3], selector=selector), compressed_tuple
        )

    def test_compress_list_with_different_size(self) -> None:
        self.assertRaisesRegex(
            AssertionError,
            re.escape("Inconsistent lengths"),
            compress_list,
            complete_list=[1, 2, 3],
            selector=(True, False),
        )


@instantiate_parametrized_tests
class GetDTypeSizeTest(unittest.TestCase):
    @parametrize(
        "dtype, size",
        ((torch.int64, 8), (torch.float32, 4), (torch.bfloat16, 2), (torch.bool, 1)),
    )
    def test_get_dtype_size(self, dtype: torch.dtype, size: int) -> None:
        self.assertEqual(get_dtype_size(dtype), size)


class GeneratePairwiseIndicesTest(unittest.TestCase):
    def test_generate_pairwise_indices(self) -> None:
        input_tuple = (1, 3, 2)
        expected_pairwise_indices = [(0, 1), (1, 4), (4, 6)]
        self.assertListEqual(
            list(generate_pairwise_indices(input_tuple)), expected_pairwise_indices
        )

    def test_generate_pairwise_indices_with_empty_list(self) -> None:
        input_tuple = ()
        expected_pairwise_indices: list[int] = []
        self.assertListEqual(
            list(generate_pairwise_indices(input_tuple)), expected_pairwise_indices
        )


@instantiate_parametrized_tests
class ComputeChunkSizesTest(unittest.TestCase):
    """Tests for _compute_chunk_sizes which mirrors torch.chunk semantics."""

    @parametrize(
        "numel, num_chunks, expected",
        [
            # numel == 0: all chunks get size 0
            (0, 1, [0]),
            (0, 4, [0, 0, 0, 0]),
            # numel >= num_chunks, evenly divisible
            (4, 4, [1, 1, 1, 1]),
            (8, 4, [2, 2, 2, 2]),
            (12, 4, [3, 3, 3, 3]),
            # numel >= num_chunks, not evenly divisible (last chunk gets remainder)
            (10, 4, [3, 3, 3, 1]),
            (7, 3, [3, 3, 1]),
            (5, 2, [3, 2]),
            (1, 1, [1]),
            # numel < num_chunks: each element is its own chunk, extras are 0
            (2, 4, [1, 1, 0, 0]),
            (1, 4, [1, 0, 0, 0]),
            (3, 5, [1, 1, 1, 0, 0]),
        ],
    )
    def test_compute_chunk_sizes(
        self, numel: int, num_chunks: int, expected: list[int]
    ) -> None:
        result = _compute_chunk_sizes(numel, num_chunks)
        self.assertEqual(result, expected)

    @parametrize("numel", (1, 5, 10, 13, 32, 100))
    @parametrize("num_chunks", (1, 2, 3, 4, 7, 8))
    def test_matches_torch_chunk(self, numel: int, num_chunks: int) -> None:
        """Verify _compute_chunk_sizes matches actual torch.chunk behavior."""
        tensor = torch.arange(numel)
        actual_chunks = torch.chunk(tensor, num_chunks, dim=0)
        actual_sizes = [c.numel() for c in actual_chunks]

        computed_sizes = _compute_chunk_sizes(numel, num_chunks)
        # torch.chunk may produce fewer chunks than num_chunks if numel < num_chunks.
        # _compute_chunk_sizes pads with zeros in that case.
        nonzero_computed = [s for s in computed_sizes if s > 0]
        self.assertEqual(nonzero_computed, actual_sizes)
        self.assertEqual(sum(computed_sizes), numel)

    def test_sum_equals_numel(self) -> None:
        """Verify all chunk sizes sum to the original numel."""
        for numel in range(0, 20):
            for num_chunks in range(1, 8):
                sizes = _compute_chunk_sizes(numel, num_chunks)
                self.assertEqual(
                    sum(sizes),
                    numel,
                    f"Sum mismatch for numel={numel}, num_chunks={num_chunks}",
                )
                self.assertEqual(
                    len(sizes),
                    num_chunks,
                    f"Length mismatch for numel={numel}, num_chunks={num_chunks}",
                )


class ParameterizeEnterExitContextTest(unittest.TestCase):
    """Test suite for the ParameterizeEnterExitContext class.

    This test case verifies the functionality of the ParameterizeEnterExitContext
    class, ensuring that the enter and exit methods are called correctly on the
    input object, and that the object's state is modified as expected.
    """

    def test_parameterize_enter_exit_context(self) -> None:
        """Test the enter and exit context management.

        This test creates an instance of a TestClass, which has enter and exit
        methods that modify an internal variable. It then uses the
        ParameterizeEnterExitContext to ensure that the enter method is called
        upon entering the context and the exit method is called upon exiting,
        verifying the changes in the internal state of the TestClass instance.
        """

        class TestClass:
            def __init__(self) -> None:
                self._test_var = 0

            def enter(self) -> None:
                self._test_var = 1

            def exit(self) -> None:
                self._test_var = -1

            @property
            def test_var(self) -> int:
                return self._test_var

        test_class = TestClass()
        with ParameterizeEnterExitContext(
            input_with_enter_exit_context=test_class,
            enter_method_caller=methodcaller("enter"),
            exit_method_caller=methodcaller("exit"),
        ):
            # Due to the invocation of test_class.enter(), the state of test_class.test_var should be 1.
            self.assertEqual(test_class.test_var, 1)

        # Due to the invocation of test_class.exit(), the state of test_class.test_var should be -1.
        self.assertEqual(test_class.test_var, -1)


@instantiate_parametrized_tests
class DistributeBufferSizesTest(unittest.TestCase):
    @staticmethod
    def empty_tensor_list(sizes: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.empty(size, device="meta", dtype=torch.bool) for size in sizes
        )

    @parametrize(
        "blocked_params, group_size, load_balancing_config, expected_result",
        (
            # Test case 1: Even distribution of buffer sizes
            (
                empty_tensor_list((128, 64, 500, 256)),
                2,
                LoadBalancingConfig(),
                (
                    (128, 1),
                    (64, 1),
                    (512, 0),
                    (256, 1),
                ),
            ),
            # Test case 2: Single group
            (
                empty_tensor_list((128, 64, 500, 256)),
                1,
                LoadBalancingConfig(),
                (
                    (128, 0),
                    (64, 0),
                    (512, 0),
                    (256, 0),
                ),
            ),
            # Test case 3: More groups than buffers
            (
                empty_tensor_list((128, 64)),
                4,
                LoadBalancingConfig(),
                ((128, 0), (64, 1)),
            ),
            # Test case 4: Empty buffer sizes
            ((), 2, LoadBalancingConfig(), ()),
            # Test case 5: Linear Compute Cost Model
            (
                empty_tensor_list((128, 64, 512, 256)),
                2,
                LoadBalancingConfig(
                    cost_model=PolynomialComputationalCostModel(coefficients=(0, 2))
                ),
                (
                    (128, 1),
                    (64, 1),
                    (512, 0),
                    (256, 1),
                ),
            ),
            # Test case 6: Quadratic Compute Cost Model
            # Rank 0 gets tensors of sizes 256, 256, and 128, Rank 1 gets a tensor of size 384.
            # The quadratic computational cost of Rank 0 equals that of Rank 1: 384² = 256² + 256² + 128²
            (
                empty_tensor_list((256, 128, 384, 256)),
                2,
                LoadBalancingConfig(
                    cost_model=PolynomialComputationalCostModel(coefficients=(0, 0, 1))
                ),
                (
                    (256, 1),
                    (128, 1),
                    (384, 0),
                    (256, 1),
                ),
            ),
        ),
    )
    def test_distribute_buffer_sizes(
        self,
        blocked_params: tuple[torch.Tensor, ...],
        group_size: int,
        load_balancing_config: LoadBalancingConfig,
        expected_result: tuple[tuple[int, int], ...],
    ) -> None:
        self.assertEqual(
            distribute_buffer_sizes(
                blocked_params=blocked_params,
                group_size=group_size,
                load_balancing_config=load_balancing_config,
            ),
            expected_result,
        )


@instantiate_parametrized_tests
class PackUpperTriangularTest(unittest.TestCase):
    def test_pack_2x2(self) -> None:
        # [[1, 2], [3, 4]] in row-major upper-triangular order: (0,0), (0,1), (1,1).
        torch.testing.assert_close(
            pack_upper_triangular(torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
            torch.tensor([1.0, 2.0, 4.0]),
        )

    def test_pack_3x3(self) -> None:
        # Row-major upper-triangular order for 3x3:
        # (0,0), (0,1), (0,2), (1,1), (1,2), (2,2).
        torch.testing.assert_close(
            pack_upper_triangular(
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            ),
            torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 9.0]),
        )

    def test_pack_1x1(self) -> None:
        torch.testing.assert_close(
            pack_upper_triangular(torch.tensor([[42.0]])),
            torch.tensor([42.0]),
        )

    @parametrize("dim", (1, 2, 3, 5, 8))
    def test_pack_size_invariant(self, dim: int) -> None:
        packed = pack_upper_triangular(torch.zeros(dim, dim))
        self.assertEqual(packed.shape, (dim * (dim + 1) // 2,))

    def test_pack_preserves_dtype_and_device(self) -> None:
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            packed = pack_upper_triangular(torch.zeros(3, 3, dtype=dtype))
            self.assertEqual(packed.dtype, dtype)


@instantiate_parametrized_tests
class UnpackUpperTriangularTest(unittest.TestCase):
    def test_unpack_2x2(self) -> None:
        # [1, 2, 4] reconstructs to symmetric matrix [[1, 2], [2, 4]].
        torch.testing.assert_close(
            unpack_upper_triangular(torch.tensor([1.0, 2.0, 4.0]), dim=2),
            torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
        )

    def test_unpack_3x3(self) -> None:
        torch.testing.assert_close(
            unpack_upper_triangular(
                torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 9.0]), dim=3
            ),
            torch.tensor([[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]]),
        )

    def test_unpack_returns_symmetric(self) -> None:
        unpacked = unpack_upper_triangular(
            torch.arange(1, 11, dtype=torch.float32), dim=4
        )
        torch.testing.assert_close(unpacked, unpacked.T)

    @parametrize("dim", (1, 2, 3, 5, 8))
    def test_pack_unpack_roundtrip_symmetric(self, dim: int) -> None:
        # Build a random symmetric matrix and verify pack -> unpack returns it.
        a = torch.randn(dim, dim)
        symmetric = a + a.T
        torch.testing.assert_close(
            unpack_upper_triangular(pack_upper_triangular(symmetric), dim=dim),
            symmetric,
        )

    def test_unpack_preserves_dtype_and_device(self) -> None:
        for dtype in (torch.float32, torch.float64, torch.bfloat16):
            unpacked = unpack_upper_triangular(torch.zeros(6, dtype=dtype), dim=3)
            self.assertEqual(unpacked.dtype, dtype)


class TriuIndicesCacheTest(unittest.TestCase):
    # The @cache on _get_triu_indices is process-wide; isolate every test
    # so assertions on cache_info are independent of test ordering.
    def setUp(self) -> None:
        _get_triu_indices.cache_clear()

    def tearDown(self) -> None:
        _get_triu_indices.cache_clear()

    def test_cache_reuse(self) -> None:
        # First call populates the cache; subsequent same-dim calls hit it.
        symmetric = torch.randn(8, 8)
        symmetric = symmetric + symmetric.T
        pack_upper_triangular(symmetric)
        info_after_first = _get_triu_indices.cache_info()
        self.assertEqual(info_after_first.misses, 1)
        self.assertEqual(info_after_first.hits, 0)

        for _ in range(5):
            pack_upper_triangular(symmetric)
            unpack_upper_triangular(torch.zeros(8 * 9 // 2), dim=8)
        info_after_repeat = _get_triu_indices.cache_info()
        self.assertEqual(info_after_repeat.misses, 1)
        # 5 iterations * (1 pack + 1 unpack) = 10 hits, exact.
        self.assertEqual(info_after_repeat.hits, 10)

        # A different dim should miss exactly once more.
        larger = torch.randn(16, 16)
        larger = larger + larger.T
        pack_upper_triangular(larger)
        self.assertEqual(_get_triu_indices.cache_info().misses, 2)

    def test_dtype_and_shape_and_values(self) -> None:
        # int32 halves the cache footprint vs torch.triu_indices' int64 default.
        # Lock dtype, shape, AND values against the reference torch.triu_indices.
        indices = _get_triu_indices(32, "cpu", 0)
        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(indices.shape, (2, 32 * 33 // 2))
        torch.testing.assert_close(
            indices, torch.triu_indices(32, 32, device="cpu").to(torch.int32)
        )

    def test_device_key_normalizes_cpu(self) -> None:
        # cpu always normalizes to ('cpu', 0).
        self.assertEqual(_device_key(torch.device("cpu")), ("cpu", 0))

    @unittest.skipUnless(_HAS_CUDA, "CUDA required")
    def test_device_key_normalizes_cuda(self) -> None:
        # cuda without explicit index must collapse to the current device,
        # so str-form aliases ('cuda' vs 'cuda:0') don't double-cache.
        self.assertEqual(
            _device_key(torch.device("cuda")),
            _device_key(torch.device("cuda", torch.cuda.current_device())),
        )

    @unittest.skipUnless(_HAS_CUDA, "CUDA required")
    def test_cuda_roundtrip(self) -> None:
        symmetric = torch.randn(16, 16, device="cuda")
        symmetric = symmetric + symmetric.T
        packed = pack_upper_triangular(symmetric)
        self.assertEqual(packed.device.type, "cuda")
        unpacked = unpack_upper_triangular(packed, dim=16)
        torch.testing.assert_close(unpacked, symmetric)


@instantiate_parametrized_tests
class SplitParamGroupsTest(unittest.TestCase):
    @staticmethod
    def _make_group(
        num_params: int,
        distributed_config: object,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        params = [torch.zeros(1) for _ in range(num_params)]
        group: dict[str, object] = {
            PARAMS: params,
            DISTRIBUTED_CONFIG: distributed_config,
        }
        if extra is not None:
            group.update(extra)
        return group

    @staticmethod
    def _make_fsdp_config(num_sub_groups: int) -> FullyShardDistributedConfig:
        return FullyShardDistributedConfig(
            param_assignment_strategy=FSDPParamAssignmentStrategy.ROUND_ROBIN,
            num_sub_groups=num_sub_groups,
        )

    def test_passthrough_when_num_sub_groups_is_1(self) -> None:
        config = self._make_fsdp_config(num_sub_groups=1)
        group = self._make_group(num_params=4, distributed_config=config)
        result = split_param_groups([group])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], group)

    def test_passthrough_for_non_fsdp_config(self) -> None:
        # SingleDeviceDistributedConfig has no num_sub_groups field; should pass through.
        group = self._make_group(
            num_params=4, distributed_config=SingleDeviceDistributedConfig()
        )
        result = split_param_groups([group])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], group)

    @parametrize(
        "num_params, num_sub_groups, expected_sizes",
        (
            (6, 3, [2, 2, 2]),
            (5, 4, [2, 1, 1, 1]),
        ),
    )
    def test_split_distribution(
        self, num_params: int, num_sub_groups: int, expected_sizes: list[int]
    ) -> None:
        config = self._make_fsdp_config(num_sub_groups=num_sub_groups)
        group = self._make_group(num_params=num_params, distributed_config=config)
        result = split_param_groups([group])
        self.assertEqual(len(result), num_sub_groups)
        self.assertEqual([len(g[PARAMS]) for g in result], expected_sizes)

    def test_preserves_other_group_keys(self) -> None:
        config = self._make_fsdp_config(num_sub_groups=2)
        group = self._make_group(
            num_params=4, distributed_config=config, extra={"lr": 0.01, "beta3": 0.9}
        )
        result = split_param_groups([group])
        for sub_group in result:
            self.assertEqual(sub_group["lr"], 0.01)
            self.assertEqual(sub_group["beta3"], 0.9)

    def test_each_sub_group_gets_distinct_config_copy(self) -> None:
        config = self._make_fsdp_config(num_sub_groups=2)
        group = self._make_group(num_params=4, distributed_config=config)
        result = split_param_groups([group])
        self.assertIsNot(result[0][DISTRIBUTED_CONFIG], result[1][DISTRIBUTED_CONFIG])
        self.assertIsNot(result[0][DISTRIBUTED_CONFIG], config)

    def test_raises_when_num_sub_groups_exceeds_num_params(self) -> None:
        config = self._make_fsdp_config(num_sub_groups=5)
        group = self._make_group(num_params=3, distributed_config=config)
        with self.assertRaisesRegex(ValueError, "num_sub_groups=5 is too large"):
            split_param_groups([group])

    def test_splits_each_input_group_independently(self) -> None:
        config = self._make_fsdp_config(num_sub_groups=2)
        group_a = self._make_group(num_params=4, distributed_config=config)
        group_b = self._make_group(
            num_params=2, distributed_config=SingleDeviceDistributedConfig()
        )
        result = split_param_groups([group_a, group_b])
        # group_a -> 2 sub-groups; group_b passes through unchanged.
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0][PARAMS]), 2)
        self.assertEqual(len(result[1][PARAMS]), 2)
        self.assertIs(result[2], group_b)

    def test_hsdp_raises_when_dominant_param_starves_a_bin(self) -> None:
        """HSDP: a single huge param can be alone in a bin, leaving fewer
        than shard_size params there. Greedy bin-packing must detect and reject."""
        # 6 params split into 2 bins with shard_size=3. Validation passes
        # (6 // 3 = 2 >= 2). One param has dominant numel; greedy puts it
        # alone in a bin → 1 < shard_size, violation.
        params = [torch.zeros(1000)] + [torch.zeros(1) for _ in range(5)]
        # Avoid real device mesh init (needs distributed setup).
        mock_mesh = MagicMock()
        mock_mesh.size.return_value = 3
        config = HybridShardDistributedConfig(
            device_mesh=mock_mesh,
            param_assignment_strategy=FSDPParamAssignmentStrategy.ROUND_ROBIN,
            num_sub_groups=2,
        )
        group: dict[str, object] = {PARAMS: params, DISTRIBUTED_CONFIG: config}
        with self.assertRaisesRegex(ValueError, "fewer than shard_size=3"):
            split_param_groups([group])


class GreedyBinPackTest(unittest.TestCase):
    def test_skewed_sizes(self) -> None:
        """Largest item should be alone; three small items in the other bin."""
        items = [1000, 100, 100, 100]
        bins, costs = greedy_bin_pack(items, num_bins=2, cost_fn=lambda x: x)
        # Bin 0 gets the largest item first, bin 1 gets the rest.
        self.assertEqual(sorted(costs), [300, 1000])
        self.assertEqual(sum(len(b) for b in bins), 4)

    def test_more_bins_than_items(self) -> None:
        """Empty bins should be present when num_bins > len(items)."""
        items = ["a", "b"]
        bins, costs = greedy_bin_pack(items, num_bins=4, cost_fn=lambda x: 1)
        non_empty = [b for b in bins if b]
        self.assertEqual(len(non_empty), 2)
        self.assertEqual(len(bins), 4)

    def test_single_bin(self) -> None:
        """All items should end up in one bin."""
        items = [10, 20, 30]
        bins, costs = greedy_bin_pack(items, num_bins=1, cost_fn=lambda x: x)
        self.assertEqual(len(bins), 1)
        self.assertEqual(costs, [60])
        self.assertEqual(sorted(bins[0]), [10, 20, 30])

    def test_equal_sizes(self) -> None:
        """Equal-cost items should be distributed round-robin across bins."""
        items = list(range(6))
        bins, costs = greedy_bin_pack(items, num_bins=3, cost_fn=lambda x: 1)
        # Each bin should get exactly 2 items.
        for b in bins:
            self.assertEqual(len(b), 2)
        self.assertEqual(costs, [2, 2, 2])

    def test_empty_items(self) -> None:
        """No items should produce all empty bins."""
        empty_items: list[int] = []
        bins, costs = greedy_bin_pack(empty_items, num_bins=3, cost_fn=lambda x: x)
        self.assertEqual(bins, [[], [], []])
        self.assertEqual(costs, [0, 0, 0])
