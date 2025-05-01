import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, cast, Mapping, Sequence, TypeVar

import torch
from distributed_shampoo.utils.shampoo_block_info import BlockInfo

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed_shampoo.shampoo_types import (
    EigenvalueCorrectedShampooPreconditionerConfig,
    ShampooPreconditionerConfig,
)
from distributed_shampoo.utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    EigendecomposedShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    PreconditionerList,
    RootInvShampooPreconditionerList,
    SGDPreconditionerList,
)

T = TypeVar("T")

# Set logger config
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Silence the instantiation logs from the preconditioner list module
logging.getLogger("distributed_shampoo.utils.shampoo_preconditioner_list").setLevel(
    logging.WARNING
)


class MockBlockInfo:
    """Mock BlockInfo for benchmarking"""

    def __init__(self, param, param_index, block_index, device="cuda"):
        self.param = param
        self.composable_block_ids = (param_index, block_index)
        self.device = device

    def allocate_zeros_tensor(self, size, dtype, device):
        return torch.zeros(size, dtype=dtype, device=device)

    def allocate_ones_tensor(self, size, dtype, device):
        return torch.ones(size, dtype=dtype, device=device)

    def allocate_eye_tensor(self, n, dtype, device):
        return torch.eye(n, dtype=dtype, device=device)

    def get_tensor(self, tensor):
        return tensor


@dataclass
class PreconditionerConfig:
    """Common configuration for all preconditioners."""

    beta2: float = 1.0
    epsilon: float = 1e-12
    use_bias_correction: bool = True
    factor_matrix_dtype: torch.dtype = torch.float


class PreconditionerFactory:
    """Factory for creating preconditioners."""

    @staticmethod
    def create(
        preconditioner_type: str,
        block_list: tuple[torch.Tensor, ...],
        block_info_list: tuple[BlockInfo, ...],
        state: Mapping[torch.Tensor, Any],
        config: PreconditionerConfig,
    ) -> PreconditionerList:
        if preconditioner_type == "SGD":
            return SGDPreconditionerList(block_list=block_list)

        if preconditioner_type == "AdaGrad":
            return AdagradPreconditionerList(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                beta2=config.beta2,
                epsilon=config.epsilon,
                use_bias_correction=False,
            )

        if preconditioner_type == "Shampoo":
            return RootInvShampooPreconditionerList(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                preconditioner_config=ShampooPreconditionerConfig(),
                beta2=config.beta2,
                epsilon=config.epsilon,
                use_bias_correction=config.use_bias_correction,
                factor_matrix_dtype=config.factor_matrix_dtype,
            )

        if preconditioner_type == "EigendecomposedShampoo":
            from matrix_functions_types import QREigendecompositionConfig

            eigen_config = ShampooPreconditionerConfig(
                amortized_computation_config=QREigendecompositionConfig(),
            )
            return EigendecomposedShampooPreconditionerList(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                preconditioner_config=eigen_config,
                beta2=config.beta2,
                epsilon=config.epsilon,
                use_bias_correction=config.use_bias_correction,
                factor_matrix_dtype=config.factor_matrix_dtype,
            )

        if preconditioner_type == "EigenvalueCorrectedShampoo":
            from matrix_functions_types import QREigendecompositionConfig

            evc_config = EigenvalueCorrectedShampooPreconditionerConfig(
                amortized_computation_config=QREigendecompositionConfig(),
            )
            return EigenvalueCorrectedShampooPreconditionerList(
                block_list=block_list,
                state=state,
                block_info_list=block_info_list,
                preconditioner_config=evc_config,
                beta2=config.beta2,
                epsilon=config.epsilon,
                use_bias_correction=config.use_bias_correction,
                factor_matrix_dtype=config.factor_matrix_dtype,
            )

        raise ValueError(f"Unknown preconditioner type: {preconditioner_type}")


class PreconditionerBenchmark:
    """Benchmark different preconditioners."""

    def __init__(
        self,
        param_shapes: Sequence[tuple[int, ...]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.param_shapes = param_shapes
        self.device = device
        self.state: dict[torch.Tensor, Any] = {}
        self.blocks: list[torch.Tensor] = []
        self.block_infos: list[MockBlockInfo] = []
        self.preconditioner_config = PreconditionerConfig()

        # Prepare parameters and blocks
        for i, shape in enumerate(param_shapes):
            param = torch.nn.Parameter(torch.randn(shape, device=device))
            self.state[param] = {}

            # Create blocks (using entire tensor as one block for simplicity)
            self.blocks.append(param.data)
            self.block_infos.append(MockBlockInfo(param, i, 0, device))

    def create_preconditioner(self, preconditioner_type: str) -> PreconditionerList:
        """Create a preconditioner of the specified type."""
        return PreconditionerFactory.create(
            preconditioner_type,
            tuple(self.blocks),
            cast(tuple[BlockInfo, ...], tuple(self.block_infos)),
            self.state,
            self.preconditioner_config,
        )

    def benchmark_preconditioner(
        self,
        preconditioner: PreconditionerList,
        preconditioner_type: str,
        num_epochs: int = 100,
        precondition_frequency: int = 5,
    ) -> dict[str, Any]:
        """Run benchmark for a preconditioner"""
        logger.info(f"Starting benchmark for {preconditioner_type}")

        results: dict[str, Any] = {
            "preconditioner": preconditioner_type,
            "epochs": [],
            "total_time": 0.0,
            "avg_time_per_epoch": 0.0,
            "memory_usage": 0,
        }

        # Measure memory usage
        if hasattr(preconditioner, "num_bytes"):
            results["memory_usage"] = preconditioner.num_bytes()

        total_start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            step = torch.tensor(epoch + 1, dtype=torch.int64, device=self.device)

            # Generate dummy gradients (scaled by 0.01 to simulate realistic scenario)
            grad_list = [torch.randn_like(block) * 0.01 for block in self.blocks]

            # Update preconditioner
            perform_amortized = (epoch + 1) % precondition_frequency == 0
            preconditioner.update_preconditioners(
                masked_grad_list=tuple(grad_list),
                step=step,
                perform_amortized_computation=perform_amortized,
            )

            epoch_time = time.time() - epoch_start_time

            # Log output every 40 epochs
            if (epoch + 1) % 40 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - {preconditioner_type} | "
                    f"Time: {epoch_time:.4f}s | "
                    f"Amortized: {perform_amortized}"
                )

            results["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "time": epoch_time,
                    "amortized": perform_amortized,
                }
            )

        total_time = time.time() - total_start_time
        results["total_time"] = total_time
        results["avg_time_per_epoch"] = total_time / num_epochs

        logger.info(
            f"Benchmark completed for {preconditioner_type} | "
            f"Total time: {total_time:.2f}s | "
            f"Avg time per epoch: {total_time/num_epochs:.4f}s | "
            f"Memory usage: {results['memory_usage']/1024/1024:.2f} MB"
        )
        logger.info("-" * 60)

        return results

    def run_all_benchmarks(self, num_epochs: int = 100) -> dict[str, Any]:
        """Run benchmarks for all preconditioners"""
        preconditioner_types = [
            "SGD",
            "AdaGrad",
            "Shampoo",
            "EigendecomposedShampoo",
            "EigenvalueCorrectedShampoo",
        ]

        all_results = {}

        for preconditioner_type in preconditioner_types:
            preconditioner = self.create_preconditioner(preconditioner_type)
            result = self.benchmark_preconditioner(
                preconditioner, preconditioner_type, num_epochs=num_epochs
            )
            all_results[preconditioner_type] = result

        # Print summary
        logger.info("=" * 60)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 60)
        for preconditioner_type, result in all_results.items():
            logger.info(
                f"{preconditioner_type}: "
                f"Time={result['total_time']:.2f}s, "
                f"Avg={result['avg_time_per_epoch']:.4f}s, "
                f"Memory={result['memory_usage']/1024/1024:.2f}MB"
            )

        return all_results


if __name__ == "__main__":
    # Run benchmark example
    param_shapes = [
        (512, 512),  # Medium matrix
        (256, 256),  # Small matrix
        (128, 256),  # Rectangular matrix
    ]

    benchmark = PreconditionerBenchmark(param_shapes)
    results = benchmark.run_all_benchmarks(num_epochs=200)
