import logging
import os
import sys
import time
from typing import Any

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed_shampoo.shampoo_types import (
    EigenvalueCorrectedShampooPreconditionerConfig,
    ShampooPreconditionerConfig,
)
from distributed_shampoo.utils.shampoo_block_info import BlockInfo
from distributed_shampoo.utils.shampoo_preconditioner_list import (
    AdagradPreconditionerList,
    EigendecomposedShampooPreconditionerList,
    EigenvalueCorrectedShampooPreconditionerList,
    PreconditionerList,
    RootInvShampooPreconditionerList,
    SGDPreconditionerList,
)

# Set logger config
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Silence the instantiation logs from the preconditioner list module
logging.getLogger("distributed_shampoo.utils.shampoo_preconditioner_list").setLevel(
    logging.WARNING
)


class MockBlockInfo(BlockInfo):
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


class PreconditionerBenchmark:
    def __init__(
        self,
        param_shapes: list[tuple],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.param_shapes = param_shapes
        self.device = device
        self.state: dict[torch.Tensor, Any] = {}  # Optimizer state
        self.blocks: list[torch.Tensor] = []
        self.block_infos: list[MockBlockInfo] = []

        # Prepare parameters and blocks
        for i, shape in enumerate(param_shapes):
            param = torch.nn.Parameter(torch.randn(shape, device=device))
            self.state[param] = {}

            # Create blocks (using entire tensor as one block for simplicity)
            block = param.data
            block_info = MockBlockInfo(param, i, 0, device)

            self.blocks.append(block)
            self.block_infos.append(block_info)

    def create_preconditioner(self, preconditioner_type: str) -> PreconditionerList:
        """Create various preconditioners"""
        block_list = tuple(self.blocks)
        block_info_list = tuple(self.block_infos)

        if preconditioner_type == "SGD":
            return SGDPreconditionerList(block_list=block_list)

        elif preconditioner_type == "AdaGrad":
            return AdagradPreconditionerList(
                block_list=block_list,
                state=self.state,
                block_info_list=block_info_list,
                beta2=1.0,
                epsilon=1e-12,
                use_bias_correction=False,
            )

        elif preconditioner_type == "Shampoo":
            config = ShampooPreconditionerConfig()
            return RootInvShampooPreconditionerList(
                block_list=block_list,
                state=self.state,
                block_info_list=block_info_list,
                preconditioner_config=config,
                beta2=1.0,
                epsilon=1e-12,
                use_bias_correction=True,
                factor_matrix_dtype=torch.float,
            )

        elif preconditioner_type == "EigendecomposedShampoo":
            from matrix_functions_types import QREigendecompositionConfig

            config = ShampooPreconditionerConfig(
                amortized_computation_config=QREigendecompositionConfig(),
            )
            return EigendecomposedShampooPreconditionerList(
                block_list=block_list,
                state=self.state,
                block_info_list=block_info_list,
                preconditioner_config=config,
                beta2=1.0,
                epsilon=1e-12,
                use_bias_correction=True,
                factor_matrix_dtype=torch.float,
            )

        elif preconditioner_type == "EigenvalueCorrectedShampoo":
            from matrix_functions_types import QREigendecompositionConfig

            eigen_config: EigenvalueCorrectedShampooPreconditionerConfig = (
                EigenvalueCorrectedShampooPreconditionerConfig(
                    amortized_computation_config=QREigendecompositionConfig(),
                )
            )
            return EigenvalueCorrectedShampooPreconditionerList(
                block_list=block_list,
                state=self.state,
                block_info_list=block_info_list,
                preconditioner_config=eigen_config,
                beta2=1.0,
                epsilon=1e-12,
                use_bias_correction=True,
                factor_matrix_dtype=torch.float,
            )

        else:
            raise ValueError(f"Unknown preconditioner type: {preconditioner_type}")

    def benchmark_preconditioner(
        self,
        preconditioner: PreconditionerList,
        preconditioner_type: str,
        num_epochs: int = 300,
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

            # Generate dummy gradients
            grad_list = []
            for block in self.blocks:
                grad = (
                    torch.randn_like(block) * 0.01
                )  # Scale gradients to simulate realistic scenario
                grad_list.append(grad)

            # Update preconditioner
            perform_amortized = (epoch + 1) % precondition_frequency == 0
            preconditioner.update_preconditioners(
                masked_grad_list=tuple(grad_list),
                step=step,
                perform_amortized_computation=perform_amortized,
            )

            epoch_time = time.time() - epoch_start_time

            # Log output every 200 epochs
            if (epoch + 1) % 200 == 0 or epoch == 0:
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

    def run_all_benchmarks(self, num_epochs: int = 1000) -> dict[str, Any]:
        """Run benchmarks for all preconditioners"""
        preconditioner_types: list[str] = [
            "SGD",
            "AdaGrad",
            "Shampoo",
            "EigendecomposedShampoo",
            "EigenvalueCorrectedShampoo",
        ]

        all_results: dict[str, dict[str, Any]] = {}

        for preconditioner_type in preconditioner_types:
            try:
                preconditioner = self.create_preconditioner(preconditioner_type)
                result = self.benchmark_preconditioner(
                    preconditioner, preconditioner_type, num_epochs=num_epochs
                )
                all_results[preconditioner_type] = result
            except Exception as e:
                logger.error(f"Failed to benchmark {preconditioner_type}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                continue

        # Print summary
        logger.info("[ BENCHMARK SUMMARY] ")
        for preconditioner_type, result in all_results.items():
            logger.info(
                f"{preconditioner_type}: "
                f"Time={result['total_time']:.2f}s, "
                f"Avg={result['avg_time_per_epoch']:.4f}s, "
                f"Memory={result['memory_usage']/1024/1024:.2f}MB"
            )

        return all_results


# Run benchmark example
if __name__ == "__main__":
    param_shapes = [
        (512, 512),  # Medium matrix
        (256, 256),  # Small matrix
        (128, 256),  # Rectangular matrix
    ]

    benchmark = PreconditionerBenchmark(param_shapes)
    results = benchmark.run_all_benchmarks(num_epochs=1000)
