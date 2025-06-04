"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import time
from dataclasses import dataclass
from typing import Optional

import torch

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
from matrix_functions_types import QREigendecompositionConfig

# Note: This is a workaround for mypy not recognizing the rich library
# Since only benchmarks requires it, we can ignore the errors here
# If you want to fundamentally change this, you can use a stub file (i.e., .pyi, .ini)
from rich.console import Console  # type: ignore
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn  # type: ignore
from rich.table import Table  # type: ignore
from torch.profiler import profile, ProfilerActivity


@dataclass
class BenchmarkResult:
    """Container for benchmark results.

    Attributes:
        preconditioner: Type of the preconditioner. (e.g., Root Inverse Shampoo, Eigendecomposed Shampoo)
        total_time: Total time taken to run all epochs (in seconds).
        avg_time_per_epoch: Average time per epoch (in seconds).
        memory_usage: Memory usage in bytes.
        gpu_utilization: GPU utilization percentage (only for CUDA devices).
    """

    preconditioner: str
    total_time: float
    avg_time_per_epoch: float
    memory_usage: float
    gpu_utilization: Optional[float] = None
    profiling_table: Optional[str] = None


class PreconditionerBenchmark:
    """Benchmark different preconditioners.

    This class provides utilities to benchmark various preconditioners
    for optimization algorithms. It measures performance metrics like
    execution time, memory usage, and GPU utilization.
    """

    # Mapping preconditioner names
    PRECONDITIONER_TYPES = {
        "SGD": SGDPreconditionerList,
        "AdaGrad": AdagradPreconditionerList,
        "Shampoo": RootInvShampooPreconditionerList,
        "EigendecomposedShampoo": EigendecomposedShampooPreconditionerList,
        "EigenvalueCorrectedShampoo": EigenvalueCorrectedShampooPreconditionerList,
    }

    def __init__(self, param_shapes: list[tuple[int, ...]], device: str):
        """Initialize benchmark with parameter shapes and device."""
        self.param_shapes = param_shapes
        self.console = Console()
        self._setup_parameters()

    def _setup_parameters(self):
        """Initialize parameters, blocks, and block_infos.

        Sets up the following instance attributes:
        - self.state: Dictionary mapping parameters to their states
        - self.blocks: List of parameter tensors
        - self.block_infos: List of BlockInfo objects
        """
        self.state: dict[torch.Tensor] = {}
        self.blocks: list[torch.Tensor] = []
        self.block_infos: list[BlockInfo] = []

        for i, shape in enumerate(self.param_shapes):
            param = torch.nn.Parameter(torch.randn(shape, device="cuda"))
            self.state[param] = {}
            self.blocks.append(param.data)
            self.block_infos.append(
                BlockInfo(param=param, composable_block_ids=(i, "block_0"))
            )

    def _get_preconditioner_config(
        self, preconditioner_type: str
    ) -> ShampooPreconditionerConfig | EigenvalueCorrectedShampooPreconditionerConfig:
        """Get configuration for specific preconditioner types.

        Args:
            preconditioner_type: The type of preconditioner to configure.

        Returns:
            A configuration object appropriate for the specified preconditioner type.
        """
        eigen_config = QREigendecompositionConfig()

        if preconditioner_type == "EigendecomposedShampoo":
            return ShampooPreconditionerConfig(
                amortized_computation_config=eigen_config
            )
        elif preconditioner_type == "EigenvalueCorrectedShampoo":
            return EigenvalueCorrectedShampooPreconditionerConfig(
                amortized_computation_config=eigen_config
            )
        return ShampooPreconditionerConfig()

    def _create_preconditioner(self, preconditioner_type: str) -> PreconditionerList:
        """Create a preconditioner of the specified type.

        Args:
            preconditioner_type: Name of the preconditioner type to create.

        Returns:
            An initialized preconditioner instance.
        """
        preconditioner_class = self.PRECONDITIONER_TYPES[preconditioner_type]

        # SGD only needs block_list (no preconditioner config)
        if preconditioner_type == "SGD":
            return preconditioner_class(block_list=tuple(self.blocks))

        # Common kwargs for other preconditioners
        kwargs = {
            "block_list": tuple(self.blocks),
            "state": self.state,
            "block_info_list": tuple(self.block_infos),
            "beta2": 0.999,
            "epsilon": 1e-8,
            "use_bias_correction": preconditioner_type != "AdaGrad",
        }
        if preconditioner_type in [
            "Shampoo",
            "EigendecomposedShampoo",
            "EigenvalueCorrectedShampoo",
        ]:
            kwargs.update(
                {
                    "preconditioner_config": self._get_preconditioner_config(
                        preconditioner_type
                    )
                }
            )

        return preconditioner_class(**kwargs)

    def _benchmark_single(
        self, preconditioner_type: str, num_epochs=200
    ) -> BenchmarkResult:
        """Benchmark a single preconditioner with profiler."""
        preconditioner = self._create_preconditioner(preconditioner_type)
        gradients = self._generate_gradients(num_epochs)
        memory_usage = getattr(preconditioner, "num_bytes", lambda: 0)()

        torch.cuda.empty_cache()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            start_time = time.time()

            # Show progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"[green]Benchmarking {preconditioner_type}...", total=num_epochs
                )

                for epoch in range(num_epochs):
                    step = torch.tensor(epoch + 1, dtype=torch.float, device="cuda")
                    # Perform amortized computation per the frequency epochs (40)
                    perform_amortized = (epoch + 1) % 40 == 0

                    preconditioner.update_preconditioners(
                        masked_grad_list=tuple(gradients[epoch]),
                        step=step,
                        perform_amortized_computation=perform_amortized,
                    )
                    progress.update(task, advance=1)

            total_time = time.time() - start_time

        # Extract profiling results
        key_averages = prof.key_averages()
        top_ops = sorted(
            key_averages, key=lambda x: getattr(x, "cuda_time_total", 0), reverse=True
        )[:8]

        # Calculate total CPU and GPU times
        total_cpu_time = sum(x.cpu_time_total for x in key_averages)
        total_device_time = sum(x.device_time_total for x in key_averages)

        lines = [
            "Name                               Self CPU %  CPU Time  Self GPU %  GPU Time  Calls",
            "-" * 85,
        ]

        # Format the top operations
        for op in top_ops:
            name = op.key[:35].ljust(35)
            self_cpu_pct = (
                f"{op.self_cpu_time_total / total_cpu_time * 100:.1f}%".rjust(9)
            )
            cpu_time = f"{op.cpu_time_total / 1000:.1f}ms".rjust(9)
            self_gpu_pct = (
                f"{op.self_device_time_total / total_device_time * 100:.1f}%".rjust(9)
            )
            gpu_time = f"{op.device_time_total / 1000:.1f}ms".rjust(8)
            calls = f"{op.count}".rjust(9)

            lines.append(
                f"{name} {self_cpu_pct} {cpu_time} {self_gpu_pct} {gpu_time} {calls}"
            )

        top_ops = "\n".join(lines)
        gpu_utilization = (
            (total_device_time / (total_time * 1e6)) * 100 if total_time > 0 else 0
        )

        return BenchmarkResult(
            preconditioner=preconditioner_type,
            total_time=total_time,
            avg_time_per_epoch=total_time / num_epochs,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            profiling_table=top_ops,
        )

    def _generate_gradients(self, num_epochs: int) -> list[list[torch.Tensor]]:
        """Pre-generate gradients for benchmarking.

        This method pre-generates random gradients for all epochs to ensure
        fair comparison between different preconditioners.

        Args:
            num_epochs: Number of epochs to generate gradients for.

        Returns:
            List of gradient lists, one per epoch.
        """
        gradients = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Generating gradients...", total=num_epochs)
            for _ in range(num_epochs):
                gradients.append(
                    [torch.randn_like(block) * 0.01 for block in self.blocks]
                )
                progress.update(task, advance=1)
        return gradients

    def _display_results(self, results: dict[str, BenchmarkResult]):
        """Display benchmark results in formatted tables.

        This method creates and displays:
        1. A performance summary table
        2. Bottleneck analysis for each preconditioner
        3. A quick comparison of relative performance

        Args:
            results: Dictionary mapping preconditioner names to BenchmarkResult objects.
        """
        # Main performance table
        table = Table(title="Performance Summary")
        table.add_column("Preconditioner", style="cyan", justify="left")
        table.add_column("Total time (s)", justify="right", style="magenta")
        table.add_column("Avg/Epoch (ms)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="green")
        table.add_column("Relative", justify="right", style="red")
        table.add_column("GPU Util %", justify="right", style="blue")

        sgd_time = results["SGD"].avg_time_per_epoch

        for preconditioner_type, result in results.items():
            row = [
                preconditioner_type,
                f"{result.total_time:.1f}",
                f"{result.avg_time_per_epoch * 1000:.1f}",
                f"{result.memory_usage / 1024 / 1024:.1f}",
                f"{result.avg_time_per_epoch / sgd_time:.1f}x",
                f"{result.gpu_utilization or 0:.0f}",
            ]

            table.add_row(*row)

        self.console.print(table)

        # Show bottleneck analysis in rich Console
        self.console.print("\n[bold]Bottleneck Analysis[/bold]")
        for preconditioner_type, result in results.items():
            if result.profiling_table:
                self.console.print(f"\n[cyan]{preconditioner_type}[/cyan]:")
                lines = result.profiling_table.split("\n")
                for line in lines:
                    self.console.print(f"  {line}")

        comparisons = []
        for preconditioner_type, result in results.items():
            ratio = result.avg_time_per_epoch / sgd_time
            comparisons.append(f"{preconditioner_type}: {ratio:.1f}x")

        self.console.print(f"Total time relative to SGD: {' | '.join(comparisons)}")

    def run_all_benchmarks(self, num_epochs: int = 200) -> dict[str, BenchmarkResult]:
        """Run benchmarks for all preconditioners.

        Args:
            num_epochs: Number of epochs to run for each benchmark.

        Returns:
            Dictionary mapping preconditioner names to their benchmark results.
        """
        self.console.print("\n[bold cyan]Preconditioner Benchmark[/bold cyan]")
        self.console.print(
            f"[dim]Device: cuda  | Shape: {self.param_shapes[0]} | Epochs: {num_epochs}[/dim]\n"
        )

        results = {}
        for preconditioner_type in self.PRECONDITIONER_TYPES:
            result = self._benchmark_single(preconditioner_type, num_epochs)
            results[preconditioner_type] = result

        self._display_results(results)
        return results


if __name__ == "__main__":
    # Run benchmark with a single 2048x2048 parameter matrix
    benchmark = PreconditionerBenchmark([(2048, 2048)], device="cuda")
    benchmark.run_all_benchmarks(num_epochs=200)
