import time
from dataclasses import dataclass
from typing import Any, Optional

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
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
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
        top_operations: List of tuples with (operation_name, time_ms) for most time-consuming operations.
    """

    preconditioner: str
    total_time: float
    avg_time_per_epoch: float
    memory_usage: float
    gpu_utilization: Optional[float] = None
    top_operations: Optional[list[tuple[str, float]]] = None


class PreconditionerBenchmark:
    """Benchmark different preconditioners.

    This class provides utilities to benchmark various preconditioners
    for optimization algorithms. It measures performance metrics like
    execution time, memory usage, and GPU utilization.
    """

    # Mapping of preconditioner names to their respective implementation classes
    PRECONDITIONER_TYPES = {
        "SGD": SGDPreconditionerList,
        "AdaGrad": AdagradPreconditionerList,
        "Shampoo": RootInvShampooPreconditionerList,
        "EigendecomposedShampoo": EigendecomposedShampooPreconditionerList,
        "EigenvalueCorrectedShampoo": EigenvalueCorrectedShampooPreconditionerList,
    }

    def __init__(self, param_shapes: list[tuple[int, ...]], device: str):
        """Initialize benchmark with parameter shapes and device.

        Args:
            param_shapes: List of tensor shapes to benchmark.
            device: Device to run benchmark on. If None, automatically selects
                   "cuda" if available, otherwise "cpu".
        """
        self.param_shapes = param_shapes
        self.device = device
        self.console = Console()
        self.config = self._get_default_config()
        self._setup_parameters()

    def _get_default_config(self):
        """Create default configuration for preconditioners.

        Returns:
            A config object with default settings for the benchmark.
        """
        return type(
            "Config",
            (),
            {
                "beta2": 1.0,
                "epsilon": 1e-12,
                "use_bias_correction": True,
                "factor_matrix_dtype": torch.float,
            },
        )()

    def _setup_parameters(self):
        """Initialize parameters, blocks, and block_infos.

        Sets up the following instance attributes:
        - self.state: Dictionary mapping parameters to their states
        - self.blocks: List of parameter tensors
        - self.block_infos: List of BlockInfo objects
        """
        self.state: dict[torch.Tensor, Any] = {}
        self.blocks: list[torch.Tensor] = []
        self.block_infos: list[BlockInfo] = []

        for i, shape in enumerate(self.param_shapes):
            param = torch.nn.Parameter(torch.randn(shape, device=self.device))
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

        # SGD only needs block_list
        if preconditioner_type == "SGD":
            return preconditioner_class(block_list=tuple(self.blocks))

        # Common kwargs for other preconditioners
        kwargs = {
            "block_list": tuple(self.blocks),
            "state": self.state,
            "block_info_list": tuple(self.block_infos),
            "beta2": self.config.beta2,
            "epsilon": self.config.epsilon,
        }

        # AdaGrad doesn't use bias correction
        if preconditioner_type == "AdaGrad":
            kwargs["use_bias_correction"] = False
        else:
            kwargs["use_bias_correction"] = self.config.use_bias_correction

        # Shampoo variants need additional config
        if preconditioner_type in [
            "Shampoo",
            "EigendecomposedShampoo",
            "EigenvalueCorrectedShampoo",
        ]:
            kwargs.update(
                {
                    "preconditioner_config": self._get_preconditioner_config(
                        preconditioner_type
                    ),
                    "factor_matrix_dtype": self.config.factor_matrix_dtype,
                }
            )

        return preconditioner_class(**kwargs)

    def _benchmark_single(
        self,
        preconditioner_type: str,
        num_epochs: int = 200,
        precondition_frequency: int = 40,
    ) -> BenchmarkResult:
        """Benchmark a single preconditioner with profiler."""
        preconditioner = self._create_preconditioner(preconditioner_type)
        gradients = self._generate_gradients(num_epochs)
        memory_usage = getattr(preconditioner, "num_bytes", lambda: 0)()

        # Configure profiler
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        # Build PyTorch profiler
        with profile(
            activities=activities,
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
                    step = torch.tensor(
                        epoch + 1, dtype=torch.int64, device=self.device
                    )
                    grad_list = gradients[epoch]

                    # Perform amortized computation per the frequency epochs (40)
                    perform_amortized = (epoch + 1) % precondition_frequency == 0

                    preconditioner.update_preconditioners(
                        masked_grad_list=tuple(grad_list),
                        step=step,
                        perform_amortized_computation=perform_amortized,
                    )
                    progress.update(task, advance=1)

            total_time = time.time() - start_time

        # Extract profiling results
        key_averages = prof.key_averages()

        # Select time attribute based on device
        time_attr = "device_time_total" if self.device == "cuda" else "cpu_time_total"

        # Sort key_averages by time
        sorted_events = sorted(
            key_averages, key=lambda x: getattr(x, time_attr, 0), reverse=True
        )

        top_ops: list[tuple[str, float]] = []
        for event in sorted_events:
            if len(top_ops) >= 5:
                break
            if "randn" not in event.key and "normal" not in event.key:
                time_ms = getattr(event, time_attr, 0) / 1000
                top_ops.append((event.key, time_ms))

        # Calculate GPU-usage if device is CUDA
        gpu_utilization = None
        if self.device == "cuda":
            device_time = sum(getattr(event, time_attr, 0) for event in key_averages)
            total_time_us = total_time * 1e6
            gpu_utilization = (
                (device_time / total_time_us) * 100 if total_time_us > 0 else 0
            )

        return BenchmarkResult(
            preconditioner=preconditioner_type,
            total_time=total_time,
            avg_time_per_epoch=total_time / num_epochs,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            top_operations=top_ops,
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
                grad_list = [torch.randn_like(block) * 0.01 for block in self.blocks]
                gradients.append(grad_list)
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
        table.add_column("Preconditioner", style="cyan", no_wrap=True)
        table.add_column("Total (s)", justify="right", style="magenta")
        table.add_column("Avg/Epoch (ms)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="green")
        table.add_column("Relative", justify="right", style="red")

        if self.device == "cuda":
            table.add_column("GPU Util %", justify="right", style="blue")

        sgd_time = results["SGD"].avg_time_per_epoch

        for preconditioner_type, result in results.items():
            relative_time = result.avg_time_per_epoch / sgd_time
            row = [
                preconditioner_type,
                f"{result.total_time:.3f}",
                f"{result.avg_time_per_epoch * 1000:.2f}",
                f"{result.memory_usage / 1024 / 1024:.1f}",
                f"{relative_time:.1f}x",
            ]

            if self.device == "cuda":
                gpu_util = result.gpu_utilization or 0
                row.append(f"{gpu_util:.1f}")

            table.add_row(*row)

        self.console.print(table)

        # Show bottleneck analysis
        self.console.print("\n[bold]Bottleneck Analysis[/bold]")
        for preconditioner_type, result in results.items():
            if result.top_operations:
                self.console.print(
                    f"\n[cyan]{preconditioner_type}[/cyan] top operations:"
                )
                for i, (op_name, time_ms) in enumerate(result.top_operations[:10]):
                    self.console.print(f"  {i+1}. {op_name}: {time_ms:.2f}ms")

        # Quick comparison
        self.console.print("\n[bold]Quick Comparison (SGD = 1.0x)[/bold]")
        comparisons = []
        for preconditioner_type, result in results.items():
            relative_time = result.avg_time_per_epoch / sgd_time
            comparisons.append(
                f"{preconditioner_type}: [bold red]{relative_time:.1f}x[/bold red]"
            )
        self.console.print(" | ".join(comparisons))

    def run_all_benchmarks(self, num_epochs: int = 100) -> dict[str, BenchmarkResult]:
        """Run benchmarks for all preconditioners.

        Args:
            num_epochs: Number of epochs to run for each benchmark.

        Returns:
            Dictionary mapping preconditioner names to their benchmark results.
        """
        self.console.print("\n[bold cyan]Preconditioner Benchmark[/bold cyan]")
        self.console.print(
            f"[dim]Device: {self.device} | Param Shape: {self.param_shapes[0]} | "
            f"Epochs: {num_epochs}[/dim]\n"
        )

        results = {}
        for preconditioner_type in self.PRECONDITIONER_TYPES:
            result = self._benchmark_single(preconditioner_type, num_epochs)
            results[preconditioner_type] = result

        self._display_results(results)
        return results


if __name__ == "__main__":
    # Run benchmark with a single 2048x2048 parameter matrix
    benchmark = PreconditionerBenchmark(
        [(2048, 2048)], device="cuda" if torch.cuda.is_available() else "cpu"
    )
    benchmark.run_all_benchmarks(num_epochs=200)
