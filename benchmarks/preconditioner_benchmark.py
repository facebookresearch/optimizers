import torch
import time
from distributed_shampoo.utils.shampoo_preconditioner_list import (
    ShampooPreconditionerList,
    EigendecomposedShampooPreconditionerList,
    AdagradPreconditionerList,
)

def benchmark_preconditioner_lists(grad_shapes, device="cuda"):
    results = {}

    # Build dummy gradient
    grads = [torch.randn(shape, device=device) for shape in grad_shapes]

    # List for simulation
    for preconditioner_cls in [ShampooPreconditionerList,
                               EigendecomposedShampooPreconditionerList,
                               AdagradPreconditionerList]:

        # Initialize start memory and time
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()

        # Benchmarking for simulation
        for step in range(100):
            # ... (simulation logics)
            """ [To-do]
            - Build simulation logics for preconditioner
            - Record score (time and gpu-usage) per specified iterations
            """

        #
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated()

        # Note. result must contain time and computation cost
        results[preconditioner_cls.__name__] = {
            "time": end_time - start_time,
            "memory": end_memory - start_memory
        }

    return results