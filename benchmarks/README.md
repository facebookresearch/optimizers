# Preconditioner Benchmark

This benchmark compares different preconditioners in [shampoo_preconditioner_list.py](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/utils/shampoo_preconditioner_list.py). It illustrates total time, time per epoch, and total GPU usage.

### Usage

```bash
uv run preconditioner_benchmark
```

### Benchmark List

- `SGDPreconditionerList` : SGD (no preconditioning)
- `AdagradPreconditionerList` : AdaGrad
- `RootInvShampooPreconditionerList` : Root Inverse Shampoo
- `EigendecomposedShampooPreconditionerList` : Eigendecomposed Shampoo
- `EigenvalueCorrectedShampooPreconditionerList` : Eigenvalue-Corrected Shampoo