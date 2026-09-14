# Preconditioner Benchmarks

This benchmark compares different preconditioners in [shampoo_preconditioner_list.py](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/utils/shampoo_preconditioner_list.py). It illustrates **time consumption, CPU-usage, and GPU-usage using PyTorch Profiler** and rich Console.

Note: You should be available to CUDA for the benchmarks

### Benchmark List

- `SGDPreconditionerList` : SGD (no preconditioning)
- `AdagradPreconditionerList` : AdaGrad
- `RootInvShampooPreconditionerList` : Root Inverse Shampoo
- `EigendecomposedShampooPreconditionerList` : Eigendecomposed Shampoo
- `EigenvalueCorrectedShampooPreconditionerList` : Eigenvalue-Corrected Shampoo


### Example

![image](https://github.com/user-attachments/assets/c92edaf0-234f-464d-b661-8dc28da84118)