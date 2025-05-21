# Preconditioner Benchmark

This benchmark compares different preconditioners in [shampoo_preconditioner_list.py](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/utils/shampoo_preconditioner_list.py). It illustrates total time, average time, GPU-usage, and bottleneck using **rich Console and PyTorch profiler**.


## Benchmark List

- `SGDPreconditionerList` : SGD (no preconditioning)
- `AdagradPreconditionerList` : AdaGrad
- `RootInvShampooPreconditionerList` : Root Inverse Shampoo
- `EigendecomposedShampooPreconditionerList` : Eigendecomposed Shampoo
- `EigenvalueCorrectedShampooPreconditionerList` : Eigenvalue-Corrected Shampoo


## Example

```zsh
➜  optimizers git: ✗ uv run benchmarks/preconditioner.py

Preconditioner Benchmark
Device: cuda | Param Shape: (2048, 2048) | Epochs: 200

                                       Performance Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Preconditioner             ┃ Total (s) ┃ Avg/Epoch (ms) ┃ Memory (MB) ┃ Relative ┃ GPU Util % ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ SGD                        │     0.021 │           0.11 │         0.0 │     1.0x │        1.6 │
│ AdaGrad                    │     0.064 │           0.32 │        16.0 │     3.0x │      164.7 │
│ EigendecomposedShampoo     │     1.260 │           6.30 │        64.0 │    58.8x │      428.9 │
│ EigenvalueCorrectedShampoo │     1.602 │           8.01 │        64.0 │    74.7x │      595.3 │
└────────────────────────────┴───────────┴────────────────┴─────────────┴──────────┴────────────┘

Bottleneck Analysis

SGD top operations:
  1. aten::to: 0.08ms
  2. aten::_to_copy: 0.08ms
  3. aten::copy_: 0.08ms
  4. Memcpy HtoD (Pageable -> Device): 0.08ms
  5. aten::empty: 0.00ms

AdaGrad top operations:
  1. ## AdagradPreconditionerList:update_preconditioners ##: 26.43ms
  2. aten::_foreach_addcmul_: 26.43ms
  3. void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<3>, at::native::(anonymous namespace)::PointwiseOpScalarFunctor<float, 3,
3, 0>, std::multiplies<float>, float>(at::native::(anonymous namespace)::TensorListMetadata<3>, at::native::(anonymous namespace)::PointwiseOpScalarFunctor<float, 3, 3, 0>, std::multiplies<float>,
float): 26.43ms
  4. ## AdagradPreconditionerList:update_preconditioners ##: 26.43ms
  5. aten::to: 0.08ms

EigendecomposedShampoo top operations:
  1. ## EigendecomposedShampooPreconditionerList:update_preconditioners ##: 899.27ms
  2. aten::mm: 675.26ms
  3. ## EigendecomposedShampooPreconditionerList:_update_factor_matrices ##: 668.67ms
  4. ## EigendecomposedShampooPreconditionerList:_update_factor_matrices ##: 654.35ms
  5. aten::tensordot: 601.84ms

EigenvalueCorrectedShampoo top operations:
  1. ## EigenvalueCorrectedShampooPreconditionerList:update_preconditioners ##: 2367.36ms
  2. aten::mm: 1210.46ms
  3. aten::tensordot: 1136.00ms
  4. void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nt_align1>(cutlass_80_simt_sgemm_256x128_8x4_nt_align1::Params): 824.67ms
  5. ## EigenvalueCorrectedShampooPreconditionerList:_update_factor_matrices ##: 652.20ms

Quick Comparison (SGD = 1.0x)
SGD: 1.0x | AdaGrad: 3.0x | EigendecomposedShampoo: 58.8x | EigenvalueCorrectedShampoo: 74.7x
```