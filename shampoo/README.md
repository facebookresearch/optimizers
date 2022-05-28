# Distributed Shampoo

An experimental distributed Shampoo implementation in PyTorch. Currently only supports dense parameters.

Key distinctives of this implementation include:
- Incorporation of learning rate grafting. Our version of grafting only grafts the second moment/diagonal preconditioner. Momentum/first moment updates are performed separate from grafting. Supports the methods:
    - SGD
    - Adagrad
    - RMSProp
    - Adam
- Supports both normal and AdamW weight decay.
- Distribution of the root inverse computation across different GPUs for the distributed data-parallel setting. Does not perform CPU offloading.
- Offers different options to handle large-dimensional tensors, including:
    - Diagonalizing the Shampoo preconditioners.
    - Using standard diagonal Adagrad.
    - Blocking the tensor and applying Shampoo to each block. (Needs to be improved.)
- Offers multiple approaches for computing the root inverse, including:
    - Using symmetric eigendecomposition.
    - Coupled inverse Newton iteration.
- Choice of precision for preconditioner accumulation and root inverse computation.
- Merging of small dimensions.

## How to Use
Given a learning rate schedule for another method, one should simply replace the optimizer with Shampoo and graft from the previous method.

For example,
```
optimizer = optim.AdamW(nn.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
```
we would instead use
```
optimizer = shampoo.Shampoo(
    nn.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    bias_correction=True,
    adam_w_mode=True,
    weight_decay=0.01,
    grafting_type=GraftingType.ADAM,
    grafting_epsilon=1e-08,
    grafting_beta2=0.999,
)
```

## References
1. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf ). Vineet Gupta, Tomer Koren, and Yoram Singer. International Conference on Machine Learning, 2018.
2. [Scalable Second-Order Optimization for Deep Learning](https://arxiv.org/pdf/2002.09018.pdf). Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Tech Report, 2021.
3. [Learning Rate Grafting: Transferability of Optimizer Tuning](https://openreview.net/pdf?id=FpKgG31Z_i9). Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Tech Report, 2021.
4. [Functions of Matrices: Theory and Computation.](https://epubs.siam.org/doi/book/10.1137/1.9780898717778) Nicholas J. Higham. SIAM, 2008.
