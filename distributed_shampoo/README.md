# PyTorch Distributed Shampoo

Distributed Shampoo is a second-order optimizer in the Adagrad family of methods [1, 2]. It converges in fewer iterations or epochs at the cost of more compute and memory. The key to tuning this optimizer is to balance accuracy, performance, and memory. We will discuss how to do this below.

Developers:
- Hao-Jun Michael Shi (Meta Platforms, Inc.)
- Tsung-Hsien Lee (Cruise)

with contributions and support from:
- Rohan Anil (Google)
- Vineet Gupta (Google)
- Shintaro Iwasaki (Meta Platforms, Inc.)
- Zhijing Li (Meta Platforms, Inc.)
- Dheevatsa Mudigere (NVIDIA)
- Mike Rabbat (Meta Platforms, Inc.)
- Kaushik Rangadurai (Meta Platforms, Inc.)

This implementation is under development. Currently only supports dense parameters.

## Features
-----------

Key distinctives of this implementation include:
- Learning rate grafting [3]. Our version of grafting only grafts the second moment/diagonal preconditioner. Momentum/first moment updates are performed separate from grafting. Supports the methods:
    - SGD
    - Adagrad
    - RMSProp
    - Adam
    - Normalized Adagrad
    - Normalized RMSProp
    - Normalized Adam
- Supports both normal and AdamW weight decay.
- Incorporates exponential moving averaging (with or without bias correction) to the estimate the first moment (akin to Adam).
- Incorporates momentum and Nesterov acceleration.
- Distributes memory and computation across different GPUs for the data-parallel setting. Supports data-parallel multi-node, multi-GPU training using `torch.nn.parallel.DistributedDataParallel`. Broadcasts are performed using `torch.dist`.
- Offers different options to handle large-dimensional tensors, including:
    - Diagonalizing the Shampoo preconditioners.
    - Using standard diagonal Adagrad.
    - Blocking the tensor and applying Shampoo to each block.
- Offers multiple approaches for computing the root inverse, including:
    - Using symmetric eigendecomposition (used by default).
    - Coupled inverse Newton iteration [4].
- Choice of precision for preconditioner accumulation and root inverse computation.
- Merging of small dimensions.

*We are in the process of optimizing the performance of this implementation. Stay tuned!*

## How to Use
-------------
**Given a learning rate schedule for your previous base optimizer, we can replace the optimizer with Shampoo and "graft" from the learning rate schedule of the base method.**

A few notes on hyperparameters:

- Notice that Shampoo contains some new hyperparameters (`max_preconditioner_dim` and `precondition_frequency`) that are important for performance. We describe how to tune these below in the section on Hyperparameter Tuning.

- Here, `betas` refer to the hyperparameters used for the exponential moving average of the gradients and Shampoo preconditioners, while `grafting_beta2` corresponds to the `beta2` used specifically for exponential moving averaging of the grafted method. This is similar for `epsilon` and `grafting_epsilon`. As a first choice, we recommend setting `betas` equal to the previous `betas` and additionally setting `grafting_beta2` equal to `betas[1]`, and set `epsilon = 1e-12` and `grafting_epsilon` equal to the previous `epsilon`.

- We also distinguish between `beta1` and `momentum`. `beta1` corresponds to the EMA of the gradients (or gradient filtering), while `momentum` corresponds to the SGD momentum formula applied to the search direction.

- We allow for decoupled and coupled weight decay. If one sets `use_decoupled_weight_decay=True`, then you are enabling AdamW-style weight decay, while `use_decoupled_weight_decay=False` corresponds to the normal L2-regularization style weight decay.

### Example 1: SGD with Momentum

If we previously used the optimizer:
```
import torch
from torch.optim import SGD

model = instantiate_model()

optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-05,
)
```
we would instead use:
```
import torch
from distributed_shampoo import DistributedShampoo
from shampoo_utils import GraftingType

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0., 0.999),
    epsilon=1e-12,
    momentum=0.9,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    grafting_type=GraftingType.SGD,
)
```


### Example 2: Adam

If we previously used the optimizer:
```
import torch
from torch.optim import Adam

model = instantiate_model()

optimizer = Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
)
```
we would instead use:
```
import torch
from distributed_shampoo import DistributedShampoo
from shampoo_utils import GraftingType

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=False,
    grafting_type=GraftingType.ADAM,
    grafting_epsilon=1e-08,
    grafting_beta2=0.999,
)
```

### Example 3: Adagrad

If we previously used the optimizer:
```
import torch
from torch.optim import Adagrad

model = instantiate_model()

optimizer = Adagrad(
    model.parameters(),
    lr=0.01,
    eps=1e-10,
    weight_decay=1e-05,
)
```
we would instead use:
```
import torch
from distributed_shampoo import DistributedShampoo
from shampoo_utils import GraftingType

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.01,
    betas=(0., 1.0),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=False,
    grafting_type=GraftingType.ADAGRAD,
    grafting_epsilon=1e-10,
)
```


### Example 4: AdamW

If we previously used the optimizer:
```
import torch
from torch.optim import AdamW

model = instantiate_model()

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
)
```
we would instead use:
```
import torch
from distributed_shampoo import DistributedShampoo
from shampoo_utils import GraftingType

model = instantiate_model()

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_type=GraftingType.ADAM,
    grafting_epsilon=1e-08,
    grafting_beta2=0.999,
)
```

## Hyperparameter Tuning
------------------------

**We want to tune Shampoo to balance model quality, memory, and efficiency/performance by applying approximations to a "pure" version of Shampoo.**

This requires adjusting the hyperparameters `max_preconditioner_dim`, `precondition_frequency`, and `start_preconditioning_step`. The general approach is to start by using as close to a “pure” version of Shampoo as possible, then incorporate approximations to ensure that one obtains fast performance. A pure version of Shampoo would set `max_preconditioner_dim = 8192` and `precondition_frequency = 1`.

With the inclusion of learning rate grafting, no additional changes are needed for your existing learning rate scheduler. Other techniques for preventing divergence (gradient clipping) may also be removed.

### Step-by-Step Guide

1. Start with the largest `max_preconditioner_dim` (i.e., 4096 or 8192) and reduce the block size.

    * The maximum effective value of this hyperparameter is the maximum value of the products of each layer’s dimensions. For example, if we have a model with three layers where the first layer is 5x5x3x6, the second layer is 3x3x3x8, and the third layer is 216x5; the products of the first, second, and third layers’ dimensions are 5x5x3x6=450, 3x3x3x8=216, and 216x10=1080, respectively. In this example, 1080 is the maximum effective value of this hyperparameter, and any value greater than 1080 will perform the same as 1080.

    * The higher this value is, the better the convergence of the algorithm will be at the cost of more memory.

    * For efficiency purposes, it is best to set this value as a multiple of 2.

    * The following is an example of setting `max_preconditioner_dim = 4096` with SGD grafting:
    ```
    optimizer = shampoo.DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        momentum=0.9,
        weight_decay=0.01,
        max_preconditioner_dim=4096,
        grafting_type=GraftingType.SGD,
    )
    ```

2. Use the smallest precondition_frequency (i.e., 1) and increase the precondition frequency.


    * This hyperparameter determines how frequently the preconditioner is computed. The smaller the value, the slower Shampoo becomes but with faster convergence. The goal is to find a value that balances convergence and speed.


    * It is normal to eventually set this hyperparameter on the order of hundreds or thousands. This is based primarily on the size of the network and the effective ratio between the cost of a single forward-backward pass + standard optimizer step to the cost of computing a series of matrix root inverses.

    * In practice, we have found that an upper bound to `precondition_frequency` is on the order of thousands. This approach will offer diminishing performance gains if the bottleneck is due to preconditioning, which is performed at every iteration.

    * The following is an example of setting `precondition_frequency = 100`:
    ```
    optimizer = shampoo.DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        momentum=0.9,
        weight_decay=0.01,
        precondition_frequency=100,
        grafting_type=GraftingType.SGD,
    )
    ```

3. Set `start_preconditioning_steps` to be consistent with the precondition frequency.

    * This hyperparameter determines when to start using Shampoo. Prior to this, the optimizer will use the grafted method. This value should generally be set larger than or equal to `precondition_frequency` except when the precondition frequency is 1. By default, `start_preconditioning_steps` is set equal to `precondition_frequency`.

    * If the precondition_frequency = 1, then set start_preconditioning_steps = 0 in order to use Shampoo from the start.

    * Following is an example of setting `start_preconditioning_steps = 300`:
    ```
    optimizer = shampoo.DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        momentum=0.9,
        weight_decay=0.01,
        start_preconditioning_steps=300,
        grafting_type=GraftingType.SGD,
    )
    ```

## References
-------------
1. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf ). Vineet Gupta, Tomer Koren, and Yoram Singer. International Conference on Machine Learning, 2018.
2. [Scalable Second-Order Optimization for Deep Learning](https://arxiv.org/pdf/2002.09018.pdf). Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Tech Report, 2021.
3. [Learning Rate Grafting: Transferability of Optimizer Tuning](https://openreview.net/pdf?id=FpKgG31Z_i9). Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Tech Report, 2021.
4. [Functions of Matrices: Theory and Computation.](https://epubs.siam.org/doi/book/10.1137/1.9780898717778) Nicholas J. Higham. SIAM, 2008.
