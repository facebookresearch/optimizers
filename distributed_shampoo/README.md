# PyTorch Distributed Shampoo

Distributed Shampoo is a preconditioned stochastic gradient optimizer in the adaptive gradient (Adagrad) family of methods [1, 2]. It converges faster by leveraging neural network-specific structures to achieve comparable model quality/accuracy in fewer iterations or epochs at the cost of additional FLOPs and memory, or achieve higher model quality in the same number of iterations or epochs. Our implementation offers specialized support for serial, [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), and [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) training.

Distributed Shampoo currently only supports dense parameters.

The key to tuning this optimizer is to balance accuracy, performance, and memory. This is discussed in the Step-by-Step Guide below.

Developers:
- Hao-Jun Michael Shi (Meta Platforms, Inc.)
- Tsung-Hsien Lee
- Anna Cai (Meta Platforms, Inc.)
- Shintaro Iwasaki (Meta Platforms, Inc.)
- Ke Sang (Meta Platforms, Inc.)
- Wang Zhou (Meta Platforms, Inc.)

with contributions and support from:

Rohan Anil (Google), Adnan Aziz (Meta), Pavan Balaji (Meta), Shuo Chang (Meta), Weiwei Chu (Meta), Assaf Eisenman (Meta), Will Feng (Meta), Zhuobo Feng (Meta), Jose Gallego-Posada (Mila / Meta Platforms, Inc.), Avirup Ghosh (Meta), Yizi Gu (Meta), Vineet Gupta (Google), Yuchen Hao (Meta), Brian Hirsh (Meta), Yusuo Hu (Meta), Yuxi Hu (Meta), Minhui Huang (Meta), Guna Lakshminarayanan (Meta), Michael Lazos (Meta), Zhijing Li (Meta), Ming Liang (Meta), Wanchao Liang (Meta), Ying Liu (Meta), Wenguang Mao (Meta), Dheevatsa Mudigere (NVIDIA), Maxim Naumov (Meta), Jongsoo Park (Meta), Mike Rabbat (Meta), Kaushik Rangadurai (Meta), Dennis van der Staay (Meta), Fei Tian (Meta), Sanjay Vishwakarma (Meta), Xunnan (Shawn) Xu (Meta), Jiyan Yang (Meta), Chunxing Yin (Meta), and Iris Zhang (Meta).

## Updates
- (2/14/24) We have released our Distributed Shampoo v2.0.0 implementation, a ground-up re-write of our PyTorch Shampoo implementation. Our v2.0.0 implementation includes:
  - Incorporates new performance optimizations, such as the usage of `torch._foreach_*` operators and PyTorch 2 compile.
  - Shared support and enablement of DDP and FSDP Shampoo, via the specification of the `distributed_config` field.
  - Cleaner API for configuring grafting methods through specifying the `grafting_config` field.
  - Deprecation of handling large tensors by diagonalizing the Shampoo preconditioners and using standard diagonal Adagrad.
  - While we do not currently support LAMB/LARS grafting, we intend to add support for this in the future.
  - We will update our [ArXiv paper](https://arxiv.org/pdf/2309.06497.pdf) to reflect our implementation changes.

## Features

Key distinctives of this implementation include:
- Homogeneous multi-node multi-GPU support in PyTorch.
- Learning rate grafting [3]. Our version of grafting only grafts the second moment/diagonal preconditioner. Momentum/first moment updates are performed separate from grafting. Supports the methods:
    - SGD
    - Adagrad
    - RMSProp
    - Adam
- Supports both normal and AdamW (decoupled) weight decay.
- Incorporates exponential moving averaging (with or without bias correction) to the estimate the first moment (akin to Adam).
- Incorporates momentum and Nesterov acceleration.
- Offers multiple approaches for computing the root inverse, including:
    - Using symmetric eigendecomposition (used by default).
    - Coupled inverse Newton iteration [4].
- Choice of precision for preconditioner accumulation and root inverse computation.
- Ability to cache split parameters.
- Merging of small dimensions.

## Requirements

We have tested this implementation on the following versions of PyTorch:

- PyTorch >= 2.0;
- Python >= 3.8;
- CUDA 11.3-11.4; 12.2+;
- [expecttest](https://github.com/ezyang/expecttest) (for distributed unit tests);
- [hypothesis](https://github.com/HypothesisWorks/hypothesis) (for distributed unit tests).

If one wants to use `DTensor` which leads to memory savings, please set the hidden default `use_dtensor = True` under `allocate_distributed_tensor` in `shampoo_dist_utils.py`. (This is on by default.) Requires PyTorch 2 nightly build.

Note: We have observed known instabilities with the torch.linalg.eigh operator on CUDA 11.6-12.1, specifically for low-rank matrices, which may appear with using a small start_preconditioning_step. Please avoid these versions of CUDA if possible. See: https://github.com/pytorch/pytorch/issues/94772.

## How to Use

**Given a learning rate schedule for your previous base optimizer, we can replace the optimizer with Shampoo and "graft" from the learning rate schedule of the base method.**

A few notes on hyperparameters:

- Notice that Shampoo contains some new hyperparameters (`max_preconditioner_dim` and `precondition_frequency`) that are important for performance. We describe how to tune these below in the section on Hyperparameter Tuning.

- Here, `betas` refer to the hyperparameters used for the exponential moving average of the gradients and Shampoo preconditioners, while `grafting_beta2` corresponds to the `beta2` used specifically for exponential moving averaging of the grafted method. This is similar for `epsilon` and `grafting_epsilon`. As a first choice, we recommend setting `betas` equal to the previous `betas` and additionally setting `grafting_beta2` equal to `betas[1]`, and set `epsilon = 1e-12` and `grafting_epsilon` equal to the previous `epsilon`.

- We also distinguish between `beta1` and `momentum`. `beta1` corresponds to the EMA of the gradients (or gradient filtering), while `momentum` corresponds to the SGD momentum formula applied to the search direction.

- We allow for decoupled and coupled weight decay. If one sets `use_decoupled_weight_decay=True`, then you are enabling AdamW-style weight decay, while `use_decoupled_weight_decay=False` corresponds to the normal L2-regularization style weight decay.

### Example 1: [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) with Momentum

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
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import SGDGraftingConfig

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
    grafting_config=SGDGraftingConfig(),
)
```


### Example 2: [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

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
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig

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
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-08,
    ),
)
```

### Example 3: [Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)

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
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdaGradGraftingConfig

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
    grafting_config=AdaGradGraftingConfig(
        epsilon=1e-10,
    ),
)
```

### Example 4: [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

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
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig

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
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
)
```

## Distributed Training Support

Our implementation offers specialized compatibility and performance optimizations for different distributed training paradigms, including Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (FSDP) training. Note that Distributed Shampoo will work out of the box for DDP training, but not for FSDP training.

### DDP Training Support

In order to support fast DDP training, our implementation offers ZeRO-1 support, which distributes the computation and memory (via `DTensor`) in order to lower both Shampoo's memory requirements and its per-iteration wall-clock time at the cost of additional (`AllGather`) communication. Our DDP Shampoo implementation can either: (1) communicate the updated parameters; or (2) communicate the parameter updates.

We support:
- Quantized (or low-precision) communications using BF16, FP16, or FP32 communications.
- Specification of the number of trainers within each process group to distribute compute and memory. This trades off the amount of communication and compute each trainer is responsible for.
- Option to communicate updated parameters.

To use DDP Shampoo, simply configure the `distributed_config` as `DDPShampooConfig`:
```
import os

import torch
import torch.distributed as dist

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, DDPShampooConfig
from torch import nn

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))
torch.cuda.set_device(LOCAL_RANK)

model = instantiate_model().to(device)
model = nn.parallel.DistributedDataParallel(
    model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK
)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=DDPShampooConfig(
        communication_dtype=CommunicationDType.FP32,
        num_trainers_per_group=8,
        communicate_params=False,
    ),
)
```
Please see `ddp_cifar10_example.py` as an example.

### FSDP Training Support

FSDP training will create flattened parameters by flattening and concatenating all parameters within each FSDP module. By default, this removes all information about each parameter's tensor shape that Shampoo aims to exploit. Therefore, in order to support FSDP training, we have to use additional FSDP metadata in order to recover valid tensor blocks of the original parameters.

Note that we only support PyTorch FSDP with the `use_orig_params=True` option.
```
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, FSDPShampooConfig
from distributed_shampoo.utils.shampoo_fsdp_utils import compile_fsdp_parameter_metadata

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK))

model = instantiate_model().to(device)
model = FSDP(model, use_orig_params=True)

optimizer = DistributedShampoo(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    epsilon=1e-12,
    weight_decay=1e-05,
    max_preconditioner_dim=8192,
    precondition_frequency=100,
    use_decoupled_weight_decay=True,
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-12,
    ),
    distributed_config=FSDPShampooConfig(
        param_to_metadata=compile_fsdp_parameter_metadata(model),
    ),
)
```
Please see `fsdp_cifar10_example.py` as an example.

## Checkpointing Support

To checkpoint Distributed Shampoo, we have to use the `torch.distributed.checkpoint` solution with `DTensor`. *Note that we do not currently support the standard PyTorch checkpointing solution because it cannot handle storing process groups or `DTensor` by default.* We have therefore disabled `state_dict` and `load_state_dict` and instead rely on `distributed_state_dict` and `load_distributed_state_dict` instead.

Distributed checkpointing requires a fully-qualified name (FQN) mapping for each parameter, unlike the identifier used in `torch.optim.Optimizer`. The easiest way to handle this requirement is to use the model's `named_parameters()` function and pass this as the `key_to_param` argument of `distributed_state_dict` and `load_distributed_state_dict`.

Given a `CHECKPOINT_DIR`, to store the checkpoint:
```
import torch.distributed.checkpoint as dist_checkpoint

state_dict = {
    "model": model.state_dict(),
    "optim": optimizer.distributed_state_dict(key_to_param=model.named_parameters()),
}
dist_checkpoint.save_state_dict(
    state_dict=state_dict,
    storage_writer=dist_checkpoint.FileSystemWriter(CHECKPOINT_DIR),
)
```

To load the checkpoint:
```
dist_checkpoint.load_state_dict(
    state_dict=state_dict,
    storage_reader=dist_checkpoint.FileSystemReader(CHECKPOINT_DIR),
)
model.load_state_dict(state_dict["model"])
optimizer.load_distributed_state_dict(state_dict["optim"], key_to_param=model.named_parameters())
```

You can also refer to `ddp_cifar10_example.py` as an example.

## Hyperparameter Tuning

**We want to tune Shampoo to balance model quality, memory, and efficiency/performance by applying approximations to a "pure" version of Shampoo.**

This requires adjusting the hyperparameters `max_preconditioner_dim`, `precondition_frequency`, and `start_preconditioning_step`. The general approach is to start by using as close to a “pure” version of Shampoo as possible, then incorporate approximations to ensure that one obtains fast performance. A pure version of Shampoo would set `max_preconditioner_dim = 8192` and `precondition_frequency = 1`.

With the inclusion of learning rate grafting, we can extract a good learning rate schedule from your existing scheduler. Other techniques for preventing divergence (gradient clipping) may also be removed.

### Step-by-Step Guide

1. Start with a reasonable `max_preconditioner_dim` (i.e., 8192) and reduce the block size as necessary for memory and performance.

    * The maximum effective value of this hyperparameter is the maximum value of the products of each layer’s dimensions. For example, if we have a model with three layers where the first layer is 5x5x3x6, the second layer is 3x3x3x8, and the third layer is 216x5; the products of the first, second, and third layers’ dimensions are 5x5x3x6=450, 3x3x3x8=216, and 216x10=1080, respectively. In this example, 1080 is the maximum effective value of this hyperparameter, and any value greater than 1080 will perform the same as 1080.

    * The higher this value is, the better the model quality we expect.

    * There is a sweet spot in terms of performance - if the number is too small, the algorithm will slow down due to kernel latency. On the other hand, using too large of a value leads to slow matrix computations (i.e., matrix root inverses), which scale as $O(n^3)$ if $n$ is the dimension of the matrix, as well as poor load-balancing. In our experience, using a `max_preconditioner_dim` between 1024 and 8192 is ideal for performance.

    * Memory varies depending on the order of the tensor. For vectors, increasing `max_preconditioner_dim` leads to increased memory costs, but for 3rd-order tensors (or higher), increasing `max_preconditioner_dim` leads to decreased memory costs. Blocked matrices yield a fixed memory cost regardless of `max_preconditioner_dim`.

    * For efficiency purposes, it is best to set this value as a multiple of 2.

    * The following is an example of setting `max_preconditioner_dim = 4096` with SGD grafting:
    ```
    optimizer = DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        momentum=0.9,
        weight_decay=0.01,
        max_preconditioner_dim=4096,
        grafting_config=SGDGraftingConfig(),
    )
    ```

2. Use the smallest `precondition_frequency` (i.e., 1) and increase the precondition frequency.

    * This hyperparameter determines how frequently the preconditioner is computed. The smaller the value, the slower Shampoo becomes but with faster convergence. The goal is to find a value that balances convergence and speed.

    * It is normal to eventually set this hyperparameter on the order of hundreds or thousands. This is based primarily on the size of the network and the effective ratio between the cost of a single forward-backward pass + standard optimizer step to the cost of computing a series of matrix root inverses.

    * In practice, we have found that an upper bound to `precondition_frequency` is on the order of thousands. This approach will offer diminishing performance gains if the bottleneck is due to preconditioning, which is performed at every iteration.

    * The following is an example of setting `precondition_frequency = 100`:
    ```
    optimizer = DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        momentum=0.9,
        weight_decay=0.01,
        precondition_frequency=100,
        grafting_config=SGDGraftingConfig(),
    )
    ```

3. Set `start_preconditioning_step` to be consistent with the precondition frequency.

    * This hyperparameter determines when to start using Shampoo. Prior to this, the optimizer will use the grafted method. This value should generally be set larger than or equal to `precondition_frequency` except when the precondition frequency is 1. By default, `start_preconditioning_step` is set equal to `precondition_frequency`.

    * If the `precondition_frequency = 1`, then set `start_preconditioning_step = 0` in order to use Shampoo from the start.

    * Following is an example of setting `start_preconditioning_step = 300`:
    ```
    optimizer = DistributedShampoo(
        nn.parameters(),
        lr=0.01,
        betas=(0., 0.999),
        momentum=0.9,
        weight_decay=0.01,
        start_preconditioning_step=300,
        grafting_config=SGDGraftingConfig(),
    )
    ```

4. To tune for better model quality, one can tune:

    * **Learning Rate** (`lr`): One can change the learning rate schedule, and potentially use a larger learning rate.
    * **Nesterov Momentum** (`momentum`, `use_nesterov`): In some cases, we have found using Nesterov momentum to substantially improve model quality. To use this, we recommend setting `momentum` to 0.5 or 0.9 and setting `use_nesterov` to True. The learning rate needs to be re-tuned with respect to this hyperparameter.
    * **Epsilon Regularization** (`epsilon`): One should typically search for a value in $\{10^{−12},10^{−11},...,10^{−2},10^{−1}\}$.
    * **Exponential Moving Average Parameters** (`betas`): One can tune the `betas = (beta1, beta2)` parameters as is typical for Adam(W).
    * **Inverse Root Override and Multiplier** (`inv_root_override`, `exponent_multiplier`): In general, we have found that using `inv_root_override = 2` xor `exponent_multiplier = 1.82` works well in practice, particularly for models dominated by fully-connected layers, such as in ranking and recommendation models.
    * **Preconditioner Data Type** (`preconditioner_dtype`): For certain models, it is necessary to use higher precision to accumulate the Shampoo factor matrices and compute its eigendecomposition to obtain high enough numerical accuracy. In those cases, one can specify this as `torch.float64`. (Note that this will use more memory.)
    * **MTML Task Weights**: Task weights may need to be re-tuned as Distributed Shampoo will better exploit certain imbalances between different task losses.

5. If enabling DDP Shampoo, you can tune for performance:

    * **Process Group Size** (`num_trainers_per_group`): For large-scale distributed jobs, this hyperparameter allows us to trade off computational and communication costs. Assuming the number of GPUs per node is 8, one should search for a value in $\{8,16,32,64\}$. This hyperparameter has no impact on model quality.
    * **Quantized Communications** (`communication_dtype`): One can enable quantized communications by setting the `communication_dtype`. We have found that using `CommunicationDType.FP16` works well in practice (with `communicate_params = False`).
    * **Communicate Updated Parameters** (`communicate_params`): If one does not enable quantized communications, one can possibly obtain better performance by communicating the updated parameters by setting this to `True`.

## References

1. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf). Vineet Gupta, Tomer Koren, and Yoram Singer. International Conference on Machine Learning, 2018.
2. [Scalable Second-Order Optimization for Deep Learning](https://arxiv.org/pdf/2002.09018.pdf). Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Tech Report, 2021.
3. [Learning Rate Grafting: Transferability of Optimizer Tuning](https://openreview.net/pdf?id=FpKgG31Z_i9). Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Tech Report, 2021.
4. [Functions of Matrices: Theory and Computation](https://epubs.siam.org/doi/book/10.1137/1.9780898717778). Nicholas J. Higham. SIAM, 2008.
5. [A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale](https://arxiv.org/pdf/2309.06497.pdf). Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat. Tech Report, 2023.
