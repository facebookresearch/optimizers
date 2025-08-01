# PyTorch Distributed Shampoo

[![arXiv](https://img.shields.io/badge/arXiv-2309.06497-b31b1b.svg)](https://arxiv.org/abs/2309.06497)


Distributed Shampoo is a preconditioned stochastic gradient optimizer in the adaptive gradient (Adagrad) family of methods [1, 2]. It converges faster by leveraging neural network-specific structures to achieve comparable model quality/accuracy in fewer iterations or epochs at the cost of additional FLOPs and memory, or achieve higher model quality in the same number of iterations or epochs. Our implementation offers specialized support for serial, [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html), [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html), [Hybrid Sharding Data Parallel (HSDP)](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html#how-to-use-devicemesh-with-hsdp), [Per-parameter Fully Sharded Data Parallel (FSDP2)](#fsdp2-training-support), and [Per-parameter Hybrid Sharded Data Parallel (HSDP2)](#hsdp2-training-support) training.

Distributed Shampoo currently only supports dense parameters.

The key to tuning this optimizer is to balance accuracy, performance, and memory. This is discussed in the Step-by-Step Guide below.

Developers:
- Hao-Jun Michael Shi (Meta Platforms, Inc.)
- Tsung-Hsien Lee
- Anna Cai (Meta Platforms, Inc.)
- Runa Eschenhagen (University of Cambridge)
- Shintaro Iwasaki (Meta Platforms, Inc.)
- Ke Sang (Meta Platforms, Inc.)
- Wang Zhou (Meta Platforms, Inc.)

with contributions and support from:

Ganesh Ajjanagadde (Meta), Rohan Anil (Google), Adnan Aziz (Meta), Pavan Balaji (Meta), Shuo Chang (Meta), Weiwei Chu (Meta), Assaf Eisenman (Meta), Will Feng (Meta), Zhuobo Feng (Meta), Jose Gallego-Posada (Mila / Meta Platforms, Inc.), Avirup Ghosh (Meta), Yizi Gu (Meta), Vineet Gupta (Google), Yuchen Hao (Meta), Brian Hirsh (Meta), Yusuo Hu (Meta), Yuxi Hu (Meta), Minhui Huang (Meta), Guna Lakshminarayanan (Meta), Michael Lazos (Meta), Zhijing Li (Meta), Ming Liang (Meta), Wanchao Liang (Meta), Ying Liu (Meta), Wenguang Mao (Meta), Dheevatsa Mudigere (NVIDIA), Maxim Naumov (Meta), Jongsoo Park (Meta), Mike Rabbat (Meta), Kaushik Rangadurai (Meta), Dennis van der Staay (Meta), Fei Tian (Meta), Rohan Varma (Meta), Sanjay Vishwakarma (Meta), Xunnan (Shawn) Xu (Meta), Jiyan Yang (Meta), Chunxing Yin (Meta), Iris Zhang (Meta), Chuanhao Zhuge (Meta), and Will Zou (Meta).

## Features

Key distinctives of this implementation include:
- Homogeneous multi-node multi-GPU support in PyTorch.
- Learning rate grafting [3]. Our version of grafting only grafts the second moment/diagonal preconditioner. Momentum/first moment updates are performed separate from grafting. Supports the methods:
    - SGD
    - Adagrad
    - RMSprop
    - Adam
- Supports both normal and AdamW (decoupled) weight decay.
- Incorporates exponential moving averaging (with or without bias correction) to the estimate the first moment (akin to Adam).
- Incorporates momentum and Nesterov acceleration.
- Offers multiple approaches for computing the root inverse, including:
    - Using symmetric eigendecomposition (used by default).
    - Using the QR algorithm to compute an approximate eigendecomposition.
    - Coupled inverse Newton iteration [4].
    - Higher-order coupled iterations with relative epsilon based on estimate of largest eigenvalue.
- Choice of precision for preconditioner accumulation and root inverse computation.
- Ability to cache split parameters.
- Merging of small dimensions.
- Option to (approximately) correct the eigenvalues/run Adam in the eigenbasis of Shampoo's preconditioner (SOAP) [2,6,7].
- Option to use an adaptive preconditioner update frequency when symmetric eigendecompositions or the QR algorithm is used [8].
- Spectral descent via reduced SVD or Newton-Schulz iteration for 2D gradients, or gradients that have been reshaped to 2D [9,10]. This can be used to implement Muon [11], see [Example 6](#example-6-muon).

## Requirements

We have tested this implementation on the following versions of PyTorch:

- PyTorch >= 2.7;
- Python >= 3.12;
- CUDA 11.3-11.4; 12.2+;

Note: We have observed known instabilities with the torch.linalg.eigh operator on CUDA 11.6-12.1, specifically for low-rank matrices, which may appear with using a small `start_preconditioning_step`. Please avoid these versions of CUDA if possible. See: https://github.com/pytorch/pytorch/issues/94772.

## How to Use

**Given a learning rate schedule for your previous base optimizer, we can replace the optimizer with Shampoo and "graft" from the learning rate schedule of the base method. Alternatively, you can consider replacing Adam(W) by eigenvalue-corrected Shampoo (SOAP).**

A few notes on hyperparameters:

- Notice that Shampoo contains some new hyperparameters (`max_preconditioner_dim` and `precondition_frequency`) that are important for performance. We describe how to tune these below in the section on Hyperparameter Tuning.

- Here, `betas` refer to the hyperparameters used for the exponential moving average of the gradients and Shampoo preconditioners, while `grafting_beta2` corresponds to the `beta2` used specifically for exponential moving averaging of the grafted method. This is similar for `epsilon` and `grafting_epsilon`. As a first choice, we recommend setting `betas` equal to the previous `betas` and additionally setting `grafting_beta2` equal to `betas[1]`, and set `epsilon = 1e-12` and `grafting_epsilon` equal to the previous `epsilon`.

- We also distinguish between `beta1` and `momentum`. `beta1` corresponds to the EMA of the gradients (or gradient filtering), while `momentum` corresponds to the SGD momentum formula applied to the search direction.

- We allow for decoupled and coupled weight decay. If one sets `use_decoupled_weight_decay=True`, then you are enabling AdamW-style weight decay, while `use_decoupled_weight_decay=False` corresponds to the normal L2-regularization style weight decay.

- When setting `preconditioner_config` as an instance of `EigenvalueCorrectedShampooPreconditionerConfig` (see Example 5), there is typically no need to use learning rate grafting from Adam (`grafting_config=None`) and, when they are available, Adam's optimal `lr`, `betas`, and `weight_decay` should be a good starting point for further tuning. However, the case of `beta2=1.0`, i.e. an AdaGrad-like accumulation, has not been explored yet.  Also, in settings where Shampoo would usually graft its learning rate from SGD, grafting might still be beneficial.

### Example 1: [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) with Momentum

If we previously used the optimizer:
```python
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
```python
import torch
from distributed_shampoo import DistributedShampoo, SGDGraftingConfig

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
```python
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
```python
import torch
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo

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
```python
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
```python
import torch
from distributed_shampoo import AdaGradGraftingConfig, DistributedShampoo

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
```python
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
```python
import torch
from distributed_shampoo import AdamGraftingConfig, DistributedShampoo

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
        epsilon=1e-08,
    ),
)
```

### Example 5: eigenvalue-corrected Shampoo/SOAP

If we previously used the optimizer:
```python
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
```python
import torch
from distributed_shampoo import (
    DistributedShampoo,
    DefaultEigenvalueCorrectedShampooConfig,
)

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
    # This can also be set to `DefaultSOAPConfig` which uses QR decompositions, hence is
    # less expensive and might thereby allow for a smaller `precondition_frequency`.
    preconditioner_config=DefaultEigenvalueCorrectedShampooConfig,
)
```

### Example 6: Muon

```python
import math

from distributed_shampoo import (
    DistributedShampoo,
    NewtonSchulzOrthogonalizationConfig,
    SpectralDescentPreconditionerConfig,
)


model = instantiate_model()
# Separate parameters into hidden layers (only 2D) and other parameters (first layer, biases and other 1D parameters, and last layer).
hidden_layer_params = ...
other_params = ...

optimizer = DistributedShampoo(
    [
        # Use spectral descent with Newton-Schulz semi-orthogonalization for hidden layer parameters.
        {
            "params": hidden_layer_params,
            "lr": 0.02,
            "preconditioner_config": SpectralDescentPreconditionerConfig(
                orthogonalization_config=NewtonSchulzOrthogonalizationConfig(
                    scale_by_dims_fn=lambda d_in, d_out: max(1, d_out / d_in)**0.5,
                ),
            ),
        },
        # Use AdamW for other parameters.
        {
            "params": other_params,
            "lr": 3e-4,
            "start_preconditioning_step", math.inf,
            "grafting_config": AdamGraftingConfig(
                beta2=0.95,
                epsilon=1e-10,
            ),
        },
    ],
    weight_decay=1e-05,
    use_decoupled_weight_decay=True,
)
```

`SpectralDescentPreconditionerConfig` can also be used to implement other variations of spectral descent.

## Distributed Training Support

Our implementation offers specialized compatibility and performance optimizations for different distributed training paradigms, including Distributed Data Parallel (DDP) and Fully Sharded Data Parallel (including FSDP and per-parameter FSDP, a.k.a. FSDP2) training. Note that Distributed Shampoo will work out of the box for DDP training, but not for FSDP training.

### DDP Training Support

In order to support fast DDP training, our implementation offers ZeRO-1 support, which distributes the computation and memory (via `DTensor`) in order to lower both Shampoo's memory requirements and its per-iteration wall-clock time at the cost of additional (`AllGather`) communication. Our DDP Shampoo implementation can either: (1) communicate the updated parameters; or (2) communicate the parameter updates.

We support:
- Quantized (or low-precision) communications using BF16, FP16, or FP32 communications.
- Specification of the number of trainers within each process group to distribute compute and memory. This trades off the amount of communication and compute each trainer is responsible for.
- Option to communicate updated parameters.

To use DDP Shampoo, simply configure the `distributed_config` as `DDPShampooConfig`:
```python
import os

import torch
import torch.distributed as dist

from distributed_shampoo import (
    AdamGraftingConfig,
    DDPShampooConfig,
    DistributedShampoo,
)
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
        communication_dtype=torch.float32,
        num_trainers_per_group=8,
        communicate_params=False,
    ),
)
```
Please see [`ddp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/ddp_cifar10_example.py) as an example.

### FSDP1 (FullyShardedDataParallel)

#### FSDP1 Training Support

FSDP training will create flattened parameters by flattening and concatenating all parameters within each FSDP module. By default, this removes all information about each parameter's tensor shape that Shampoo aims to exploit. Therefore, in order to support FSDP training, we have to use additional FSDP metadata in order to recover valid tensor blocks of the original parameters.

Note that we only support PyTorch FSDP with the `use_orig_params=True` option.
```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from distributed_shampoo import (
    AdamGraftingConfig,
    compile_fsdp_parameter_metadata,
    DistributedShampoo,
    FSDPShampooConfig,
)

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
Please see [`fsdp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/fsdp_cifar10_example.py) as an example.

#### HSDP1 Training Support

Note that we only support PyTorch HSDP with `sharding_strategy=ShardingStrategy.HYBRID_SHARD` and the `use_orig_params=True` option.
```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from distributed_shampoo import (
    AdamGraftingConfig,
    compile_fsdp_parameter_metadata,
    DistributedShampoo,
    HSDPShampooConfig,
)

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    rank=WORLD_RANK,
    world_size=WORLD_SIZE,
)
device = torch.device("cuda:{}".format(LOCAL_RANK)

# Instantiate device mesh for HSDP Shampoo.
# Assuming 8 GPUs, a 2 x 4 mesh will be initialized.
# This means we shard model into four shards, and each sub-model has two replicas.
# [0, 1, 2, 3] and [4, 5, 6, 7] are the two shard groups.
# [0, 4], [1, 5], [2, 6], [3, 7] are the four replicate groups.
device_mesh = init_device_mesh("cuda", (2, 4))

model = instantiate_model().to(device)
model = FSDP(model, device_mesh=device_mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD, use_orig_params=True)

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
    distributed_config=HSDPShampooConfig(
        param_to_metadata=compile_fsdp_parameter_metadata(model),
        device_mesh=device_mesh,
    ),
)
```
Please see [`hsdp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/hsdp_cifar10_example.py) as an example.

### FSDP2 (fully_shard)

#### FSDP2 Training Support

Per-parameter sharding FSDP, also known as FSDP2, is the new fully sharded data parallelism implementation, which uses ``DTensor``-based dim-0 per-parameter sharding for a simpler sharding representation compared to FSDP1's flat-parameter sharding, while preserving similar throughput performance. In short, FSDP2 chunks each parameter on dim-0 across the data parallel workers (using ``torch.chunk(dim=0)``). To support Shampoo with FSDP2, we implement a new distributor that creates Shampoo preconditioner tensor blocks based on the rank local tensors of the dim-0 sharded ``DTensor`` parameters. One simplification brought by FSDP2 to Shampoo is that tensor blocks are local to each rank, so we don't need the ``tensor block recovery`` algorithm implemented for FSDP1 (where parameters are flatten and then sharded).

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from distributed_shampoo import (
    AdamGraftingConfig,
    DistributedShampoo,
    FullyShardShampooConfig,
)

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
model = fully_shard(model)

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
    distributed_config=FullyShardShampooConfig(),
)
```

Please see [`fully_shard_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/fully_shard_cifar10_example.py) as an example.

#### HSDP2 Training Support

We support PyTorch HSDP for FSDP2 (fully_shard).

```python
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from distributed_shampoo import (
    AdamGraftingConfig,
    DistributedShampoo,
    HybridShardShampooConfig,
)

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

# Instantiate device mesh for HSDP Shampoo.
# Assuming 8 GPUs, a 2 x 4 mesh will be initialized.
# This means we shard model into four shards, and each sub-model has two replicas.
# [0, 1, 2, 3] and [4, 5, 6, 7] are the two shard groups.
# [0, 4], [1, 5], [2, 6], [3, 7] are the four replicate groups.
device_mesh = init_device_mesh("cuda", (2, WORLD_SIZE // 2))

model = instantiate_model().to(device)
model = fully_shard(model, mesh=device_mesh)

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
    distributed_config=HybridShardShampooConfig(device_mesh=device_mesh),
)
```
Please see [`hybrid_shard_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/hybrid_shard_cifar10_example.py) as an example.

## Checkpointing Support

To checkpoint Distributed Shampoo, we have to use the `torch.distributed.checkpoint` solution with `DTensor`. *Note that we do not currently support the standard PyTorch checkpointing solution because it cannot handle storing process groups or `DTensor` by default.* We have therefore disabled `state_dict` and `load_state_dict` and instead rely on `distributed_state_dict` and `load_distributed_state_dict` instead.

Distributed checkpointing requires a fully-qualified name (FQN) mapping for each parameter, unlike the identifier used in `torch.optim.Optimizer`. The easiest way to handle this requirement is to use the model's `named_parameters()` function and pass this as the `key_to_param` argument of `distributed_state_dict` and `load_distributed_state_dict`.

Given a `CHECKPOINT_DIR`, to store the checkpoint:
```python
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
```python
dist_checkpoint.load_state_dict(
    state_dict=state_dict,
    storage_reader=dist_checkpoint.FileSystemReader(CHECKPOINT_DIR),
)
model.load_state_dict(state_dict["model"])
optimizer.load_distributed_state_dict(state_dict["optim"], key_to_param=model.named_parameters())
```

You can also refer to [`ddp_cifar10_example.py`](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/examples/ddp_cifar10_example.py) as an example.

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
    ```python
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
    ```python
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

    * If the `precondition_frequency = 1`, then set `start_preconditioning_step = -1` in order to use Shampoo from the start.

    * Following is an example of setting `start_preconditioning_step = 300`:
    ```python
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
    * **Preconditioner Data Type** (`preconditioner_dtype`): For certain models, it is necessary to use higher precision to accumulate the Shampoo factor matrices and compute its eigendecomposition to obtain high enough numerical accuracy. In those cases, one can specify this as `torch.float64`. (Note that this will use more memory.)
    * **MTML Task Weights**: Task weights may need to be re-tuned as Distributed Shampoo will better exploit certain imbalances between different task losses.

5. If enabling DDP Shampoo, you can tune for performance:

    * **Process Group Size** (`num_trainers_per_group`): For large-scale distributed jobs, this hyperparameter allows us to trade off computational and communication costs. Assuming the number of GPUs per node is 8, one should search for a value in $\{8,16,32,64\}$. This hyperparameter has no impact on model quality.
    * **Quantized Communications** (`communication_dtype`): One can enable quantized communications by setting the `communication_dtype`. We have found that using `torch.float16` works well in practice (with `communicate_params = False`).
    * **Communicate Updated Parameters** (`communicate_params`): If one does not enable quantized communications, one can possibly obtain better performance by communicating the updated parameters by setting this to `True`.

## Commmon Questions

### Encountering `NaN/Inf` numerical error:

When gradients are `NaN/Inf`, most optimizers still proceed smoothly and modify model weights with those `NaN/Inf` values but Shampoo reacts with error messages like "Encountered nan values ...".

When encountering those errors, following are things you could try:

1. Decrease the learning rate.
2. Adjust the learning rate scheduler.
3. Increase `start_preconditioning_step`.
4. Consider applying gradient clipping.

## Citing PyTorch Distributed Shampoo

If you use PyTorch Distributed Shampoo in your work, please use the following BibTeX entry.

```BibTeX
@misc{shi2023pytorchshampoo,
    title={A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale},
    author={Hao-Jun Michael Shi and Tsung-Hsien Lee and Shintaro Iwasaki and Jose Gallego-Posada and Zhijing Li and Kaushik Rangadurai and Dheevatsa Mudigere and Michael Rabbat},
    howpublished={\url{https://github.com/facebookresearch/optimizers/tree/main/distributed_shampoo}},
    year ={2023},
    eprint={2309.06497},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## References

1. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf). Vineet Gupta, Tomer Koren, and Yoram Singer. International Conference on Machine Learning, 2018.
2. [Scalable Second-Order Optimization for Deep Learning](https://arxiv.org/pdf/2002.09018.pdf). Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, and Yoram Singer. Tech report, 2021.
3. [Learning Rate Grafting: Transferability of Optimizer Tuning](https://openreview.net/pdf?id=FpKgG31Z_i9). Naman Agarwal, Rohan Anil, Elad Hazan, Tomer Koren, and Cyril Zhang. Tech report, 2021.
4. [Functions of Matrices: Theory and Computation](https://epubs.siam.org/doi/book/10.1137/1.9780898717778). Nicholas J. Higham. SIAM, 2008.
5. [A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer for Training Neural Networks At-Scale](https://arxiv.org/pdf/2309.06497.pdf). Hao-Jun Michael Shi, Tsung-Hsien Lee, Shintaro Iwasaki, Jose Gallego-Posada, Zhijing Li, Kaushik Rangadurai, Dheevatsa Mudigere, and Michael Rabbat. Tech report, 2023.
6. [Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis](https://arxiv.org/abs/1806.03884). Thomas George, César Laurent, Xavier Bouthillier, Nicolas Ballas, Pascal Vincent. NeurIPS, 2018.
7. [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321). Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade. ICLR, 2025.
8. [Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](https://www.arxiv.org/abs/2506.03595). Runa Eschenhagen, Aaron Defazio, Tsung-Hsien Lee, Richard E. Turner, Hao-Jun Michael Shi. Tech report, 2025.
9. [Preconditioned Spectral Descent for Deep Learning](https://papers.nips.cc/paper_files/paper/2015/hash/f50a6c02a3fc5a3a5d4d9391f05f3efc-Abstract.html). David E. Carlson, Edo Collins, Ya-Ping Hsieh, Lawrence Carin, Volkan Cevher. NIPS, 2015.
10. [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325). Jeremy Bernstein, Laker Newhouse. Tech report, 2024.
11. [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/). Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, Jeremy Bernstein. Blog post, 2024.
