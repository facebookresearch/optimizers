# Optimizers

*Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.*

## Description
Optimizers is a Github repository of PyTorch optimization algorithms. It is designed for external collaboration and development.

Currently includes the optimizers:
- Distributed Shampoo

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
Optimizers is BSD licensed, as found in the LICENSE file.

## Installation and Dependencies
This code requires `python>=3.8` and (as of 5 May 2023) requires the PyTorch nightly build. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install PyTorch. Installing PyTorch with CUDA and NCCL support is required.

Install `distributed_shampoo`:
```
git clone git@github.com:facebookresearch/optimizers.git
cd optimizers
pip install -e .
```

## Usage

After installation, basic usage looks like:
```
import torch
from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig

model = ...  # Instantiate model

optim = DistributedShampoo(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    epsilon=1e-8,
    grafting_config=AdamGraftingConfig(
        beta2=0.999,
        epsilon=1e-8,
    ),
)
```

For more, please see the [additional documentation here](./distributed_shampoo/README.md) and especially the [How to Use](./distributed_shampoo/README.md#how-to-use) section.
