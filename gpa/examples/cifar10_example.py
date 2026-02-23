"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CIFAR-10 Training Example with GPAAdamW Optimizer.

Single GPU example demonstrating GPA optimizer usage with a simple ConvNet.

Requirements:
    - Python 3.10 or above
    - PyTorch / TorchVision

To run this training script:

    python -m gpa.examples.cifar10_example

With custom parameters:

    python -m gpa.examples.cifar10_example --lr 0.001 --epochs 10

"""

import argparse

import torch
from gpa.gpa_adamw import GPAAdamW
from torch import nn
from torchvision import datasets, transforms


class ConvNet(nn.Module):
    """Simple two-layer convolutional network for CIFAR-10 classification."""

    def __init__(self, height: int = 32, width: int = 32, out_channels: int = 64):
        super().__init__()
        kernel_size, stride, padding = 3, 1, 1
        self.conv = nn.Conv2d(3, out_channels, kernel_size, stride, padding, bias=False)
        self.activation = nn.ReLU()
        conv_h = (height - kernel_size + 2 * padding) // stride + 1
        conv_w = (width - kernel_size + 2 * padding) // stride + 1
        self.linear = nn.Linear(conv_h * conv_w * out_channels, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.flatten(self.activation(self.conv(x)), start_dim=1))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 GPA Training Example")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--train-interp-coeff", type=float, default=0.7, help="GPA train coefficient"
    )
    parser.add_argument(
        "--eval-interp-coeff", type=float, default=0.9967, help="GPA eval coefficient"
    )
    parser.add_argument(
        "--data-path", type=str, default="/tmp/cifar10", help="Data directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    torch.manual_seed(args.seed)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        args.data_path, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Model and optimizer
    model = ConvNet().to(device)
    optimizer = GPAAdamW(
        model.parameters(),
        lr=args.lr,
        train_interp_coeff=args.train_interp_coeff,
        eval_interp_coeff=args.eval_interp_coeff,
    )
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        optimizer.train()  # Switch optimizer to train mode (use y-sequence)

        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(data), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Evaluation (optional if y and x sequences are similar)
        model.eval()
        optimizer.eval()  # Switch optimizer to eval mode (use x-sequence)
        print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(train_loader):.4f}")

    print("Training complete!")

    # Checkpoint save example (optional if y and x sequences are similar)
    optimizer.eval()  # Switch to eval mode before saving
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    # torch.save(checkpoint, "checkpoint.pt")
    print("Checkpoint would be saved here (uncomment to enable)")

"""
## Extending to Distributed Training

GPA optimizers work seamlessly with DDP and FSDP. Simply wrap your model
and create the optimizer afterward:

### DDP Example:

    from torch.nn.parallel import DistributedDataParallel as DDP

    model = DDP(model, device_ids=[local_rank])
    optimizer = GPAAdamW(model.parameters(), lr=1e-3)

    # Training loop remains identical - GPA's train/eval mode
    # switching works the same in distributed settings.

### FSDP Example:

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    model = FSDP(model, ...)
    optimizer = GPAAdamW(model.parameters(), lr=1e-3)

    # No additional changes needed for GPA with FSDP.

"""
