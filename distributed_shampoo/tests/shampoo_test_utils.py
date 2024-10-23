"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
from typing import Optional

import torch
from torch import nn


class _ModelWithLinearAndDeadLayers(nn.Module):
    def __init__(
        self,
        model_linear_layers_dims: tuple[int, ...],
        model_dead_layer_dims: Optional[tuple[int, ...]],
        bias: bool = False,
    ) -> None:
        super().__init__()
        # fully_shard doesn't support containers so we fall back to use nn.Sequential
        self.linear_layers: nn.Sequential = nn.Sequential(
            *(
                nn.Linear(a, b, bias=bias)
                for a, b in itertools.pairwise(model_linear_layers_dims)
            )
        )
        if model_dead_layer_dims is not None:
            self.dead_layers: nn.Sequential = nn.Sequential(
                *(
                    nn.Linear(a, b, bias=False)
                    for a, b in itertools.pairwise(model_dead_layer_dims)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear_layer in self.linear_layers:
            x = linear_layer(x)
        return x


def construct_training_problem(
    model_linear_layers_dims: tuple[int, ...],
    model_dead_layer_dims: Optional[tuple[int, ...]] = (10, 10),
    device: Optional[torch.device] = None,
    bias: bool = False,
    fill: float | tuple[float, ...] = 0.0,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
    """
    Constructs a training problem (model, loss, data, target) with the given model dimensions and attributes.

    Args:
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the model linear layers.
        model_dead_layer_dims (Optional[tuple[int, ...]]): The dimensions of the model dead linear layers. (Default: (10, 10))
        device (Optional[torch.device]): The device to use. (Default: None)
        bias (bool): Whether to use bias in the linear (non-dead) layers. (Default: False)
        fill (float | tuple[float, ...]): The value(s) to fill the model parameters. If a tuple, each element should correspond to one layer. (Default: 0.0)

    Returns:
        model (nn.Module): The model as specified from the input arguments.
        loss (nn.Module): The loss function (currently always set to MSE).
        data (torch.Tensor): A randomly generated input tensor corresponding to the input dimension.
        target (torch.Tensor): A target tensor of zeros corresponding to the output dimension.
    """
    data = torch.arange(model_linear_layers_dims[0], dtype=torch.float, device=device)
    data /= torch.norm(data)

    model = _ModelWithLinearAndDeadLayers(
        model_linear_layers_dims=model_linear_layers_dims,
        model_dead_layer_dims=model_dead_layer_dims,
        bias=bias,
    ).to(device=device)

    if isinstance(fill, float):
        fill = (fill,) * (len(model.linear_layers))
    for m, f in zip(model.linear_layers, fill, strict=True):
        m.weight.data.fill_(f)
        if bias:
            m.bias.data.fill_(f)

    loss = nn.MSELoss()

    target = torch.tensor([0.0] * model_linear_layers_dims[-1]).to(device=device)

    return model, loss, data, target
