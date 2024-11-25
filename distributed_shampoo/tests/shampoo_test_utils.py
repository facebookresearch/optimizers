"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import itertools
from collections.abc import Callable
from functools import reduce

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim.optimizer import ParamsT


class _ModelWithLinearAndDeadLayers(nn.Module):
    def __init__(
        self,
        model_linear_layers_dims: tuple[int, ...],
        model_dead_layer_dims: tuple[int, ...] | None,
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
        return reduce(lambda x, layer: layer(x), self.linear_layers, x)


def construct_training_problem(
    model_linear_layers_dims: tuple[int, ...],
    model_dead_layer_dims: tuple[int, ...] | None = (10, 10),
    device: torch.device | None = None,
    bias: bool = False,
    fill: float | tuple[float, ...] = 0.0,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]:
    """
    Constructs a training problem (model, loss, data, target) with the given model dimensions and attributes.

    Args:
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the model linear layers.
        model_dead_layer_dims (tuple[int, ...] | None): The dimensions of the model dead linear layers. (Default: (10, 10))
        device (torch.device | None): The device to use. (Default: None)
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


def compare_two_optimizers_on_weight_and_loss(
    control_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    experimental_optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    model_linear_layers_dims: tuple[int, ...] = (10, 1, 1),
    model_dead_layer_dims: tuple[int, ...] | None = None,
    device: torch.device | None = None,
    fill: float | tuple[float, ...] = 1.0,
    total_steps: int = 5,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """
    Compare the performance of two optimizers on a simple neural network.
    This function trains two identical neural networks using different optimizers and compares their weights and losses after training.

    Args:
        control_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the control optimizer.
        experimental_optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the experimental optimizer.
        model_linear_layers_dims (tuple[int, ...]): The dimensions of the linear layers in the neural network. (Defaults: (10, 1, 1))
        model_dead_layer_dims (tuple[int, ...] | None): The dimensions of the dead layers in the neural network. (Defaults: None)
        device (torch.device | None): The device to use for training. (Defaults: None)
        fill (float | tuple[float, ...]): The value(s) to fill the model parameters. If a tuple, each element should correspond to one layer. (Default: 0.0)
        total_steps (int): The number of training steps. (Defaults: 5)
        rtol (float | None): The relative tolerance for comparing weights and losses. (Defaults: None)
        atol (float | None): The absolute tolerance for comparing weights and losses. (Defaults: None)

    Returns:
        None
    """

    def train(
        optim_factory: Callable[[ParamsT], torch.optim.Optimizer],
    ) -> tuple[list[Parameter], torch.Tensor]:
        """
        Train a neural network using the given optimizer.

        Args:
            optim_factory (Callable[[ParamsT], torch.optim.Optimizer]): A factory function that returns an instance of the optimizer.

        Returns:
            weight_and_loss (tuple[list[torch.Tensor], torch.Tensor]): A tuple containing the trained weight and loss.
        """
        model, loss, data, target = construct_training_problem(
            model_linear_layers_dims=model_linear_layers_dims,
            model_dead_layer_dims=model_dead_layer_dims,
            device=device,
            fill=fill,
        )
        optimizer = optim_factory(model.parameters())
        for _ in range(total_steps):
            optimizer.zero_grad()
            objective = loss(model(data), target)
            objective.backward()
            optimizer.step()

        # Retrun the weight of linear_layers (i.e., non-dead layers) only because dead layers weights are random and never-changed.
        return list(
            model.get_submodule("linear_layers").parameters()
        ), objective.detach()

    control_params, control_loss = train(control_optim_factory)
    experimental_params, experimental_loss = train(experimental_optim_factory)
    torch.testing.assert_close(experimental_loss, control_loss, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        experimental_params, control_params, rtol=rtol, atol=atol
    )
