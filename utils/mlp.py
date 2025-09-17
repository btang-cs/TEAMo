from typing import Iterable, Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Configurable multilayer perceptron with SiLU activations."""

    def __init__(self, dims: Sequence[int], dropout: float = 0.0) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP requires at least input and output dimensions")

        layers: Iterable[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != dims[-1]:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
