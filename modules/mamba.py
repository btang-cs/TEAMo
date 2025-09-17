"""Selective state-space blocks used across TEAMo components.

The implementation mimics the behaviour of the Mamba architecture through a
lightweight state-space update equipped with dynamic gates and depth-wise
convolution. The blocks are intentionally compact yet expressive enough for the
TEAMo modules described in the paper.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SelectiveStateSpaceBlock(nn.Module):
    """Single selective state-space block approximating a Mamba layer."""

    def __init__(
        self,
        model_dim: int,
        state_dim: Optional[int] = None,
        conv_kernel: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd to preserve tensor length")

        state_dim = state_dim or model_dim
        self.model_dim = model_dim
        self.state_dim = state_dim

        self.input_layer = nn.Linear(model_dim, 2 * state_dim)
        self.gate_layer = nn.Linear(model_dim, state_dim)
        self.conv = nn.Conv1d(
            in_channels=state_dim,
            out_channels=state_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=state_dim,
        )
        self.output_layer = nn.Linear(state_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        state_and_skip = self.input_layer(x)
        state_update, skip = state_and_skip.chunk(2, dim=-1)

        state_update = state_update.transpose(1, 2)
        state_update = self.conv(state_update)
        state_update = state_update.transpose(1, 2)

        gate = torch.sigmoid(self.gate_layer(x))
        updated_state = (state_update + skip) * gate
        updated_state = self.output_layer(updated_state)
        updated_state = self.dropout(updated_state)
        return residual + updated_state


class SelectiveStateSpaceStack(nn.Module):
    """Stack of selective state-space blocks."""

    def __init__(
        self,
        depth: int,
        model_dim: int,
        state_dim: Optional[int] = None,
        conv_kernel: int = 5,
        dropout: float = 0.0,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        state_dim = state_dim or model_dim
        self.layers = nn.ModuleList(
            [
                SelectiveStateSpaceBlock(
                    model_dim=model_dim,
                    state_dim=state_dim,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim) if final_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x


def build_state_space_stack(
    depth: int,
    model_dim: int,
    state_dim: Optional[int] = None,
    conv_kernel: int = 5,
    dropout: float = 0.0,
    final_norm: bool = True,
) -> SelectiveStateSpaceStack:
    return SelectiveStateSpaceStack(
        depth=depth,
        model_dim=model_dim,
        state_dim=state_dim,
        conv_kernel=conv_kernel,
        dropout=dropout,
        final_norm=final_norm,
    )
