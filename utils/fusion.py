from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn


class AdaptiveConditionFusion(nn.Module):
    """Implements the learnable modality weighting"""

    def __init__(self, modalities: Iterable[str], dim: int) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.dim = dim
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
                for name in self.modalities
            }
        )
        self.weight_proj = nn.Linear(len(self.modalities) * dim, len(self.modalities))

    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        projected = []
        for name in self.modalities:
            if name not in features:
                raise KeyError(f"Missing modality '{name}' in fusion input")
            projected.append(self.projections[name](features[name]))

        concat = torch.cat(projected, dim=-1)
        weights = torch.softmax(self.weight_proj(concat), dim=-1)

        fused = torch.zeros_like(projected[0])
        weight_dict: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.modalities):
            weight = weights[..., idx : idx + 1]
            fused = fused + weight * projected[idx]
            weight_dict[name] = weight.squeeze(-1)

        return fused, weight_dict
