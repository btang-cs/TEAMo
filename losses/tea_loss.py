"""Trait-Emotion-Action loss (Eq. (12)–(14) in the TEAMo paper)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TEALossConfig:
    trait_dim: int
    emotion_dim: int
    motion_dim: int
    hidden_dim: int = 256
    cycle_weight: float = 1.0


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TEALoss(nn.Module):
    """Psychologically grounded regularisation enforcing the TEA pathway."""

    def __init__(self, cfg: TEALossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trait_to_emotion = _build_mlp(cfg.trait_dim, cfg.hidden_dim, cfg.emotion_dim)
        self.emotion_to_motion = _build_mlp(cfg.emotion_dim, cfg.hidden_dim, cfg.motion_dim)
        self.cycle_weight = cfg.cycle_weight

    def forward(
        self,
        z_trait: torch.Tensor,
        z_emotion: torch.Tensor,
        z_motion: torch.Tensor,
        trait_indices: Optional[Sequence[int]] = None,
        emotion_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the TEA loss components.

        Args:
            z_trait: Latent trait embeddings ``z_T``.
            z_emotion: Latent emotion embeddings ``z_E``.
            z_motion: Latent motion embeddings ``z_M``.
            trait_indices: Optional subset of indices corresponding to set ``S``
                in Eq. (12). If ``None`` the full batch is used.
            emotion_indices: Optional subset of indices corresponding to set
                ``S'`` in Eq. (12).
        Returns:
            Dictionary containing individual terms and the combined loss.
        """

        if trait_indices is not None:
            z_trait_subset = z_trait[trait_indices]
            z_emotion_trait = z_emotion[trait_indices]
        else:
            z_trait_subset = z_trait
            z_emotion_trait = z_emotion

        if emotion_indices is not None:
            z_emotion_subset = z_emotion[emotion_indices]
            z_motion_subset = z_motion[emotion_indices]
        else:
            z_emotion_subset = z_emotion
            z_motion_subset = z_motion

        trait_to_emotion = F.mse_loss(
            self.trait_to_emotion(z_trait_subset),
            z_emotion_trait,
        )

        emotion_to_motion = F.mse_loss(
            self.emotion_to_motion(z_emotion_subset),
            z_motion_subset,
        )

        reference_motion = z_motion[trait_indices] if trait_indices is not None else z_motion
        cycle_loss = F.mse_loss(
            self.emotion_to_motion(self.trait_to_emotion(z_trait_subset)),
            reference_motion,
        )

        total = trait_to_emotion + emotion_to_motion + self.cycle_weight * cycle_loss
        return {
            "loss": total,
            "trait_to_emotion": trait_to_emotion,
            "emotion_to_motion": emotion_to_motion,
            "cycle": cycle_loss,
        }
