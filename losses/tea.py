from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from ..utils import MLP


@dataclass
class TEALossOutput:
    total: torch.Tensor
    mapping: torch.Tensor
    cycle: torch.Tensor
    trait_to_emotion_pred: torch.Tensor
    emotion_to_motion_pred: torch.Tensor


class TEALoss(nn.Module):
    """Trait-Emotion-Action loss"""

    def __init__(
        self,
        *,
        trait_dim: int,
        emotion_dim: int,
        motion_dim: int,
        trait_hidden: Tuple[int, ...] = (256, 128),
        emotion_hidden: Tuple[int, ...] = (256, 128),
        lambda_cycle: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_cycle = lambda_cycle
        t2e_dims = [trait_dim, *trait_hidden, emotion_dim]
        e2m_dims = [emotion_dim, *emotion_hidden, motion_dim]
        self.trait_to_emotion = MLP(t2e_dims)
        self.emotion_to_motion = MLP(e2m_dims)
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        trait_latent: torch.Tensor,
        emotion_latent: torch.Tensor,
        motion_latent: torch.Tensor,
        *,
        trait_mask: Optional[torch.Tensor] = None,
        motion_mask: Optional[torch.Tensor] = None,
    ) -> TEALossOutput:
        """Compute the TEA loss.

        Args:
            trait_latent: [batch, trait_dim]
            emotion_latent: [batch, emotion_dim]
            motion_latent: [batch, motion_dim]
            trait_mask: optional mask over samples for the trait-emotion term.
            motion_mask: optional mask over samples for emotion-motion terms.
        """

        pred_emotion = self.trait_to_emotion(trait_latent)
        pred_motion = self.emotion_to_motion(emotion_latent)

        mapping_loss = self._masked_mean(
            self.mse(pred_emotion, emotion_latent),
            mask=trait_mask,
        ) + self._masked_mean(
            self.mse(pred_motion, motion_latent),
            mask=motion_mask,
        )

        cycle_pred_motion = self.emotion_to_motion(pred_emotion)
        cycle_loss = self._masked_mean(
            self.mse(cycle_pred_motion, motion_latent),
            mask=motion_mask,
        )

        total = mapping_loss + self.lambda_cycle * cycle_loss
        return TEALossOutput(
            total=total,
            mapping=mapping_loss,
            cycle=cycle_loss,
            trait_to_emotion_pred=pred_emotion,
            emotion_to_motion_pred=pred_motion,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return values.mean()
        mask = mask.to(values.dtype)
        while mask.dim() < values.dim():
            mask = mask.unsqueeze(-1)
        masked = values * mask
        denom = mask.sum()
        if denom <= 0:
            return values.mean()
        return masked.sum() / denom
