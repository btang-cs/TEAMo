"""Unified TEAMo core module tying together Tagger, TEADM, and TEA loss."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .tagger import MambaTagger, MambaTaggerConfig
from .teadm import TEADM, TEADMConfig
from ..losses.tea_loss import TEALoss, TEALossConfig


@dataclass
class TEAMoCoreConfig:
    mamba_tagger: MambaTaggerConfig
    teadm: TEADMConfig
    tea_loss: TEALossConfig
    lambda_tea: float = 1.0
    lambda_trait: float = 1.0


class TEAMoCore(nn.Module):
    """High-level module encapsulating the TEAMo learning objective."""

    def __init__(self, cfg: TEAMoCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tagger = MambaTagger(cfg.mamba_tagger)
        self.denoiser = TEADM(cfg.teadm)
        self.tea_loss = TEALoss(cfg.tea_loss)
        self.lambda_tea = cfg.lambda_tea
        self.lambda_trait = cfg.lambda_trait

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        latent: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        tagger_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        refined_latent, aux = self.denoiser(latent, conditions)
        outputs = {"latent": refined_latent, **aux}
        if tagger_inputs is not None:
            outputs["trait_prediction"] = self.tagger(tagger_inputs)
        return outputs

    def compute_losses(
        self,
        latent: torch.Tensor,
        target_latent: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        tagger_inputs: Dict[str, torch.Tensor],
        target_traits: torch.Tensor,
        trait_indices: Optional[Sequence[int]] = None,
        emotion_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.forward(latent, conditions, tagger_inputs)
        refined_latent = outputs["latent"]
        projected = outputs["projected_modalities"]

        recon_loss = F.mse_loss(refined_latent, target_latent)
        trait_pred = outputs["trait_prediction"]
        trait_loss = self.tagger.trait_loss(trait_pred, target_traits)

        z_trait = projected["trait"]
        z_emotion = projected["emotion"]
        z_motion = refined_latent.mean(dim=1)

        tea_losses = self.tea_loss(
            z_trait=z_trait,
            z_emotion=z_emotion,
            z_motion=z_motion,
            trait_indices=trait_indices,
            emotion_indices=emotion_indices,
        )

        total = recon_loss
        total = total + self.lambda_trait * trait_loss
        total = total + self.lambda_tea * tea_losses["loss"]

        components = {
            "total": total,
            "reconstruction": recon_loss,
            "trait": trait_loss,
            "tea": tea_losses["loss"],
        }
        components.update({f"tea/{k}": v for k, v in tea_losses.items() if k != "loss"})
        return total, components
