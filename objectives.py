from __future__ import annotations

from typing import NamedTuple

import torch

from .losses.tea import TEALossOutput


class ObjectiveBreakdown(NamedTuple):
    total: torch.Tensor
    reconstruction: torch.Tensor
    tea: torch.Tensor
    trait: torch.Tensor


def compute_total_loss(
    reconstruction_loss: torch.Tensor,
    tea_output: TEALossOutput,
    trait_loss: torch.Tensor,
    *,
    lambda_tea: float = 1.0,
    lambda_trait: float = 1.0,
) -> ObjectiveBreakdown:
    """Combine the individual terms"""

    tea_term = lambda_tea * tea_output.total
    trait_term = lambda_trait * trait_loss
    total = reconstruction_loss + tea_term + trait_term
    return ObjectiveBreakdown(
        total=total,
        reconstruction=reconstruction_loss,
        tea=tea_term,
        trait=trait_term,
    )
