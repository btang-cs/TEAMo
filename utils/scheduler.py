"""Scheduler helpers."""
from __future__ import annotations

import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    minimum_factor: float = 0.1,
) -> LambdaLR:
    """Return a LambdaLR implementing warmup followed by cosine decay."""

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps and warmup_steps > 0:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
