"""Training configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..data.schema import ExperimentConfig
from ..models.core import TEAMoCoreConfig


@dataclass
class OptimizationConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: Optional[float] = 1.0
    ema_decay: Optional[float] = None
    warmup_steps: int = 2000


@dataclass
class LoggingConfig:
    log_dir: Path = Path("runs")
    log_every_steps: int = 100
    checkpoint_every_epochs: int = 1
    keep_last_k: int = 5


@dataclass
class TrainingConfig:
    experiment: ExperimentConfig
    model: TEAMoCoreConfig
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    max_epochs: int = 100
    precision: str = "fp32"
    device: str = "cuda"
    seed: int = 42
