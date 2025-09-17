"""Inference utilities for TEAMo."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from .models.core import TEAMoCore
from .training import TEAMoTrainer, TrainingConfig
from .train import load_config

MotionDecoder = Callable[[torch.Tensor], torch.Tensor]


class TEAMoInferenceEngine:
    """Load TEAMo checkpoints and perform annotation / generation."""

    def __init__(
        self,
        cfg: TrainingConfig,
        checkpoint_path: Path,
        device: Optional[torch.device] = None,
        motion_decoder: Optional[MotionDecoder] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.motion_decoder = motion_decoder
        self.model = TEAMoCore(cfg.model).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @classmethod
    def from_run_directory(
        cls,
        run_dir: Path,
        checkpoint_name: str = "latest",
        device: Optional[torch.device] = None,
        motion_decoder: Optional[MotionDecoder] = None,
    ) -> "TEAMoInferenceEngine":
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Could not find config.json in {run_dir}")
        cfg = load_config(config_path)
        if checkpoint_name == "latest":
            checkpoints = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found in run directory")
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        return cls(cfg, checkpoint_path, device=device, motion_decoder=motion_decoder)

    def annotate_traits(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in features.items()}
        with torch.no_grad():
            predictions = self.model.tagger(inputs)
        return predictions.cpu()

    def generate(
        self,
        latent: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
        tagger_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        latent = latent.to(self.device)
        conditioned = {k: v.to(self.device) for k, v in conditions.items()}
        tagger_inputs = tagger_inputs or {}
        tagger_inputs = {k: v.to(self.device) for k, v in tagger_inputs.items()}
        with torch.no_grad():
            outputs = self.model(latent, conditioned, tagger_inputs)
        if self.motion_decoder is not None and "latent" in outputs:
            outputs["motion"] = self.motion_decoder(outputs["latent"])
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}


def load_trainer_for_inference(run_dir: Path) -> TEAMoTrainer:
    cfg = load_config(run_dir / "config.json")
    cfg = replace(cfg, logging=replace(cfg.logging, log_dir=run_dir))
    trainer = TEAMoTrainer(cfg)
    return trainer
