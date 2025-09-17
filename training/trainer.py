"""Training loop implementation for TEAMo."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from ..data import TEAMoDataModule
from ..models.core import TEAMoCore
from ..training.config import LoggingConfig, OptimizationConfig, TrainingConfig
from ..utils.scheduler import build_warmup_cosine_scheduler


class TEAMoTrainer:
    """Orchestrates TEAMo end-to-end training."""

    def __init__(self, cfg: TrainingConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.device = device or torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self._setup_dirs()
        torch.manual_seed(cfg.seed)

        self.datamodule = TEAMoDataModule(cfg.experiment.dataset, cfg.experiment.preprocessing)
        self.datamodule.setup()

        self.model = TEAMoCore(cfg.model).to(self.device)
        self.condition_dims = cfg.model.teadm.condition_dims
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.optimization.learning_rate,
            betas=cfg.optimization.betas,
            weight_decay=cfg.optimization.weight_decay,
        )
        self.scheduler = self._build_scheduler(cfg.optimization)
        self.ema_model: Optional[nn.Module] = None
        if cfg.optimization.ema_decay is not None:
            self.ema_model = TEAMoCore(cfg.model).to(self.device)
            self._update_ema(0.0)

        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.precision == "mixed")
        self.global_step = 0

    def _setup_dirs(self) -> None:
        self.cfg.logging.log_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.logging.log_dir / "checkpoints").mkdir(exist_ok=True)
        config_path = self.cfg.logging.log_dir / "config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(asdict(self.cfg), indent=2, default=str))

    def _build_scheduler(self, optim_cfg: OptimizationConfig) -> LambdaLR:
        return build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            warmup_steps=optim_cfg.warmup_steps,
            total_steps=self.cfg.max_epochs * max(1, len(self.datamodule.get_dataloader("train"))),
        )

    def _update_ema(self, decay: float) -> None:
        if self.ema_model is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_feature(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        if tensor.dim() == 2:
            return tensor
        batch = tensor.size(0)
        return tensor.view(batch, -1)

    def _prepare_condition(self, tensor: Optional[torch.Tensor], key: str, batch_size: int) -> torch.Tensor:
        target_dim = self.condition_dims[key]
        if tensor is None:
            return torch.zeros(batch_size, target_dim, device=self.device)
        tensor = tensor.to(self.device).float()
        flattened = self._flatten_feature(tensor)
        if flattened.size(1) > target_dim:
            flattened = flattened[:, :target_dim]
        elif flattened.size(1) < target_dim:
            pad = torch.zeros(batch_size, target_dim - flattened.size(1), device=self.device, dtype=flattened.dtype)
            flattened = torch.cat([flattened, pad], dim=1)
        return flattened

    def _reduce_for_tagger(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor.mean(dim=1)
        if tensor.dim() == 4:
            return tensor.mean(dim=(1, 2))
        return tensor

    def _prepare_tagger(self) -> None:
        train_loader = self.datamodule.get_dataloader("train")
        audio_features = []
        text_features = []
        motion_features = []
        trait_vectors = []

        for batch in train_loader:
            if "traits" not in batch:
                continue
            trait_vectors.append(batch["traits"].float())
            if "audio" in batch:
                audio_features.append(self._reduce_for_tagger(batch["audio"].float()))
            if "text" in batch:
                text_features.append(self._reduce_for_tagger(batch["text"].float()))
            if "motion" in batch:
                motion_features.append(self._reduce_for_tagger(batch["motion"].float()))
        if not trait_vectors:
            raise RuntimeError("Training data must include trait annotations for Mamba Tagger")

        labeled_traits = torch.cat(trait_vectors, dim=0).to(self.device)
        labeled_features: Dict[str, torch.Tensor] = {}
        if audio_features:
            labeled_features["audio"] = torch.cat(audio_features, dim=0).to(self.device)
        if text_features:
            labeled_features["text"] = torch.cat(text_features, dim=0).to(self.device)
        if motion_features:
            labeled_features["motion"] = torch.cat(motion_features, dim=0).to(self.device)
        if not labeled_features:
            raise RuntimeError("No modalities available to train the Mamba Tagger")
        self.model.tagger.fit(labeled_features, labeled_traits)

    # ------------------------------------------------------------------
    # Training / evaluation loops
    # ------------------------------------------------------------------
    def train(self) -> None:
        self._prepare_tagger()
        train_loader = self.datamodule.get_dataloader("train")
        val_loader = self.datamodule.get_dataloader("val") if "val" in self.datamodule.datasets else None

        for epoch in range(1, self.cfg.max_epochs + 1):
            epoch_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._evaluate(val_loader) if val_loader is not None else None
            self._save_checkpoint(epoch, epoch_loss, val_loss)

    def _train_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for batch in progress:
            self.global_step += 1
            loss = self._compute_loss(batch)
            loss_value = float(loss.detach())
            running_loss += loss_value

            self.scaler.scale(loss).backward()
            if self.cfg.optimization.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimization.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            if self.ema_model is not None:
                self._update_ema(self.cfg.optimization.ema_decay)
            if self.global_step % self.cfg.logging.log_every_steps == 0:
                progress.set_postfix(loss=loss_value)
        return running_loss / max(1, len(loader))

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        latent = batch.get("latent")
        target_latent = batch.get("latent_target", latent)
        if latent is None:
            if "motion" not in batch:
                raise RuntimeError("Batch must include 'latent' or 'motion' to supervise reconstruction")
            latent = batch["motion"].float()
            target_latent = latent
        batch_size = latent.size(0)

        conditions = {
            "audio": self._prepare_condition(batch.get("audio"), "audio", batch_size),
            "text": self._prepare_condition(batch.get("text"), "text", batch_size),
            "trait": self._prepare_condition(batch.get("traits"), "trait", batch_size),
            "emotion": self._prepare_condition(batch.get("emotion"), "emotion", batch_size),
        }
        tagger_inputs = {k: v for k, v in batch.items() if k in {"audio", "text", "motion"}}
        target_traits = batch.get("traits")
        if target_traits is None:
            target_traits = torch.zeros(batch_size, self.model.tagger.num_traits, device=self.device)
        total_loss, _ = self.model.compute_losses(
            latent=latent,
            target_latent=target_latent,
            conditions=conditions,
            tagger_inputs=tagger_inputs,
            target_traits=target_traits,
        )
        return total_loss

    def _evaluate(self, loader) -> Optional[float]:
        if loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                loss = self._compute_loss(batch)
                total_loss += float(loss)
        return total_loss / max(1, len(loader))

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        ckpt_dir = self.cfg.logging.log_dir / "checkpoints"
        ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "global_step": self.global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if self.ema_model is not None:
            payload["ema_model"] = self.ema_model.state_dict()
        torch.save(payload, ckpt_path)
        self._prune_checkpoints()

    def _prune_checkpoints(self) -> None:
        ckpt_dir = self.cfg.logging.log_dir / "checkpoints"
        checkpoints = sorted(ckpt_dir.glob("epoch_*.pt"))
        excess = len(checkpoints) - self.cfg.logging.keep_last_k
        for path in checkpoints[: max(0, excess)]:
            path.unlink()
