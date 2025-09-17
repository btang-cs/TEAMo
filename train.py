"""CLI entry for training TEAMo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - yaml optional
    yaml = None

from .data.schema import (
    AudioPreprocessConfig,
    DatasetConfig,
    DatasetSplitConfig,
    ExperimentConfig,
    MotionPreprocessConfig,
    PreprocessingConfig,
    TextPreprocessConfig,
)
from .losses import TEALossConfig
from .models import MambaTaggerConfig, TEADMConfig
from .models.core import TEAMoCoreConfig
from .training import LoggingConfig, OptimizationConfig, TEAMoTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TEAMo")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON config file")
    return parser.parse_args()


def load_config(path: Path) -> TrainingConfig:
    data = _read_config_file(path)
    experiment_block = data.get("experiment", {})
    dataset_cfg = _build_dataset_config(experiment_block["dataset"])
    preprocessing_cfg = _build_preprocessing_config(experiment_block.get("preprocessing", {}))
    experiment_cfg = ExperimentConfig(dataset=dataset_cfg, preprocessing=preprocessing_cfg)

    model_cfg = _build_model_config(data["model"])
    optimization_cfg = OptimizationConfig(**data.get("optimization", {}))
    logging_cfg = LoggingConfig(**data.get("logging", {}))
    training_kwargs = {k: v for k, v in data.items() if k in {"max_epochs", "precision", "device", "seed"}}
    return TrainingConfig(
        experiment=experiment_cfg,
        model=model_cfg,
        optimization=optimization_cfg,
        logging=logging_cfg,
        **training_kwargs,
    )


def _read_config_file(path: Path) -> Dict[str, Any]:
    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to parse YAML config files")
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _build_dataset_config(raw: Dict[str, Any]) -> DatasetConfig:
    splits = {}
    for name, split_data in raw["splits"].items():
        splits[name] = DatasetSplitConfig(
            name=name,
            metadata_path=Path(split_data["metadata_path"]),
            feature_root=Path(split_data["feature_root"]),
            shuffle=split_data.get("shuffle", name == "train"),
            drop_last=split_data.get("drop_last", name == "train"),
        )
    kwargs = {k: v for k, v in raw.items() if k not in {"splits"}}
    return DatasetConfig(splits=splits, **kwargs)


def _build_preprocessing_config(raw: Dict[str, Any]) -> PreprocessingConfig:
    audio_cfg = AudioPreprocessConfig(**raw.get("audio", {}))
    text_cfg = TextPreprocessConfig(**raw.get("text", {}))
    motion_cfg = MotionPreprocessConfig(**raw.get("motion", {}))
    return PreprocessingConfig(audio=audio_cfg, text=text_cfg, motion=motion_cfg)


def _build_model_config(raw: Dict[str, Any]) -> TEAMoCoreConfig:
    tagger_cfg = MambaTaggerConfig(**raw["mamba_tagger"])
    teada_cfg = TEADMConfig(**raw["teadm"])
    tea_loss_cfg = TEALossConfig(**raw["tea_loss"])
    lambda_tea = raw.get("lambda_tea", 1.0)
    lambda_trait = raw.get("lambda_trait", 1.0)
    return TEAMoCoreConfig(
        mamba_tagger=tagger_cfg,
        teadm=teada_cfg,
        tea_loss=tea_loss_cfg,
        lambda_tea=lambda_tea,
        lambda_trait=lambda_trait,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    trainer = TEAMoTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
