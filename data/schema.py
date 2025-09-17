"""Configuration dataclasses for TEAMo dataset handling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DatasetSplitConfig:
    """Location and parameters for a single dataset split."""

    name: str
    metadata_path: Path
    feature_root: Path
    shuffle: bool = True
    drop_last: bool = False


@dataclass
class DatasetConfig:
    """Configuration for loading the TEAMo datasets."""

    splits: Dict[str, DatasetSplitConfig]
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    trait_key: str = "traits"
    emotion_key: str = "emotion"
    motion_key: str = "motion"
    audio_key: str = "audio"
    text_key: str = "text"
    max_sequence_length: Optional[int] = None
    chunk_length: Optional[int] = None


@dataclass
class AudioPreprocessConfig:
    sample_rate: int = 16000
    mel_bins: int = 64
    hop_length: int = 160
    win_length: int = 400
    f_min: float = 0.0
    f_max: Optional[float] = None
    normalize: bool = True


@dataclass
class TextPreprocessConfig:
    vocab_path: Optional[Path] = None
    lowercase: bool = True
    max_tokens: int = 128


@dataclass
class MotionPreprocessConfig:
    frame_rate: int = 60
    smoothing_window: int = 5
    keypoints: Optional[List[int]] = None
    normalize: bool = True


@dataclass
class PreprocessingConfig:
    audio: AudioPreprocessConfig = field(default_factory=AudioPreprocessConfig)
    text: TextPreprocessConfig = field(default_factory=TextPreprocessConfig)
    motion: MotionPreprocessConfig = field(default_factory=MotionPreprocessConfig)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
