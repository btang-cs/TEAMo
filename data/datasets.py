"""Dataset and dataloader utilities for TEAMo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess import PreprocessingPipeline
from .schema import DatasetConfig, DatasetSplitConfig, PreprocessingConfig


def _load_tensor(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(path)
    if suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if suffix == ".npz":
        data = np.load(path)
        if "arr_0" in data:
            return torch.from_numpy(data["arr_0"])
        raise KeyError(f"npz file {path} does not contain 'arr_0'")
    raise ValueError(f"Unsupported tensor file format: {path}")


class TEAMoDataset(Dataset):
    """Loads multimodal samples as described in the TEAMo paper."""

    def __init__(
        self,
        split_cfg: DatasetSplitConfig,
        preprocessing: PreprocessingConfig,
        lazy_audio: bool = True,
    ) -> None:
        super().__init__()
        self.split_cfg = split_cfg
        self.feature_root = split_cfg.feature_root
        self.pipeline = PreprocessingPipeline(preprocessing)
        self.lazy_audio = lazy_audio
        metadata_text = split_cfg.metadata_path.read_text()
        raw_metadata = json.loads(metadata_text)
        if not isinstance(raw_metadata, list):
            raise ValueError("Metadata file must contain a list of sample descriptions")
        self.metadata: List[Dict[str, object]] = raw_metadata

    @property
    def text_tokenizer(self):
        return self.pipeline.text_tokenizer

    def build_text_vocab(self) -> None:
        texts: List[str] = []
        for meta in self.metadata:
            text_value = self._load_text(meta)
            if text_value is not None:
                texts.append(text_value)
        if texts:
            self.pipeline.text_tokenizer.build_vocab(texts)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[index]
        audio_tensor = self._load_optional(meta, "audio")
        text = self._load_text(meta)
        motion_tensor = self._load_optional(meta, "motion")
        trait_tensor = self._load_optional(meta, "traits")
        emotion_tensor = self._load_optional(meta, "emotion")

        outputs = self.pipeline.process_sample(audio_tensor, text, motion_tensor)
        if trait_tensor is not None:
            outputs["traits"] = trait_tensor.float()
        if emotion_tensor is not None:
            outputs["emotion"] = emotion_tensor.float()
        for optional_key in ("latent", "latent_target", "latent_mask"):
            optional_tensor = self._load_optional(meta, optional_key)
            if optional_tensor is not None:
                outputs[optional_key] = optional_tensor.float()
        outputs["id"] = torch.tensor(index, dtype=torch.long)
        return outputs

    def _load_optional(self, meta: Dict[str, object], key: str) -> Optional[torch.Tensor]:
        if key not in meta:
            return None
        value = meta[key]
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return torch.tensor(value)
        path = Path(value)
        if not path.is_absolute():
            path = self.feature_root / path
        if key == "audio" and self.lazy_audio:
            # Audio will be loaded as waveform; expect torchaudio available
            import torchaudio  # type: ignore

            waveform, _ = torchaudio.load(path)
            return waveform
        return _load_tensor(path)

    @staticmethod
    def _load_text(meta: Dict[str, object]) -> Optional[str]:
        value = meta.get("text")
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise TypeError("Text entry in metadata must be a string")


def _collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    output: Dict[str, List[torch.Tensor]] = {}
    for sample in batch:
        for key, tensor in sample.items():
            output.setdefault(key, []).append(tensor)

    padded_batch: Dict[str, torch.Tensor] = {}
    for key, tensors in output.items():
        if key in {"audio", "motion"}:
            padded_batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True
            )
        elif key == "text":
            padded_batch[key] = torch.nn.utils.rnn.pad_sequence(
                tensors, batch_first=True
            )
        else:
            padded_batch[key] = torch.stack(tensors)
    return padded_batch


class TEAMoDataModule:
    """Utility to create PyTorch dataloaders for TEAMo experiments."""

    def __init__(self, cfg: DatasetConfig, preprocessing: PreprocessingConfig) -> None:
        self.cfg = cfg
        self.preprocessing = preprocessing
        self.datasets: Dict[str, TEAMoDataset] = {}

    def setup(self) -> None:
        for split_name, split_cfg in self.cfg.splits.items():
            dataset = TEAMoDataset(split_cfg, self.preprocessing)
            if split_name == "train":
                dataset.build_text_vocab()
            self.datasets[split_name] = dataset
        train_dataset = self.datasets.get("train")
        if train_dataset is not None:
            shared_vocab = dict(train_dataset.text_tokenizer.vocab)
            for name, dataset in self.datasets.items():
                if name == "train":
                    continue
                dataset.text_tokenizer.vocab = shared_vocab

    def get_dataloader(self, split: str) -> DataLoader:
        if split not in self.datasets:
            raise KeyError(f"Split '{split}' has not been set up")
        split_cfg = self.cfg.splits[split]
        dataset = self.datasets[split]
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=split_cfg.shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            drop_last=split_cfg.drop_last,
            collate_fn=_collate_batch,
        )
