"""Preprocessing utilities for TEAMo datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .schema import (
    AudioPreprocessConfig,
    MotionPreprocessConfig,
    PreprocessingConfig,
    TextPreprocessConfig,
)

try:  # Optional dependency
    import torchaudio  # type: ignore
except ImportError:  # pragma: no cover - torchaudio is optional
    torchaudio = None


class AudioFeatureExtractor(nn.Module):
    """Extract log-mel spectrogram features from raw audio."""

    def __init__(self, cfg: AudioPreprocessConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if torchaudio is None:
            raise ImportError(
                "torchaudio is required for AudioFeatureExtractor. Install torchaudio to "
                "enable audio preprocessing."
            )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.win_length,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.mel_bins,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.melspec(waveform)
        spec = self.amplitude_to_db(spec)
        if self.cfg.normalize:
            spec = (spec - spec.mean(dim=-1, keepdim=True)) / (spec.std(dim=-1, keepdim=True) + 1e-5)
        return spec


class TextTokenizer:
    """Simple word-level tokenizer with optional vocabulary persistence."""

    def __init__(self, cfg: TextPreprocessConfig) -> None:
        self.cfg = cfg
        self.vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        if cfg.vocab_path and cfg.vocab_path.exists():
            self.vocab.update(json.loads(cfg.vocab_path.read_text()))

    def build_vocab(self, texts: Iterable[str]) -> None:
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
        if self.cfg.vocab_path:
            self.cfg.vocab_path.write_text(json.dumps(self.vocab, ensure_ascii=True, indent=2))

    def encode(self, text: str) -> torch.Tensor:
        tokens = self._tokenize(text)
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        if len(ids) > self.cfg.max_tokens:
            ids = ids[: self.cfg.max_tokens]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        tokens = [inverse_vocab.get(int(idx), "<unk>") for idx in token_ids]
        return " ".join(tokens)

    def pad_batch(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        max_len = max(t.size(0) for t in batch)
        padded = torch.full((len(batch), max_len), self.vocab["<pad>"], dtype=torch.long)
        for i, tensor in enumerate(batch):
            padded[i, : tensor.size(0)] = tensor
        return padded

    def _tokenize(self, text: str) -> List[str]:
        if self.cfg.lowercase:
            text = text.lower()
        tokens = text.strip().split()
        return tokens or ["<unk>"]


class MotionPreprocessor:
    """Normalise and smooth pose sequences."""

    def __init__(self, cfg: MotionPreprocessConfig) -> None:
        self.cfg = cfg
        self.window = cfg.smoothing_window

    def __call__(self, motion: torch.Tensor) -> torch.Tensor:
        if motion.dim() != 3:
            raise ValueError("motion tensor must have shape [frames, joints, dims]")
        if self.cfg.normalize:
            motion = (motion - motion.mean(dim=0, keepdim=True)) / (motion.std(dim=0, keepdim=True) + 1e-5)
        if self.window > 1:
            motion = self._smooth(motion)
        if self.cfg.keypoints is not None:
            motion = motion[:, self.cfg.keypoints, :]
        return motion

    def _smooth(self, motion: torch.Tensor) -> torch.Tensor:
        pad = self.window // 2
        padded = torch.nn.functional.pad(motion, (0, 0, 0, 0, pad, pad), mode="replicate")
        kernel = torch.ones(self.window, device=motion.device, dtype=motion.dtype) / self.window
        kernel = kernel.view(1, 1, -1)
        smoothed = torch.nn.functional.conv1d(
            padded.permute(1, 2, 0), kernel, padding=0
        )  # [joints, dims, frames]
        smoothed = smoothed.permute(2, 0, 1)
        return smoothed


class PreprocessingPipeline:
    """Convenience wrapper that batches preprocessing modules."""

    def __init__(self, cfg: PreprocessingConfig) -> None:
        self.cfg = cfg
        self.audio_extractor = None
        if torchaudio is not None:
            self.audio_extractor = AudioFeatureExtractor(cfg.audio)
        self.text_tokenizer = TextTokenizer(cfg.text)
        self.motion_preprocessor = MotionPreprocessor(cfg.motion)

    def process_sample(
        self,
        audio: Optional[torch.Tensor],
        text: Optional[str],
        motion: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        if audio is not None:
            if self.audio_extractor is None:
                raise RuntimeError("Audio extractor is not available; install torchaudio.")
            outputs["audio"] = self.audio_extractor(audio)
        if text is not None:
            outputs["text"] = self.text_tokenizer.encode(text)
        if motion is not None:
            outputs["motion"] = self.motion_preprocessor(motion)
        return outputs
