from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

from ..modules import (
    LayerNorm,
    MultiHeadCrossAttention,
    build_state_space_stack,
)
from ..utils import AdaptiveConditionFusion


@dataclass
class TEADMConfig:
    latent_dim: int
    condition_dims: Dict[str, int]
    fused_condition_dim: int = 256
    global_num_queries: int = 4
    global_num_heads: int = 4
    global_depth: int = 2
    global_state_dim: int = 256
    global_dropout: float = 0.1
    num_local_blocks: int = 4
    local_num_heads: int = 4
    local_num_queries: int = 8
    local_depth: int = 1
    local_state_dim: int = 256
    local_dropout: float = 0.1
    use_local_self_attention: bool = True


def _pool_feature(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.mean(dim=1)
    if x.dim() != 2:
        raise ValueError(
            f"Expected feature with rank 2 or 3, received tensor of shape {tuple(x.shape)}"
        )
    return x


class GlobalBlock(nn.Module):
    def __init__(self, cfg: TEADMConfig) -> None:
        super().__init__()
        latent_dim = cfg.latent_dim
        self.query_tokens = nn.Parameter(torch.randn(1, cfg.global_num_queries, latent_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.cross_attn = MultiHeadCrossAttention(
            dim=latent_dim,
            num_heads=cfg.global_num_heads,
            dropout=cfg.global_dropout,
        )
        self.state_stack = build_state_space_stack(
            depth=cfg.global_depth,
            model_dim=latent_dim,
            state_dim=cfg.global_state_dim,
            dropout=cfg.global_dropout,
        )
        combined_dim = cfg.fused_condition_dim + cfg.condition_dims["trait"]
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, latent_dim),
        )
        self.summary_proj = nn.Linear(latent_dim * 2, latent_dim)
        self.dropout = nn.Dropout(cfg.global_dropout)
        self.norm = LayerNorm(latent_dim)

    def forward(
        self,
        fused_condition: torch.Tensor,
        trait_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = fused_condition.size(0)
        condition = torch.cat([fused_condition, trait_embedding], dim=-1)
        condition = self.condition_proj(condition).unsqueeze(1)

        queries = self.query_tokens.expand(batch, -1, -1)
        attn_out = self.cross_attn(queries, condition)
        attn_out = self.state_stack(self.norm(attn_out))
        pooled = attn_out.mean(dim=1)
        summary = self.summary_proj(torch.cat([pooled, condition.squeeze(1)], dim=-1))
        summary = self.dropout(summary)
        return summary, attn_out


class LocalBlock(nn.Module):
    def __init__(self, cfg: TEADMConfig) -> None:
        super().__init__()
        latent_dim = cfg.latent_dim
        self.query_tokens = nn.Parameter(torch.randn(1, cfg.local_num_queries, latent_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.cross_attn = MultiHeadCrossAttention(
            dim=latent_dim,
            num_heads=cfg.local_num_heads,
            dropout=cfg.local_dropout,
        )
        self.self_attn = (
            nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=cfg.local_num_heads,
                dropout=cfg.local_dropout,
                batch_first=True,
            )
            if cfg.use_local_self_attention
            else None
        )
        self.state_stack = build_state_space_stack(
            depth=cfg.local_depth,
            model_dim=latent_dim,
            state_dim=cfg.local_state_dim,
            dropout=cfg.local_dropout,
        )
        self.condition_adapter = nn.Sequential(
            nn.LayerNorm(cfg.fused_condition_dim + cfg.condition_dims["emotion"]),
            nn.Linear(cfg.fused_condition_dim + cfg.condition_dims["emotion"], latent_dim),
        )
        self.pre_norm = LayerNorm(latent_dim)
        self.post_norm = LayerNorm(latent_dim)
        self.dropout = nn.Dropout(cfg.local_dropout)

    def forward(
        self,
        segment: torch.Tensor,
        fused_condition: torch.Tensor,
        emotion_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch, seg_len, latent_dim = segment.shape
        condition = torch.cat([fused_condition, emotion_embedding], dim=-1)
        condition_token = self.condition_adapter(condition).unsqueeze(1)
        segment = segment + condition_token

        queries = self.query_tokens.expand(batch, -1, -1)
        if seg_len != queries.size(1):
            repeat = (seg_len + queries.size(1) - 1) // queries.size(1)
            queries = queries.repeat(1, repeat, 1)[:, :seg_len, :]
        attn_out = self.cross_attn(queries, segment)
        attn_out = self.dropout(attn_out)
        x = self.pre_norm(segment + attn_out)

        if self.self_attn is not None and seg_len > 1:
            self_attn_out, _ = self.self_attn(x, x, x, need_weights=False)
            x = x + self.dropout(self_attn_out)
        x = self.post_norm(x)

        return self.state_stack(x)


class TEADM(nn.Module):
    def __init__(self, cfg: TEADMConfig) -> None:
        super().__init__()
        required = {"audio", "text", "trait", "emotion"}
        if set(cfg.condition_dims.keys()) != required:
            missing = required.difference(cfg.condition_dims)
            extra = set(cfg.condition_dims).difference(required)
            raise ValueError(
                "condition_dims must match {'audio','text','trait','emotion'} exactly. "
                f"Missing: {sorted(missing)}, Extra: {sorted(extra)}"
            )

        self.cfg = cfg
        self.condition_projs = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, cfg.fused_condition_dim),
                )
                for name, dim in cfg.condition_dims.items()
            }
        )
        self.fusion = AdaptiveConditionFusion(cfg.condition_dims.keys(), cfg.fused_condition_dim)
        self.global_block = GlobalBlock(cfg)
        self.local_blocks = nn.ModuleList(
            [LocalBlock(cfg) for _ in range(cfg.num_local_blocks)]
        )
        self.latent_norm = LayerNorm(cfg.latent_dim)
        self.output_proj = nn.Linear(cfg.latent_dim, cfg.latent_dim)

    def forward(
        self,
        latent_sequence: torch.Tensor,
        conditions: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if latent_sequence.dim() != 3:
            raise ValueError(
                "latent_sequence must have shape (batch, seq_len, latent_dim)"
            )
        if latent_sequence.size(-1) != self.cfg.latent_dim:
            raise ValueError(
                "Mismatch between latent dimension and TEADM configuration"
            )
        if latent_sequence.size(1) < self.cfg.num_local_blocks:
            raise ValueError("Sequence length must be >= number of local blocks")

        processed = {
            name: self.condition_projs[name](_pool_feature(conditions[name]))
            for name in self.condition_projs
        }
        fused_condition, modality_weights = self.fusion(processed)
        trait_embedding = processed["trait"]
        emotion_embedding = processed["emotion"]

        global_summary, global_tokens = self.global_block(fused_condition, trait_embedding)
        latent = self.latent_norm(latent_sequence) + global_summary.unsqueeze(1)

        segments = torch.chunk(latent, self.cfg.num_local_blocks, dim=1)
        refined_segments = []
        for segment, block in zip(segments, self.local_blocks):
            refined_segments.append(block(segment, fused_condition, emotion_embedding))
        refined = torch.cat(refined_segments, dim=1)

        denoised = latent_sequence + self.output_proj(refined)
        aux = {
            "fused_condition": fused_condition,
            "modality_weights": modality_weights,
            "projected_modalities": processed,
            "global_summary": global_summary,
            "global_tokens": global_tokens,
        }
        return denoised, aux
