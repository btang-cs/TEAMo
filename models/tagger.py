from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from ..modules import LayerNorm, SelectiveStateSpaceBlock
from ..utils import MLP, PersonalityNeighborSelector


class MambaTagger(nn.Module):
    """Personality label transfer module following the TEAMo description."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_modalities: int,
        trait_dim: int = 5,
        k_neighbors: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trait_dim = trait_dim
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        self.k_neighbors = k_neighbors

        self.neighbor_selector = PersonalityNeighborSelector(
            num_traits=trait_dim,
            num_modalities=num_modalities,
            k=k_neighbors,
        )

        self.input_projection = nn.Linear(embedding_dim + 1, hidden_dim)
        self.layers = nn.ModuleList(
            [SelectiveStateSpaceBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.output_head = MLP([hidden_dim, hidden_dim // 2, 1], dropout=dropout)

    def forward(
        self,
        unlabeled_embeddings: torch.Tensor,
        labeled_embeddings: torch.Tensor,
        labeled_traits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        neighbor_embeddings, neighbor_traits, neighbor_indices = self.neighbor_selector(
            labeled_embeddings=labeled_embeddings,
            labeled_traits=labeled_traits,
            unlabeled_embeddings=unlabeled_embeddings,
        )

        batch_size = unlabeled_embeddings.size(0)
        device = unlabeled_embeddings.device

        neighbor_embeddings = neighbor_embeddings.mean(dim=3)
        unlabeled_condensed = unlabeled_embeddings.mean(dim=1)

        masked_trait = torch.zeros(
            batch_size, self.trait_dim, 1, 1, device=device, dtype=labeled_traits.dtype
        )

        neighbor_features = torch.cat(
            [
                neighbor_embeddings,
                unlabeled_condensed.unsqueeze(1).unsqueeze(2).expand(-1, self.trait_dim, 1, -1),
            ],
            dim=2,
        )

        trait_tokens = torch.cat(
            [neighbor_traits.unsqueeze(-1), masked_trait],
            dim=2,
        )

        sequence_tokens = torch.cat([neighbor_features, trait_tokens], dim=-1)
        batch_trait = batch_size * self.trait_dim
        seq_len = sequence_tokens.size(2)
        sequence_tokens = sequence_tokens.view(batch_trait, seq_len, -1)

        x = self.input_projection(sequence_tokens)
        for norm, layer in zip(self.norms, self.layers):
            x = layer(norm(x))

        masked_state = x[:, -1, :]
        predictions = self.output_head(masked_state).view(batch_size, self.trait_dim)

        aux = {
            "neighbor_indices": neighbor_indices,
            "neighbor_traits": neighbor_traits,
        }
        return predictions, aux
