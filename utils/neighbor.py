from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class PersonalityNeighborSelector(nn.Module):
    """Multi-head personality-aware neighbor selector.

    Follows the paper's design where each Big Five dimension maintains its own
    notion of semantic proximity over multimodal embeddings. The module takes
    normalized features from a labeled (trait annotated) corpus and an unlabeled
    corpus, and retrieves ``K`` nearest neighbors per trait dimension for every
    unlabeled sample.
    """

    def __init__(self, num_traits: int = 5, num_modalities: int = 3, k: int = 8, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_traits = num_traits
        self.num_modalities = num_modalities
        self.k = k
        self.eps = eps
        # Learnable modality importance per trait dimension.
        self.trait_modality_logits = nn.Parameter(torch.zeros(num_traits, num_modalities))

    def forward(
        self,
        labeled_embeddings: torch.Tensor,
        labeled_traits: torch.Tensor,
        unlabeled_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return neighbor features, traits and indices.

        Args:
            labeled_embeddings: Tensor [N_labeled, M, D] with modality-specific features.
            labeled_traits: Tensor [N_labeled, num_traits] containing OCEAN annotations.
            unlabeled_embeddings: Tensor [N_unlabeled, M, D] with modality-specific features.

        Returns:
            neighbor_embeddings: Tensor [N_unlabeled, num_traits, k, M, D].
            neighbor_traits: Tensor [N_unlabeled, num_traits, k].
            neighbor_indices: Long tensor [N_unlabeled, num_traits, k].
        """

        if labeled_embeddings.dim() != 3 or unlabeled_embeddings.dim() != 3:
            raise ValueError("Embeddings must be [batch, modality, dim]")
        if labeled_embeddings.size(1) != self.num_modalities:
            raise ValueError("Mismatch between configured modalities and labeled embeddings")
        if unlabeled_embeddings.size(1) != self.num_modalities:
            raise ValueError("Mismatch between configured modalities and unlabeled embeddings")
        if labeled_traits.size(1) != self.num_traits:
            raise ValueError("Trait dimension mismatch")

        n_labeled, _, dim = labeled_embeddings.shape
        n_unlabeled = unlabeled_embeddings.size(0)

        labeled_norm = labeled_embeddings / (labeled_embeddings.norm(dim=-1, keepdim=True) + self.eps)
        unlabeled_norm = unlabeled_embeddings / (unlabeled_embeddings.norm(dim=-1, keepdim=True) + self.eps)

        neighbor_indices = []
        modality_weights = torch.softmax(self.trait_modality_logits, dim=-1)

        # Pre-compute modality-wise similarity matrices to avoid recomputation.
        modality_sim = []
        for m in range(self.num_modalities):
            # [N_unlabeled, dim] @ [dim, N_labeled] -> [N_unlabeled, N_labeled]
            sim = torch.matmul(unlabeled_norm[:, m, :], labeled_norm[:, m, :].transpose(0, 1))
            modality_sim.append(sim)

        neighbor_embeddings = torch.zeros(
            n_unlabeled,
            self.num_traits,
            self.k,
            self.num_modalities,
            dim,
            device=labeled_embeddings.device,
            dtype=labeled_embeddings.dtype,
        )
        neighbor_traits = torch.zeros(
            n_unlabeled,
            self.num_traits,
            self.k,
            device=labeled_traits.device,
            dtype=labeled_traits.dtype,
        )

        neighbor_indices_tensor = torch.zeros(
            n_unlabeled,
            self.num_traits,
            self.k,
            device=labeled_embeddings.device,
            dtype=torch.long,
        )

        for trait_idx in range(self.num_traits):
            weights = modality_weights[trait_idx]
            combined_sim = 0.0
            for m in range(self.num_modalities):
                combined_sim = combined_sim + weights[m] * modality_sim[m]

            topk = torch.topk(combined_sim, k=min(self.k, n_labeled), dim=-1)
            neighbor_indices_tensor[:, trait_idx, : topk.indices.size(-1)] = topk.indices

            gathered_embeddings = labeled_embeddings[topk.indices]  # [N_unlabeled, k, M, D]
            gathered_traits = labeled_traits[topk.indices, trait_idx]  # [N_unlabeled, k]

            neighbor_embeddings[:, trait_idx, : gathered_embeddings.size(1)] = gathered_embeddings
            neighbor_traits[:, trait_idx, : gathered_traits.size(1)] = gathered_traits

        return neighbor_embeddings, neighbor_traits, neighbor_indices_tensor
