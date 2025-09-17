import torch
from torch import nn


class LearnedQueryBank(nn.Module):
    """Repository of learnable query embeddings used in TEADM blocks."""

    def __init__(self, num_queries: int, dim: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.queries.unsqueeze(0).expand(batch_size, -1, -1)
