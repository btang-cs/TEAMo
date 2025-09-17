import math
from typing import Optional

import torch
from torch import nn


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention with residual connection.

    This module follows the description from the TEAMo paper where
    cross-attention is used to align learned queries with conditioning
    features. It accepts a set of queries Q and context K/V tensors,
    and returns attention-refined queries of the same shape as Q.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute cross-attention.

        Args:
            query: Tensor of shape [batch, num_queries, dim].
            context: Tensor of shape [batch, context_len, dim].
            mask: Optional boolean tensor of shape [batch, context_len] where True
                indicates positions to mask.

        Returns:
            Tensor of shape [batch, num_queries, dim] with attention-refined queries.
        """

        batch, q_len, _ = query.size()
        _, ctx_len, _ = context.size()

        q = self.q_proj(query).view(batch, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(batch, ctx_len, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(batch, ctx_len, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)  # [batch, heads, q_len, head_dim]
        k = k.transpose(1, 2)  # [batch, heads, ctx_len, head_dim]
        v = v.transpose(1, 2)  # [batch, heads, ctx_len, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        attended = torch.matmul(attn, v)
        attended = attended.transpose(1, 2).contiguous().view(batch, q_len, self.dim)
        output = self.out_proj(attended)
        return output + query
