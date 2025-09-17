from torch import nn


class LayerNorm(nn.Module):
    """Layer normalization wrapper with eps tuned for sequence modeling."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
