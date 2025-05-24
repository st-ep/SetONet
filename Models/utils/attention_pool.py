import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    k‑token multi‑head attention aggregator (Set‑Transformer style).
    If `n_tokens = 1` this is identical to the old single‑token pool.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns pooled : (B, n_tokens * d_model) after flattening
        """
        B = x.size(0)
        q = self.query_tokens.expand(B, -1, -1)         # (B, k, d)
        pooled, _ = self.attn(q, x, x)                  # (B, k, d)
        return pooled.flatten(1)                        # (B, k·d) 