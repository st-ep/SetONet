from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PetrovGalerkinHead(nn.Module):
    """
    Petrov-Galerkin attention head for SetONet branch network.

    Uses learned query tokens as "test functions" to compute weighted projections
    of sensor embeddings via attention mechanism.
    """

    def __init__(
        self,
        *,
        p: int,
        dx_enc: int,
        du: int,
        dout: int,
        dk: int,
        dv: int,
        hidden: int,
        activation_fn,
        eps: float = 1e-8,
        n_refine_iters: int = 0,  # Number of query refinement iterations (0 = disabled)
    ) -> None:
        super().__init__()

        self.p = int(p)
        self.dk = int(dk)
        self.dv = int(dv)
        self.dout = int(dout)
        self.eps = float(eps)
        self.n_refine_iters = int(n_refine_iters)

        self.key_net = nn.Sequential(
            nn.Linear(dx_enc, hidden),
            activation_fn(),
            nn.Linear(hidden, hidden),
            activation_fn(),
            nn.Linear(hidden, dk),
        )

        self.value_net = nn.Sequential(
            nn.Linear(dx_enc + du, hidden),
            activation_fn(),
            nn.Linear(hidden, hidden),
            activation_fn(),
            nn.Linear(hidden, dv),
        )

        self.query_tokens = nn.Parameter(torch.randn(1, p, dk))

        self.rho_token = (
            nn.Identity()
            if dv == dout
            else nn.Sequential(
                nn.Linear(dv, hidden),
                activation_fn(),
                nn.Linear(hidden, dout),
            )
        )

        # Iterative query refinement layers (only created if n_refine_iters > 0)
        # Uses residual update: Q = LayerNorm(Q + MLP(pooled))
        self.refine_mlp = None
        self.refine_ln = None
        if self.n_refine_iters > 0:
            self.refine_mlp = nn.Sequential(
                nn.Linear(dv, hidden),
                activation_fn(),
                nn.Linear(hidden, dk),
            )
            self.refine_ln = nn.LayerNorm(dk)

    def forward(
        self,
        x_enc: torch.Tensor,
        u: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        sensor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_enc: (B, N, dx_enc) - encoded sensor positions
            u:     (B, N, du) - sensor values
            sensor_mask: (B, N) bool, True = valid sensor
            sensor_weights: (B, N) nonnegative weights (optional)
        Returns:
            b: (B, p, dout)
        """
        if x_enc.dim() != 3 or u.dim() != 3:
            raise ValueError(f"x_enc and u must be 3D, got {x_enc.shape=} and {u.shape=}")
        if x_enc.shape[:2] != u.shape[:2]:
            raise ValueError(f"x_enc and u must share (B, N), got {x_enc.shape=} and {u.shape=}")

        batch_size, n_sensors = x_enc.shape[0], x_enc.shape[1]

        if sensor_mask is not None:
            if sensor_mask.dim() == 3 and sensor_mask.shape[-1] == 1:
                sensor_mask = sensor_mask.squeeze(-1)
            if sensor_mask.shape != (batch_size, n_sensors):
                raise ValueError(f"{sensor_mask.shape=} must be (B, N) = {(batch_size, n_sensors)}")
            sensor_mask = sensor_mask.to(device=x_enc.device).bool()

        if sensor_weights is not None:
            if sensor_weights.dim() == 3 and sensor_weights.shape[-1] == 1:
                sensor_weights = sensor_weights.squeeze(-1)
            if sensor_weights.shape != (batch_size, n_sensors):
                raise ValueError(f"{sensor_weights.shape=} must be (B, N) = {(batch_size, n_sensors)}")
            sensor_weights = sensor_weights.to(device=x_enc.device, dtype=x_enc.dtype)

        # 1) Keys / values (computed once, reused across iterations)
        K = self.key_net(x_enc)  # (B, N, dk)
        V = self.value_net(torch.cat([x_enc, u], dim=-1))  # (B, N, dv)

        # 2) Initialize queries
        Q = self.query_tokens.expand(batch_size, -1, -1)  # (B, p, dk)

        # 3) Iterative refinement loop
        # n_refine_iters=0: just one pass (original behavior)
        # n_refine_iters=1: two passes (initial + 1 refinement)
        # etc.
        for iter_idx in range(self.n_refine_iters + 1):
            # Compute attention scores
            scores = torch.einsum("bpk,bnk->bpn", Q, K) / math.sqrt(self.dk)  # (B, p, N)

            # Optional quadrature weights in log-space
            if sensor_weights is not None:
                scores = scores + torch.log(sensor_weights.clamp_min(self.eps)).unsqueeze(1)

            # Mask padded sensors
            if sensor_mask is not None:
                scores = scores.masked_fill(~sensor_mask.unsqueeze(1), -float("inf"))

            # Normalized attention weights
            A = torch.softmax(scores, dim=-1)  # (B, p, N)

            # Weighted sum / coefficients in value space
            pooled = torch.einsum("bpn,bnd->bpd", A, V)  # (B, p, dv)

            # Refine queries for next iteration (skip on final iteration)
            if self.refine_mlp is not None and iter_idx < self.n_refine_iters:
                delta_Q = self.refine_mlp(pooled)  # (B, p, dk)
                Q = self.refine_ln(Q + delta_Q)  # Residual + LayerNorm

        # 4) Token-wise map to output coefficient dimension
        return self.rho_token(pooled)  # (B, p, dout)

