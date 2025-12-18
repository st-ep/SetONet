from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveQuadratureHead(nn.Module):
    """
    Input-adaptive quadrature branch head for SetONet.

    Replaces fixed query tokens Q^(0) with input-adaptive tokens:
        Q(g) = Q^(0) + ΔQ(g),
    where ΔQ(g) is a low-rank update predicted from a permutation-invariant global
    context vector built from value embeddings V(x_i, u_i).

    Pooling uses additive quadrature sums (no softmax over sensors).
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
        adapt_rank: int = 4,
        adapt_hidden: Optional[int] = 64,
        adapt_scale: float = 0.1,
        use_value_context: bool = True,
        normalize: str = "total",
        learn_temperature: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.p = int(p)
        self.dk = int(dk)
        self.dv = int(dv)
        self.dout = int(dout)
        self.eps = float(eps)
        self.normalize = normalize.lower()
        self.learn_temperature = bool(learn_temperature)

        self.use_value_context = bool(use_value_context)
        if not self.use_value_context:
            raise NotImplementedError("Only value-based context is supported (use_value_context=True).")

        self.adapt_rank = int(adapt_rank)
        if self.adapt_rank <= 0:
            raise ValueError(f"adapt_rank must be positive, got {adapt_rank}")
        self.adapt_scale = float(adapt_scale)

        if self.normalize not in ["none", "total", "token"]:
            raise ValueError(f"normalize must be 'none', 'total', or 'token', got {normalize}")

        # Position-only key network: defines test/basis dependence on x
        self.key_net = nn.Sequential(
            nn.Linear(dx_enc, hidden),
            activation_fn(),
            nn.Linear(hidden, hidden),
            activation_fn(),
            nn.Linear(hidden, dk),
        )

        # Value network: defines integrand dependence on (x, u)
        self.value_net = nn.Sequential(
            nn.Linear(dx_enc + du, hidden),
            activation_fn(),
            nn.Linear(hidden, hidden),
            activation_fn(),
            nn.Linear(hidden, dv),
        )

        # Base learnable query tokens (test functions)
        self.query_tokens = nn.Parameter(torch.randn(1, p, dk))

        # Low-rank token adaptation: ΔQ = a(c) @ B_dirs
        if adapt_hidden is None:
            adapt_hidden = min(64, int(hidden))
        self.adapt_hidden = int(adapt_hidden)
        if self.adapt_hidden <= 0:
            raise ValueError(f"adapt_hidden must be positive, got {adapt_hidden}")

        self.B_dirs = nn.Parameter(torch.randn(self.adapt_rank, dk))
        self.adapter_net = nn.Sequential(
            nn.Linear(dv, self.adapt_hidden),
            activation_fn(),
            nn.Linear(self.adapt_hidden, p * self.adapt_rank),
        )

        # Optional learnable temperature (softness of φ_k activation)
        if self.learn_temperature:
            self.log_tau = nn.Parameter(torch.zeros(1))
        else:
            self.log_tau = None

        # Map pooled value embeddings to branch coefficient channels
        self.rho_token = (
            nn.Identity()
            if dv == dout
            else nn.Sequential(
                nn.Linear(dv, hidden),
                activation_fn(),
                nn.Linear(hidden, dout),
            )
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        u: torch.Tensor,
        sensor_mask: Optional[torch.Tensor] = None,
        sensor_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_enc: (B, N, dx_enc) encoded sensor coordinates
            u:     (B, N, du) sensor values
            sensor_mask: (B, N) bool, True = valid sensor (optional)
            sensor_weights: (B, N) nonnegative quadrature weights (optional)
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

        # 1) Compute keys/values
        K = self.key_net(x_enc)  # (B, N, dk)
        V = self.value_net(torch.cat([x_enc, u], dim=-1))  # (B, N, dv)

        # 2) Quadrature weights and mask (match QuadratureHead semantics)
        if sensor_weights is None:
            w = torch.ones((batch_size, n_sensors), device=x_enc.device, dtype=x_enc.dtype)
        else:
            w = torch.clamp(sensor_weights, min=0.0).to(dtype=x_enc.dtype)

        if sensor_mask is not None:
            m = sensor_mask.to(dtype=x_enc.dtype)
            w = w * m
            V = V * m.unsqueeze(-1)

        # 3) Permutation-invariant global context c from value embeddings
        denom = w.sum(dim=1).clamp_min(self.eps)  # (B,)
        c = (w.unsqueeze(-1) * V).sum(dim=1) / denom.unsqueeze(-1)  # (B, dv)

        # 4) Low-rank adaptive query update
        a_raw = self.adapter_net(c).reshape(batch_size, self.p, self.adapt_rank)  # (B, p, R)
        a = torch.tanh(a_raw) * self.adapt_scale  # bounded update
        delta_Q = torch.einsum("bpr,rk->bpk", a, self.B_dirs)  # (B, p, dk)

        Q = self.query_tokens.expand(batch_size, -1, -1) + delta_Q  # (B, p, dk)

        # 5) Test functions and quadrature pooling
        scores = torch.einsum("bpk,bnk->bpn", Q, K) / math.sqrt(self.dk)  # (B, p, N)
        if self.learn_temperature:
            tau = torch.exp(self.log_tau) + self.eps
            scores = scores / tau
        Phi = F.softplus(scores)  # (B, p, N)

        if sensor_mask is not None:
            Phi = Phi * sensor_mask.to(dtype=x_enc.dtype).unsqueeze(1)

        pooled = torch.einsum("bpn,bn,bnd->bpd", Phi, w, V)  # (B, p, dv)

        # 6) Optional normalization (match QuadratureHead)
        if self.normalize == "total":
            pooled = pooled / denom.view(batch_size, 1, 1)
        elif self.normalize == "token":
            mass = torch.einsum("bpn,bn->bp", Phi, w).clamp_min(self.eps)  # (B, p)
            pooled = pooled / mass.unsqueeze(-1)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize option: {self.normalize}")

        return self.rho_token(pooled)  # (B, p, dout)

