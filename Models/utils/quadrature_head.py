from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuadratureHead(nn.Module):
    """
    Quadrature-style branch head for SetONet.

    Key idea: represent learned test functions φ_k(x) via learned query tokens
    against position-only keys, then compute additive quadrature sums

        b_k = Σ_i w_i φ_k(x_i) V(x_i, u_i)

    Unlike attention pooling over sensors, φ_k(x_i) does not normalize over sensors
    (so refining the sensor set can converge to a continuum integral when weights
    correspond to quadrature cell volumes).
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
        key_hidden: int | None = None,
        key_layers: int = 3,
        phi_activation: str = "tanh",
        value_mode: str = "linear_u",
        eps: float = 1e-8,
        normalize: str = "total",
        learn_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.p = int(p)
        self.dk = int(dk)
        self.dv = int(dv)
        self.dout = int(dout)
        self.eps = float(eps)
        self.normalize = normalize.lower()
        self.learn_temperature = bool(learn_temperature)
        self.phi_activation = phi_activation.lower()
        self.value_mode = value_mode.lower()

        if self.normalize not in ["none", "total", "token"]:
            raise ValueError(f"normalize must be 'none', 'total', or 'token', got {normalize}")
        if self.phi_activation not in ["tanh", "softsign", "softplus"]:
            raise ValueError(
                f"phi_activation must be 'tanh', 'softsign', or 'softplus', got {phi_activation}"
            )
        if self.value_mode not in ["linear_u", "mlp_u", "mlp_xu"]:
            raise ValueError(
                f"value_mode must be 'linear_u', 'mlp_u', or 'mlp_xu', got {value_mode}"
            )

        # Position-only key network: defines test/basis dependence on x
        key_hidden_dim = int(hidden if key_hidden is None else key_hidden)
        key_layers = int(key_layers)
        if key_layers < 2:
            raise ValueError(f"key_layers must be >= 2, got {key_layers}")

        key_layers_list = [nn.Linear(dx_enc, key_hidden_dim), activation_fn()]
        for _ in range(key_layers - 2):
            key_layers_list.append(nn.Linear(key_hidden_dim, key_hidden_dim))
            key_layers_list.append(activation_fn())
        key_layers_list.append(nn.Linear(key_hidden_dim, dk))
        self.key_net = nn.Sequential(*key_layers_list)

        # Value network: configurable input and depth
        if self.value_mode == "linear_u":
            self.value_net = nn.Linear(du, dv)
            self._value_includes_x = False
        elif self.value_mode == "mlp_u":
            self.value_net = nn.Sequential(
                nn.Linear(du, hidden),
                activation_fn(),
                nn.Linear(hidden, hidden),
                activation_fn(),
                nn.Linear(hidden, dv),
            )
            self._value_includes_x = False
        else:
            self.value_net = nn.Sequential(
                nn.Linear(dx_enc + du, hidden),
                activation_fn(),
                nn.Linear(hidden, hidden),
                activation_fn(),
                nn.Linear(hidden, dv),
            )
            self._value_includes_x = True

        # Learnable query tokens (test functions)
        self.query_tokens = nn.Parameter(torch.randn(1, p, dk))

        # Optional learnable temperature (softness of φ_k activation)
        if self.learn_temperature:
            self.log_tau = nn.Parameter(torch.zeros(1))
        else:
            self.log_tau = None

        # Map value embeddings to branch coefficient channels
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
        if self._value_includes_x:
            V = self.value_net(torch.cat([x_enc, u], dim=-1))  # (B, N, dv)
        else:
            V = self.value_net(u)  # (B, N, dv)

        # 2) Token parameters
        Q = self.query_tokens.expand(batch_size, -1, -1)  # (B, p, dk)

        # 3) Scores and test function evaluation φ_k(x_i) >= 0
        scores = torch.einsum("bpk,bnk->bpn", Q, K) / math.sqrt(self.dk)  # (B, p, N)
        if self.learn_temperature:
            tau = torch.exp(self.log_tau) + self.eps
            scores = scores / tau

        if self.phi_activation == "tanh":
            Phi = torch.tanh(scores)  # (B, p, N)
        elif self.phi_activation == "softsign":
            Phi = scores / (1.0 + scores.abs())  # (B, p, N)
        else:
            Phi = F.softplus(scores)  # (B, p, N)

        # 4) Quadrature weights and mask
        if sensor_weights is None:
            w = torch.ones((batch_size, n_sensors), device=x_enc.device, dtype=x_enc.dtype)
        else:
            w = torch.clamp(sensor_weights, min=0.0).to(dtype=x_enc.dtype)

        if sensor_mask is not None:
            m = sensor_mask.to(dtype=x_enc.dtype)
            w = w * m
            V = V * m.unsqueeze(-1)
            Phi = Phi * m.unsqueeze(1)

        # 5) Quadrature sum: Σ_i w_i φ_k(x_i) V_i
        pooled = torch.einsum("bpn,bn,bnd->bpd", Phi, w, V)  # (B, p, dv)

        # 6) Optional normalization
        if self.normalize == "total":
            denom = w.sum(dim=1).clamp_min(self.eps)  # (B,)
            pooled = pooled / denom.view(batch_size, 1, 1)
        elif self.normalize == "token":
            mass = torch.einsum("bpn,bn->bp", Phi, w).clamp_min(self.eps)  # (B, p)
            pooled = pooled / mass.unsqueeze(-1)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize option: {self.normalize}")

        return self.rho_token(pooled)  # (B, p, dout)
