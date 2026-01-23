from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class GalerkinPoUHead(nn.Module):
    """
    Galerkin partition-of-unity (PoU) head for SetONet branch network.

    Computes Galerkin-style quadrature sums using a learned partition-of-unity over tokens.

    KEY DIFFERENCE from attention pooling / PetrovGalerkinHead:
    - Old heads: softmax over sensors → "each token selects sensors then averages" (conditional expectation)
    - This head: softmax over tokens → "each sensor assigns its mass to tokens via PoU basis evaluation",
      then coefficients are quadrature sums over sensors.

    Mathematical interpretation:
    - Tokens k represent basis/test functions φ_k
    - Sensors i represent quadrature points x_i with weights w_i
    - Φ_{k,i} = softmax_k(Q_k · K_i / √d_k) enforces partition-of-unity: Σ_k Φ_{k,i} = 1 for each sensor i
    - Coefficient: c_k = Σ_i w_i Φ_{k,i} V_i  (Galerkin quadrature sum)

    This makes the key/value split principled:
    - Keys depend only on sensor position x (basis evaluation depends only on spatial location)
    - Values depend on (x, u) (the integrand carries both position and sensor value)
    - Weights w_i multiply the integrand in the sum (standard quadrature), NOT added to logits
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
        normalize: str = "total",
        learn_temperature: bool = False,
    ) -> None:
        """
        Args:
            p: Number of tokens / output coefficients (matches SetONet latent dimension)
            dx_enc: Dimension of encoded sensor coordinate (pos encoding dim or raw x dim)
            du: Sensor value dimension
            dout: Branch output channel dimension (matches SetONet output_size_tgt)
            dk: Key/query dimension
            dv: Value embedding dimension
            hidden: MLP hidden layer width
            activation_fn: Activation function class (e.g., nn.ReLU)
            eps: Small constant for numerical stability
            normalize: Normalization strategy for quadrature sums
                - "none": c_k = Σ_i w_i Φ_{k,i} V_i
                - "total": c_k = (Σ_i w_i Φ_{k,i} V_i) / (Σ_i w_i)  [recommended; scale-invariant w.r.t. M]
                - "token": c_k = (Σ_i w_i Φ_{k,i} V_i) / (Σ_i w_i Φ_{k,i})  [per-token mass normalization]
            learn_temperature: If True, learn a scalar temperature parameter for softmax sharpness
        """
        super().__init__()

        self.p = int(p)
        self.dk = int(dk)
        self.dv = int(dv)
        self.dout = int(dout)
        self.eps = float(eps)
        self.normalize = normalize.lower()
        self.learn_temperature = learn_temperature

        if self.normalize not in ["none", "total", "token"]:
            raise ValueError(f"normalize must be 'none', 'total', or 'token', got {normalize}")

        # Key network: maps sensor positions to key space
        # In Galerkin view: encodes basis evaluation dependency on spatial location
        self.key_net = nn.Sequential(
            nn.Linear(dx_enc, hidden),
            activation_fn(),
            nn.Linear(hidden, hidden),
            activation_fn(),
            nn.Linear(hidden, dk),
        )

        # Value network: maps (position, sensor_value) to value space
        # In Galerkin view: encodes the integrand at each quadrature point
        self.value_net = nn.Sequential(
            nn.Linear(dx_enc + du, hidden),
            activation_fn(),
            nn.Linear(hidden, hidden),
            activation_fn(),
            nn.Linear(hidden, dv),
        )

        # Learnable query tokens (basis/test function parameters)
        self.query_tokens = nn.Parameter(torch.randn(1, p, dk))

        # Optional learnable temperature parameter
        # Stored as log_tau for numerical stability; tau = exp(log_tau) + eps
        if self.learn_temperature:
            self.log_tau = nn.Parameter(torch.zeros(1))
        else:
            self.log_tau = None

        # Output projection: maps value embeddings to output dimension
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
            x_enc: (B, N, dx_enc) - encoded sensor positions
            u:     (B, N, du) - sensor values
            sensor_mask: (B, N) bool, True = valid sensor
            sensor_weights: (B, N) nonnegative quadrature weights (optional)
        Returns:
            b: (B, p, dout) - branch output coefficients
        """
        if x_enc.dim() != 3 or u.dim() != 3:
            raise ValueError(f"x_enc and u must be 3D, got {x_enc.shape=} and {u.shape=}")
        if x_enc.shape[:2] != u.shape[:2]:
            raise ValueError(f"x_enc and u must share (B, N), got {x_enc.shape=} and {u.shape=}")

        batch_size, n_sensors = x_enc.shape[0], x_enc.shape[1]

        # Handle sensor mask (may come as (B, N, 1), squeeze to (B, N))
        if sensor_mask is not None:
            if sensor_mask.dim() == 3 and sensor_mask.shape[-1] == 1:
                sensor_mask = sensor_mask.squeeze(-1)
            if sensor_mask.shape != (batch_size, n_sensors):
                raise ValueError(f"{sensor_mask.shape=} must be (B, N) = {(batch_size, n_sensors)}")
            sensor_mask = sensor_mask.to(device=x_enc.device).bool()

        # Handle sensor weights (may come as (B, N, 1), squeeze to (B, N))
        if sensor_weights is not None:
            if sensor_weights.dim() == 3 and sensor_weights.shape[-1] == 1:
                sensor_weights = sensor_weights.squeeze(-1)
            if sensor_weights.shape != (batch_size, n_sensors):
                raise ValueError(f"{sensor_weights.shape=} must be (B, N) = {(batch_size, n_sensors)}")
            sensor_weights = sensor_weights.to(device=x_enc.device, dtype=x_enc.dtype)

        # 1) Compute keys and values
        K = self.key_net(x_enc)  # (B, N, dk)
        V = self.value_net(torch.cat([x_enc, u], dim=-1))  # (B, N, dv)

        # 2) Initialize queries (learnable token parameters)
        Q = self.query_tokens.expand(batch_size, -1, -1)  # (B, p, dk)

        # 3) Compute attention scores
        scores = torch.einsum("bpk,bnk->bpn", Q, K) / math.sqrt(self.dk)  # (B, p, N)

        # Apply learnable temperature if enabled
        if self.learn_temperature:
            tau = torch.exp(self.log_tau) + self.eps
            scores = scores / tau

        # 4) CRITICAL: Partition-of-unity over TOKENS for each sensor
        # Softmax over dim=1 (token axis) ensures Σ_k Φ_{k,i} = 1 for each sensor i
        Phi = torch.softmax(scores, dim=1)  # (B, p, N)

        # 5) Prepare quadrature weights per sensor
        if sensor_weights is None:
            w = torch.ones((batch_size, n_sensors), device=x_enc.device, dtype=x_enc.dtype)
        else:
            # Ensure nonnegative weights
            w = torch.clamp(sensor_weights, min=0.0).to(dtype=x_enc.dtype)

        # Apply mask to weights (invalid sensors contribute zero weight)
        if sensor_mask is not None:
            w = w * sensor_mask.to(dtype=x_enc.dtype)
            # Also zero out values for invalid sensors to be extra safe
            V = V * sensor_mask.unsqueeze(-1).to(dtype=x_enc.dtype)

        # 6) Galerkin quadrature sum: c_k = Σ_i w_i Φ_{k,i} V_i
        # einsum: for each batch b, token p: sum over sensors n of Phi[b,p,n] * w[b,n] * V[b,n,d]
        pooled = torch.einsum("bpn,bn,bnd->bpd", Phi, w, V)  # (B, p, dv)

        # 7) Optional normalization
        if self.normalize == "total":
            # Divide by total quadrature weight: scale-invariant w.r.t. number of sensors
            denom = w.sum(dim=1).clamp_min(self.eps)  # (B,)
            pooled = pooled / denom.view(batch_size, 1, 1)
        elif self.normalize == "token":
            # Divide by per-token mass: Σ_i w_i Φ_{k,i}
            mass = torch.einsum("bpn,bn->bp", Phi, w).clamp_min(self.eps)  # (B, p)
            pooled = pooled / mass.unsqueeze(-1)
        elif self.normalize == "none":
            pass  # No normalization
        else:
            raise ValueError(f"Unknown normalize option: {self.normalize}")

        # 8) Map to output dimension
        b = self.rho_token(pooled)  # (B, p, dout)

        return b
