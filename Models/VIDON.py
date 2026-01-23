import math
import torch
import torch.nn as nn
from tqdm import trange

from .utils.helper_utils import calculate_l2_relative_error

# Implements VIDON from:
# "Variable-Input Deep Operator Networks" (Prasthofer*, De Ryck*, Mishra)


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron with configurable depth.
    Interprets n_layers as the number of Linear layers (incl. output layer).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 4,
        activation_fn: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        layers: list[nn.Module] = []
        if n_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_fn())
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VIDON(nn.Module):
    """
    Variable-Input Deep Operator Network (VIDON), Section 2 in the paper.

    Forward API (consistent with repo):
        forward(xs, us, ys, sensor_mask=None, sensor_weights=None) -> [B, Q, d_out]

    Shapes:
        xs: [B, S, dx]    sensor coordinates
        us: [B, S, du]    sensor values
        ys: [B, Q, dy]    query / trunk inputs
        sensor_mask: [B, S] (bool) True = valid sensor
        sensor_weights: [B, S] (float, nonnegative) optional prior weights (incorporated as log-weights in logits)

    Notes:
        - Permutation-invariant over sensors by construction.
        - Supports variable number of sensors per sample via sensor_mask.
        - Implements τ0(y) term exactly as in the paper by including an extra trunk basis element and
          fixing the corresponding branch coefficient to 1.
    """

    def __init__(
        self,
        input_size_src: int,      # dx
        output_size_src: int,     # du
        input_size_tgt: int,      # dy
        output_size_tgt: int,     # d_out
        p: int = 32,              # number of trunk basis functions excluding τ0
        n_heads: int = 4,         # H
        d_enc: int = 40,          # d_enc (encoding dimension)
        head_output_size: int = 64,   # dimension of each head output (paper uses p; experiments use smaller)
        # Encoders Ψc, Ψv
        enc_hidden_size: int = 40,
        enc_n_layers: int = 4,
        # Head MLPs ωe^{(l)} and νe^{(l)}
        head_hidden_size: int = 128,
        head_n_layers: int = 4,
        # Combiner Φ
        combine_hidden_size: int = 256,
        combine_n_layers: int = 4,
        # Trunk τ (outputs (p+1) * d_out)
        trunk_hidden_size: int = 256,
        n_trunk_layers: int = 4,
        activation_fn: type[nn.Module] = nn.ReLU,
        # Optimizer / schedule (kept consistent with other models)
        initial_lr: float = 5e-4,
        lr_schedule_steps=None,
        lr_schedule_gammas=None,
        # Numerics
        eps: float = 1e-8,
    ):
        super().__init__()

        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if d_enc <= 0:
            raise ValueError(f"d_enc must be positive, got {d_enc}")
        if head_output_size <= 0:
            raise ValueError(f"head_output_size must be positive, got {head_output_size}")
        if output_size_tgt <= 0:
            raise ValueError(f"output_size_tgt must be positive, got {output_size_tgt}")

        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt

        self.p = p
        self.n_heads = n_heads
        self.d_enc = d_enc
        self.head_output_size = head_output_size

        self.eps = float(eps)

        # ---------------------------------------------------------------------
        # Encoders Ψc and Ψv, eq. (2.3): ψ = Ψc(x) + Ψv(u)
        # ---------------------------------------------------------------------
        self.coord_encoder = MLP(
            input_dim=input_size_src,
            hidden_dim=enc_hidden_size,
            output_dim=d_enc,
            n_layers=enc_n_layers,
            activation_fn=activation_fn,
        )
        self.value_encoder = MLP(
            input_dim=output_size_src,
            hidden_dim=enc_hidden_size,
            output_dim=d_enc,
            n_layers=enc_n_layers,
            activation_fn=activation_fn,
        )

        # ---------------------------------------------------------------------
        # Heads: per head ℓ, we have:
        #   νe^(ℓ): R^{d_enc} -> R^{head_output_size}
        #   ωe^(ℓ): R^{d_enc} -> R
        # with normalized weights ω^(ℓ) via softmax across sensors, eq. (2.4),
        # and head output eq. (2.5).
        # ---------------------------------------------------------------------
        self.head_value_nets = nn.ModuleList(
            [
                MLP(
                    input_dim=d_enc,
                    hidden_dim=head_hidden_size,
                    output_dim=head_output_size,
                    n_layers=head_n_layers,
                    activation_fn=activation_fn,
                )
                for _ in range(n_heads)
            ]
        )
        self.head_weight_nets = nn.ModuleList(
            [
                MLP(
                    input_dim=d_enc,
                    hidden_dim=head_hidden_size,
                    output_dim=1,
                    n_layers=head_n_layers,
                    activation_fn=activation_fn,
                )
                for _ in range(n_heads)
            ]
        )

        # ---------------------------------------------------------------------
        # Combiner Φ: R^{H * head_output_size} -> R^{p * d_out}
        # Paper uses output R^p (scalar output); we generalize to vector outputs.
        # ---------------------------------------------------------------------
        self.combiner = MLP(
            input_dim=n_heads * head_output_size,
            hidden_dim=combine_hidden_size,
            output_dim=p * output_size_tgt,
            n_layers=combine_n_layers,
            activation_fn=activation_fn,
        )

        # ---------------------------------------------------------------------
        # Trunk τ: U -> R^{(p+1) * d_out}  (includes τ0)
        # ---------------------------------------------------------------------
        self.trunk = MLP(
            input_dim=input_size_tgt,
            hidden_dim=trunk_hidden_size,
            output_dim=(p + 1) * output_size_tgt,
            n_layers=n_trunk_layers,
            activation_fn=activation_fn,
        )

        # ---------------------------------------------------------------------
        # Optimizer and LR scheduling (consistent with SetONet)
        # ---------------------------------------------------------------------
        self.initial_lr = float(initial_lr)
        self.lr_schedule_steps = None
        self.lr_schedule_rates = None
        self.lr_schedule_gammas = None

        if lr_schedule_steps is not None:
            if lr_schedule_gammas is None or len(lr_schedule_steps) != len(lr_schedule_gammas):
                raise ValueError(
                    "lr_schedule_gammas must be provided and have the same length as lr_schedule_steps if scheduling is used."
                )
            self.lr_schedule_steps = sorted(lr_schedule_steps)
            self.lr_schedule_gammas = lr_schedule_gammas
            self.lr_schedule_rates = [self.initial_lr]
            current_lr = self.initial_lr
            for gamma in lr_schedule_gammas:
                current_lr *= gamma
                self.lr_schedule_rates.append(current_lr)

        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        self.total_steps = 0

        # Holdovers (kept for compatibility with any existing code expecting these)
        self.method = "VIDON"
        self.average_function = None

    @staticmethod
    def _squeeze_last_if_needed(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.dim() == 3 and x.shape[-1] == 1:
            return x.squeeze(-1)
        return x

    def _normalize_sensor_mask(self, sensor_mask: torch.Tensor | None, B: int, S: int, device) -> torch.Tensor | None:
        sensor_mask = self._squeeze_last_if_needed(sensor_mask)
        if sensor_mask is None:
            return None
        if sensor_mask.shape != (B, S):
            raise ValueError(f"sensor_mask must have shape (B, S) = {(B, S)}, got {tuple(sensor_mask.shape)}")
        return sensor_mask.to(device=device).bool()

    def _normalize_sensor_weights(self, sensor_weights: torch.Tensor | None, B: int, S: int, device, dtype) -> torch.Tensor | None:
        sensor_weights = self._squeeze_last_if_needed(sensor_weights)
        if sensor_weights is None:
            return None
        if sensor_weights.shape != (B, S):
            raise ValueError(f"sensor_weights must have shape (B, S) = {(B, S)}, got {tuple(sensor_weights.shape)}")
        w = sensor_weights.to(device=device, dtype=dtype)
        return w

    def forward_branch(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Branch forward per VIDON (Section 2).
        Returns:
            branch_out: [B, p, d_out]
        """
        if xs.dim() != 3 or us.dim() != 3:
            raise ValueError(f"xs and us must be 3D tensors, got xs={xs.shape}, us={us.shape}")

        B, S, dx = xs.shape
        Bu, Su, du = us.shape
        if Bu != B or Su != S:
            raise ValueError(f"xs and us must share (B, S); got xs={xs.shape}, us={us.shape}")
        if dx != self.input_size_src:
            raise ValueError(f"xs last dim must be input_size_src={self.input_size_src}, got {dx}")
        if du != self.output_size_src:
            raise ValueError(f"us last dim must be output_size_src={self.output_size_src}, got {du}")

        device = xs.device
        dtype = xs.dtype

        mask = self._normalize_sensor_mask(sensor_mask, B, S, device)
        weights = self._normalize_sensor_weights(sensor_weights, B, S, device, dtype)

        # ψ = Ψc(x) + Ψv(u), eq. (2.3)
        psi = self.coord_encoder(xs) + self.value_encoder(us)  # [B, S, d_enc]

        # Multi-head: compute head outputs ν^(ℓ), eq. (2.5)
        head_outs: list[torch.Tensor] = []
        scale = 1.0 / math.sqrt(float(self.d_enc))

        for l in range(self.n_heads):
            # logits: [B, S]
            logits = self.head_weight_nets[l](psi).squeeze(-1) * scale

            # Optional sensor prior weights: incorporate as log-weights into logits (still convex after softmax)
            if weights is not None:
                logits = logits + torch.log(weights.clamp_min(self.eps))

            # Mask invalid sensors
            if mask is not None:
                logits = logits.masked_fill(~mask, -1.0e9)

            # Softmax over sensors -> convex weights, eq. (2.4)
            attn = torch.softmax(logits, dim=1)  # [B, S]

            # Ensure masked sensors contribute exactly 0 and renormalize over valid sensors.
            if mask is not None:
                m = mask.to(dtype=attn.dtype)
                attn = attn * m
                attn = attn / (attn.sum(dim=1, keepdim=True) + self.eps)

            # values: [B, S, head_output_size]
            vals = self.head_value_nets[l](psi)

            # weighted sum over sensors: [B, head_output_size], eq. (2.5)
            head_vec = torch.sum(attn.unsqueeze(-1) * vals, dim=1)
            head_outs.append(head_vec)

        # Concatenate heads: [B, H * head_output_size]
        multihead = torch.cat(head_outs, dim=-1)

        # Combine through Φ: [B, p * d_out]
        branch_flat = self.combiner(multihead)

        # Reshape: [B, p, d_out]
        branch_out = branch_flat.reshape(B, self.p, self.output_size_tgt)
        return branch_out

    def forward_trunk(self, ys: torch.Tensor) -> torch.Tensor:
        """
        Trunk forward.
        Returns:
            trunk_out: [B, Q, (p+1), d_out]  (includes τ0)
        """
        if ys.dim() != 3:
            raise ValueError(f"ys must be 3D [B, Q, dy], got {ys.shape}")

        B, Q, dy = ys.shape
        if dy != self.input_size_tgt:
            raise ValueError(f"ys last dim must be input_size_tgt={self.input_size_tgt}, got {dy}")

        trunk_flat = self.trunk(ys)  # [B, Q, (p+1) * d_out]
        trunk_out = trunk_flat.reshape(B, Q, self.p + 1, self.output_size_tgt)
        return trunk_out

    def forward(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        ys: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Full VIDON forward.
        Returns:
            G_u_ys: [B, Q, d_out]
        """
        B = xs.shape[0]

        # Branch: [B, p, d_out]
        b = self.forward_branch(xs, us, sensor_mask=sensor_mask, sensor_weights=sensor_weights)

        # Augment branch with β0 = 1 to include τ0(y) term exactly as in (2.6):
        #   τ0(y) + sum_{k=1}^p β_k τ_k(y)  ==  sum_{k=0}^p β̃_k τ_k(y), with β̃_0 = 1
        ones = torch.ones((B, 1, self.output_size_tgt), device=b.device, dtype=b.dtype)
        b_aug = torch.cat([ones, b], dim=1)  # [B, p+1, d_out]

        # Trunk: [B, Q, p+1, d_out]
        t = self.forward_trunk(ys)

        # Combine via einsum (SetONet-style)
        G_u_y = torch.einsum("bpz,bdpz->bdz", b_aug, t)
        return G_u_y

    # -------------------------------------------------------------------------
    # LR scheduling + training loop (kept consistent with SetONet style)
    # -------------------------------------------------------------------------
    def _get_current_lr(self) -> float:
        if self.lr_schedule_steps is None or self.lr_schedule_rates is None:
            return self.initial_lr

        lr = self.initial_lr
        milestone_idx = -1
        for i, step_milestone in enumerate(self.lr_schedule_steps):
            if self.total_steps >= step_milestone:
                milestone_idx = i
            else:
                break
        if milestone_idx != -1:
            lr = self.lr_schedule_rates[milestone_idx + 1]
        return lr

    def _update_lr(self) -> float:
        new_lr = self._get_current_lr()
        for param_group in self.opt.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def train_model(
        self,
        dataset,
        epochs: int,
        progress_bar: bool = True,
        callback=None,
    ):
        device = next(self.parameters()).device

        if callback is not None:
            callback.on_training_start(locals())

        bar = trange(epochs) if progress_bar else range(epochs)
        for _ in bar:
            current_lr = self._update_lr()

            # dataset.sample should return: xs, us, ys, G_u_ys, sensor_mask
            xs, us, ys, G_u_ys, sensor_mask = dataset.sample(device=device)

            estimated_G_u_ys = self.forward(xs, us, ys, sensor_mask=sensor_mask)
            prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)

            with torch.no_grad():
                pred_flat = (
                    estimated_G_u_ys.squeeze(-1)
                    if estimated_G_u_ys.shape[-1] == 1
                    else estimated_G_u_ys.reshape(estimated_G_u_ys.shape[0], -1)
                )
                target_flat = (
                    G_u_ys.squeeze(-1) if G_u_ys.shape[-1] == 1 else G_u_ys.reshape(G_u_ys.shape[0], -1)
                )
                rel_l2_error = calculate_l2_relative_error(pred_flat, target_flat)

            loss = prediction_loss

            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.opt.step()

            self.total_steps += 1

            if progress_bar:
                bar.set_description(
                    f"Step {self.total_steps} | Loss: {loss.item():.4e} | Rel L2: {rel_l2_error.item():.4f} | "
                    f"Grad Norm: {float(norm):.2f} | LR: {current_lr:.2e}"
                )

            if callback is not None:
                callback.on_step(locals())

        if callback is not None:
            callback.on_training_end(locals())
