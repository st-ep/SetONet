from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import trange

from .SetONet import SetONet
from .utils.helper_utils import calculate_l2_relative_error


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, activation_fn):
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers = [nn.Linear(in_dim, hidden_dim), activation_fn()]
    for _ in range(n_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class KoopmanSetONet(nn.Module):
    """
    Lifted linear operator: nonlinear encoders/decoders + linear operator in latent space.
    - Encode u -> z_u
    - Linear operator A in latent: z_s = A z_u (SetONet with quadrature head, linear in z_u)
    - Decode z_s -> s
    """

    def __init__(
        self,
        *,
        input_size_src: int,
        output_size_src: int,
        input_size_tgt: int,
        output_size_tgt: int,
        latent_dim: int = 16,
        lowrank_r: int | None = None,
        encoder_hidden: int = 64,
        encoder_layers: int = 2,
        encoder_type: str = "mlp",
        encoder_transformer_dim: int | None = None,
        encoder_transformer_layers: int = 2,
        encoder_transformer_heads: int = 4,
        encoder_transformer_dropout: float = 0.0,
        encoder_transformer_ff: int | None = None,
        decoder_hidden: int = 64,
        decoder_layers: int = 2,
        activation_fn=nn.ReLU,
        initial_lr: float = 5e-4,
        lr_schedule_steps=None,
        lr_schedule_gammas=None,
        # SetONet operator params
        p: int = 32,
        trunk_hidden_size: int = 256,
        n_trunk_layers: int = 4,
        pos_encoding_type: str = "sinusoidal",
        pos_encoding_dim: int = 64,
        pos_encoding_max_freq: float = 0.1,
        quad_dk: int | None = None,
        quad_key_hidden: int | None = None,
        quad_key_layers: int = 3,
        quad_phi_activation: str = "softplus",
        quad_value_mode: str = "linear_u",
        quad_normalize: str = "total",
        quad_learn_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt
        self.latent_dim = int(latent_dim)
        self.lowrank_r = None if lowrank_r is None else int(lowrank_r)
        if self.lowrank_r is not None and self.lowrank_r <= 0:
            raise ValueError("lowrank_r must be a positive integer or None.")

        self.encoder_type = encoder_type
        if self.encoder_type not in {"mlp", "transformer"}:
            raise ValueError("encoder_type must be 'mlp' or 'transformer'.")

        # Encoder for u
        if self.encoder_type == "transformer":
            d_model = encoder_transformer_dim or encoder_hidden
            if d_model <= 0:
                raise ValueError("encoder_transformer_dim must be positive.")
            if encoder_transformer_heads <= 0:
                raise ValueError("encoder_transformer_heads must be positive.")
            if d_model % encoder_transformer_heads != 0:
                raise ValueError("encoder_transformer_heads must divide encoder_transformer_dim.")
            ff_dim = encoder_transformer_ff or 4 * d_model
            act = "gelu" if activation_fn == nn.GELU else "relu"
            self.encoder_in = nn.Linear(input_size_src + output_size_src, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=encoder_transformer_heads,
                dim_feedforward=ff_dim,
                dropout=encoder_transformer_dropout,
                activation=act,
                batch_first=True,
            )
            self.encoder_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=encoder_transformer_layers,
            )
            self.encoder_out = nn.Linear(d_model, self.latent_dim) if d_model != self.latent_dim else nn.Identity()
            self.encoder_u = None
        else:
            # Encoders/decoders (pointwise)
            self.encoder_u = _build_mlp(output_size_src, encoder_hidden, self.latent_dim, encoder_layers, activation_fn)
            self.encoder_in = None
            self.encoder_transformer = None
            self.encoder_out = None

        self.decoder_u = _build_mlp(self.latent_dim, decoder_hidden, output_size_src, decoder_layers, activation_fn)
        self.encoder_s = _build_mlp(output_size_tgt, encoder_hidden, self.latent_dim, encoder_layers, activation_fn)
        self.decoder_s = _build_mlp(self.latent_dim, decoder_hidden, output_size_tgt, decoder_layers, activation_fn)

        operator_dim = self.latent_dim
        if self.lowrank_r is not None and self.lowrank_r < self.latent_dim:
            operator_dim = self.lowrank_r
            self.lowrank_in = nn.Linear(self.latent_dim, operator_dim, bias=False)
            self.lowrank_out = nn.Linear(operator_dim, self.latent_dim, bias=False)
        else:
            self.lowrank_r = None
            self.lowrank_in = nn.Identity()
            self.lowrank_out = nn.Identity()

        # Linear operator in latent space (quadrature head, linear in z_u)
        self.operator = SetONet(
            input_size_src=input_size_src,
            output_size_src=operator_dim,
            input_size_tgt=input_size_tgt,
            output_size_tgt=operator_dim,
            p=p,
            phi_hidden_size=256,
            rho_hidden_size=256,
            trunk_hidden_size=trunk_hidden_size,
            n_trunk_layers=n_trunk_layers,
            activation_fn=activation_fn,
            use_deeponet_bias=False,
            phi_output_size=32,
            initial_lr=initial_lr,
            lr_schedule_steps=lr_schedule_steps,
            lr_schedule_gammas=lr_schedule_gammas,
            pos_encoding_type=pos_encoding_type,
            pos_encoding_dim=pos_encoding_dim,
            pos_encoding_max_freq=pos_encoding_max_freq,
            aggregation_type="mean",
            use_positional_encoding=(pos_encoding_type != "skip"),
            attention_n_tokens=1,
            branch_head_type="quadrature",
            quad_dk=quad_dk,
            quad_dv=self.latent_dim,
            quad_key_hidden=quad_key_hidden,
            quad_key_layers=quad_key_layers,
            quad_phi_activation=quad_phi_activation,
            quad_value_mode=quad_value_mode,
            quad_normalize=quad_normalize,
            quad_learn_temperature=quad_learn_temperature,
        )

        # Force rho_token to be linear/identity via dv == dout above.
        self.initial_lr = initial_lr
        self.lr_schedule_steps = lr_schedule_steps
        self.lr_schedule_gammas = lr_schedule_gammas
        self.lr_schedule_rates = None
        if lr_schedule_steps is not None:
            if lr_schedule_gammas is None or len(lr_schedule_steps) != len(lr_schedule_gammas):
                raise ValueError("lr_schedule_gammas must be provided and have the same length as lr_schedule_steps.")
            self.lr_schedule_steps = sorted(lr_schedule_steps)
            self.lr_schedule_rates = [initial_lr]
            current_lr = initial_lr
            for gamma in lr_schedule_gammas:
                current_lr *= gamma
                self.lr_schedule_rates.append(current_lr)

        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        self.total_steps = 0

    def _apply_pointwise(self, x: torch.Tensor, net: nn.Module) -> torch.Tensor:
        b, n, d = x.shape
        out = net(x.reshape(b * n, d))
        return out.reshape(b, n, -1)

    @staticmethod
    def _apply_linear_T(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        if isinstance(layer, nn.Identity):
            return x
        if not isinstance(layer, nn.Linear):
            raise TypeError("Expected nn.Linear or nn.Identity for transpose application.")
        w = layer.weight  # (out_dim, in_dim)
        return torch.matmul(x, w)

    def encode_u(self, us: torch.Tensor, xs: torch.Tensor | None = None) -> torch.Tensor:
        if self.encoder_type == "transformer":
            if xs is None:
                raise ValueError("encode_u requires xs when encoder_type='transformer'.")
            tokens = torch.cat([xs, us], dim=-1)
            h = self.encoder_in(tokens)
            h = self.encoder_transformer(h)
            return self.encoder_out(h)
        return self._apply_pointwise(us, self.encoder_u)

    def decode_u(self, z_u: torch.Tensor) -> torch.Tensor:
        return self._apply_pointwise(z_u, self.decoder_u)

    def encode_s(self, s: torch.Tensor) -> torch.Tensor:
        return self._apply_pointwise(s, self.encoder_s)

    def decode_s(self, z_s: torch.Tensor) -> torch.Tensor:
        return self._apply_pointwise(z_s, self.decoder_s)

    def forward_latent(
        self,
        xs: torch.Tensor,
        z_u: torch.Tensor,
        ys: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_u_op = self._apply_pointwise(z_u, self.lowrank_in)
        z_s_op = self.operator(xs, z_u_op, ys, sensor_mask=sensor_mask, sensor_weights=sensor_weights)
        return self._apply_pointwise(z_s_op, self.lowrank_out)

    def forward(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        ys: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_u = self.encode_u(us, xs=xs)
        z_s = self.forward_latent(xs, z_u, ys, sensor_mask=sensor_mask, sensor_weights=sensor_weights)
        return self.decode_s(z_s)

    def apply_adjoint(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        z_s: torch.Tensor,
        z_u: torch.Tensor | None = None,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Compute A^T z_s in latent space. Uses explicit adjoint for quadrature head when available."""
        if self.operator.branch_head_type == "quadrature":
            quad_head = getattr(self.operator, "quadrature_head", None)
            quad_ok = (
                quad_head is not None
                and quad_head.value_mode in {"linear_u", "gated_linear"}
                and isinstance(quad_head.rho_token, (nn.Identity, nn.Linear))
            )
            if quad_ok:
                z_s_op = self._apply_linear_T(z_s, self.lowrank_out)
                if z_u is None:
                    b, n, _ = xs.shape
                    z_u_op = torch.zeros((b, n, z_s_op.shape[-1]), device=xs.device, dtype=xs.dtype)
                else:
                    if z_u.shape[-1] == z_s_op.shape[-1]:
                        z_u_op = z_u
                    else:
                        z_u_op = self._apply_pointwise(z_u, self.lowrank_in)
                z_u_op = self.operator.apply_operator_adjoint(xs, z_u_op, ys, z_s_op)
                return self._apply_linear_T(z_u_op, self.lowrank_in)

        b, n, _ = xs.shape
        z_u = torch.zeros((b, n, self.latent_dim), device=xs.device, dtype=xs.dtype, requires_grad=True)
        z_s_pred = self.forward_latent(xs, z_u, ys)
        grad = torch.autograd.grad(
            z_s_pred,
            z_u,
            grad_outputs=z_s,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )[0]
        return grad

    def apply_full_adjoint(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        ys: torch.Tensor,
        v: torch.Tensor,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """
        Compute the exact adjoint of the full Koopman model:
        J_Eu^T * A^T * J_Ds^T * v.
        """
        us_req = us.detach().requires_grad_(True)
        z_u = self.encode_u(us_req, xs=xs)
        z_s = self.forward_latent(xs, z_u, ys)

        z_s_req = z_s.detach().requires_grad_(True)
        s_out = self.decode_s(z_s_req)
        scalar = (s_out * v).sum()
        grad_z_s = torch.autograd.grad(
            scalar,
            z_s_req,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )[0]

        grad_z_u = self.apply_adjoint(xs, ys, grad_z_s, z_u=z_u, create_graph=create_graph)
        if not create_graph:
            grad_z_u = grad_z_u.detach()

        scalar_u = (z_u * grad_z_u).sum()
        grad_u = torch.autograd.grad(
            scalar_u,
            us_req,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,
        )[0]
        if not create_graph:
            grad_u = grad_u.detach()
        return grad_u

    @staticmethod
    def _cg_solve(apply_mat, b, max_iter=25, tol=1e-5):
        if not torch.isfinite(b).all():
            return torch.zeros_like(b)
        x = torch.zeros_like(b)
        r = b - apply_mat(x)
        if not torch.isfinite(r).all():
            return torch.zeros_like(b)
        p = r.clone()
        rsold = (r * r).sum(dim=(1, 2))
        rs0 = rsold.clone()
        for _ in range(max_iter):
            Ap = apply_mat(p)
            if not torch.isfinite(Ap).all():
                return torch.zeros_like(b)
            denom = (p * Ap).sum(dim=(1, 2)).clamp_min(1e-12)
            alpha = rsold / denom
            x = x + alpha.view(-1, 1, 1) * p
            r = r - alpha.view(-1, 1, 1) * Ap
            if not torch.isfinite(x).all() or not torch.isfinite(r).all():
                return torch.zeros_like(b)
            rsnew = (r * r).sum(dim=(1, 2))
            rel = torch.sqrt(rsnew / (rs0 + 1e-12))
            if torch.max(rel) < tol:
                break
            beta = rsnew / (rsold + 1e-12)
            p = r + beta.view(-1, 1, 1) * p
            rsold = rsnew
        return x

    def solve_latent_ls(
        self,
        xs: torch.Tensor,
        ys: torch.Tensor,
        z_s: torch.Tensor,
        lambda_reg: float = 1e-4,
        cg_iters: int = 25,
        cg_tol: float = 1e-5,
    ) -> torch.Tensor:
        def apply_A(z_u):
            return self.forward_latent(xs, z_u, ys)

        def apply_AT(z_s_in):
            with torch.enable_grad():
                return self.apply_adjoint(xs, ys, z_s_in, create_graph=False)

        b = apply_AT(z_s)

        def apply_normal(z_u):
            return apply_AT(apply_A(z_u)) + lambda_reg * z_u

        return self._cg_solve(apply_normal, b, max_iter=cg_iters, tol=cg_tol)

    def _get_current_lr(self):
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

    def _update_lr(self):
        new_lr = self._get_current_lr()
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def train_model(
        self,
        dataset,
        epochs: int,
        progress_bar: bool = True,
        callback=None,
        lambda_inverse: float = 1.0,
        lambda_adjoint: float = 0.0,
        lambda_latent: float = 1.0,
        lambda_range: float = 0.0,
        lambda_recon_u: float = 0.0,
        lambda_recon_s: float = 0.0,
        adjoint_mode: str = "full",
    ):
        device = next(self.parameters()).device
        if callback is not None:
            callback.on_training_start(locals())

        bar = trange(epochs) if progress_bar else range(epochs)
        for _ in bar:
            current_lr = self._update_lr()
            xs, us, ys, s_true, sensor_mask = dataset.sample(device=device)
            us = us.detach().requires_grad_(lambda_adjoint > 0.0)

            z_u = self.encode_u(us, xs=xs)
            z_s_pred = self.forward_latent(xs, z_u, ys, sensor_mask=sensor_mask)
            s_pred = self.decode_s(z_s_pred)

            forward_loss = torch.nn.MSELoss()(s_pred, s_true)

            z_s_true = self.encode_s(s_true)
            latent_loss = torch.nn.MSELoss()(z_s_pred, z_s_true)
            # Range alignment: push encoder_s outputs into Range(A) without moving A in this loss.
            range_loss = torch.nn.MSELoss()(z_s_true, z_s_pred.detach())
            inverse_loss = torch.tensor(0.0, device=device)
            if lambda_inverse > 0.0:
                z_u_hat = self.apply_adjoint(xs, ys, z_s_true, create_graph=True)
                if torch.isfinite(z_u_hat).all():
                    u_hat = self.decode_u(z_u_hat)
                    inverse_loss = torch.nn.MSELoss()(u_hat, us)

            adjoint_loss = torch.tensor(0.0, device=device)
            if lambda_adjoint > 0.0:
                if adjoint_mode != "full":
                    raise ValueError("adjoint_mode must be 'full' when lambda_adjoint > 0.")
                v = (s_pred - s_true).detach()
                scalar = (s_pred * v).sum()
                grad_u_true = torch.autograd.grad(scalar, us, create_graph=False, retain_graph=True)[0].detach()
                z_v = self.encode_s(v)
                z_u_adj = self.apply_adjoint(xs, ys, z_v, create_graph=True)
                u_adj = self.decode_u(z_u_adj)
                adjoint_loss = torch.nn.MSELoss()(u_adj, grad_u_true)

            recon_u_loss = torch.tensor(0.0, device=device)
            recon_s_loss = torch.tensor(0.0, device=device)
            if lambda_recon_u > 0.0:
                u_recon = self.decode_u(z_u)
                recon_u_loss = torch.nn.MSELoss()(u_recon, us)
            if lambda_recon_s > 0.0:
                s_recon = self.decode_s(z_s_true)
                recon_s_loss = torch.nn.MSELoss()(s_recon, s_true)

            loss = forward_loss + lambda_inverse * inverse_loss + lambda_latent * latent_loss + lambda_range * range_loss
            if lambda_adjoint > 0.0:
                loss = loss + lambda_adjoint * adjoint_loss
            if lambda_recon_u > 0.0:
                loss = loss + lambda_recon_u * recon_u_loss
            if lambda_recon_s > 0.0:
                loss = loss + lambda_recon_s * recon_s_loss

            if not torch.isfinite(loss):
                self.opt.zero_grad(set_to_none=True)
                self.total_steps += 1
                if progress_bar:
                    bar.set_description(
                        f"Step {self.total_steps} | Loss: nan | Rel L2: nan | Grad Norm: nan | LR: {current_lr:.2e}"
                    )
                continue

            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            self.total_steps += 1

            with torch.no_grad():
                pred_flat = s_pred.squeeze(-1)
                target_flat = s_true.squeeze(-1)
                rel_l2_error = calculate_l2_relative_error(pred_flat, target_flat)

            if progress_bar:
                grad_norm = float(norm)
                bar.set_description(
                    f"Step {self.total_steps} | Loss: {loss.item():.4e} | Rel L2: {rel_l2_error.item():.4f} | "
                    f"Grad Norm: {grad_norm:.2f} | LR: {current_lr:.2e}"
                )

            if callback is not None:
                callback.on_step(locals())

        if callback is not None:
            callback.on_training_end(locals())
