import torch
import torch.nn as nn
# from FunctionEncoder import BaseDataset, BaseCallback # Keep BaseDataset/Callback if used
from tqdm import trange
from torch.optim.lr_scheduler import _LRScheduler # Import base class for type hinting if needed
from .utils.helper_utils import calculate_l2_relative_error

# This implements a DeepOSet using PyTorch
class SetONet(torch.nn.Module):

    def __init__(self,
                 input_size_src,      # Dimensionality of sensor location x_i (e.g., 1 for 1D)
                 output_size_src,     # Dimensionality of sensor value u(x_i) (e.g., 1 for scalar)
                 input_size_tgt,      # Dimensionality of trunk input y (e.g., 1 for 1D)
                 output_size_tgt,     # Dimensionality of final output G(u)(y) (e.g., 1 for scalar)
                 p=32,                # Latent dimension for the branch/trunk cross product (Default)
                 phi_hidden_size=256, # Hidden layer size for the phi network (Default)
                 rho_hidden_size=256, # Hidden layer size for the rho network (Default)
                 trunk_hidden_size=256,# Hidden layer size for the trunk network (MLP) (Default)
                 n_trunk_layers=4,    # Number of layers in the trunk network (Default)
                 activation_fn=nn.ReLU, # Activation function
                 use_deeponet_bias=True, # Whether to use a bias term after the cross product
                 phi_output_size=128, # Output dimension of the phi network before aggregation (Default)
                 initial_lr=5e-4,     # Initial learning rate
                 lr_schedule_steps=None, # Steps for LR decay (optional)
                 lr_schedule_gammas=None, # Multiplicative factor for each step (optional)
                 use_positional_encoding=True, # Flag to enable/disable positional encoding
                 pos_encoding_dim=64, # Dimension for sinusoidal positional encoding
                 pos_encoding_type='skip', # Type: 'sinusoidal', or 'skip'
                 pos_encoding_max_freq=0.1, # Max frequency/scale for sinusoidal encoding
                 encoding_strategy='concatenate', # Strategy for combining positional and sensor features. Only 'concatenate' is supported.
                 aggregation_type: str = "attention",  # 'mean' or 'attention'
                 attention_n_tokens: int = 1,     # k – number of learnable query tokens
	                 branch_head_type: str = "standard",  # "standard" | "petrov_attention" | "galerkin_pou" | "quadrature" | "adaptive_quadrature"
                 pg_dk: int = None,  # key/query dim (default = phi_output_size)
                 pg_dv: int = None,  # value dim (default = phi_output_size)
                 pg_use_softmax: bool = True,  # only softmax supported for now
                 pg_use_logw: bool = True,  # include log(w_i) term when weights provided
                 pg_n_refine_iters: int = 0,  # number of query refinement iterations (0 = disabled)
                 galerkin_dk: int = None,  # Galerkin PoU key/query dim (default = phi_output_size)
                 galerkin_dv: int = None,  # Galerkin PoU value dim (default = phi_output_size)
                 galerkin_normalize: str = "total",  # Galerkin normalization: "none" | "total" | "token"
                 galerkin_learn_temperature: bool = False,  # Learn temperature parameter for Galerkin PoU softmax
	                 quad_dk: int = None,  # Quadrature head key/query dim (default = phi_output_size)
	                 quad_dv: int = None,  # Quadrature head value dim (default = phi_output_size)
	                 quad_normalize: str = "total",  # Quadrature normalization: "none" | "total" | "token"
	                 quad_learn_temperature: bool = False,  # Learn temperature parameter for quadrature test functions
	                 adapt_quad_rank: int = 4,  # Low-rank adaptation rank R for adaptive_quadrature
	                 adapt_quad_hidden: int | None = 64,  # Hidden dim for the adapter MLP (None -> infer)
	                 adapt_quad_scale: float = 0.1,  # Tanh-bounded adapter scale
	                 adapt_quad_use_value_context: bool = True,  # Only value-based context supported
	                 ):
        super().__init__()

        # set hyperparameters
        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt
        # Note: use_positional_encoding is True if type is 'mlp' or 'sinusoidal'
        self.use_positional_encoding = use_positional_encoding and pos_encoding_type != 'skip' # True if type is 'sinusoidal' and use_positional_encoding arg is True
        self.pos_encoding_dim = pos_encoding_dim if self.use_positional_encoding else 0
        self.pos_encoding_type = pos_encoding_type
        self.pos_encoding_max_freq = pos_encoding_max_freq # Store max frequency/scale
        self.encoding_strategy = encoding_strategy

        self.p = p
        self.phi_hidden_size = phi_hidden_size
        self.phi_output_size = phi_output_size
        self.rho_hidden_size = rho_hidden_size
        self.trunk_hidden_size = trunk_hidden_size
        self.n_trunk_layers = n_trunk_layers

        # ---------------------------------------------------------------------
        # Branch head choice ("standard" | "petrov_attention" | "galerkin_pou" | "quadrature" | "adaptive_quadrature")
        # ---------------------------------------------------------------------
        self.branch_head_type = branch_head_type.lower()
        if self.branch_head_type not in [
            "standard",
            "petrov_attention",
            "galerkin_pou",
            "quadrature",
            "adaptive_quadrature",
        ]:
            raise ValueError(
                "branch_head_type must be one of 'standard', 'petrov_attention', 'galerkin_pou', 'quadrature', or 'adaptive_quadrature'"
            )
        self.pg_use_softmax = pg_use_softmax
        self.pg_use_logw = pg_use_logw
        if self.branch_head_type == "petrov_attention" and not self.pg_use_softmax:
            raise NotImplementedError("Only softmax-normalized PG attention is supported (pg_use_softmax=True).")

        # ---------------------------------------------------------------------
        # Aggregation choice ('mean' | 'sum' | 'attention')
        # ---------------------------------------------------------------------
        self.aggregation = aggregation_type.lower()
        if self.aggregation not in ["mean", "sum", "attention"]:
            raise ValueError("aggregation_type must be one of 'mean', 'sum', or 'attention'")
        self.attention_n_tokens = attention_n_tokens
        self.attention_n_heads = 4 # Default or make configurable if needed

        # Attention pool is only used by "standard" branch head
        self.pool = None
        if self.aggregation == "attention" and self.branch_head_type == "standard":
            from .utils.attention_pool import AttentionPool
            self.pool = AttentionPool(phi_output_size,
                                      n_heads=4,
                                      n_tokens=self.attention_n_tokens)

        # Validate encoding strategy
        if self.encoding_strategy != 'concatenate':
            raise ValueError("encoding_strategy must be 'concatenate'. 'film' is no longer supported.")
        # Validate pos_encoding_type
        if self.pos_encoding_type not in ['sinusoidal', 'skip']:
             raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}. Choose 'sinusoidal' or 'skip'.")
        if self.use_positional_encoding:
            if self.pos_encoding_dim % (2 * self.input_size_src) != 0:
                raise ValueError(
                    f"For sinusoidal encoding, pos_encoding_dim ({self.pos_encoding_dim}) must be divisible by "
                    f"2 * input_size_src ({2 * self.input_size_src})."
                )

        # Store LR schedule parameters
        self.initial_lr = initial_lr
        self.lr_schedule_steps = None
        self.lr_schedule_rates = None
        self.lr_schedule_gammas = None # Store gammas as well
        if lr_schedule_steps is not None:
            if lr_schedule_gammas is None or len(lr_schedule_steps) != len(lr_schedule_gammas):
                raise ValueError("lr_schedule_gammas must be provided and have the same length as lr_schedule_steps if scheduling is used.")
            self.lr_schedule_steps = sorted(lr_schedule_steps) # Ensure steps are sorted
            self.lr_schedule_gammas = lr_schedule_gammas # Store the provided gammas
            # Calculate the actual learning rates at each step based on gammas
            self.lr_schedule_rates = [initial_lr]
            current_lr = initial_lr
            for gamma in lr_schedule_gammas:
                current_lr *= gamma
                self.lr_schedule_rates.append(current_lr)

        # --- Branch Network (Deep Sets) ---
        # self.pos_encoder_mlp = None # REMOVED: For concatenate strategy with MLP encoding

        # --- Phi Network (only needed for "standard" branch head) ---
        # petrov_attention has its own internal networks
        self.phi = None
        if self.branch_head_type == "standard":
            if self.encoding_strategy == 'concatenate':
                # Determine phi input dimension
                if self.use_positional_encoding: # True only for 'sinusoidal' type and if use_positional_encoding arg is True
                    phi_input_dim = self.pos_encoding_dim + output_size_src
                    # No pos_encoder_mlp is initialized or used for sinusoidal.
                else: # This case handles pos_encoding_type == 'skip' or if use_positional_encoding (arg) was explicitly False
                    # Original input: raw position + sensor value
                    phi_input_dim = input_size_src + output_size_src

                # Phi network: processes concatenated (encoded_location, value) or (location, value) pairs
                self.phi = nn.Sequential(
                    nn.Linear(phi_input_dim, phi_hidden_size),
                    activation_fn(),
                    nn.Linear(phi_hidden_size, phi_hidden_size),
                    activation_fn(),
                    nn.Linear(phi_hidden_size, phi_output_size)
                )

        # --- Rho Network (only needed for "standard" branch head) ---
        # petrov_attention has its own output projection
        self.rho = None
        if self.branch_head_type == "standard":
            rho_input_dim = (self.phi_output_size *
                             (self.attention_n_tokens if self.aggregation == "attention" else 1))

            self.rho = nn.Sequential(
                nn.Linear(rho_input_dim, rho_hidden_size),
                activation_fn(),
                nn.Linear(rho_hidden_size, output_size_tgt * p)
            )
        # --- End Branch Network ---

        # --- Petrov–Galerkin (PG) attention head (Approach 2) ---
        self.pg_head = None
        if self.branch_head_type == "petrov_attention":
            from .utils.petrov_galerkin_head import PetrovGalerkinHead

            dx_enc = self.pos_encoding_dim if self.use_positional_encoding else self.input_size_src
            dk = pg_dk if pg_dk is not None else self.phi_output_size
            dv = pg_dv if pg_dv is not None else self.phi_output_size
            self.pg_head = PetrovGalerkinHead(
                p=self.p,
                dx_enc=dx_enc,
                du=self.output_size_src,
                dout=self.output_size_tgt,
                dk=dk,
                dv=dv,
                hidden=self.rho_hidden_size,
                activation_fn=activation_fn,
                n_refine_iters=pg_n_refine_iters,
            )
        # --- End PG head ---

        # --- Galerkin partition-of-unity (PoU) head ---
        self.galerkin_head = None
        if self.branch_head_type == "galerkin_pou":
            from .utils.galerkin_head import GalerkinPoUHead

            dx_enc = self.pos_encoding_dim if self.use_positional_encoding else self.input_size_src
            dk = galerkin_dk if galerkin_dk is not None else self.phi_output_size
            dv = galerkin_dv if galerkin_dv is not None else self.phi_output_size
            self.galerkin_head = GalerkinPoUHead(
                p=self.p,
                dx_enc=dx_enc,
                du=self.output_size_src,
                dout=self.output_size_tgt,
                dk=dk,
                dv=dv,
                hidden=self.rho_hidden_size,
                activation_fn=activation_fn,
                normalize=galerkin_normalize,
                learn_temperature=galerkin_learn_temperature,
            )
        # --- End Galerkin PoU head ---

        # --- Quadrature head (non-normalized test functions, additive quadrature sum) ---
        self.quadrature_head = None
        if self.branch_head_type == "quadrature":
            from .utils.quadrature_head import QuadratureHead

            dx_enc = self.pos_encoding_dim if self.use_positional_encoding else self.input_size_src
            dk = quad_dk if quad_dk is not None else self.phi_output_size
            dv = quad_dv if quad_dv is not None else self.phi_output_size
            self.quadrature_head = QuadratureHead(
                p=self.p,
                dx_enc=dx_enc,
                du=self.output_size_src,
                dout=self.output_size_tgt,
                dk=dk,
                dv=dv,
                hidden=self.rho_hidden_size,
                activation_fn=activation_fn,
                normalize=quad_normalize,
                learn_temperature=quad_learn_temperature,
            )
        # --- End Quadrature head ---

        # --- Input-adaptive quadrature head (context-conditioned test functions) ---
        self.adaptive_quadrature_head = None
        if self.branch_head_type == "adaptive_quadrature":
            from .utils.adaptive_quadrature_head import AdaptiveQuadratureHead

            dx_enc = self.pos_encoding_dim if self.use_positional_encoding else self.input_size_src
            dk = quad_dk if quad_dk is not None else self.phi_output_size
            dv = quad_dv if quad_dv is not None else self.phi_output_size
            self.adaptive_quadrature_head = AdaptiveQuadratureHead(
                p=self.p,
                dx_enc=dx_enc,
                du=self.output_size_src,
                dout=self.output_size_tgt,
                dk=dk,
                dv=dv,
                hidden=self.rho_hidden_size,
                activation_fn=activation_fn,
                adapt_rank=adapt_quad_rank,
                adapt_hidden=adapt_quad_hidden,
                adapt_scale=adapt_quad_scale,
                use_value_context=adapt_quad_use_value_context,
                normalize=quad_normalize,
                learn_temperature=quad_learn_temperature,
            )
        # --- End adaptive quadrature head ---

        # --- Trunk Network (MLP) ---
        # Maps y to t_1, ..., t_p (potentially multi-dimensional output_size_tgt)
        trunk_layers = []
        trunk_layers.append(nn.Linear(input_size_tgt, trunk_hidden_size))
        trunk_layers.append(activation_fn())
        for _ in range(n_trunk_layers - 2):
            trunk_layers.append(nn.Linear(trunk_hidden_size, trunk_hidden_size))
            trunk_layers.append(activation_fn())
        trunk_layers.append(nn.Linear(trunk_hidden_size, output_size_tgt * p))

        # trunk_layers.append(torch.nn.Sigmoid())
        self.trunk = torch.nn.Sequential(*trunk_layers)
        # --- End Trunk Network ---

        # an optional bias, see equation 2 in the DeepONet paper.
        self.bias = torch.nn.Parameter(torch.randn(output_size_tgt) * 0.1) if use_deeponet_bias else None

        # create optimizer with the initial learning rate
        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr)

        # Initialize step counter for LR scheduling
        self.total_steps = 0

        # holdovers from function encoder code, these do nothing
        self.method = "deepOSet"
        self.average_function = None

    def _sinusoidal_encoding(self, coords):
        """Applies fixed sinusoidal encoding to coordinates."""
        # coords shape: (batch_size, n_sensors, input_size_src)
        # Output shape: (batch_size, n_sensors, pos_encoding_dim)

        # Make encoding depend on coords' last dimension (works for both x and y)
        coord_dim = coords.shape[-1]
        # Ensure pos_encoding_dim is divisible by 2*coord_dim
        dims_per_coord = self.pos_encoding_dim // coord_dim
        half_dim = dims_per_coord // 2

        # Frequency bands
        # Shape: (half_dim,)
        div_term = torch.exp(torch.arange(half_dim, device=coords.device) * -(torch.log(torch.tensor(self.pos_encoding_max_freq, device=coords.device)) / half_dim))

        # Expand div_term for broadcasting: (1, 1, 1, half_dim)
        div_term = div_term.reshape(1, 1, 1, half_dim)

        # Expand coords for broadcasting: (batch, n_sensors, input_size_src, 1)
        coords_expanded = coords.unsqueeze(-1)

        # Calculate arguments for sin/cos: (batch, n_sensors, input_size_src, half_dim)
        angles = coords_expanded * div_term

        # Calculate sin and cos embeddings: (batch, n_sensors, input_size_src, half_dim)
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)

        # Interleave sin and cos and flatten the last two dimensions
        # Shape: (batch, n_sensors, input_size_src, dims_per_coord)
        encoding = torch.cat([sin_embed, cos_embed], dim=-1).reshape(
            coords.shape[0], coords.shape[1], coord_dim, dims_per_coord
        )

        # Reshape to final desired dimension: (batch, n_sensors, pos_encoding_dim)
        encoding = encoding.reshape(coords.shape[0], coords.shape[1], self.pos_encoding_dim)

        return encoding

    def _infer_sensor_weights(self, xs: torch.Tensor, sensor_mask: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor:
        """
        Infer simple quadrature weights from sensor coordinates.

        - For 1D sensors: trapezoidal/Voronoi-style cell sizes from sorted coordinates.
        - For higher-dimensional sensors: fall back to uniform weights (per valid sensor).

        Args:
            xs: (B, N, dx) raw sensor coordinates
            sensor_mask: (B, N) bool mask (optional)
            eps: numerical stability constant
        Returns:
            w: (B, N) nonnegative weights (zeros for masked-out sensors)
        """
        if xs.dim() != 3:
            raise ValueError(f"xs must be 3D (B, N, dx), got {xs.shape=}")

        batch_size, n_sensors, dx = xs.shape
        device, dtype = xs.device, xs.dtype

        mask = None
        if sensor_mask is not None:
            if sensor_mask.dim() == 3 and sensor_mask.shape[-1] == 1:
                sensor_mask = sensor_mask.squeeze(-1)
            if sensor_mask.shape != (batch_size, n_sensors):
                raise ValueError(f"{sensor_mask.shape=} must be (B, N) = {(batch_size, n_sensors)}")
            mask = sensor_mask.to(device=device).bool()

        # Higher-dimensional: uniform over valid sensors (weight scale handled by head normalization).
        if dx != 1:
            w = torch.ones((batch_size, n_sensors), device=device, dtype=dtype)
            if mask is not None:
                w = w * mask.to(dtype)
            return w

        # 1D: trapezoidal weights from sorted coordinates.
        x = xs[..., 0].detach()
        w = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)

        # Fast path: no mask (all sensors valid)
        if mask is None:
            if n_sensors == 1:
                return torch.ones((batch_size, n_sensors), device=device, dtype=dtype)

            x_sorted, sort_idx = torch.sort(x, dim=1)
            dx_sorted = (x_sorted[:, 1:] - x_sorted[:, :-1]).clamp_min(0.0)  # (B, N-1)

            w_sorted = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)
            w_sorted[:, 0] = dx_sorted[:, 0] / 2.0
            w_sorted[:, -1] = dx_sorted[:, -1] / 2.0
            if n_sensors > 2:
                w_sorted[:, 1:-1] = (dx_sorted[:, :-1] + dx_sorted[:, 1:]) / 2.0

            w.scatter_(1, sort_idx, w_sorted)
            return w

        # Masked path: per-sample valid set (variable number of sensors).
        x = x.detach()  # weights are geometry-derived; do not backprop through sorting
        for b in range(batch_size):
            idx_valid = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            n_valid = int(idx_valid.numel())
            if n_valid == 0:
                continue
            if n_valid == 1:
                w[b, idx_valid[0]] = 1.0
                continue

            x_valid = x[b, idx_valid]
            x_sorted, perm = torch.sort(x_valid)
            idx_sorted = idx_valid[perm]

            dx_sorted = (x_sorted[1:] - x_sorted[:-1]).clamp_min(0.0)  # (n_valid-1,)
            w_sorted = torch.zeros((n_valid,), device=device, dtype=dtype)
            w_sorted[0] = dx_sorted[0] / 2.0
            w_sorted[-1] = dx_sorted[-1] / 2.0
            if n_valid > 2:
                w_sorted[1:-1] = (dx_sorted[:-1] + dx_sorted[1:]) / 2.0

            w[b, idx_sorted] = w_sorted

        return w

    def forward_branch(self, xs, us, ys=None, sensor_mask=None, sensor_weights=None):
        """
        Forward pass for the Deep Sets Branch.
        Args:
            xs (torch.Tensor): Sensor locations, shape (batch_size, n_sensors, input_size_src)
            us (torch.Tensor): Sensor values, shape (batch_size, n_sensors, output_size_src)
            ys (torch.Tensor | None): Target locations (unused, kept for API compatibility)
            sensor_mask (torch.Tensor | None): (batch_size, n_sensors) bool mask, True = valid
            sensor_weights (torch.Tensor | None): (batch_size, n_sensors) nonnegative weights
        Returns:
            torch.Tensor: Branch output, shape (batch_size, p, output_size_tgt)
        """
        # --- Petrov-Galerkin attention head ---
        if self.branch_head_type == "petrov_attention":
            x_enc = self._sinusoidal_encoding(xs) if self.use_positional_encoding else xs
            weights = sensor_weights if (sensor_weights is not None and self.pg_use_logw) else None
            return self.pg_head(x_enc, us, sensor_mask=sensor_mask, sensor_weights=weights)

        # --- Galerkin partition-of-unity head ---
        if self.branch_head_type == "galerkin_pou":
            x_enc = self._sinusoidal_encoding(xs) if self.use_positional_encoding else xs
            # For Galerkin head, weights are used multiplicatively in the quadrature sum (not in logits)
            return self.galerkin_head(
                x_enc,
                us,
                sensor_mask=sensor_mask,
                sensor_weights=sensor_weights,
            )

        # --- Quadrature head (non-normalized test functions) ---
        if self.branch_head_type == "quadrature":
            x_enc = self._sinusoidal_encoding(xs) if self.use_positional_encoding else xs
            inferred_weights = (
                self._infer_sensor_weights(xs, sensor_mask) if sensor_weights is None else sensor_weights
            )
            return self.quadrature_head(
                x_enc,
                us,
                sensor_mask=sensor_mask,
                sensor_weights=inferred_weights,
            )

        # --- Input-adaptive quadrature head (context-conditioned test functions) ---
        if self.branch_head_type == "adaptive_quadrature":
            x_enc = self._sinusoidal_encoding(xs) if self.use_positional_encoding else xs
            inferred_weights = (
                self._infer_sensor_weights(xs, sensor_mask) if sensor_weights is None else sensor_weights
            )
            return self.adaptive_quadrature_head(
                x_enc,
                us,
                sensor_mask=sensor_mask,
                sensor_weights=inferred_weights,
            )

        batch_size = xs.shape[0]
        n_sensors = xs.shape[1]

        # Reshape inputs for element-wise processing
        # Shape: (batch * n_sensors, *)
        xs_reshaped = xs.reshape(batch_size * n_sensors, self.input_size_src)
        us_reshaped = us.reshape(batch_size * n_sensors, self.output_size_src)

        phi_input_reshaped = None

        # --- Apply Encoding Strategy ---
        if self.encoding_strategy == 'concatenate':
            if self.use_positional_encoding: # True only for 'sinusoidal' type and if use_positional_encoding arg is True
                # Calculate sinusoidal encoding
                # Shape: (batch, n_sensors, pos_encoding_dim)
                encoded_xs_full = self._sinusoidal_encoding(xs)
                # Reshape: (batch * n_sensors, pos_encoding_dim)
                encoded_xs = encoded_xs_full.reshape(batch_size * n_sensors, self.pos_encoding_dim)

                # Concatenate the encoded location and value
                # Shape: (batch * n_sensors, pos_encoding_dim + output_size_src)
                phi_input_reshaped = torch.cat((encoded_xs, us_reshaped), dim=1)
            else: # Handles 'skip' type or explicitly disabled positional encoding
                # Original: Concatenate raw location and value
                # Shape: (batch * n_sensors, input_size_src + output_size_src)
                phi_input_reshaped = torch.cat((xs_reshaped, us_reshaped), dim=1)

        # --- End Encoding Strategy ---

        # Apply phi network
        # Input shape: (batch_size * n_sensors, phi_input_dim)
        phi_output = self.phi(phi_input_reshaped)

        # Reshape back for aggregation
        # Shape: (batch_size, n_sensors, phi_output_size)
        phi_output_reshaped = phi_output.reshape(batch_size, n_sensors, self.phi_output_size)

        # ---- Aggregation over sensors ---------------------------------------
        if self.aggregation == "mean":
            aggregated = torch.mean(phi_output_reshaped, dim=1)            # (B, dφ)
        elif self.aggregation == "sum":
            aggregated = torch.sum(phi_output_reshaped, dim=1)             # (B, dφ)
        elif self.aggregation == "attention": # Ensure this covers all valid cases
            aggregated = self.pool(phi_output_reshaped)                    # (B, dφ)
        else:
            # This case should ideally not be reached if aggregation_type is validated in __init__
            raise ValueError(f"Unsupported aggregation type: {self.aggregation}")

        # Apply rho to the aggregated representation
        # Shape: (batch_size, output_size_tgt * p)
        rho_output = self.rho(aggregated)

        # Reshape rho output to match the desired structure for einsum
        # Shape: (batch_size, p, output_size_tgt)
        branch_out = rho_output.reshape(batch_size, self.p, self.output_size_tgt)

        return branch_out

    def forward_trunk(self, ys):
        """
        Forward pass for the Trunk Network.
        Args:
            ys (torch.Tensor): Trunk input locations, shape (batch_size, n_points, input_size_tgt)
        Returns:
            torch.Tensor: Trunk output, shape (batch_size, n_points, p, output_size_tgt)
        """
        batch_size = ys.shape[0]
        n_points = ys.shape[1]

        # Reshape for trunk's Linear layers if needed (MLP handles batch dim)
        # Input shape: (batch_size, n_points, input_size_tgt)
        # Output shape: (batch_size, n_points, output_size_tgt * p)
        trunk_out_flat = self.trunk(ys)

        # Reshape trunk output for einsum
        # Shape: (batch_size, n_points, p, output_size_tgt)
        trunk_out = trunk_out_flat.reshape(batch_size, n_points, self.p, self.output_size_tgt)
        return trunk_out

    def forward(self, xs, us, ys, sensor_mask=None, sensor_weights=None):
        """
        Full forward pass for DeepOSet.
        Args:
            xs (torch.Tensor): Sensor locations, shape (batch_size, n_sensors, input_size_src)
            us (torch.Tensor): Sensor values, shape (batch_size, n_sensors, output_size_src)
            ys (torch.Tensor): Trunk input locations, shape (batch_size, n_points, input_size_tgt)
            sensor_mask (torch.Tensor | None): (batch_size, n_sensors) bool mask, True = valid
            sensor_weights (torch.Tensor | None): (batch_size, n_sensors) nonnegative weights
        Returns:
            torch.Tensor: Predicted output G(u)(y), shape (batch_size, n_points, output_size_tgt)
        """
        # Get branch output: (batch, p, out_tgt)
        b = self.forward_branch(xs, us, ys=ys, sensor_mask=sensor_mask, sensor_weights=sensor_weights)

        # Get trunk output: (batch, n_points, p, out_tgt)
        t = self.forward_trunk(ys)

        # Combine using einsum (dot product over latent dimension p)
        # einsum: "bpz, bdpz -> bdz" (b=batch, d=n_points, p=latent_dim, z=out_tgt_dim)
        G_u_y = torch.einsum("bpz,bdpz->bdz", b, t)

        # optionally add bias
        if self.bias is not None:
            # Bias shape is (output_size_tgt), needs broadcasting to (batch, n_points, output_size_tgt)
            G_u_y = G_u_y + self.bias  # Broadcasting handles the addition

        return G_u_y

    def _get_current_lr(self):
        """ Gets the learning rate based on the current total_steps. """
        # If no schedule is defined, always return the initial LR
        if self.lr_schedule_steps is None or self.lr_schedule_rates is None:
            return self.initial_lr

        lr = self.initial_lr
        # Find the correct LR based on the number of steps completed
        milestone_idx = -1
        for i, step_milestone in enumerate(self.lr_schedule_steps):
            if self.total_steps >= step_milestone:
                milestone_idx = i
            else:
                break # Stop checking once we are below a milestone

        # If we passed any milestones, use the corresponding rate
        if milestone_idx != -1:
             # +1 because lr_schedule_rates[0] is initial LR
            lr = self.lr_schedule_rates[milestone_idx + 1]
        return lr

    def _update_lr(self):
        """ Updates the optimizer's learning rate based on total_steps. """
        new_lr = self._get_current_lr()
        # Update learning rate for all parameter groups in the optimizer
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr
        return new_lr # Return the new LR for logging/display

    # This is the main training loop, kept consistent with the function encoder code.
    def train_model(self,
                    dataset, # : BaseDataset,
                    epochs: int, # Note: This loop runs 'epochs' times, each is one step
                    progress_bar=True,
                    callback=None): # : BaseCallback = None):
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            # Pass initial state including the step counter if needed
            callback.on_training_start(locals())


        losses = []
        # Treat 'epochs' here as the number of steps for this training call
        bar = trange(epochs) if progress_bar else range(epochs)
        for step_in_epoch in bar: # Renamed 'epoch' to 'step_in_epoch' for clarity
            # Update Learning Rate based on total steps *before* optimizer step
            current_lr = self._update_lr()

            # sample input data - dataset needs to provide xs, us, ys, G_u_ys
            # Ensure dataset.sample() returns locations (xs) and values (us) separately
            xs, us, ys, G_u_ys, sensor_mask = dataset.sample(device=device)


            # approximate functions, compute error
            estimated_G_u_ys = self.forward(xs, us, ys, sensor_mask=sensor_mask)
            prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)

            # Calculate relative L2 error for progress bar
            with torch.no_grad():
                # Reshape to (batch_size, n_points) for helper function if needed
                pred_flat = estimated_G_u_ys.squeeze(-1) if estimated_G_u_ys.shape[-1] == 1 else estimated_G_u_ys.reshape(estimated_G_u_ys.shape[0], -1)
                target_flat = G_u_ys.squeeze(-1) if G_u_ys.shape[-1] == 1 else G_u_ys.reshape(G_u_ys.shape[0], -1)
                rel_l2_error = calculate_l2_relative_error(pred_flat, target_flat)

            # add loss components (can add regularization later if needed)
            loss = prediction_loss

            # backprop with gradient clipping
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()

            # Increment total steps *after* optimizer step
            self.total_steps += 1

            # update progress bar
            if progress_bar:
                grad_norm = float(norm)
                # Display current LR and relative L2 error in the progress bar
                bar.set_description(f"Step {self.total_steps} | Loss: {loss.item():.4e} | Rel L2: {rel_l2_error.item():.4f} | Grad Norm: {grad_norm:.2f} | LR: {current_lr:.2e}")

            # callbacks
            if callback is not None:
                # Pass current state including step counter and LR
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())
