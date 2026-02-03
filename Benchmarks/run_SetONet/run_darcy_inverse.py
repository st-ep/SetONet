import torch
import numpy as np
import sys
import os
import time
from datetime import datetime
import argparse

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.koopman_setonet import KoopmanSetONet
from Models.SetONet import SetONet
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Data.darcy_1d_data.darcy_1d_dataset import (
    load_darcy_dataset, DarcyDataGenerator, create_sensor_points,
    create_query_points, setup_parameters
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Koopman SetONet for Darcy 1D and evaluate linear operator diagnostics."
    )

    # Data parameters
    parser.add_argument('--data_path', type=str, default="Data/darcy_1d_data/darcy_1d_dataset_501",
                        help='Path to Darcy 1D dataset')
    parser.add_argument('--sensor_size', type=int, default=10, help='Number of sensor locations (max 501 for Darcy 1D grid)')

    # Koopman / lifted operator architecture
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for Koopman lift')
    parser.add_argument('--encoder_hidden', type=int, default=64, help='Hidden size for encoders')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoders')
    parser.add_argument('--encoder_type', type=str, default="transformer", choices=["mlp", "transformer"],
                        help='Encoder type for u: mlp or transformer')
    parser.add_argument('--encoder_transformer_dim', type=int, default=None,
                        help='Transformer model dim for encoder (default: encoder_hidden)')
    parser.add_argument('--encoder_transformer_layers', type=int, default=2,
                        help='Number of Transformer encoder layers for u')
    parser.add_argument('--encoder_transformer_heads', type=int, default=4,
                        help='Number of attention heads for Transformer encoder')
    parser.add_argument('--encoder_transformer_dropout', type=float, default=0.0,
                        help='Dropout for Transformer encoder')
    parser.add_argument('--encoder_transformer_ff', type=int, default=None,
                        help='Feedforward dimension for Transformer encoder (default: 4 * d_model)')
    parser.add_argument('--decoder_hidden', type=int, default=64, help='Hidden size for decoders')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders')
    parser.add_argument('--son_p_dim', type=int, default=64, help='Latent dimension p for SetONet operator')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for networks')
    parser.add_argument('--son_quad_dk', type=int, default=64, help='Quadrature key/query dim (default: 64)')
    parser.add_argument('--son_quad_key_hidden', type=int, default=None, help='Quadrature key MLP hidden width (default: son_trunk_hidden)')
    parser.add_argument('--son_quad_key_layers', type=int, default=3, help='Quadrature key MLP depth (>=2)')
    parser.add_argument('--son_quad_phi_activation', type=str, default="softplus", choices=["tanh", "softsign", "softplus"], help='Quadrature Phi activation')
    parser.add_argument('--son_quad_value_mode', type=str, default="gated_linear", choices=["linear_u", "gated_linear"], help='Quadrature value net mode')
    parser.add_argument('--son_quad_normalize', type=str, default="total", choices=["none", "total", "token"], help='Quadrature normalization')
    parser.add_argument('--son_quad_learn_temperature', action='store_true', help='Learn temperature for quadrature test functions')
    parser.add_argument('--lowrank_r', type=int, default=None,
                        help='Low-rank operator dimension (<= latent_dim). If unset, uses full latent_dim.')

    # Training parameters
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=35000, help='Number of epochs for SetONet')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency for sinusoidal positional encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    parser.add_argument('--lambda_latent', type=float, default=0.0, help='Weight for latent consistency loss')
    parser.add_argument('--lambda_range', type=float, default=0.0, help='Weight for range alignment loss')
    parser.add_argument('--lambda_recon_u', type=float, default=0.0, help='Weight for u autoencoder reconstruction loss')
    parser.add_argument('--lambda_recon_s', type=float, default=0.0, help='Weight for s autoencoder reconstruction loss')

    # Sensor variability
    parser.add_argument('--variable_sensors', dest='variable_sensors', action='store_true', default=True,
                        help='Use different random sensor locations each batch (default: True)')
    parser.add_argument('--fixed_sensors', dest='variable_sensors', action='store_false',
                        help='Use fixed evenly spaced sensors')

    # Sensor dropout and evaluation (kept for compatibility; overridden to 0 for fixed sensors)
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation (0.0-1.0)')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors')
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during training (0.0-1.0)')
    parser.add_argument('--n_test_samples_eval', type=int, default=1000, help='Number of test samples for evaluation')
    parser.add_argument('--n_query_points', type=int, default=300, help='Number of query points for evaluation')

    # Model loading and misc
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:1', help='Torch device to use.')

    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='TensorBoard evaluation frequency (steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples for TensorBoard')

    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None, help='Custom log directory (overrides default timestamped dir)')
    parser.add_argument('--benchmark_linearity', action='store_true',
                        help='Evaluate operator linearity in latent: ||A(z1+z2)-A(z1)-A(z2)||')
    parser.add_argument('--benchmark_resolution', action='store_true',
                        help='Evaluate resolution invariance across sensor counts')
    parser.add_argument('--resolution_sizes', type=int, nargs='+', default=None,
                        help='Sensor counts to test (e.g., 50 100 200 300)')
    parser.add_argument('--resolution_mode', type=str, default="even", choices=["even", "random"],
                        help='Resolution sampling: even (linspace) or random subsets')
    parser.add_argument('--resolution_trials', type=int, default=5,
                        help='Trials for random resolution sampling')
    parser.add_argument('--benchmark_additivity', action='store_true',
                        help='Evaluate split-additivity: G(u) vs G(u1)+G(u2)')
    parser.add_argument('--benchmark_lowrank', action='store_true',
                        help='Sweep low-rank values and plot scaling curve')
    parser.add_argument('--lowrank_r_values', type=int, nargs='+', default=None,
                        help='List of low-rank values to sweep (e.g., 4 8 16 32 64)')
    parser.add_argument('--lowrank_epochs', type=int, default=None,
                        help='Epochs per rank when sweeping (default: son_epochs)')
    parser.add_argument('--benchmark_latent', action='store_true',
                        help='Sweep latent_dim values and plot scaling curve')
    parser.add_argument('--latent_dim_values', type=int, nargs='+', default=None,
                        help='List of latent_dim values to sweep (e.g., 8 16 32 64 128)')
    parser.add_argument('--latent_dim_epochs', type=int, default=None,
                        help='Epochs per latent_dim when sweeping (default: son_epochs)')
    parser.add_argument('--baseline_attention', action='store_true',
                        help='Train/eval baseline SetONet with attention pooling')
    parser.add_argument('--baseline_epochs', type=int, default=None,
                        help='Epochs for baseline attention SetONet (default: son_epochs)')
    parser.add_argument('--baseline_lr', type=float, default=None,
                        help='Learning rate for baseline attention SetONet (default: son_lr)')
    parser.add_argument('--baseline_p_dim', type=int, default=32, help='Baseline SetONet latent dim p')
    parser.add_argument('--baseline_phi_hidden', type=int, default=256, help='Baseline phi hidden size')
    parser.add_argument('--baseline_rho_hidden', type=int, default=256, help='Baseline rho hidden size')
    parser.add_argument('--baseline_trunk_hidden', type=int, default=256, help='Baseline trunk hidden size')
    parser.add_argument('--baseline_n_trunk_layers', type=int, default=4, help='Baseline trunk layers')
    parser.add_argument('--baseline_phi_output_size', type=int, default=32, help='Baseline phi output size')

    return parser.parse_args()


def setup_logging(custom_log_dir=None):
    """Setup logging directory or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join("logs")
        model_folder_name = "KoopmanSetONet_darcy_1d_linear"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir


def get_activation_function(activation_name):
    """Get activation function by name."""
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'swish': nn.SiLU  # SiLU is equivalent to Swish
    }
    return activation_map.get(activation_name.lower(), nn.ReLU)


def create_setonet_model(args, device):
    """Create Koopman SetONet model for Darcy 1D problem."""
    print(f"\n--- Initializing SetONet Model for {args.benchmark} ---")
    print(f"Using activation function: {args.activation_fn}")

    activation_fn = get_activation_function(args.activation_fn)

    model = KoopmanSetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        latent_dim=args.latent_dim,
        lowrank_r=args.lowrank_r,
        encoder_hidden=args.encoder_hidden,
        encoder_layers=args.encoder_layers,
        encoder_type=args.encoder_type,
        encoder_transformer_dim=args.encoder_transformer_dim,
        encoder_transformer_layers=args.encoder_transformer_layers,
        encoder_transformer_heads=args.encoder_transformer_heads,
        encoder_transformer_dropout=args.encoder_transformer_dropout,
        encoder_transformer_ff=args.encoder_transformer_ff,
        decoder_hidden=args.decoder_hidden,
        decoder_layers=args.decoder_layers,
        activation_fn=activation_fn,
        initial_lr=args.son_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
        p=args.son_p_dim,
        trunk_hidden_size=args.son_trunk_hidden,
        n_trunk_layers=args.son_n_trunk_layers,
        pos_encoding_type=args.pos_encoding_type,
        pos_encoding_dim=args.pos_encoding_dim,
        pos_encoding_max_freq=args.pos_encoding_max_freq,
        quad_dk=args.son_quad_dk,
        quad_key_hidden=args.son_quad_key_hidden,
        quad_key_layers=args.son_quad_key_layers,
        quad_phi_activation=args.son_quad_phi_activation,
        quad_value_mode=args.son_quad_value_mode,
        quad_normalize=args.son_quad_normalize,
        quad_learn_temperature=args.son_quad_learn_temperature,
    ).to(device)

    return model


def create_attention_baseline(args, device):
    """Create baseline SetONet with attention pooling."""
    activation_fn = get_activation_function(args.activation_fn)
    lr = args.son_lr if args.baseline_lr is None else args.baseline_lr
    model = SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=args.baseline_p_dim,
        phi_hidden_size=args.baseline_phi_hidden,
        rho_hidden_size=args.baseline_rho_hidden,
        trunk_hidden_size=args.baseline_trunk_hidden,
        n_trunk_layers=args.baseline_n_trunk_layers,
        activation_fn=activation_fn,
        use_deeponet_bias=True,
        phi_output_size=args.baseline_phi_output_size,
        initial_lr=lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
        pos_encoding_type=args.pos_encoding_type,
        pos_encoding_dim=args.pos_encoding_dim,
        pos_encoding_max_freq=args.pos_encoding_max_freq,
        aggregation_type="attention",
        use_positional_encoding=(args.pos_encoding_type != 'skip'),
        attention_n_tokens=1,
        branch_head_type="standard",
    ).to(device)
    return model


def configure_linear_quadrature(model, args):
    """Ensure the latent operator is linear-in-u (required for A/A^T)."""
    if args.son_quad_value_mode not in {"linear_u", "gated_linear"}:
        raise ValueError("Quadrature operator requires --son_quad_value_mode linear_u or gated_linear.")


def load_pretrained_model(setonet_model, args, device):
    """Load a pre-trained model if path is provided."""
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            setonet_model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print(f"Loaded pre-trained SetONet model from: {args.load_model_path}")
            return True
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")
            args.load_model_path = None

    return False


def evaluate_forward_only(
    model,
    sensor_x,
    query_x,
    test_data,
    sensor_indices,
    query_indices,
    device,
    n_test,
    batch_size=64,
):
    """Evaluate forward-only performance."""
    sensor_idx = sensor_indices.to(device)
    query_idx = query_indices.to(device)

    total_forward = 0.0
    total_count = 0

    model.eval()
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        u_list = []
        s_list = []
        for i in range(start, end):
            sample = test_data[i]
            u = torch.tensor(sample['u'], device=device)[sensor_idx]
            s = torch.tensor(sample['s'], device=device)[query_idx]
            u_list.append(u)
            s_list.append(s)

        u_batch = torch.stack(u_list).unsqueeze(-1)
        s_batch = torch.stack(s_list).unsqueeze(-1)
        xs_batch = sensor_x.unsqueeze(0).expand(u_batch.shape[0], -1, -1)
        ys_batch = query_x.unsqueeze(0).expand(u_batch.shape[0], -1, -1)

        with torch.no_grad():
            s_pred = model(xs_batch, u_batch, ys_batch)
            forward_rel = calculate_l2_relative_error(
                s_pred.squeeze(-1), s_batch.squeeze(-1)
            ).item()

        batch_count = end - start
        total_forward += forward_rel * batch_count
        total_count += batch_count

    forward_rel = total_forward / max(1, total_count)
    return torch.tensor(forward_rel)


def evaluate_operator_linearity(model, xs, us, ys):
    """Check linearity of the latent operator A (in z_u)."""
    with torch.no_grad():
        z_u = model.encode_u(us)
        z1 = z_u
        z2 = torch.roll(z_u, shifts=1, dims=0)
        z1z2 = z1 + z2
        s1 = model.forward_latent(xs, z1, ys)
        s2 = model.forward_latent(xs, z2, ys)
        s12 = model.forward_latent(xs, z1z2, ys)
        num = torch.norm((s12 - s1 - s2).reshape(s12.shape[0], -1), dim=1)
        denom = torch.norm(s1.reshape(s1.shape[0], -1), dim=1) + torch.norm(s2.reshape(s2.shape[0], -1), dim=1) + 1e-12
        return (num / denom).mean().item()


def evaluate_split_additivity(
    model,
    xs,
    us,
    ys,
    device,
):
    """Check G(u) ≈ G(u1)+G(u2) for a random sensor split."""
    bsz, n_sensors, _ = us.shape
    perm = torch.randperm(n_sensors, device=device)
    mid = n_sensors // 2
    idx1 = perm[:mid]
    idx2 = perm[mid:]

    u1 = us.clone()
    u2 = us.clone()
    u1[:, idx2, :] = 0.0
    u2[:, idx1, :] = 0.0

    with torch.no_grad():
        g = model(xs, us, ys)
        g1 = model(xs, u1, ys)
        g2 = model(xs, u2, ys)
        num = torch.norm((g - g1 - g2).reshape(bsz, -1), dim=1)
        denom = torch.norm(g.reshape(bsz, -1), dim=1) + 1e-12
        return (num / denom).mean().item()


def evaluate_resolution_invariance(
    model,
    dataset,
    grid_points,
    query_indices,
    query_x,
    device,
    n_test,
    batch_size,
    sensor_sizes,
    resolution_mode="even",
    resolution_trials=1,
):
    """Evaluate forward error as sensor resolution changes."""
    grid_points = grid_points.to(device)
    test_data = dataset['test']
    results = {}
    max_idx = len(grid_points) - 1
    for sensor_size in sensor_sizes:
        trial_vals = []
        trials = resolution_trials if resolution_mode == "random" else 1
        for t in range(trials):
            if resolution_mode == "random":
                perm = torch.randperm(max_idx + 1, device=device)
                sensor_indices = perm[:sensor_size].sort()[0].to(dtype=torch.long)
            else:
                sensor_indices = torch.linspace(0, max_idx, sensor_size, dtype=torch.long, device=device)
            sensor_x = grid_points[sensor_indices].view(-1, 1)

            total_forward = 0.0
            total_count = 0
            for start in range(0, n_test, batch_size):
                end = min(start + batch_size, n_test)
                u_list = []
                s_list = []
                for i in range(start, end):
                    sample = test_data[i]
                    u = torch.tensor(sample['u'], device=device)[sensor_indices]
                    s = torch.tensor(sample['s'], device=device)[query_indices]
                    u_list.append(u)
                    s_list.append(s)

                u_batch = torch.stack(u_list).unsqueeze(-1)
                s_batch = torch.stack(s_list).unsqueeze(-1)
                xs_batch = sensor_x.unsqueeze(0).expand(u_batch.shape[0], -1, -1)
                ys_batch = query_x.unsqueeze(0).expand(u_batch.shape[0], -1, -1)

                with torch.no_grad():
                    s_pred = model(xs_batch, u_batch, ys_batch)
                    forward_rel = calculate_l2_relative_error(
                        s_pred.squeeze(-1), s_batch.squeeze(-1)
                    ).item()

                batch_count = end - start
                total_forward += forward_rel * batch_count
                total_count += batch_count

            trial_vals.append(total_forward / max(1, total_count))

        results[sensor_size] = float(np.mean(trial_vals))

    return results


def run_lowrank_scaling(
    args,
    dataset,
    data_generator,
    grid_points,
    sensor_x,
    query_x,
    sensor_indices,
    query_indices,
    device,
    log_dir,
):
    ranks = args.lowrank_r_values or [4, 8, 16, 32, 64]
    ranks = sorted({int(r) for r in ranks if int(r) > 0})
    ranks = [r for r in ranks if r <= args.latent_dim]
    if len(ranks) == 0:
        print("No valid low-rank values provided; skipping low-rank benchmark.")
        return

    test_data = dataset['test']
    n_test = min(args.n_test_samples_eval, len(test_data))
    epochs = args.son_epochs if args.lowrank_epochs is None else args.lowrank_epochs
    results = []

    # Populate fields expected by config_utils for compatibility
    args.son_phi_hidden = 0
    args.son_rho_hidden = 0
    args.son_phi_output_size = 0
    args.son_aggregation = "mean"

    for r in ranks:
        print(f"\n=== Low-rank sweep: r={r} ===")
        args.lowrank_r = r
        run_dir = os.path.join(log_dir, f"rank_{r}")
        os.makedirs(run_dir, exist_ok=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        model = create_setonet_model(args, device)
        configure_linear_quadrature(model, args)

        callback = None
        if args.enable_tensorboard:
            tb_log_dir = os.path.join(run_dir, "tensorboard")
            callback = TensorBoardCallback(
                log_dir=tb_log_dir,
                dataset=dataset,
                dataset_wrapper=data_generator,
                device=device,
                eval_frequency=args.tb_eval_frequency,
                n_test_samples=args.tb_test_samples,
                eval_sensor_dropoff=args.eval_sensor_dropoff,
                replace_with_nearest=args.replace_with_nearest,
            )

        print(f"Training for {epochs} epochs...")
        model.train_model(
            dataset=data_generator,
            epochs=epochs,
            progress_bar=True,
            callback=callback,
            lambda_inverse=0.0,
            lambda_latent=args.lambda_latent,
            lambda_range=args.lambda_range,
            lambda_recon_u=args.lambda_recon_u,
            lambda_recon_s=args.lambda_recon_s,
        )

        forward_rel = evaluate_forward_only(
            model,
            sensor_x,
            query_x,
            test_data,
            sensor_indices,
            query_indices,
            device,
            n_test,
            args.batch_size,
        )
        results.append((r, float(forward_rel.item())))
        print(f"Rank {r} Forward Relative L2: {forward_rel.item():.6f}")

        model_save_path = os.path.join(run_dir, "darcy1d_koopman_setonet_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

        test_results = {
            "relative_l2_error": forward_rel.item(),
            "forward_relative_l2_error": forward_rel.item(),
            "n_test_samples": n_test,
        }
        save_experiment_configuration(
            args,
            model,
            dataset,
            dataset_wrapper=data_generator,
            device=device,
            log_dir=run_dir,
            dataset_type="darcy_1d",
            test_results=test_results,
        )

    csv_path = os.path.join(log_dir, "lowrank_scaling.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("rank,forward_rel_l2\n")
        for r, err in results:
            f.write(f"{r},{err}\n")
    print(f"Saved scaling CSV to: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        r_vals = [r for r, _ in results]
        errs = [e for _, e in results]
        plt.figure(figsize=(6, 4))
        plt.plot(r_vals, errs, marker="o")
        plt.xlabel("Low-rank r")
        plt.ylabel("Forward Relative L2")
        plt.title("Low-rank scaling (Darcy 1D)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(log_dir, "lowrank_scaling_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved scaling plot to: {plot_path}")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


def run_latent_scaling(
    args,
    dataset,
    data_generator,
    grid_points,
    sensor_x,
    query_x,
    sensor_indices,
    query_indices,
    device,
    log_dir,
):
    dims = args.latent_dim_values or [8, 16, 32, 64, 128]
    dims = sorted({int(d) for d in dims if int(d) > 0})
    if len(dims) == 0:
        print("No valid latent_dim values provided; skipping latent sweep.")
        return

    test_data = dataset['test']
    n_test = min(args.n_test_samples_eval, len(test_data))
    epochs = args.son_epochs if args.latent_dim_epochs is None else args.latent_dim_epochs
    results = []

    # Populate fields expected by config_utils for compatibility
    args.son_phi_hidden = 0
    args.son_rho_hidden = 0
    args.son_phi_output_size = 0
    args.son_aggregation = "mean"

    base_latent = args.latent_dim
    base_lowrank = args.lowrank_r

    for d in dims:
        print(f"\n=== Latent sweep: latent_dim={d} ===")
        args.latent_dim = d
        if base_lowrank is not None and base_lowrank > d:
            args.lowrank_r = None
        run_dir = os.path.join(log_dir, f"latent_{d}")
        os.makedirs(run_dir, exist_ok=True)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

        model = create_setonet_model(args, device)
        configure_linear_quadrature(model, args)

        callback = None
        if args.enable_tensorboard:
            tb_log_dir = os.path.join(run_dir, "tensorboard")
            callback = TensorBoardCallback(
                log_dir=tb_log_dir,
                dataset=dataset,
                dataset_wrapper=data_generator,
                device=device,
                eval_frequency=args.tb_eval_frequency,
                n_test_samples=args.tb_test_samples,
                eval_sensor_dropoff=args.eval_sensor_dropoff,
                replace_with_nearest=args.replace_with_nearest,
            )

        print(f"Training for {epochs} epochs...")
        model.train_model(
            dataset=data_generator,
            epochs=epochs,
            progress_bar=True,
            callback=callback,
            lambda_inverse=0.0,
            lambda_latent=args.lambda_latent,
            lambda_range=args.lambda_range,
            lambda_recon_u=args.lambda_recon_u,
            lambda_recon_s=args.lambda_recon_s,
        )

        forward_rel = evaluate_forward_only(
            model,
            sensor_x,
            query_x,
            test_data,
            sensor_indices,
            query_indices,
            device,
            n_test,
            args.batch_size,
        )
        results.append((d, float(forward_rel.item())))
        print(f"Latent {d} Forward Relative L2: {forward_rel.item():.6f}")

        model_save_path = os.path.join(run_dir, "darcy1d_koopman_setonet_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

        test_results = {
            "relative_l2_error": forward_rel.item(),
            "forward_relative_l2_error": forward_rel.item(),
            "n_test_samples": n_test,
        }
        save_experiment_configuration(
            args,
            model,
            dataset,
            dataset_wrapper=data_generator,
            device=device,
            log_dir=run_dir,
            dataset_type="darcy_1d",
            test_results=test_results,
        )

    args.latent_dim = base_latent
    args.lowrank_r = base_lowrank

    csv_path = os.path.join(log_dir, "latent_scaling.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("latent_dim,forward_rel_l2\n")
        for d, err in results:
            f.write(f"{d},{err}\n")
    print(f"Saved latent scaling CSV to: {csv_path}")

    try:
        import matplotlib.pyplot as plt

        d_vals = [d for d, _ in results]
        errs = [e for _, e in results]
        plt.figure(figsize=(6, 4))
        plt.plot(d_vals, errs, marker="o")
        plt.xlabel("latent_dim")
        plt.ylabel("Forward Relative L2")
        plt.title("Latent scaling (Darcy 1D)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(log_dir, "latent_scaling_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved latent scaling plot to: {plot_path}")
    except Exception as exc:
        print(f"Plotting skipped: {exc}")


def main():
    """Main training function."""
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Add benchmark argument for compatibility
    args.benchmark = 'darcy_1d'

    # No inverse/invertible options: keep standard Koopman SetONet settings.

    # Enforce fixed sensors (no dropoff)
    if not args.variable_sensors and (args.train_sensor_dropoff != 0.0 or args.eval_sensor_dropoff != 0.0):
        print("Fixed-sensor mode: overriding sensor dropoff to 0.0")
        args.train_sensor_dropoff = 0.0
        args.eval_sensor_dropoff = 0.0

    # Load dataset
    dataset = load_darcy_dataset(args.data_path)
    grid_points = torch.tensor(dataset['train'][0]['X'], dtype=torch.float32)

    log_dir = setup_logging(args.log_dir)
    params = setup_parameters(args)

    # Set random seed and ensure reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU setups

    # For better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Create sensor and query points
    sensor_x, sensor_indices = create_sensor_points(params, device, grid_points)
    query_x, query_indices = create_query_points(params, device, grid_points, args.n_query_points)

    print(f"Sensor points: {len(sensor_x)}, Query points: {len(query_x)}")

    # Create data generator
    data_generator = DarcyDataGenerator(dataset, sensor_indices, query_indices, device, params, grid_points)

    if args.benchmark_lowrank:
        run_lowrank_scaling(
            args,
            dataset,
            data_generator,
            grid_points,
            sensor_x,
            query_x,
            sensor_indices,
            query_indices,
            device,
            log_dir,
        )
        return

    if args.benchmark_latent:
        run_latent_scaling(
            args,
            dataset,
            data_generator,
            grid_points,
            sensor_x,
            query_x,
            sensor_indices,
            query_indices,
            device,
            log_dir,
        )
        return

    # Create model
    setonet_model = create_setonet_model(args, device)
    configure_linear_quadrature(setonet_model, args)

    # Load pre-trained model if specified
    model_was_loaded = load_pretrained_model(setonet_model, args, device)

    # Setup TensorBoard callback if enabled
    callback = None
    if args.enable_tensorboard:
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        callback = TensorBoardCallback(
            log_dir=tb_log_dir,
            dataset=dataset,
            dataset_wrapper=data_generator,
            device=device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=0.0,
            replace_with_nearest=False
        )
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # Training
    if not model_was_loaded:
        print(f"\nStarting training for {args.son_epochs} epochs...")
        setonet_model.train_model(
            dataset=data_generator,
            epochs=args.son_epochs,
            progress_bar=True,
            callback=callback,
            lambda_inverse=0.0,
            lambda_latent=args.lambda_latent,
            lambda_range=args.lambda_range,
            lambda_recon_u=args.lambda_recon_u,
            lambda_recon_s=args.lambda_recon_s,
        )
    else:
        print(f"\nSetONet Darcy 1D model loaded. Skipping training.")

    test_data = dataset['test']
    n_test = min(args.n_test_samples_eval, len(test_data))

    if args.baseline_attention:
        baseline_epochs = args.son_epochs if args.baseline_epochs is None else args.baseline_epochs
        print("\n--- Baseline: SetONet (attention) ---")
        baseline_model = create_attention_baseline(args, device)
        print(f"Training baseline for {baseline_epochs} epochs...")
        baseline_model.train_model(
            dataset=data_generator,
            epochs=baseline_epochs,
            progress_bar=True,
            callback=None,
        )

        print("Evaluating baseline forward error...")
        baseline_forward = evaluate_forward_only(
            baseline_model,
            sensor_x,
            query_x,
            dataset['test'],
            sensor_indices,
            query_indices,
            device,
            n_test,
            args.batch_size,
        )
        print(f"Baseline Forward Relative L2: {baseline_forward.item():.6f}")
        if args.benchmark_resolution:
            sizes = args.resolution_sizes
            if sizes is None:
                sizes = [
                    max(16, args.sensor_size // 4),
                    max(32, args.sensor_size // 2),
                    args.sensor_size,
                    min(len(grid_points), args.sensor_size * 2),
                ]
            sizes = sorted({s for s in sizes if 2 <= s <= len(grid_points)})
            if len(sizes) > 0:
                res_results = evaluate_resolution_invariance(
                    baseline_model,
                    dataset,
                    grid_points,
                    query_indices,
                    query_x,
                    device,
                    n_test,
                    args.batch_size,
                    sizes,
                    resolution_mode=args.resolution_mode,
                    resolution_trials=args.resolution_trials,
                )
                print("Baseline resolution invariance (sensor_size -> Rel L2):")
                for s in sizes:
                    print(f"  {s}: {res_results[s]:.6f}")

    if args.benchmark_linearity:
        xs_batch, us_batch, ys_batch, _, _ = data_generator.sample(device=device)
        lin_err = evaluate_operator_linearity(setonet_model, xs_batch, us_batch, ys_batch)
        print(f"\nOperator linearity error (latent superposition): {lin_err:.6e}")

    # Evaluate forward performance
    print("\nEvaluating forward performance...")
    forward_rel = evaluate_forward_only(
        setonet_model,
        sensor_x,
        query_x,
        test_data,
        sensor_indices,
        query_indices,
        device,
        n_test,
        args.batch_size,
    )
    print(f"Forward Relative L2 (A u -> s): {forward_rel.item():.6f}")

    if args.benchmark_resolution:
        if args.resolution_sizes is None:
            sizes = [
                max(16, args.sensor_size // 4),
                max(32, args.sensor_size // 2),
                args.sensor_size,
                min(len(grid_points), args.sensor_size * 2),
            ]
        else:
            sizes = args.resolution_sizes
        sizes = sorted({s for s in sizes if 2 <= s <= len(grid_points)})
        if len(sizes) == 0:
            print("No valid resolution sizes provided; skipping resolution benchmark.")
        else:
            res_results = evaluate_resolution_invariance(
                setonet_model,
                dataset,
                grid_points,
                query_indices,
                query_x,
                device,
                n_test,
                args.batch_size,
                sizes,
                resolution_mode=args.resolution_mode,
                resolution_trials=args.resolution_trials,
            )
            print("\nResolution invariance (sensor_size -> Rel L2):")
            for s in sizes:
                print(f"  {s}: {res_results[s]:.6f}")

    if args.benchmark_additivity:
        xs_batch, us_batch, ys_batch, _, _ = data_generator.sample(device=device)
        add_err = evaluate_split_additivity(setonet_model, xs_batch, us_batch, ys_batch, device)
        print(f"\nSplit-additivity error: {add_err:.6e}")

    # Prepare test results for configuration saving
    test_results = {
        "relative_l2_error": forward_rel.item(),
        "forward_relative_l2_error": forward_rel.item(),
        "n_test_samples": n_test
    }

    # Populate fields expected by config_utils for compatibility
    args.son_phi_hidden = 0
    args.son_rho_hidden = 0
    args.son_phi_output_size = 0
    args.son_aggregation = "mean"

    # Save model
    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, "darcy1d_koopman_setonet_model.pth")
        torch.save(setonet_model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    # Save experiment configuration with test results
    save_experiment_configuration(
        args,
        setonet_model,
        dataset,
        dataset_wrapper=data_generator,
        device=device,
        log_dir=log_dir,
        dataset_type="darcy_1d",
        test_results=test_results
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
