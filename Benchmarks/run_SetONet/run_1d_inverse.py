import torch
import numpy as np
import sys
import os
from datetime import datetime
import argparse

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.koopman_setonet import KoopmanSetONet
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils import save_experiment_configuration
from Data.synthetic_1d_data import Synthetic1DDataGenerator
from Data.data_utils import generate_batch


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Koopman SetONet for synthetic 1D and evaluate forward/inverse with A/A^T in latent space."
    )

    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True, choices=['integral', 'derivative'],
                        help='Benchmark task: integral (f\' -> f) or derivative (f -> f\')')

    # Koopman / lifted operator architecture
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent dimension for Koopman lift')
    parser.add_argument('--encoder_hidden', type=int, default=64, help='Hidden size for encoders')
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers for encoders')
    parser.add_argument('--decoder_hidden', type=int, default=64, help='Hidden size for decoders')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders')
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet operator')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for networks')
    parser.add_argument('--son_quad_dk', type=int, default=64, help='Quadrature key/query dim (default: 64)')
    parser.add_argument('--son_quad_key_hidden', type=int, default=None, help='Quadrature key MLP hidden width (default: son_trunk_hidden)')
    parser.add_argument('--son_quad_key_layers', type=int, default=3, help='Quadrature key MLP depth (>=2)')
    parser.add_argument('--son_quad_phi_activation', type=str, default="softplus", choices=["tanh", "softsign", "softplus"], help='Quadrature Phi activation')
    parser.add_argument('--son_quad_value_mode', type=str, default="linear_u", choices=["linear_u", "gated_linear"], help='Quadrature value net mode')
    parser.add_argument('--son_quad_normalize', type=str, default="total", choices=["none", "total", "token"], help='Quadrature normalization')
    parser.add_argument('--son_quad_learn_temperature', action='store_true', help='Learn temperature for quadrature test functions')

    # Training parameters
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=5000, help='Number of epochs for SetONet')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=32, help='Dimension for sinusoidal positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency/scale for sinusoidal encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    # Inverse training loss removed; keep eval-only inverse options below.
    parser.add_argument('--lambda_latent', type=float, default=1.0, help='Weight for latent consistency loss')
    parser.add_argument('--lambda_range', type=float, default=0.0, help='Weight for range alignment loss')
    parser.add_argument('--lambda_recon_u', type=float, default=0.0, help='Weight for u autoencoder reconstruction loss')
    parser.add_argument('--lambda_recon_s', type=float, default=0.0, help='Weight for s autoencoder reconstruction loss')
    parser.add_argument('--adjoint_mode', type=str, default="full", choices=["full"],
                        help='Adjoint mode (full autograd) for adjoint matching')

    # Data generation
    parser.add_argument('--variable_sensors', action='store_true', help='Use different random sensor locations for each sample')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training/eval')

    # Evaluation
    parser.add_argument('--n_test_samples_eval', type=int, default=1000, help='Number of test samples for evaluation')
    parser.add_argument('--inverse_eval_mode', type=str, default="none", choices=["ls", "adjoint", "none"],
                        help='Inverse evaluation mode: ls, adjoint, or none')
    parser.add_argument('--inverse_lambda', type=float, default=1e-4, help='Tikhonov regularization for inverse (A^T A + λI)')
    parser.add_argument('--inverse_cg_iters', type=int, default=25, help='CG iterations for inverse solve')
    parser.add_argument('--inverse_cg_tol', type=float, default=1e-5, help='CG relative residual tolerance')

    # Model loading and misc
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use.')

    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None, help='Custom log directory (overrides default timestamped dir)')

    return parser.parse_args()


def setup_logging(benchmark_type, custom_log_dir=None):
    """Setup logging directory based on benchmark type or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join("logs")
        model_folder_name = f"KoopmanSetONet_{benchmark_type}_inverse"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir


def setup_parameters(args):
    """Setup problem parameters."""
    return {
        'input_range': [-1, 1],
        'scale': 0.1,
        'sensor_size': 100,
        'batch_size_train': args.batch_size,
        'n_trunk_points_train': 200,
        'n_test_samples_eval': args.n_test_samples_eval,
        'sensor_seed': args.seed,
        'variable_sensors': args.variable_sensors,
        'eval_sensor_dropoff': 0.0,
        'replace_with_nearest': False,
    }


def create_sensor_points(params, device):
    if params.get('variable_sensors', False):
        print("Using variable random sensor locations (same within batch, different between batches)")
        return None
    torch.manual_seed(params['sensor_seed'])
    sensor_x = torch.rand(params['sensor_size'], device=device) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
    sensor_x = sensor_x.sort()[0].view(-1, 1)
    print(f"Using {params['sensor_size']} fixed random sensor locations (sorted, seed={params['sensor_seed']})")
    return sensor_x


def get_activation_function(activation_name):
    activation_map = {
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh,
        'gelu': torch.nn.GELU,
        'swish': torch.nn.SiLU
    }
    return activation_map.get(activation_name.lower(), torch.nn.ReLU)


def create_model(args, device):
    activation_fn = get_activation_function(args.activation_fn)
    return KoopmanSetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        latent_dim=args.latent_dim,
        encoder_hidden=args.encoder_hidden,
        encoder_layers=args.encoder_layers,
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


def load_pretrained_model(model, args, device):
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print(f"Loaded pre-trained model from: {args.load_model_path}")
            return True
        print(f"Warning: Model path not found: {args.load_model_path}")
        args.load_model_path = None
    return False


def evaluate_forward_backward(model, generator, sensor_x, params, args):
    model.eval()
    n_batches = max(1, args.n_test_samples_eval // args.batch_size)
    total_forward = 0.0
    total_backward = 0.0
    total_count = 0

    for batch_idx in range(n_batches):
        torch.manual_seed(42 + batch_idx)
        if params['variable_sensors']:
            batch_data = generate_batch(
                batch_size=args.batch_size,
                n_trunk_points=params['n_trunk_points_train'],
                sensor_x=None,
                scale=params['scale'],
                input_range=params['input_range'],
                device=generator.device,
                constant_zero=True,
                variable_sensors=True,
                sensor_size=params['sensor_size']
            )
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval, sensor_x_batch = batch_data
            sensor_x_used = sensor_x_batch
        else:
            batch_data = generate_batch(
                batch_size=args.batch_size,
                n_trunk_points=params['n_trunk_points_train'],
                sensor_x=sensor_x,
                scale=params['scale'],
                input_range=params['input_range'],
                device=generator.device,
                constant_zero=True
            )
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval = batch_data
            sensor_x_used = sensor_x

            if args.benchmark == 'derivative':
                xs = sensor_x_used.unsqueeze(0).expand(args.batch_size, -1, -1)
                us = f_at_sensors.unsqueeze(-1)
                ys = x_eval.unsqueeze(0).expand(args.batch_size, -1, -1)
                target = f_prime_at_trunk.T.unsqueeze(-1)
            else:
                xs = sensor_x_used.unsqueeze(0).expand(args.batch_size, -1, -1)
                us = f_prime_at_sensors.unsqueeze(-1)
                ys = x_eval.unsqueeze(0).expand(args.batch_size, -1, -1)
                target = f_at_trunk.T.unsqueeze(-1)

        with torch.no_grad():
            s_pred = model(xs, us, ys)
            forward_rel = calculate_l2_relative_error(s_pred.squeeze(-1), target.squeeze(-1)).item()
            z_s = model.encode_s(target)

        backward_rel = None
        if args.inverse_eval_mode != "none":
            if args.inverse_eval_mode == "ls":
                z_u_hat = model.solve_latent_ls(
                    xs,
                    ys,
                    z_s,
                    lambda_reg=args.inverse_lambda,
                    cg_iters=args.inverse_cg_iters,
                    cg_tol=args.inverse_cg_tol,
                )
            elif args.inverse_eval_mode == "adjoint":
                with torch.enable_grad():
                    z_u_hat = model.apply_adjoint(xs, ys, z_s, create_graph=False).detach()
            else:
                raise ValueError("inverse_eval_mode must be 'ls', 'adjoint', or 'none'.")

            with torch.no_grad():
                u_hat = model.decode_u(z_u_hat)
                backward_rel = calculate_l2_relative_error(u_hat.squeeze(-1), us.squeeze(-1)).item()

        total_forward += forward_rel * args.batch_size
        if backward_rel is not None:
            total_backward += backward_rel * args.batch_size
        total_count += args.batch_size

    forward_rel = total_forward / max(1, total_count)
    if args.inverse_eval_mode == "none":
        return forward_rel, float("nan")
    backward_rel = total_backward / max(1, total_count)
    return forward_rel, backward_rel


def main():
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")

    print(f"Running benchmark: {args.benchmark}")
    log_dir = setup_logging(args.benchmark, args.log_dir)
    params = setup_parameters(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sensor_x = create_sensor_points(params, device)

    model = create_model(args, device)
    model_was_loaded = load_pretrained_model(model, args, device)

    generator = Synthetic1DDataGenerator(params, device, sensor_x, args.benchmark)

    if not model_was_loaded:
        print(f"\nStarting training for {args.son_epochs} epochs...")
        model.train_model(
            dataset=generator,
            epochs=args.son_epochs,
            progress_bar=True,
            callback=None,
            lambda_inverse=0.0,
            lambda_latent=args.lambda_latent,
            lambda_range=args.lambda_range,
            lambda_recon_u=args.lambda_recon_u,
            lambda_recon_s=args.lambda_recon_s,
            adjoint_mode=args.adjoint_mode,
        )
    else:
        print("\nModel loaded. Skipping training.")

    print("\nEvaluating forward and inverse performance...")
    forward_rel, backward_rel = evaluate_forward_backward(model, generator, sensor_x, params, args)
    print(f"Forward Relative L2: {forward_rel:.6f}")
    if args.inverse_eval_mode != "none":
        label = "LS" if args.inverse_eval_mode == "ls" else "A^T"
        print(f"Backward Relative L2 ({label}): {backward_rel:.6f}")

    test_results = {
        "relative_l2_error": forward_rel,
        "forward_relative_l2_error": forward_rel,
        "backward_relative_l2_error": backward_rel if args.inverse_eval_mode != "none" else None,
        "n_test_samples": args.n_test_samples_eval
    }

    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, f"koopman_setonet_{args.benchmark}_inverse_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    dummy_dataset = {'train': [], 'test': []}
    args.data_path = f"synthetic_1d_{args.benchmark}"
    args.son_phi_hidden = 256
    args.son_rho_hidden = 256
    args.son_phi_output_size = 32
    args.son_aggregation = "mean"
    save_experiment_configuration(
        args, model, dummy_dataset, dataset_wrapper=generator, device=device,
        log_dir=log_dir, dataset_type=args.benchmark, test_results=test_results
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
