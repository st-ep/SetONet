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
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Data.heat_data.heat_2d_dataset import load_heat_dataset


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Koopman SetONet for 2D heat problem.")

    # Data parameters
    default_data_path = os.path.join(
        project_root,
        "Data",
        "heat_data",
        "pcb_heat_adaptive_dataset8.0_n8192_N25_P30",
    )
    parser.add_argument('--data_path', type=str, default=default_data_path, help='Path to Heat 2D dataset')

    # Koopman / lifted operator architecture
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for Koopman lift')
    parser.add_argument('--encoder_hidden', type=int, default=128, help='Hidden size for encoders')
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
    parser.add_argument('--decoder_hidden', type=int, default=128, help='Hidden size for decoders')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of layers for decoders')
    parser.add_argument('--son_p_dim', type=int, default=64, help='Latent dimension p for SetONet operator')
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
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--son_epochs', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type')
    parser.add_argument('--pos_encoding_dim', type=int, default=128, help='Dimension for positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.01, help='Max frequency for sinusoidal positional encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[15000, 30000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    parser.add_argument('--lambda_latent', type=float, default=1.0, help='Weight for latent consistency loss')
    parser.add_argument('--lambda_range', type=float, default=0.0, help='Weight for range alignment loss')
    parser.add_argument('--lambda_recon_u', type=float, default=0.0, help='Weight for u autoencoder reconstruction loss')
    parser.add_argument('--lambda_recon_s', type=float, default=0.0, help='Weight for s autoencoder reconstruction loss')

    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained model')

    # Random seed and device
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:1', help='Torch device to use.')

    # Evaluation
    parser.add_argument('--n_test_samples', type=int, default=100, help='Number of test samples for evaluation')

    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='TensorBoard evaluation frequency (steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples for TensorBoard')

    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None, help='Custom log directory (overrides default timestamped dir)')

    return parser.parse_args()


def setup_logging(custom_log_dir=None):
    """Setup logging directory or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join(project_root, "logs")
        model_folder_name = "KoopmanSetONet_heat2d_linear"
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


def create_model(args, device):
    """Create Koopman SetONet model for 2D heat problem."""
    activation_fn = get_activation_function(args.activation_fn)

    model = KoopmanSetONet(
        input_size_src=2,  # 2D coordinates (x, y) of sources
        output_size_src=1,  # Scalar power values
        input_size_tgt=2,  # 2D coordinates (x, y) of grid points
        output_size_tgt=1,  # Scalar temperature values
        latent_dim=args.latent_dim,
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


def evaluate_model(model, dataset, heat_dataset, device, n_test_samples=100):
    """Evaluate the model on test data."""
    model.eval()
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))

    total_loss = 0.0
    total_rel_error = 0.0

    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]

            sources = torch.tensor(np.array(sample['sources']), device=device, dtype=torch.float32)
            source_coords = sources[:, :2].unsqueeze(0)  # (1, n_sources, 2)
            source_powers = sources[:, 2:3].unsqueeze(0)  # (1, n_sources, 1)

            if heat_dataset.is_adaptive:
                target_coords = torch.tensor(np.array(sample['grid_coords']), device=device, dtype=torch.float32).unsqueeze(0)
                target_temps = torch.tensor(np.array(sample['field_values']), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            else:
                target_coords = heat_dataset.grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
                temp_field = torch.tensor(np.array(sample['field'])[:, :, 0].flatten(), device=device, dtype=torch.float32)
                target_temps = temp_field.unsqueeze(0).unsqueeze(-1)  # (1, n_grid_points, 1)

            pred = model(source_coords, source_powers, target_coords)

            mse_loss = torch.nn.MSELoss()(pred, target_temps)
            total_loss += mse_loss.item()

            rel_error = calculate_l2_relative_error(pred, target_temps)
            total_rel_error += rel_error.item()

    avg_loss = total_loss / n_test
    avg_rel_error = total_rel_error / n_test

    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")

    model.train()
    return avg_loss, avg_rel_error


def main():
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_dir = setup_logging(args.log_dir)

    dataset, heat_dataset = load_heat_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device
    )
    if dataset is None or heat_dataset is None:
        return

    print("Creating Koopman SetONet model...")
    model = create_model(args, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    model_was_loaded = False
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            print(f"Loading pre-trained model from: {args.load_model_path}")
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            model_was_loaded = True
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")

    callback = None
    if args.enable_tensorboard:
        print("Setting up TensorBoard logging...")
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        callback = TensorBoardCallback(
            log_dir=tb_log_dir,
            dataset=dataset,
            dataset_wrapper=heat_dataset,
            device=device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=0.0,
            replace_with_nearest=False
        )
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    if not model_was_loaded:
        print(f"\nStarting training for {args.son_epochs} epochs...")
        model.train_model(
            dataset=heat_dataset,
            epochs=args.son_epochs,
            progress_bar=True,
            callback=callback,
            lambda_inverse=0.0,
            lambda_adjoint=0.0,
            lambda_latent=args.lambda_latent,
            lambda_range=args.lambda_range,
            lambda_recon_u=args.lambda_recon_u,
            lambda_recon_s=args.lambda_recon_s,
        )
    else:
        print("\nModel loaded. Skipping training.")

    print("\nEvaluating model...")
    avg_loss, avg_rel_error = evaluate_model(model, dataset, heat_dataset, device, n_test_samples=args.n_test_samples)

    test_results = {
        "relative_l2_error": avg_rel_error,
        "mse_loss": avg_loss,
        "n_test_samples": args.n_test_samples
    }

    # Populate fields expected by config_utils for compatibility
    args.son_phi_hidden = 0
    args.son_rho_hidden = 0
    args.son_phi_output_size = 0
    args.son_aggregation = "mean"

    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, "heat2d_koopman_setonet_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    save_experiment_configuration(
        args,
        model,
        dataset,
        heat_dataset,
        device,
        log_dir,
        dataset_type="heat_2d",
        test_results=test_results
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
