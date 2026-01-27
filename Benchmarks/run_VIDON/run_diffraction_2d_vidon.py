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

# Import required modules
from Models.VIDON import VIDON
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils_vidon import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Data.diffraction_2d_data.diffraction_2d_dataset import load_diffraction_dataset
from Plotting.plot_diffraction_2d_utils import plot_diffraction_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VIDON for 2D diffraction problem.")

    # Data parameters
    parser.add_argument('--data_path', type=str, default="Data/diffraction_data/phase_screen_dataset",
                        help='Path to Diffraction 2D dataset')

    # Model architecture - VIDON specific
    parser.add_argument('--vidon_p_dim', type=int, default=128, help='Number of trunk basis functions (excluding τ0)')
    parser.add_argument('--vidon_n_heads', type=int, default=4, help='Number of attention heads (H)')
    parser.add_argument('--vidon_d_enc', type=int, default=40, help='Encoding dimension (d_enc)')
    parser.add_argument('--vidon_head_output_size', type=int, default=64, help='Output dimension of each head')

    # Encoder networks (Ψc, Ψv)
    parser.add_argument('--vidon_enc_hidden', type=int, default=40, help='Hidden size for encoder networks')
    parser.add_argument('--vidon_enc_n_layers', type=int, default=4, help='Number of layers in encoder networks')

    # Head MLPs (ωe, νe)
    parser.add_argument('--vidon_head_hidden', type=int, default=128, help='Hidden size for head MLPs')
    parser.add_argument('--vidon_head_n_layers', type=int, default=4, help='Number of layers in head MLPs')

    # Combiner Φ
    parser.add_argument('--vidon_combine_hidden', type=int, default=256, help='Hidden size for combiner network')
    parser.add_argument('--vidon_combine_n_layers', type=int, default=4, help='Number of layers in combiner network')

    # Trunk network τ
    parser.add_argument('--vidon_trunk_hidden', type=int, default=256, help='Hidden size for trunk network')
    parser.add_argument('--vidon_n_trunk_layers', type=int, default=4, help='Number of layers in trunk network')

    parser.add_argument('--activation_fn', type=str, default="relu",
                        choices=["relu", "tanh", "gelu", "swish"], help='Activation function for networks')

    # Training parameters
    parser.add_argument('--vidon_lr', type=float, default=5e-4, help='Learning rate for VIDON')
    parser.add_argument('--vidon_epochs', type=int, default=50000, help='Number of epochs for VIDON')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+',
                        default=[15000, 30000, 125000, 175000, 1250000, 1500000],
                        help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+',
                        default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5],
                        help="List of multiplicative factors for LR decay.")

    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained VIDON model')

    # Random seed and device
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:1', help='Torch device to use.')

    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True,
                        help='Enable TensorBoard logging of training metrics')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000,
                        help='How often to evaluate on test set for TensorBoard logging (in steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100,
                        help='Number of test samples to use for TensorBoard evaluation')

    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Custom log directory (overrides default timestamped dir)')

    return parser.parse_args()


def setup_logging(project_root, custom_log_dir=None):
    """Setup logging directory or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join(project_root, "logs")
        model_folder_name = "VIDON_diffraction2d"
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
    """Create VIDON model for 2D diffraction problem."""
    activation_fn = get_activation_function(args.activation_fn)

    model = VIDON(
        input_size_src=2,   # bump coordinates (x, y)
        output_size_src=2,  # bump features (alpha, ell)
        input_size_tgt=2,   # grid coordinates (x, y)
        output_size_tgt=2,  # complex field values (Re, Im)
        p=args.vidon_p_dim,
        n_heads=args.vidon_n_heads,
        d_enc=args.vidon_d_enc,
        head_output_size=args.vidon_head_output_size,
        enc_hidden_size=args.vidon_enc_hidden,
        enc_n_layers=args.vidon_enc_n_layers,
        head_hidden_size=args.vidon_head_hidden,
        head_n_layers=args.vidon_head_n_layers,
        combine_hidden_size=args.vidon_combine_hidden,
        combine_n_layers=args.vidon_combine_n_layers,
        trunk_hidden_size=args.vidon_trunk_hidden,
        n_trunk_layers=args.vidon_n_trunk_layers,
        activation_fn=activation_fn,
        initial_lr=args.vidon_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
    ).to(device)

    return model


def evaluate_model(model, dataset, diffraction_dataset, device, n_test_samples=100):
    """Evaluate the model on test data."""
    model.eval()
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))

    total_loss = 0.0
    total_rel_error = 0.0

    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]

            # Prepare test data
            bumps = torch.tensor(np.array(sample['bumps']), device=device, dtype=torch.float32)
            bump_coords = bumps[:, :2].unsqueeze(0)  # (1, n_bumps, 2)
            bump_feats = bumps[:, 2:4].unsqueeze(0)  # (1, n_bumps, 2)

            # Handle adaptive vs uniform mesh
            if diffraction_dataset.is_adaptive:
                target_coords = torch.tensor(np.array(sample['grid_coords']), device=device,
                                             dtype=torch.float32).unsqueeze(0)
                target_field = torch.tensor(np.array(sample['field_values']), device=device,
                                            dtype=torch.float32).unsqueeze(0)
            else:
                target_coords = diffraction_dataset.grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
                field = np.array(sample['field'])
                target_field = torch.tensor(field.reshape(-1, 2), device=device, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            pred = model(bump_coords, bump_feats, target_coords)

            # Calculate metrics
            mse_loss = torch.nn.MSELoss()(pred, target_field)
            total_loss += mse_loss.item()

            pred_flat = pred.reshape(pred.shape[0], -1)
            target_flat = target_field.reshape(target_field.shape[0], -1)
            rel_error = calculate_l2_relative_error(pred_flat, target_flat)
            total_rel_error += rel_error.item()

    avg_loss = total_loss / n_test
    avg_rel_error = total_rel_error / n_test

    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")

    model.train()
    return avg_loss, avg_rel_error


def main():
    """Main training function."""
    # Parse arguments and setup
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Validate arguments
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")

    # Set random seed and ensure reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU setups

    # For better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup logging
    log_dir = setup_logging(project_root, args.log_dir)

    # Load dataset using the new function
    dataset, diffraction_dataset = load_diffraction_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device
    )

    if dataset is None or diffraction_dataset is None:
        return

    # Create model
    print("Creating VIDON model...")
    model = create_model(args, device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    # Load pre-trained model if specified
    model_was_loaded = False
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            print(f"Loading pre-trained model from: {args.load_model_path}")
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            model_was_loaded = True
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")

    # Setup TensorBoard callback if enabled
    callback = None
    if args.enable_tensorboard:
        print("Setting up TensorBoard logging...")
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        callback = TensorBoardCallback(
            log_dir=tb_log_dir,
            dataset=dataset,
            dataset_wrapper=diffraction_dataset,
            device=device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=0.0,  # No sensor dropout for diffraction
            replace_with_nearest=False
        )
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {tb_log_dir}")

    # Train model
    if not model_was_loaded:
        print(f"\nStarting training for {args.vidon_epochs} epochs...")
        model.train_model(
            dataset=diffraction_dataset,
            epochs=args.vidon_epochs,
            progress_bar=True,
            callback=callback
        )
    else:
        print(f"\nVIDON diffraction 2D model loaded. Skipping training.")

    # Evaluate model
    print("\nEvaluating model...")
    avg_loss, avg_rel_error = evaluate_model(model, dataset, diffraction_dataset, device, n_test_samples=100)

    # Prepare test results for configuration saving
    test_results = {
        "relative_l2_error": avg_rel_error,
        "mse_loss": avg_loss,
        "n_test_samples": 100,
    }

    # Plot results
    print("Generating plots...")
    # Plot 3 test samples
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"diffraction_results_test_sample_{i}.png")
        plot_diffraction_results(
            model,
            dataset,
            diffraction_dataset,
            device,
            sample_idx=i,
            save_path=plot_save_path,
            dataset_split="test",
        )

    # Plot 3 train samples
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"diffraction_results_train_sample_{i}.png")
        plot_diffraction_results(
            model,
            dataset,
            diffraction_dataset,
            device,
            sample_idx=i,
            save_path=plot_save_path,
            dataset_split="train",
        )

    # Save experiment configuration with test results
    save_experiment_configuration(
        args,
        model,
        dataset,
        diffraction_dataset,
        device,
        log_dir,
        dataset_type="diffraction_2d",
        test_results=test_results,
    )

    # Save model
    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, "diffraction2d_vidon_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
