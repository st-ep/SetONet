import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse
import json

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.VIDON import VIDON
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils_vidon import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Data.darcy_1d_data.darcy_1d_dataset import (
    load_darcy_dataset, DarcyDataGenerator, create_sensor_points, 
    create_query_points, setup_parameters
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VIDON for Darcy 1D equation.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="Data/darcy_1d_data/darcy_1d_dataset_501", 
                       help='Path to Darcy 1D dataset')
    parser.add_argument('--sensor_size', type=int, default=300, help='Number of sensor locations (max 501 for Darcy 1D grid)')
    
    # Model architecture - VIDON specific
    parser.add_argument('--vidon_p_dim', type=int, default=32, help='Number of trunk basis functions (excluding τ0)')
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
    
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for networks')
    
    # Training parameters
    parser.add_argument('--vidon_lr', type=float, default=5e-4, help='Learning rate for VIDON')
    parser.add_argument('--vidon_epochs', type=int, default=125000, help='Number of epochs for VIDON')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Sensor dropout and evaluation
    # VIDON can handle variable input sizes like SetONet, so it uses removal/replacement (not interpolation)
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation (0.0-1.0). Sensors are removed or replaced.')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors instead of removing them')
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during training (0.0-1.0)')
    parser.add_argument('--n_test_samples_eval', type=int, default=1000, help='Number of test samples for evaluation')
    parser.add_argument('--n_query_points', type=int, default=300, help='Number of query points for evaluation')
    
    # Model loading and misc
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained VIDON model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use.')
    
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
        logs_base_in_project = os.path.join("logs")
        model_folder_name = "VIDON_darcy_1d"
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

def create_vidon_model(args, device):
    """Create VIDON model for Darcy 1D problem."""
    print(f"\n--- Initializing VIDON Model for Darcy 1D ---")
    print(f"Using activation function: {args.activation_fn}")
    
    activation_fn = get_activation_function(args.activation_fn)
    
    model = VIDON(
        input_size_src=1,  # 1D coordinates (x)
        output_size_src=1,  # Scalar force values
        input_size_tgt=1,  # 1D coordinates (x)
        output_size_tgt=1,  # Scalar displacement values
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
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    return model

def load_pretrained_model(model, args, device):
    """Load a pre-trained model if path is provided."""
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print(f"Loaded pre-trained VIDON model from: {args.load_model_path}")
            return True
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")
            args.load_model_path = None
    
    return False


def main():
    """Main training function."""
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Validate arguments
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")
    
    if not 0.0 <= args.eval_sensor_dropoff <= 1.0:
        raise ValueError("--eval_sensor_dropoff must be between 0.0 and 1.0")
    
    if args.eval_sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Will test robustness with sensor drop-off rate: {args.eval_sensor_dropoff:.1%} using {replacement_mode}")
        print("(Training will use full sensor data)")
    
    # Add benchmark argument for compatibility
    args.benchmark = 'darcy_1d'
    
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
    
    # Create model
    vidon_model = create_vidon_model(args, device)
    
    # Load pre-trained model if specified
    model_was_loaded = load_pretrained_model(vidon_model, args, device)
    
    # Create data generator
    data_generator = DarcyDataGenerator(dataset, sensor_indices, query_indices, device, params, grid_points)
    
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
            eval_sensor_dropoff=args.eval_sensor_dropoff,
            replace_with_nearest=args.replace_with_nearest
        )
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
    
    # Training
    if not model_was_loaded:
        print(f"\nStarting training for {args.vidon_epochs} epochs...")
        vidon_model.train_model(
            dataset=data_generator,
            epochs=args.vidon_epochs,
            progress_bar=True,
            callback=callback
        )
    else:
        print(f"\nVIDON Darcy 1D model loaded. Skipping training.")
    
    # Evaluate model
    print("\nEvaluating model...")
    from Data.data_utils import apply_sensor_dropoff
    
    vidon_model.eval()
    test_data = dataset['test']
    n_test = min(args.n_test_samples_eval, len(test_data))
    total_loss = 0.0
    total_rel_error = 0.0
    
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            
            # Get sensor values at sensor indices
            us_data = torch.tensor(sample['u'], device=device)[sensor_indices]
            ys_data = query_x.unsqueeze(0)
            target = torch.tensor(sample['s'], device=device)[query_indices].unsqueeze(0).unsqueeze(-1)
            
            # Apply sensor dropout if specified
            # VIDON can handle variable input sizes, so we use removal/replacement like SetONet
            if args.eval_sensor_dropoff > 0.0:
                sensor_x_used, us_used = apply_sensor_dropoff(
                    sensor_x, us_data, 
                    args.eval_sensor_dropoff, args.replace_with_nearest
                )
                # Reshape for model input: [B, S, 1]
                xs_input = sensor_x_used.unsqueeze(0)  # [1, S_remaining, 1]
                us_input = us_used.unsqueeze(0).unsqueeze(-1)  # [1, S_remaining, 1]
            else:
                xs_input = sensor_x.unsqueeze(0)  # [1, S, 1]
                us_input = us_data.unsqueeze(0).unsqueeze(-1)  # [1, S, 1]
            
            pred = vidon_model(xs_input, us_input, ys_data)
            
            mse_loss = torch.nn.MSELoss()(pred, target)
            total_loss += mse_loss.item()
            
            rel_error = calculate_l2_relative_error(pred, target)
            total_rel_error += rel_error.item()
    
    avg_loss = total_loss / n_test
    avg_rel_error = total_rel_error / n_test
    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    
    # Prepare test results for configuration saving
    test_results = {
        "relative_l2_error": avg_rel_error,
        "mse_loss": avg_loss,
        "n_test_samples": n_test
    }
    
    # Generate plots
    print("Generating plots...")
    from Plotting.plot_darcy_1d_utils import plot_darcy_comparison
    
    # Plot 3 test samples
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"darcy_results_test_sample_{i}.png")
        plot_darcy_comparison(
            model_to_use=vidon_model, dataset=dataset, sensor_x=sensor_x, query_x=query_x,
            sensor_indices=sensor_indices, query_indices=query_indices, log_dir=log_dir,
            num_samples_to_plot=1, plot_filename_prefix=f"darcy_1d_test_{i}",
            sensor_dropoff=args.eval_sensor_dropoff, replace_with_nearest=args.replace_with_nearest,
            dataset_split="test", batch_size=args.batch_size, variable_sensors=False,
            grid_points=grid_points, sensors_to_plot_fraction=0.05
        )
    
    # Save model
    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, "darcy1d_vidon_model.pth")
        torch.save(vidon_model.state_dict(), model_save_path)
        print(f"VIDON model saved to: {model_save_path}")
    
    # Save experiment configuration with test results
    save_experiment_configuration(args, vidon_model, dataset, dataset_wrapper=data_generator, device=device, log_dir=log_dir, dataset_type="darcy_1d", test_results=test_results)
    
    print("Training completed!")

if __name__ == "__main__":
    main()

