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

from Models.utils.config_utils import save_experiment_configuration
from Data.synthetic_1d_data import (
    Synthetic1DDataGenerator, create_synthetic_setonet_model, 
    load_synthetic_pretrained_model
)
from Plotting.plotting_utils import plot_synthetic_1d_comparison

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for synthetic 1D benchmark tasks.")
    
    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True, choices=['integral', 'derivative'], 
                       help='Benchmark task: integral (f\' -> f) or derivative (f -> f\')')
    
    # Model architecture
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention", "sum"], help='Aggregation type for SetONet')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for SetONet networks')
    parser.add_argument(
        '--son_branch_head_type',
        type=str,
        default="standard",
        choices=["standard", "petrov_attention", "galerkin_pou", "quadrature", "adaptive_quadrature"],
        help="Branch head type: standard (pool+rho), petrov_attention (Petrov-Galerkin attention), galerkin_pou (Galerkin partition-of-unity), quadrature (additive quadrature over learned test functions), or adaptive_quadrature (input-adaptive quadrature).",
    )
    parser.add_argument('--son_pg_dk', type=int, default=None, help='PG attention key/query dim (default: son_phi_output_size)')
    parser.add_argument('--son_pg_dv', type=int, default=None, help='PG attention value dim (default: son_phi_output_size)')
    parser.add_argument(
        '--son_pg_no_logw',
        action='store_true',
        help='Disable adding log(sensor_weights) to PG attention logits (weights are unused by default).',
    )
    parser.add_argument('--son_galerkin_dk', type=int, default=None, help='Galerkin PoU key/query dim (default: son_phi_output_size)')
    parser.add_argument('--son_galerkin_dv', type=int, default=None, help='Galerkin PoU value dim (default: son_phi_output_size)')
    parser.add_argument('--son_quad_dk', type=int, default=None, help='Quadrature/adaptive quadrature key/query dim (default: son_phi_output_size)')
    parser.add_argument('--son_quad_dv', type=int, default=None, help='Quadrature/adaptive quadrature value dim (default: son_phi_output_size)')
    parser.add_argument(
        '--son_galerkin_normalize',
        type=str,
        default="total",
        choices=["none", "total", "token"],
        help='Galerkin PoU normalization: "none" (no norm), "total" (divide by total weight), "token" (per-token mass norm).',
    )
    parser.add_argument(
        '--son_galerkin_learn_temperature',
        action='store_true',
        help='Learn temperature parameter for Galerkin PoU softmax sharpness.',
    )
    parser.add_argument('--son_adapt_quad_rank', type=int, default=4, help='Adaptive quadrature low-rank update rank R')
    parser.add_argument('--son_adapt_quad_hidden', type=int, default=64, help='Adaptive quadrature adapter MLP hidden dimension')
    parser.add_argument('--son_adapt_quad_scale', type=float, default=0.1, help='Adaptive quadrature tanh-bounded update scale')

    # Training parameters
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=125000, help='Number of epochs for SetONet')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for sinusoidal positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency/scale for sinusoidal encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Data generation
    parser.add_argument('--variable_sensors', action='store_true', help='Use different random sensor locations for each sample (more challenging)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    
    # Evaluation robustness testing (sensor failures)
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation only (0.0-1.0). Simulates sensor failures during testing')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors instead of removing them (leverages permutation invariance)')
    parser.add_argument('--n_test_samples_eval', type=int, default=1000, help='Number of test samples for evaluation')
    
    # Model loading and misc
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use.')
    
    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='TensorBoard evaluation frequency (steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples for TensorBoard')
    
    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None, help='Custom log directory (overrides default timestamped dir)')
    
    return parser.parse_args()

def setup_logging(benchmark_type, custom_log_dir=None):
    """Setup logging directory based on benchmark type or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join("logs")
        model_folder_name = f"SetONet_{benchmark_type}"  # SetONet_derivative or SetONet_integral
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
        'eval_sensor_dropoff': args.eval_sensor_dropoff,
        'replace_with_nearest': args.replace_with_nearest,
    }

def create_sensor_points(params, device):
    """Create sensor points - either fixed random or variable per batch."""
    if params.get('variable_sensors', False):
        # Sensor points will be generated per batch in data generator
        print(f"Using variable random sensor locations (same within batch, different between batches)")
        return None  # Signal that sensors are variable
    else:
        # Use fixed random sensor locations (same across all samples)
        torch.manual_seed(params['sensor_seed'])  # Fixed seed for reproducible sensor locations
        sensor_x = torch.rand(params['sensor_size'], device=device) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
        sensor_x = sensor_x.sort()[0]  # Sort for better visualization
        sensor_x = sensor_x.view(-1, 1)
        print(f"Using {params['sensor_size']} fixed random sensor locations (sorted, seed={params['sensor_seed']})")
        return sensor_x

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
    
    print(f"Running benchmark: {args.benchmark}")
    if args.eval_sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Will test robustness with sensor drop-off rate: {args.eval_sensor_dropoff:.1%} using {replacement_mode}")
        print("(Training will use full sensor data)")
    
    log_dir = setup_logging(args.benchmark, args.log_dir)
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
    
    # Create sensor points
    sensor_x_original = create_sensor_points(params, device)
    
    # Create model
    setonet_model = create_synthetic_setonet_model(args, device)
    
    # Load pre-trained model if specified
    model_was_loaded = load_synthetic_pretrained_model(setonet_model, args, device)
    
    # Create data generator
    data_generator = Synthetic1DDataGenerator(params, device, sensor_x_original, args.benchmark)
    
    # Setup TensorBoard callback if enabled
    callback = None
    if args.enable_tensorboard:
        callback = data_generator.get_tensorboard_callback(log_dir, args)
    
    # Training
    if not model_was_loaded:
        print(f"\nStarting training for {args.son_epochs} epochs...")
        setonet_model.train_model(
            dataset=data_generator,
            epochs=args.son_epochs,
            progress_bar=True,
            callback=callback
        )
    else:
        print(f"\nSetONet {args.benchmark} model loaded. Skipping training.")
    
    # Evaluate model
    print("\nEvaluating model...")
    avg_loss, avg_rel_error = data_generator.evaluate_model(
        setonet_model, 
        args.n_test_samples_eval, 
        args.batch_size, 
        args.eval_sensor_dropoff, 
        args.replace_with_nearest
    )
    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    
    # Prepare test results for configuration saving
    test_results = data_generator.prepare_test_results(avg_loss, avg_rel_error, args.n_test_samples_eval)
    
    # Generate plots
    plot_synthetic_1d_comparison(setonet_model, data_generator, params, log_dir, args, show_sensor_markers=False)
    
    # Save model
    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, f"setonet_{args.benchmark}_model.pth")
        torch.save(setonet_model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
    
    # Save experiment configuration with test results
    dummy_dataset = {'train': [], 'test': []}  # Create dummy dataset for config saving
    args.data_path = f"synthetic_1d_{args.benchmark}"  # Add missing attribute expected by config_utils
    save_experiment_configuration(args, setonet_model, dummy_dataset, dataset_wrapper=data_generator, device=device, log_dir=log_dir, dataset_type=args.benchmark, test_results=test_results)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
