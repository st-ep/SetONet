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

# Import required modules
from Models.SetONet import SetONet
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Plotting.plot_elastic_2d_utils import plot_elastic_results
from Data.elastic_2d_data.elastic_2d_dataset import load_elastic_dataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for Elastic 2D plate problem.")
    
    # Data parameters
    # Default path relative to project root
    default_data_path = os.path.join(project_root, "Data", "elastic_2d_data", "elastic_dataset")
    parser.add_argument('--data_path', type=str, default=default_data_path, 
                       help='Path to Elastic 2D dataset')
    
    # Model architecture
    parser.add_argument('--son_p_dim', type=int, default=128, help='Latent dimension p for SetONet')
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
    parser.add_argument('--son_quad_dk', type=int, default=64, help='Quadrature/adaptive quadrature key/query dim (default: 64)')
    parser.add_argument('--son_quad_dv', type=int, default=None, help='Quadrature/adaptive quadrature value dim (default: son_phi_output_size)')
    parser.add_argument('--son_quad_key_hidden', type=int, default=None, help='Quadrature key MLP hidden width (default: son_rho_hidden)')
    parser.add_argument('--son_quad_key_layers', type=int, default=3, help='Quadrature key MLP depth (>=2)')
    parser.add_argument('--son_quad_phi_activation', type=str, default="tanh", choices=["tanh", "softsign", "softplus"], help='Quadrature Phi activation')
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
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency for sinusoidal positional encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    
    # Sensor dropout for training and evaluation
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during training (0.0-1.0). Makes model more robust to sensor failures')
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation only (0.0-1.0). Simulates sensor failures during testing')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors instead of removing them (leverages permutation invariance)')
    
    # Random seed and device
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use.')
    
    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging of training metrics')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='How often to evaluate on test set for TensorBoard logging (in steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples to use for TensorBoard evaluation')
    
    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None, help='Custom log directory (overrides default timestamped dir)')
    
    return parser.parse_args()

def setup_logging(project_root, custom_log_dir=None):
    """Setup logging directory or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join(project_root, "logs")
        model_folder_name = "SetONet_elastic2d"
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
    """Create SetONet model for Elastic 2D problem."""
    activation_fn = get_activation_function(args.activation_fn)
    
    model = SetONet(
        input_size_src=2,  # 2D coordinates (x, y)
        output_size_src=1,  # Scalar force values
        input_size_tgt=2,  # 2D coordinates (x, y)
        output_size_tgt=1,  # Scalar displacement values
        p=args.son_p_dim,
        phi_hidden_size=args.son_phi_hidden,
        rho_hidden_size=args.son_rho_hidden,
        trunk_hidden_size=args.son_trunk_hidden,
        n_trunk_layers=args.son_n_trunk_layers,
        activation_fn=activation_fn,
        use_deeponet_bias=True,
        phi_output_size=args.son_phi_output_size,
        initial_lr=args.son_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
        pos_encoding_type=args.pos_encoding_type,  # Use the argument instead of hardcoded 'skip'
        pos_encoding_dim=args.pos_encoding_dim,  # Add positional encoding dimension
        pos_encoding_max_freq=args.pos_encoding_max_freq,  # Add max frequency parameter
        aggregation_type=args.son_aggregation,
        use_positional_encoding=(args.pos_encoding_type != 'skip'),  # Enable if not 'skip'
        attention_n_tokens=1,
        branch_head_type=args.son_branch_head_type,
        pg_dk=args.son_pg_dk,
        pg_dv=args.son_pg_dv,
        pg_use_logw=(not args.son_pg_no_logw),
        galerkin_dk=args.son_galerkin_dk,
        galerkin_dv=args.son_galerkin_dv,
        quad_dk=args.son_quad_dk,
        quad_dv=args.son_quad_dv,
        quad_key_hidden=args.son_quad_key_hidden,
        quad_key_layers=args.son_quad_key_layers,
        quad_phi_activation=args.son_quad_phi_activation,
        galerkin_normalize=args.son_galerkin_normalize,
        galerkin_learn_temperature=args.son_galerkin_learn_temperature,
        adapt_quad_rank=args.son_adapt_quad_rank,
        adapt_quad_hidden=args.son_adapt_quad_hidden,
        adapt_quad_scale=args.son_adapt_quad_scale,
    ).to(device)

    return model

def evaluate_model(model, dataset, elastic_dataset, device, n_test_samples=100, eval_sensor_dropoff=0.0, replace_with_nearest=False):
    """Evaluate the model on test data."""
    model.eval()
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))
    
    # Print sensor dropout configuration
    if eval_sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if replace_with_nearest else "removal"
        print(f"Applying sensor drop-off rate: {eval_sensor_dropoff:.1%} with {replacement_mode}")
        print("(This tests model robustness to sensor failures)")
    
    total_loss = 0.0
    total_rel_error = 0.0
    
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            
            # Load pre-normalized data
            xs_norm = torch.tensor(sample['X'], dtype=torch.float32, device=device)
            xs = xs_norm.unsqueeze(0)
            
            us_norm = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)
            us = us_norm.unsqueeze(-1)
            
            ys_norm = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
            ys = ys_norm.unsqueeze(0)
            
            target_norm = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)
            target = target_norm.unsqueeze(-1)
            
            # Apply sensor dropout if specified
            xs_used = xs
            us_used = us
            if eval_sensor_dropoff > 0.0:
                from Data.data_utils import apply_sensor_dropoff
                
                # Apply dropout to sensor data (remove batch dimension for dropout function)
                xs_dropped, us_dropped = apply_sensor_dropoff(
                    xs.squeeze(0),  # Remove batch dimension: (n_sensors, 2)
                    us.squeeze(0).squeeze(-1),  # Remove batch and feature dimensions: (n_sensors,)
                    eval_sensor_dropoff,
                    replace_with_nearest
                )
                
                # Add batch dimension back
                xs_used = xs_dropped.unsqueeze(0)  # (1, n_remaining_sensors, 2)
                us_used = us_dropped.unsqueeze(0).unsqueeze(-1)  # (1, n_remaining_sensors, 1)
            
            # Forward pass
            pred_norm = model(xs_used, us_used, ys)
            
            # Calculate metrics
            mse_loss = torch.nn.MSELoss()(pred_norm, target)
            total_loss += mse_loss.item()
            
            rel_error = calculate_l2_relative_error(pred_norm, target)
            total_rel_error += rel_error.item()
    
    avg_loss = total_loss / n_test
    avg_rel_error = total_rel_error / n_test
    
    if eval_sensor_dropoff > 0.0:
        replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
        print(f"Test Results with {eval_sensor_dropoff:.1%} sensor dropout ({replacement_mode}) - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    else:
        print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    
    model.train()
    return avg_loss, avg_rel_error



def main():
    """Main training function."""
    # Parse arguments and setup
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed and ensure reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # For multi-GPU setups
    
    # For better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Validate arguments
    if not 0.0 <= args.train_sensor_dropoff <= 1.0:
        raise ValueError("--train_sensor_dropoff must be between 0.0 and 1.0")
    if not 0.0 <= args.eval_sensor_dropoff <= 1.0:
        raise ValueError("--eval_sensor_dropoff must be between 0.0 and 1.0")
    
    # Print sensor dropout configuration
    if args.train_sensor_dropoff > 0.0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Training with sensor drop-off rate: {args.train_sensor_dropoff:.1%} ({replacement_mode})")
        print("(This makes the model more robust to sensor failures)")
    
    # Setup logging
    log_dir = setup_logging(project_root, args.log_dir)
    
    # Load dataset using the new function
    dataset, elastic_dataset = load_elastic_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=str(device),
        train_sensor_dropoff=args.train_sensor_dropoff,
        replace_with_nearest=args.replace_with_nearest
    )
    
    if dataset is None or elastic_dataset is None:
        return
    
    # Create model
    print("Creating SetONet model...")
    model = create_model(args, device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load pre-trained model if specified
    if args.load_model_path:
        print(f"Loading pre-trained model from: {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
    
    # Setup TensorBoard callback if enabled
    callback = None
    if args.enable_tensorboard:
        print("Setting up TensorBoard logging...")
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        callback = TensorBoardCallback(
            log_dir=tb_log_dir,
            dataset=dataset,
            dataset_wrapper=elastic_dataset,
            device=device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=args.eval_sensor_dropoff,
            replace_with_nearest=args.replace_with_nearest
        )
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {tb_log_dir}")
    
    # Train model
    print(f"\nStarting training for {args.son_epochs} epochs...")
    
    model.train_model(
        dataset=elastic_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=callback
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    avg_loss, avg_rel_error = evaluate_model(model, dataset, elastic_dataset, device, n_test_samples=100, 
                                            eval_sensor_dropoff=args.eval_sensor_dropoff, 
                                            replace_with_nearest=args.replace_with_nearest)
    
    # Prepare test results for configuration saving
    test_results = {
        "relative_l2_error": avg_rel_error,
        "mse_loss": avg_loss,
        "n_test_samples": 100
    }
    
    # Plot results
    print("Generating plots...")
    # Plot 3 test samples
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"elastic_results_test_sample_{i}.png")
        plot_elastic_results(model, dataset, elastic_dataset, device, sample_idx=i, 
                           save_path=plot_save_path, dataset_split="test",
                           eval_sensor_dropoff=args.eval_sensor_dropoff, 
                           replace_with_nearest=args.replace_with_nearest)
    
    # Plot 3 train samples  
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"elastic_results_train_sample_{i}.png")
        plot_elastic_results(model, dataset, elastic_dataset, device, sample_idx=i, 
                           save_path=plot_save_path, dataset_split="train",
                           eval_sensor_dropoff=args.eval_sensor_dropoff, 
                           replace_with_nearest=args.replace_with_nearest)
    
    # Save experiment configuration with test results
    save_experiment_configuration(args, model, dataset, elastic_dataset, device, log_dir, dataset_type="elastic_2d", test_results=test_results)
    
    # Save model
    model_save_path = os.path.join(log_dir, "elastic2d_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()
