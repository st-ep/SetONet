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
from Plotting.plot_chladni_utils import plot_chladni_results
from Data.chladni_data.chladni_2d_dataset import load_chladni_dataset, ChladniDataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for Chladni plate problem.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None, 
                       help='Path to Chladni dataset (if None, uses Data/chladni_data/chladni_dataset relative to project root)')
    
    # Model architecture
    parser.add_argument('--son_p_dim', type=int, default=128, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention"], help='Aggregation type for SetONet')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for SetONet networks')
    
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
    
    # Evaluation robustness testing (sensor failures)
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation only (0.0-1.0). Simulates sensor failures during testing')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors instead of removing them (leverages permutation invariance)')
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during training (0.0-1.0). Makes model more robust to sensor failures')
    
    # GPU selection
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (0, 1, etc.). If not specified, uses CUDA_VISIBLE_DEVICES or auto-detects')
    
    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging of training metrics')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='How often to evaluate on test set for TensorBoard logging (in steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples to use for TensorBoard evaluation')
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet_chladni"
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
    """Create SetONet model for Chladni problem."""
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
    ).to(device)
    
    return model

def evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100, eval_sensor_dropoff=0.0, replace_with_nearest=False):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate arguments
    if not 0.0 <= args.eval_sensor_dropoff <= 1.0:
        raise ValueError("--eval_sensor_dropoff must be between 0.0 and 1.0")
    
    if not 0.0 <= args.train_sensor_dropoff <= 1.0:
        raise ValueError("--train_sensor_dropoff must be between 0.0 and 1.0")

    if args.train_sensor_dropoff > 0.0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Training with sensor drop-off rate: {args.train_sensor_dropoff:.1%} ({replacement_mode})")
        print("(This makes the model more robust to sensor failures)")
    
    # Setup data path
    if args.data_path is None:
        args.data_path = os.path.join(project_root, "Data", "chladni_data", "chladni_dataset")
    
    # Setup logging
    log_dir = setup_logging(project_root)
    
    # Load dataset using the new function
    dataset, _ = load_chladni_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=str(device)  # Convert device to string
    )
    
    if dataset is None:
        return
    
    # Create dataset wrapper with corresponding normalization stats path
    normalization_stats_path = os.path.join(os.path.dirname(args.data_path), 'chladni_normalization_stats.json')
    chladni_dataset = ChladniDataset(dataset, batch_size=args.batch_size, device=str(device), 
                                   normalization_stats_path=normalization_stats_path,
                                   train_sensor_dropoff=args.train_sensor_dropoff,
                                   replace_with_nearest=args.replace_with_nearest)
    
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
            dataset_wrapper=chladni_dataset,
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
        dataset=chladni_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=callback
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    avg_loss, avg_rel_error = evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100, 
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
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"chladni_results_sample_{i}.png")
        plot_chladni_results(model, dataset, chladni_dataset, device, sample_idx=i, save_path=plot_save_path,
                            eval_sensor_dropoff=args.eval_sensor_dropoff, 
                            replace_with_nearest=args.replace_with_nearest)
    
    # Save experiment configuration with test results
    save_experiment_configuration(args, model, dataset, chladni_dataset, device, log_dir, dataset_type="chladni_2d", test_results=test_results)
    
    # Save model
    model_save_path = os.path.join(log_dir, "chladni_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()