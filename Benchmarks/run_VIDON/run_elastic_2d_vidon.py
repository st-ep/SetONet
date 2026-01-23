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
from Models.VIDON import VIDON
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils_vidon import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Plotting.plot_elastic_2d_utils import plot_elastic_results
from Data.elastic_2d_data.elastic_2d_dataset import load_elastic_dataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VIDON for Elastic 2D plate problem.")
    
    # Data parameters
    # Default path relative to project root
    default_data_path = os.path.join(project_root, "Data", "elastic_2d_data", "elastic_dataset")
    parser.add_argument('--data_path', type=str, default=default_data_path, 
                       help='Path to Elastic 2D dataset')
    
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
    
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for networks')
    
    # Training parameters
    parser.add_argument('--vidon_lr', type=float, default=5e-4, help='Learning rate for VIDON')
    parser.add_argument('--vidon_epochs', type=int, default=125000, help='Number of epochs for VIDON')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained VIDON model')
    
    # Sensor dropout for training and evaluation
    # VIDON can handle variable input sizes like SetONet, so it uses removal/replacement (not interpolation)
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during training (0.0-1.0). Makes model more robust to sensor failures')
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation only (0.0-1.0). Sensors are removed or replaced.')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors instead of removing them')
    
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
        model_folder_name = "VIDON_elastic2d"
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
    """Create VIDON model for Elastic 2D problem."""
    print(f"\n--- Initializing VIDON Model for Elastic 2D ---")
    print(f"Using activation function: {args.activation_fn}")
    
    activation_fn = get_activation_function(args.activation_fn)
    
    model = VIDON(
        input_size_src=2,  # 2D coordinates (x, y)
        output_size_src=1,  # Scalar force values
        input_size_tgt=2,  # 2D coordinates (x, y)
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
            # VIDON can handle variable input sizes, so we use removal/replacement like SetONet
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
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")
    if not 0.0 <= args.train_sensor_dropoff <= 1.0:
        raise ValueError("--train_sensor_dropoff must be between 0.0 and 1.0")
    if not 0.0 <= args.eval_sensor_dropoff <= 1.0:
        raise ValueError("--eval_sensor_dropoff must be between 0.0 and 1.0")
    
    # Print sensor dropout configuration
    if args.train_sensor_dropoff > 0.0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Training with sensor drop-off rate: {args.train_sensor_dropoff:.1%} ({replacement_mode})")
        print("(This makes the model more robust to sensor failures)")
    
    if args.eval_sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Will test robustness with sensor drop-off rate: {args.eval_sensor_dropoff:.1%} using {replacement_mode}")
    
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
    print("Creating VIDON model...")
    model = create_model(args, device)
    
    # Load pre-trained model if specified
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            print(f"Loading pre-trained model from: {args.load_model_path}")
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")
            args.load_model_path = None
    
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
    if args.load_model_path is None:
        print(f"\nStarting training for {args.vidon_epochs} epochs...")
        
        model.train_model(
            dataset=elastic_dataset,
            epochs=args.vidon_epochs,
            progress_bar=True,
            callback=callback
        )
    else:
        print("\nVIDON Elastic 2D model loaded. Skipping training.")
    
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
    if args.load_model_path is None:
        model_save_path = os.path.join(log_dir, "elastic2d_vidon_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"VIDON model saved to: {model_save_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()

