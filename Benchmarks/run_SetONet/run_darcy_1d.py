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
    parser = argparse.ArgumentParser(description="Train SetONet for Darcy 1D equation.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="Data/darcy_1d_data/darcy_1d_dataset_501", 
                       help='Path to Darcy 1D dataset')
    parser.add_argument('--sensor_size', type=int, default=300, help='Number of sensor locations (max 501 for Darcy 1D grid)')
    
    # Model architecture
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
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
    
    # Sensor dropout and evaluation
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation (0.0-1.0)')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors')
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during training (0.0-1.0)')
    parser.add_argument('--n_test_samples_eval', type=int, default=1000, help='Number of test samples for evaluation')
    parser.add_argument('--n_query_points', type=int, default=300, help='Number of query points for evaluation')
    
    # Model loading and misc
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use.')
    
    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='TensorBoard evaluation frequency (steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples for TensorBoard')
    
    return parser.parse_args()

def setup_logging():
    """Setup logging directory."""
    logs_base_in_project = os.path.join("logs")
    model_folder_name = "SetONet_darcy_1d"
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
    """Create SetONet model for Darcy 1D problem."""
    print(f"\n--- Initializing SetONet Model for {args.benchmark} ---")
    print(f"Using activation function: {args.activation_fn}")
    
    activation_fn = get_activation_function(args.activation_fn)
    
    model = SetONet(
        input_size_src=1,  # 1D coordinates (x)
        output_size_src=1,  # Scalar force values
        input_size_tgt=1,  # 1D coordinates (x)
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
        pos_encoding_type=args.pos_encoding_type,
        pos_encoding_dim=args.pos_encoding_dim,
        pos_encoding_max_freq=args.pos_encoding_max_freq,
        aggregation_type=args.son_aggregation,
        use_positional_encoding=(args.pos_encoding_type != 'skip'),
        attention_n_tokens=1,
    ).to(device)
    
    return model

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





def main():
    """Main training function."""
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Add benchmark argument for compatibility
    args.benchmark = 'darcy_1d'
    
    # Load dataset
    dataset = load_darcy_dataset(args.data_path)
    grid_points = torch.tensor(dataset['train'][0]['X'], dtype=torch.float32)
    
    log_dir = setup_logging()
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
    setonet_model = create_setonet_model(args, device)
    
    # Load pre-trained model if specified
    model_was_loaded = load_pretrained_model(setonet_model, args, device)
    
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
        print(f"\nStarting training for {args.son_epochs} epochs...")
        setonet_model.train_model(
            dataset=data_generator,
            epochs=args.son_epochs,
            progress_bar=True,
            callback=callback
        )
    else:
        print(f"\nSetONet Darcy 1D model loaded. Skipping training.")
    
    # Evaluate model
    print("\nEvaluating model...")
    from Data.data_utils import apply_sensor_dropoff
    
    setonet_model.eval()
    test_data = dataset['test']
    n_test = min(100, len(test_data))
    total_loss = 0.0
    total_rel_error = 0.0
    
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            
            xs_data = torch.tensor(sample['u'], device=device)[sensor_indices].unsqueeze(0).unsqueeze(-1)
            ys_data = query_x.unsqueeze(0)
            target = torch.tensor(sample['s'], device=device)[query_indices].unsqueeze(0).unsqueeze(-1)
            
            # Apply sensor dropout if specified
            if args.eval_sensor_dropoff > 0.0:
                xs_dropped, us_dropped = apply_sensor_dropoff(
                    sensor_x, xs_data.squeeze(0).squeeze(-1), 
                    args.eval_sensor_dropoff, args.replace_with_nearest
                )
                xs_data = us_dropped.unsqueeze(0).unsqueeze(-1)
                sensor_x_used = xs_dropped.unsqueeze(0)
            else:
                sensor_x_used = sensor_x.unsqueeze(0)
            
            pred = setonet_model(sensor_x_used, xs_data, ys_data)
            
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
            model_to_use=setonet_model, dataset=dataset, sensor_x=sensor_x, query_x=query_x,
            sensor_indices=sensor_indices, query_indices=query_indices, log_dir=log_dir,
            num_samples_to_plot=1, plot_filename_prefix=f"darcy_1d_test_{i}",
            sensor_dropoff=args.eval_sensor_dropoff, replace_with_nearest=args.replace_with_nearest,
            dataset_split="test", batch_size=args.batch_size, variable_sensors=False,
            grid_points=grid_points, sensors_to_plot_fraction=0.05
        )
    
    # Save model
    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, "darcy1d_setonet_model.pth")
        torch.save(setonet_model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")
    
    # Save experiment configuration with test results
    save_experiment_configuration(args, setonet_model, dataset, dataset_wrapper=data_generator, device=device, log_dir=log_dir, dataset_type="darcy_1d", test_results=test_results)
    
    print("Training completed!")

if __name__ == "__main__":
    main() 