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
from Models.SetONet import SetONet
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Models.utils.config_utils import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Data.heat_data.heat_2d_dataset import load_heat_dataset
from Plotting.plot_heat_2d_utils import plot_heat_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for 2D heat problem.")
    
    # Data parameters
    default_data_path = os.path.join(project_root, "Data", "heat_data", "pcb_heat_adaptive_dataset8.0_n8192_N25_P30")
    parser.add_argument('--data_path', type=str, default=default_data_path, 
                       help='Path to Heat 2D dataset')
    parser.add_argument('--adaptive_mesh', action='store_true', help='Use adaptive mesh dataset (auto-detected from data)')
    
    # Model architecture
    parser.add_argument('--son_p_dim', type=int, default=128, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention"], help='Aggregation type for SetONet')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for SetONet networks')
    parser.add_argument(
        '--son_branch_head_type',
        type=str,
        default="standard",
        choices=["standard", "petrov_attention", "galerkin_pou"],
        help="Branch head type: standard (pool+rho), petrov_attention (Petrov-Galerkin attention), or galerkin_pou (Galerkin partition-of-unity).",
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

    # Training parameters
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=50000, help='Number of epochs for SetONet')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.01, help='Max frequency for sinusoidal positional encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[15000, 30000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    
    # Random seed and device
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:1', help='Torch device to use.')
    
    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging of training metrics')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000, help='How often to evaluate on test set for TensorBoard logging (in steps)')
    parser.add_argument('--tb_test_samples', type=int, default=100, help='Number of test samples to use for TensorBoard evaluation')
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet_heat2d"
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
    """Create SetONet model for 2D heat problem."""
    activation_fn = get_activation_function(args.activation_fn)
    
    model = SetONet(
        input_size_src=2,  # 2D coordinates (x, y) of sources
        output_size_src=1,  # Scalar power values
        input_size_tgt=2,  # 2D coordinates (x, y) of grid points
        output_size_tgt=1,  # Scalar temperature values
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
        branch_head_type=args.son_branch_head_type,
        pg_dk=args.son_pg_dk,
        pg_dv=args.son_pg_dv,
        pg_use_logw=(not args.son_pg_no_logw),
        galerkin_dk=args.son_galerkin_dk,
        galerkin_dv=args.son_galerkin_dv,
        galerkin_normalize=args.son_galerkin_normalize,
        galerkin_learn_temperature=args.son_galerkin_learn_temperature,
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
            
            # Prepare test data
            sources = torch.tensor(np.array(sample['sources']), device=device, dtype=torch.float32)
            source_coords = sources[:, :2].unsqueeze(0)  # (1, n_sources, 2)
            source_powers = sources[:, 2:3].unsqueeze(0)  # (1, n_sources, 1)
            
            # Handle adaptive vs uniform mesh
            if heat_dataset.is_adaptive:
                # Adaptive mesh: different grid points per sample
                target_coords = torch.tensor(np.array(sample['grid_coords']), device=device, dtype=torch.float32).unsqueeze(0)
                target_temps = torch.tensor(np.array(sample['field_values']), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            else:
                # Uniform mesh: same grid for all samples
                target_coords = heat_dataset.grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
                temp_field = torch.tensor(np.array(sample['field'])[:, :, 0].flatten(), device=device, dtype=torch.float32)
                target_temps = temp_field.unsqueeze(0).unsqueeze(-1)  # (1, n_grid_points, 1)
            
            # Forward pass
            pred = model(source_coords, source_powers, target_coords)
            
            # Calculate metrics
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
    
    # Setup logging
    log_dir = setup_logging(project_root)
    
    # Load dataset using the new function
    dataset, heat_dataset = load_heat_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device
    )
    
    if dataset is None or heat_dataset is None:
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
            dataset_wrapper=heat_dataset,
            device=device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=0.0,  # No sensor dropout for Heat
            replace_with_nearest=False
        )
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {tb_log_dir}")
    
    # Train model
    print(f"\nStarting training for {args.son_epochs} epochs...")
    
    model.train_model(
        dataset=heat_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=callback
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    avg_loss, avg_rel_error = evaluate_model(model, dataset, heat_dataset, device, n_test_samples=100)
    
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
        plot_save_path = os.path.join(log_dir, f"heat_results_test_sample_{i}.png")
        plot_heat_results(model, dataset, heat_dataset, device, sample_idx=i, 
                         save_path=plot_save_path, dataset_split="test")
    
    # Plot 3 train samples  
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"heat_results_train_sample_{i}.png")
        plot_heat_results(model, dataset, heat_dataset, device, sample_idx=i, 
                         save_path=plot_save_path, dataset_split="train")
    
    # Save experiment configuration with test results
    save_experiment_configuration(args, model, dataset, heat_dataset, device, log_dir, dataset_type="heat_2d", test_results=test_results)
    
    # Save model
    model_save_path = os.path.join(log_dir, "heat2d_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()
