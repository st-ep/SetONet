import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from Models.SetONet import SetONet
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Data.concentration_data.concentration_2d_dataset import load_concentration_dataset
from Plotting.plot_consentration_2d_utils import plot_concentration_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for 2D concentration problem.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="Data/concentration_data/chem_plume_dataset", 
                       help='Path to Concentration 2D dataset')
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
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet_concentration2d"
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
    """Create SetONet model for 2D concentration problem."""
    activation_fn = get_activation_function(args.activation_fn)
    
    model = SetONet(
        input_size_src=2,  # 2D coordinates (x, y) of sources
        output_size_src=1,  # Scalar emission rate values
        input_size_tgt=2,  # 2D coordinates (x, y) of grid points
        output_size_tgt=1,  # Scalar concentration values
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

def evaluate_model(model, dataset, concentration_dataset, device, n_test_samples=100):
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
            source_rates = sources[:, 2:3].unsqueeze(0)  # (1, n_sources, 1)
            
            # Handle adaptive vs uniform mesh
            if concentration_dataset.is_adaptive:
                # Adaptive mesh: different grid points per sample
                target_coords = torch.tensor(np.array(sample['grid_coords']), device=device, dtype=torch.float32).unsqueeze(0)
                target_concs = torch.tensor(np.array(sample['field_values']), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            else:
                # Uniform mesh: same grid for all samples
                target_coords = concentration_dataset.grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
                conc_field = torch.tensor(np.array(sample['field'])[:, :, 0].flatten(), device=device, dtype=torch.float32)
                target_concs = conc_field.unsqueeze(0).unsqueeze(-1)  # (1, n_grid_points, 1)
            
            # Forward pass
            pred = model(source_coords, source_rates, target_coords)
            
            # Calculate metrics
            mse_loss = torch.nn.MSELoss()(pred, target_concs)
            total_loss += mse_loss.item()
            
            rel_error = calculate_l2_relative_error(pred, target_concs)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = setup_logging(project_root)
    
    # Load dataset using the new function
    dataset, concentration_dataset = load_concentration_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device
    )
    
    if dataset is None or concentration_dataset is None:
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
    
    # Train model
    print(f"\nStarting training for {args.son_epochs} epochs...")
    
    model.train_model(
        dataset=concentration_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=None
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, dataset, concentration_dataset, device, n_test_samples=100)
    
    # Plot results
    print("Generating plots...")
    # Plot 3 test samples
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"concentration_results_test_sample_{i}.png")
        plot_concentration_results(model, dataset, concentration_dataset, device, sample_idx=i, 
                                 save_path=plot_save_path, dataset_split="test")
    
    # Plot 3 train samples  
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"concentration_results_train_sample_{i}.png")
        plot_concentration_results(model, dataset, concentration_dataset, device, sample_idx=i, 
                                 save_path=plot_save_path, dataset_split="train")
    
    # Save model
    model_save_path = os.path.join(log_dir, "concentration2d_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()