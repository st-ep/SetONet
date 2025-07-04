import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse
from datasets import load_from_disk

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from Models.SetONet import SetONet
import torch.nn as nn
from Models.utils.helper_utils import calculate_l2_relative_error
from Plotting.plot_chladni_utils import plot_chladni_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for Chladni plate problem.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="/home/titanv/Stepan/setprojects/SetONet/Data/chladni_dataset", 
                       help='Path to Chladni dataset')
    
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

class ChladniDataset:
    """Dataset wrapper for Chladni plate data that uses pre-normalized data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("Loading pre-normalized Chladni dataset...")
        
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        self.n_points = len(sample_0['X'])
        self.input_dim = len(sample_0['X'][0])
        
        # Pre-allocate tensors on GPU for ALL data
        self.X_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.u_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        self.Y_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        
        # Load pre-normalized data to GPU
        for i in range(self.n_samples):
            sample = train_data[i]
            self.X_data[i] = torch.tensor(sample['X'], device=device, dtype=torch.float32)
            self.u_data[i] = torch.tensor(sample['u'], device=device, dtype=torch.float32)
            self.Y_data[i] = torch.tensor(sample['Y'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(sample['s'], device=device, dtype=torch.float32)
        
        print(f"Dataset loaded: {self.n_samples} samples, {self.n_points} points")
        
        # Load normalization statistics
        import json
        with open('Data/chladni_normalization_stats.json', 'r') as f:
            stats = json.load(f)
        
        self.u_mean = torch.tensor(stats['u_mean'], device=device, dtype=torch.float32)
        self.u_std = torch.tensor(stats['u_std'], device=device, dtype=torch.float32)
        self.s_mean = torch.tensor(stats['s_mean'], device=device, dtype=torch.float32)
        self.s_std = torch.tensor(stats['s_std'], device=device, dtype=torch.float32)
        self.xy_mean = torch.tensor(stats['xy_mean'], device=device, dtype=torch.float32)
        self.xy_std = torch.tensor(stats['xy_std'], device=device, dtype=torch.float32)
        
    def sample(self, device=None):
        """Sample a batch using pre-normalized GPU tensors."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        xs = self.X_data[indices]
        us = self.u_data[indices].unsqueeze(-1)
        ys = self.Y_data[indices]
        G_u_ys = self.s_data[indices].unsqueeze(-1)
        
        return xs, us, ys, G_u_ys, None
    
    def denormalize_displacement(self, s_norm):
        """Denormalize displacement predictions."""
        return s_norm * (self.s_std + 1e-8) + self.s_mean
    
    def denormalize_force(self, u_norm):
        """Denormalize force values."""
        return u_norm * (self.u_std + 1e-8) + self.u_mean
    
    def denormalize_coordinates(self, coords_norm):
        """Denormalize coordinates."""
        return coords_norm * self.xy_std + self.xy_mean

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
    
    # Setup logging
    log_dir = setup_logging(project_root)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    try:
        dataset = load_from_disk(args.data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/chladni_plate_generator.py")
        return
    
    # Create dataset wrapper
    chladni_dataset = ChladniDataset(dataset, batch_size=args.batch_size, device=device)
    
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
        dataset=chladni_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=None
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100, 
                   eval_sensor_dropoff=args.eval_sensor_dropoff, 
                   replace_with_nearest=args.replace_with_nearest)
    
    # Plot results
    print("Generating plots...")
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"chladni_results_sample_{i}.png")
        plot_chladni_results(model, dataset, chladni_dataset, device, sample_idx=i, save_path=plot_save_path,
                            eval_sensor_dropoff=args.eval_sensor_dropoff, 
                            replace_with_nearest=args.replace_with_nearest)
    
    # Save model
    model_save_path = os.path.join(log_dir, "chladni_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()