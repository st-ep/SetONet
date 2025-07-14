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
from Plotting.plot_darcy_2d_utils import plot_multiple_darcy_results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for 2D Darcy flow problem.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="/home/titanv/Stepan/setprojects/SetONet/Data/darcy_2d_data/darcy64", 
                       help='Path to Darcy 2D dataset')
    
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
    parser.add_argument('--son_epochs', type=int, default=5000, help='Number of epochs for SetONet')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency for sinusoidal positional encoding')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet_darcy2d"
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

class DarcyDataset:
    """Dataset wrapper for 2D Darcy flow data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("Loading Darcy 2D dataset...")
        
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        k_field = np.array(sample_0['k'])  # permeability field (65x65)
        
        self.grid_size = k_field.shape[0]  # Should be 65 for 64x64 grid
        self.n_points = self.grid_size * self.grid_size  # Total number of points
        self.input_dim = 2  # 2D coordinates (x, y)
        
        print(f"Grid size: {self.grid_size}x{self.grid_size}, Total points: {self.n_points}")
        
        # Create coordinate grid (same for all samples)
        x = np.linspace(0.0, 1.0, self.grid_size)
        y = np.linspace(0.0, 1.0, self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
        self.coords = torch.tensor(coords, device=device, dtype=torch.float32)
        
        # Pre-allocate tensors on GPU for ALL data
        self.k_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        self.p_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        
        # Load data to GPU (flatten the 2D fields)
        for i in range(self.n_samples):
            sample = train_data[i]
            self.k_data[i] = torch.tensor(np.array(sample['k']).flatten(), device=device, dtype=torch.float32)
            self.p_data[i] = torch.tensor(np.array(sample['p']).flatten(), device=device, dtype=torch.float32)
        
        print(f"Dataset loaded: {self.n_samples} samples, {self.n_points} points per sample")
        
    def sample(self, device=None):
        """Sample a batch for training."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        # Source coordinates (same for all samples)
        xs = self.coords.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Source values (permeability k)
        us = self.k_data[indices].unsqueeze(-1)
        
        # Target coordinates (same as source coordinates for Darcy problem)
        ys = xs.clone()
        
        # Target values (pressure p)
        G_u_ys = self.p_data[indices].unsqueeze(-1)
        
        return xs, us, ys, G_u_ys, None

def create_model(args, device):
    """Create SetONet model for 2D Darcy problem."""
    activation_fn = get_activation_function(args.activation_fn)
    
    model = SetONet(
        input_size_src=2,  # 2D coordinates (x, y)
        output_size_src=1,  # Scalar permeability values
        input_size_tgt=2,  # 2D coordinates (x, y)
        output_size_tgt=1,  # Scalar pressure values
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

def evaluate_model(model, dataset, darcy_dataset, device, n_test_samples=100):
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
            k_field = torch.tensor(np.array(sample['k']).flatten(), dtype=torch.float32, device=device)
            p_field = torch.tensor(np.array(sample['p']).flatten(), dtype=torch.float32, device=device)
            
            # Use the same coordinates as training
            xs = darcy_dataset.coords.unsqueeze(0)  # (1, n_points, 2)
            us = k_field.unsqueeze(0).unsqueeze(-1)  # (1, n_points, 1)
            ys = xs.clone()  # Same coordinates for target
            target = p_field.unsqueeze(0).unsqueeze(-1)  # (1, n_points, 1)
            
            # Forward pass
            pred = model(xs, us, ys)
            
            # Calculate metrics
            mse_loss = torch.nn.MSELoss()(pred, target)
            total_loss += mse_loss.item()
            
            rel_error = calculate_l2_relative_error(pred, target)
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
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    try:
        # Load train and test datasets separately
        from datasets import DatasetDict
        train_dataset = load_from_disk(os.path.join(args.data_path, "train"))
        test_dataset = load_from_disk(os.path.join(args.data_path, "test"))
        
        # Combine into DatasetDict
        dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/darcy_2d_data/make_dataset.py")
        return
    
    # Create dataset wrapper
    darcy_dataset = DarcyDataset(dataset, batch_size=args.batch_size, device=device)
    
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
        dataset=darcy_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=None
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, dataset, darcy_dataset, device, n_test_samples=100)
    
    # Generate plots
    print("\nGenerating result plots...")
    plot_multiple_darcy_results(model, dataset, darcy_dataset, device, log_dir, n_samples=3)
    
    # Save model
    model_save_path = os.path.join(log_dir, "darcy2d_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main()