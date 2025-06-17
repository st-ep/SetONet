import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse
from datasets import load_from_disk
import matplotlib.pyplot as plt

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from Models.SetONet import SetONet
import torch.nn as nn

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
    parser.add_argument('--son_epochs', type=int, default=175000, help='Number of epochs for SetONet')
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
    """OPTIMIZED Dataset wrapper for Chladni plate data that implements the interface expected by SetONet."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("📊 Pre-loading and optimizing Chladni dataset...")
        
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        self.n_points = len(sample_0['X'])  # Number of points in the grid
        self.input_dim = len(sample_0['X'][0])  # Should be 2 for (x, y)
        
        print(f"⚡ PRE-LOADING {self.n_samples} samples to GPU for maximum efficiency...")
        
        # Pre-allocate tensors on GPU for ALL data
        self.X_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.u_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        self.Y_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        
        # Load ALL data to GPU once (much faster than per-batch loading)
        for i in range(self.n_samples):
            sample = train_data[i]
            self.X_data[i] = torch.tensor(sample['X'], device=device, dtype=torch.float32)
            self.u_data[i] = torch.tensor(sample['u'], device=device, dtype=torch.float32)
            self.Y_data[i] = torch.tensor(sample['Y'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(sample['s'], device=device, dtype=torch.float32)
        
        print(f"✅ Dataset optimized: {self.n_samples} samples, {self.n_points} points, {self.input_dim}D coordinates")
        print(f"💾 GPU memory usage: ~{(self.X_data.numel() + self.u_data.numel() + self.Y_data.numel() + self.s_data.numel()) * 4 / 1024**2:.1f} MB")
        
        # ------------------------------------------------------------------
        # NORMALISE ALL INPUT FEATURES (forces AND coordinates)
        # ------------------------------------------------------------------
        print("📏 Applying data normalization (forces, displacements, coordinates)...")
        # Force & displacement stats (per-dataset)
        self.u_mean = self.u_data.mean();  self.u_std = self.u_data.std()
        self.s_mean = self.s_data.mean();  self.s_std = self.s_data.std()
        self.u_data_norm = (self.u_data - self.u_mean) / (self.u_std + 1e-8)
        self.s_data_norm = (self.s_data - self.s_mean) / (self.s_std + 1e-8)
        
        # Coordinate stats (global over all samples & points)
        self.xy_mean = self.X_data.mean(dim=(0,1))   # (2,) - mean for each coordinate
        self.xy_std  = self.X_data.std( dim=(0,1)) + 1e-8  # (2,) - std for each coordinate
        self.X_data_norm = (self.X_data - self.xy_mean) / self.xy_std
        self.Y_data_norm = (self.Y_data - self.xy_mean) / self.xy_std
        print("✅ Data normalized: x,y,u,z now zero-mean / unit-variance")
        
    def sample(self, device=None):
        """
        OPTIMIZED sampling using pre-loaded GPU tensors.
        Returns: xs, us, ys, G_u_ys, _ (following SetONet interface)
        """
        # Sample random indices
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        # FAST: Direct indexing from pre-loaded GPU tensors (no CPU-GPU transfers!)
        batch_X = self.X_data_norm[indices]  # normalized coordinates
        batch_u = self.u_data_norm[indices]  # [batch_size, n_points] - NORMALIZED
        batch_Y = self.Y_data_norm[indices]  # normalized coords for trunk
        batch_s = self.s_data_norm[indices]  # [batch_size, n_points] - NORMALIZED
        
        # For SetONet interface:
        # xs: sensor locations (input coordinates)
        # us: sensor values (input function values)  
        # ys: target locations (output coordinates)
        # G_u_ys: target values (output function values)
        
        xs = batch_X  # Input coordinates (x, y)
        us = batch_u.unsqueeze(-1)  # Input forces - add channel dimension
        ys = batch_Y  # Output coordinates (same as input for this problem)
        G_u_ys = batch_s.unsqueeze(-1)  # Output displacements - add channel dimension
        
        return xs, us, ys, G_u_ys, None
    
    def denormalize_displacement(self, s_norm):
        """Denormalize displacement predictions back to original scale."""
        return s_norm * (self.s_std + 1e-8) + self.s_mean
    
    def denormalize_force(self, u_norm):
        """Denormalize force values back to original scale."""
        return u_norm * (self.u_std + 1e-8) + self.u_mean

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

def evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100):
    """Evaluate the model on test data using normalized loss."""
    model.eval()
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))
    
    total_loss = 0.0
    total_rel_error = 0.0
    
    print(f"Evaluating on {n_test} test samples...")
    
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            
            # Convert coordinate data directly without unnecessary reshaping
            # sample['X'] should already be shape (n_points, 2)
            xs_raw = torch.tensor(sample['X'], dtype=torch.float32, device=device)
            # Ensure we have the expected shape (n_points, 2)
            if xs_raw.shape[1] != 2:
                raise ValueError(f"Expected coordinate shape (n_points, 2), got {xs_raw.shape}")
            
            # Normalize coordinates using dataset statistics
            xs_norm = (xs_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
            xs = xs_norm.unsqueeze(0)  # Add batch dimension: (1, n_points, 2)
            
            # Process input function values (forces)
            us_orig = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_points)
            
            # Process output coordinates (same as input coordinates for this problem)
            ys_raw = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
            # Ensure we have the expected shape (n_points, 2)
            if ys_raw.shape[1] != 2:
                raise ValueError(f"Expected coordinate shape (n_points, 2), got {ys_raw.shape}")
            
            ys_norm = (ys_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
            ys = ys_norm.unsqueeze(0)  # Add batch dimension: (1, n_points, 2)
            
            # Process target function values (displacements)
            target_orig = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_points)
            
            # Normalize inputs and targets for model
            us_norm = ((us_orig - chladni_dataset.u_mean) / (chladni_dataset.u_std + 1e-8)).unsqueeze(-1)  # (1, n_points, 1)
            target_norm = ((target_orig - chladni_dataset.s_mean) / (chladni_dataset.s_std + 1e-8)).unsqueeze(-1)  # (1, n_points, 1)
            
            # Forward pass (model works with normalized data)
            pred_norm = model(xs, us_norm, ys)
            
            # Calculate loss in normalized space
            mse_loss = torch.nn.MSELoss()(pred_norm, target_norm)
            total_loss += mse_loss.item()
            
            # Calculate relative error in normalized space
            rel_error = torch.norm(pred_norm - target_norm) / torch.norm(target_norm)
            total_rel_error += rel_error.item()
    
    avg_loss = total_loss / n_test
    avg_rel_error = total_rel_error / n_test
    
    print(f"Test Results:")
    print(f"  Average MSE Loss: {avg_loss:.6e}")
    print(f"  Average Relative Error: {avg_rel_error:.6f}")
    
    model.train()
    return avg_loss, avg_rel_error

def plot_results(model, dataset, chladni_dataset, device, sample_idx=0, save_path=None):
    """Plot input forces, predicted displacements, and ground truth for a sample."""
    model.eval()
    test_data = dataset['test']
    
    # Get a test sample
    sample = test_data[sample_idx]
    
    # Convert coordinate data directly without unnecessary reshaping
    # sample['X'] should already be shape (n_points, 2)
    xs_raw = torch.tensor(sample['X'], dtype=torch.float32, device=device)
    # Ensure we have the expected shape (n_points, 2)
    if xs_raw.shape[1] != 2:
        raise ValueError(f"Expected coordinate shape (n_points, 2), got {xs_raw.shape}")
    
    # Normalize coordinates using dataset statistics
    xs_norm = (xs_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
    xs = xs_norm.unsqueeze(0)  # Add batch dimension: (1, n_points, 2)
    
    # Process input function values (forces)
    us_orig = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_points)
    
    # Process output coordinates (same as input coordinates for this problem)
    ys_raw = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
    # Ensure we have the expected shape (n_points, 2)
    if ys_raw.shape[1] != 2:
        raise ValueError(f"Expected coordinate shape (n_points, 2), got {ys_raw.shape}")
    
    ys_norm = (ys_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
    ys = ys_norm.unsqueeze(0)  # Add batch dimension: (1, n_points, 2)
    
    # Process target function values (displacements)
    target_orig = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_points)
    
    # Normalize input for model
    us_norm = ((us_orig - chladni_dataset.u_mean) / (chladni_dataset.u_std + 1e-8)).unsqueeze(-1)  # (1, n_points, 1)
    
    # Get prediction
    with torch.no_grad():
        pred_norm = model(xs, us_norm, ys)
        # Denormalize prediction
        pred_orig = chladni_dataset.denormalize_displacement(pred_norm.squeeze(-1))
    
    # Convert back to numpy and reshape to 2D grid (25x25)
    grid_size = int(np.sqrt(len(sample['X'])))  # Should be 25
    
    # Get coordinate grids for plotting
    coords = np.array(sample['X'])
    x_coords = coords[:, 0].reshape(grid_size, grid_size)
    y_coords = coords[:, 1].reshape(grid_size, grid_size)
    
    # Reshape data to 2D grids
    forces_2d = us_orig.cpu().numpy().reshape(grid_size, grid_size)
    pred_2d = pred_orig.cpu().numpy().reshape(grid_size, grid_size)
    target_2d = target_orig.cpu().numpy().reshape(grid_size, grid_size)
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.style.use('default')  # Use default style for better visibility
    
    # Plot 1: Input Forces
    im1 = axes[0].contourf(x_coords, y_coords, forces_2d, levels=20, cmap='RdBu_r')
    axes[0].set_title(f'Input Forces S(x,y)\nSample #{sample_idx}')
    axes[0].set_xlabel('X position (m)')
    axes[0].set_ylabel('Y position (m)')
    axes[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0], label='Force magnitude')
    
    # Plot 2: Predicted Displacements
    im2 = axes[1].contourf(x_coords, y_coords, pred_2d, levels=20, cmap='viridis')
    axes[1].set_title('Predicted Displacements Z(x,y)')
    axes[1].set_xlabel('X position (m)')
    axes[1].set_ylabel('Y position (m)')
    axes[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[1], label='Displacement magnitude')
    
    # Plot 3: Ground Truth Displacements
    im3 = axes[2].contourf(x_coords, y_coords, target_2d, levels=20, cmap='viridis')
    axes[2].set_title('Ground Truth Displacements Z(x,y)')
    axes[2].set_xlabel('X position (m)')
    axes[2].set_ylabel('Y position (m)')
    axes[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes[2], label='Displacement magnitude')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print some statistics
    print(f"\nSample {sample_idx} Statistics:")
    print(f"Force range: [{forces_2d.min():.6f}, {forces_2d.max():.6f}]")
    print(f"Predicted displacement range: [{pred_2d.min():.6f}, {pred_2d.max():.6f}]")
    print(f"Ground truth displacement range: [{target_2d.min():.6f}, {target_2d.max():.6f}]")
    
    # Calculate error metrics
    mse_error = np.mean((pred_2d - target_2d)**2)
    rel_error = np.linalg.norm(pred_2d - target_2d) / np.linalg.norm(target_2d)
    print(f"MSE Error: {mse_error:.6e}")
    print(f"Relative L2 Error: {rel_error:.6f}")
    
    model.train()

def debug_data_shapes(model, dataset, chladni_dataset, device):
    """Debug function to print data shapes and verify compatibility with model."""
    print("\n" + "="*50)
    print("DEBUG: Data Shape Verification")
    print("="*50)
    
    # Check training data shape
    xs, us, ys, G_u_ys, _ = chladni_dataset.sample(device=device)
    print(f"Training data shapes:")
    print(f"  xs (input coordinates): {xs.shape}")
    print(f"  us (input function values): {us.shape}")
    print(f"  ys (output coordinates): {ys.shape}")
    print(f"  G_u_ys (target function values): {G_u_ys.shape}")
    
    # Check test data shape
    test_sample = dataset['test'][0]
    print(f"\nTest data shapes (raw):")
    print(f"  sample['X']: {np.array(test_sample['X']).shape}")
    print(f"  sample['u']: {np.array(test_sample['u']).shape}")
    print(f"  sample['Y']: {np.array(test_sample['Y']).shape}")
    print(f"  sample['s']: {np.array(test_sample['s']).shape}")
    
    # Test model forward pass with training data
    print(f"\nModel forward pass test:")
    model.eval()
    with torch.no_grad():
        try:
            pred = model(xs, us, ys)
            print(f"  Model output shape: {pred.shape}")
            print(f"  Model forward pass: SUCCESS ✓")
        except Exception as e:
            print(f"  Model forward pass: FAILED ✗")
            print(f"  Error: {e}")
    
    # Test model with evaluation data format
    print(f"\nEvaluation data format test:")
    xs_raw = torch.tensor(test_sample['X'], dtype=torch.float32, device=device)
    xs_norm = (xs_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
    xs_eval = xs_norm.unsqueeze(0)
    
    us_orig = torch.tensor(test_sample['u'], dtype=torch.float32, device=device).unsqueeze(0)
    us_norm = ((us_orig - chladni_dataset.u_mean) / (chladni_dataset.u_std + 1e-8)).unsqueeze(-1)
    
    ys_raw = torch.tensor(test_sample['Y'], dtype=torch.float32, device=device)
    ys_norm = (ys_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
    ys_eval = ys_norm.unsqueeze(0)
    
    print(f"  xs_eval shape: {xs_eval.shape}")
    print(f"  us_norm shape: {us_norm.shape}")
    print(f"  ys_eval shape: {ys_eval.shape}")
    
    with torch.no_grad():
        try:
            pred_eval = model(xs_eval, us_norm, ys_eval)
            print(f"  Evaluation forward pass: SUCCESS ✓")
            print(f"  Evaluation output shape: {pred_eval.shape}")
        except Exception as e:
            print(f"  Evaluation forward pass: FAILED ✗")
            print(f"  Error: {e}")
    
    # Check model configuration
    print(f"\nModel configuration:")
    print(f"  pos_encoding_type: {model.pos_encoding_type}")
    print(f"  use_positional_encoding: {model.use_positional_encoding}")
    print(f"  pos_encoding_dim: {model.pos_encoding_dim}")
    print(f"  pos_encoding_max_freq: {model.pos_encoding_max_freq}")
    print(f"  aggregation: {model.aggregation}")
    
    model.train()
    print("="*50)

def main():
    """Main training function."""
    # Parse arguments and setup
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = setup_logging(project_root)
    
    # Load dataset
    print(f"Loading Chladni dataset from: {args.data_path}")
    try:
        dataset = load_from_disk(args.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure the Chladni dataset has been generated.")
        print("Run: python Data/chladni_plate_generator.py")
        return
    
    # Create dataset wrapper
    chladni_dataset = ChladniDataset(dataset, batch_size=args.batch_size, device=device)
    
    # Create model
    print("Creating SetONet model for Chladni plate problem...")
    model = create_model(args, device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load pre-trained model if specified
    if args.load_model_path:
        print(f"Loading pre-trained model from: {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        print("Pre-trained model loaded successfully!")
    
    # Debug data shapes before training (helpful for troubleshooting)
    debug_data_shapes(model, dataset, chladni_dataset, device)
    
    # Train model
    print(f"\nStarting training for {args.son_epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.son_lr}")
    print(f"LR schedule steps: {args.lr_schedule_steps}")
    print(f"LR schedule gammas: {args.lr_schedule_gammas}")
    
    # Train using SetONet's built-in training loop
    model.train_model(
        dataset=chladni_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=None
    )
    
    # Evaluate model
    print("\nTraining completed! Evaluating model...")
    evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100)
    
    # Plot results for a few test samples
    print("\nGenerating result plots...")
    for i in range(3):  # Plot first 3 test samples
        plot_save_path = os.path.join(log_dir, f"chladni_results_sample_{i}.png")
        plot_results(model, dataset, chladni_dataset, device, sample_idx=i, save_path=plot_save_path)
    
    # Save model
    model_save_path = os.path.join(log_dir, "chladni_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()