import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse
from pathlib import Path

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.setonet_factory import create_setonet_model, load_pretrained_model
from Models.utils.experiment_utils import save_experiment_config
from Models.utils.helper_utils import calculate_l2_relative_error

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for Coulomb potential prediction.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="coulomb_2D_10particles_safe.npz", 
                       help='Path to Coulomb dataset (.npz file)')
    parser.add_argument('--n_particles', type=int, default=50, help='Number of particles (charges) per sample')
    parser.add_argument('--n_query_points', type=int, default=100, help='Number of query points for evaluation')
    
    # Model architecture - same as Darcy for consistency
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention"], help='Aggregation type for SetONet')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function for SetONet networks')
    
    # Training parameters - same as Darcy
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=75000, help='Number of epochs for SetONet')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Evaluation parameters
    parser.add_argument('--n_test_samples_eval', type=int, default=100, help='Number of test samples for evaluation')
    parser.add_argument('--test_split', type=float, default=0.1, help='Fraction of data to use for testing')
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet_coulomb"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def load_coulomb_dataset(data_path, test_split=0.1):
    """Load and split the Coulomb dataset."""
    try:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        
        # Load the .npz file (allow_pickle=True for dictionary objects)
        data = np.load(data_path, allow_pickle=True)
        
        # Extract all samples
        samples = []
        for key in sorted(data.keys()):
            if key.startswith('s'):  # sample keys like 's0', 's1', etc.
                sample = data[key].item()  # .item() to get the dict from numpy array
                samples.append(sample)
        
        n_samples = len(samples)
        n_test = int(n_samples * test_split)
        n_train = n_samples - n_test
        
        # Split into train/test
        train_samples = samples[:n_train]
        test_samples = samples[n_train:]
        
        print(f"Loaded Coulomb dataset from: {data_path}")
        print(f"Total samples: {n_samples}")
        print(f"Train samples: {len(train_samples)}")
        print(f"Test samples: {len(test_samples)}")
        
        # Print sample structure
        sample_0 = samples[0]
        print(f"Sample structure:")
        for key, value in sample_0.items():
            print(f"  {key}: {value.shape} {value.dtype}")
        
        return {'train': train_samples, 'test': test_samples}
        
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        raise

def setup_parameters(args):
    """Setup problem parameters."""
    return {
        'input_range': [0, 1],  # Coulomb problem domain [0,1] x [0,1]
        'scale': 1.0,  # Not directly applicable to Coulomb, but needed for compatibility
        'sensor_size': args.n_particles,  # Number of charge particles (sensors)
        'n_trunk_points_train': args.n_query_points,  # Number of query points (trunk points)
        'n_particles': args.n_particles,
        'n_query_points': args.n_query_points,
        'batch_size_train': args.batch_size,
        'n_test_samples_eval': args.n_test_samples_eval,
        'test_split': args.test_split,
        'variable_sensors': False,  # Fixed number of particles
        'sensor_seed': 42,
    }

class CoulombDataGenerator:
    """Data generator for Coulomb dataset with proper normalization."""
    
    def __init__(self, dataset_samples, device):
        print("📊 Pre-loading and optimizing Coulomb dataset...")
        
        n_samples = len(dataset_samples)
        
        # Get dimensions from first sample
        sample_0 = dataset_samples[0]
        n_particles = len(sample_0['positions'])
        n_queries = len(sample_0['queries'])
        
        print(f"Dataset info: {n_samples} samples, {n_particles} particles, {n_queries} queries per sample")
        
        # Pre-allocate tensors on GPU
        self.positions = torch.zeros(n_samples, n_particles, 2, device=device, dtype=torch.float32)
        self.charges = torch.zeros(n_samples, n_particles, device=device, dtype=torch.float32)
        self.queries = torch.zeros(n_samples, n_queries, 2, device=device, dtype=torch.float32)
        self.potentials = torch.zeros(n_samples, n_queries, device=device, dtype=torch.float32)
        
        # Load all data to GPU once
        for i, sample in enumerate(dataset_samples):
            self.positions[i] = torch.tensor(sample['positions'], device=device, dtype=torch.float32)
            self.charges[i] = torch.tensor(sample['charges'], device=device, dtype=torch.float32)
            self.queries[i] = torch.tensor(sample['queries'], device=device, dtype=torch.float32)
            self.potentials[i] = torch.tensor(sample['potentials'], device=device, dtype=torch.float32)
        
        # CRITICAL: Compute normalization statistics for potentials
        print("🔧 Computing normalization statistics...")
        self.potential_mean = self.potentials.mean()
        self.potential_std = self.potentials.std()
        
        print(f"Potential stats before normalization:")
        print(f"  Range: [{self.potentials.min():.3f}, {self.potentials.max():.3f}]")
        print(f"  Mean: {self.potential_mean:.3f}, Std: {self.potential_std:.3f}")
        
        # Normalize potentials to have mean=0, std=1
        self.potentials = (self.potentials - self.potential_mean) / self.potential_std
        
        print(f"Potential stats after normalization:")
        print(f"  Range: [{self.potentials.min():.3f}, {self.potentials.max():.3f}]")
        print(f"  Mean: {self.potentials.mean():.3f}, Std: {self.potentials.std():.3f}")
        
        self.device = device
        self.n_samples = n_samples
        self.n_particles = n_particles
        self.n_queries = n_queries
        
        print(f"✅ Coulomb dataset optimized: {n_samples} samples pre-loaded to GPU")
        
    def generate_batch(self, batch_size):
        """Generate a batch using pre-loaded GPU data."""
        # Random sampling directly on GPU
        indices = torch.randint(0, self.n_samples, (batch_size,), device=self.device)
        
        # Direct indexing into pre-loaded GPU tensors
        batch_positions = self.positions[indices]    # [batch_size, n_particles, 2]
        batch_charges = self.charges[indices]        # [batch_size, n_particles]
        batch_queries = self.queries[indices]        # [batch_size, n_queries, 2]
        batch_potentials = self.potentials[indices]  # [batch_size, n_queries]
        
        return batch_positions, batch_charges, batch_queries, batch_potentials

def train_coulomb_model(setonet_model, args, data_generator, device, log_dir):
    """Train SetONet model on Coulomb data."""
    print(f"\n--- Training Coulomb Model ---")
    print(f"Training with {data_generator.n_particles} particles and {data_generator.n_queries} query points")
    print(f"⚡ Training for {args.son_epochs} epochs, batch_size={args.batch_size}")
    
    # Initialize TensorBoard writer if log_dir is provided
    writer = None
    if log_dir:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = f"{log_dir}/tensorboard"
        writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logging to: {tensorboard_dir}")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(setonet_model.parameters(), lr=args.son_lr)
    loss_fn = torch.nn.MSELoss()
    
    from tqdm import tqdm
    
    print(f"\nTraining SetONet for Coulomb potential prediction...")
    epoch_pbar = tqdm(range(args.son_epochs), desc="Training SetONet (Coulomb)")
    
    for epoch in epoch_pbar:
        setonet_model.train()
        
        # Generate batch
        positions, charges, queries, potentials = data_generator.generate_batch(args.batch_size)
        
        # Prepare inputs for SetONet
        # xs: particle positions [batch_size, n_particles, 2]
        # us: particle charges [batch_size, n_particles, 1]
        # ys: query positions [batch_size, n_queries, 2]
        # targets: potential values [batch_size, n_queries]
        
        xs = positions  # [batch_size, n_particles, 2]
        us = charges.unsqueeze(-1)  # [batch_size, n_particles, 1]
        ys = queries    # [batch_size, n_queries, 2]
        targets = potentials  # [batch_size, n_queries]
        
        # Forward pass
        pred = setonet_model(xs, us, ys)
        pred = pred.squeeze(-1)  # [batch_size, n_queries]
        
        # Compute loss
        loss = loss_fn(pred, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        current_iteration = epoch + 1
        if current_iteration in args.lr_schedule_steps:
            milestone_idx = args.lr_schedule_steps.index(current_iteration)
            gamma = args.lr_schedule_gammas[milestone_idx]
            
            old_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
            new_lr = optimizer.param_groups[0]['lr']
            print(f"\nIteration {current_iteration}: LR decayed from {old_lr:.2e} to {new_lr:.2e} (factor {gamma}).")
        
        # Calculate L2 relative error
        with torch.no_grad():
            rel_l2_error = calculate_l2_relative_error(pred, targets)
        
        # TensorBoard logging
        if writer and epoch % 100 == 0:
            writer.add_scalar('Loss/Training', loss.item(), epoch)
            writer.add_scalar('L2_Error/Training', rel_l2_error.item(), epoch)
            writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        progress_info = {
            "Loss": f"{loss.item():.3e}",
            "L2_Error": f"{rel_l2_error.item():.3e}",
        }
        
        epoch_pbar.set_postfix(progress_info)
    
    # Close TensorBoard writer
    if writer:
        writer.close()
        print(f"\nTensorBoard logs saved. To view, run: tensorboard --logdir={tensorboard_dir}")

def evaluate_coulomb_model(setonet_model, test_data_generator, params, device):
    """Evaluate SetONet model on Coulomb test data."""
    setonet_model.eval()
    
    n_test_samples = min(test_data_generator.n_samples, params['n_test_samples_eval'])
    batch_size = params['batch_size_train']
    
    print(f"\n--- Evaluating Coulomb Model ---")
    print(f"Evaluation using {test_data_generator.n_particles} particles and {test_data_generator.n_queries} query points")
    print(f"Using normalization: mean={test_data_generator.potential_mean:.3f}, std={test_data_generator.potential_std:.3f}")
    
    total_l2_error = 0.0
    n_batches = (n_test_samples + batch_size - 1) // batch_size
    
    from tqdm import tqdm
    
    with torch.no_grad():
        samples_processed = 0
        for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_test_samples)
            current_batch_size = end_idx - start_idx
            
            # Get batch by indices
            indices = torch.arange(start_idx, end_idx, device=device)
            positions = test_data_generator.positions[indices]
            charges = test_data_generator.charges[indices]
            queries = test_data_generator.queries[indices]
            potentials_normalized = test_data_generator.potentials[indices]  # Already normalized
            
            # Prepare inputs for SetONet
            xs = positions
            us = charges.unsqueeze(-1)
            ys = queries
            targets = potentials_normalized  # Use normalized targets
            
            # Forward pass
            pred = setonet_model(xs, us, ys)
            pred = pred.squeeze(-1)
            
            # Calculate L2 relative error for this batch (on normalized values)
            batch_l2_error = calculate_l2_relative_error(pred, targets)
            total_l2_error += batch_l2_error.item() * current_batch_size
            samples_processed += current_batch_size
    
    avg_l2_error = total_l2_error / samples_processed
    print(f"Final L2 relative error (Coulomb, normalized): {avg_l2_error:.6f}")
    
    return {
        'final_l2_relative_error': avg_l2_error,
        'benchmark_task': 'coulomb',
        'n_test_samples': samples_processed,
    }

def coulomb_potential_on_grid(positions, charges, n=120, min_distance=0.02, use_soft_core=True):
    """Evaluate Coulomb potential on an n×n grid using the analytical formula with safe regularization."""
    xv = np.linspace(0.0, 1.0, n)
    yv = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xv, yv, indexing="xy")
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)     # (n², 2)
    
    # Coulomb potential with safe regularization: V(y) = (1 / 2πε₀) Σ_i q_i / sqrt(||y – x_i||² + r_c²)
    diff = grid_pts[:, None, :] - positions[None, :, :]    # shape (n², M, 2)
    r_squared = np.sum(diff**2, axis=-1)                   # squared distances
    
    if use_soft_core:
        # Soft-core potential: smoother and more stable
        r_core_squared = min_distance**2
        r_regularized = np.sqrt(r_squared + r_core_squared)
        V = np.sum(charges / r_regularized, axis=1) / (2.0 * np.pi * 1.0)  # ε₀ = 1.0
    else:
        # Hard cutoff: enforce minimum distance
        r = np.sqrt(r_squared)
        r_regularized = np.maximum(r, min_distance)
        V = np.sum(charges / r_regularized, axis=1) / (2.0 * np.pi * 1.0)  # ε₀ = 1.0
    
    return xx, yy, V.reshape(n, n)

def setonet_potential_on_grid(setonet_model, positions, charges, device, potential_mean, potential_std, n=120):
    """Evaluate SetONet prediction on an n×n grid and denormalize."""
    # Create grid points
    xv = np.linspace(0.0, 1.0, n)
    yv = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xv, yv, indexing="xy")
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)     # (n², 2)
    
    # Convert to tensors
    grid_tensor = torch.tensor(grid_pts, device=device, dtype=torch.float32).unsqueeze(0)  # [1, n², 2]
    pos_tensor = positions.unsqueeze(0) if len(positions.shape) == 2 else positions  # [1, n_particles, 2]
    charge_tensor = charges.unsqueeze(0) if len(charges.shape) == 1 else charges    # [1, n_particles]
    
    # SetONet inputs
    xs = pos_tensor                           # [1, n_particles, 2]
    us = charge_tensor.unsqueeze(-1)         # [1, n_particles, 1]
    ys = grid_tensor                         # [1, n², 2]
    
    # Predict
    setonet_model.eval()
    with torch.no_grad():
        pred = setonet_model(xs, us, ys)
        pred_normalized = pred.squeeze().cpu().numpy()  # [n²]
        
        # CRITICAL: Denormalize the predictions
        pred_potentials = pred_normalized * potential_std.cpu().numpy() + potential_mean.cpu().numpy()
    
    return xx, yy, pred_potentials.reshape(n, n)

def generate_coulomb_plots(setonet_model, dataset, test_data_generator, params, log_dir, device):
    """Generate plots for Coulomb predictions with proper field visualization."""
    print("\n--- Generating Coulomb Plots ---")
    print("Generating field visualization plots for test data")
    
    # Plot first 3 test samples
    test_samples = dataset['test'][:3]
    
    import matplotlib.pyplot as plt
    
    for i, sample in enumerate(test_samples):
        positions_np = sample['positions']  # numpy array [n_particles, 2]
        charges_np = sample['charges']      # numpy array [n_particles]
        
        # Convert to tensors for SetONet
        positions = torch.tensor(positions_np, device=device, dtype=torch.float32)
        charges = torch.tensor(charges_np, device=device, dtype=torch.float32)
        
        # Evaluate potentials on dense grids
        print(f"  Evaluating sample {i} on dense grid...")
        xx_true, yy_true, V_true = coulomb_potential_on_grid(positions_np, charges_np, n=120)
        xx_pred, yy_pred, V_pred = setonet_potential_on_grid(
            setonet_model, positions, charges, device, 
            test_data_generator.potential_mean, test_data_generator.potential_std, n=120
        )
        
        # Create plot with proper field visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Use dynamic color scaling based on actual data
        vmin_true, vmax_true = np.percentile(V_true, [5, 95])
        vmin_pred, vmax_pred = np.percentile(V_pred, [5, 95])
        vmin_common, vmax_common = min(vmin_true, vmin_pred), max(vmax_true, vmax_pred)
        
        # Ground truth field
        im1 = ax1.imshow(
            V_true,
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation="bilinear",
            vmin=vmin_common,
            vmax=vmax_common,
            cmap="RdBu_r"
        )
        # Overlay charge positions
        sizes = 8 + 4 * np.abs(charges_np)
        ax1.scatter(
            positions_np[:, 0], positions_np[:, 1],
            s=sizes, c=charges_np, cmap="RdBu_r",
            edgecolors="black", linewidths=0.5, alpha=0.8,
            vmin=-3, vmax=3
        )
        ax1.set_title(f'Ground Truth Potential Field ({len(charges_np)} charges)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1, label="V", shrink=0.8)
        
        # SetONet prediction field
        im2 = ax2.imshow(
            V_pred,
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation="bilinear",
            vmin=vmin_common,
            vmax=vmax_common,
            cmap="RdBu_r"
        )
        # Overlay charge positions
        ax2.scatter(
            positions_np[:, 0], positions_np[:, 1],
            s=sizes, c=charges_np, cmap="RdBu_r",
            edgecolors="black", linewidths=0.5, alpha=0.8,
            vmin=-3, vmax=3
        )
        ax2.set_title('SetONet Prediction')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2, label="V", shrink=0.8)
        
        # Error field
        error_field = V_pred - V_true
        error_max = np.max(np.abs(error_field))
        im3 = ax3.imshow(
            error_field,
            origin="lower",
            extent=[0, 1, 0, 1],
            interpolation="bilinear",
            vmin=-error_max,
            vmax=error_max,
            cmap="seismic"
        )
        ax3.set_title('Prediction Error')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3, label="Error", shrink=0.8)
        
        # Add charge range info
        charge_range_text = f"Charge range: [{charges_np.min():.1f}, {charges_np.max():.1f}]"
        ax1.text(0.02, 0.98, charge_range_text, 
                transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(log_dir, f"coulomb_field_sample_{i}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Field plot saved → {plot_path}")
        
        # Calculate errors
        field_rel_error = np.linalg.norm(error_field) / np.linalg.norm(V_true)
        print(f"Sample {i} field relative L2 error: {field_rel_error:.6f}")
        
        # Also evaluate on the original query points for comparison
        queries = torch.tensor(sample['queries'], device=device, dtype=torch.float32).unsqueeze(0)
        true_potentials = sample['potentials']  # Original unnormalized potentials
        
        xs = positions.unsqueeze(0)
        us = charges.unsqueeze(0).unsqueeze(-1)
        ys = queries
        
        with torch.no_grad():
            pred = setonet_model(xs, us, ys)
            pred_normalized = pred.squeeze().cpu().numpy()
            # Denormalize predictions for fair comparison
            pred_potentials = pred_normalized * test_data_generator.potential_std.cpu().numpy() + test_data_generator.potential_mean.cpu().numpy()
        
        query_rel_error = np.linalg.norm(pred_potentials - true_potentials) / np.linalg.norm(true_potentials)
        print(f"Sample {i} query points relative L2 error: {query_rel_error:.6f}")

def save_model(setonet_model, log_dir, model_was_loaded):
    """Save trained model."""
    if not model_was_loaded:
        model_path = os.path.join(log_dir, "setonet_model_coulomb.pth")
        torch.save(setonet_model.state_dict(), model_path)
        print(f"SetONet Coulomb model saved to {model_path}")
    else:
        print(f"\nSkipping model saving as pre-trained Coulomb model was loaded.")

def main():
    """Main execution function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args = parse_arguments()
    
    # Add benchmark argument for compatibility with existing infrastructure
    args.benchmark = 'coulomb'
    
    # Validate arguments
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")
    
    print(f"Training SetONet for Coulomb potential prediction")
    
    # Load dataset
    dataset = load_coulomb_dataset(args.data_path, args.test_split)
    
    log_dir = setup_logging(project_root)
    params = setup_parameters(args)
    
    # Fix random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Initialize model
    setonet_model = create_setonet_model(args, device)
    
    # Load pre-trained model if available
    model_was_loaded = load_pretrained_model(setonet_model, args, device)
    
    # Create data generators
    train_data_generator = CoulombDataGenerator(dataset['train'], device)
    test_data_generator = CoulombDataGenerator(dataset['test'], device)
    
    # Training
    if not model_was_loaded:
        train_coulomb_model(setonet_model, args, train_data_generator, device, log_dir)
    else:
        print(f"\nSetONet Coulomb model loaded. Skipping training.")
    
    # Evaluation
    eval_result = evaluate_coulomb_model(setonet_model, test_data_generator, params, device)
    
    # Save model
    save_model(setonet_model, log_dir, model_was_loaded)
    
    # Generate plots
    generate_coulomb_plots(setonet_model, dataset, test_data_generator, params, log_dir, device)
    
    # Save experiment configuration
    save_experiment_config(args, params, log_dir, device, model_was_loaded, eval_result, 'coulomb', setonet_model)
    
    print("\nScript finished.")

if __name__ == "__main__":
    main() 