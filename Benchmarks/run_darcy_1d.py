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

from Models.setonet_factory import create_setonet_model, load_pretrained_model
from Models.utils.experiment_utils import save_experiment_config
from Models.utils.helper_utils import calculate_l2_relative_error

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for Darcy 1D equation.")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="/home/titanv/Stepan/setprojects/SetONet/Data/darcy_1d_dataset", 
                       help='Path to Darcy 1D dataset')
    parser.add_argument('--sensor_size', type=int, default=50, help='Number of sensor locations (max 101 for Darcy 1D grid)')
    
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
    parser.add_argument('--son_epochs', type=int, default=50000, help='Number of epochs for SetONet')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Data generation
    parser.add_argument('--variable_sensors', action='store_true', help='Use different random sensor locations for each sample (more challenging)')
    
    # Evaluation robustness testing (sensor failures)
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate during evaluation only (0.0-1.0). Simulates sensor failures during testing')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors instead of removing them (leverages permutation invariance)')
    
    # Evaluation parameters
    parser.add_argument('--n_test_samples_eval', type=int, default=100, help='Number of test samples for evaluation (max 100 for Darcy 1D test set)')
    parser.add_argument('--n_query_points', type=int, default=101, help='Number of query points for evaluation (max 101 for Darcy 1D grid)')
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet_darcy_1d"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def load_darcy_dataset(data_path):
    """Load and return the Darcy 1D dataset."""
    try:
        dataset = load_from_disk(data_path)
        print(f"Loaded Darcy 1D dataset from: {data_path}")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        raise

def setup_parameters(args):
    """Setup problem parameters."""
    return {
        'input_range': [0, 1],  # Darcy problem is on [0,1] (keep domain-specific)
        'scale': 0.1,  # Same as run_1d.py
        'sensor_size': args.sensor_size,
        'batch_size_train': args.batch_size,
        'n_trunk_points_train': args.n_query_points,  # Query points are trunk points for Darcy
        'n_test_samples_eval': args.n_test_samples_eval,
        'sensor_seed': 42,
        'variable_sensors': args.variable_sensors,
        'eval_sensor_dropoff': args.eval_sensor_dropoff,
        'replace_with_nearest': args.replace_with_nearest,
    }

def create_sensor_points(params, device, grid_points):
    """Create fixed sensor points from the grid."""
    # Use fixed sensor locations - evenly spaced subset of grid points
    sensor_indices = torch.linspace(0, len(grid_points)-1, params['sensor_size'], dtype=torch.long)
    sensor_x = grid_points[sensor_indices].to(device).view(-1, 1)
    print(f"Using {params['sensor_size']} fixed sensor locations (evenly spaced)")
    return sensor_x, sensor_indices

def create_query_points(params, device, grid_points, n_query_points):
    """Create query points for evaluation."""
    # Use evenly spaced query points
    query_indices = torch.linspace(0, len(grid_points)-1, n_query_points, dtype=torch.long)
    query_x = grid_points[query_indices].to(device).view(-1, 1)
    return query_x, query_indices

class DarcyDataGenerator:
    """OPTIMIZED Data generator for Darcy 1D dataset."""
    
    def __init__(self, dataset, sensor_indices, query_indices, device):
        print("📊 Pre-loading and optimizing dataset...")
        
        # PRE-LOAD all data to GPU for maximum efficiency
        train_data = dataset['train']
        n_train = len(train_data)
        n_grid = len(train_data[0]['u'])
        
        # Pre-allocate tensors on GPU
        self.u_data = torch.zeros(n_train, n_grid, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(n_train, n_grid, device=device, dtype=torch.float32)
        
        # Load all data to GPU once (much faster than per-batch loading)
        for i in range(n_train):
            self.u_data[i] = torch.tensor(train_data[i]['u'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(train_data[i]['s'], device=device, dtype=torch.float32)
        
        self.sensor_indices = sensor_indices.to(device)
        self.query_indices = query_indices.to(device)
        self.device = device
        self.n_train = n_train
        
        # Pre-extract sensor and query data for even faster access
        self.u_sensors = self.u_data[:, self.sensor_indices]  # [n_train, n_sensors]
        self.s_queries = self.s_data[:, self.query_indices]   # [n_train, n_queries]
        
        print(f"✅ Dataset optimized: {n_train} samples pre-loaded to GPU")
        
    def generate_batch(self, batch_size):
        """OPTIMIZED: Generate a batch using pre-loaded GPU data."""
        # Random sampling directly on GPU (much faster)
        indices = torch.randint(0, self.n_train, (batch_size,), device=self.device)
        
        # Direct indexing into pre-loaded GPU tensors (extremely fast)
        u_at_sensors = self.u_sensors[indices]  # [batch_size, n_sensors]
        s_at_queries = self.s_queries[indices]  # [batch_size, n_queries]
        
        return u_at_sensors, s_at_queries

def train_darcy_model(setonet_model, args, data_generator, sensor_x, query_x, device, log_dir):
    """Train SetONet model on Darcy data with OPTIMIZED data loading."""
    print(f"\n--- Training Darcy 1D Model ---")
    print(f"Training with {len(sensor_x)} FIXED sensor locations and {len(query_x)} query points")
    print(f"⚡ EFFICIENCY: Pre-loaded GPU data for faster training, {args.son_epochs} epochs, batch_size={args.batch_size}")
    
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
    
    print(f"\nTraining SetONet for Darcy 1D equation...")
    epoch_pbar = tqdm(range(args.son_epochs), desc="Training SetONet (Darcy 1D)")
    
    for epoch in epoch_pbar:
        setonet_model.train()
        
        # Generate batch
        u_at_sensors, s_at_queries = data_generator.generate_batch(args.batch_size)
        
        # Prepare inputs for SetONet
        batch_size = u_at_sensors.shape[0]
        n_sensors = sensor_x.shape[0]
        n_queries = query_x.shape[0]
        
        # Expand sensor and query locations for batch
        xs = sensor_x.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_sensors, 1]
        ys = query_x.unsqueeze(0).expand(batch_size, -1, -1)   # [batch_size, n_queries, 1]
        us = u_at_sensors.unsqueeze(-1)  # [batch_size, n_sensors, 1]
        
        # Forward pass
        pred = setonet_model(xs, us, ys)
        pred = pred.squeeze(-1)  # [batch_size, n_queries]
        
        # Compute loss
        loss = loss_fn(pred, s_at_queries)
        
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
        
        # Calculate L2 relative error using standard function
        with torch.no_grad():
            rel_l2_error = calculate_l2_relative_error(pred, s_at_queries)
        
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

def evaluate_darcy_model(setonet_model, dataset, sensor_x, query_x, sensor_indices, query_indices, params, device):
    """Evaluate SetONet model on Darcy test data with OPTIMIZED data loading."""
    setonet_model.eval()
    
    test_data = dataset['test']
    n_test_samples = min(len(test_data), params['n_test_samples_eval'])
    batch_size = params['batch_size_train']
    
    print(f"\n--- Evaluating Darcy 1D Model ---")
    print(f"Evaluation using SAME {len(sensor_x)} sensor locations and {len(query_x)} query points as training")
    # Verify sensor consistency
    assert len(sensor_indices) == params['sensor_size'], f"Sensor count mismatch: {len(sensor_indices)} vs {params['sensor_size']}"
    print(f"✓ Sensor configuration verified: {len(sensor_indices)} sensors at same locations as training")
    
    # OPTIMIZATION: Pre-load all test data to GPU
    print("📊 Pre-loading test data to GPU...")
    n_grid = len(test_data[0]['u'])
    test_u_data = torch.zeros(n_test_samples, n_grid, device=device, dtype=torch.float32)
    test_s_data = torch.zeros(n_test_samples, n_grid, device=device, dtype=torch.float32)
    
    for i in range(n_test_samples):
        test_u_data[i] = torch.tensor(test_data[i]['u'], device=device, dtype=torch.float32)
        test_s_data[i] = torch.tensor(test_data[i]['s'], device=device, dtype=torch.float32)
    
    # Pre-extract sensor and query data
    sensor_indices_gpu = sensor_indices.to(device)
    query_indices_gpu = query_indices.to(device)
    test_u_sensors = test_u_data[:, sensor_indices_gpu]  # [n_test, n_sensors]
    test_s_queries = test_s_data[:, query_indices_gpu]   # [n_test, n_queries]
    
    total_l2_error = 0.0
    n_batches = (n_test_samples + batch_size - 1) // batch_size
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_test_samples)
            current_batch_size = end_idx - start_idx
            
            # OPTIMIZED: Direct tensor slicing (much faster)
            u_at_sensors = test_u_sensors[start_idx:end_idx]  # [current_batch_size, n_sensors]
            s_at_queries = test_s_queries[start_idx:end_idx]  # [current_batch_size, n_queries]
            
            # Prepare inputs for SetONet
            xs = sensor_x.unsqueeze(0).expand(current_batch_size, -1, -1)
            ys = query_x.unsqueeze(0).expand(current_batch_size, -1, -1)
            us = u_at_sensors.unsqueeze(-1)
            
            # Forward pass
            pred = setonet_model(xs, us, ys)
            pred = pred.squeeze(-1)
            
            # Calculate L2 relative error for this batch using standard function
            batch_l2_error = calculate_l2_relative_error(pred, s_at_queries)
            total_l2_error += batch_l2_error.item() * current_batch_size
    
    avg_l2_error = total_l2_error / n_test_samples
    print(f"Final L2 relative error (Darcy 1D): {avg_l2_error:.6f}")
    
    return {
        'final_l2_relative_error': avg_l2_error,
        'benchmark_task': 'darcy_1d',
        'n_test_samples': n_test_samples,
    }

def generate_darcy_plots(setonet_model, dataset, sensor_x, query_x, sensor_indices, query_indices, log_dir, device):
    """Generate plots for Darcy 1D predictions."""
    print("\n--- Generating Darcy 1D Plots ---")
    print(f"Plotting using SAME {len(sensor_x)} sensor locations as training/testing")
    
    import matplotlib.pyplot as plt
    
    setonet_model.eval()
    test_data = dataset['test']
    
    # Plot first 3 test samples
    n_plot_samples = min(3, len(test_data))
    
    fig, axes = plt.subplots(n_plot_samples, 3, figsize=(15, 5*n_plot_samples))
    if n_plot_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(n_plot_samples):
            sample = test_data[i]
            x_grid = torch.tensor(sample['X'], dtype=torch.float32)
            u_full = torch.tensor(sample['u'], dtype=torch.float32).to(device)
            s_full = torch.tensor(sample['s'], dtype=torch.float32).to(device)
            
            # Get sensor and query values
            u_at_sensors = u_full[sensor_indices]
            s_at_queries_true = s_full[query_indices]
            
            # Prepare inputs for SetONet
            xs = sensor_x.unsqueeze(0)  # [1, n_sensors, 1]
            ys = query_x.unsqueeze(0)   # [1, n_queries, 1]
            us = u_at_sensors.unsqueeze(0).unsqueeze(-1)  # [1, n_sensors, 1]
            
            # Predict
            pred = setonet_model(xs, us, ys)
            s_at_queries_pred = pred.squeeze()
            
            # Plot source term
            axes[i, 0].plot(x_grid.cpu(), u_full.cpu(), 'b-', label='Source term u(x)', linewidth=2)
            axes[i, 0].scatter(sensor_x.cpu().squeeze(), u_at_sensors.cpu(), c='red', s=30, label='Sensor locations', zorder=5)
            axes[i, 0].set_xlabel('x')
            axes[i, 0].set_ylabel('u(x)')
            axes[i, 0].set_title(f'Sample {i+1}: Source Term')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            
            # Plot true solution
            axes[i, 1].plot(x_grid.cpu(), s_full.cpu(), 'g-', label='True solution s(x)', linewidth=2)
            axes[i, 1].scatter(query_x.cpu().squeeze(), s_at_queries_true.cpu(), c='red', s=30, label='Query locations', zorder=5)
            axes[i, 1].set_xlabel('x')
            axes[i, 1].set_ylabel('s(x)')
            axes[i, 1].set_title(f'Sample {i+1}: True Solution')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
            
            # Plot comparison
            axes[i, 2].plot(query_x.cpu().squeeze(), s_at_queries_true.cpu(), 'g-', label='True', linewidth=2, marker='o', markersize=4)
            axes[i, 2].plot(query_x.cpu().squeeze(), s_at_queries_pred.cpu(), 'r--', label='Predicted', linewidth=2, marker='s', markersize=4)
            
            # Calculate error for this sample using standard approach (with epsilon for stability)
            error = torch.norm(s_at_queries_pred - s_at_queries_true) / (torch.norm(s_at_queries_true) + 1e-8)
            axes[i, 2].set_xlabel('x')
            axes[i, 2].set_ylabel('s(x)')
            axes[i, 2].set_title(f'Sample {i+1}: Comparison (L2 error: {error:.4f})')
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'darcy_1d_predictions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Darcy 1D plots saved to: {plot_path}")

def save_model(setonet_model, log_dir, model_was_loaded):
    """Save trained model."""
    if not model_was_loaded:
        model_path = os.path.join(log_dir, "setonet_model_darcy_1d.pth")
        torch.save(setonet_model.state_dict(), model_path)
        print(f"SetONet Darcy 1D model saved to {model_path}")
    else:
        print(f"\nSkipping model saving as pre-trained Darcy 1D model was loaded.")

def main():
    """Main execution function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args = parse_arguments()
    
    # Add benchmark argument for compatibility with existing infrastructure
    args.benchmark = 'darcy_1d'
    
    # Validate arguments
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")
    
    if not 0.0 <= args.eval_sensor_dropoff <= 1.0:
        raise ValueError("--eval_sensor_dropoff must be between 0.0 and 1.0")
    
    print(f"Training SetONet for Darcy 1D equation")
    if args.eval_sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if args.replace_with_nearest else "removal"
        print(f"Will test robustness with sensor drop-off rate: {args.eval_sensor_dropoff:.1%} using {replacement_mode}")
        print("(Training will use full sensor data)")
    
    # Load dataset
    dataset = load_darcy_dataset(args.data_path)
    
    # Get grid points from first sample
    sample_0 = dataset['train'][0]
    grid_points = torch.tensor(sample_0['X'], dtype=torch.float32)
    n_grid_points = len(grid_points)
    print(f"Grid points: {n_grid_points}")
    
    # Validate sensor and query point counts
    if args.sensor_size > n_grid_points:
        raise ValueError(f"sensor_size ({args.sensor_size}) cannot exceed grid points ({n_grid_points})")
    if args.n_query_points > n_grid_points:
        raise ValueError(f"n_query_points ({args.n_query_points}) cannot exceed grid points ({n_grid_points})")
    
    log_dir = setup_logging(project_root)
    params = setup_parameters(args)
    
    # Fix random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create sensor and query points (SAME for both training and testing)
    sensor_x, sensor_indices = create_sensor_points(params, device, grid_points)
    query_x, query_indices = create_query_points(params, device, grid_points, args.n_query_points)
    
    print(f"Sensor points: {len(sensor_x)} (FIXED - same for train & test)")
    print(f"Query points: {len(query_x)} (FIXED - same for train & test)")
    print(f"Sensor indices: {sensor_indices[:5]}... to {sensor_indices[-5:]}")
    print(f"Query indices: {query_indices[:5]}... to {query_indices[-5:]}")
    
    # Initialize model
    setonet_model = create_setonet_model(args, device)
    
    # Load pre-trained model if available
    model_was_loaded = load_pretrained_model(setonet_model, args, device)
    
    # Create data generator (uses SAME sensor_indices for training)
    data_generator = DarcyDataGenerator(dataset, sensor_indices, query_indices, device)
    
    # Training (uses SAME sensor_x and sensor_indices as testing)
    if not model_was_loaded:
        train_darcy_model(setonet_model, args, data_generator, sensor_x, query_x, device, log_dir)
    else:
        print(f"\nSetONet Darcy 1D model loaded. Skipping training.")
    
    # Evaluation (uses IDENTICAL sensor_x, query_x, sensor_indices, query_indices as training)
    eval_result = evaluate_darcy_model(setonet_model, dataset, sensor_x, query_x, 
                                     sensor_indices, query_indices, params, device)
    
    # Save model
    save_model(setonet_model, log_dir, model_was_loaded)
    
    # Generate plots
    generate_darcy_plots(setonet_model, dataset, sensor_x, query_x, 
                        sensor_indices, query_indices, log_dir, device)
    
    # Save experiment configuration
    save_experiment_config(args, params, log_dir, device, model_was_loaded, eval_result, 'darcy_1d', setonet_model)
    
    print("\nScript finished.")

if __name__ == "__main__":
    main() 