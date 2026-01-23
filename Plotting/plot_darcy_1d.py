import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from datetime import datetime

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.SetONet import SetONet
from Models.DeepONet import DeepONetWrapper
import torch.nn as nn
from Data.darcy_1d_data.darcy_1d_dataset import (
    load_darcy_dataset, create_sensor_points, create_query_points, setup_parameters
)
from Data.data_utils import apply_sensor_dropoff, apply_sensor_dropoff_with_interpolation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot comparison of SetONet and DeepONet for Darcy 1D problem.")
    
    # Model paths
    parser.add_argument('--setonet_model_path', type=str, default=None, 
                       help='Path to pre-trained SetONet model')
    parser.add_argument('--deeponet_model_path', type=str, default=None,
                       help='Path to pre-trained DeepONet model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="Data/darcy_1d_data/darcy_1d_dataset_501", 
                       help='Path to Darcy 1D dataset')
    parser.add_argument('--sensor_size', type=int, default=300, help='Number of sensor locations')
    parser.add_argument('--n_query_points', type=int, default=300, help='Number of query points')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (for compatibility)')
    parser.add_argument('--train_sensor_dropoff', type=float, default=0.0, help='Training sensor dropoff (not used in plotting)')
    parser.add_argument('--n_test_samples_eval', type=int, default=1000, help='Number of test samples (not used in plotting)')
    
    # SetONet architecture (needed to reconstruct model)
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention"], 
                       help='Aggregation type for SetONet')
    parser.add_argument(
        '--son_branch_head_type',
        type=str,
        default="standard",
        choices=["standard", "petrov_attention"],
        help="Branch head type: standard (pool+rho) or petrov_attention (PG attention projection).",
    )
    parser.add_argument('--son_pg_dk', type=int, default=None, help='PG key/query dim (default: son_phi_output_size)')
    parser.add_argument('--son_pg_dv', type=int, default=None, help='PG value dim (default: son_phi_output_size)')
    parser.add_argument(
        '--son_pg_no_logw',
        action='store_true',
        help='Disable adding log(sensor_weights) to PG attention logits (weights are unused by default).',
    )
    
    # DeepONet architecture (needed to reconstruct model)
    parser.add_argument('--don_p_dim', type=int, default=32, help='Latent dimension p for DeepONet')
    parser.add_argument('--don_trunk_hidden', type=int, default=256, help='Hidden size for DeepONet trunk network')
    parser.add_argument('--don_n_trunk_layers', type=int, default=4, help='Number of layers in DeepONet trunk network')
    parser.add_argument('--don_branch_hidden', type=int, default=128, help='Hidden size for DeepONet branch network')
    parser.add_argument('--don_n_branch_layers', type=int, default=3, help='Number of layers in DeepONet branch network')
    
    # Common parameters
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], 
                       help='Activation function')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], 
                       help='Positional encoding type for SetONet')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency for positional encoding')
    
    # Learning rate parameters (needed for model creation but not used)
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate for DeepONet')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', 
                       default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="LR decay milestones")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', 
                       default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="LR decay factors")
    
    # Sensor dropout and evaluation
    parser.add_argument('--eval_sensor_dropoff', type=float, default=0.0, 
                       help='Sensor drop-off rate during evaluation (0.0-1.0)')
    parser.add_argument('--replace_with_nearest', action='store_true', 
                       help='Replace dropped sensors with nearest remaining sensors (SetONet only)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sample plots to generate')
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'test'], 
                       help='Which dataset split to use')
    
    # Misc
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use')
    parser.add_argument('--benchmark', type=str, default='darcy_1d', help='Benchmark name (for compatibility)')
    
    return parser.parse_args()


def get_activation_function(activation_name):
    """Get activation function by name."""
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'swish': nn.SiLU
    }
    return activation_map.get(activation_name.lower(), nn.ReLU)


def create_setonet_model(args, device):
    """Create SetONet model for Darcy 1D problem."""
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
        branch_head_type=args.son_branch_head_type,
        pg_dk=args.son_pg_dk,
        pg_dv=args.son_pg_dv,
        pg_use_logw=(not args.son_pg_no_logw),
    ).to(device)
    
    return model


def create_deeponet_model(args, device):
    """Create DeepONet model for Darcy 1D problem."""
    activation_fn = get_activation_function(args.activation_fn)
    
    model = DeepONetWrapper(
        branch_input_dim=args.sensor_size,
        p=args.don_p_dim,
        trunk_hidden_size=args.don_trunk_hidden,
        n_trunk_layers=args.don_n_trunk_layers,
        branch_hidden_size=args.don_branch_hidden,
        n_branch_layers=args.don_n_branch_layers,
        activation_fn=activation_fn,
        initial_lr=args.don_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
    ).to(device)
    
    return model


def plot_dual_darcy_comparison(
    setonet_model,
    deeponet_model,
    dataset,
    sensor_x,
    query_x,
    sensor_indices,
    query_indices,
    save_dir,
    sample_idx,
    dataset_split="test",
    sensor_dropoff=0.0,
    replace_with_nearest=False,
    grid_points=None
):
    """
    Plots comparison between true solution and predictions from both SetONet and DeepONet.
    Style follows plot_1d.py: clean, no titles, no sensor locations shown.
    """
    if setonet_model is not None:
        setonet_model.eval()
        device = next(setonet_model.parameters()).device
    elif deeponet_model is not None:
        deeponet_model.eval()
        device = next(deeponet_model.parameters()).device
    else:
        raise ValueError("At least one model must be provided")
    
    # Set larger font sizes for better readability (matching plot_1d.py style)
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'axes.titlesize': 18
    })
    
    # Use constrained layout and make each subplot square
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, squeeze=False)
    for ax in axs.ravel():
        ax.set_box_aspect(1)  # enforce square axes
        ax.margins(x=0.01, y=0.02)  # reduce internal padding around data
    
    # Get sample data
    data_split = dataset[dataset_split]
    sample = data_split[sample_idx]
    
    # Extract source term and solution
    u_full = torch.tensor(sample['u'], device=device, dtype=torch.float32)  # Source term
    s_full = torch.tensor(sample['s'], device=device, dtype=torch.float32)  # Solution
    
    # Get sensor and query data
    u_sensors = u_full[sensor_indices]  # Source at sensor locations
    s_queries_true = s_full[query_indices]  # True solution at query points
    
    # Convert grid points to CPU for plotting
    if grid_points is not None:
        x_plot = grid_points.cpu().numpy()
    else:
        x_plot = np.linspace(0, 1, len(u_full))
    
    # Apply sensor dropout if specified
    if sensor_dropoff > 0.0:
        # For SetONet: use apply_sensor_dropoff with optional nearest neighbor replacement
        sensor_x_dropped_son, u_sensors_dropped_son = apply_sensor_dropoff(
            sensor_x, u_sensors, sensor_dropoff, replace_with_nearest
        )
        
        # For DeepONet: use interpolation (never uses nearest neighbor)
        _, u_sensors_interpolated_don = apply_sensor_dropoff_with_interpolation(
            sensor_x, u_sensors, sensor_dropoff
        )
        
        # SetONet inputs
        sensor_x_model_son = sensor_x_dropped_son.unsqueeze(0)
        u_sensors_model_son = u_sensors_dropped_son.unsqueeze(0).unsqueeze(-1)
        
        # DeepONet inputs (always uses original sensor locations with interpolated values)
        sensor_x_model_don = sensor_x.unsqueeze(0)
        u_sensors_model_don = u_sensors_interpolated_don.unsqueeze(0).unsqueeze(-1)
    else:
        # No dropout
        sensor_x_model_son = sensor_x.unsqueeze(0)
        u_sensors_model_son = u_sensors.unsqueeze(0).unsqueeze(-1)
        sensor_x_model_don = sensor_x.unsqueeze(0)
        u_sensors_model_don = u_sensors.unsqueeze(0).unsqueeze(-1)
    
    # Left subplot: Source term (no sensor locations shown, following plot_1d.py style)
    axs[0, 0].plot(x_plot, u_full.cpu().numpy(), 'darkorange', linewidth=2, label='Source term')
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel(r'$f(x)$')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    # Get predictions from both models
    with torch.no_grad():
        query_x_model = query_x.unsqueeze(0)
        
        # SetONet prediction (if model provided)
        s_pred_son = None
        if setonet_model is not None:
            s_pred_son = setonet_model(sensor_x_model_son, u_sensors_model_son, query_x_model)
            s_pred_son = s_pred_son.squeeze().cpu().numpy()
        
        # DeepONet prediction (if model provided)
        s_pred_don = None
        if deeponet_model is not None:
            s_pred_don = deeponet_model(sensor_x_model_don, u_sensors_model_don, query_x_model)
            s_pred_don = s_pred_don.squeeze().cpu().numpy()
    
    # Right subplot: Solution comparison with both models
    query_x_plot = query_x.squeeze().cpu().numpy()
    s_true_plot = s_queries_true.cpu().numpy()
    
    # Plot with different markers to distinguish overlapping predictions
    axs[0, 1].plot(query_x_plot, s_true_plot, 'g-', linewidth=2, label='True')
    if s_pred_son is not None:
        axs[0, 1].plot(query_x_plot, s_pred_son, 'ro', markersize=4, markevery=10, label='SetONet')
    if s_pred_don is not None:
        axs[0, 1].plot(query_x_plot, s_pred_don, 'bs', markersize=4, markevery=(5, 10), label='DeepONet')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel(r'$u(x)$')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    # Tighten spacing
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.005, hspace=0.01)
    
    # Save plot
    replacement_suffix = "_nearest" if replace_with_nearest and sensor_dropoff > 0 else ""
    dropoff_suffix = f"_dropoff_{sensor_dropoff:.1f}{replacement_suffix}" if sensor_dropoff > 0 else ""
    save_path = os.path.join(save_dir, f"darcy_comparison_sample_{sample_idx}{dropoff_suffix}.png")
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved Darcy comparison plot {sample_idx} to {save_path}")
    plt.close(fig)
    
    # Reset font parameters to default
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'axes.titlesize': 12
    })


def main():
    """Main function to load models and generate comparison plots."""
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # For better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join("results", f"darcy_comparison_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Load dataset
    dataset = load_darcy_dataset(args.data_path)
    grid_points = torch.tensor(dataset['train'][0]['X'], dtype=torch.float32)
    
    # Setup parameters
    args.benchmark = 'darcy_1d'  # Add for compatibility
    params = setup_parameters(args)
    
    # Create sensor and query points
    sensor_x, sensor_indices = create_sensor_points(params, device, grid_points)
    query_x, query_indices = create_query_points(params, device, grid_points, args.n_query_points)
    
    print(f"Sensor points: {len(sensor_x)}, Query points: {len(query_x)}")
    
    # Check that at least one model path is provided
    if args.setonet_model_path is None and args.deeponet_model_path is None:
        raise ValueError("At least one model path must be provided (--setonet_model_path or --deeponet_model_path)")
    
    # Create and load SetONet model if path provided
    setonet_model = None
    if args.setonet_model_path:
        print("\nCreating and loading SetONet model...")
        setonet_model = create_setonet_model(args, device)
        if os.path.exists(args.setonet_model_path):
            setonet_model.load_state_dict(torch.load(args.setonet_model_path, map_location=device))
            print(f"Loaded SetONet from: {args.setonet_model_path}")
        else:
            raise ValueError(f"SetONet model not found: {args.setonet_model_path}")
    
    # Create and load DeepONet model if path provided
    deeponet_model = None
    if args.deeponet_model_path:
        print("Creating and loading DeepONet model...")
        deeponet_model = create_deeponet_model(args, device)
        if os.path.exists(args.deeponet_model_path):
            deeponet_model.load_state_dict(torch.load(args.deeponet_model_path, map_location=device))
            print(f"Loaded DeepONet from: {args.deeponet_model_path}")
        else:
            raise ValueError(f"DeepONet model not found: {args.deeponet_model_path}")
    
    # Generate plots
    print(f"\nGenerating {args.num_samples} comparison plots...")
    
    data_split = dataset[args.dataset_split]
    for i in range(min(args.num_samples, len(data_split))):
        plot_dual_darcy_comparison(
            setonet_model=setonet_model,
            deeponet_model=deeponet_model,
            dataset=dataset,
            sensor_x=sensor_x,
            query_x=query_x,
            sensor_indices=sensor_indices,
            query_indices=query_indices,
            save_dir=results_dir,
            sample_idx=i,
            dataset_split=args.dataset_split,
            sensor_dropoff=args.eval_sensor_dropoff,
            replace_with_nearest=args.replace_with_nearest,
            grid_points=grid_points
        )
    
    # Save configuration info
    config_path = os.path.join(results_dir, "plot_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"SetONet model: {args.setonet_model_path}\n")
        f.write(f"DeepONet model: {args.deeponet_model_path}\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Dataset split: {args.dataset_split}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Sensor dropoff: {args.eval_sensor_dropoff}\n")
        f.write(f"Replace with nearest: {args.replace_with_nearest}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    print(f"\nPlotting completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
