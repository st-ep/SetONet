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

from Models.utils.helper_utils import prepare_setonet_inputs
from Data.synthetic_1d_data import (
    Synthetic1DDataGenerator, 
    create_synthetic_setonet_model, 
    load_synthetic_pretrained_model,
    create_synthetic_deeponet_model,
    load_synthetic_pretrained_deeponet_model
)
from Data.data_utils import apply_sensor_dropoff


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot comparison of SetONet and DeepONet for synthetic 1D benchmarks.")
    
    # Model paths
    parser.add_argument('--setonet_model_path', type=str, default=None, 
                       help='Path to pre-trained SetONet model')
    parser.add_argument('--deeponet_model_path', type=str, default=None,
                       help='Path to pre-trained DeepONet model')
    
    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True, choices=['integral', 'derivative'], 
                       help='Benchmark task: integral (f\' -> f) or derivative (f -> f\')')
    
    # SetONet architecture (needed to reconstruct model)
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention", "sum"], help='Aggregation type for SetONet')
    
    # DeepONet architecture (needed to reconstruct model)
    parser.add_argument('--don_p_dim', type=int, default=32, help='Latent dimension p for DeepONet')
    parser.add_argument('--don_trunk_hidden', type=int, default=256, help='Hidden size for DeepONet trunk network')
    parser.add_argument('--don_n_trunk_layers', type=int, default=4, help='Number of layers in DeepONet trunk network')
    parser.add_argument('--don_branch_hidden', type=int, default=128, help='Hidden size for DeepONet branch network')
    parser.add_argument('--don_n_branch_layers', type=int, default=3, help='Number of layers in DeepONet branch network')
    
    # Common parameters
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type')
    parser.add_argument('--pos_encoding_dim', type=int, default=64, help='Dimension for sinusoidal positional encoding')
    parser.add_argument('--pos_encoding_max_freq', type=float, default=0.1, help='Max frequency/scale for sinusoidal encoding')
    
    # Learning rate parameters (needed for model creation but not used for training)
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet (not used, just for model creation)')
    parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate for DeepONet (not used, just for model creation)')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="LR decay milestones")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="LR decay factors")
    
    # Data parameters
    parser.add_argument('--variable_sensors', action='store_true', help='Use different random sensor locations for each sample')
    parser.add_argument('--sensor_dropoff', type=float, default=0.0, help='Sensor drop-off rate (0.0-1.0)')
    parser.add_argument('--replace_with_nearest', action='store_true', help='Replace dropped sensors with nearest remaining sensors')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of sample plots to generate')

    
    # Misc
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use')
    
    return parser.parse_args()


def plot_dual_model_comparison(
    setonet_model,
    deeponet_model,
    branch_input_locations,
    trunk_query_locations,
    input_range,
    scale,
    save_dir,
    sample_idx,
    benchmark_type,
    is_inverse_task=False,
    use_zero_constant=True,
    sensor_dropoff=0.0,
    replace_with_nearest=False
):
    """
    Plots comparison between true function and predictions from both SetONet and DeepONet.
    
    This is a modified version of plot_operator_comparison that shows both model predictions.
    """
    print(f"Generating plot {sample_idx} with {len(branch_input_locations)} branch points and {len(trunk_query_locations)} trunk points...")
    
    if sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if replace_with_nearest else "removal"
        print(f"🔧 Applying sensor drop-off rate: {sensor_dropoff:.1%} with {replacement_mode}")
    
    if setonet_model is not None:
        setonet_model.eval()
        device = next(setonet_model.parameters()).device
    elif deeponet_model is not None:
        deeponet_model.eval()
        device = next(deeponet_model.parameters()).device
    else:
        raise ValueError("At least one model must be provided")
    
    # Ensure locations are on CPU for numpy ops and plotting
    branch_input_locs_cpu = branch_input_locations.cpu()
    trunk_query_locs_cpu = trunk_query_locations.cpu().squeeze()
    
    # Set larger font sizes for better readability (matching original design)
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
    
    # Generate coefficients using the SAME method as training data
    # Set seed for this specific sample to ensure reproducibility
    torch.manual_seed(123 + sample_idx)  # Use a fixed offset to get consistent samples
    np.random.seed(123 + sample_idx)
    
    a_coeff = (torch.rand(1).item() * 2 - 1) * scale
    b_coeff = (torch.rand(1).item() * 2 - 1) * scale
    c_coeff = (torch.rand(1).item() * 2 - 1) * scale
    if use_zero_constant:
        d_coeff = 0.0
    else:
        d_coeff = (torch.rand(1).item() * 2 - 1) * scale
    
    # True function f(x) and its derivative f'(x)
    def f_true(x_coords):
        return a_coeff * x_coords**3 + b_coeff * x_coords**2 + c_coeff * x_coords + d_coeff
    def df_true(x_coords):
        return 3*a_coeff*x_coords**2 + 2*b_coeff*x_coords + c_coeff
    
    # Determine what the branch sees and what the true output is based on task
    if not is_inverse_task:  # Forward task: f -> f' (derivative benchmark)
        branch_values_true = f_true(branch_input_locs_cpu)
        operator_output_true = df_true(trunk_query_locs_cpu)
        plot_ylabel_left = r'$f(x)$'
        plot_ylabel_right = r'$f^{\prime}(x)$'
    else:  # Inverse task: f' -> f (integral benchmark)
        branch_values_true = df_true(branch_input_locs_cpu)
        operator_output_true = f_true(trunk_query_locs_cpu)
        plot_ylabel_left = r'$f^{\prime}(x)$'
        plot_ylabel_right = r'$f(x)$'
    
    # Convert to numpy for plotting
    branch_values_true_np = branch_values_true.numpy() if hasattr(branch_values_true, 'numpy') else branch_values_true
    operator_output_true_np = operator_output_true.numpy() if hasattr(operator_output_true, 'numpy') else operator_output_true
    
    # Apply sensor drop-off if specified
    if sensor_dropoff > 0.0:
        # Convert to torch tensors for drop-off function
        branch_locs_torch = branch_input_locations.clone().to(device)
        branch_values_torch = torch.tensor(branch_values_true_np, device=device, dtype=torch.float32).squeeze()
        
        # Apply drop-off
        branch_locs_dropped, branch_values_dropped = apply_sensor_dropoff(
            branch_locs_torch, branch_values_torch, sensor_dropoff, replace_with_nearest
        )
        
        # Convert back to CPU for plotting
        branch_input_locs_plot = branch_locs_dropped.cpu()
        branch_values_plot = branch_values_dropped.cpu().numpy()
        
        # For model input, keep on device
        branch_input_locs_model = branch_locs_dropped
        branch_values_model = branch_values_dropped
        actual_n_sensors = len(branch_input_locs_plot)
    else:
        # No drop-off
        branch_input_locs_plot = branch_input_locs_cpu
        branch_values_plot = branch_values_true_np
        branch_input_locs_model = branch_input_locations
        branch_values_model = torch.tensor(branch_values_true_np, device=device, dtype=torch.float32).squeeze()
        actual_n_sensors = len(branch_input_locs_cpu)
    
    # Plot the input function (left subplot) - EXACTLY THE SAME AS ORIGINAL
    axs[0, 0].plot(branch_input_locs_plot.squeeze(), branch_values_plot.squeeze(), 'darkorange', linewidth=2, label='Input Function')
    
    axs[0, 0].set_xlabel('$x$')
    axs[0, 0].set_ylabel(plot_ylabel_left)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()
    
    # Get predictions from BOTH models
    with torch.no_grad():
        trunk_query_locs_model_dev = trunk_query_locations.clone().to(device)
        
        # SetONet prediction (if model provided)
        setonet_pred = None
        if setonet_model is not None:
            xs, us, ys = prepare_setonet_inputs(
                branch_input_locs_model,
                1,  # batch_size = 1 for single sample
                branch_values_model.unsqueeze(-1),  # Add feature dimension
                trunk_query_locs_model_dev,
                actual_n_sensors
            )
            setonet_pred = setonet_model(xs, us, ys).squeeze().cpu().numpy()
        
        # DeepONet prediction (if model provided)
        deeponet_pred = None
        if deeponet_model is not None:
            xs_don = branch_input_locs_model.unsqueeze(0)  # [1, actual_n_sensors, 1]
            us_don = branch_values_model.unsqueeze(0).unsqueeze(-1)  # [1, actual_n_sensors, 1]
            ys_don = trunk_query_locs_model_dev.unsqueeze(0)  # [1, n_trunk, 1]
            deeponet_pred = deeponet_model(xs_don, us_don, ys_don).squeeze().cpu().numpy()
    
    # Plot the output comparison (right subplot) - WITH AVAILABLE MODELS
    # Use different markers to distinguish overlapping accurate predictions
    axs[0, 1].plot(trunk_query_locs_cpu, operator_output_true_np.squeeze(), 'g-', linewidth=2, label='True')
    if setonet_pred is not None:
        axs[0, 1].plot(trunk_query_locs_cpu, setonet_pred, 'ro', markersize=4, markevery=6, label='SetONet')
    if deeponet_pred is not None:
        axs[0, 1].plot(trunk_query_locs_cpu, deeponet_pred, 'bs', markersize=4, markevery=(3, 6), label='DeepONet')
    axs[0, 1].set_xlabel('$x$')
    axs[0, 1].set_ylabel(plot_ylabel_right)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()
    
    # Tighten spacing specifically
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, wspace=0.005, hspace=0.01)
    
    # Save plot
    replacement_suffix = "_nearest" if replace_with_nearest and sensor_dropoff > 0 else ""
    dropoff_suffix = f"_dropoff_{sensor_dropoff:.1f}{replacement_suffix}" if sensor_dropoff > 0 else ""
    save_path = os.path.join(save_dir, f"{benchmark_type}_comparison_sample_{sample_idx}{dropoff_suffix}.png")
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved {benchmark_type} comparison plot {sample_idx} to {save_path}")
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
    results_dir = os.path.join("results", f"{args.benchmark}_comparison_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Setup parameters (matching the training scripts)
    params = {
        'input_range': [-1, 1],
        'scale': 0.1,
        'sensor_size': 100,
        'sensor_seed': args.seed,
        'variable_sensors': args.variable_sensors,
    }
    
    # Create sensor points
    if params.get('variable_sensors', False):
        print(f"Using variable random sensor locations")
        sensor_x_original = None
    else:
        torch.manual_seed(params['sensor_seed'])
        sensor_x = torch.rand(params['sensor_size'], device=device) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
        sensor_x = sensor_x.sort()[0]
        sensor_x_original = sensor_x.view(-1, 1)
        print(f"Using {params['sensor_size']} fixed random sensor locations")
    
    # Check that at least one model path is provided
    if args.setonet_model_path is None and args.deeponet_model_path is None:
        raise ValueError("At least one model path must be provided (--setonet_model_path or --deeponet_model_path)")
    
    # Create and load SetONet model if path provided
    setonet_model = None
    if args.setonet_model_path:
        print("\nCreating SetONet model...")
        setonet_model = create_synthetic_setonet_model(args, device)
        print(f"Loading SetONet from: {args.setonet_model_path}")
        args.load_model_path = args.setonet_model_path
        if not load_synthetic_pretrained_model(setonet_model, args, device):
            raise ValueError(f"Failed to load SetONet model from {args.setonet_model_path}")
    
    # Create and load DeepONet model if path provided
    deeponet_model = None
    if args.deeponet_model_path:
        print("Creating DeepONet model...")
        deeponet_model = create_synthetic_deeponet_model(args, params, device)
        print(f"Loading DeepONet from: {args.deeponet_model_path}")
        args.load_model_path = args.deeponet_model_path
        if not load_synthetic_pretrained_deeponet_model(deeponet_model, args, device):
            raise ValueError(f"Failed to load DeepONet model from {args.deeponet_model_path}")
    
    # Create dense evaluation points for plotting
    x_dense_plot = torch.linspace(params['input_range'][0], params['input_range'][1], 200, device=device).view(-1, 1)
    
    # Generate plots
    print(f"\nGenerating {args.num_samples} comparison plots...")
    
    if args.benchmark == 'derivative':
        # For derivative: f -> f'
        is_inverse_task = False
        print("Plotting for Derivative benchmark (f -> f')")
    else:
        # For integral: f' -> f
        is_inverse_task = True
        print("Plotting for Integral benchmark (f' -> f)")
    
    # Generate plots
    for i in range(1, args.num_samples + 1):
        if args.variable_sensors:
            # Generate random sensor locations for this sample
            torch.manual_seed(args.seed + i)  # Different seed for each sample
            sensor_x = torch.rand(params['sensor_size'], device=device) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
            sensor_x = sensor_x.sort()[0].view(-1, 1)
        else:
            sensor_x = sensor_x_original
        
        plot_dual_model_comparison(
            setonet_model=setonet_model,
            deeponet_model=deeponet_model,
            branch_input_locations=sensor_x,
            trunk_query_locations=x_dense_plot,
            input_range=params['input_range'],
            scale=params['scale'],
            save_dir=results_dir,
            sample_idx=i,
            benchmark_type=args.benchmark,
            is_inverse_task=is_inverse_task,
            use_zero_constant=True,
            sensor_dropoff=args.sensor_dropoff,
            replace_with_nearest=args.replace_with_nearest
        )
    
    # Save configuration info
    config_path = os.path.join(results_dir, "plot_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Benchmark: {args.benchmark}\n")
        f.write(f"SetONet model: {args.setonet_model_path}\n")
        f.write(f"DeepONet model: {args.deeponet_model_path}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"Sensor dropoff: {args.sensor_dropoff}\n")
        f.write(f"Variable sensors: {args.variable_sensors}\n")
        f.write(f"Replace with nearest: {args.replace_with_nearest}\n")
        f.write(f"Timestamp: {timestamp}\n")
    
    print(f"\nPlotting completed! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
