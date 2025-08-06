import torch
import matplotlib.pyplot as plt
import os
import numpy as np
# Assuming normalization functions are accessible, e.g. if Plotting is a sibling to Models
# Adjust path if necessary, or ensure helper_utils is in PYTHONPATH
from Models.utils.helper_utils import prepare_setonet_inputs

def plot_operator_comparison(
    model_to_use,
    branch_input_locations, # Locations for the function fed into the branch (e.g., x_i for T, y_j for T_inv)
    trunk_query_locations,  # Locations where the output is queried (e.g., y_j for T, x_i for T_inv)
    input_range,
    scale,
    log_dir,
    num_samples_to_plot=3,
    plot_filename_prefix="", # New argument
    is_inverse_task=False,    # New argument
    use_zero_constant: bool = True, # New argument to control d_coeff for plotting
    sensor_dropoff: float = 0.0,  # New argument for sensor drop-off
    replace_with_nearest: bool = False,  # New argument for nearest neighbor replacement
    show_sensor_markers: bool = True  # Whether to show sensor location markers
):
    """
    Plots comparison between true and predicted functions for operator learning.
    
    Args:
        model_to_use: Trained SetONet model
        branch_input_locations: Locations where branch input is sampled [n_branch, 1]
        trunk_query_locations: Locations where trunk output is queried [n_trunk, 1] 
        input_range: Input domain range
        scale: Scale for polynomial coefficients
        log_dir: Directory to save plots
        num_samples_to_plot: Number of sample functions to plot
        plot_filename_prefix: Prefix for plot filenames
        is_inverse_task: Whether this is inverse task (f' -> f) or forward (f -> f')
        use_zero_constant: Whether to use zero integration constant
        sensor_dropoff: Sensor drop-off rate to apply (same as training/evaluation)
        replace_with_nearest: Whether to replace dropped sensors with nearest neighbors
        show_sensor_markers: Whether to show sensor location markers (plots every ~20th sensor)
    """
    print(f"Generating plots with {len(branch_input_locations)} branch points and {len(trunk_query_locations)} trunk points...")
    
    # Check if sensors have unique locations (for variable sensor detection)
    unique_branch_locs = len(torch.unique(branch_input_locations.squeeze(), dim=0))
    print(f"📍 Branch sensor locations: {len(branch_input_locations)} total, {unique_branch_locs} unique")
    
    if sensor_dropoff > 0:
        # Detect if this is DeepONet based on calling context
        import inspect
        calling_file = inspect.stack()[2].filename if len(inspect.stack()) > 2 else ""
        
        if 'DeepONet' in calling_file or 'don' in calling_file:
            replacement_mode = "interpolation"
        else:
            replacement_mode = "nearest neighbor replacement" if replace_with_nearest else "removal"
        print(f"🔧 Applying sensor drop-off rate: {sensor_dropoff:.1%} with {replacement_mode}")
    
    if show_sensor_markers:
        print(f"🎯 Will plot every ~{max(1, len(branch_input_locations) // 20)}th sensor for visualization clarity") 
    
    # Note: For variable sensor training, we use fixed sensor locations for plotting
    # to ensure consistent visualization across different runs
    
    model_to_use.eval()
    device = next(model_to_use.parameters()).device

    if not model_to_use:
        print("No model provided to plot_operator_comparison. Skipping plot.")
        return

    if hasattr(model_to_use, 'forward_branch'):
        model_name_str = "SetONet"
    else:
        model_name_str = "DeepONet"

    # Ensure locations are on CPU for numpy ops and plotting, but keep original device versions for model input
    branch_input_locs_cpu = branch_input_locations.cpu()
    trunk_query_locs_cpu = trunk_query_locations.cpu().squeeze() # Squeeze for 1D plotting

    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,          # Default text size
        'axes.labelsize': 16,     # X and Y labels
        'xtick.labelsize': 16,    # X axis tick labels
        'ytick.labelsize': 16,    # Y axis tick labels
        'legend.fontsize': 16,    # Legend text (line labels)
        'axes.titlesize': 18      # Title text (not used but set for consistency)
    })

    for i in range(num_samples_to_plot):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)

        # Generate coefficients using the SAME method as training data
        # Use torch.rand instead of torch.randn to match training distribution
        a_coeff = (torch.rand(1).item() * 2 - 1) * scale
        b_coeff = (torch.rand(1).item() * 2 - 1) * scale
        c_coeff = (torch.rand(1).item() * 2 - 1) * scale
        if use_zero_constant:
            d_coeff = 0.0
        else:
            d_coeff = (torch.rand(1).item() * 2 - 1) * scale

        # True function f(x) and its derivative f'(x) - DEFINE BEFORE USING
        def f_true(x_coords):
            return a_coeff * x_coords**3 + b_coeff * x_coords**2 + c_coeff * x_coords + d_coeff
        def df_true(x_coords):
            return 3*a_coeff*x_coords**2 + 2*b_coeff*x_coords + c_coeff

        # Determine what the branch sees and what the true output is based on task
        if not is_inverse_task: # Forward task: f -> f' (derivative benchmark)
            branch_values_true = f_true(branch_input_locs_cpu) # f(x_i)
            operator_output_true = df_true(trunk_query_locs_cpu) # f'(y_j)
            
            plot_ylabel_left = '$f(x)$'
            plot_ylabel_right = '$f\'(x)$'
            task_type_str = "derivative"
            
        else: # Inverse task: f' -> f (integral benchmark)
            branch_values_true = df_true(branch_input_locs_cpu) # f'(x_i)
            operator_output_true = f_true(trunk_query_locs_cpu) # f(y_j)
            
            plot_ylabel_left = '$f\'(x)$'
            plot_ylabel_right = '$f(x)$'
            task_type_str = "integral"

        # Convert to numpy for plotting
        branch_values_true_np = branch_values_true.numpy() if hasattr(branch_values_true, 'numpy') else branch_values_true
        operator_output_true_np = operator_output_true.numpy() if hasattr(operator_output_true, 'numpy') else operator_output_true

        # Apply sensor drop-off if specified (same as training/evaluation)
        if sensor_dropoff > 0.0:
            from Data.data_utils import apply_sensor_dropoff
            
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

        # Plot the input function (left subplot) with optional sensor locations
        axs[0, 0].plot(branch_input_locs_plot.squeeze(), branch_values_plot.squeeze(), 'b-', linewidth=2, label='Input Function')
        
        if show_sensor_markers:
            # Add sensor locations (plot every 20th sensor for clarity)
            sensor_step = max(1, len(branch_input_locs_plot) // 20)  # Show ~20 sensors max
            sensor_indices = range(0, len(branch_input_locs_plot), sensor_step)
            
            sensor_x_subset = branch_input_locs_plot.squeeze()[sensor_indices]
            sensor_y_subset = branch_values_plot.squeeze()[sensor_indices]
            
            axs[0, 0].scatter(sensor_x_subset, sensor_y_subset, c='red', s=50, zorder=5, 
                             label=f'Sensors (every {sensor_step}th, {len(sensor_indices)} shown)')
        
        # Count unique sensor locations
        unique_sensors = len(torch.unique(branch_input_locs_plot.squeeze(), dim=0))
        
        # Add text box with sensor info
        sensor_info = f'Total sensors: {actual_n_sensors}\nUnique locations: {unique_sensors}'
        if sensor_dropoff > 0:
            original_sensors = len(branch_input_locations) if sensor_dropoff > 0 else actual_n_sensors
            sensor_info += f'\nOriginal: {original_sensors} (dropoff: {sensor_dropoff:.1%})'
        
        axs[0, 0].text(0.02, 0.98, sensor_info, transform=axs[0, 0].transAxes, 
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axs[0, 0].set_xlabel('$x$')
        axs[0, 0].set_ylabel(plot_ylabel_left)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Get model prediction
        with torch.no_grad():
            # Prepare inputs for the model (must be on model_device)
            trunk_query_locs_model_dev = trunk_query_locations.clone().to(device)

            if hasattr(model_to_use, 'forward_branch'):  # SetONet
                xs, us, ys = prepare_setonet_inputs(
                    branch_input_locs_model,
                    1,  # batch_size = 1 for single sample
                    branch_values_model.unsqueeze(-1),  # Add feature dimension
                    trunk_query_locs_model_dev,
                    actual_n_sensors  # Use actual number of sensors after drop-off
                )

                # Get model prediction
                operator_output_pred = model_to_use(xs, us, ys)
                operator_output_pred = operator_output_pred.squeeze().cpu().numpy()
            else:  # DeepONet
                # Prepare inputs for DeepONetWrapper which expects (xs, us, ys)
                xs = branch_input_locs_model.unsqueeze(0)  # [1, actual_n_sensors, 1] - sensor locations (ignored by DeepONet but required)
                us = branch_values_model.unsqueeze(0).unsqueeze(-1)  # [1, actual_n_sensors, 1] - sensor values
                ys = trunk_query_locs_model_dev.unsqueeze(0)  # [1, n_trunk, 1] - query locations
                operator_output_pred = model_to_use(xs, us, ys).squeeze().cpu().numpy()

        # Plot the output comparison (right subplot)
        axs[0, 1].plot(trunk_query_locs_cpu, operator_output_true_np.squeeze(), 'g-', linewidth=2, label='True')
        axs[0, 1].plot(trunk_query_locs_cpu, operator_output_pred, 'r--', linewidth=2, label=f'{model_name_str} Prediction')
        axs[0, 1].set_xlabel('$x$')
        axs[0, 1].set_ylabel(plot_ylabel_right)
        # NO TITLE: axs[0, 1].set_title(plot_title_right)
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        plt.tight_layout()
        
        # Save plot
        replacement_suffix = "_nearest" if replace_with_nearest and sensor_dropoff > 0 else ""
        dropoff_suffix = f"_dropoff_{sensor_dropoff:.1f}{replacement_suffix}" if sensor_dropoff > 0 else ""
        save_path = os.path.join(log_dir, f"{plot_filename_prefix}{task_type_str}_sample_{i+1}{dropoff_suffix}.png")
        plt.savefig(save_path)
        print(f"Saved {task_type_str} plot for sample {i+1} to {save_path}")
        plt.close(fig)

    # Reset font parameters to default to avoid affecting other plots
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'axes.titlesize': 12
    })


def plot_synthetic_1d_comparison(model, data_generator, params, log_dir, args, show_sensor_markers=True):
    """Plot synthetic 1D benchmark results with proper handling of variable sensors and benchmarks."""
    from Data.data_utils import generate_batch
    
    print("Generating plots...")
    
    device = next(model.parameters()).device
    
    # Create dense evaluation points for plotting
    x_dense_plot = torch.linspace(params['input_range'][0], params['input_range'][1], 200, device=device).view(-1, 1)
    
    # Generate both train and test plots
    for data_type in ['train', 'test']:
        print(f"\n--- Generating {data_type.upper()} plots ---")
        
        # Determine sensor dropoff based on data type
        current_sensor_dropoff = params.get('eval_sensor_dropoff', 0.0) if data_type == 'test' else 0.0
        
        if args.benchmark == 'derivative':
            # For derivative: sensor locations -> dense plot locations
            print(f"Plotting for Derivative Model (f -> f') - {data_type.upper()} data")
            
            if params.get('variable_sensors', False):
                # For variable sensor training, use actual sensor locations from evaluation batches
                print("Using actual sensor locations from evaluation batches")
                
                # Generate 3 different evaluation batches to show different sensor configurations
                for batch_idx in range(3):
                    print(f"Generating {data_type} plot for evaluation batch {batch_idx + 1}")
                    
                    batch_data = generate_batch(
                        batch_size=1,
                        n_trunk_points=params['n_trunk_points_train'],
                        sensor_x=None,
                        scale=params['scale'],
                        input_range=params['input_range'],
                        device=device,
                        constant_zero=True,
                        variable_sensors=True,
                        sensor_size=params['sensor_size']
                    )
                    
                    actual_sensor_locations = batch_data[5]
                    
                    plot_operator_comparison(
                        model_to_use=model,
                        branch_input_locations=actual_sensor_locations,
                        trunk_query_locations=x_dense_plot,
                        input_range=params['input_range'],
                        scale=params['scale'],
                        log_dir=log_dir,
                        num_samples_to_plot=1,
                        plot_filename_prefix=f"{data_type}_{args.benchmark}_eval_batch_{batch_idx+1}_",
                        is_inverse_task=False,
                        use_zero_constant=True,
                        sensor_dropoff=current_sensor_dropoff,
                        replace_with_nearest=params.get('replace_with_nearest', False),
                        show_sensor_markers=show_sensor_markers
                    )
            else:
                # Fixed sensors - use the original sensor locations
                plot_operator_comparison(
                    model_to_use=model,
                    branch_input_locations=data_generator.sensor_x_original,
                    trunk_query_locations=x_dense_plot,
                    input_range=params['input_range'],
                    scale=params['scale'],
                    log_dir=log_dir,
                    num_samples_to_plot=3,
                    plot_filename_prefix=f"{data_type}_{args.benchmark}_",
                    is_inverse_task=False,
                    use_zero_constant=True,
                    sensor_dropoff=current_sensor_dropoff,
                    replace_with_nearest=params.get('replace_with_nearest', False),
                    show_sensor_markers=show_sensor_markers
                )
        elif args.benchmark == 'integral':
            # For integral: f' -> f (dense plot locations -> sensor locations)
            print(f"Plotting for Integral Model (f' -> f) - {data_type.upper()} data")
            
            if params.get('variable_sensors', False):
                # For variable sensor training, generate different batches to show different sensor configurations
                print("Using variable sensor locations (different query sensor locations per batch)")
                
                # Generate 3 different evaluation batches to show different sensor configurations
                for batch_idx in range(3):
                    print(f"Generating {data_type} plot for evaluation batch {batch_idx + 1}")
                    
                    batch_data = generate_batch(
                        batch_size=1,
                        n_trunk_points=params['n_trunk_points_train'],
                        sensor_x=None,
                        scale=params['scale'],
                        input_range=params['input_range'],
                        device=device,
                        constant_zero=True,
                        variable_sensors=True,
                        sensor_size=params['sensor_size']
                    )
                    
                    actual_sensor_locations = batch_data[5]
                    
                    plot_operator_comparison(
                        model_to_use=model,
                        branch_input_locations=actual_sensor_locations,
                        trunk_query_locations=x_dense_plot,
                        input_range=params['input_range'],
                        scale=params['scale'],
                        log_dir=log_dir,
                        num_samples_to_plot=1,
                        plot_filename_prefix=f"{data_type}_{args.benchmark}_eval_batch_{batch_idx+1}_",
                        is_inverse_task=True,
                        use_zero_constant=True,
                        sensor_dropoff=current_sensor_dropoff,
                        replace_with_nearest=params.get('replace_with_nearest', False),
                        show_sensor_markers=show_sensor_markers
                    )
            else:
                # Fixed sensors - use the original sensor locations
                plot_sensor_x = data_generator.sensor_x_original if data_generator.sensor_x_original is not None else torch.rand(params['sensor_size'], device=device).sort()[0].view(-1, 1) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
                plot_operator_comparison(
                    model_to_use=model,
                    branch_input_locations=plot_sensor_x,
                    trunk_query_locations=x_dense_plot,
                    input_range=params['input_range'],
                    scale=params['scale'],
                    log_dir=log_dir,
                    num_samples_to_plot=3,
                    plot_filename_prefix=f"{data_type}_{args.benchmark}_",
                    is_inverse_task=True,
                    use_zero_constant=True,
                    sensor_dropoff=current_sensor_dropoff,
                    replace_with_nearest=params.get('replace_with_nearest', False),
                    show_sensor_markers=show_sensor_markers
                ) 