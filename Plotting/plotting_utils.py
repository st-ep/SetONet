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
    replace_with_nearest: bool = False  # New argument for nearest neighbor replacement
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
    """
    print(f"Generating plots with {len(branch_input_locations)} branch points and {len(trunk_query_locations)} trunk points...")
    if sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if replace_with_nearest else "removal"
        print(f"Applying sensor drop-off rate: {sensor_dropoff:.1%} with {replacement_mode}")
    
    # Note: For variable sensor training, we use fixed sensor locations for plotting
    # to ensure consistent visualization across different runs
    
    model_to_use.eval()
    device = next(model_to_use.parameters()).device

    if not model_to_use:
        print("No model provided to plot_operator_comparison. Skipping plot.")
        return

    model_name_str = "SetONet"

    # Ensure locations are on CPU for numpy ops and plotting, but keep original device versions for model input
    branch_input_locs_cpu = branch_input_locations.cpu()
    trunk_query_locs_cpu = trunk_query_locations.cpu().squeeze() # Squeeze for 1D plotting

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
            
            plot_title_left = f'Input Function $f(x)$ (Sample {i+1})'
            plot_ylabel_left = '$f(x)$'
            plot_title_right = f'Output: True vs Predicted $f\'(x)$ (Sample {i+1})'
            plot_ylabel_right = '$f\'(x)$'
            task_type_str = "derivative"
            
        else: # Inverse task: f' -> f (integral benchmark)
            branch_values_true = df_true(branch_input_locs_cpu) # f'(x_i)
            operator_output_true = f_true(trunk_query_locs_cpu) # f(y_j)
            
            plot_title_left = f'Input Function $f\'(x)$ (Sample {i+1})'
            plot_ylabel_left = '$f\'(x)$'
            plot_title_right = f'Output: True vs Predicted $f(x)$ (Sample {i+1})'
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

        # Plot the input function (left subplot)
        axs[0, 0].plot(branch_input_locs_plot.squeeze(), branch_values_plot.squeeze(), 'b-', linewidth=2, label='Input Function')
        
        # Plot sensor points (every 20th sensor point to avoid clutter)
        if replace_with_nearest and sensor_dropoff > 0:
            # For nearest neighbor replacement, plot based on unique sensors only
            unique_sensor_locs, unique_indices = torch.unique(branch_input_locs_plot, dim=0, return_inverse=True)
            unique_sensor_values = []
            
            # Get values corresponding to unique locations
            for i, unique_loc in enumerate(unique_sensor_locs):
                # Find first occurrence of this unique location
                first_idx = (branch_input_locs_plot == unique_loc.unsqueeze(0)).all(dim=1).nonzero()[0].item()
                unique_sensor_values.append(branch_values_plot[first_idx])
            
            unique_sensor_values = np.array(unique_sensor_values)
            
            # Plot every 20th unique sensor
            sensor_indices = range(0, len(unique_sensor_locs), 20)
            sensor_x_subset = unique_sensor_locs[sensor_indices]
            sensor_y_subset = unique_sensor_values[sensor_indices]
            
            axs[0, 0].scatter(sensor_x_subset.squeeze(), sensor_y_subset.squeeze(), 
                             c='red', s=50, zorder=5, alpha=0.8, 
                             label=f'Unique sensors (every 20th)')
        else:
            # Normal case: plot every 20th sensor
            sensor_indices = range(0, len(branch_input_locs_plot), 20)
            sensor_x_subset = branch_input_locs_plot[sensor_indices]
            sensor_y_subset = branch_values_plot[sensor_indices]
            
            axs[0, 0].scatter(sensor_x_subset.squeeze(), sensor_y_subset.squeeze(), 
                             c='red', s=50, zorder=5, alpha=0.8, 
                             label=f'Sensors (every 20th)')
        
        axs[0, 0].set_xlabel('$x$')
        axs[0, 0].set_ylabel(plot_ylabel_left)
        axs[0, 0].set_title(plot_title_left)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Get model prediction
        with torch.no_grad():
            # Prepare inputs for the model (must be on model_device)
            trunk_query_locs_model_dev = trunk_query_locations.clone().to(device)

            # Use prepare_setonet_inputs
            from Models.utils.helper_utils import prepare_setonet_inputs
            
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

        # Plot the output comparison (right subplot)
        axs[0, 1].plot(trunk_query_locs_cpu, operator_output_true_np.squeeze(), 'g-', linewidth=2, label='True')
        axs[0, 1].plot(trunk_query_locs_cpu, operator_output_pred, 'r--', linewidth=2, label=f'{model_name_str} Prediction')
        axs[0, 1].set_xlabel('$x$')
        axs[0, 1].set_ylabel(plot_ylabel_right)
        axs[0, 1].set_title(plot_title_right)
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

def plot_trunk_basis_functions(model, x_basis, p_dim, log_dir, model_name="SetONet"):
    """
    Plots the trunk basis functions for a neural operator model.
    All basis functions for a model are plotted on a single subplot.
    Saves the plots to the specified log directory.
    
    Args:
        model: The neural operator model (SetONet or DeepONet)
        x_basis: Input coordinates for plotting basis functions
        p_dim: Latent dimension (number of basis functions)
        log_dir: Directory to save plots
        model_name: Name of the model for filename
    """
    if model is None:
        print(f"No {model_name} model provided to plot_trunk_basis_functions. Skipping plots.")
        return

    x_basis_cpu = x_basis.cpu().numpy().squeeze()

    try:
        with torch.no_grad():
            # For SetONet, use the trunk method; for DeepONet, use trunk_net
            if hasattr(model, 'trunk'):
                basis_output = model.trunk(x_basis).cpu().numpy()
            elif hasattr(model, 'trunk_net'):
                basis_output = model.trunk_net(x_basis).cpu().numpy()
            else:
                print(f"Model does not have 'trunk' or 'trunk_net' method. Cannot plot basis functions.")
                return
        
        actual_p_dim = basis_output.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot up to p_dim basis functions (or all if fewer available)
        num_to_plot = min(p_dim, actual_p_dim) if p_dim else actual_p_dim
        
        for i in range(num_to_plot):
            ax.plot(x_basis_cpu, basis_output[:, i], label=f"Basis {i+1}", alpha=0.8)
        
        ax.set_xlabel("Input coordinate")
        ax.set_ylabel("Trunk Output Value")
        ax.set_title(f"{model_name} Trunk Basis Functions (showing {num_to_plot}/{actual_p_dim})")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only show legend if not too many basis functions
        if num_to_plot <= 10:
            ax.legend(loc='best')
        
        plt.tight_layout()
        plot_filename = os.path.join(log_dir, f"{model_name.lower()}_trunk_basis_plot.png")
        plt.savefig(plot_filename)
        print(f"{model_name} trunk basis plot saved to {plot_filename}")
        plt.close(fig)
        
    except Exception as e:
        print(f"Could not plot {model_name} trunk basis functions: {e}")

# Legacy function for backward compatibility
def plot_trunk_basis_functions_legacy(deeponet_model, setonet_model, x_basis, setonet_p_dim, log_dir):
    """
    Legacy function for backward compatibility.
    Plots the trunk basis functions for DeepONet or SetONet.
    """
    if deeponet_model:
        plot_trunk_basis_functions(deeponet_model, x_basis, None, log_dir, "DeepONet")
    
    if setonet_model:
        plot_trunk_basis_functions(setonet_model, x_basis, setonet_p_dim, log_dir, "SetONet")

def plot_darcy_comparison(
    model_to_use,
    dataset,
    sensor_x,
    query_x,
    sensor_indices,
    query_indices,
    log_dir,
    num_samples_to_plot=3,
    plot_filename_prefix="darcy_1d",
    sensor_dropoff: float = 0.0,
    replace_with_nearest: bool = False,
    dataset_split="test",
    batch_size=64,  # Add batch_size parameter to determine sample spacing
    variable_sensors=False,  # Add parameter to handle variable sensor plotting
    grid_points=None  # Grid points for variable sensor interpolation
):
    """
    Plots comparison between true and predicted solutions for Darcy 1D dataset.
    Creates separate plots for each sample, with 2 subplots per plot (similar to derivative benchmark).
    
    Args:
        model_to_use: Trained SetONet model
        dataset: Darcy 1D dataset (HuggingFace dataset)
        sensor_x: Sensor locations [n_sensors, 1] (used for test plots or when variable_sensors=False)
        query_x: Query locations [n_queries, 1] 
        sensor_indices: Indices for sensor locations in the grid (used when variable_sensors=False)
        query_indices: Indices for query locations in the grid
        log_dir: Directory to save plots
        num_samples_to_plot: Number of sample functions to plot
        plot_filename_prefix: Prefix for plot filenames
        sensor_dropoff: Sensor drop-off rate to apply (same as evaluation)
        replace_with_nearest: Whether to replace dropped sensors with nearest neighbors
        dataset_split: Which dataset split to use ("train" or "test")
        batch_size: Batch size used during evaluation (to select samples from different batches)
        variable_sensors: Whether variable sensors were used during training
        grid_points: Grid points for variable sensor interpolation (required if variable_sensors=True)
    """
    print(f"Generating Darcy 1D plots with {len(sensor_x)} sensor points and {len(query_x)} query points...")
    if sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if replace_with_nearest else "removal"
        print(f"Applying sensor drop-off rate: {sensor_dropoff:.1%} with {replacement_mode}")
    
    # Print sensor mode for clarity
    if variable_sensors:
        print(f"🔄 Using VARIABLE sensor locations for both train and test plots (same as training/evaluation)")
        print(f"Note: Each plot will show different random sensor locations, reflecting actual model usage")
    else:
        print(f"📍 Using FIXED sensor locations for both train and test plots")
    
    model_to_use.eval()
    device = next(model_to_use.parameters()).device

    if not model_to_use:
        print("No model provided to plot_darcy_comparison. Skipping plot.")
        return

    # Get data from specified split
    split_data = dataset[dataset_split]
    n_available = len(split_data)
    
    # Select samples from different batches to get diverse sensor dropout patterns
    # This ensures we see different sensor configurations when dropout is applied
    sample_indices = []
    for i in range(num_samples_to_plot):
        # Select first sample from each batch: batch_i * batch_size + 0
        sample_idx = i * batch_size
        if sample_idx < n_available:
            sample_indices.append(sample_idx)
        else:
            # Fall back to sequential if we run out of batches
            sample_indices.append(i)
    
    n_to_plot = len(sample_indices)
    
    print(f"Plotting {n_to_plot} samples from {dataset_split} set")
    print(f"Sample indices: {sample_indices} (from different batches for diverse sensor patterns)")
    
    # Convert query locations to CPU for plotting
    query_x_cpu = query_x.cpu().squeeze()

    for plot_idx, sample_idx in enumerate(sample_indices):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)

        # Get sample data
        sample = split_data[sample_idx]
        x_grid = torch.tensor(sample['X'], dtype=torch.float32)
        u_full = torch.tensor(sample['u'], dtype=torch.float32).to(device)
        s_full = torch.tensor(sample['s'], dtype=torch.float32).to(device)
        
        # Determine sensor locations based on variable_sensors and dataset_split
        if variable_sensors and dataset_split == "train":
            # TRAIN plots with variable sensors: Generate new random sensor locations for each sample
            # This reflects what actually happens during training
            from Data.data_utils import sample_variable_sensor_points
            
            # Generate random sensor locations from continuous domain [0,1] (same as training)
            current_sensor_x = sample_variable_sensor_points(
                len(sensor_x), 
                [0, 1],  # Darcy domain
                device
            ).squeeze(-1)  # [n_sensors] for interpolation
            
            # Interpolate sensor values at these arbitrary locations
            # We need to import the interpolation function from the Darcy script
            from Benchmarks.run_darcy_1d import interpolate_sensor_values
            u_at_sensors_true = interpolate_sensor_values(
                u_full.unsqueeze(0), 
                grid_points.to(device), 
                current_sensor_x, 
                device
            ).squeeze(0)  # Remove batch dimension
            
            # Convert sensor locations for plotting and model input
            current_sensor_x = current_sensor_x.view(-1, 1)  # [n_sensors, 1]
            sensor_x_for_plot = current_sensor_x
            sensor_indices_for_plot = None  # Not applicable for variable sensors
        elif variable_sensors and dataset_split == "test":
            # TEST plots with variable sensors: Also generate random sensor locations
            # This reflects what actually happens during evaluation (now that we fixed it)
            from Data.data_utils import sample_variable_sensor_points
            
            # Generate random sensor locations from continuous domain [0,1] (same as evaluation)
            current_sensor_x = sample_variable_sensor_points(
                len(sensor_x), 
                [0, 1],  # Darcy domain
                device
            ).squeeze(-1)  # [n_sensors] for interpolation
            
            # Interpolate sensor values at these arbitrary locations
            from Benchmarks.run_darcy_1d import interpolate_sensor_values
            u_at_sensors_true = interpolate_sensor_values(
                u_full.unsqueeze(0), 
                grid_points.to(device), 
                current_sensor_x, 
                device
            ).squeeze(0)  # Remove batch dimension
            
            # Convert sensor locations for plotting and model input
            current_sensor_x = current_sensor_x.view(-1, 1)  # [n_sensors, 1]
            sensor_x_for_plot = current_sensor_x
            sensor_indices_for_plot = None  # Not applicable for variable sensors
        else:
            # FIXED sensors: Use the provided fixed sensor locations
            u_at_sensors_true = u_full[sensor_indices]
            sensor_x_for_plot = sensor_x
            sensor_indices_for_plot = sensor_indices
        
        s_at_queries_true = s_full[query_indices]

        # Apply sensor drop-off if specified (same as evaluation)
        if sensor_dropoff > 0.0:
            from Data.data_utils import apply_sensor_dropoff
            
            # Apply drop-off to sensor data using the correct sensor locations
            sensor_x_device = sensor_x_for_plot.clone().to(device)
            sensor_locs_dropped, sensor_values_dropped = apply_sensor_dropoff(
                sensor_x_device, u_at_sensors_true, sensor_dropoff, replace_with_nearest
            )
            
            # For plotting
            sensor_x_plot = sensor_locs_dropped.cpu()
            u_at_sensors_plot = sensor_values_dropped.cpu().numpy()
            
            # For model input
            sensor_x_model = sensor_locs_dropped
            u_at_sensors_model = sensor_values_dropped
            actual_n_sensors = len(sensor_x_plot)
        else:
            # No drop-off - use the determined sensor locations
            sensor_x_plot = sensor_x_for_plot.cpu()
            u_at_sensors_plot = u_at_sensors_true.cpu().numpy()
            sensor_x_model = sensor_x_for_plot
            u_at_sensors_model = u_at_sensors_true
            actual_n_sensors = len(sensor_x_plot)

        # Plot input source term u(x) with sensor points (left subplot)
        axs[0, 0].plot(x_grid.cpu(), u_full.cpu(), 'b-', linewidth=2, label='Input Source Term $u(x)$')
        
        # Plot sensor points (every nth sensor point to avoid clutter)
        if replace_with_nearest and sensor_dropoff > 0:
            # For nearest neighbor replacement, plot based on unique sensors only
            unique_sensor_locs, unique_indices = torch.unique(sensor_x_plot, dim=0, return_inverse=True)
            unique_sensor_values = []
            
            # Get values corresponding to unique locations
            for j, unique_loc in enumerate(unique_sensor_locs):
                # Find first occurrence of this unique location
                first_idx = (sensor_x_plot == unique_loc.unsqueeze(0)).all(dim=1).nonzero()[0].item()
                unique_sensor_values.append(u_at_sensors_plot[first_idx])
            
            unique_sensor_values = np.array(unique_sensor_values)
            
            # Plot every nth unique sensor
            step_size = max(1, len(unique_sensor_locs)//10)
            sensor_indices_plot = range(0, len(unique_sensor_locs), step_size)
            sensor_x_subset = unique_sensor_locs[sensor_indices_plot]
            sensor_y_subset = unique_sensor_values[sensor_indices_plot]
            
            axs[0, 0].scatter(sensor_x_subset.squeeze(), sensor_y_subset.squeeze(), 
                             c='red', s=50, zorder=5, alpha=0.8, 
                             label=f'Unique sensors (every {step_size}th)')
        else:
            # Normal case: plot every nth sensor
            step_size = max(1, len(sensor_x_plot)//10)
            sensor_indices_plot = range(0, len(sensor_x_plot), step_size)
            sensor_x_subset = sensor_x_plot[sensor_indices_plot]
            sensor_y_subset = u_at_sensors_plot[sensor_indices_plot]
            
            axs[0, 0].scatter(sensor_x_subset.squeeze(), sensor_y_subset.squeeze(), 
                             c='red', s=50, zorder=5, alpha=0.8, 
                             label=f'Sensors (every {step_size}th)')
        
        axs[0, 0].set_xlabel('$x$')
        axs[0, 0].set_ylabel('$u(x)$')
        
        # Create title that reflects sensor type
        if variable_sensors:
            title_suffix = f"(Sample {sample_idx}, Variable Sensors)"
        else:
            title_suffix = f"(Sample {sample_idx}, Fixed Sensors)"
        axs[0, 0].set_title(f'Input Source Term $u(x)$ {title_suffix}')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Get model prediction
        with torch.no_grad():
            # Prepare inputs for the model
            query_x_model = query_x.clone().to(device)

            # Use prepare_setonet_inputs for consistency
            from Models.utils.helper_utils import prepare_setonet_inputs
            
            xs, us, ys = prepare_setonet_inputs(
                sensor_x_model,
                1,  # batch_size = 1 for single sample
                u_at_sensors_model.unsqueeze(-1),  # Add feature dimension
                query_x_model,
                actual_n_sensors
            )

            # Get model prediction
            s_at_queries_pred = model_to_use(xs, us, ys)
            s_at_queries_pred = s_at_queries_pred.squeeze().cpu().numpy()

        # Plot true vs predicted solution s(x) (right subplot)
        axs[0, 1].plot(query_x_cpu, s_at_queries_true.cpu().numpy(), 'g-', linewidth=2, label='True')
        axs[0, 1].plot(query_x_cpu, s_at_queries_pred, 'r--', linewidth=2, label='SetONet Prediction')
        axs[0, 1].set_xlabel('$x$')
        axs[0, 1].set_ylabel('$s(x)$')
        axs[0, 1].set_title(f'Output: True vs Predicted $s(x)$ {title_suffix}')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()

        plt.tight_layout()
        
        # Save plot
        replacement_suffix = "_nearest" if replace_with_nearest and sensor_dropoff > 0 else ""
        dropoff_suffix = f"_dropoff_{sensor_dropoff:.1f}{replacement_suffix}" if sensor_dropoff > 0 else ""
        save_path = os.path.join(log_dir, f"{plot_filename_prefix}_{dataset_split}_sample_{sample_idx}_plot_{plot_idx+1}{dropoff_suffix}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Darcy 1D {dataset_split} plot for sample {sample_idx} (plot {plot_idx+1}) to {save_path}")
        plt.close(fig) 