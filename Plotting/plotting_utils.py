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
    use_zero_constant: bool = True # New argument to control d_coeff for plotting
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
    """
    print(f"Generating plots with {len(branch_input_locations)} branch points and {len(trunk_query_locations)} trunk points...")
    
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

        # True function f(x) and its derivative f'(x)
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

        # Plot the input function (left subplot)
        axs[0, 0].plot(branch_input_locs_cpu.squeeze(), branch_values_true_np.squeeze(), 'b-', linewidth=2, label='Input Function')
        
        # Plot sensor points (every 10th sensor point to avoid clutter)
        sensor_indices = range(0, len(branch_input_locs_cpu), max(1, len(branch_input_locs_cpu) // 10))
        sensor_x_subset = branch_input_locs_cpu[sensor_indices]
        sensor_y_subset = branch_values_true_np[sensor_indices]
        axs[0, 0].scatter(sensor_x_subset.squeeze(), sensor_y_subset.squeeze(), 
                         c='red', s=50, zorder=5, alpha=0.8, 
                         label=f'Sensor Points (every {max(1, len(branch_input_locs_cpu) // 10)}th)')
        
        axs[0, 0].set_xlabel('$x$')
        axs[0, 0].set_ylabel(plot_ylabel_left)
        axs[0, 0].set_title(plot_title_left)
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()

        # Get model prediction
        with torch.no_grad():
            # Prepare inputs for the model (must be on model_device)
            branch_locs_model_dev = branch_input_locations.clone().to(device)
            trunk_query_locs_model_dev = trunk_query_locations.clone().to(device)

            # Branch values (u(x_i) or u(y_j)) are used as is (original scale)
            branch_values_torch = torch.tensor(branch_values_true_np, device=device, dtype=torch.float32)

            # Use prepare_setonet_inputs
            from Models.utils.helper_utils import prepare_setonet_inputs
            
            xs, us, ys = prepare_setonet_inputs(
                branch_locs_model_dev,
                1,  # batch_size = 1 for single sample
                branch_values_torch.unsqueeze(-1),  # Add feature dimension
                trunk_query_locs_model_dev,
                len(branch_input_locations)
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
        save_path = os.path.join(log_dir, f"{plot_filename_prefix}{task_type_str}_sample_{i+1}.png")
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