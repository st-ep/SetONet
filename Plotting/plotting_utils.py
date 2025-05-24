import torch
import matplotlib.pyplot as plt
import os
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
    Plots the results for a SetONet-like operator.
    If is_inverse_task is False (default): Assumes T: u(branch_locs) -> G(u)(trunk_locs) (e.g., f -> f')
    If is_inverse_task is True: Assumes T_inv: u(branch_locs) -> G(u)(trunk_locs) (e.g., f' -> f)
    Saves plots for a specified number of samples to the log directory.
    """
    if not model_to_use:
        print("No model provided to plot_operator_comparison. Skipping plot.")
        return

    model_name_str = "SetONet" # Assuming SetONet for now

    # Ensure locations are on CPU for numpy ops and plotting, but keep original device versions for model input
    branch_input_locs_cpu = branch_input_locations.cpu()
    trunk_query_locs_cpu = trunk_query_locations.cpu().squeeze() # Squeeze for 1D plotting

    model_device = next(model_to_use.parameters()).device

    for i in range(num_samples_to_plot):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)

        # Generate coefficients using the SAME method as training data
        # Use torch.rand instead of torch.randn to match training distribution
        a_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn
        b_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn
        c_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn
        if use_zero_constant:
            d_coeff = 0.0
        else:
            d_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn

        # True function f(x) and its derivative f'(x)
        def f_true(x_coords):
            return a_coeff * x_coords**3 + b_coeff * x_coords**2 + c_coeff * x_coords + d_coeff
        def df_true(x_coords):
            return 3*a_coeff*x_coords**2 + 2*b_coeff*x_coords + c_coeff

        # Determine what the branch sees and what the true output is based on task
        if not is_inverse_task: # Forward task: f -> f'
            branch_values_true = f_true(branch_input_locs_cpu) # f(x_i)
            operator_output_true = df_true(trunk_query_locs_cpu) # f'(y_j)
            
            plot_title_left = f'Input Function $f(x)$ (Sample {i+1})'
            plot_ylabel_left = '$f(x)$'
            plot_legend_left_true = 'True $f(x)$'
            
            plot_title_right = "Output Derivative $f'(y)$ (Sample {})".format(i+1)
            plot_ylabel_right = '$f\'(y)$'
            plot_legend_right_true = 'True $f\'(y)$'
            plot_legend_right_pred = "{} $\hat{{f}}'(y)$".format(model_name_str)
            
            # For smooth plotting of the input function f(x)
            dense_plot_locs_left = torch.linspace(input_range[0], input_range[1], 200, device='cpu')
            dense_plot_vals_left = f_true(dense_plot_locs_left)

        else: # Inverse task: f' -> f
            branch_values_true = df_true(branch_input_locs_cpu) # f'(y_j)
            operator_output_true = f_true(trunk_query_locs_cpu) # f(x_i)

            plot_title_left = "Input Derivative $f'(y)$ (Sample {})".format(i+1)
            plot_ylabel_left = '$f\'(y)$'
            plot_legend_left_true = 'True $f\'(y)$'

            plot_title_right = f'Output Function $f(x)$ (Sample {i+1})'
            plot_ylabel_right = '$f(x)$'
            plot_legend_right_true = 'True $f(x)$'
            plot_legend_right_pred = "{} $\hat{{f}}(x)$".format(model_name_str)

            # For smooth plotting of the input derivative f'(y)
            dense_plot_locs_left = torch.linspace(input_range[0], input_range[1], 200, device='cpu')
            dense_plot_vals_left = df_true(dense_plot_locs_left)


        predicted_output_denorm_for_plot = None

        with torch.no_grad():
            model_to_use.eval()
            
            # Prepare inputs for the model (must be on model_device)
            # Coordinates are used as is (assumed to be in the correct range, e.g., input_range)
            branch_locs_model_dev = branch_input_locations.clone().to(model_device)
            trunk_query_locs_model_dev = trunk_query_locations.clone().to(model_device)

            # Branch values (u(x_i) or u(y_j)) are used as is (original scale)
            branch_values_torch = torch.tensor(branch_values_true.numpy(), device=model_device, dtype=torch.float32)

            # Use prepare_setonet_inputs
            # batch_size is 1 for plotting individual samples
            # global_sensor_size for prepare_setonet_inputs is the number of points in branch_input_locations
            num_branch_points = branch_input_locations.shape[0]

            xs_model, us_model, ys_model = prepare_setonet_inputs(
                sensor_x_global=branch_locs_model_dev.view(-1,1), # [NumBranchPts, 1]
                current_batch_size=1,
                batch_f_values_norm_expanded=branch_values_torch.view(1, num_branch_points, 1), # [1, NumBranchPts, 1] - original scale
                batch_x_eval_norm=trunk_query_locs_model_dev.view(-1,1), # [NumTrunkQueryPts, 1]
                global_sensor_size=num_branch_points
            )
            
            # Model predicts in original target scale
            predicted_operator_output = model_to_use(xs_model, us_model, ys_model) # [1, NumTrunkQueryPts, 1]
            
            predicted_output_denorm_for_plot = predicted_operator_output.squeeze().cpu().numpy()

        # Plot 1: Input to the operator (either f(x) or f'(y))
        axs[0, 0].plot(dense_plot_locs_left.numpy(), dense_plot_vals_left.numpy(), label=plot_legend_left_true, color='blue', linestyle='-')
        axs[0, 0].set_xlabel('Domain coordinate')
        axs[0, 0].set_ylabel(plot_ylabel_left)
        axs[0, 0].set_title(plot_title_left)
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot 2: Output of the operator (either f'(y) or f(x))
        axs[0, 1].plot(trunk_query_locs_cpu.numpy(), operator_output_true.numpy(), label=plot_legend_right_true, color='green', linestyle='-')
        if predicted_output_denorm_for_plot is not None:
            axs[0, 1].plot(trunk_query_locs_cpu.numpy(), predicted_output_denorm_for_plot, label=plot_legend_right_pred, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Domain coordinate')
        axs[0, 1].set_ylabel(plot_ylabel_right)
        axs[0, 1].set_title(plot_title_right)
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        task_type_str = "inverse_reconstruction" if is_inverse_task else "forward_derivative"
        fig.suptitle(f'{model_name_str}: {task_type_str.replace("_", " ").title()} - Sample {i+1}', fontsize=16)
        
        # Use plot_filename_prefix
        final_plot_filename = f"{plot_filename_prefix}{task_type_str}_{model_name_str.lower()}_sample_{i+1}.png"
        save_path = os.path.join(log_dir, final_plot_filename)
        plt.savefig(save_path)
        print(f"Saved {task_type_str} plot for sample {i+1} to {save_path}")
        plt.close(fig)

def plot_cycle_consistency_reconstruction(
    model_1, # First model in the cycle (e.g., T for f->f'->f, or T_inv for f'->f->f')
    model_2, # Second model in the cycle (e.g., T_inv for f->f'->f, or T for f'->f->f')
    locs_A_orig, # Locations for initial input (to model_1) and final output (from model_2)
    locs_B_orig, # Locations for intermediate prediction (output of model_1, input to model_2)
    input_range,
    scale,
    log_dir,
    num_samples_to_plot=3,
    plot_filename_prefix="",
    cycle_starts_with_f=True, # True for f->f'->f, False for f'->f->f'
    use_zero_constant: bool = True # New argument to control d_coeff for plotting
):
    """
    Plots the cycle consistency reconstruction.
    If cycle_starts_with_f is True: Visualizes model_2(model_1(f)) vs. f.
    If cycle_starts_with_f is False: Visualizes model_2(model_1(f')) vs. f'.
    """
    if not model_1 or not model_2:
        print("One or both models not provided for cycle consistency plot. Skipping.")
        return

    model_1_name = "Model1" # Placeholder, can be made more specific if needed
    model_2_name = "Model2"
    
    # Determine device from model parameters
    try:
        model_device = next(model_1.parameters()).device
    except StopIteration: # Handle case where model might have no parameters (though unlikely for NNs)
        try:
            model_device = next(model_2.parameters()).device
        except StopIteration:
            print("Could not determine model device. Assuming CPU for plotting cycle consistency.")
            model_device = torch.device('cpu')


    locs_A_cpu = locs_A_orig.cpu().squeeze() # For plotting original and final reconstruction
    locs_B_cpu = locs_B_orig.cpu().squeeze() # For intermediate step (not directly plotted but used for queries)

    # Use original locations directly on the model's device
    locs_A_model_dev = locs_A_orig.clone().to(model_device)
    locs_B_model_dev = locs_B_orig.clone().to(model_device)

    num_locs_A = locs_A_orig.shape[0]
    num_locs_B = locs_B_orig.shape[0]

    for i in range(num_samples_to_plot):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), squeeze=False)

        # Generate coefficients using the SAME method as training data
        a_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn
        b_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn
        c_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn
        if use_zero_constant:
            d_coeff = 0.0
        else:
            d_coeff = (torch.rand(1).item() * 2 - 1) * scale  # Changed from torch.randn

        def f_true_fn(x_coords):
            return a_coeff * x_coords**3 + b_coeff * x_coords**2 + c_coeff * x_coords + d_coeff
        def df_true_fn(x_coords):
            return 3*a_coeff*x_coords**2 + 2*b_coeff*x_coords + c_coeff

        original_input_for_cycle_cpu = None
        original_input_for_cycle_torch_model_dev = None # [1, num_locs_A]
        
        plot_title_left_panel = ""
        plot_ylabel_left_panel = ""
        plot_legend_left_panel_true = ""
        plot_title_right_panel = ""
        plot_ylabel_right_panel = ""
        plot_legend_right_panel_true = ""
        plot_legend_right_panel_reconstructed = ""
        cycle_description = ""

        if cycle_starts_with_f: # Cycle: f -> model_1 (T) -> f' -> model_2 (T_inv) -> f_reconstructed
            original_input_for_cycle_cpu = f_true_fn(locs_A_cpu) # f_true at locs_A
            original_input_for_cycle_torch_model_dev = torch.tensor(original_input_for_cycle_cpu.numpy(), device=model_device, dtype=torch.float32).unsqueeze(0)
            
            plot_title_left_panel = f'Original $f(x)$ (Sample {i+1})'
            plot_ylabel_left_panel = '$f(x)$'
            plot_legend_left_panel_true = 'Original $f(x)$'
            plot_title_right_panel = f'Reconstructed $f(x)$ vs. Original (Sample {i+1})'
            plot_ylabel_right_panel = '$f(x)$'
            plot_legend_right_panel_true = 'Original $f(x)$'
            plot_legend_right_panel_reconstructed = 'Reconstructed $T_{inv}(T(f))$'
            cycle_description = "f_to_f_reconstruction"

            # Branch input for model_1 is f at locs_A, trunk query for model_1 is locs_B
            branch_locs_m1 = locs_A_model_dev
            branch_val_count_m1 = num_locs_A
            trunk_locs_m1 = locs_B_model_dev
            
            # Branch input for model_2 is f' (output of m1) at locs_B, trunk query for model_2 is locs_A
            branch_locs_m2 = locs_B_model_dev
            branch_val_count_m2 = num_locs_B
            trunk_locs_m2 = locs_A_model_dev

        else: # Cycle: f' -> model_1 (T_inv) -> f -> model_2 (T) -> f'_reconstructed
            original_input_for_cycle_cpu = df_true_fn(locs_A_cpu) # df_true at locs_A
            original_input_for_cycle_torch_model_dev = torch.tensor(original_input_for_cycle_cpu.numpy(), device=model_device, dtype=torch.float32).unsqueeze(0)

            plot_title_left_panel = f"Original $f'(y)$ (Sample {i+1})"
            plot_ylabel_left_panel = "$f'(y)$"
            plot_legend_left_panel_true = "Original $f'(y)$"
            plot_title_right_panel = f"Reconstructed $f'(y)$ vs. Original (Sample {i+1})"
            plot_ylabel_right_panel = "$f'(y)$"
            plot_legend_right_panel_true = "Original $f'(y)$"
            plot_legend_right_panel_reconstructed = "Reconstructed $T(T_{inv}(f'))$"
            cycle_description = "fprime_to_fprime_reconstruction"

            # Branch input for model_1 is f' at locs_A, trunk query for model_1 is locs_B
            branch_locs_m1 = locs_A_model_dev
            branch_val_count_m1 = num_locs_A
            trunk_locs_m1 = locs_B_model_dev

            # Branch input for model_2 is f (output of m1) at locs_B, trunk query for model_2 is locs_A
            branch_locs_m2 = locs_B_model_dev
            branch_val_count_m2 = num_locs_B
            trunk_locs_m2 = locs_A_model_dev
            
        # Use original input directly
        initial_input_m1_values = original_input_for_cycle_torch_model_dev # Shape [1, num_locs_A]

        reconstructed_denorm_plot = None
        with torch.no_grad():
            model_1.eval()
            model_2.eval()

            # Step 1: Pass through model_1
            xs_m1, us_m1, ys_m1 = prepare_setonet_inputs(
                branch_locs_m1, 1, initial_input_m1_values.unsqueeze(-1), trunk_locs_m1, branch_val_count_m1
            )
            intermediate_prediction = model_1(xs_m1, us_m1, ys_m1) # [1, num_locs_B, 1] - original scale of model_1's target

            # Step 2: Pass through model_2
            # The output of model_1 (intermediate_prediction) is the branch input values for model_2
            xs_m2, us_m2, ys_m2 = prepare_setonet_inputs(
                branch_locs_m2, 1, intermediate_prediction, trunk_locs_m2, branch_val_count_m2
            )
            final_reconstruction = model_2(xs_m2, us_m2, ys_m2) # [1, num_locs_A, 1] - original scale of model_2's target
                                                                    # which should be the original input's scale.

            reconstructed_denorm_plot = final_reconstruction.squeeze().cpu().numpy()

        # Plot 1 (Left Panel): Original input to the cycle
        # For smooth plotting of the original function/derivative
        dense_plot_locs_cpu_A = torch.linspace(input_range[0], input_range[1], 200, device='cpu')
        dense_plot_vals_original_A = f_true_fn(dense_plot_locs_cpu_A) if cycle_starts_with_f else df_true_fn(dense_plot_locs_cpu_A)
        
        axs[0, 0].plot(dense_plot_locs_cpu_A.numpy(), dense_plot_vals_original_A.numpy(), label=plot_legend_left_panel_true, color='blue', linestyle='-')
        axs[0, 0].set_xlabel('Domain coordinate')
        axs[0, 0].set_ylabel(plot_ylabel_left_panel)
        axs[0, 0].set_title(plot_title_left_panel)
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot 2 (Right Panel): Reconstructed vs. Original
        axs[0, 1].plot(locs_A_cpu.numpy(), original_input_for_cycle_cpu.numpy(), label=plot_legend_right_panel_true, color='green', linestyle='-')
        if reconstructed_denorm_plot is not None:
            axs[0, 1].plot(locs_A_cpu.numpy(), reconstructed_denorm_plot, label=plot_legend_right_panel_reconstructed, color='red', linestyle='--')
        axs[0, 1].set_xlabel('Domain coordinate')
        axs[0, 1].set_ylabel(plot_ylabel_right_panel)
        axs[0, 1].set_title(plot_title_right_panel)
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(f'Cycle Consistency: {cycle_description.replace("_", " ").title()} - Sample {i+1}', fontsize=16)
        
        final_plot_filename = f"{plot_filename_prefix}{cycle_description}_sample_{i+1}.png"
        save_path = os.path.join(log_dir, final_plot_filename)
        plt.savefig(save_path)
        print(f"Saved cycle consistency plot for sample {i+1} to {save_path}")
        plt.close(fig)

def plot_trunk_basis_functions(deeponet_model, setonet_model, x_basis, setonet_p_dim, log_dir):
    """
    Plots the trunk basis functions for DeepONet or SetONet.
    All basis functions for a model are plotted on a single subplot.
    Saves the plots to the specified log directory.
    """
    if deeponet_model is None and setonet_model is None:
        print("No models provided to plot_trunk_basis_functions. Skipping plots.")
        return

    x_basis_cpu = x_basis.cpu().numpy().squeeze()

    if deeponet_model:
        try:
            with torch.no_grad():
                basis_deeponet = deeponet_model.trunk_net(x_basis).cpu().numpy()
            
            p_deeponet = basis_deeponet.shape[1]
            
            fig_don, ax_don = plt.subplots(figsize=(10, 6)) # Single subplot
            # fig_don.suptitle("DeepONet Trunk Basis Functions", fontsize=16) # Removed title
            
            for i in range(p_deeponet):
                ax_don.plot(x_basis_cpu, basis_deeponet[:, i], label=f"Basis {i+1}") # Label kept for potential future use, but legend removed
            
            ax_don.set_xlabel("y")
            ax_don.set_ylabel("Trunk Output Value")
            # ax_don.legend(loc='best') # Removed legend
            ax_don.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Keep tight_layout, adjust if suptitle was the only reason for rect
            plot_filename_don = os.path.join(log_dir, "deeponet_trunk_basis_plot.png")
            plt.savefig(plot_filename_don)
            print(f"DeepONet trunk basis plot saved to {plot_filename_don}")
            plt.close(fig_don)
        except Exception as e:
            print(f"Could not plot DeepONet trunk basis functions: {e}")


    if setonet_model and setonet_p_dim is not None and setonet_p_dim > 0:
        try:
            with torch.no_grad():
                basis_setonet = setonet_model.trunk(x_basis).cpu().numpy()

            if basis_setonet.shape[1] == setonet_p_dim * setonet_model.output_size_tgt:
                if setonet_model.output_size_tgt > 1:
                    print(f"Warning: SetONet output_size_tgt is {setonet_model.output_size_tgt}. Plotting assumes it's effectively 1 for basis visualization.")
                
                fig_son, ax_son = plt.subplots(figsize=(10, 6)) # Single subplot
                # fig_son.suptitle(f"SetONet Trunk Basis Functions (p={setonet_p_dim})", fontsize=16) # Removed title

                for i in range(setonet_p_dim):
                    # Assuming output_size_tgt is 1, so we take the i-th column directly
                    ax_son.plot(x_basis_cpu, basis_setonet[:, i], label=f"Basis {i+1}") # Label kept, legend removed
                
                ax_son.set_xlabel("y")
                ax_son.set_ylabel("Trunk Output Value")
                # ax_son.legend(loc='best') # Removed legend
                ax_son.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0, 1, 0.96]) # Keep tight_layout
                plot_filename_son = os.path.join(log_dir, "setonet_trunk_basis_plot.png")
                plt.savefig(plot_filename_son)
                print(f"SetONet trunk basis plot saved to {plot_filename_son}")
                plt.close(fig_son)
            else:
                print(f"Could not plot SetONet trunk basis functions: Trunk output shape {basis_setonet.shape} not compatible with p={setonet_p_dim} and assumed output_size_tgt=1.")
        except Exception as e:
            print(f"Could not plot SetONet trunk basis functions: {e}")
    elif setonet_model and (setonet_p_dim is None or setonet_p_dim <= 0):
        print("SetONet model provided but setonet_p_dim is invalid. Skipping SetONet trunk basis plot.") 