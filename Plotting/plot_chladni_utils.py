import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.axes import Axes

def plot_chladni_results(model, dataset, chladni_dataset, device, sample_idx=0, save_path=None, eval_sensor_dropoff=0.0, replace_with_nearest=False) -> plt.Figure:
    """Plot input forces, predicted displacements, and ground truth for Chladni plate."""
    model.eval()
    test_data = dataset['test']
    sample = test_data[sample_idx]
    
    # Load pre-normalized data
    xs_norm = torch.tensor(sample['X'], dtype=torch.float32, device=device)
    xs = xs_norm.unsqueeze(0)
    
    us_norm = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)
    us = us_norm.unsqueeze(-1)
    
    ys_norm = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
    ys = ys_norm.unsqueeze(0)
    
    target_norm = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Apply sensor dropout if specified
    xs_used = xs
    us_used = us
    n_sensors_orig = xs.shape[1]
    dropout_info = ""
    
    if eval_sensor_dropoff > 0.0:
        # Check if this is SetONet or DeepONet based on model capabilities
        if hasattr(model, 'forward_branch'):  # SetONet
            from Data.data_utils import apply_sensor_dropoff
            
            # SetONet can handle variable input sizes, so we use removal/replacement
            xs_dropped, us_dropped = apply_sensor_dropoff(
                xs.squeeze(0),  # Remove batch dimension: (n_sensors, 2)
                us.squeeze(0).squeeze(-1),  # Remove batch and feature dimensions: (n_sensors,)
                eval_sensor_dropoff,
                replace_with_nearest
            )
            
            # Add batch dimension back
            xs_used = xs_dropped.unsqueeze(0)  # (1, n_remaining_sensors, 2)
            us_used = us_dropped.unsqueeze(0).unsqueeze(-1)  # (1, n_remaining_sensors, 1)
            
            n_sensors_remaining = xs_used.shape[1]
            replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
            dropout_info = f" (Dropout: {eval_sensor_dropoff:.1%}, {n_sensors_remaining}/{n_sensors_orig} sensors, {replacement_mode})"
            
        else:  # DeepONet
            from Data.data_utils import apply_sensor_dropoff_with_2d_interpolation
            
            # For Chladni 2D problem, use bilinear interpolation to maintain fixed input size
            # Chladni uses a 32x32 regular grid (1024 points total)
            grid_shape = (32, 32)  # Based on numPoints in chladni_plate_generator.py
            
            # Apply dropout with bilinear interpolation 
            _, us_interpolated = apply_sensor_dropoff_with_2d_interpolation(
                xs.squeeze(0),  # 2D coordinates [n_sensors, 2]
                us.squeeze(0).squeeze(-1),  # Sensor values [n_sensors]
                grid_shape,
                eval_sensor_dropoff
            )
            
            # Keep original sensor locations, use interpolated values
            xs_used = xs  # (1, n_sensors, 2) - unchanged
            us_used = us_interpolated.unsqueeze(0).unsqueeze(-1)  # (1, n_sensors, 1)
            
            dropout_info = f" (Dropout: {eval_sensor_dropoff:.1%}, bilinear interpolation)"
    else:
        us_used = us
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'forward_branch'):  # SetONet
            pred_norm = model(xs_used, us_used, ys)
        else:  # DeepONet
            pred_norm = model(xs_used, us_used, ys)  # xs_used is dummy, ignored by DeepONet
    
    # Get prediction and denormalize
    pred_orig = chladni_dataset.denormalize_displacement(pred_norm.squeeze(-1))
    
    # Use original data if available, otherwise denormalize
    if 'u_orig' in sample:
        forces_orig = torch.tensor(sample['u_orig'], dtype=torch.float32, device=device)
        target_orig = torch.tensor(sample['s_orig'], dtype=torch.float32, device=device)
        coords_orig = torch.tensor(sample['X_orig'], dtype=torch.float32, device=device)
    else:
        forces_orig = chladni_dataset.denormalize_force(us_norm.squeeze(0))
        target_orig = chladni_dataset.denormalize_displacement(target_norm.squeeze(0))
        coords_orig = chladni_dataset.denormalize_coordinates(xs_norm)
    
    # Reshape to 2D grid
    grid_size = int(np.sqrt(len(sample['X'])))
    coords = coords_orig.cpu().numpy()
    x_coords = coords[:, 0].reshape(grid_size, grid_size)
    y_coords = coords[:, 1].reshape(grid_size, grid_size)
    
    forces_2d = forces_orig.cpu().numpy().reshape(grid_size, grid_size)
    pred_2d = pred_orig.cpu().numpy().reshape(grid_size, grid_size)
    target_2d = target_orig.cpu().numpy().reshape(grid_size, grid_size)
    
    # Calculate absolute error
    abs_error = np.abs(pred_2d - target_2d)
    
    # Compute shared vmin/vmax for prediction and ground truth displacements
    vmin_shared = min(pred_2d.min(), target_2d.min())
    vmax_shared = max(pred_2d.max(), target_2d.max())
    
    # Compute scaling orders for colorbars
    forces_vabs = max(abs(forces_2d.min()), abs(forces_2d.max()))
    forces_order = int(np.floor(np.log10(forces_vabs))) if forces_vabs > 0 else 0
    
    shared_vabs = max(abs(vmin_shared), abs(vmax_shared))
    shared_order = int(np.floor(np.log10(shared_vabs))) if shared_vabs > 0 else 0
    
    error_vabs = np.max(abs_error)
    error_order = int(np.floor(np.log10(error_vabs))) if error_vabs > 0 else 0
    
    # Create figure with GridSpec for custom widths and variable spacing
    fig = plt.figure(figsize=(28, 6))
    plot_ratios = [1, 1, 1, 1]  # Equal ratios for all four plots
    has_colorbar = [True, False, True, True]  # Input Forces: yes, Predicted: no, Ground Truth: yes, Error: yes
    cb_ratio = 0.05  # Colorbar width ratio
    cb_spacer = 0.001  # Small spacer between plot and its colorbar
    wspaces = [0.2, 0.05, 0.2]  # Custom spaces: normal between 1-2, tight between 2-3, normal between 3-4
    
    # Build full ratio list including spaces and colorbars
    num_plots = len(plot_ratios)
    avg_width = sum(plot_ratios) / num_plots
    spacer_widths = [w * avg_width for w in wspaces]
    full_ratios = []
    col_starts = []
    current_col = 0
    
    for i in range(num_plots):
        if i > 0:
            full_ratios.append(spacer_widths[i-1])
            current_col += 1
        full_ratios.append(plot_ratios[i])
        col_starts.append(current_col)
        current_col += 1
        if has_colorbar[i]:
            # Add small spacer before colorbar
            full_ratios.append(cb_spacer)
            current_col += 1
            full_ratios.append(cb_ratio)
            current_col += 1
    
    gs = fig.add_gridspec(1, len(full_ratios), width_ratios=full_ratios, wspace=0)
    axes = [fig.add_subplot(gs[0, col_starts[j]]) for j in range(num_plots)]
    
    # Create colorbar axes where needed
    cb_axes = {}
    for i in range(num_plots):
        if has_colorbar[i]:
            cb_col = col_starts[i] + 2  # +2 because of the spacer
            cb_axes[i] = fig.add_subplot(gs[0, cb_col])
    
    # Apply scaling to data
    forces_scaled = forces_2d * (10 ** -forces_order)
    pred_scaled = pred_2d * (10 ** -shared_order)
    target_scaled = target_2d * (10 ** -shared_order)
    error_scaled = abs_error * (10 ** -error_order)
    
    # Scale vmin/vmax for shared colorbar
    vmin_scaled = vmin_shared * (10 ** -shared_order)
    vmax_scaled = vmax_shared * (10 ** -shared_order)
    
    # Input forces
    im1 = axes[0].contourf(x_coords, y_coords, forces_scaled, levels=20, cmap='Spectral_r')
    axes[0].set_xlabel('X (m)', fontsize=16)
    axes[0].set_ylabel('Y (m)', fontsize=16)
    axes[0].set_aspect('equal', adjustable='box')  # Ensure square aspect ratio
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar1 = plt.colorbar(im1, cax=cb_axes[0], format='%.1f')
    cbar1.ax.set_title(f'10^{{{forces_order}}}', fontsize=16, pad=20)
    cbar1.ax.tick_params(labelsize=16)
    
    # Predicted displacements (no colorbar, shared scale)
    im2 = axes[1].contourf(x_coords, y_coords, pred_scaled, levels=20, cmap='RdBu_r', vmin=vmin_scaled, vmax=vmax_scaled)
    # Add zero displacement contour lines (Chladni nodal lines)
    axes[1].contour(x_coords, y_coords, pred_scaled, levels=[0], colors='gold', linewidths=2, alpha=1.0, linestyles='-')
    axes[1].set_xlabel('X (m)', fontsize=16)
    axes[1].set_ylabel('Y (m)', fontsize=16)
    axes[1].set_aspect('equal', adjustable='box')  # Ensure square aspect ratio
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Ground truth displacements (with shared colorbar)
    im3 = axes[2].contourf(x_coords, y_coords, target_scaled, levels=20, cmap='RdBu_r', vmin=vmin_scaled, vmax=vmax_scaled)
    # Add zero displacement contour lines (Chladni nodal lines)
    axes[2].contour(x_coords, y_coords, target_scaled, levels=[0], colors='gold', linewidths=2, alpha=1.0, linestyles='-')
    axes[2].set_xlabel('X (m)', fontsize=16)
    axes[2].set_ylabel('Y (m)', fontsize=16)
    axes[2].set_aspect('equal', adjustable='box')  # Ensure square aspect ratio
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar3 = plt.colorbar(im3, cax=cb_axes[2], format='%.1f')
    cbar3.ax.set_title(f'10^{{{shared_order}}}', fontsize=16, pad=20)
    cbar3.ax.tick_params(labelsize=16)
    
    # Absolute error (with its own colorbar)
    im4 = axes[3].contourf(x_coords, y_coords, error_scaled, levels=20, cmap='coolwarm')
    axes[3].set_xlabel('X (m)', fontsize=16)
    axes[3].set_ylabel('Y (m)', fontsize=16)
    axes[3].set_aspect('equal', adjustable='box')  # Ensure square aspect ratio
    axes[3].tick_params(axis='both', which='major', labelsize=14)
    axes[3].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar4 = plt.colorbar(im4, cax=cb_axes[3], format='%.1f')
    cbar4.ax.set_title(f'10^{{{error_order}}}', fontsize=16, pad=20)
    cbar4.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Error metrics
    mse_error = np.mean((pred_2d - target_2d)**2)
    rel_error = np.linalg.norm(pred_2d - target_2d) / np.linalg.norm(target_2d)
    
    if eval_sensor_dropoff > 0.0:
        replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
        print(f"Sample {sample_idx} with {eval_sensor_dropoff:.1%} sensor dropout ({replacement_mode}) - MSE: {mse_error:.6e}, Rel Error: {rel_error:.6f}")
    else:
        print(f"Sample {sample_idx} - MSE: {mse_error:.6e}, Rel Error: {rel_error:.6f}")
    
    model.train()
