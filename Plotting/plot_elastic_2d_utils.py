#!/usr/bin/env python
"""
Plotting utilities for Elastic 2D plate results.
Creates plots with 4 subplots: forcing function, prediction, ground truth, and absolute error.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch

def create_circular_mask(x, y, center_x=0.5, center_y=0.5, radius=0.25):
    """Create a circular mask for the hole in the plate."""
    return (x - center_x)**2 + (y - center_y)**2 <= radius**2

def plot_displacement_field(coords, displacement, title, ax, colorbar_label='x-displacement', add_colorbar=True, vmin=None, vmax=None, cax=None, scaling_order=None):
    """Plot displacement field with circular hole."""
    # Extract coordinates
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Create regular grid for interpolation
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create dense grid for smooth visualization
    xi = np.linspace(x_min, x_max, 200)
    yi = np.linspace(y_min, y_max, 200)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate displacement values onto regular grid
    Zi = griddata((x_coords, y_coords), displacement, (Xi, Yi), method='cubic')
    
    # Scale data if scaling_order provided
    if scaling_order is not None:
        scale = 10 ** -scaling_order
        Zi *= scale
        if vmin is not None:
            vmin *= scale
        if vmax is not None:
            vmax *= scale
    
    # Create circular mask for the hole
    hole_mask = create_circular_mask(Xi, Yi)
    
    # Set hole region to NaN so it appears empty
    Zi[hole_mask] = np.nan
    
    # Plot displacement field with optional vmin/vmax
    im = ax.contourf(Xi, Yi, Zi, levels=100, cmap='jet', vmin=vmin, vmax=vmax)
    
    # Set color limits if provided
    if vmin is not None and vmax is not None:
        im.set_clim(vmin, vmax)
    
    if cax is not None:
        cbar = plt.colorbar(im, cax=cax, format='%.1f')
        cbar.set_label(colorbar_label, rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.get_offset_text().set_size(16)
        if scaling_order is not None:
            cbar.ax.set_title('10^{%d}' % scaling_order, fontsize=16, pad=10)
    elif add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format='%.1f')
        cbar.set_label(colorbar_label, rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.yaxis.get_offset_text().set_size(16)
        if scaling_order is not None:
            cbar.ax.set_title('10^{%d}' % scaling_order, fontsize=16, pad=10)
    
    # Set equal aspect ratio and add axes
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

def plot_forcing_function(force_coords, force_values, title, ax):
    """Plot the forcing function as a 1D plot."""
    # Extract y-coordinates and force values
    force_y = force_coords[:, 1]  # y-coordinates of force points
    force_x = force_values  # force magnitudes
    
    # Plot force as horizontal line plot
    ax.plot(force_x, force_y, 'r-', linewidth=2)
    
    # Set up force subplot
    ax.set_ylim(force_y.min(), force_y.max())
    ax.set_xlabel('Force value', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Invert x-axis so zero is at the right (force applied from right side)
    ax.invert_xaxis()
    
    # Add vertical line at zero force
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)

def plot_elastic_results(model, dataset, elastic_dataset, device, sample_idx=0, save_path=None, dataset_split="test", 
                        eval_sensor_dropoff=0.0, replace_with_nearest=False):
    """Plot input forces, predicted displacements, and ground truth for Elastic 2D plate."""
    
    # Get the appropriate dataset split
    if dataset_split == "test":
        data_split = dataset['test']
    else:
        data_split = dataset['train']
    
    # Get sample
    sample = data_split[sample_idx]
    
    # Prepare data
    xs_norm = torch.tensor(sample['X'], dtype=torch.float32, device=device)
    xs = xs_norm.unsqueeze(0)
    
    us_norm = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)
    us = us_norm.unsqueeze(-1)
    
    ys_norm = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
    ys = ys_norm.unsqueeze(0)
    
    target_norm = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)
    target = target_norm.unsqueeze(-1)
    
    # Apply sensor dropout if specified
    xs_used = xs
    us_used = us
    dropout_info = ""
    if eval_sensor_dropoff > 0.0:
        from Data.data_utils import apply_sensor_dropoff
        
        # Apply dropout to sensor data (remove batch dimension for dropout function)
        xs_dropped, us_dropped = apply_sensor_dropoff(
            xs.squeeze(0),  # Remove batch dimension: (n_sensors, 2)
            us.squeeze(0).squeeze(-1),  # Remove batch and feature dimensions: (n_sensors,)
            eval_sensor_dropoff,
            replace_with_nearest
        )
        
        # Add batch dimension back
        xs_used = xs_dropped.unsqueeze(0)  # (1, n_remaining_sensors, 2)
        us_used = us_dropped.unsqueeze(0).unsqueeze(-1)  # (1, n_remaining_sensors, 1)
        
        replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
        dropout_info = f" (w/ {eval_sensor_dropoff:.1%} sensor dropout - {replacement_mode})"
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'forward_branch'):  # SetONet
            pred_norm = model(xs_used, us_used, ys)
        else:  # DeepONet
            pred_norm = model(xs_used, us_used, ys)  # xs_used is dummy, ignored by DeepONet
    
    # Convert to numpy and denormalize
    pred_orig = elastic_dataset.denormalize_displacement(pred_norm.squeeze(-1))
    pred_orig = pred_orig.cpu().numpy().squeeze(0)
    
    # Get coordinates and denormalize
    coords_orig = elastic_dataset.denormalize_coordinates(xs_norm)
    coords_orig = coords_orig.cpu().numpy()
    
    # Get forcing function values and denormalize
    forces_orig = elastic_dataset.denormalize_force(us_norm.squeeze(0))
    forces_orig = forces_orig.cpu().numpy()
    
    # Get target coordinates and values
    target_coords = elastic_dataset.denormalize_coordinates(ys_norm)
    target_coords = target_coords.cpu().numpy()
    
    target_orig = elastic_dataset.denormalize_displacement(target_norm.squeeze(0))
    target_orig = target_orig.cpu().numpy()
    
    # Calculate absolute error
    abs_error = np.abs(pred_orig - target_orig)
    
    # Compute shared vmin/vmax for prediction and ground truth
    vmin_shared = min(pred_orig.min(), target_orig.min())
    vmax_shared = max(pred_orig.max(), target_orig.max())

    # Compute scaling orders
    shared_vabs = max(abs(vmin_shared), abs(vmax_shared))
    shared_order = np.floor(np.log10(shared_vabs)) if shared_vabs > 0 else 0
    error_vabs = np.max(abs_error)
    error_order = np.floor(np.log10(error_vabs)) if error_vabs > 0 else 0
    
    # Create figure with GridSpec for custom widths and variable spacing
    fig = plt.figure(figsize=(20, 5))
    plot_ratios = [0.3, 1, 1, 1]
    has_colorbar = [False, False, True, True]
    cb_ratio = 0.07
    wspaces = [0.15, 0.05, 0.35]  # Custom spaces between plot units: between 1-2, 2-3, 3-4
    num_plots = len(plot_ratios)
    assert len(wspaces) == num_plots - 1
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
            full_ratios.append(cb_ratio)
            current_col += 1
    gs = fig.add_gridspec(1, len(full_ratios), width_ratios=full_ratios, wspace=0)
    axes = [fig.add_subplot(gs[0, col_starts[j]]) for j in range(num_plots)]
    cb_axes = [None] * num_plots
    for i in range(num_plots):
        if has_colorbar[i]:
            cb_col = col_starts[i] + 1
            cb_axes[i] = fig.add_subplot(gs[0, cb_col])
    
    # Plot 1: Forcing function (narrow)
    plot_forcing_function(coords_orig, forces_orig, 'Forcing Function', axes[0])
    
    # Plot 2: Prediction (no colorbar, shared scale)
    plot_displacement_field(target_coords, pred_orig, 'Prediction', axes[1], add_colorbar=False, vmin=vmin_shared, vmax=vmax_shared)
    
    # Plot 3: Ground truth (with colorbar, shared scale)
    plot_displacement_field(target_coords, target_orig, 'Ground Truth', axes[2], add_colorbar=False, vmin=vmin_shared, vmax=vmax_shared, cax=cb_axes[2], scaling_order=shared_order)
    
    # Plot 4: Absolute error (with its own colorbar)
    plot_displacement_field(target_coords, abs_error, 'Absolute Error', axes[3], colorbar_label='Error', add_colorbar=False, cax=cb_axes[3], scaling_order=error_order)
    
    # Adjust layout with reduced spacing between subplots
    plt.tight_layout(pad=0, w_pad=0)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    # Return figure for further use if needed
    return fig
