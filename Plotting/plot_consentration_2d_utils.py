#!/usr/bin/env python
"""plot_concentration_2d_utils.py
----------------------------------
Plotting utilities for 2D concentration problem results visualization.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from matplotlib.gridspec import GridSpec  # re-import for custom spacing
from scipy.interpolate import griddata
from matplotlib.patches import FancyArrowPatch

def add_wind_arrow(ax, wind_angle):
    """Add wind direction arrow to plot."""
    # Calculate arrow start and end points based on wind direction
    arrow_length = 0.2  # length of arrow in axes coordinates (increased from 0.1)
    center_x, center_y = 0.9, 0.95  # center position of arrow

    # Wind direction: 0 = +x, π/2 = +y, π = -x, 3π/2 = -y
    wind_x = np.cos(wind_angle)
    wind_y = np.sin(wind_angle)

    start_x = center_x - arrow_length/2 * wind_x
    start_y = center_y - arrow_length/2 * wind_y
    end_x = center_x + arrow_length/2 * wind_x
    end_y = center_y + arrow_length/2 * wind_y

    # Create arrow using FancyArrowPatch
    arrow = FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='->',
        lw=5,
        color='deepskyblue',
        mutation_scale=30,  # Size of arrow head
        transform=ax.transAxes,
        zorder=10
    )
    ax.add_patch(arrow)

def plot_concentration_results(model, dataset, concentration_dataset, device, sample_idx=0, save_path=None, dataset_split="test"):
    """
    Plot concentration 2D results: prediction, ground truth, and absolute error.
    
    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset
        concentration_dataset: ConcentrationDataset wrapper
        device: PyTorch device
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
        dataset_split: 'train' or 'test'
    """
    model.eval()
    
    # Get the sample
    data = dataset[dataset_split]
    sample = data[sample_idx]
    
    # Extract chemical sources
    sources = np.array(sample["sources"])  # shape: (n_sources, 3) -> [x, y, rate]
    wind_angle = sample.get("wind_angle", 0.0)  # wind direction in radians (fixed at 0.0 for +x direction)
    
    # Check if this is adaptive mesh or uniform grid
    is_adaptive = 'grid_coords' in sample
    
    # UNIFIED APPROACH: Always use standard visualization grid for both cases
    grid_n = 64  # Standard visualization resolution
    x = np.linspace(0, 1, grid_n)
    y = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Create target coordinates for prediction (always use regular grid for consistent visualization)
    viz_coords = np.column_stack([X.flatten(), Y.flatten()])  # (grid_n*grid_n, 2)
    
    # Get ground truth on visualization grid
    if is_adaptive:
        # Adaptive mesh: interpolate ground truth to visualization grid
        grid_coords_gt = np.array(sample["grid_coords"])  # (n_points, 2) -> [x, y]
        field_values_gt = np.array(sample["field_values"])  # (n_points,)
        field_gt = griddata(grid_coords_gt, field_values_gt, (X, Y), method='linear', fill_value=0.0)
    else:
        # Uniform grid: interpolate to standard visualization grid if needed
        original_field = np.array(sample["field"]).squeeze()  # (original_grid_n, original_grid_n)
        original_grid_n = original_field.shape[0]
        
        if original_grid_n != grid_n:
            # Interpolate from original grid to visualization grid
            x_orig = np.linspace(0, 1, original_grid_n)
            y_orig = np.linspace(0, 1, original_grid_n)
            X_orig, Y_orig = np.meshgrid(x_orig, y_orig, indexing='xy')
            orig_coords = np.column_stack([X_orig.flatten(), Y_orig.flatten()])
            orig_values = original_field.flatten()
            field_gt = griddata(orig_coords, orig_values, (X, Y), method='linear', fill_value=0.0)
        else:
            field_gt = original_field
    
    # Get model prediction on visualization grid (same for both adaptive and uniform)
    with torch.no_grad():
        source_coords = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_sources, 2)
        source_rates = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_sources, 1)
        target_coords = torch.tensor(viz_coords, device=device, dtype=torch.float32).unsqueeze(0)  # (1, grid_n*grid_n, 2)
        
        # Get prediction on visualization grid
        pred = model(source_coords, source_rates, target_coords)
        pred_field = pred.squeeze().cpu().numpy().reshape(grid_n, grid_n)
    
    # Calculate absolute error (same for both adaptive and uniform)
    abs_error = np.abs(field_gt - pred_field)
    
    # Determine common color scale based on ground truth for consistent comparison
    vmin = float(np.min(field_gt))
    vmax = float(np.max(field_gt))
    
    # -------------------------------------------------------------------
    # Custom layout with unequal gaps via intermediate blank columns.
    # Column pattern: 0-Prediction | 1-gap(0.1) | 2-GroundTruth | 3-gap(0.7) | 4-AbsoluteError
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(21.5, 6))
    gs = GridSpec(1, 5, width_ratios=[1, 0.04, 1, 0.32, 1], wspace=0.0)

    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 2])
    ax_err = fig.add_subplot(gs[0, 4])
    
    # Plot 1: Prediction
    ax = ax_pred
    # Use plasma colormap for concentration (better than hot for concentration visualization)
    im1 = ax.contourf(X, Y, pred_field, levels=100, cmap='plasma', vmin=vmin, vmax=vmax)
    
    # Add chemical sources
    if len(sources) > 0:
        ax.scatter(sources[:, 0], sources[:, 1], 
                  c=sources[:, 2], s=100, 
                  cmap='viridis', edgecolors='white', linewidth=2)
    
    # Note: Using unified visualization grid for consistent comparison
    # Wind direction is fixed at +x direction (0 radians) for all samples
    
    # ax.set_title('Prediction')
    ax.set_xlabel('X position', fontsize=16)
    ax.set_ylabel('Y position', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add wind direction arrow
    add_wind_arrow(ax, wind_angle)
    
    # Plot 2: Ground Truth
    ax = ax_gt
    im2 = ax.contourf(X, Y, field_gt, levels=100, cmap='plasma', vmin=vmin, vmax=vmax)
    
    # Add chemical sources
    if len(sources) > 0:
        ax.scatter(sources[:, 0], sources[:, 1], 
                  c=sources[:, 2], s=100, 
                  cmap='viridis', edgecolors='white', linewidth=2)
    
    # Note: Using unified visualization grid for consistent comparison
    # Wind direction is fixed at +x direction (0 radians) for all samples
    
    # ax.set_title('Ground Truth')
    ax.set_xlabel('X position', fontsize=16)
    ax.set_ylabel('Y position', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Add wind direction arrow
    add_wind_arrow(ax, wind_angle)
    
    # Plot 3: Absolute Error
    ax = ax_err
    im3 = ax.contourf(X, Y, abs_error, levels=100, cmap='Reds')
    
    # Add chemical sources
    if len(sources) > 0:
        ax.scatter(sources[:, 0], sources[:, 1], 
                  c='blue', s=100, 
                  edgecolors='white', linewidth=2, alpha=0.7)
    
    # Wind direction is fixed at +x direction (0 radians) for all samples
    
    # ax.set_title('Absolute Error')
    ax.set_xlabel('X position', fontsize=16)
    ax.set_ylabel('Y position', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Calculate and display error statistics
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    rmse = np.sqrt(np.mean(abs_error**2))
    
    # Add text box with error statistics - REMOVED per user request
    # mesh_type = "Adaptive" if is_adaptive else "Uniform"
    # error_text = f'{mesh_type} Dataset\n(Unified {grid_n}×{grid_n} Viz)\nMax Error: {max_error:.4f}\nMean Error: {mean_error:.4f}\nRMSE: {rmse:.4f}'
    # ax.text(0.02, 0.98, error_text, transform=ax.transAxes, 
    #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # First, tighten layout so subplots take their final positions
    fig.tight_layout()

    # Retrieve bounding boxes AFTER layout adjustment
    pos_gt = ax_gt.get_position()
    pos_err = ax_err.get_position()

    # Define gap and colorbar width (figure fraction)
    gap_cb = 0.015  # spacing between axis and its colorbar
    gap_axes = 0.02  # extra space between GT colorbar and Error subplot
    cb_width = 0.02

    # Concentration colorbar – right of GT subplot
    left_conc = pos_gt.x1 + gap_cb
    bottom = pos_gt.y0
    height = pos_gt.height

    cbar_ax1 = fig.add_axes([left_conc, bottom, cb_width, height])
    conc_cbar = fig.colorbar(im2, cax=cbar_ax1)
    conc_cbar.set_label('Concentration', rotation=270, labelpad=20, fontsize=16)
    conc_cbar.ax.tick_params(labelsize=16)

    # Absolute-error colorbar – right of Error subplot
    left_err = pos_err.x1 + gap_cb + 0.0 # place immediately after Error axis

    cbar_ax2 = fig.add_axes([left_err, bottom, cb_width, height])
    err_cbar = fig.colorbar(im3, cax=cbar_ax2)
    err_cbar.set_label('|Error|', rotation=270, labelpad=20, fontsize=16)
    err_cbar.ax.tick_params(labelsize=16)

    # Redraw the canvas to ensure everything renders correctly
    fig.canvas.draw()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved concentration results plot to {save_path}")
    
    return fig
