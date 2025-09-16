#!/usr/bin/env python
"""plot_heat_2d_utils.py
----------------------------------
Plotting utilities for 2D heat problem results visualization.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from matplotlib.gridspec import GridSpec  # re-import for custom spacing
from scipy.interpolate import griddata

def plot_heat_results(model, dataset, heat_dataset, device, sample_idx=0, save_path=None, dataset_split="test"):
    """
    Plot heat 2D results: prediction, ground truth, and absolute error.
    
    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset
        heat_dataset: HeatDataset wrapper
        device: PyTorch device
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
        dataset_split: 'train' or 'test'
    """
    model.eval()
    
    # Get the sample
    data = dataset[dataset_split]
    sample = data[sample_idx]
    
    # Extract heat sources
    sources = np.array(sample["sources"])  # shape: (n_sources, 3) -> [x, y, power]
    
    # Check if this is adaptive mesh or uniform grid
    is_adaptive = 'grid_coords' in sample
    
    # Create visualization grid (always uniform for plotting)
    grid_n = 64  # Standard visualization resolution
    x = np.linspace(0, 1, grid_n)
    y = np.linspace(0, 1, grid_n)
    # Use consistent indexing: X[i,j] corresponds to x[j], Y[i,j] corresponds to y[i]
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    if is_adaptive:
        # Adaptive mesh data
        grid_coords_gt = np.array(sample["grid_coords"])  # (n_points, 2) -> [x, y]
        field_values_gt = np.array(sample["field_values"])  # (n_points,)
        
        # Interpolate ground truth to regular grid for visualization
        # grid_coords_gt[:, 0] = x coordinates, grid_coords_gt[:, 1] = y coordinates
        field_gt = griddata(grid_coords_gt, field_values_gt, (X, Y), method='cubic', fill_value=np.nan)
        
        with torch.no_grad():
            # Prepare model inputs - use same adaptive grid points as ground truth
            source_coords = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_sources, 2)
            source_powers = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_sources, 1)
            target_coords = torch.tensor(grid_coords_gt, device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_points, 2)
            
            # Get prediction on adaptive grid
            pred = model(source_coords, source_powers, target_coords)
            pred_values = pred.squeeze().cpu().numpy()  # (n_points,)
            
            # Interpolate prediction to regular grid for visualization
            pred_field = griddata(grid_coords_gt, pred_values, (X, Y), method='cubic', fill_value=np.nan)
    else:
        # Uniform grid data (original format)
        field_gt = np.array(sample["field"]).squeeze()  # Ground truth temperature field
        grid_n = field_gt.shape[0]
        
        # Update visualization grid to match data grid
        x = np.linspace(0, 1, grid_n)
        y = np.linspace(0, 1, grid_n)
        X, Y = np.meshgrid(x, y, indexing='xy')
        
        with torch.no_grad():
            # Prepare model inputs
            source_coords = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_sources, 2)
            source_powers = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_sources, 1)
            target_coords = heat_dataset.grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
            
            # Get prediction
            pred = model(source_coords, source_powers, target_coords)
            pred_field = pred.squeeze().cpu().numpy().reshape(grid_n, grid_n)
    
    # Calculate absolute error (works for both adaptive and uniform)
    abs_error = np.abs(field_gt - pred_field)
    
    # Handle NaN values in interpolated data
    if is_adaptive:
        # For adaptive mesh, we might have NaN values from interpolation
        # Replace NaN with 0 for error calculation (or use nanmax/nanmean)
        abs_error = np.nan_to_num(abs_error, nan=0.0)
        field_gt = np.nan_to_num(field_gt, nan=0.0)
        pred_field = np.nan_to_num(pred_field, nan=0.0)
    
    # Determine common color scale based on ground truth for consistent comparison
    vmin = float(np.nanmin(field_gt))
    vmax = float(np.nanmax(field_gt))
    
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
    # For contour plots with meshgrid indexing='xy': X[i,j] = x[j], Y[i,j] = y[i]
    # So field[i,j] corresponds to location (x[j], y[i])
    # We need to transpose field to match this: field.T[i,j] corresponds to (x[i], y[j])
    im1 = ax.contourf(X, Y, pred_field, levels=100, cmap='hot', vmin=vmin, vmax=vmax)
    
    # Add heat sources
    if len(sources) > 0:
        ax.scatter(sources[:, 0], sources[:, 1], 
                  c=sources[:, 2], s=100, 
                  cmap='viridis', edgecolors='white', linewidth=2)
    
    # ax.set_title('Prediction')
    ax.set_xlabel('X position', fontsize=16)
    ax.set_ylabel('Y position', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 2: Ground Truth
    ax = ax_gt
    im2 = ax.contourf(X, Y, field_gt, levels=100, cmap='hot', vmin=vmin, vmax=vmax)
    
    # Add heat sources
    if len(sources) > 0:
        ax.scatter(sources[:, 0], sources[:, 1], 
                  c=sources[:, 2], s=100, 
                  cmap='viridis', edgecolors='white', linewidth=2)
    
    # ax.set_title('Ground Truth')
    ax.set_xlabel('X position', fontsize=16)
    ax.set_ylabel('Y position', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 3: Absolute Error
    ax = ax_err
    im3 = ax.contourf(X, Y, abs_error, levels=100, cmap='Reds')
    
    # Add heat sources
    if len(sources) > 0:
        ax.scatter(sources[:, 0], sources[:, 1], 
                  c='blue', s=100, 
                  edgecolors='white', linewidth=2, alpha=0.7)
    
    # ax.set_title('Absolute Error')
    ax.set_xlabel('X position', fontsize=16)
    ax.set_ylabel('Y position', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Calculate and display error statistics
    max_error = np.nanmax(abs_error)
    mean_error = np.nanmean(abs_error)
    rmse = np.sqrt(np.nanmean(abs_error**2))
    
    # First, tighten layout so subplots take their final positions
    fig.tight_layout()

    # Retrieve bounding boxes AFTER layout adjustment
    pos_gt = ax_gt.get_position()
    pos_err = ax_err.get_position()

    # Define gap and colorbar width (figure fraction)
    gap_cb = 0.015  # spacing between axis and its colorbar
    gap_axes = 0.02  # extra space between GT colorbar and Error subplot
    cb_width = 0.02

    # Temperature colorbar – right of GT subplot
    left_temp = pos_gt.x1 + gap_cb
    bottom = pos_gt.y0
    height = pos_gt.height

    cbar_ax1 = fig.add_axes([left_temp, bottom, cb_width, height])
    temp_cbar = fig.colorbar(im2, cax=cbar_ax1)
    temp_cbar.set_label('Temperature', rotation=270, labelpad=20, fontsize=16)
    temp_cbar.ax.tick_params(labelsize=16)
    temp_cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # Absolute-error colorbar – right of Error subplot
    left_err = pos_err.x1 + gap_cb + 0.0 # place immediately after Error axis

    cbar_ax2 = fig.add_axes([left_err, bottom, cb_width, height])
    err_cbar = fig.colorbar(im3, cax=cbar_ax2)
    err_cbar.set_label('|Error|', rotation=270, labelpad=20, fontsize=16)
    err_cbar.ax.tick_params(labelsize=16)
    err_cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    # Redraw the canvas to ensure everything renders correctly
    fig.canvas.draw()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heat results plot to {save_path}")
    
    return fig