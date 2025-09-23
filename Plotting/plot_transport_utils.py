#!/usr/bin/env python
"""plot_transport_utils.py
----------------------------------
Plotting utilities for SetONet transport results.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_transport_results(model, dataset, transport_dataset, device, sample_idx=0, 
                          save_path=None, dataset_split="test"):
    """
    Plot transport results showing prediction, ground truth, and absolute error.
    
    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset
        transport_dataset: TransportDataset wrapper
        device: Device for computations
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
        dataset_split: 'train' or 'test'
    """
    model.eval()
    
    # Get sample data
    data_split = dataset[dataset_split]
    if sample_idx >= len(data_split):
        sample_idx = 0
    
    sample = data_split[sample_idx]
    
    # Extract ground truth data
    source_points = torch.tensor(np.array(sample['source_points']), device=device, dtype=torch.float32)
    velocity_field_gt = torch.tensor(np.array(sample['velocity_field']), device=device, dtype=torch.float32)
    grid_coords = torch.tensor(np.array(sample['grid_coords']), device=device, dtype=torch.float32)
    
    # Prepare input for model
    source_coords = source_points.unsqueeze(0)  # (1, n_sources, 2)
    source_weights = torch.ones(1, source_points.shape[0], 1, device=device, dtype=torch.float32)  # (1, n_sources, 1)
    target_coords = grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
    
    # Get model prediction
    with torch.no_grad():
        pred_velocity = model(source_coords, source_weights, target_coords)  # (1, n_grid_points, 2)
    
    # Convert to numpy and reshape
    grid_h, grid_w = velocity_field_gt.shape[0], velocity_field_gt.shape[1]
    
    # Ground truth velocity field
    V_gt = velocity_field_gt.cpu().numpy()  # (grid_h, grid_w, 2)
    
    # Predicted velocity field 
    V_pred = pred_velocity.squeeze(0).reshape(grid_h, grid_w, 2).cpu().numpy()  # (grid_h, grid_w, 2)
    
    # Absolute error
    V_error = np.abs(V_pred - V_gt)  # (grid_h, grid_w, 2)
    
    # Calculate velocity magnitudes
    V_gt_mag = np.sqrt(V_gt[:, :, 0]**2 + V_gt[:, :, 1]**2)
    V_pred_mag = np.sqrt(V_pred[:, :, 0]**2 + V_pred[:, :, 1]**2)
    V_error_mag = np.sqrt(V_error[:, :, 0]**2 + V_error[:, :, 1]**2)
    
    # Prepare grid coordinates for plotting
    grid_coords_np = grid_coords.cpu().numpy()
    xx = grid_coords_np[:, 0].reshape(grid_h, grid_w)
    yy = grid_coords_np[:, 1].reshape(grid_h, grid_w)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Determine shared color scale for prediction and ground truth
    vmin_vel = min(V_gt_mag.min(), V_pred_mag.min())
    vmax_vel = max(V_gt_mag.max(), V_pred_mag.max())
    
    # Plot 1: Ground Truth (ONLY magnitude, no vectors)
    ax1 = axes[0]
    im1 = ax1.contourf(xx, yy, V_gt_mag, levels=20, cmap='viridis', vmin=vmin_vel, vmax=vmax_vel)
    ax1.contour(xx, yy, V_gt_mag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1, label='Velocity Magnitude')
    cbar1.set_label('Velocity Magnitude', fontsize=18)
    cbar1.ax.tick_params(labelsize=18)
    
    ax1.set_xlim(xx.min(), xx.max())
    ax1.set_ylim(yy.min(), yy.max())
    ax1.set_xlabel('x', fontsize=18)
    ax1.set_ylabel('y', fontsize=18)
    ax1.set_aspect('equal')
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # Plot 2: Prediction (ONLY magnitude, no vectors)
    ax2 = axes[1]
    im2 = ax2.contourf(xx, yy, V_pred_mag, levels=20, cmap='viridis', vmin=vmin_vel, vmax=vmax_vel)
    ax2.contour(xx, yy, V_pred_mag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = plt.colorbar(im2, cax=cax2, label='Velocity Magnitude')
    cbar2.set_label('Velocity Magnitude', fontsize=18)
    cbar2.ax.tick_params(labelsize=18)
    
    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())
    ax2.set_xlabel('x', fontsize=18)
    ax2.set_ylabel('y', fontsize=18)
    ax2.set_aspect('equal')
    ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # Plot 3: Absolute Error
    ax3 = axes[2]
    im3 = ax3.contourf(xx, yy, V_error_mag, levels=20, cmap='plasma')
    ax3.contour(xx, yy, V_error_mag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    cbar3 = plt.colorbar(im3, cax=cax3, label='Error Magnitude')
    cbar3.set_label('Error Magnitude', fontsize=18)
    cbar3.ax.tick_params(labelsize=18)
    
    ax3.set_xlim(xx.min(), xx.max())
    ax3.set_ylim(yy.min(), yy.max())
    ax3.set_xlabel('x', fontsize=18)
    ax3.set_ylabel('y', fontsize=18)
    ax3.set_aspect('equal')
    ax3.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved transport results plot to {save_path}")
    
    model.train()
    return fig

def plot_transport_vectors_and_maps(model, dataset, transport_dataset, device, sample_idx=0, 
                                   save_path=None, dataset_split="test"):
    """
    Plot transport vectors and maps showing predicted vs ground truth.
    
    Creates 4 subplots in one row:
    1. Predicted velocity vectors
    2. Ground Truth velocity vectors  
    3. Predicted transport map
    4. Ground Truth transport map
    
    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset
        transport_dataset: TransportDataset wrapper
        device: Device for computations
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
        dataset_split: 'train' or 'test'
    """
    model.eval()
    
    # Get sample data
    data_split = dataset[dataset_split]
    if sample_idx >= len(data_split):
        sample_idx = 0
    
    sample = data_split[sample_idx]
    
    # Extract ground truth data
    source_points = torch.tensor(np.array(sample['source_points']), device=device, dtype=torch.float32)
    target_points = torch.tensor(np.array(sample['target_points']), device=device, dtype=torch.float32)
    velocity_field_gt = torch.tensor(np.array(sample['velocity_field']), device=device, dtype=torch.float32)
    grid_coords = torch.tensor(np.array(sample['grid_coords']), device=device, dtype=torch.float32)
    
    # Prepare input for model - velocity field prediction
    source_coords = source_points.unsqueeze(0)  # (1, n_sources, 2)
    source_weights = torch.ones(1, source_points.shape[0], 1, device=device, dtype=torch.float32)
    target_coords = grid_coords.unsqueeze(0)  # (1, n_grid_points, 2)
    
    # Get model prediction for velocity field
    with torch.no_grad():
        pred_velocity_field = model(source_coords, source_weights, target_coords)  # (1, n_grid_points, 2)
    
    # Also get velocity predictions at source points for transport map
    source_coords_for_transport = source_points.unsqueeze(0)  # (1, n_sources, 2)
    with torch.no_grad():
        pred_velocity_at_sources = model(source_coords, source_weights, source_coords_for_transport)  # (1, n_sources, 2)
    
    # Convert to numpy
    grid_h, grid_w = velocity_field_gt.shape[0], velocity_field_gt.shape[1]
    
    # Ground truth data
    V_gt = velocity_field_gt.cpu().numpy()  # (grid_h, grid_w, 2)
    source_pts_np = source_points.cpu().numpy()  # (n_sources, 2)
    target_pts_np = target_points.cpu().numpy()  # (n_sources, 2)
    
    # Predicted data
    V_pred = pred_velocity_field.squeeze(0).reshape(grid_h, grid_w, 2).cpu().numpy()  # (grid_h, grid_w, 2)
    pred_velocity_at_sources_np = pred_velocity_at_sources.squeeze(0).cpu().numpy()  # (n_sources, 2)
    pred_target_pts_np = source_pts_np + pred_velocity_at_sources_np  # Predicted target points
    
    # Prepare grid coordinates for plotting
    grid_coords_np = grid_coords.cpu().numpy()
    xx = grid_coords_np[:, 0].reshape(grid_h, grid_w)
    yy = grid_coords_np[:, 1].reshape(grid_h, grid_w)
    
    # Create figure with 4 subplots in one row
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Common settings for all plots - use actual grid extent with small padding
    padding = 0.2
    xlim = (xx.min() - padding, xx.max() + padding)
    ylim = (yy.min() - padding, yy.max() + padding)
    
    # Plot 1: Predicted Velocity Vectors (ONLY vectors, no magnitude)
    ax1 = axes[0]
    
    # Create indices that always include boundaries
    n_vectors = 10  # Number of vectors along each axis
    x_indices = np.linspace(0, grid_h-1, n_vectors, dtype=int)
    y_indices = np.linspace(0, grid_w-1, n_vectors, dtype=int)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices, indexing='ij')
    
    # Plot ONLY velocity vectors on clean background
    ax1.quiver(xx[x_grid, y_grid], yy[x_grid, y_grid],
              V_pred[x_grid, y_grid, 0], V_pred[x_grid, y_grid, 1],
              color='orange', angles='xy', scale_units='xy', scale=1, alpha=0.8, width=0.003)
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('x', fontsize=18)
    ax1.set_ylabel('y', fontsize=18)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # Plot 2: Ground Truth Velocity Vectors (ONLY vectors, no magnitude)
    ax2 = axes[1]
    
    # Plot ONLY velocity vectors on clean background (using same indices)
    ax2.quiver(xx[x_grid, y_grid], yy[x_grid, y_grid],
              V_gt[x_grid, y_grid, 0], V_gt[x_grid, y_grid, 1],
              color='orange', angles='xy', scale_units='xy', scale=1, alpha=0.8, width=0.003)
    
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('x', fontsize=18)
    ax2.set_ylabel('y', fontsize=18)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    
    # Plot 3: Predicted Transport Map
    ax3 = axes[2]
    
    # Plot source points
    ax3.scatter(source_pts_np[:, 0], source_pts_np[:, 1], c='blue', s=15, alpha=0.7, label='Source points')
    # Plot predicted target points
    ax3.scatter(pred_target_pts_np[:, 0], pred_target_pts_np[:, 1], c='red', s=15, alpha=0.7, label='Predicted targets')
    
    # Draw transport trajectories as dashed lines (showing all points)
    n_arrows = len(source_pts_np)  # Use all points
    indices = np.arange(len(source_pts_np))  # All indices
    for i in indices:
        ax3.plot([source_pts_np[i, 0], pred_target_pts_np[i, 0]], 
                [source_pts_np[i, 1], pred_target_pts_np[i, 1]],
                '--', color='gray', alpha=0.6, linewidth=1)
    
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_xlabel('x', fontsize=18)
    ax3.set_ylabel('y', fontsize=18)
    ax3.legend(fontsize=18)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.tick_params(axis='both', which='major', labelsize=18)
    
    # Plot 4: Ground Truth Transport Map
    ax4 = axes[3]
    
    # Plot source points
    ax4.scatter(source_pts_np[:, 0], source_pts_np[:, 1], c='blue', s=15, alpha=0.7, label='Source points')
    # Plot ground truth target points
    ax4.scatter(target_pts_np[:, 0], target_pts_np[:, 1], c='green', s=15, alpha=0.7, label='True targets')
    
    # Draw transport trajectories as dashed lines
    for i in indices:  # Use same indices as predicted for fair comparison
        ax4.plot([source_pts_np[i, 0], target_pts_np[i, 0]], 
                [source_pts_np[i, 1], target_pts_np[i, 1]],
                '--', color='gray', alpha=0.6, linewidth=1)
    
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_xlabel('x', fontsize=18)
    ax4.set_ylabel('y', fontsize=18)
    ax4.legend(fontsize=18)
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    ax4.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved transport vectors and maps plot to {save_path}")
    
    model.train()
    return fig


