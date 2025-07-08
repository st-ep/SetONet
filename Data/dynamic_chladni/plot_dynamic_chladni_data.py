#!/usr/bin/env python
"""plot_dynamic_chladni_data.py
----------------------------------
Visualization script for the Dynamic Chladni dataset with point cloud forces.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from pathlib import Path
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dynamic_chladni_dataset import load_dynamic_chladni_dataset

def plot_single_sample(sample, sample_idx=0, split_name='train', grid_n=64, save_path=None):
    """Plot a single dynamic Chladni sample with displacement field and force locations."""
    
    # Extract data
    sources = np.array(sample["sources"])  # shape: (n_forces, 3) -> [x_norm, y_norm, force_magnitude]
    field = np.array(sample["field"]).squeeze()  # remove channel dimension
    
    # Create coordinate grids (normalized [0,1])
    x = np.linspace(0, 1, grid_n)
    y = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot displacement field as filled contours
    contour_filled = ax.contourf(X, Y, field.T, levels=20, cmap='RdBu_r', alpha=0.8)
    
    # Create divider for proper colorbar sizing
    divider = make_axes_locatable(ax)
    
    # Add colorbar for displacement (same height as subplot)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(contour_filled, cax=cax1, label='Displacement')
    
    # Plot zero-level contours (Chladni nodal lines) 
    contour_lines = ax.contour(X, Y, field.T, levels=[0], 
                              colors='black', linewidths=3, alpha=0.9)
    
    # Plot force locations
    if len(sources) > 0:
        scatter = ax.scatter(sources[:, 0], sources[:, 1], 
                           c=sources[:, 2], s=150, 
                           cmap='viridis', edgecolors='white', linewidth=2,
                           label=f'{len(sources)} Force Points', zorder=5)
        
        # Add second colorbar for force magnitudes (same height as subplot)
        cax2 = divider.append_axes("right", size="5%", pad=0.3)
        cbar2 = plt.colorbar(scatter, cax=cax2, label='Force Magnitude')
    
    ax.set_xlabel('X position (normalized)', fontsize=12)
    ax.set_ylabel('Y position (normalized)', fontsize=12)
    ax.set_title(f'Dynamic Chladni {split_name.capitalize()} Sample #{sample_idx+1}\n'
                f'{len(sources)} forces, Grid: {grid_n}×{grid_n}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    if len(sources) > 0:
        ax.legend(loc='upper right', fontsize=10)
    
    # Add displacement statistics as text
    disp_min, disp_max = field.min(), field.max()
    disp_range = disp_max - disp_min
    force_range = f"{sources[:, 2].min():.3f} - {sources[:, 2].max():.3f}" if len(sources) > 0 else "N/A"
    
    stats_text = f'Displacement: [{disp_min:.4f}, {disp_max:.4f}] (range: {disp_range:.4f})\n'
    stats_text += f'Force Range: {force_range}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
           verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    return fig



def main():
    # Dataset settings
    dataset_path = "Data/dynamic_chladni/dynamic_chladni_dataset"  # Path relative to project root
    output_dir = Path("plots")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset, wrapper = load_dynamic_chladni_dataset(dataset_path)
    
    if dataset is None or wrapper is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    
    # Get grid size from wrapper
    grid_n = wrapper.grid_size
    print(f"Grid resolution: {grid_n}×{grid_n}")
    
    # Plot 3 samples from train set
    print("Plotting 3 samples from training set...")
    for i in range(min(3, len(dataset['train']))):
        sample = dataset['train'][i]
        save_path = output_dir / f"train_sample_{i+1}.png"
        plot_single_sample(sample, i, 'train', grid_n, save_path)
        plt.close()  # Close figure to save memory
    
    # Plot 3 samples from test set
    print("Plotting 3 samples from test set...")
    for i in range(min(3, len(dataset['test']))):
        sample = dataset['test'][i]
        save_path = output_dir / f"test_sample_{i+1}.png"
        plot_single_sample(sample, i, 'test', grid_n, save_path)
        plt.close()  # Close figure to save memory
    
    print("✅ Generated plots:")
    print("   - 3 training samples: train_sample_1.png, train_sample_2.png, train_sample_3.png")
    print("   - 3 test samples: test_sample_1.png, test_sample_2.png, test_sample_3.png")
    print(f"   - All saved in: {output_dir}")
    print(f"   - Grid resolution: {grid_n}×{grid_n}")

if __name__ == "__main__":
    main()
