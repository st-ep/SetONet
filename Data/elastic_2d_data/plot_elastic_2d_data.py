#!/usr/bin/env python
"""
Plot elastic plate dataset samples showing x-displacement fields.
Creates 6 plots total: 3 training samples and 3 test samples.
Each plot shows the x-displacement field with a circular hole in the center.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from datasets import load_from_disk
import json
import os

# Get paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

def load_elastic_data():
    """Load the elastic dataset and normalization stats."""
    try:
        # Load dataset
        dataset_path = os.path.join(current_dir, 'elastic_dataset')
        dataset = load_from_disk(dataset_path)
        
        # Load normalization stats
        stats_path = os.path.join(current_dir, 'elastic_normalization_stats.json')
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        return dataset, stats
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/elastic_2d_data/get_elastic_data.py first")
        return None, None

def denormalize_displacement(s_norm, s_mean, s_std):
    """Denormalize displacement values."""
    return s_norm * s_std + s_mean

def create_circular_mask(x, y, center_x=0.5, center_y=0.5, radius=0.25):
    """Create a circular mask for the hole in the plate."""
    return (x - center_x)**2 + (y - center_y)**2 <= radius**2

def plot_displacement_field(coords, displacement, force_coords, force_values, sample_idx, is_train=True):
    """Plot displacement field with circular hole and forcing function."""
    # Extract coordinates
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Create regular grid for interpolation
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    # Create dense grid for smooth visualization
    xi = np.linspace(x_min, x_max, 400)
    yi = np.linspace(y_min, y_max, 400)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate displacement values onto regular grid using cubic interpolation for smoother results
    Zi = griddata((x_coords, y_coords), displacement, (Xi, Yi), method='cubic')
    
    # Create circular mask for the hole
    hole_mask = create_circular_mask(Xi, Yi)
    
    # Set hole region to NaN so it appears empty
    Zi[hole_mask] = np.nan
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [1, 3]})
    
    # Plot displacement field - on the right (set this first to determine height)
    im = ax2.contourf(Xi, Yi, Zi, levels=100, cmap='RdBu_r')
    
    # Add colorbar with same height as plot
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('x-displacement', rotation=270, labelpad=20)
    
    # Set equal aspect ratio and add axes for displacement plot
    ax2.set_aspect('equal')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Plot forcing function (1D plot) - on the left
    # Extract y-coordinates and force values from force data
    force_y = force_coords[:, 1]  # y-coordinates of force points
    force_x = force_values  # force magnitudes
    
    # Plot force as horizontal line plot with zero at right
    ax1.plot(force_x, force_y, 'b-', linewidth=2)
    
    # Set up force subplot to match the height of displacement subplot
    ax1.set_ylim(y_min, y_max)  # Match y-axis with displacement plot
    ax1.set_xlabel('Force value')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # Get positions and align heights
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    # Match height and y-position of displacement plot to force plot
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, pos1.height])
    
    # Invert x-axis so zero is at the right (force applied from right side)
    ax1.invert_xaxis()
    
    # Add vertical line at zero force
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # First apply tight layout to let matplotlib compute positions
    fig.tight_layout()
    # Redraw to ensure positions are updated
    fig.canvas.draw()
    
    # Retrieve updated positions
    pos2 = ax2.get_position()
    pos1 = ax1.get_position()
    # Make ax1 (force) match the exact vertical size of ax2
    ax1.set_position([pos1.x0, pos2.y0, pos1.width, pos2.height])
    
    return fig

def plot_elastic_samples():
    """Create plots for 3 training and 3 test samples."""
    # Load data
    dataset, stats = load_elastic_data()
    if dataset is None:
        return
    
    # Sample indices to plot
    train_indices = [0, 100, 500]  # 3 different training samples
    test_indices = [0, 25, 75]     # 3 different test samples
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(current_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot training samples
    print("Plotting training samples...")
    for i, idx in enumerate(train_indices):
        sample = dataset['train'][idx]
        
        # Get coordinates and normalized displacement
        coords = np.array(sample['Y'])  # Coordinates (already in original scale)
        displacement = np.array(sample['s'])  # Normalized displacement
        
        # Denormalize displacement
        displacement = displacement * stats['s_std'] + stats['s_mean']
        
        # Get force coordinates and normalized values
        force_coords = np.array(sample['X'])  # Force coordinates
        force_values = np.array(sample['u'])  # Normalized force values
        # Denormalize force values
        force_values = force_values * stats['u_std'] + stats['u_mean']
        
        # Create plot
        fig = plot_displacement_field(coords, displacement, force_coords, force_values, idx, is_train=True)
        
        # Save plot
        plt.savefig(f'{plots_dir}/train_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved train_sample_{i+1}.png")
    
    # Plot test samples
    print("Plotting test samples...")
    for i, idx in enumerate(test_indices):
        sample = dataset['test'][idx]
        
        # Get coordinates and normalized displacement
        coords = np.array(sample['Y'])  # Coordinates (already in original scale)
        displacement = np.array(sample['s'])  # Normalized displacement
        
        # Denormalize displacement
        displacement = displacement * stats['s_std'] + stats['s_mean']
        
        # Get force coordinates and normalized values
        force_coords = np.array(sample['X'])  # Force coordinates
        force_values = np.array(sample['u'])  # Normalized force values
        # Denormalize force values
        force_values = force_values * stats['u_std'] + stats['u_mean']
        
        # Create plot
        fig = plot_displacement_field(coords, displacement, force_coords, force_values, idx, is_train=False)
        
        # Save plot
        plt.savefig(f'{plots_dir}/test_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved test_sample_{i+1}.png")
    
    print("All plots saved successfully!")

if __name__ == "__main__":
    plot_elastic_samples()
