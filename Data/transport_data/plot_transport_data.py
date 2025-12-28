#!/usr/bin/env python
"""plot_transport_data.py
----------------------------------
Visualization script for the optimal transport dataset.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as colors
from datasets import load_from_disk
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _compute_grid_coords(sample):
    if "grid_coords" in sample:
        return np.array(sample["grid_coords"], dtype=np.float32)

    domain_size = float(sample.get("domain_size", 5.0))
    velocity_field = np.array(sample["velocity_field"])
    grid_h, grid_w = velocity_field.shape[0], velocity_field.shape[1]
    xs = np.linspace(-domain_size, domain_size, grid_h, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_w, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

def load_transport_dataset(dataset_path: str = "Data/transport_data/transport_dataset"):
    """Load the transport dataset from disk and extract parameters."""
    try:
        dataset = load_from_disk(dataset_path)
        train_data = dataset['train']
        
        print(f"Dataset loaded successfully:")
        print(f"  - Number of train samples: {len(train_data)}")
        print(f"  - Number of test samples: {len(dataset['test'])}")
        
        # Extract sample info
        sample_0 = train_data[0]
        source_points = np.array(sample_0['source_points'])
        target_points = np.array(sample_0['target_points'])
        velocity_field = np.array(sample_0['velocity_field'])
        grid_coords = _compute_grid_coords(sample_0)
        
        print(f"  - Source points per sample: {len(source_points)}")
        print(f"  - Target points per sample: {len(target_points)}")
        print(f"  - Velocity field grid: {velocity_field.shape[0]}×{velocity_field.shape[1]}")
        print(f"  - Grid points: {len(grid_coords)}")
        
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return None

def plot_single_sample(dataset, sample_idx=0, split='train', save_path=None):
    """Plot a single transport sample with source/target points and velocity field."""
    
    # Extract data for this sample
    sample = dataset[split][sample_idx]
    X = np.array(sample['source_points'])  # source points (256, 2)
    Y = np.array(sample['target_points'])  # target points (256, 2)
    V = np.array(sample['velocity_field'])  # velocity field (80, 80, 2)
    grid_pts = _compute_grid_coords(sample)  # grid coordinates (n_grid_points, 2)
    
    # Reshape grid for plotting
    grid_size = V.shape[0]  # Get actual grid size from velocity field
    xx = grid_pts[:, 0].reshape(grid_size, grid_size)
    yy = grid_pts[:, 1].reshape(grid_size, grid_size)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Source and target densities with transport map
    ax1 = axes[0]
    
    # Plot source points (red) and target points (blue)
    ax1.scatter(X[:, 0], X[:, 1], c='red', s=20, alpha=0.7, label='Source points')
    ax1.scatter(Y[:, 0], Y[:, 1], c='blue', s=20, alpha=0.7, label='Target points')
    
    # Draw transport arrows (subsample for visibility)
    n_arrows = min(50, len(X))
    indices = np.random.choice(len(X), n_arrows, replace=False)
    for i in indices:
        ax1.arrow(X[i, 0], X[i, 1], 
                 Y[i, 0] - X[i, 0], Y[i, 1] - X[i, 1],
                 head_width=0.02, head_length=0.03, 
                 fc='gray', ec='gray', alpha=0.6, length_includes_head=True)
    
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Sample {sample_idx+1}: Transport Map')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Velocity field (vector field)
    ax2 = axes[1]
    
    # Plot velocity field with quiver (subsample for visibility)
    skip = 4  # plot every 4th vector to avoid clutter
    ax2.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
              V[::skip, ::skip, 0], V[::skip, ::skip, 1],
              color='red', angles='xy', scale_units='xy', scale=1, alpha=0.8)
    
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'Sample {sample_idx+1}: Velocity Field')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Velocity magnitude as heatmap
    ax3 = axes[2]
    
    # Compute velocity magnitude
    V_mag = np.sqrt(V[:, :, 0]**2 + V[:, :, 1]**2)
    
    # Plot velocity magnitude as contour/heatmap
    im = ax3.contourf(xx, yy, V_mag, levels=20, cmap='viridis')
    ax3.contour(xx, yy, V_mag, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add colorbar with same height as subplot
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label='Velocity Magnitude')
    
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-5, 5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title(f'Sample {sample_idx+1}: Velocity Magnitude')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    return fig

def plot_transport_statistics(dataset, save_path=None):
    """Plot statistics of the transport dataset."""
    
    train_data = dataset['train']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Distribution of transport distances
    ax1 = axes[0, 0]
    transport_distances = []
    for i in range(len(train_data)):
        sample = train_data[i]
        X = np.array(sample['source_points'])
        Y = np.array(sample['target_points'])
        distances = np.linalg.norm(Y - X, axis=1)
        transport_distances.extend(distances)
    
    ax1.hist(transport_distances, bins=50, alpha=0.7, density=True)
    ax1.set_xlabel('Transport Distance')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of Transport Distances')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of velocity magnitudes
    ax2 = axes[0, 1]
    velocity_magnitudes = []
    for i in range(len(train_data)):
        sample = train_data[i]
        V = np.array(sample['velocity_field'])
        V_mag = np.sqrt(V[:, :, 0]**2 + V[:, :, 1]**2)
        velocity_magnitudes.extend(V_mag.flatten())
    
    ax2.hist(velocity_magnitudes, bins=50, alpha=0.7, density=True)
    ax2.set_xlabel('Velocity Magnitude')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Velocity Magnitudes')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Source distribution centers
    ax3 = axes[1, 0]
    mean1_all = []
    mean2_all = []
    for i in range(len(train_data)):
        sample = train_data[i]
        params = sample['source_params']
        mean1_all.append(params['mean1'])
        mean2_all.append(params['mean2'])
    
    mean1_all = np.array(mean1_all)
    mean2_all = np.array(mean2_all)
    
    ax3.scatter(mean1_all[:, 0], mean1_all[:, 1], c='red', alpha=0.6, s=30, label='Gaussian 1 centers')
    ax3.scatter(mean2_all[:, 0], mean2_all[:, 1], c='blue', alpha=0.6, s=30, label='Gaussian 2 centers')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Source Distribution Centers')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Average velocity field across all samples
    ax4 = axes[1, 1]
    V_all = []
    for i in range(len(train_data)):
        sample = train_data[i]
        V = np.array(sample['velocity_field'])
        V_all.append(V)
    
    V_mean = np.mean(V_all, axis=0)  # Average over all samples
    sample_0 = train_data[0]
    grid_pts = _compute_grid_coords(sample_0)
    grid_size = V_mean.shape[0]
    xx = grid_pts[:, 0].reshape(grid_size, grid_size)
    yy = grid_pts[:, 1].reshape(grid_size, grid_size)
    
    V_mag_mean = np.sqrt(V_mean[:, :, 0]**2 + V_mean[:, :, 1]**2)
    im = ax4.contourf(xx, yy, V_mag_mean, levels=20, cmap='plasma')
    
    # Add some velocity vectors
    skip = 8
    ax4.quiver(xx[::skip, ::skip], yy[::skip, ::skip],
              V_mean[::skip, ::skip, 0], V_mean[::skip, ::skip, 1],
              color='white', angles='xy', scale_units='xy', scale=1, alpha=0.8)
    
    # Add colorbar with same height as subplot
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax4, label='Mean Velocity Magnitude')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Average Velocity Field')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved statistics plot to {save_path}")
    
    return fig

def main():
    # Hardcoded settings
    dataset_path = "Data/transport_data/transport_dataset"
    output_dir = Path("Data/transport_data/plots")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_transport_dataset(dataset_path)
    
    if dataset is None:
        print("Failed to load dataset. Make sure to run generate_transport_data.py first.")
        return
    
    print(f"Dataset loaded with {len(dataset['train'])} train samples and {len(dataset['test'])} test samples")
    
    # Plot sample transport maps from train set (up to 10)
    n_train_plots = min(10, len(dataset['train']))
    print(f"Plotting {n_train_plots} sample transport maps from train set...")
    for i in range(n_train_plots):
        save_path = output_dir / f"train_transport_sample_{i+1}.png"
        plot_single_sample(dataset, i, 'train', save_path)
        plt.close()  # Close figure to save memory
    
    # Plot sample transport maps from test set (up to 10)
    n_test_plots = min(10, len(dataset['test']))
    print(f"Plotting {n_test_plots} sample transport maps from test set...")
    for i in range(n_test_plots):
        save_path = output_dir / f"test_transport_sample_{i+1}.png"
        plot_single_sample(dataset, i, 'test', save_path)
        plt.close()  # Close figure to save memory
    
    # Plot dataset statistics
    print("Plotting dataset statistics...")
    stats_path = output_dir / "transport_statistics.png"
    plot_transport_statistics(dataset, stats_path)
    plt.close()
    
    total_plots = n_train_plots + n_test_plots + 1
    print(f"✅ Generated {total_plots} plots total:")
    print(f"   - {n_train_plots} train transport samples")
    print(f"   - {n_test_plots} test transport samples") 
    print("   - 1 statistics plot: transport_statistics.png")
    print(f"   - All saved in: {output_dir}")

if __name__ == "__main__":
    main() 
