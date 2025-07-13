#!/usr/bin/env python
"""plot_concentration_2d_data.py
----------------------------------
Visualization script for the chemical concentration dataset using 2D advection-diffusion Green's function.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from pathlib import Path
import matplotlib.colors as colors
from scipy.interpolate import griddata

def load_concentration_dataset(dataset_path: str = "Data/concentration_data/chem_plume_dataset"):
    """Load the concentration dataset from disk and extract parameters."""
    try:
        dataset = load_from_disk(dataset_path)
        
        # Check if this is adaptive mesh or uniform grid
        first_sample = dataset['train'][0]
        is_adaptive = 'grid_coords' in first_sample
        
        if is_adaptive:
            # Adaptive mesh dataset
            grid_coords = np.array(first_sample['grid_coords'])
            field_values = np.array(first_sample['field_values'])
            print(f"Dataset type: Adaptive mesh")
            print(f"  - Sample grid points: {len(grid_coords)}")
            grid_n = 64  # Default visualization grid size for adaptive mesh
        else:
            # Uniform grid dataset
            field_shape = np.array(first_sample['field']).shape
            grid_n = field_shape[0]  # field shape is (grid_n, grid_n, 1)
            print(f"Dataset type: Uniform grid")
            print(f"  - Grid size: {grid_n}×{grid_n}")
        
        # Analyze dataset to extract generation parameters
        train_data = dataset['train']
        source_counts = []
        all_rates = []
        
        # Sample a few examples to get parameter ranges
        sample_size = min(100, len(train_data))
        for i in range(sample_size):
            sources = np.array(train_data[i]['sources'])
            source_counts.append(len(sources))
            if len(sources) > 0:
                all_rates.extend(sources[:, 2].tolist())
        
        n_min = min(source_counts) if source_counts else 0
        n_max = max(source_counts) if source_counts else 0
        rate_min = min(all_rates) if all_rates else 0
        rate_max = max(all_rates) if all_rates else 0
        
        print(f"Dataset parameters detected:")
        print(f"  - Source count range: {n_min} to {n_max}")
        print(f"  - Rate range: {rate_min:.4f} to {rate_max:.4f}")
        print(f"  - Train samples: {len(dataset['train'])}")
        print(f"  - Test samples: {len(dataset['test'])}")
        
        return dataset, grid_n, is_adaptive
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return None, None, None

def plot_single_sample(sample, sample_idx=0, grid_n=64, save_path=None):
    """Plot a single concentration sample with concentration field and source locations."""
    
    # Extract data
    sources = np.array(sample["sources"])  # shape: (n_sources, 3) -> [x, y, rate]
    wind_angle = sample.get("wind_angle", 0.0)  # wind direction in radians (default to 0 if not present)
    
    # Check if this is adaptive mesh or uniform grid
    is_adaptive = 'grid_coords' in sample
    
    # Create visualization grid (always uniform for plotting)
    x = np.linspace(0, 1, grid_n)
    y = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    if is_adaptive:
        # Adaptive mesh data - interpolate to regular grid for visualization
        grid_coords = np.array(sample["grid_coords"])  # (n_points, 2) -> [x, y]
        field_values = np.array(sample["field_values"])  # (n_points,)
        
        # Interpolate to regular grid for visualization
        field = griddata(grid_coords, field_values, (X, Y), method='cubic', fill_value=np.nan)
        field = np.nan_to_num(field, nan=0.0)  # Replace NaN with 0
        
        # Store adaptive points for visualization
        adaptive_coords = grid_coords
    else:
        # Uniform grid data
        field = np.array(sample["field"]).squeeze()  # remove channel dimension (raw concentration field)
        # Update grid to match data resolution
        actual_grid_n = field.shape[0]
        if actual_grid_n != grid_n:
            x = np.linspace(0, 1, actual_grid_n)
            y = np.linspace(0, 1, actual_grid_n)
            X, Y = np.meshgrid(x, y, indexing='xy')
        adaptive_coords = None
    
    # Single subplot layout: [ax_main | cbar_conc | cbar_rate]
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1, 0.04, 0.04],
        left=0.1,
        right=0.9,
        wspace=0.05,  # bring concentration colorbar closer
    )
    ax_main = fig.add_subplot(gs[0, 0])
    cax_conc = fig.add_subplot(gs[0, 1])
    cax_rate = fig.add_subplot(gs[0, 2])
    
    # Plot concentration field - use a colormap suitable for concentration (higher values = more concentrated)
    im = ax_main.contourf(X, Y, field, levels=20, cmap="plasma")
    ax_main.contour(X, Y, field, levels=10, colors="black", alpha=0.3, linewidths=0.5)
    
    # Add chemical sources
    if len(sources) > 0:
        scatter = ax_main.scatter(
            sources[:, 0],
            sources[:, 1],
            c=sources[:, 2],
            s=100,
            cmap="viridis",
            edgecolors="white",
            linewidth=2,
            label="Chemical Sources",
        )
    
    # Add adaptive points visualization for adaptive mesh (subsample for visibility)
    if is_adaptive and adaptive_coords is not None:
        n_show = min(1000, len(adaptive_coords))  # Show at most 1000 points
        indices = np.random.choice(len(adaptive_coords), n_show, replace=False)
        ax_main.scatter(adaptive_coords[indices, 0], adaptive_coords[indices, 1], 
                       c='cyan', s=0.5, alpha=0.4, 
                       label=f'Adaptive points ({n_show}/{len(adaptive_coords)})')
        ax_main.legend(loc='upper right', fontsize=8)
    
    # Concentration colorbar (initial position from GridSpec)
    fig.colorbar(im, cax=cax_conc, label="Concentration")
    # Manually shift the concentration colorbar slightly left to hug the main plot
    conc_pos = cax_conc.get_position()
    shift = 0.085  # fraction of figure width to shift left
    cax_conc.set_position([conc_pos.x0 - shift, conc_pos.y0, conc_pos.width, conc_pos.height])
    
    # Add source rate colorbar (right)
    if len(sources) > 0:
        fig.colorbar(scatter, cax=cax_rate, label="Source Rate")
    else:
        # If no sources, hide the rate colorbar axis
        cax_rate.set_visible(False)
    
    ax_main.set_xlabel("X position")
    ax_main.set_ylabel("Y position")
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect("equal", adjustable="box")
    
    # Add wind direction arrow to show plume direction
    # Calculate arrow start and end points based on wind direction
    arrow_length = 0.1  # length of arrow in axes coordinates
    center_x, center_y = 0.9, 0.95  # center position of arrow
    
    # Wind direction: 0 = +x, π/2 = +y, π = -x, 3π/2 = -y
    wind_x = np.cos(wind_angle)
    wind_y = np.sin(wind_angle)
    
    start_x = center_x - arrow_length/2 * wind_x
    start_y = center_y - arrow_length/2 * wind_y
    end_x = center_x + arrow_length/2 * wind_x
    end_y = center_y + arrow_length/2 * wind_y
    
    ax_main.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                    transform=ax_main.transAxes)
    
    # Wind direction label
    wind_deg = np.degrees(wind_angle) % 360
    if wind_deg == 0:
        wind_label = 'Wind (+x)'
    elif wind_deg == 90:
        wind_label = 'Wind (+y)'
    elif wind_deg == 180:
        wind_label = 'Wind (-x)'
    elif wind_deg == 270:
        wind_label = 'Wind (-y)'
    else:
        wind_label = f'Wind ({wind_deg:.0f}°)'
    
    ax_main.text(center_x, center_y + 0.03, wind_label, transform=ax_main.transAxes, 
                ha='center', va='bottom', color='red', fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    return fig



def main():
    # Hardcoded settings
    dataset_path = "Data/concentration_data/chem_plume_dataset"
    output_dir = Path("Data/concentration_data/plots_uniform")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset and auto-detect parameters
    print(f"Loading dataset from {dataset_path}...")
    dataset, grid_n, is_adaptive = load_concentration_dataset(dataset_path)
    
    if dataset is None or grid_n is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    
    # Plot 3 samples from train set
    print("Plotting 3 samples from training set...")
    for i in range(3):
        sample = dataset['train'][i]
        save_path = output_dir / f"train_sample_{i+1}.png"
        plot_single_sample(sample, i, grid_n, save_path)
        plt.close()  # Close figure to save memory
    
    # Plot 3 samples from test set
    print("Plotting 3 samples from test set...")
    for i in range(3):
        sample = dataset['test'][i]
        save_path = output_dir / f"test_sample_{i+1}.png"
        plot_single_sample(sample, i, grid_n, save_path)
        plt.close()  # Close figure to save memory
    
    print("✅ Generated 6 plots total:")
    print("   - 3 training samples: train_sample_1.png, train_sample_2.png, train_sample_3.png")
    print("   - 3 test samples: test_sample_1.png, test_sample_2.png, test_sample_3.png")
    print(f"   - All saved in: {output_dir}")
    print(f"   - Plots generated for {grid_n}×{grid_n} grid resolution")

if __name__ == "__main__":
    main()
