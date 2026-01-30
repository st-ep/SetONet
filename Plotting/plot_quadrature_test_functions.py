#!/usr/bin/env python3
"""
Visualize how Quadrature Test Functions learn SPATIAL GEOMETRY.

This script demonstrates that learned test functions φ_k(x,y) capture
geometric and spatial structure, not just random patterns.

Key visualizations:
1. Spatial receptive fields - where each test function activates
2. Dominant function map - geometric partitioning of the domain
3. Integration with real data - shows spatial correlation

Usage:
    python Plotting/plot_quadrature_test_functions.py \
        --checkpoint_path logs_all/concentration_2d/setonet_quadrature/seed_0/concentration2d_setonet_model.pth
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from Models.SetONet import SetONet


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize geometric learning in quadrature test functions"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/quadrature_viz",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=100,
        help="Number of points per dimension in 2D evaluation grid",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Test sample index to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation",
    )
    return parser.parse_args()


def load_model_and_config(checkpoint_path, device):
    """Load trained SetONet model and configuration."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config_path = checkpoint_path.parent / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        full_config = json.load(f)

    arch = full_config.get("model_architecture", {})

    activation_map = {
        "relu": torch.nn.ReLU,
        "gelu": torch.nn.GELU,
        "tanh": torch.nn.Tanh,
    }
    activation_fn = activation_map.get(arch.get("activation_fn", "relu").lower(), torch.nn.ReLU)

    model = SetONet(
        input_size_src=arch.get("input_size_src", 2),
        output_size_src=arch.get("output_size_src", 1),
        input_size_tgt=arch.get("input_size_tgt", 2),
        output_size_tgt=arch.get("output_size_tgt", 1),
        p=arch.get("son_p_dim", 32),
        phi_hidden_size=arch.get("son_phi_hidden", 256),
        rho_hidden_size=arch.get("son_rho_hidden", 200),
        trunk_hidden_size=arch.get("son_trunk_hidden", 256),
        n_trunk_layers=arch.get("son_n_trunk_layers", 4),
        activation_fn=activation_fn,
        use_deeponet_bias=arch.get("use_deeponet_bias", True),
        phi_output_size=arch.get("son_phi_output_size", 32),
        pos_encoding_type=arch.get("pos_encoding_type", "sinusoidal"),
        pos_encoding_dim=arch.get("pos_encoding_dim", 64),
        pos_encoding_max_freq=arch.get("pos_encoding_max_freq", 0.1),
        use_positional_encoding=arch.get("use_positional_encoding", True),
        aggregation_type=arch.get("son_aggregation", "attention"),
        attention_n_tokens=arch.get("attention_n_tokens", 1),
        branch_head_type="quadrature",
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    if not hasattr(model, "quadrature_head") or model.quadrature_head is None:
        raise ValueError("Model does not have a quadrature head")

    print(f"Loaded SetONet model with quadrature head")
    print(f"  p (latent dim): {model.p}")
    print(f"  dk (key dim): {model.quadrature_head.dk}")
    print(f"  dv (value dim): {model.quadrature_head.dv}")

    eval_results = full_config.get("evaluation_results", {})
    config = {
        "arch": arch,
        "test_l2_error": eval_results.get("test_relative_l2_error"),
        "test_mse": eval_results.get("test_mse_loss"),
    }

    return model, config


def load_sample_data(sample_idx=0):
    """Load a test sample from the concentration dataset."""
    data_path = project_root / "Data" / "concentration_data" / "chem_plume_dataset"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    dataset = load_from_disk(str(data_path))
    test_data = dataset["test"]

    if sample_idx >= len(test_data):
        sample_idx = 0
        print(f"Warning: sample_idx out of range, using sample 0")

    sample = test_data[sample_idx]

    # Extract sources
    sources = np.array(sample["sources"])  # (n_sources, 3) - [x, y, rate]
    source_coords = sources[:, :2]
    source_rates = sources[:, 2]

    # Extract field
    is_adaptive = 'grid_coords' in sample

    if is_adaptive:
        # Adaptive mesh: need coords + values for interpolation
        grid_coords = np.array(sample['grid_coords'])
        field_values = np.array(sample['field_values'])
        field_2d = None  # Will interpolate later
    else:
        # Uniform grid: use field directly (no interpolation)
        field = np.array(sample['field']).squeeze()  # (grid_n, grid_n)
        field_2d = field  # Store 2D field directly
        grid_coords = None
        field_values = None

    return {
        'source_coords': source_coords,
        'source_rates': source_rates,
        'is_adaptive': is_adaptive,
        'field_2d': field_2d,  # For uniform grid
        'grid_coords': grid_coords,  # For adaptive grid
        'field_values': field_values,  # For adaptive grid
    }


def compute_test_functions_2d(model, x_grid, y_grid):
    """Compute test function values φ_k(x,y) over 2D grid."""
    device = next(model.parameters()).device
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)

    # Flatten and combine coordinates
    x_flat = x_grid.flatten().unsqueeze(-1)
    y_flat = y_grid.flatten().unsqueeze(-1)
    coords = torch.cat([x_flat, y_flat], dim=-1).unsqueeze(0)  # (1, N*N, 2)

    with torch.no_grad():
        # Apply positional encoding
        x_enc = model._sinusoidal_encoding(coords)  # (1, N*N, pos_enc_dim)

        # Compute test functions
        Q = model.quadrature_head.query_tokens  # (1, p, dk)
        K = model.quadrature_head.key_net(x_enc)  # (1, N*N, dk)

        scores = torch.einsum('bpk,bnk->bpn', Q, K) / math.sqrt(model.quadrature_head.dk)

        if model.quadrature_head.learn_temperature:
            tau = torch.exp(model.quadrature_head.log_tau) + model.quadrature_head.eps
            scores = scores / tau

        Phi = F.softplus(scores).squeeze(0)  # (p, N*N)
        Phi = Phi / Phi.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # Reshape to 2D
    grid_size = x_grid.shape[0]
    Phi_2d = Phi.cpu().numpy().reshape(-1, grid_size, grid_size)

    return Phi_2d, x_grid.cpu().numpy(), y_grid.cpu().numpy()


def compute_spatial_geometry_analysis(model, sample_data, x_grid, y_grid, Phi):
    """
    Compute spatial geometry analysis showing how test functions learn structure.

    Returns:
        dict with geometric analysis results
    """
    device = next(model.parameters()).device
    p = Phi.shape[0]

    # Prepare sample data
    source_coords = torch.tensor(sample_data['source_coords'], dtype=torch.float32, device=device)
    source_rates = torch.tensor(sample_data['source_rates'], dtype=torch.float32, device=device).unsqueeze(-1)

    xs_batch = source_coords.unsqueeze(0)  # (1, N_src, 2)
    us_batch = source_rates.unsqueeze(0)   # (1, N_src, 1)

    # Prepare evaluation grid
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    eval_coords = torch.tensor(
        np.stack([x_flat, y_flat], axis=1),
        dtype=torch.float32, device=device
    ).unsqueeze(0)  # (1, N*N, 2)

    with torch.no_grad():
        # 1) Evaluate test functions at source locations
        x_enc_sources = model._sinusoidal_encoding(xs_batch)
        Q = model.quadrature_head.query_tokens
        K_sources = model.quadrature_head.key_net(x_enc_sources)

        scores_sources = torch.einsum('bpk,bnk->bpn', Q, K_sources) / math.sqrt(model.quadrature_head.dk)
        Phi_sources = F.softplus(scores_sources).squeeze(0)  # (p, N_src)
        Phi_sources = Phi_sources / Phi_sources.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # 2) Compute weighted activations (test functions weighted by source rates)
        # This shows which test functions respond to which source patterns
        source_weights = source_rates.squeeze(-1)  # (N_src,)
        weighted_activations = torch.einsum('pn,n->p', Phi_sources, source_weights)  # (p,)
        weighted_activations = weighted_activations / weighted_activations.sum()

        # 3) Run forward pass
        prediction = model(xs_batch, us_batch, eval_coords).squeeze().cpu().numpy()

    # Reshape prediction
    grid_size = x_grid.shape[0]
    prediction_2d = prediction.reshape(grid_size, grid_size)

    # Get ground truth
    if sample_data['is_adaptive']:
        # Adaptive mesh: interpolate to regular grid
        ground_truth_2d = griddata(
            sample_data['grid_coords'],
            sample_data['field_values'],
            (x_grid, y_grid),
            method='cubic',
            fill_value=0.0
        )
    else:
        # Uniform grid: use field directly or interpolate if resolution differs
        field_2d = sample_data['field_2d']
        if field_2d.shape == (grid_size, grid_size):
            # Same resolution: use directly
            ground_truth_2d = field_2d
        else:
            # Different resolution: need to interpolate
            # Create coordinate grid for original field
            orig_size = field_2d.shape[0]
            x_orig = np.linspace(0, 1, orig_size)
            y_orig = np.linspace(0, 1, orig_size)
            X_orig, Y_orig = np.meshgrid(x_orig, y_orig, indexing='xy')
            orig_coords = np.stack([X_orig.flatten(), Y_orig.flatten()], axis=1)
            orig_values = field_2d.flatten()

            ground_truth_2d = griddata(
                orig_coords,
                orig_values,
                (x_grid, y_grid),
                method='cubic',
                fill_value=0.0
            )

    # Compute dominant test function at each location
    dominant_function = np.argmax(Phi, axis=0)  # (grid_size, grid_size)

    # Compute "mass center" of each test function (geometric centroid)
    mass_centers = np.zeros((p, 2))
    for k in range(p):
        mass = Phi[k].sum()
        if mass > 0:
            mass_centers[k, 0] = (Phi[k] * x_grid).sum() / mass
            mass_centers[k, 1] = (Phi[k] * y_grid).sum() / mass

    # Compute effective "radius" (spread) of each test function
    radii = np.zeros(p)
    for k in range(p):
        cx, cy = mass_centers[k]
        dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
        radii[k] = np.sqrt((Phi[k] * dist_sq).sum() / (Phi[k].sum() + 1e-8))

    return {
        'weighted_activations': weighted_activations.cpu().numpy(),
        'dominant_function': dominant_function,
        'mass_centers': mass_centers,
        'radii': radii,
        'prediction_2d': prediction_2d,
        'ground_truth_2d': ground_truth_2d,
        'Phi_at_sources': Phi_sources.cpu().numpy(),
    }


def plot_geometric_learning(model, sample_data, x_grid, y_grid, Phi, analysis, output_path, config):
    """
    Create visualization showing geometric/spatial learning in test functions.

    Layout (3 rows x 4 columns):
    Row 1: Overview (input sources, ground truth, prediction, dominant function map)
    Row 2-3: 8 most important test functions with geometric annotations
    """
    p = Phi.shape[0]
    weighted_acts = analysis['weighted_activations']
    sorted_indices = np.argsort(weighted_acts)[::-1]

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # ========== ROW 1: Overview ==========

    # Panel 1: Input sources
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(sample_data['source_coords'][:, 0],
                     sample_data['source_coords'][:, 1],
                     c=sample_data['source_rates'],
                     s=300, cmap='Reds', alpha=0.8, edgecolors='black', linewidths=2)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(f'Input: {len(sample_data["source_coords"])} Sources',
                  fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax1, label='Rate', fraction=0.046)

    # Panel 2: Ground truth
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(x_grid, y_grid, analysis['ground_truth_2d'], levels=20, cmap='viridis')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Ground Truth', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # Panel 3: Prediction
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(x_grid, y_grid, analysis['prediction_2d'], levels=20, cmap='viridis')
    error = np.linalg.norm(analysis['prediction_2d'] - analysis['ground_truth_2d']) / \
            np.linalg.norm(analysis['ground_truth_2d'])
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('y', fontsize=12)
    ax3.set_title(f'Prediction (L2: {error:.4f})', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, fraction=0.046)

    # Panel 4: Dominant function map (geometric partitioning)
    ax4 = fig.add_subplot(gs[0, 3])
    # Create discrete colormap
    n_colors = min(p, 32)
    cmap_discrete = plt.cm.tab20(np.linspace(0, 1, n_colors))
    im4 = ax4.contourf(x_grid, y_grid, analysis['dominant_function'],
                       levels=np.arange(p+1)-0.5, cmap=ListedColormap(cmap_discrete))

    # Overlay mass centers
    centers = analysis['mass_centers']
    for k in sorted_indices[:8]:  # Show top 8
        ax4.plot(centers[k, 0], centers[k, 1], 'w*', markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)
        ax4.text(centers[k, 0], centers[k, 1], f'{k}',
                fontsize=8, ha='center', va='center', fontweight='bold')

    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('y', fontsize=12)
    ax4.set_title('Dominant Test Function\n(Geometric Partitioning)',
                  fontsize=13, fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')

    # ========== ROWS 2-3: Top 8 Test Functions with Geometry ==========

    n_show = 8
    for i in range(n_show):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        k = sorted_indices[i]
        activation_pct = 100 * weighted_acts[k]

        # Plot test function
        im = ax.contourf(x_grid, y_grid, Phi[k], levels=20, cmap='plasma')

        # Overlay geometric center and radius
        cx, cy = analysis['mass_centers'][k]
        radius = analysis['radii'][k]

        circle = plt.Circle((cx, cy), radius, fill=False,
                           edgecolor='white', linewidth=2, linestyle='--', alpha=0.8)
        ax.add_patch(circle)
        ax.plot(cx, cy, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=1.5)

        # Overlay sources
        ax.scatter(sample_data['source_coords'][:, 0],
                  sample_data['source_coords'][:, 1],
                  c='white', s=40, alpha=0.6, marker='x', linewidths=1.5)

        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'φ_{k}: {activation_pct:.1f}% | Center=({cx:.2f},{cy:.2f}) | R={radius:.2f}',
                    fontsize=10, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Overall title
    test_l2 = config.get("test_l2_error")
    title = f"Geometric Learning in Quadrature Test Functions\n"
    title += f"Test functions learn SPATIAL STRUCTURE: localized regions, geometric centers, varying scales"
    if test_l2:
        title += f" | Model Test L2: {test_l2:.4f}"

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved geometric learning visualization to: {output_path}")
    plt.close()


def main():
    """Main execution."""
    args = parse_arguments()

    print(f"=== Quadrature Geometric Learning Visualization ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output: {args.output_dir}")
    print(f"Grid size: {args.grid_size}")
    print()

    try:
        # 1. Load model
        print("[1/5] Loading model...")
        model, config = load_model_and_config(args.checkpoint_path, args.device)
        print()

        # 2. Load sample
        print("[2/5] Loading test sample...")
        sample_data = load_sample_data(args.sample_idx)
        print()

        # 3. Generate grid
        print("[3/5] Generating evaluation grid...")
        x = torch.linspace(0, 1, args.grid_size)
        y = torch.linspace(0, 1, args.grid_size)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        print()

        # 4. Compute test functions
        print("[4/5] Computing test functions...")
        Phi, x_grid_np, y_grid_np = compute_test_functions_2d(model, x_grid, y_grid)
        print(f"  Computed {Phi.shape[0]} test functions")
        print()

        # 5. Analyze and visualize
        print("[5/5] Analyzing geometric structure...")
        analysis = compute_spatial_geometry_analysis(model, sample_data,
                                                     x_grid_np, y_grid_np, Phi)

        output_path = Path(args.output_dir) / "quadrature_geometric_learning.png"
        plot_geometric_learning(model, sample_data, x_grid_np, y_grid_np,
                               Phi, analysis, output_path, config)
        print()

        print("=== Done! ===")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
