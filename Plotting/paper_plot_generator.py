#!/usr/bin/env python
"""paper_plot_generator.py - Generate paper-quality stacked comparison figures.

Usage:
    python Plotting/paper_plot_generator.py --benchmarks heat_2d_P10 elastic_2d
    python Plotting/paper_plot_generator.py --n_samples 5 --device cuda:0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from Plotting.paper_plot_config import (
    BENCHMARKS, FIGURE_WIDTH, MODEL_ORDER, OUTPUT_FORMATS, 
    PAPER_STYLE, PNG_DPI, DEFAULT_SEED, DEFAULT_N_SAMPLES,
)
from Plotting.paper_plot_utils import load_model, load_dataset

_project_root = Path(__file__).parent.parent


# =============================================================================
# Plotting Functions
# =============================================================================

def _get_viz_grid(grid_n=64):
    """Create visualization grid coordinates."""
    x = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, x, indexing='xy')
    return X, Y, np.column_stack([X.flatten(), Y.flatten()])


def _plot_field(ax, X, Y, field, sources, cmap, vmin=None, vmax=None, show_sources=True):
    """Plot a 2D field with optional source markers."""
    im = ax.contourf(X, Y, field, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    if show_sources and len(sources) > 0:
        if sources.shape[1] > 2:
            ax.scatter(sources[:, 0], sources[:, 1], c=sources[:, 2],
                       s=30, cmap='viridis', edgecolors='white', linewidth=1)
        else:
            ax.scatter(sources[:, 0], sources[:, 1], c='blue',
                       s=30, edgecolors='white', linewidth=1)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=10)
    return im


def plot_heat_row(axes, model, dataset, wrapper, idx, device, vmin=None, vmax=None):
    """Plot heat benchmark row: Prediction, Ground Truth, Error."""
    sample = dataset['test'][idx]
    sources = np.array(sample["sources"])
    X, Y, viz_coords = _get_viz_grid()
    
    # Get ground truth
    if 'grid_coords' in sample:
        gt_coords = np.array(sample["grid_coords"])
        gt_values = np.array(sample["field_values"])
        field_gt = griddata(gt_coords, gt_values, (X, Y), method='cubic', fill_value=0)
    else:
        field_gt = np.array(sample["field"]).squeeze()
    
    # Predict
    with torch.no_grad():
        src_xy = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
        src_val = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(gt_coords if 'grid_coords' in sample else viz_coords, 
                          device=device, dtype=torch.float32).unsqueeze(0)
        pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy()
    
    if 'grid_coords' in sample:
        pred_field = griddata(gt_coords, pred, (X, Y), method='cubic', fill_value=0)
    else:
        pred_field = pred.reshape(64, 64)
    
    field_gt = np.nan_to_num(field_gt, 0)
    pred_field = np.nan_to_num(pred_field, 0)
    error = np.abs(field_gt - pred_field)
    
    if vmin is None:
        vmin, vmax = field_gt.min(), field_gt.max()
    
    _plot_field(axes[0], X, Y, pred_field, sources, 'hot', vmin, vmax)
    _plot_field(axes[1], X, Y, field_gt, sources, 'hot', vmin, vmax)
    _plot_field(axes[2], X, Y, error, sources[:, :2].reshape(-1, 2), 'Reds')
    
    return vmin, vmax


def plot_concentration_row(axes, model, dataset, wrapper, idx, device, vmin=None, vmax=None):
    """Plot concentration benchmark row: Prediction, Ground Truth, Error."""
    sample = dataset['test'][idx]
    sources = np.array(sample["sources"])
    X, Y, viz_coords = _get_viz_grid()
    
    # Get ground truth
    if 'grid_coords' in sample:
        gt_coords = np.array(sample["grid_coords"])
        gt_values = np.array(sample["field_values"])
        field_gt = griddata(gt_coords, gt_values, (X, Y), method='linear', fill_value=0)
    else:
        field_gt = np.array(sample["field"]).squeeze()
    
    # Predict on viz grid
    with torch.no_grad():
        src_xy = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
        src_val = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)
        tgt = torch.tensor(viz_coords, device=device, dtype=torch.float32).unsqueeze(0)
        pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy().reshape(64, 64)
    
    error = np.abs(field_gt - pred)
    
    if vmin is None:
        vmin, vmax = field_gt.min(), field_gt.max()
    
    _plot_field(axes[0], X, Y, pred, sources, 'plasma', vmin, vmax)
    _plot_field(axes[1], X, Y, field_gt, sources, 'plasma', vmin, vmax)
    _plot_field(axes[2], X, Y, error, sources[:, :2].reshape(-1, 2), 'Reds')
    
    return vmin, vmax


def _circular_mask(X, Y, cx=0.5, cy=0.5, r=0.25):
    """Create circular mask for elastic plate hole."""
    return (X - cx)**2 + (Y - cy)**2 <= r**2


def plot_elastic_row(axes, model, dataset, wrapper, idx, device, vmin=None, vmax=None):
    """Plot elastic benchmark row: Force, Prediction, Ground Truth, Error."""
    sample = dataset['test'][idx]
    
    xs = torch.tensor(sample['X'], device=device, dtype=torch.float32).unsqueeze(0)
    us = torch.tensor(sample['u'], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    ys = torch.tensor(sample['Y'], device=device, dtype=torch.float32).unsqueeze(0)
    gt_norm = torch.tensor(sample['s'], device=device, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred_norm = model(xs, us, ys)
    
    # Denormalize
    pred = wrapper.denormalize_displacement(pred_norm.squeeze(-1)).cpu().numpy().squeeze()
    gt = wrapper.denormalize_displacement(gt_norm).cpu().numpy().squeeze()
    coords = wrapper.denormalize_coordinates(xs.squeeze(0)).cpu().numpy()
    force = wrapper.denormalize_force(us.squeeze(0).squeeze(-1)).cpu().numpy()
    tgt_coords = wrapper.denormalize_coordinates(ys.squeeze(0)).cpu().numpy()
    
    error = np.abs(pred - gt)
    
    if vmin is None:
        vmin, vmax = min(pred.min(), gt.min()), max(pred.max(), gt.max())
    
    # Plot force profile
    axes[0].plot(force, coords[:, 1], 'r-', linewidth=2)
    axes[0].set_xlabel('Force', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.5)
    axes[0].tick_params(labelsize=10)
    
    # Plot displacement fields
    for ax, vals in [(axes[1], pred), (axes[2], gt), (axes[3], error)]:
        x, y = tgt_coords[:, 0], tgt_coords[:, 1]
        xi = np.linspace(x.min(), x.max(), 200)
        yi = np.linspace(y.min(), y.max(), 200)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), vals, (Xi, Yi), method='cubic')
        Zi[_circular_mask(Xi, Yi)] = np.nan
        
        cmap = 'jet' if ax != axes[3] else 'Reds'
        v = (vmin, vmax) if ax != axes[3] else (error.min(), error.max())
        ax.contourf(Xi, Yi, Zi, levels=100, cmap=cmap, vmin=v[0], vmax=v[1])
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.tick_params(labelsize=10)
    
    return vmin, vmax


def plot_transport_row(axes, model, dataset, wrapper, idx, device):
    """Plot transport benchmark row: Pred, GT, Arrows, Error."""
    from matplotlib.ticker import MultipleLocator
    from scipy.spatial import cKDTree
    
    sample = dataset['test'][idx]
    src_pts = np.array(sample["source_points"])
    query_pts = np.array(sample["query_points"])
    gt_vec = np.array(sample["query_vectors"])
    
    with torch.no_grad():
        src_t = torch.tensor(src_pts, device=device, dtype=torch.float32).unsqueeze(0)
        wts = torch.ones(1, len(src_pts), 1, device=device)
        qry_t = torch.tensor(query_pts, device=device, dtype=torch.float32).unsqueeze(0)
        pred_vec = model(src_t, wts, qry_t).squeeze(0).cpu().numpy()
    
    transported_pred = query_pts + pred_vec
    transported_gt = query_pts + gt_vec
    error_mag = np.linalg.norm(pred_vec - gt_vec, axis=1)
    
    domain = float(sample.get("domain_size", 5.0))
    lim = (-domain - 0.5, domain + 0.5)
    
    # Subsample for arrows
    n_arr = min(30, len(src_pts))
    arr_idx = np.linspace(0, len(src_pts) - 1, n_arr, dtype=int)
    src_sub = src_pts[arr_idx]
    tree = cKDTree(query_pts)
    _, nearest = tree.query(src_sub)
    gt_arr = gt_vec[nearest]
    
    with torch.no_grad():
        src_sub_t = torch.tensor(src_sub, device=device, dtype=torch.float32).unsqueeze(0)
        pred_arr = model(src_t, wts, src_sub_t).squeeze(0).cpu().numpy()
    
    # Plot predicted transport
    axes[0].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=8, alpha=0.5)
    axes[0].scatter(transported_pred[:, 0], transported_pred[:, 1], c='red', s=8, alpha=0.6)
    
    # Plot GT transport
    axes[1].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=8, alpha=0.5)
    axes[1].scatter(transported_gt[:, 0], transported_gt[:, 1], c='green', s=8, alpha=0.6)
    
    # Plot arrows
    axes[2].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=4, alpha=0.3)
    axes[2].quiver(src_sub[:, 0], src_sub[:, 1], gt_arr[:, 0], gt_arr[:, 1],
                   color='green', alpha=0.8, angles='xy', scale_units='xy', scale=1, width=0.008)
    axes[2].quiver(src_sub[:, 0], src_sub[:, 1], pred_arr[:, 0], pred_arr[:, 1],
                   color='red', alpha=0.7, angles='xy', scale_units='xy', scale=1, width=0.008)
    
    # Zoom arrows to relevant region
    all_pts = np.vstack([src_sub, src_sub + gt_arr, src_sub + pred_arr])
    pad = max(all_pts.max(0) - all_pts.min(0)) * 0.15
    center = (all_pts.max(0) + all_pts.min(0)) / 2
    rng = max(all_pts.max(0) - all_pts.min(0)) / 2 + pad
    axes[2].set_xlim(center[0] - rng, center[0] + rng)
    axes[2].set_ylim(center[1] - rng, center[1] + rng)
    
    # Plot error
    axes[3].scatter(query_pts[:, 0], query_pts[:, 1], c=error_mag, cmap='magma', s=8, alpha=0.85)
    axes[3].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=5, alpha=0.3, marker='s')
    
    # Common formatting
    for i, ax in enumerate(axes):
        if i != 2:
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(MultipleLocator(2.0))


# =============================================================================
# Main Generator
# =============================================================================

PLOT_FUNCS = {
    'heat': plot_heat_row,
    'concentration': plot_concentration_row,
    'elastic': plot_elastic_row,
    'transport_q': plot_transport_row,
}


def generate_figure(benchmark: str, sample_idx: int, models: dict, 
                   dataset, wrapper, device: str, output_dir: Path):
    """Generate a stacked multi-model figure."""
    plt.rcParams.update(PAPER_STYLE)
    
    cfg = BENCHMARKS[benchmark]
    n_cols = cfg['n_cols']
    row_h = cfg['row_height']
    plot_fn = PLOT_FUNCS[cfg['plot_type']]
    
    avail = [(n, m) for n, m in models.items() if m is not None]
    if not avail:
        return
    
    fig, axes = plt.subplots(len(avail), n_cols, figsize=(FIGURE_WIDTH, row_h * len(avail)), squeeze=False)
    
    # First pass for shared scales (heat/concentration/elastic)
    shared = None
    for row, (name, model) in enumerate(avail):
        if cfg['plot_type'] in ('heat', 'concentration', 'elastic'):
            result = plot_fn(axes[row], model, dataset, wrapper, sample_idx, device, 
                           shared[0] if shared else None, shared[1] if shared else None)
            if shared is None:
                shared = result
        else:
            plot_fn(axes[row], model, dataset, wrapper, sample_idx, device)
    
    plt.tight_layout()
    
    base = f"fig_{benchmark}_sample_{sample_idx}"
    for fmt in OUTPUT_FORMATS:
        fig.savefig(output_dir / f"{base}.{fmt}", format=fmt, 
                   dpi=PNG_DPI if fmt == 'png' else None, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {base}.pdf and .png")


def main():
    parser = argparse.ArgumentParser(description="Paper figure generator")
    parser.add_argument('--benchmarks', nargs='+', default=list(BENCHMARKS.keys()))
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output_dir', default='paper_figures')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--logs_dir', default='logs_all')
    args = parser.parse_args()
    
    logs_dir = _project_root / args.logs_dir
    output_dir = _project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cpu'
    
    print(f"Paper Figure Generator\n{'='*50}")
    print(f"Device: {device}, Benchmarks: {args.benchmarks}, Samples: {args.n_samples}\n")
    
    for benchmark in args.benchmarks:
        if benchmark not in BENCHMARKS:
            print(f"Unknown benchmark: {benchmark}")
            continue
        
        print(f"\n{'='*50}\nProcessing: {benchmark}\n{'='*50}")
        
        bench_out = output_dir / benchmark
        bench_out.mkdir(parents=True, exist_ok=True)
        
        print("\nLoading dataset...")
        try:
            dataset, wrapper = load_dataset(benchmark, device)
            if dataset is None:
                continue
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        print("\nLoading models...")
        models = {}
        for name in MODEL_ORDER:
            if name in BENCHMARKS[benchmark]['available_models']:
                models[name] = load_model(benchmark, name, args.seed, logs_dir, device)
        
        avail = sum(1 for m in models.values() if m)
        print(f"  {avail}/{len(BENCHMARKS[benchmark]['available_models'])} models loaded")
        
        if not avail:
            continue
        
        print("\nGenerating figures...")
        n = min(args.n_samples, len(dataset['test']))
        for i in range(n):
            generate_figure(benchmark, i, models, dataset, wrapper, device, bench_out)
        
        print(f"\nCompleted: {n} figures")
    
    print(f"\n{'='*50}\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
