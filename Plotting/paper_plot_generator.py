#!/usr/bin/env python
"""paper_plot_generator.py - Generate stacked multi-model comparison figures.

Creates figures with SetONet on top, DeepONet (if exists) in middle, VIDON at bottom.
Uses consistent GridSpec-based layout with benchmark-specific spacing.

Usage:
    python Plotting/paper_plot_generator.py --benchmarks heat_2d_P10 elastic_2d
    python Plotting/paper_plot_generator.py --n_samples 5 --device cuda:1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from Plotting.paper_plot_config import (
    BENCHMARKS, MODEL_ORDER, OUTPUT_FORMATS, MODEL_DISPLAY_NAMES,
    PNG_DPI, DEFAULT_SEED, DEFAULT_N_SAMPLES,
)
from Plotting.paper_plot_utils import load_model, load_dataset

_project_root = Path(__file__).parent.parent


# =============================================================================
# Helper Functions
# =============================================================================

def _add_wind_arrow(ax, wind_angle=0.0):
    """Add wind direction arrow."""
    arrow_length = 0.2
    cx, cy = 0.9, 0.95
    wx, wy = np.cos(wind_angle), np.sin(wind_angle)
    arrow = FancyArrowPatch(
        (cx - arrow_length/2 * wx, cy - arrow_length/2 * wy),
        (cx + arrow_length/2 * wx, cy + arrow_length/2 * wy),
        arrowstyle='->', lw=5, color='deepskyblue', mutation_scale=30,
        transform=ax.transAxes, zorder=10
    )
    ax.add_patch(arrow)


def _circular_mask(X, Y, cx=0.5, cy=0.5, r=0.25):
    return (X - cx)**2 + (Y - cy)**2 <= r**2


# =============================================================================
# Error Computation Helpers (for two-pass approach)
# =============================================================================

def _compute_heat_conc_error(model, dataset, wrapper, idx, device, is_concentration=False):
    """Compute error for heat/concentration without plotting. Returns error_max."""
    sample = dataset['test'][idx]
    sources = np.array(sample["sources"])
    
    grid_n = 64
    x = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, x, indexing='xy')
    viz_coords = np.column_stack([X.flatten(), Y.flatten()])
    
    is_adaptive = 'grid_coords' in sample
    if is_adaptive:
        gt_coords = np.array(sample["grid_coords"])
        gt_values = np.array(sample["field_values"])
        method = 'linear' if is_concentration else 'cubic'
        field_gt = griddata(gt_coords, gt_values, (X, Y), method=method, fill_value=0)
        
        with torch.no_grad():
            src_xy = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
            src_val = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)
            tgt = torch.tensor(gt_coords, device=device, dtype=torch.float32).unsqueeze(0)
            pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy()
        pred_field = griddata(gt_coords, pred, (X, Y), method=method, fill_value=0)
    else:
        field_gt = np.array(sample["field"]).squeeze()
        with torch.no_grad():
            src_xy = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
            src_val = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)
            if is_concentration:
                tgt = torch.tensor(viz_coords, device=device, dtype=torch.float32).unsqueeze(0)
                pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy().reshape(grid_n, grid_n)
            else:
                tgt = wrapper.grid_coords.unsqueeze(0)
                pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy()
        pred_field = pred.reshape(field_gt.shape) if not is_concentration else pred
    
    field_gt = np.nan_to_num(field_gt, 0)
    pred_field = np.nan_to_num(pred_field, 0)
    error = np.abs(field_gt - pred_field)
    
    return float(error.max())


def _compute_elastic_error(model, dataset, wrapper, idx, device):
    """Compute error for elastic without plotting. Returns error_max, error_order."""
    sample = dataset['test'][idx]
    
    xs = torch.tensor(sample['X'], device=device, dtype=torch.float32).unsqueeze(0)
    us = torch.tensor(sample['u'], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    ys = torch.tensor(sample['Y'], device=device, dtype=torch.float32).unsqueeze(0)
    gt_norm = torch.tensor(sample['s'], device=device, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred_norm = model(xs, us, ys)
    
    pred = wrapper.denormalize_displacement(pred_norm.squeeze(-1)).cpu().numpy().squeeze()
    gt = wrapper.denormalize_displacement(gt_norm).cpu().numpy().squeeze()
    
    error = np.abs(pred - gt)
    error_vmax = float(np.max(error))
    error_order = int(np.floor(np.log10(error_vmax))) if error_vmax > 0 else 0
    
    return error_vmax, error_order


def _compute_transport_error(model, dataset, idx, device):
    """Compute error for transport without plotting. Returns error_max."""
    sample = dataset['test'][idx]
    src_pts = np.array(sample["source_points"])
    qry_pts = np.array(sample["query_points"])
    gt_vec = np.array(sample["query_vectors"])
    
    with torch.no_grad():
        src_t = torch.tensor(src_pts, device=device, dtype=torch.float32).unsqueeze(0)
        wts = torch.ones(1, len(src_pts), 1, device=device)
        qry_t = torch.tensor(qry_pts, device=device, dtype=torch.float32).unsqueeze(0)
        pred_vec = model(src_t, wts, qry_t).squeeze(0).cpu().numpy()
    
    error_mag = np.linalg.norm(pred_vec - gt_vec, axis=1)
    return float(error_mag.max())


# =============================================================================
# Heat/Concentration Plotting
# =============================================================================

def _plot_heat_conc_row(axes, cb_axes, model, dataset, wrapper, idx, device, vmin, vmax, 
                        cmap, label, is_concentration=False, shared_error_vmax=None):
    """Plot heat or concentration row into provided axes.
    
    Args:
        shared_error_vmax: If provided, use this as the error scale max (from SetONet)
    
    Returns:
        im_gt, im_err, vmin, vmax, label, error_vmax
    """
    sample = dataset['test'][idx]
    sources = np.array(sample["sources"])
    wind_angle = sample.get("wind_angle", 0.0) if is_concentration else None
    
    grid_n = 64
    x = np.linspace(0, 1, grid_n)
    X, Y = np.meshgrid(x, x, indexing='xy')
    viz_coords = np.column_stack([X.flatten(), Y.flatten()])
    
    is_adaptive = 'grid_coords' in sample
    if is_adaptive:
        gt_coords = np.array(sample["grid_coords"])
        gt_values = np.array(sample["field_values"])
        method = 'linear' if is_concentration else 'cubic'
        field_gt = griddata(gt_coords, gt_values, (X, Y), method=method, fill_value=0)
        
        with torch.no_grad():
            src_xy = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
            src_val = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)
            tgt = torch.tensor(gt_coords, device=device, dtype=torch.float32).unsqueeze(0)
            pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy()
        pred_field = griddata(gt_coords, pred, (X, Y), method=method, fill_value=0)
    else:
        field_gt = np.array(sample["field"]).squeeze()
        with torch.no_grad():
            src_xy = torch.tensor(sources[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
            src_val = torch.tensor(sources[:, 2:3], device=device, dtype=torch.float32).unsqueeze(0)
            if is_concentration:
                tgt = torch.tensor(viz_coords, device=device, dtype=torch.float32).unsqueeze(0)
                pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy().reshape(grid_n, grid_n)
            else:
                tgt = wrapper.grid_coords.unsqueeze(0)
                pred = model(src_xy, src_val, tgt).squeeze().cpu().numpy()
        pred_field = pred.reshape(field_gt.shape) if not is_concentration else pred
    
    field_gt = np.nan_to_num(field_gt, 0)
    pred_field = np.nan_to_num(pred_field, 0)
    error = np.abs(field_gt - pred_field)
    
    if vmin is None:
        vmin, vmax = float(field_gt.min()), float(field_gt.max())
    
    # Compute this model's error max
    this_error_vmax = float(error.max())
    # Use shared error scale if provided (from SetONet)
    error_vmax_to_use = shared_error_vmax if shared_error_vmax is not None else this_error_vmax
    
    im_list = []
    # Plot Prediction (0), GT (1), Error (2)
    # axes provided map to these: axes[0]=Pred, axes[1]=GT, axes[2]=Error
    
    # Prediction
    im_pred = axes[0].contourf(X, Y, pred_field, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    im_list.append(im_pred)
    
    # GT
    im_gt = axes[1].contourf(X, Y, field_gt, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    im_list.append(im_gt)
    if cb_axes[0] is not None:
        cbar = plt.colorbar(im_gt, cax=cb_axes[0], format='%.2f')
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        
    # Error - use shared error scale with explicit levels for consistent colorbar
    error_levels = np.linspace(0, error_vmax_to_use, 101)
    im_err = axes[2].contourf(X, Y, error, levels=error_levels, cmap='Reds', vmin=0, vmax=error_vmax_to_use)
    im_list.append(im_err)
    if cb_axes[1] is not None:
        cbar = plt.colorbar(im_err, cax=cb_axes[1], format='%.2f')
        cbar.set_label('|Error|', rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        # Set explicit ticks for consistent colorbar across models
        tick_values = np.linspace(0, error_vmax_to_use, 6)
        cbar.set_ticks(tick_values)

    # Common styling
    for i, ax in enumerate(axes):
        if len(sources) > 0:
            c = sources[:, 2] if i < 2 else 'blue'
            cmap_s = 'viridis' if i < 2 else None
            ax.scatter(sources[:, 0], sources[:, 1], c=c, s=100, cmap=cmap_s,
                      edgecolors='white', linewidth=2, alpha=0.7 if i == 2 else 1.0)
        
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_aspect('equal', adjustable='box')
        
        if is_concentration and i < 2:
            _add_wind_arrow(ax, wind_angle)
    
    return im_list[1], im_list[2], vmin, vmax, label, this_error_vmax


def generate_heat_conc_stacked(benchmark, sample_idx, models, dataset, wrapper, device, output_dir,
                                cmap, field_label):
    """Generate stacked heat/concentration figure with integrated GridSpec layout."""
    avail = [(n, m) for n, m in models.items() if m is not None]
    if not avail:
        return
    
    n_rows = len(avail)
    is_conc = 'concentration' in benchmark
    
    # === PASS 1: Compute global max error across all models ===
    global_error_vmax = 0.0
    for model_name, model in avail:
        err_max = _compute_heat_conc_error(model, dataset, wrapper, sample_idx, device, is_conc)
        global_error_vmax = max(global_error_vmax, err_max)
    
    # Layout configuration
    plot_ratios = [1, 1, 1]  # Pred, GT, Error
    has_colorbar = [False, True, True]  # CB for GT and Error
    cb_ratio = 0.07
    
    # Spacing adjustments:
    # wspaces[0]: Gap between Pred and GT (tighter)
    # wspaces[1]: Gap between GT+CB and Error (keep room for colorbar labels)
    wspaces = [0.08, 0.3]
    
    avg_width = sum(plot_ratios) / len(plot_ratios)
    spacer_widths = [w * avg_width for w in wspaces]
    
    full_ratios = []
    col_starts = []
    cb_cols = []
    current_col = 0
    
    for i in range(len(plot_ratios)):
        if i > 0:
            full_ratios.append(spacer_widths[i-1])
            current_col += 1
        full_ratios.append(plot_ratios[i])
        col_starts.append(current_col)
        current_col += 1
        if has_colorbar[i]:
            full_ratios.append(cb_ratio)
            cb_cols.append(current_col)
            current_col += 1
            
    # Figure size
    fig = plt.figure(figsize=(19.5, 6 * n_rows))
    gs = fig.add_gridspec(n_rows, len(full_ratios), width_ratios=full_ratios, wspace=0, hspace=0.4)
    
    shared_vmin, shared_vmax = None, None
    row_axes = []  # Store first axis of each row for title positioning
    bottom_axes = None
    
    # === PASS 2: Plot all models with global error scale ===
    for row, (model_name, model) in enumerate(avail):
        # Extract axes
        axes = [fig.add_subplot(gs[row, col_starts[j]]) for j in range(len(plot_ratios))]
        row_axes.append((model_name, axes[0]))
        if row == n_rows - 1:
            bottom_axes = axes
        
        # Extract colorbar axes (mapped to which plot has one)
        cb_axes = [None] * len(plot_ratios)
        cb_idx = 0
        for i in range(len(plot_ratios)):
            if has_colorbar[i]:
                cb_axes[i] = fig.add_subplot(gs[row, cb_cols[cb_idx]])
                cb_idx += 1
        
        # Only pass relevant CB axes to plotting function (GT and Error)
        active_cb_axes = [cb_axes[1], cb_axes[2]]
        
        _, _, v_min, v_max, _, _ = _plot_heat_conc_row(
            axes, active_cb_axes, model, dataset, wrapper, sample_idx, device,
            shared_vmin, shared_vmax, cmap, field_label, is_conc,
            shared_error_vmax=global_error_vmax
        )
        
        # Capture field scale from first model
        if shared_vmin is None:
            shared_vmin, shared_vmax = v_min, v_max
    
    # Skip tight_layout to preserve GridSpec spacing (especially hspace for row gaps)
    # Use subplots_adjust for fine-tuning instead
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.08)
    
    # Add row titles (centered across entire figure width)
    for model_name, ax in row_axes:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        pos = ax.get_position()
        # Position title just above the subplot (small offset to avoid overlap with row above)
        fig.text(0.5, pos.y1 + 0.03, display_name, fontsize=22, fontweight='bold',
                 ha='center', va='bottom', transform=fig.transFigure)

    # Add column labels under bottom row
    if bottom_axes is not None:
        col_labels = ["(Prediction)", "(Ground truth)", "(Error)"]
        label_y = min(ax.get_position().y0 for ax in bottom_axes) - 0.09
        label_y = max(label_y, 0.01)
        for ax, label in zip(bottom_axes, col_labels):
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1) / 2, label_y, label, ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Save
    base = f"fig_{benchmark}_sample_{sample_idx}"
    for fmt in OUTPUT_FORMATS:
        fig.savefig(output_dir / f"{base}.{fmt}", format=fmt, 
                   dpi=PNG_DPI if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {base}")


# =============================================================================
# Elastic Plotting
# =============================================================================

def _plot_elastic_row(axes, cb_axes, model, dataset, wrapper, idx, device, vmin, vmax,
                      shared_error_vmax=None, shared_error_order=None):
    """Plot elastic row into provided axes.
    
    Args:
        shared_error_vmax: If provided, use this as the error scale max (from SetONet)
        shared_error_order: If provided, use this as the error order (from SetONet)
    
    Returns:
        vmin, vmax, error_vmax, error_order
    """
    sample = dataset['test'][idx]
    
    xs = torch.tensor(sample['X'], device=device, dtype=torch.float32).unsqueeze(0)
    us = torch.tensor(sample['u'], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    ys = torch.tensor(sample['Y'], device=device, dtype=torch.float32).unsqueeze(0)
    gt_norm = torch.tensor(sample['s'], device=device, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred_norm = model(xs, us, ys)
    
    pred = wrapper.denormalize_displacement(pred_norm.squeeze(-1)).cpu().numpy().squeeze()
    gt = wrapper.denormalize_displacement(gt_norm).cpu().numpy().squeeze()
    coords = wrapper.denormalize_coordinates(xs.squeeze(0)).cpu().numpy()
    force = wrapper.denormalize_force(us.squeeze(0).squeeze(-1)).cpu().numpy()
    tgt_coords = wrapper.denormalize_coordinates(ys.squeeze(0)).cpu().numpy()
    
    error = np.abs(pred - gt)
    
    if vmin is None:
        vmin = float(min(pred.min(), gt.min()))
        vmax = float(max(pred.max(), gt.max()))
    
    # Compute scaling orders (like original)
    shared_vabs = max(abs(vmin), abs(vmax))
    shared_order = int(np.floor(np.log10(shared_vabs))) if shared_vabs > 0 else 0
    
    # Error scale: use shared if provided, otherwise compute from this model's error
    this_error_vmax = float(np.max(error))
    this_error_order = int(np.floor(np.log10(this_error_vmax))) if this_error_vmax > 0 else 0
    
    # Use shared error scale if provided (from SetONet)
    error_vmax_to_use = shared_error_vmax if shared_error_vmax is not None else this_error_vmax
    error_order_to_use = shared_error_order if shared_error_order is not None else this_error_order
    
    # Force plot
    ax = axes[0]
    ax.plot(force, coords[:, 1], 'r-', linewidth=2)
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.set_xlabel('Force value', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=16)
    
    # Displacement plots
    x, y = tgt_coords[:, 0], tgt_coords[:, 1]
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    
    for i, (ax, vals, cax, is_err) in enumerate([
        (axes[1], pred, None, False),
        (axes[2], gt, cb_axes[0], False),
        (axes[3], error, cb_axes[1], True)
    ]):
        Zi = griddata((x, y), vals, (Xi, Yi), method='cubic')
        if np.isnan(Zi).any():
            Zi_nearest = griddata((x, y), vals, (Xi, Yi), method='nearest')
            Zi = np.where(np.isnan(Zi), Zi_nearest, Zi)
        Zi[_circular_mask(Xi, Yi)] = np.nan
        
        # Use shared error scale for error plots
        if is_err:
            v = (0.0, error_vmax_to_use)
            order = error_order_to_use
        else:
            v = (vmin, vmax)
            order = shared_order
        
        scale = 10 ** -order
        Zi_scaled = Zi * scale
        v_scaled = (v[0] * scale, v[1] * scale)
        if is_err:
            Zi_scaled = np.clip(Zi_scaled, v_scaled[0], v_scaled[1])
        
        cmap_name = 'jet'
        
        # Only use explicit levels for error plots (for consistent colorbar)
        if is_err:
            plot_levels = np.linspace(v_scaled[0], v_scaled[1], 101)
            im = ax.contourf(Xi, Yi, Zi_scaled, levels=plot_levels, cmap=cmap_name, 
                            vmin=v_scaled[0], vmax=v_scaled[1])
        else:
            im = ax.contourf(Xi, Yi, Zi_scaled, levels=100, cmap=cmap_name, 
                            vmin=v_scaled[0], vmax=v_scaled[1])
        
        ax.set_aspect('equal')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.tick_params(labelsize=16)
        
        if cax is not None:
            cbar = plt.colorbar(im, cax=cax, format='%.1f')
            label = '|Error|' if is_err else 'x-displacement'
            cbar.set_label(label, rotation=270, labelpad=20, fontsize=16)
            cbar.ax.tick_params(labelsize=16)
            # Use shared error order for colorbar title
            order_for_title = error_order_to_use if is_err else order
            cbar.ax.set_title(f'$10^{{{order_for_title}}}$', fontsize=16, pad=10)
            # Set explicit ticks only for error colorbar
            if is_err:
                tick_values = np.linspace(v_scaled[0], v_scaled[1], 6)
                cbar.set_ticks(tick_values)
    
    # Return this model's error stats (used by SetONet to set shared scale)
    return vmin, vmax, this_error_vmax, this_error_order


def generate_elastic_stacked(benchmark, sample_idx, models, dataset, wrapper, device, output_dir):
    """Generate stacked elastic figure matching original utility layout."""
    avail = [(n, m) for n, m in models.items() if m is not None]
    if not avail:
        return
    
    n_rows = len(avail)
    
    # === PASS 1: Compute global max error across all models ===
    global_error_vmax = 0.0
    global_error_order = 0
    for model_name, model in avail:
        err_max, err_order = _compute_elastic_error(model, dataset, wrapper, sample_idx, device)
        if err_max > global_error_vmax:
            global_error_vmax = err_max
            global_error_order = err_order
    
    # Layout configuration
    plot_ratios = [0.3, 1, 1, 1]
    has_colorbar = [False, False, True, True]
    cb_ratio = 0.07
    
    # Spacing adjustments:
    # 0: Force -> Pred (0.2 increased)
    # 1: Pred -> GT (0.1 increased)
    # 2: GT+CB -> Error (0.4 increased to avoid overlap)
    wspaces = [0.2, 0.1, 0.4]
    
    avg_width = sum(plot_ratios) / len(plot_ratios)
    spacer_widths = [w * avg_width for w in wspaces]
    
    full_ratios = []
    col_starts = []
    cb_cols = []
    current_col = 0
    for i in range(len(plot_ratios)):
        if i > 0:
            full_ratios.append(spacer_widths[i-1])
            current_col += 1
        full_ratios.append(plot_ratios[i])
        col_starts.append(current_col)
        current_col += 1
        if has_colorbar[i]:
            full_ratios.append(cb_ratio)
            cb_cols.append(current_col)
            current_col += 1
    
    # Figure size
    fig = plt.figure(figsize=(19.5, 5.5 * n_rows))
    gs = fig.add_gridspec(n_rows, len(full_ratios), width_ratios=full_ratios, wspace=0, hspace=0.5)
    
    shared_vmin, shared_vmax = None, None
    row_axes = []  # Store first axis of each row for title positioning
    bottom_axes = None
    
    # === PASS 2: Plot all models with global error scale ===
    for row, (model_name, model) in enumerate(avail):
        axes = [fig.add_subplot(gs[row, col_starts[j]]) for j in range(len(plot_ratios))]
        row_axes.append((model_name, axes[0]))
        if row == n_rows - 1:
            bottom_axes = axes
        cb_axes = [fig.add_subplot(gs[row, c]) for c in cb_cols]
        
        v_min, v_max, _, _ = _plot_elastic_row(
            axes, cb_axes, model, dataset, wrapper, 
            sample_idx, device, shared_vmin, shared_vmax,
            shared_error_vmax=global_error_vmax, shared_error_order=global_error_order
        )
        
        # Capture displacement scale from first model
        if shared_vmin is None:
            shared_vmin, shared_vmax = v_min, v_max
    
    # Skip tight_layout to preserve GridSpec spacing (especially hspace for row gaps)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)
    
    # Add row titles (centered across entire figure width)
    for model_name, ax in row_axes:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        pos = ax.get_position()
        fig.text(0.5, pos.y1 + 0.03, display_name, fontsize=22, fontweight='bold',
                 ha='center', va='bottom', transform=fig.transFigure)

    # Add column labels under bottom row
    if bottom_axes is not None:
        col_labels = ["(Input)", "(Prediction)", "(Ground truth)", "(Error)"]
        label_y = min(ax.get_position().y0 for ax in bottom_axes) - 0.09
        label_y = max(label_y, 0.01)
        for ax, label in zip(bottom_axes, col_labels):
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1) / 2, label_y, label, ha='center', va='top', fontsize=18, fontweight='bold')
    
    base = f"fig_{benchmark}_sample_{sample_idx}"
    for fmt in OUTPUT_FORMATS:
        fig.savefig(output_dir / f"{base}.{fmt}", format=fmt,
                   dpi=PNG_DPI if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {base}")


# =============================================================================
# Transport Plotting
# =============================================================================

def _plot_transport_row(axes, cb_axes, model, dataset, wrapper, idx, device, shared_error_vmax=None):
    """Plot transport row.
    
    Args:
        shared_error_vmax: If provided, use this as the error scale max (from SetONet)
    
    Returns:
        sc, error_vmax
    """
    sample = dataset['test'][idx]
    src_pts = np.array(sample["source_points"])
    qry_pts = np.array(sample["query_points"])
    gt_vec = np.array(sample["query_vectors"])
    
    with torch.no_grad():
        src_t = torch.tensor(src_pts, device=device, dtype=torch.float32).unsqueeze(0)
        wts = torch.ones(1, len(src_pts), 1, device=device)
        qry_t = torch.tensor(qry_pts, device=device, dtype=torch.float32).unsqueeze(0)
        pred_vec = model(src_t, wts, qry_t).squeeze(0).cpu().numpy()
    
    transported_pred = qry_pts + pred_vec
    transported_gt = qry_pts + gt_vec
    error_mag = np.linalg.norm(pred_vec - gt_vec, axis=1)
    
    # Compute this model's error max
    this_error_vmax = float(error_mag.max())
    # Use shared error scale if provided (from SetONet)
    error_vmax_to_use = shared_error_vmax if shared_error_vmax is not None else this_error_vmax
    
    domain = float(sample.get("domain_size", 5.0))
    lim = (-domain - 0.5, domain + 0.5)
    
    n_arr = min(30, len(src_pts))
    arr_idx = np.linspace(0, len(src_pts) - 1, n_arr, dtype=int)
    src_sub = src_pts[arr_idx]
    tree = cKDTree(qry_pts)
    _, nearest = tree.query(src_sub)
    gt_arr = gt_vec[nearest]
    
    with torch.no_grad():
        src_sub_t = torch.tensor(src_sub, device=device, dtype=torch.float32).unsqueeze(0)
        pred_arr = model(src_t, wts, src_sub_t).squeeze(0).cpu().numpy()
    
    # Plot 1: Predicted
    axes[0].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=15, alpha=0.5)
    axes[0].scatter(qry_pts[:, 0], qry_pts[:, 1], c='gray', s=8, alpha=0.3, marker='x')
    axes[0].scatter(transported_pred[:, 0], transported_pred[:, 1], c='red', s=15, alpha=0.6)
    
    # Plot 2: GT
    axes[1].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=15, alpha=0.5)
    axes[1].scatter(qry_pts[:, 0], qry_pts[:, 1], c='gray', s=8, alpha=0.3, marker='x')
    axes[1].scatter(transported_gt[:, 0], transported_gt[:, 1], c='green', s=15, alpha=0.6)
    
    # Plot 3: Arrows
    axes[2].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=8, alpha=0.3)
    axes[2].quiver(src_sub[:, 0], src_sub[:, 1], gt_arr[:, 0], gt_arr[:, 1],
                   color='green', alpha=0.8, angles='xy', scale_units='xy', scale=1,
                   width=0.008, headwidth=3, headlength=4)
    axes[2].quiver(src_sub[:, 0], src_sub[:, 1], pred_arr[:, 0], pred_arr[:, 1],
                   color='red', alpha=0.7, angles='xy', scale_units='xy', scale=1,
                   width=0.008, headwidth=3, headlength=4)
    axes[2].scatter(src_sub[:, 0], src_sub[:, 1], c='blue', s=25, alpha=0.8)
    
    all_pts = np.vstack([src_sub, src_sub + gt_arr, src_sub + pred_arr])
    pad = max(all_pts.max(0) - all_pts.min(0)) * 0.15
    center = (all_pts.max(0) + all_pts.min(0)) / 2
    rng = max(all_pts.max(0) - all_pts.min(0)) / 2 + pad
    axes[2].set_xlim(center[0] - rng, center[0] + rng)
    axes[2].set_ylim(center[1] - rng, center[1] + rng)
    
    # Plot 4: Error - use shared error scale
    sc = axes[3].scatter(qry_pts[:, 0], qry_pts[:, 1], c=error_mag, cmap='magma', s=15, alpha=0.85,
                         vmin=0, vmax=error_vmax_to_use)
    axes[3].scatter(src_pts[:, 0], src_pts[:, 1], c='blue', s=10, alpha=0.3, marker='s')
    
    # Colorbar with explicit ticks for consistency
    if cb_axes[0] is not None:
        cbar = plt.colorbar(sc, cax=cb_axes[0], format='%.2f')
        cbar.set_label('|Error|', rotation=270, labelpad=20, fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        # Set explicit ticks for consistent colorbar across models
        tick_values = np.linspace(0, error_vmax_to_use, 6)
        cbar.set_ticks(tick_values)
    
    for i, ax in enumerate(axes):
        if i != 2:
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        ax.set_xlabel('x', fontsize=16)
        ax.set_ylabel('y', fontsize=16)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.xaxis.set_major_locator(MultipleLocator(2.0))
    
    return sc, this_error_vmax


def generate_transport_stacked(benchmark, sample_idx, models, dataset, wrapper, device, output_dir):
    """Generate stacked transport figure matching original utility layout."""
    avail = [(n, m) for n, m in models.items() if m is not None]
    if not avail:
        return
    
    n_rows = len(avail)
    
    # === PASS 1: Compute global max error across all models ===
    global_error_vmax = 0.0
    for model_name, model in avail:
        err_max = _compute_transport_error(model, dataset, sample_idx, device)
        global_error_vmax = max(global_error_vmax, err_max)
    
    # Layout configuration
    plot_ratios = [1, 1, 1, 1]
    has_colorbar = [False, False, False, True]  # only Error has colorbar
    cb_ratio = 0.07
    
    # Spacing adjustments:
    # 0-2: Spacers between plots (tighter)
    # 3: Spacer before Error Colorbar (keep tight)
    wspaces = [0.08, 0.08, 0.08, 0.02]
    
    avg_width = sum(plot_ratios) / len(plot_ratios)
    spacer_widths = [w * avg_width for w in wspaces]
    
    full_ratios = []
    col_starts = []
    cb_cols = []
    current_col = 0
    
    for i in range(len(plot_ratios)):
        full_ratios.append(plot_ratios[i])
        col_starts.append(current_col)
        current_col += 1
        
        # Add spacer after plot (except last if no colorbar)
        if i < len(plot_ratios) - 1:
            full_ratios.append(spacer_widths[i])
            current_col += 1
        elif has_colorbar[i]:
            # Spacer before colorbar
            full_ratios.append(spacer_widths[i])
            current_col += 1
            # Colorbar
            full_ratios.append(cb_ratio)
            cb_cols.append(current_col)
            current_col += 1
            
    # Figure size
    fig = plt.figure(figsize=(19.5, 5.2 * n_rows))
    gs = fig.add_gridspec(n_rows, len(full_ratios), width_ratios=full_ratios, wspace=0, hspace=0.5)
    
    row_axes = []  # Store first axis of each row for title positioning
    bottom_axes = None
    
    # === PASS 2: Plot all models with global error scale ===
    for row, (model_name, model) in enumerate(avail):
        axes = [fig.add_subplot(gs[row, col_starts[j]]) for j in range(len(plot_ratios))]
        row_axes.append((model_name, axes[0]))
        if row == n_rows - 1:
            bottom_axes = axes
        cb_axes = [fig.add_subplot(gs[row, c]) for c in cb_cols]
        
        sc, _ = _plot_transport_row(
            axes, cb_axes, model, dataset, wrapper, sample_idx, device,
            shared_error_vmax=global_error_vmax
        )
    
    # Skip tight_layout to preserve GridSpec spacing (especially hspace for row gaps)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.08)
    
    # Add row titles (centered across entire figure width)
    for model_name, ax in row_axes:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        pos = ax.get_position()
        fig.text(0.5, pos.y1 + 0.04, display_name, fontsize=22, fontweight='bold',
                 ha='center', va='bottom', transform=fig.transFigure)

    # Add column labels under bottom row
    if bottom_axes is not None:
        col_labels = ["(Prediction)", "(Ground truth)", "(Displacement)", "(Error)"]
        label_y = min(ax.get_position().y0 for ax in bottom_axes) - 0.09
        label_y = max(label_y, 0.01)
        for ax, label in zip(bottom_axes, col_labels):
            pos = ax.get_position()
            fig.text((pos.x0 + pos.x1) / 2, label_y, label, ha='center', va='top', fontsize=18, fontweight='bold')
    
    base = f"fig_{benchmark}_sample_{sample_idx}"
    for fmt in OUTPUT_FORMATS:
        fig.savefig(output_dir / f"{base}.{fmt}", format=fmt,
                   dpi=PNG_DPI if fmt == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {base}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper figure generator")
    parser.add_argument('--benchmarks', nargs='+', default=list(BENCHMARKS.keys()))
    parser.add_argument('--n_samples', type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output_dir', default='paper_figures')
    parser.add_argument('--device', default='cuda:1')
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
        
        print("\nGenerating stacked figures...")
        n = min(args.n_samples, len(dataset['test']))
        
        plot_type = BENCHMARKS[benchmark]['plot_type']
        
        for i in range(n):
            if plot_type == 'heat':
                generate_heat_conc_stacked(benchmark, i, models, dataset, wrapper, device, 
                                          bench_out, 'hot', 'Temperature')
            elif plot_type == 'concentration':
                generate_heat_conc_stacked(benchmark, i, models, dataset, wrapper, device,
                                          bench_out, 'plasma', 'Concentration')
            elif plot_type == 'elastic':
                generate_elastic_stacked(benchmark, i, models, dataset, wrapper, device, bench_out)
            elif plot_type == 'transport_q':
                generate_transport_stacked(benchmark, i, models, dataset, wrapper, device, bench_out)
        
        print(f"\nCompleted: {n} figures")
    
    print(f"\n{'='*50}\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
