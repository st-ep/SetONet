#!/usr/bin/env python
"""plot_transport_q_utils.py
----------------------------------
Plotting utilities for SetONet transport-Q results (decoupled queries).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def plot_transport_q_overlay(model, dataset, transport_dataset, device, sample_idx=0,
                              save_path=None, dataset_split="test", n_arrows=30):
    """
    Visualization for transport-Q benchmark with decoupled queries.

    4-panel layout:
    1. Predicted Transport (scatter)
    2. Ground Truth Transport (scatter)
    3. Transport Trajectories (arrows from source points: green=GT, red=Pred)
    4. Error Magnitude (scatter with colorbar)

    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset
        transport_dataset: TransportDataset wrapper
        device: Device for computations
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
        dataset_split: 'train' or 'test'
        n_arrows: Number of source points to show arrows for
    """
    model.eval()

    data_split = dataset[dataset_split]
    if sample_idx >= len(data_split):
        sample_idx = 0

    sample = data_split[sample_idx]

    # Extract data from sample
    source_points = np.array(sample["source_points"])  # (n_sources, 2)
    query_points = np.array(sample["query_points"])    # (n_queries, 2)
    query_vectors_gt = np.array(sample["query_vectors"])  # (n_queries, 2) - displacement

    # Ground truth transported positions
    transported_gt = query_points + query_vectors_gt  # (n_queries, 2)

    # Get model prediction
    with torch.no_grad():
        source_t = torch.tensor(source_points, device=device, dtype=torch.float32).unsqueeze(0)
        weights_t = torch.ones(1, source_points.shape[0], 1, device=device, dtype=torch.float32)
        query_t = torch.tensor(query_points, device=device, dtype=torch.float32).unsqueeze(0)

        pred_vectors = model(source_t, weights_t, query_t)  # (1, n_queries, 2)
        pred_vectors = pred_vectors.squeeze(0).cpu().numpy()

    # Predicted transported positions
    transported_pred = query_points + pred_vectors  # (n_queries, 2)

    # Error computation
    error_vec = pred_vectors - query_vectors_gt
    error_mag = np.linalg.norm(error_vec, axis=1)

    # Domain settings
    domain_size = float(sample.get("domain_size", 5.0))
    padding = 0.5
    xlim = (-domain_size - padding, domain_size + padding)
    ylim = (-domain_size - padding, domain_size + padding)

    # Subsample source points for arrow visualization
    n_sub = min(n_arrows, len(source_points))
    sub_idx = np.linspace(0, len(source_points) - 1, n_sub, dtype=int)
    source_sub = source_points[sub_idx]

    # Interpolate GT vectors to source points (nearest neighbor from query points)
    from scipy.spatial import cKDTree
    tree = cKDTree(query_points)
    _, nearest_idx = tree.query(source_sub)
    gt_vec_source = query_vectors_gt[nearest_idx]

    # Get model prediction at subsampled source points
    with torch.no_grad():
        source_sub_t = torch.tensor(source_sub, device=device, dtype=torch.float32).unsqueeze(0)
        pred_vec_source = model(source_t, weights_t, source_sub_t)
        pred_vec_source = pred_vec_source.squeeze(0).cpu().numpy()

    # -------------------------------------------------------------------
    # 4-panel layout: Pred | GT | Arrows | Error
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(21.5, 4.6))
    gs = GridSpec(1, 7, width_ratios=[1, 0.02, 1, 0.02, 1, 0.02, 1], wspace=0.0)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[0, 6])

    # --- Plot 1: Predicted Transport ---
    ax1.scatter(source_points[:, 0], source_points[:, 1],
                c='blue', s=15, alpha=0.5)
    ax1.scatter(query_points[:, 0], query_points[:, 1],
                c='gray', s=8, alpha=0.3, marker='x')
    ax1.scatter(transported_pred[:, 0], transported_pred[:, 1],
                c='red', s=15, alpha=0.6)

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel('X position', fontsize=16)
    ax1.set_ylabel('Y position', fontsize=16)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.xaxis.set_major_locator(MultipleLocator(2.0))

    # --- Plot 2: Ground Truth Transport ---
    ax2.scatter(source_points[:, 0], source_points[:, 1],
                c='blue', s=15, alpha=0.5)
    ax2.scatter(query_points[:, 0], query_points[:, 1],
                c='gray', s=8, alpha=0.3, marker='x')
    ax2.scatter(transported_gt[:, 0], transported_gt[:, 1],
                c='green', s=15, alpha=0.6)

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xlabel('X position', fontsize=16)
    ax2.set_ylabel('Y position', fontsize=16)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.xaxis.set_major_locator(MultipleLocator(2.0))

    # --- Plot 3: Transport Trajectories from Source Points ---
    # Show all source points as small dots for context
    ax3.scatter(source_points[:, 0], source_points[:, 1],
                c='blue', s=8, alpha=0.3, marker='o', zorder=2)

    # GT transport arrows (green) - from source to transported location
    ax3.quiver(source_sub[:, 0], source_sub[:, 1],
               gt_vec_source[:, 0], gt_vec_source[:, 1],
               color='green', alpha=0.8, angles='xy', scale_units='xy', scale=1,
               width=0.008, headwidth=3, headlength=4, zorder=3)

    # Pred transport arrows (red)
    ax3.quiver(source_sub[:, 0], source_sub[:, 1],
               pred_vec_source[:, 0], pred_vec_source[:, 1],
               color='red', alpha=0.7, angles='xy', scale_units='xy', scale=1,
               width=0.008, headwidth=3, headlength=4, zorder=4)

    # Highlight the subsampled source points
    ax3.scatter(source_sub[:, 0], source_sub[:, 1],
                c='blue', s=25, alpha=0.8, marker='o', zorder=5)

    # Zoom to region containing arrows (source points + displacement vectors)
    arrow_endpoints = source_sub + np.maximum(np.abs(gt_vec_source), np.abs(pred_vec_source)) * np.sign(gt_vec_source)
    all_arrow_points = np.vstack([source_sub, source_sub + gt_vec_source, source_sub + pred_vec_source])
    arrow_xmin, arrow_ymin = all_arrow_points.min(axis=0)
    arrow_xmax, arrow_ymax = all_arrow_points.max(axis=0)
    arrow_pad = max(arrow_xmax - arrow_xmin, arrow_ymax - arrow_ymin) * 0.15
    arrow_center_x = (arrow_xmin + arrow_xmax) / 2
    arrow_center_y = (arrow_ymin + arrow_ymax) / 2
    arrow_range = max(arrow_xmax - arrow_xmin, arrow_ymax - arrow_ymin) / 2 + arrow_pad
    ax3.set_xlim(arrow_center_x - arrow_range, arrow_center_x + arrow_range)
    ax3.set_ylim(arrow_center_y - arrow_range, arrow_center_y + arrow_range)
    ax3.xaxis.set_major_locator(MultipleLocator(2.0))
    ax3.yaxis.set_major_locator(MultipleLocator(2.0))
    ax3.set_xlabel('X position', fontsize=16)
    ax3.set_ylabel('Y position', fontsize=16)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    # --- Plot 4: Error Magnitude at Query Points ---
    sc = ax4.scatter(query_points[:, 0], query_points[:, 1],
                     c=error_mag, cmap='magma', s=15, alpha=0.85)
    ax4.scatter(source_points[:, 0], source_points[:, 1],
                c='blue', s=10, alpha=0.3, marker='s')

    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim)
    ax4.set_xlabel('X position', fontsize=16)
    ax4.set_ylabel('Y position', fontsize=16)
    ax4.set_aspect('equal', adjustable='box')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    ax4.xaxis.set_major_locator(MultipleLocator(2.0))

    # Tighten layout
    fig.tight_layout()

    # Error colorbar
    pos_err = ax4.get_position()
    gap_cb = 0.015
    cb_width = 0.015

    left_err = pos_err.x1 + gap_cb
    bottom = pos_err.y0
    height = pos_err.height

    cbar_ax = fig.add_axes([left_err, bottom, cb_width, height])
    err_cbar = fig.colorbar(sc, cax=cbar_ax, format='%.2f')
    err_cbar.set_label('|Error|', rotation=270, labelpad=20, fontsize=16)
    err_cbar.ax.tick_params(labelsize=16)

    # Redraw the canvas
    fig.canvas.draw()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved transport-Q overlay plot to {save_path}")

    model.train()
    return fig
