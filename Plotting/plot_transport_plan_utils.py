#!/usr/bin/env python
"""plot_transport_plan_utils.py
----------------------------------
Plotting utilities for transport-plan results visualization.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch


def _normalize_rows(w, eps=1e-12):
    w = torch.clamp(w, min=0.0)
    row_sums = w.sum(dim=1, keepdim=True).clamp_min(eps)
    return w / row_sums


def _select_source_indices(source_np, n_pick=3):
    if n_pick <= 0:
        return []
    order = np.argsort(source_np[:, 0])
    pick = np.linspace(0, len(order) - 1, n_pick).astype(int)
    return [int(order[i]) for i in pick]


def plot_transport_barycentric_results(
    model,
    dataset,
    device,
    sample_idx=0,
    save_path=None,
    dataset_split="test",
    mode="plan",
):
    """Plot predicted vs true barycentric maps and error."""
    model.eval()

    data_split = dataset[dataset_split]
    if sample_idx >= len(data_split):
        sample_idx = 0

    sample = data_split[sample_idx]
    source_points = torch.tensor(np.array(sample["source_points"]), device=device, dtype=torch.float32)
    target_points = torch.tensor(np.array(sample["target_points"]), device=device, dtype=torch.float32)
    plan = torch.tensor(np.array(sample["transport_plan"]), device=device, dtype=torch.float32)

    xs = source_points.unsqueeze(0)
    us = torch.ones(1, source_points.shape[0], 1, device=device, dtype=torch.float32)
    ys = xs

    with torch.no_grad():
        pred = model(xs, us, ys)

    w_true = _normalize_rows(plan)
    if mode == "barycentric":
        pred_bary = pred.squeeze(0)
    else:
        w_pred = torch.softmax(pred, dim=-1).squeeze(0)
        pred_bary = w_pred @ target_points

    true_bary = w_true @ target_points

    pred_bary_np = pred_bary.cpu().numpy()
    true_bary_np = true_bary.cpu().numpy()
    source_np = source_points.cpu().numpy()
    err = np.linalg.norm(pred_bary_np - true_bary_np, axis=1)

    domain_size = float(sample.get("domain_size", 5.0))
    padding = 0.2
    xlim = (-domain_size - padding, domain_size + padding)
    ylim = (-domain_size - padding, domain_size + padding)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax1 = axes[0]
    ax1.scatter(source_np[:, 0], source_np[:, 1], c="blue", s=18, alpha=0.7, label="Source")
    ax1.scatter(pred_bary_np[:, 0], pred_bary_np[:, 1], c="red", s=18, alpha=0.7, label="Predicted")
    ax1.set_title("Predicted Barycentric")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2 = axes[1]
    ax2.scatter(source_np[:, 0], source_np[:, 1], c="blue", s=18, alpha=0.7, label="Source")
    ax2.scatter(true_bary_np[:, 0], true_bary_np[:, 1], c="green", s=18, alpha=0.7, label="True")
    ax2.set_title("True Barycentric")
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    ax3 = axes[2]
    sc = ax3.scatter(source_np[:, 0], source_np[:, 1], c=err, cmap="magma", s=22, alpha=0.85)
    ax3.set_title("Barycentric Error")
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="Error magnitude")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved barycentric plot to {save_path}")

    model.train()
    return fig


def plot_transport_row_distribution_results(
    model,
    dataset,
    device,
    sample_idx=0,
    save_path=None,
    dataset_split="test",
    mode="plan",
    n_sources=3,
):
    """Plot row distributions over target geometry for selected source points."""
    if mode == "barycentric":
        print("Skipping row distribution plot for barycentric mode.")
        return None

    model.eval()

    data_split = dataset[dataset_split]
    if sample_idx >= len(data_split):
        sample_idx = 0

    sample = data_split[sample_idx]
    source_points = torch.tensor(np.array(sample["source_points"]), device=device, dtype=torch.float32)
    target_points = torch.tensor(np.array(sample["target_points"]), device=device, dtype=torch.float32)
    plan = torch.tensor(np.array(sample["transport_plan"]), device=device, dtype=torch.float32)

    xs = source_points.unsqueeze(0)
    us = torch.ones(1, source_points.shape[0], 1, device=device, dtype=torch.float32)
    ys = xs

    with torch.no_grad():
        pred = model(xs, us, ys)

    w_true = _normalize_rows(plan)
    w_pred = torch.softmax(pred, dim=-1).squeeze(0)

    w_true_np = w_true.cpu().numpy()
    w_pred_np = w_pred.cpu().numpy()
    source_np = source_points.cpu().numpy()
    target_np = target_points.cpu().numpy()

    source_indices = _select_source_indices(source_np, n_pick=n_sources)

    weights_all = np.concatenate([w_true_np[source_indices], w_pred_np[source_indices]], axis=0).ravel()
    weights_all = weights_all[weights_all > 0.0]
    vmin = float(np.percentile(weights_all, 5)) if weights_all.size else 1e-8
    vmax = float(np.percentile(weights_all, 99.5)) if weights_all.size else 1.0
    vmin = max(vmin, 1e-8)
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10.0))

    domain_size = float(sample.get("domain_size", 5.0))
    padding = 0.2
    xlim = (-domain_size - padding, domain_size + padding)
    ylim = (-domain_size - padding, domain_size + padding)

    fig, axes = plt.subplots(2, len(source_indices), figsize=(5 * len(source_indices), 10))
    if len(source_indices) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, src_idx in enumerate(source_indices):
        ax = axes[0, col]
        weights = w_true_np[src_idx]
        sc = ax.scatter(
            target_np[:, 0],
            target_np[:, 1],
            c=weights,
            cmap="viridis",
            norm=norm,
            s=18,
            alpha=0.85,
        )
        ax.scatter(source_np[src_idx, 0], source_np[src_idx, 1], c="black", s=70, marker="*")
        ax.set_title(f"True W | src {src_idx}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1, col]
        weights = w_pred_np[src_idx]
        sc = ax.scatter(
            target_np[:, 0],
            target_np[:, 1],
            c=weights,
            cmap="viridis",
            norm=norm,
            s=18,
            alpha=0.85,
        )
        ax.scatter(source_np[src_idx, 0], source_np[src_idx, 1], c="black", s=70, marker="*")
        ax.set_title(f"Pred W | src {src_idx}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02, label="Row probability")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved row distribution plot to {save_path}")

    model.train()
    return fig


def plot_transport_plan_results(
    model,
    dataset,
    device,
    sample_idx=0,
    save_path=None,
    dataset_split="test",
    mode="plan",
):
    """Deprecated wrapper for backward compatibility."""
    return plot_transport_barycentric_results(
        model,
        dataset,
        device,
        sample_idx=sample_idx,
        save_path=save_path,
        dataset_split=dataset_split,
        mode=mode,
    )
