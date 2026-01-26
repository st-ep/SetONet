#!/usr/bin/env python
"""plot_diffraction_2d_utils.py
--------------------------------
Plotting utilities for 2D diffraction problem results visualization.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _phase_screen_from_bumps(bumps: np.ndarray, X: np.ndarray, Y: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    if bumps.size == 0:
        return np.zeros_like(X, dtype=np.float64)

    bx = bumps[:, 0].astype(np.float64)
    by = bumps[:, 1].astype(np.float64)
    alpha = bumps[:, 2].astype(np.float64)
    ell = bumps[:, 3].astype(np.float64)

    phi = np.zeros_like(X, dtype=np.float64)
    X64 = X.astype(np.float64, copy=False)
    Y64 = Y.astype(np.float64, copy=False)

    n_bumps = bx.shape[0]
    for s in range(0, n_bumps, chunk_size):
        e = min(s + chunk_size, n_bumps)
        bxs = bx[s:e][None, None, :]
        bys = by[s:e][None, None, :]
        alphas = alpha[s:e][None, None, :]
        ells = ell[s:e][None, None, :]

        dx = np.mod(X64[:, :, None] - bxs + 0.5, 1.0) - 0.5
        dy = np.mod(Y64[:, :, None] - bys + 0.5, 1.0) - 0.5
        r2 = dx * dx + dy * dy
        denom = 2.0 * (ells * ells) + 1e-18
        phi += np.sum(alphas * np.exp(-r2 / denom), axis=-1)

    phi_mod = (phi + np.pi) % (2.0 * np.pi) - np.pi
    return phi_mod


def plot_diffraction_results(
    model,
    dataset,
    diffraction_dataset,
    device,
    sample_idx: int = 0,
    save_path: str | None = None,
    dataset_split: str = "test",
):
    """
    Plot diffraction 2D results: Re/Im predictions, ground truth, and signed errors.

    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset
        diffraction_dataset: DiffractionDataset wrapper
        device: PyTorch device
        sample_idx: Index of sample to plot
        save_path: Path to save the plot
        dataset_split: 'train' or 'test'
    """
    model.eval()

    data = dataset[dataset_split]
    sample = data[sample_idx]

    bumps = np.array(sample["bumps"])
    field = np.array(sample["field"])
    grid_n = field.shape[0]

    x = np.linspace(0.0, 1.0, grid_n, endpoint=False)
    y = x
    X, Y = np.meshgrid(x, y, indexing="ij")
    viz_coords = np.column_stack([X.reshape(-1), Y.reshape(-1)])

    re_gt = field[:, :, 0]
    im_gt = field[:, :, 1]

    with torch.no_grad():
        bump_coords = torch.tensor(bumps[:, :2], device=device, dtype=torch.float32).unsqueeze(0)
        bump_feats = torch.tensor(bumps[:, 2:4], device=device, dtype=torch.float32).unsqueeze(0)
        target_coords = torch.tensor(viz_coords, device=device, dtype=torch.float32).unsqueeze(0)

        pred = model(bump_coords, bump_feats, target_coords)
        pred_field = pred.squeeze(0).cpu().numpy().reshape(grid_n, grid_n, 2)

    re_pred = pred_field[:, :, 0]
    im_pred = pred_field[:, :, 1]

    err_re = np.abs(re_gt - re_pred)
    err_im = np.abs(im_gt - im_pred)

    vlim_re = max(np.max(np.abs(re_gt)), np.max(np.abs(re_pred)), 1e-8)
    vlim_im = max(np.max(np.abs(im_gt)), np.max(np.abs(im_pred)), 1e-8)
    err_lim_re = max(np.max(err_re), 1e-8)
    err_lim_im = max(np.max(err_im), 1e-8)

    ell = bumps[:, 3] if bumps.shape[1] > 3 else np.zeros((bumps.shape[0],), dtype=np.float32)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
    cmap = "RdBu_r"
    err_cmap = "Reds"

    phi_mod = _phase_screen_from_bumps(bumps, X, Y)
    phi_plot = phi_mod.T
    xt = X.T
    yt = Y.T

    def draw_input(ax):
        im = ax.imshow(
            phi_plot,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.contour(xt, yt, phi_plot, levels=12, colors="black", linewidths=0.3, alpha=0.35)
        for x0, y0, _, ell in bumps:
            ax.add_patch(
                Circle((float(x0), float(y0)), float(ell), edgecolor="white", facecolor="none", linewidth=0.4, alpha=0.45)
            )
        ax.scatter(
            bumps[:, 0],
            bumps[:, 1],
            s=66,
            c="white",
            edgecolors="black",
            linewidths=0.3,
            alpha=0.9,
        )
        ax.set_title("Input phase screen Φ(x,y)")
        ax.set_xlabel("X position")
        ax.set_ylabel("Y position")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cb_in = fig.colorbar(im, cax=cax)
        cb_in.set_label("Phase [rad]")
        cb_in.set_ticks([-np.pi / 2, 0, np.pi / 2])
        cb_in.set_ticklabels([r"$-\pi/2$", r"$0$", r"$\pi/2$"])

    draw_input(axes[0, 0])
    draw_input(axes[1, 0])

    ax = axes[0, 1]
    im = ax.imshow(re_pred.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=-vlim_re, vmax=vlim_re)
    ax.set_title("Re(u) prediction")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    ax = axes[0, 2]
    im = ax.imshow(re_gt.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=-vlim_re, vmax=vlim_re)
    ax.set_title("Re(u) ground truth")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    ax = axes[0, 3]
    im = ax.imshow(err_re.T, origin="lower", extent=(0, 1, 0, 1), cmap=err_cmap, vmin=0.0, vmax=err_lim_re)
    ax.set_title("|Re(u) error|")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    ax = axes[1, 1]
    im = ax.imshow(im_pred.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=-vlim_im, vmax=vlim_im)
    ax.set_title("Im(u) prediction")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    ax = axes[1, 2]
    im = ax.imshow(im_gt.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=-vlim_im, vmax=vlim_im)
    ax.set_title("Im(u) ground truth")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    ax = axes[1, 3]
    im = ax.imshow(err_im.T, origin="lower", extent=(0, 1, 0, 1), cmap=err_cmap, vmin=0.0, vmax=err_lim_im)
    ax.set_title("|Im(u) error|")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(im, cax=cax)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved diffraction results plot to {save_path}")

    return fig
