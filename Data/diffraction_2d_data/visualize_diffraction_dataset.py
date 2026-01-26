#!/usr/bin/env python
"""
visualize_diffraction_dataset.py
--------------------------------
Publication-quality visualization script for the Phase-Screen Diffraction benchmark dataset.

Loads a HuggingFace dataset saved to disk and generates two distinct figure types
(5 random samples per type by default; 10 total figures), saving both PNG and PDF.

Figure Type A (3-panel): Input phase-bump point cloud + intensity |u|^2 + phase arg(u)
Figure Type B (2×2): Complex-field decomposition: Re(u), Im(u), log10(|u|^2+ε), arg(u) (masked)

Works for uniform-grid datasets:
- Sample has "field" of shape (grid_n, grid_n, 2)

Usage:
  python Data/diffraction_data/visualize_diffraction_dataset.py \
      --data_path Data/diffraction_data/phase_screen_dataset \
      --out_dir Data/diffraction_data_plots \
      --seed 0 \
      --n_per_type 5
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
from datasets import load_from_disk

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -----------------------------
# Styling (journal-quality)
# -----------------------------
def set_publication_style() -> None:
    """Set consistent journal-quality matplotlib styling."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class FieldData:
    """Unified representation of a complex field on a regular grid."""
    u: np.ndarray        # complex, shape (M, M), indexing='ij' (first axis x, second axis y)
    x: np.ndarray        # shape (M,), in [0,1)
    y: np.ndarray        # shape (M,), in [0,1)
    M: int               # grid size


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def robust_eps(x: np.ndarray, rel: float = 1e-6, abs_eps: float = 1e-12) -> float:
    """Compute a robust epsilon for log scaling."""
    xmax = float(np.nanmax(x)) if np.size(x) else 0.0
    return max(abs_eps, rel * xmax)


def get_field_data_from_sample(sample: Dict) -> FieldData:
    """Extract a complex field on a regular grid from a dataset sample."""
    if "field" not in sample:
        raise ValueError("Adaptive diffraction datasets are no longer supported. Expected 'field' in sample.")

    field = np.array(sample["field"], dtype=np.float32)  # (M,M,2)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError(f"Unexpected 'field' shape {field.shape}; expected (M,M,2).")
    M = int(field.shape[0])
    u = field[:, :, 0].astype(np.float32) + 1j * field[:, :, 1].astype(np.float32)
    u = u.astype(np.complex64, copy=False)
    x = np.linspace(0.0, 1.0, M, endpoint=False, dtype=np.float64)
    y = x.copy()
    return FieldData(u=u, x=x, y=y, M=M)


def subsample_bumps_for_plot(
    bumps: np.ndarray,  # (N,4)
    max_bumps: int,
    seed: int,
    sample_idx: int,
) -> np.ndarray:
    """Subsample bumps deterministically for plotting (reproducible)."""
    N = bumps.shape[0]
    if max_bumps <= 0 or N <= max_bumps:
        return bumps

    # Deterministic RNG per (seed, sample_idx) to make subsampling reproducible and stable.
    local_seed = (np.uint64(seed) + np.uint64(sample_idx) * np.uint64(1_000_003)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(local_seed)
    keep = rng.choice(N, size=max_bumps, replace=False)
    keep = np.sort(keep)
    return bumps[keep]


def add_ell_size_legend(ax: plt.Axes, ell: np.ndarray, size_map_func) -> None:
    """Add a compact size legend for ell marker sizes."""
    if ell.size == 0:
        return
    q = np.quantile(ell, [0.2, 0.5, 0.8])
    q = np.unique(np.round(q, decimals=4))
    if q.size == 0:
        return

    handles = []
    labels = []
    for e in q:
        s = float(size_map_func(e))
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="None", markersize=np.sqrt(s),
                                  markerfacecolor="none", markeredgecolor="0.25", markeredgewidth=0.8))
        labels.append(rf"$\ell={e:.3f}$")
    ax.legend(handles, labels, loc="upper right", frameon=True, framealpha=0.9,
              borderpad=0.35, handletextpad=0.5, labelspacing=0.35, fontsize=8.0)


def format_spatial_axes(ax: plt.Axes) -> None:
    """Consistent formatting for spatial axes on [0,1)."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])


def save_figure(fig: plt.Figure, out_dir: str, basename: str, dpi: int) -> None:
    """Save figure as PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------
# Figure Type A (UNCHANGED)
# -----------------------------
def plot_type_a(
    *,
    sample: Dict,
    sample_idx: int,
    field: FieldData,
    bumps: np.ndarray,  # (N,4)
    out_dir: str,
    dpi: int,
    seed: int,
) -> None:
    """Figure Type A: Input bumps + log-intensity + phase (masked low amplitude)."""
    u = field.u
    M = field.M

    I = (u.real.astype(np.float64) ** 2 + u.imag.astype(np.float64) ** 2)
    logI = np.log10(I + robust_eps(I, rel=1e-8, abs_eps=1e-14))
    phase = wrap_to_pi(np.angle(u).astype(np.float64))

    # Phase mask for low amplitude
    Imax = float(np.max(I))
    thr = max(1e-14, 1e-3 * Imax)
    phase_masked = phase.copy()
    phase_masked[I < thr] = np.nan

    # Plot settings
    cmap_phase = plt.get_cmap("twilight").copy()
    cmap_phase.set_bad(color=(1, 1, 1, 0))  # transparent where masked
    cmap_int = plt.get_cmap("magma")

    alpha = wrap_to_pi(bumps[:, 2].astype(np.float64))
    ell = bumps[:, 3].astype(np.float64)

    # Size mapping for ell (robust)
    ell_med = float(np.median(ell))
    ell_med = ell_med if ell_med > 0 else 1.0

    def size_map(e):
        s = 40.0 * (float(e) / ell_med) ** 2
        return float(np.clip(s, 10.0, 180.0))

    sizes = np.array([size_map(e) for e in ell], dtype=np.float64)

    fig = plt.figure(figsize=(12.6, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0])

    # Panel 1: bumps only
    ax0 = fig.add_subplot(gs[0, 0])
    norm_alpha = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    sc0 = ax0.scatter(
        bumps[:, 0], bumps[:, 1],
        c=alpha, s=sizes,
        cmap=cmap_phase, norm=norm_alpha,
        edgecolors="0.20", linewidths=0.4,
        alpha=0.95,
        rasterized=True,
    )
    format_spatial_axes(ax0)
    ax0.set_title("Input phase-bump point cloud")

    # Alpha colorbar (panel 1)
    div0 = make_axes_locatable(ax0)
    cax0 = div0.append_axes("right", size="4.5%", pad=0.06)
    cb0 = fig.colorbar(sc0, cax=cax0)
    cb0.set_label(r"Phase amplitude $\alpha$ [rad]")
    cb0.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb0.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    add_ell_size_legend(ax0, ell, size_map)

    # Panel 2: log intensity with bump overlay
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(
        logI.T, origin="lower", extent=(0, 1, 0, 1),
        cmap=cmap_int, interpolation="nearest",
    )
    ax1.scatter(
        bumps[:, 0], bumps[:, 1],
        s=np.clip(0.35 * sizes, 6.0, 80.0),
        facecolors="none", edgecolors="white", linewidths=0.55, alpha=0.7,
        rasterized=True,
    )
    format_spatial_axes(ax1)
    ax1.set_title(r"Intensity $\log_{10}(|u|^2 + \varepsilon)$ at $t=t_0$")

    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="4.5%", pad=0.06)
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label(r"$\log_{10}(|u|^2 + \varepsilon)$")

    # Panel 3: phase (masked) with bump overlay (subtle)
    ax2 = fig.add_subplot(gs[0, 2])
    phase_ma = np.ma.masked_invalid(phase_masked)
    im2 = ax2.imshow(
        phase_ma.T, origin="lower", extent=(0, 1, 0, 1),
        cmap=cmap_phase, vmin=-np.pi, vmax=np.pi, interpolation="nearest",
    )
    ax2.scatter(
        bumps[:, 0], bumps[:, 1],
        s=np.clip(0.25 * sizes, 5.0, 65.0),
        facecolors="none", edgecolors="0.15", linewidths=0.45, alpha=0.35,
        rasterized=True,
    )
    format_spatial_axes(ax2)
    ax2.set_title(r"Phase $\arg(u)$ (masked where $|u|^2$ is small)")

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="4.5%", pad=0.06)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(r"$\arg(u)$ [rad]")
    cb2.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb2.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    # Title with metadata
    t0 = float(sample.get("t0", np.nan))
    sigma_env = float(sample.get("sigma_env", np.nan))
    kx = float(sample.get("k_x", np.nan))
    ky = float(sample.get("k_y", np.nan))
    mode = "uniform-grid"
    fig.suptitle(
        rf"Phase-Screen Diffraction sample #{sample_idx}  "
        rf"({mode}, bumps: {bumps.shape[0]})   "
        rf"$t_0={t0:g}$, $\sigma_{{env}}={sigma_env:g}$, $k=({kx:g},{ky:g})$",
        y=1.02,
        fontsize=11,
    )

    basename = f"diffraction_typeA_{mode}_idx{sample_idx:05d}"
    save_figure(fig, out_dir, basename, dpi=dpi)


# -----------------------------
# Figure Type B (NEW: Complex-field decomposition 2×2)
# -----------------------------
def plot_type_b(
    *,
    sample: Dict,
    sample_idx: int,
    field: FieldData,
    bumps: np.ndarray,
    out_dir: str,
    dpi: int,
) -> None:
    """Figure Type B: 2×2 complex-field decomposition: Re, Im, log-intensity, phase (masked).

    Overlays bump locations as subtle hollow-circle outlines on all panels.
    Does NOT re-encode bump alpha/ell (Type A already covers that encoding).
    """
    u = field.u
    M = field.M

    re = u.real.astype(np.float64)
    im = u.imag.astype(np.float64)

    I = (re * re + im * im)
    logI = np.log10(I + robust_eps(I, rel=1e-8, abs_eps=1e-14))

    phase = wrap_to_pi(np.angle(u).astype(np.float64))

    # Phase mask for low amplitude
    Imax = float(np.max(I))
    thr = max(1e-14, 1e-3 * Imax)
    phase_masked = phase.copy()
    phase_masked[I < thr] = np.nan

    # Colormaps
    cmap_div = plt.get_cmap("RdBu_r")
    cmap_int = plt.get_cmap("magma")
    cmap_phase = plt.get_cmap("twilight").copy()
    cmap_phase.set_bad(color=(1, 1, 1, 0))  # transparent masked phase

    # Symmetric robust scaling for Re/Im
    q = 0.995
    vlim_re = float(np.quantile(np.abs(re), q)) if np.isfinite(re).any() else 0.0
    vlim_im = float(np.quantile(np.abs(im), q)) if np.isfinite(im).any() else 0.0
    vlim = max(vlim_re, vlim_im, 1e-8)
    norm_div = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    # Subtle bump overlay style (single consistent marker; no re-encoding)
    bump_style = dict(
        s=14.0,
        facecolors="none",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.35,
        rasterized=True,
    )

    fig = plt.figure(figsize=(12.2, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])

    # (1) Re(u)
    im00 = ax00.imshow(
        re.T, origin="lower", extent=(0, 1, 0, 1),
        cmap=cmap_div, norm=norm_div, interpolation="nearest",
    )
    ax00.scatter(bumps[:, 0], bumps[:, 1], **bump_style)
    format_spatial_axes(ax00)
    ax00.set_title("Re(u)")
    div00 = make_axes_locatable(ax00)
    cax00 = div00.append_axes("right", size="4.2%", pad=0.06)
    cb00 = fig.colorbar(im00, cax=cax00)
    cb00.set_label("Re(u)")

    # (2) Im(u)
    im01 = ax01.imshow(
        im.T, origin="lower", extent=(0, 1, 0, 1),
        cmap=cmap_div, norm=norm_div, interpolation="nearest",
    )
    ax01.scatter(bumps[:, 0], bumps[:, 1], **bump_style)
    format_spatial_axes(ax01)
    ax01.set_title("Im(u)")
    div01 = make_axes_locatable(ax01)
    cax01 = div01.append_axes("right", size="4.2%", pad=0.06)
    cb01 = fig.colorbar(im01, cax=cax01)
    cb01.set_label("Im(u)")

    # (3) log10(|u|^2 + ε)
    im10 = ax10.imshow(
        logI.T, origin="lower", extent=(0, 1, 0, 1),
        cmap=cmap_int, interpolation="nearest",
    )
    ax10.scatter(bumps[:, 0], bumps[:, 1], **bump_style)
    format_spatial_axes(ax10)
    ax10.set_title(r"log10(|u|^2 + ε)")
    div10 = make_axes_locatable(ax10)
    cax10 = div10.append_axes("right", size="4.2%", pad=0.06)
    cb10 = fig.colorbar(im10, cax=cax10)
    cb10.set_label(r"$\log_{10}(|u|^2 + \varepsilon)$")

    # (4) arg(u) masked
    phase_ma = np.ma.masked_invalid(phase_masked)
    im11 = ax11.imshow(
        phase_ma.T, origin="lower", extent=(0, 1, 0, 1),
        cmap=cmap_phase, vmin=-np.pi, vmax=np.pi, interpolation="nearest",
    )
    ax11.scatter(bumps[:, 0], bumps[:, 1], **bump_style)
    format_spatial_axes(ax11)
    ax11.set_title("arg(u) (masked)")
    div11 = make_axes_locatable(ax11)
    cax11 = div11.append_axes("right", size="4.2%", pad=0.06)
    cb11 = fig.colorbar(im11, cax=cax11)
    cb11.set_label(r"$\arg(u)$ [rad]")
    cb11.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb11.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])

    # Reduce label clutter (keep axis labels on left column + bottom row)
    for ax in (ax00, ax01):
        ax.set_xlabel("")
    for ax in (ax01, ax11):
        ax.set_ylabel("")

    # Suptitle with metadata (match Type A)
    t0 = float(sample.get("t0", np.nan))
    sigma_env = float(sample.get("sigma_env", np.nan))
    kx = float(sample.get("k_x", np.nan))
    ky = float(sample.get("k_y", np.nan))
    mode = "uniform-grid"
    fig.suptitle(
        rf"Phase-Screen Diffraction sample #{sample_idx}  "
        rf"({mode}, bumps: {bumps.shape[0]})   "
        rf"$t_0={t0:g}$, $\sigma_{{env}}={sigma_env:g}$, $k=({kx:g},{ky:g})$",
        y=1.01,
        fontsize=11,
    )

    basename = f"diffraction_typeB_{mode}_idx{sample_idx:05d}"
    save_figure(fig, out_dir, basename, dpi=dpi)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Phase-Screen Diffraction dataset (publication-quality figures).")
    parser.add_argument("--data_path", type=str, default="Data/diffraction_data/phase_screen_dataset",
                        help="Path to HuggingFace dataset saved with save_to_disk().")
    parser.add_argument("--out_dir", type=str, default="Data/diffraction_data_plots",
                        help="Output directory for figures (PNG + PDF).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed controlling chosen sample indices.")
    parser.add_argument("--n_per_type", type=int, default=5, help="# samples per figure type (total figures = 2*n_per_type).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Dataset split to visualize.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PNG output.")
    parser.add_argument("--max_bumps_to_show", type=int, default=200,
                        help="Max bumps to display per sample (deterministic subsampling if exceeded).")

    args = parser.parse_args()

    set_publication_style()

    print(f"[•] Loading dataset from: {args.data_path}")
    ds = load_from_disk(args.data_path)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found. Available: {list(ds.keys())}")
    split_ds = ds[args.split]
    n_total = len(split_ds)
    if n_total == 0:
        raise ValueError(f"Split '{args.split}' is empty.")

    sample0 = split_ds[0]
    if "field" not in sample0:
        raise ValueError("Adaptive diffraction datasets are no longer supported. Regenerate uniform-grid data.")
    print("[•] Detected dataset mode: uniform-grid")
    print(f"[•] Split '{args.split}': {n_total} samples")

    rng = np.random.default_rng(args.seed)
    n_needed = 2 * int(args.n_per_type)
    replace = n_total < n_needed
    chosen = rng.choice(n_total, size=n_needed, replace=replace)
    chosen = chosen.astype(int).tolist()

    idxs_a = chosen[: args.n_per_type]
    idxs_b = chosen[args.n_per_type :]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[•] Saving figures to: {args.out_dir}")
    print(f"[•] Type A sample indices: {idxs_a}")
    print(f"[•] Type B sample indices: {idxs_b}")

    # Generate Type A figures
    for idx in idxs_a:
        sample = split_ds[idx]
        bumps = np.array(sample["bumps"], dtype=np.float32)  # (N,4) [x,y,alpha,ell]
        bumps_plot = subsample_bumps_for_plot(bumps, args.max_bumps_to_show, seed=args.seed, sample_idx=idx)
        field = get_field_data_from_sample(sample)
        plot_type_a(
            sample=sample,
            sample_idx=idx,
            field=field,
            bumps=bumps_plot,
            out_dir=args.out_dir,
            dpi=args.dpi,
            seed=args.seed,
        )
        print(f"  ✓ Type A saved for idx={idx}")

    # Generate Type B figures
    for idx in idxs_b:
        sample = split_ds[idx]
        bumps = np.array(sample["bumps"], dtype=np.float32)
        bumps_plot = subsample_bumps_for_plot(bumps, args.max_bumps_to_show, seed=args.seed, sample_idx=idx)
        field = get_field_data_from_sample(sample)
        plot_type_b(
            sample=sample,
            sample_idx=idx,
            field=field,
            bumps=bumps_plot,
            out_dir=args.out_dir,
            dpi=args.dpi,
        )
        print(f"  ✓ Type B saved for idx={idx}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
