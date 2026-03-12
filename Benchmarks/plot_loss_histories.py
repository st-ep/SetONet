#!/usr/bin/env python3
"""
Generate publication-quality loss-history figures from exported CSV logs.

The loader supports both of the CSV layouts used in this repository:

1. Merged exports:
   logs_csv/<benchmark>/<model>/eval_mse.csv
   with columns: step, seed_0, seed_1, ..., seed_4

2. Per-seed exports (current workspace layout):
   logs_csv/<benchmark>/<model>/seed_<n>/eval_mse.csv
   with columns: step, wall_time, value

For each requested figure, the script creates either a standalone panel or a
side-by-side benchmark pair with one mean curve per model and an uncertainty band
based on one standard deviation in log10 space across seeds at each step.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext, LogLocator, MaxNLocator


EXPECTED_SEED_COUNT = 5
CSV_FILENAME = "eval_mse.csv"
DEFAULT_DPI = 400
DEFAULT_FIGSIZE = (4.8, 4.8)
BAND_ALPHA = 0.14
UNIFORM_LINEWIDTH = 1.75

FIGURE_SPECS = [
    ("derivative_fixed_variable", ("1d_derivative", "1d_derivative_varsens")),
    ("integral_fixed_variable", ("1d_integral", "1d_integral_varsens")),
    ("darcy_elastic", ("darcy_1d", "elastic_2d")),
    ("heat_M10_M30", ("heat_2d_P10", "heat_2d_P30")),
    ("concentration_diffraction", ("concentration_2d", "diffraction_2d")),
    ("transport", ("transport",)),
]

BENCHMARK_SUBTITLES = {
    "1d_derivative": "Derivative (Fixed)",
    "1d_derivative_varsens": "Derivative (Variable)",
    "1d_integral": "Integral (Fixed)",
    "1d_integral_varsens": "Integral (Variable)",
    "darcy_1d": "Darcy 1D",
    "elastic_2d": "Elastic Plate",
    "heat_2d_P10": "Heat (M=10)",
    "heat_2d_P30": "Heat (M=30)",
    "concentration_2d": "Advection Diffusion",
    "diffraction_2d": "Diffraction",
    "transport": "Optimal Transport",
}

MODEL_ORDER = [
    "setonet_quadrature",
    "setonet_attention",
    "setonet_mean",
    "setonet_sum",
    "deeponet",
    "vidon",
]

MODEL_LABELS = {
    "deeponet": "DeepONet",
    "vidon": "VIDON",
    "setonet_attention": "SetONet (Attention)",
    "setonet_mean": "SetONet (Mean)",
    "setonet_sum": "SetONet (Sum)",
    "setonet_quadrature": "SetONet-Key",
}

MODEL_COLORS = {
    "deeponet": "#7570b3",
    "vidon": "#e7298a",
    "setonet_attention": "#d95f02",
    "setonet_mean": "#7570b3",
    "setonet_sum": "#1f78b4",
    "setonet_quadrature": "#1b9e77",
}

FALLBACK_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#9C755F",
    "#B279A2",
]


@dataclass
class ModelCurve:
    model_name: str
    seed_names: list[str]
    steps: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def should_exclude_model(benchmark_name: str, model_name: str) -> bool:
    """Apply benchmark-specific figure exclusions."""
    if model_name == "deeponet":
        return ("varsens" in benchmark_name) or ("robust" in benchmark_name)

    if model_name == "setonet_attention":
        return benchmark_name in {"1d_integral", "1d_derivative"}

    if model_name in {"setonet_mean", "setonet_sum"}:
        excluded_families = ("darcy", "derivative", "integral", "elastic")
        return any(family in benchmark_name for family in excluded_families)

    return False


def configure_matplotlib() -> None:
    """Apply a restrained paper-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "lines.antialiased": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Generate publication-quality benchmark loss-history figures from logs_csv."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "logs_csv",
        help="Directory containing benchmark/model CSV exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "paper_figures" / "loss_histories",
        help="Directory where the figures will be written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"PNG export DPI (default: {DEFAULT_DPI}).",
    )
    return parser.parse_args()


def model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model_name), model_name)
    except ValueError:
        return (len(MODEL_ORDER), model_name)


def pretty_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name.replace("_", " ").title())


def get_model_color(model_name: str) -> str:
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    return FALLBACK_COLORS[hash(model_name) % len(FALLBACK_COLORS)]


def sanitize_benchmark_name(benchmark_name: str) -> str:
    return benchmark_name.replace("/", "_")


def parse_numeric(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def load_single_seed_csv(csv_path: Path) -> dict[int, float]:
    """Load a per-seed CSV with columns step, wall_time, value."""
    series: dict[int, float] = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "step" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise ValueError(f"Unexpected per-seed CSV schema in {csv_path}")
        for row in reader:
            step = int(float(row["step"]))
            value = parse_numeric(row["value"])
            if value is not None:
                series[step] = value
    return dict(sorted(series.items()))


def load_merged_csv(csv_path: Path) -> dict[str, dict[int, float]]:
    """Load a merged CSV with columns step, seed_0, ..., seed_4."""
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "step" not in reader.fieldnames:
            raise ValueError(f"Unexpected merged CSV schema in {csv_path}")
        seed_columns = [name for name in reader.fieldnames if name.startswith("seed_")]
        if not seed_columns:
            raise ValueError(f"No seed columns found in merged CSV {csv_path}")

        series_by_seed = {seed_name: {} for seed_name in seed_columns}
        for row in reader:
            step = int(float(row["step"]))
            for seed_name in seed_columns:
                value = parse_numeric(row.get(seed_name, ""))
                if value is not None:
                    series_by_seed[seed_name][step] = value

    return {
        seed_name: dict(sorted(step_values.items()))
        for seed_name, step_values in series_by_seed.items()
        if step_values
    }


def load_model_seed_series(model_dir: Path) -> dict[str, dict[int, float]]:
    """
    Load one model's exported data, inferring the CSV layout automatically.
    """
    merged_csv = model_dir / CSV_FILENAME
    if merged_csv.is_file():
        return load_merged_csv(merged_csv)

    seed_files = sorted(model_dir.glob(f"seed_*/{CSV_FILENAME}"))
    if seed_files:
        return {
            seed_file.parent.name: load_single_seed_csv(seed_file)
            for seed_file in seed_files
        }

    return {}


def aggregate_model_curve(model_name: str, seed_series: dict[str, dict[int, float]]) -> ModelCurve:
    if not seed_series:
        raise ValueError(f"No seed data found for model '{model_name}'")

    seed_names = sorted(seed_series)
    all_steps = sorted({step for series in seed_series.values() for step in series})
    if not all_steps:
        raise ValueError(f"No steps found for model '{model_name}'")

    step_to_idx = {step: idx for idx, step in enumerate(all_steps)}
    matrix = np.full((len(all_steps), len(seed_names)), np.nan, dtype=float)

    for col, seed_name in enumerate(seed_names):
        for step, value in seed_series[seed_name].items():
            matrix[step_to_idx[step], col] = value

    safe_matrix = np.where(np.isfinite(matrix), np.clip(matrix, np.finfo(float).tiny, None), np.nan)
    mean = np.nanmean(safe_matrix, axis=1)

    # Use a multiplicative spread on the log-loss scale so the uncertainty band
    # matches the geometry of the log y-axis and stays strictly positive.
    log_std = np.nanstd(np.log10(safe_matrix), axis=1)
    band_factor = np.power(10.0, log_std)
    lower = mean / band_factor
    upper = mean * band_factor
    valid_rows = np.isfinite(mean) & np.isfinite(lower) & np.isfinite(upper)

    return ModelCurve(
        model_name=model_name,
        seed_names=seed_names,
        steps=np.asarray(all_steps, dtype=float)[valid_rows],
        mean=mean[valid_rows],
        lower=lower[valid_rows],
        upper=upper[valid_rows],
    )


def discover_benchmark_curves(input_dir: Path) -> dict[str, list[ModelCurve]]:
    benchmarks: dict[str, list[ModelCurve]] = {}

    for benchmark_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        model_curves: list[ModelCurve] = []
        for model_dir in sorted(path for path in benchmark_dir.iterdir() if path.is_dir()):
            if should_exclude_model(benchmark_dir.name, model_dir.name):
                continue
            seed_series = load_model_seed_series(model_dir)
            if not seed_series:
                continue
            model_curves.append(aggregate_model_curve(model_dir.name, seed_series))

        if model_curves:
            model_curves.sort(key=lambda curve: model_sort_key(curve.model_name))
            benchmarks[benchmark_dir.name] = model_curves

    return benchmarks


def format_gradient_steps(value: float, _position: float) -> str:
    if abs(value) >= 1000.0:
        scaled = value / 1000.0
        if abs(scaled - round(scaled)) < 1e-8:
            return f"{int(round(scaled))}k"
        return f"{scaled:.1f}k"
    if abs(value - round(value)) < 1e-8:
        return f"{int(round(value))}"
    return f"{value:.0f}"


def legend_ncols(num_curves: int) -> int:
    if num_curves <= 3:
        return 1
    return 2


def reorder_legend_entries(handles, labels, ncol: int):
    """
    Reorder legend handles so the displayed reading order matches the input order.

    Matplotlib fills multi-column legends column-first. We instead want the legend
    to read row-first using the canonical model order.
    """
    if ncol <= 1 or len(handles) <= 1:
        return handles, labels

    items = list(zip(handles, labels))
    nrows = math.ceil(len(items) / ncol)

    grid: list[list[tuple[object, str] | None]] = [[None for _ in range(ncol)] for _ in range(nrows)]
    idx = 0
    for row in range(nrows):
        for col in range(ncol):
            if idx < len(items):
                grid[row][col] = items[idx]
                idx += 1

    reordered: list[tuple[object, str]] = []
    for col in range(ncol):
        for row in range(nrows):
            item = grid[row][col]
            if item is not None:
                reordered.append(item)

    handles_out, labels_out = zip(*reordered)
    return list(handles_out), list(labels_out)


def compute_y_limits(curves: list[ModelCurve]) -> tuple[float, float]:
    positive_values = np.concatenate(
        [
            np.concatenate(
                [
                    curve.mean[curve.mean > 0.0],
                    curve.lower[curve.lower > 0.0],
                    curve.upper[curve.upper > 0.0],
                ]
            )
            for curve in curves
        ]
    )
    min_positive = float(np.min(positive_values))
    max_positive = float(np.max(positive_values))

    log_min = math.log10(min_positive)
    log_max = math.log10(max_positive)
    log_span = max(log_max - log_min, 1.0e-6)

    # Use a modest padding in log space so the curves are not pressed against the
    # axes, but avoid snapping to full decades which creates large empty margins.
    pad_decades = min(max(0.06, 0.08 * log_span), 0.16)
    lower = 10.0 ** (log_min - pad_decades)
    upper = 10.0 ** (log_max + pad_decades)
    if lower == upper:
        upper *= 10.0
    return lower, upper


def compute_legend_overlap_score(ax, curves: list[ModelCurve], legend: Legend) -> float:
    """
    Score how much a legend overlaps plotted data in axes coordinates.

    Lower is better.
    """
    renderer = ax.figure.canvas.get_renderer()
    bbox_axes = legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())

    # Reject placements that extend outside the plotting area.
    if bbox_axes.x0 < 0.0 or bbox_axes.y0 < 0.0 or bbox_axes.x1 > 1.0 or bbox_axes.y1 > 1.0:
        return float("inf")

    margin_x = 0.015
    margin_y = 0.02
    x0 = bbox_axes.x0 - margin_x
    y0 = bbox_axes.y0 - margin_y
    x1 = bbox_axes.x1 + margin_x
    y1 = bbox_axes.y1 + margin_y

    score = 0.0
    for curve in curves:
        for values, weight in ((curve.mean, 1.0), (curve.lower, 0.45), (curve.upper, 0.45)):
            points = np.column_stack(
                [
                    curve.steps,
                    np.clip(values, np.finfo(float).tiny, None),
                ]
            )
            axes_points = ax.transAxes.inverted().transform(ax.transData.transform(points))
            inside = (
                (axes_points[:, 0] >= x0)
                & (axes_points[:, 0] <= x1)
                & (axes_points[:, 1] >= y0)
                & (axes_points[:, 1] <= y1)
            )
            score += weight * float(np.count_nonzero(inside))

    return score


def place_legend_inside_plot(ax, curves: list[ModelCurve]) -> None:
    """Place a frameless legend inside the axes with minimal overlap."""
    candidate_positions = [
        ("upper right", (0.985, 0.985)),
        ("upper left", (0.015, 0.985)),
        ("lower left", (0.015, 0.015)),
        ("lower right", (0.985, 0.015)),
        ("upper center", (0.5, 0.985)),
        ("center right", (0.985, 0.5)),
        ("center left", (0.015, 0.5)),
    ]

    ncol = legend_ncols(len(curves))
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = reorder_legend_entries(handles, labels, ncol)
    best_score = float("inf")
    best_position: tuple[str, tuple[float, float]] | None = None

    for loc, anchor in candidate_positions:
        legend = ax.legend(
            handles,
            labels,
            loc=loc,
            bbox_to_anchor=anchor,
            bbox_transform=ax.transAxes,
            ncol=ncol,
            frameon=False,
            handlelength=2.2,
            handletextpad=0.45,
            columnspacing=1.0,
            borderaxespad=0.0,
            labelspacing=0.35,
        )
        ax.figure.canvas.draw()
        score = compute_legend_overlap_score(ax, curves, legend)
        legend.remove()

        if score < best_score:
            best_score = score
            best_position = (loc, anchor)

    if best_position is None:
        best_position = ("upper right", (0.985, 0.985))

    loc, anchor = best_position
    ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        bbox_transform=ax.transAxes,
        ncol=ncol,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.45,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.35,
    )


def draw_model_curves(ax, curves: list[ModelCurve]) -> None:
    for curve in curves:
        color = get_model_color(curve.model_name)
        band_lower = np.clip(curve.lower, np.finfo(float).tiny, None)
        band_upper = np.clip(curve.upper, np.finfo(float).tiny, None)
        mean = np.clip(curve.mean, np.finfo(float).tiny, None)

        ax.fill_between(
            curve.steps,
            band_lower,
            band_upper,
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            curve.steps,
            mean,
            color=color,
            linewidth=UNIFORM_LINEWIDTH,
            label=pretty_model_name(curve.model_name),
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
            zorder=2,
        )


def style_loss_axis(ax, curves: list[ModelCurve]) -> None:
    max_step = max(float(np.max(curve.steps)) for curve in curves)
    y_lower, y_upper = compute_y_limits(curves)

    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Test MSE")
    ax.set_xlim(0.0, max_step)
    ax.set_ylim(y_lower, y_upper)
    ax.set_yscale("log")
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out", pad=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(format_gradient_steps))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    place_legend_inside_plot(ax, curves)


def add_panel_subtitle(ax, panel_index: int, benchmark_name: str) -> None:
    subtitle = BENCHMARK_SUBTITLES.get(benchmark_name, benchmark_name.replace("_", " ").title())
    panel_tag = chr(ord("a") + panel_index)
    ax.text(
        0.5,
        -0.18,
        f"({panel_tag}) {subtitle}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

def save_figure(
    figure_stem: str,
    benchmark_panels: list[tuple[str, list[ModelCurve]]],
    output_dir: Path,
    dpi: int,
) -> tuple[Path, Path]:
    num_panels = len(benchmark_panels)
    if num_panels == 1:
        figsize = DEFAULT_FIGSIZE
    else:
        figsize = (DEFAULT_FIGSIZE[0] * num_panels, DEFAULT_FIGSIZE[1])

    fig, axes = plt.subplots(1, num_panels, figsize=figsize, squeeze=False)

    for panel_index, (ax, (benchmark_name, curves)) in enumerate(zip(axes[0], benchmark_panels)):
        draw_model_curves(ax, curves)
        style_loss_axis(ax, curves)
        add_panel_subtitle(ax, panel_index, benchmark_name)

    if num_panels == 1:
        fig.subplots_adjust(left=0.14, right=0.98, bottom=0.24, top=0.96)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.24, top=0.96, wspace=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{sanitize_benchmark_name(figure_stem)}_loss_history"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    configure_matplotlib()
    args = parse_arguments()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    benchmarks = discover_benchmark_curves(input_dir)
    if not benchmarks:
        raise RuntimeError(f"No benchmark CSV exports found under {input_dir}")

    print(f"Discovered {len(benchmarks)} benchmark(s) in {input_dir}")
    for benchmark_name, curves in benchmarks.items():
        for curve in curves:
            if len(curve.seed_names) != EXPECTED_SEED_COUNT:
                print(
                    f"Warning: {benchmark_name}/{curve.model_name} has {len(curve.seed_names)} seed(s); "
                    f"expected {EXPECTED_SEED_COUNT}."
                )

    for figure_stem, benchmark_names in FIGURE_SPECS:
        missing = [name for name in benchmark_names if name not in benchmarks]
        if missing:
            raise RuntimeError(
                f"Cannot create '{figure_stem}' because benchmark export(s) are missing: {', '.join(missing)}"
            )

        benchmark_panels = [(benchmark_name, benchmarks[benchmark_name]) for benchmark_name in benchmark_names]
        png_path, pdf_path = save_figure(
            figure_stem=figure_stem,
            benchmark_panels=benchmark_panels,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        print(
            f"Wrote {figure_stem}: {len(benchmark_panels)} panel(s) -> "
            f"{png_path.name}, {pdf_path.name}"
        )


if __name__ == "__main__":
    main()
