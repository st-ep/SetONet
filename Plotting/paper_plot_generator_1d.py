#!/usr/bin/env python
"""paper_plot_generator_1d.py - Generate 1D paper figures (Darcy + synthetic).

Creates figures with 3 rows (Fixed, Variable, Drop-off) and 5 columns:
Input, Model 1, Model 2, Model 3, Residuals.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from Data.data_utils import apply_sensor_dropoff, sample_trunk_points
from Plotting.plot_burgers_1d_utils import interpolate_to_sensors
from Plotting.paper_plot_config_1d import (
    BENCHMARK_CONFIGS,
    DEFAULT_LOGS_DIR,
    DEFAULT_N_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    MODEL_STYLES,
    ROW_CONFIGS_BY_BENCHMARK,
)
from Plotting.paper_plot_utils_1d import (
    eval_synthetic_function,
    infer_sensor_and_query_counts,
    load_dataset_for_benchmark,
    load_models_for_rows,
    sample_sensor_points,
    sample_synthetic_coeffs,
)


PROJECT_ROOT = Path(__file__).parent.parent


def _to_1d(arr) -> np.ndarray:
    return np.asarray(arr).squeeze()


def _denormalize(values: torch.Tensor, stats: dict | None, mean_key: str, std_key: str) -> np.ndarray:
    if stats is None:
        return values.detach().cpu().numpy()
    return (values * stats[std_key] + stats[mean_key]).detach().cpu().numpy()


def _plot_missing(ax, label: str):
    ax.text(0.5, 0.5, f"Missing {label}", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _set_three_yticks(ax, ymin: float, ymax: float):
    if ymax == ymin:
        pad = abs(ymin) * 0.1 if ymin != 0 else 1.0
        ymin -= pad
        ymax += pad
    if ymax > ymin:
        span = ymax - ymin
        ticks = [ymin + span * frac for frac in (0.25, 0.5, 0.75)]
    else:
        ticks = [ymin - 1.0, ymin, ymin + 1.0]
    ax.set_yticks(ticks)


def _display_label(label: str) -> str:
    return label.replace("SetONet (Quadrature)", "SetONet-Key")


def _plot_row(axs, row_payload: dict, bench_cfg: dict):
    ax_input, ax_pred1, ax_pred2, ax_pred3, ax_res = axs
    pred_keys = row_payload["pred_keys"]
    pred_series = row_payload["pred_series"]

    input_x = _to_1d(row_payload["input_x"])
    input_vals = _to_1d(row_payload["input_vals"])
    x_query = _to_1d(row_payload["x_query"])
    s_true = _to_1d(row_payload["s_true"])

    ax_input.plot(input_x, input_vals, color="black", linewidth=2)
    ax_input.set_xlabel("x")
    ax_input.set_ylabel(bench_cfg["input_label"])
    ax_input.grid(True, alpha=0.3)
    input_ymin = float(np.min(input_vals))
    input_ymax = float(np.max(input_vals))
    input_pad = 0.05 * (input_ymax - input_ymin) if input_ymax > input_ymin else 1.0
    ax_input.set_ylim(input_ymin - input_pad, input_ymax + input_pad)
    _set_three_yticks(ax_input, input_ymin - input_pad, input_ymax + input_pad)
    ax_input.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    outputs = [s_true]
    for key in pred_keys:
        vals = pred_series.get(key)
        if vals is not None:
            outputs.append(_to_1d(vals))
    y_min = min(val.min() for val in outputs)
    y_max = max(val.max() for val in outputs)
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    pred_axes = [ax_pred1, ax_pred2, ax_pred3]
    for ax, pred_key in zip(pred_axes, pred_keys):
        ax.plot(x_query, s_true, color="black", linewidth=2)
        pred_vals = pred_series.get(pred_key)
        if pred_vals is None:
            _plot_missing(ax, _display_label(MODEL_STYLES[pred_key]["label"]))
        else:
            ax.plot(
                x_query,
                _to_1d(pred_vals),
                color=MODEL_STYLES[pred_key]["color"],
                linewidth=2,
            )
        ax.set_xlabel("x")
        ax.set_ylabel(bench_cfg["output_label"])
        ax.set_xlim(x_query.min(), x_query.max())
        ax.set_ylim(y_min - pad, y_max + pad)
        _set_three_yticks(ax, y_min - pad, y_max + pad)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)

    residuals = {}
    for key in pred_keys:
        pred_vals = pred_series.get(key)
        if pred_vals is not None:
            residuals[key] = _to_1d(pred_vals) - s_true

    if residuals:
        res_min = min(val.min() for val in residuals.values())
        res_max = max(val.max() for val in residuals.values())
        res_pad = 0.05 * (res_max - res_min) if res_max > res_min else 1.0
    else:
        res_min, res_max, res_pad = -1.0, 1.0, 0.1

    ax_res.axhline(0.0, color="black", linewidth=1, alpha=0.6)
    for key, vals in residuals.items():
        ax_res.plot(
            x_query,
            vals,
            color=MODEL_STYLES[key]["color"],
            linewidth=2,
            label=_display_label(MODEL_STYLES[key]["label"]),
        )
    ax_res.set_xlabel("x")
    ax_res.set_ylabel("pred - true")
    ax_res.set_xlim(x_query.min(), x_query.max())
    ax_res.set_ylim(res_min - res_pad, res_max + res_pad)
    _set_three_yticks(ax_res, res_min - res_pad, res_max + res_pad)
    ax_res.grid(True, alpha=0.3)

    max_abs = max(abs(res_min), abs(res_max))
    scale_exp = int(np.floor(np.log10(max_abs))) if max_abs > 0 else 0
    scale = 10 ** scale_exp

    ax_res.yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val / scale:.2f}"))
    ax_res.text(
        -0.28,
        1.04,
        rf"$\times 10^{{{scale_exp}}}$",
        transform=ax_res.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )

    for ax in axs:
        ax.set_box_aspect(1)


def _render_figure(row_payloads: list[dict], bench_cfg: dict, output_path: Path):
    fig = plt.figure(figsize=(19.5, 10.64))
    gs = fig.add_gridspec(3, 5, wspace=0.35, hspace=0.65)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(3)]

    for r, row_payload in enumerate(row_payloads):
        _plot_row(axes[r], row_payload, bench_cfg)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)

    for r, row_payload in enumerate(row_payloads):
        row_axes = axes[r]
        row_y0 = min(ax.get_position().y0 for ax in row_axes)
        row_y1 = max(ax.get_position().y1 for ax in row_axes)
        row_center = (row_y0 + row_y1) / 2
        fig.text(
            0.012,
            row_center,
            row_payload["title"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

        label_y = row_y0 - 0.07
        label_y = max(label_y, 0.01)
        for ax, label in zip(row_axes, row_payload["col_labels"]):
            pos = ax.get_position()
            fig.text(
                (pos.x0 + pos.x1) / 2,
                label_y,
                _display_label(label),
                ha="center",
                va="top",
                fontsize=18,
                fontweight="bold",
            )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="1D paper figure generator (Darcy + synthetic)")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["darcy_1d", "1d_derivative", "1d_integral"],
    )
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logs_dir", default=DEFAULT_LOGS_DIR)
    parser.add_argument(
        "--darcy_data_path",
        default=BENCHMARK_CONFIGS["darcy_1d"]["data_path"],
        help="Path to Darcy 1D dataset (load_from_disk format).",
    )
    args = parser.parse_args()

    logs_root = PROJECT_ROOT / args.logs_dir

    device = args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"

    print(f"Device: {device}")

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.titlesize": 16,
    })

    for benchmark in args.benchmarks:
        if benchmark == "burgers_1d":
            print("Skipping burgers_1d: this generator no longer includes Burgers.")
            continue
        if benchmark not in BENCHMARK_CONFIGS:
            print(f"Unknown benchmark: {benchmark}")
            continue

        print(f"\nProcessing {benchmark}...")
        bench_cfg = BENCHMARK_CONFIGS[benchmark]
        row_configs = ROW_CONFIGS_BY_BENCHMARK[benchmark]

        output_root = PROJECT_ROOT / args.output_dir / benchmark
        output_root.mkdir(parents=True, exist_ok=True)

        if bench_cfg["data_kind"] == "synthetic":
            input_range = bench_cfg["input_range"]
            scale = bench_cfg["scale"]
            sensor_size, n_query_points = infer_sensor_and_query_counts(logs_root, benchmark)
            query_x = sample_trunk_points(n_query_points, input_range, device)

            print("Loading models...")
            row_models = load_models_for_rows(row_configs, bench_cfg, logs_root, device)

            sample_indices = list(range(args.n_samples))
            for plot_idx, sample_idx in enumerate(sample_indices, start=1):
                coeffs = sample_synthetic_coeffs(args.seed + sample_idx, scale, device)
                f_query, fprime_query = eval_synthetic_function(query_x, coeffs)

                if benchmark == "1d_derivative":
                    input_vals = f_query
                    s_true = fprime_query
                else:
                    input_vals = fprime_query
                    s_true = f_query

                input_vals_np = _denormalize(input_vals, None, "", "").squeeze()
                s_true_np = _denormalize(s_true, None, "", "").squeeze()

                x_query = query_x.detach().cpu().numpy().squeeze()
                row_payloads = []
                for r, row_cfg in enumerate(row_configs):
                    pred_series = {}
                    for pred_key in row_cfg["pred_keys"]:
                        model_info = row_models[r].get(pred_key)
                        if model_info is None or model_info.get("model") is None:
                            pred_series[pred_key] = None
                            continue
                        model = model_info["model"]
                        model_sensor_size = model_info.get("n_force_points") or sensor_size

                        if row_cfg.get("variable_sensors", False):
                            sensor_seed = args.seed + sample_idx + r * 1000
                        else:
                            sensor_seed = args.seed

                        sensor_x = sample_sensor_points(
                            model_sensor_size, input_range, device, sensor_seed
                        )
                        f_sensors, fprime_sensors = eval_synthetic_function(sensor_x, coeffs)
                        if benchmark == "1d_derivative":
                            u_sensors = f_sensors.squeeze(-1)
                        else:
                            u_sensors = fprime_sensors.squeeze(-1)

                        torch.manual_seed(args.seed + sample_idx + r * 1000)
                        if row_cfg["eval_dropoff"] > 0.0:
                            sensor_x_used, u_sensors_used = apply_sensor_dropoff(
                                sensor_x,
                                u_sensors,
                                row_cfg["eval_dropoff"],
                                row_cfg["replace_with_nearest"],
                            )
                        else:
                            sensor_x_used, u_sensors_used = sensor_x, u_sensors

                        xs_input = sensor_x_used.unsqueeze(0)
                        us_input = u_sensors_used.unsqueeze(0).unsqueeze(-1)
                        ys_input = query_x.unsqueeze(0)

                        with torch.no_grad():
                            pred = model(xs_input, us_input, ys_input).squeeze(0).squeeze(-1)
                        pred_series[pred_key] = _denormalize(pred, None, "", "")

                    row_payloads.append({
                        "title": row_cfg["title"],
                        "col_labels": row_cfg["col_labels"],
                        "pred_keys": row_cfg["pred_keys"],
                        "input_x": x_query,
                        "input_vals": input_vals_np,
                        "x_query": x_query,
                        "s_true": s_true_np,
                        "pred_series": pred_series,
                    })

                base = f"fig_{benchmark}_sample_{sample_idx}"
                _render_figure(row_payloads, bench_cfg, output_root / f"{base}.png")
                print(f"Saved {base}.png ({plot_idx}/{len(sample_indices)})")
            continue

        dataset, stats = load_dataset_for_benchmark(benchmark, device, args.darcy_data_path)
        grid_points = torch.tensor(dataset["test"][0]["X"], device=device, dtype=torch.float32)

        _, n_query_points = infer_sensor_and_query_counts(logs_root, benchmark)
        query_x, query_indices = bench_cfg["query_points_fn"]({}, device, grid_points, n_query_points)

        print("Loading models...")
        row_models = load_models_for_rows(row_configs, bench_cfg, logs_root, device, grid_points)

        rng = np.random.default_rng(args.seed)
        all_indices = np.arange(len(dataset["test"]))
        rng.shuffle(all_indices)
        sample_indices = all_indices[: min(args.n_samples, len(all_indices))]

        for plot_idx, sample_idx in enumerate(sample_indices, start=1):
            sample_idx = int(sample_idx)
            sample = dataset["test"][sample_idx]
            u_full = torch.tensor(sample["u"], device=device, dtype=torch.float32)
            s_full = torch.tensor(sample["s"], device=device, dtype=torch.float32)

            s_true = s_full[query_indices]

            u_full_denorm = _denormalize(u_full, stats, "u_mean", "u_std")
            s_true_denorm = _denormalize(s_true, stats, "s_mean", "s_std")
            x_full = grid_points.detach().cpu().numpy()
            x_query = query_x.detach().cpu().numpy().squeeze()

            row_payloads = []
            for r, row_cfg in enumerate(row_configs):
                pred_series = {}
                for pred_key in row_cfg["pred_keys"]:
                    model_info = row_models[r].get(pred_key)
                    if model_info is None or model_info.get("model") is None:
                        pred_series[pred_key] = None
                        continue
                    model = model_info["model"]
                    model_sensor_x = model_info["sensor_x"]
                    model_sensor_indices = model_info["sensor_indices"]

                    if model_sensor_indices is not None:
                        u_sensors = u_full[model_sensor_indices]
                    else:
                        u_sensors = interpolate_to_sensors(
                            u_full.unsqueeze(0), grid_points, model_sensor_x
                        ).squeeze(0)

                    torch.manual_seed(args.seed + sample_idx + r * 1000)
                    if row_cfg["eval_dropoff"] > 0.0:
                        sensor_x_used, u_sensors_used = apply_sensor_dropoff(
                            model_sensor_x,
                            u_sensors,
                            row_cfg["eval_dropoff"],
                            row_cfg["replace_with_nearest"],
                        )
                    else:
                        sensor_x_used, u_sensors_used = model_sensor_x, u_sensors

                    xs_input = sensor_x_used.unsqueeze(0)
                    us_input = u_sensors_used.unsqueeze(0).unsqueeze(-1)
                    ys_input = query_x.unsqueeze(0)

                    with torch.no_grad():
                        pred = model(xs_input, us_input, ys_input).squeeze(0).squeeze(-1)
                    pred_series[pred_key] = _denormalize(pred, stats, "s_mean", "s_std")

                row_payloads.append({
                    "title": row_cfg["title"],
                    "col_labels": row_cfg["col_labels"],
                    "pred_keys": row_cfg["pred_keys"],
                    "input_x": x_full,
                    "input_vals": u_full_denorm,
                    "x_query": x_query,
                    "s_true": s_true_denorm,
                    "pred_series": pred_series,
                })

            base = f"fig_{benchmark}_sample_{sample_idx}"
            _render_figure(row_payloads, bench_cfg, output_root / f"{base}.png")
            print(f"Saved {base}.png ({plot_idx}/{len(sample_indices)})")

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        "axes.titlesize": 12,
    })

    print(f"\nDone. Output: {PROJECT_ROOT / args.output_dir}")


if __name__ == "__main__":
    main()
