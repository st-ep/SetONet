#!/usr/bin/env python
"""transport_diagnostics.py
----------------------------------
Diagnostics for the transport dataset:
  1) Consistency between transport vectors and velocity field (interpolated).
  2) Per-sample linear baseline for transport vectors.
  3) Transport vector magnitude statistics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys

import numpy as np
from datasets import DatasetDict, load_from_disk
from scipy.ndimage import map_coordinates


# Ensure project root is on sys.path for local imports.
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

@dataclass
class SummaryStats:
    mean: float
    median: float
    p90: float
    p95: float
    max: float


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred - target
    return float(np.linalg.norm(diff) / (np.linalg.norm(target) + 1e-8))


def _summary_stats(values: np.ndarray) -> SummaryStats:
    return SummaryStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        max=float(np.max(values)),
    )


def _subsample_pairs(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_points: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or source_points.shape[0] <= max_points:
        return source_points, target_points
    idx = rng.choice(source_points.shape[0], size=max_points, replace=False)
    return source_points[idx], target_points[idx]


def _interpolate_velocity_field(
    velocity_field: np.ndarray,
    points: np.ndarray,
    domain_size: float,
) -> np.ndarray:
    grid_h, grid_w, _ = velocity_field.shape
    points_clipped = np.clip(points, -domain_size, domain_size)
    grid_indices = (points_clipped + domain_size) * (np.array([grid_h - 1, grid_w - 1]) / (2 * domain_size))
    coords = grid_indices.T  # (2, n_points)

    v_x = map_coordinates(velocity_field[:, :, 0], coords, order=1, mode="nearest")
    v_y = map_coordinates(velocity_field[:, :, 1], coords, order=1, mode="nearest")
    return np.stack([v_x, v_y], axis=-1)


def _grid_coords_from_sample(sample: dict) -> np.ndarray:
    if "grid_coords" in sample:
        return np.array(sample["grid_coords"], dtype=np.float32)
    velocity_field = np.array(sample["velocity_field"], dtype=np.float32)
    grid_h, grid_w = velocity_field.shape[0], velocity_field.shape[1]
    domain_size = float(sample.get("domain_size", 5.0))
    xs = np.linspace(-domain_size, domain_size, grid_h, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_w, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def _load_samples(dataset, split: str, n_samples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    total = len(dataset[split])
    n_samples = min(n_samples, total)
    indices = rng.choice(total, size=n_samples, replace=False)
    return [dataset[split][int(i)] for i in indices]


def check_consistency(samples: list[dict], max_points: int | None, seed: int) -> SummaryStats:
    rng = np.random.default_rng(seed)
    rel_errors = []

    for sample in samples:
        source_points = np.array(sample["source_points"], dtype=np.float32)
        target_points = np.array(sample["target_points"], dtype=np.float32)
        velocity_field = np.array(sample["velocity_field"], dtype=np.float32)
        domain_size = float(sample.get("domain_size", 5.0))

        source_points, target_points = _subsample_pairs(source_points, target_points, max_points, rng)

        transport_vectors = target_points - source_points
        v_interp = _interpolate_velocity_field(velocity_field, source_points, domain_size)

        rel_errors.append(_relative_l2(v_interp.reshape(-1), transport_vectors.reshape(-1)))

    return _summary_stats(np.array(rel_errors, dtype=np.float64))


def check_linear_baseline(samples: list[dict], max_points: int | None, seed: int) -> SummaryStats:
    rng = np.random.default_rng(seed)
    rel_errors = []

    for sample in samples:
        source_points = np.array(sample["source_points"], dtype=np.float32)
        target_points = np.array(sample["target_points"], dtype=np.float32)

        source_points, target_points = _subsample_pairs(source_points, target_points, max_points, rng)

        transport_vectors = target_points - source_points
        X = np.concatenate([source_points, np.ones((source_points.shape[0], 1), dtype=np.float32)], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(X, transport_vectors, rcond=None)
        pred = X @ coeffs

        rel_errors.append(_relative_l2(pred.reshape(-1), transport_vectors.reshape(-1)))

    return _summary_stats(np.array(rel_errors, dtype=np.float64))


def check_magnitude_stats(samples: list[dict], max_points: int | None, seed: int) -> SummaryStats:
    rng = np.random.default_rng(seed)
    norms = []

    for sample in samples:
        source_points = np.array(sample["source_points"], dtype=np.float32)
        target_points = np.array(sample["target_points"], dtype=np.float32)

        source_points, target_points = _subsample_pairs(source_points, target_points, max_points, rng)

        transport_vectors = target_points - source_points
        norms.extend(np.linalg.norm(transport_vectors, axis=1))

    return _summary_stats(np.array(norms, dtype=np.float64))


def run_overfit_check(
    dataset,
    split: str,
    n_samples: int,
    seed: int,
    mode: str,
    model_type: str,
    steps: int,
    batch_size: int,
    device: str,
    lr: float,
) -> None:
    import torch

    from Models.SetONet import SetONet
    from Models.VIDON import VIDON
    from Models.utils.helper_utils import calculate_l2_relative_error
    from Data.transport_data.transport_dataset import TransportDataset

    rng = np.random.default_rng(seed)
    total = len(dataset[split])
    n_samples = min(n_samples, total)
    indices = rng.choice(total, size=n_samples, replace=False)
    subset = dataset[split].select([int(i) for i in indices])
    subset_dict = DatasetDict({"train": subset, "test": subset})

    torch_device = torch.device(device)
    wrapper = TransportDataset(subset_dict, batch_size=batch_size, device=torch_device, mode=mode)

    if model_type == "setonet":
        model = SetONet(
            input_size_src=2,
            output_size_src=1,
            input_size_tgt=2,
            output_size_tgt=2,
            p=32,
            phi_hidden_size=64,
            rho_hidden_size=64,
            trunk_hidden_size=64,
            n_trunk_layers=3,
            activation_fn=torch.nn.ReLU,
            use_deeponet_bias=True,
            phi_output_size=32,
            initial_lr=lr,
            lr_schedule_steps=None,
            lr_schedule_gammas=None,
            pos_encoding_type="sinusoidal",
            pos_encoding_dim=32,
            pos_encoding_max_freq=0.01,
            aggregation_type="attention",
            use_positional_encoding=True,
            attention_n_tokens=1,
            branch_head_type="standard",
        ).to(torch_device)
    elif model_type == "vidon":
        model = VIDON(
            input_size_src=2,
            output_size_src=1,
            input_size_tgt=2,
            output_size_tgt=2,
            p=32,
            n_heads=2,
            d_enc=32,
            head_output_size=32,
            enc_hidden_size=32,
            enc_n_layers=2,
            head_hidden_size=64,
            head_n_layers=2,
            combine_hidden_size=64,
            combine_n_layers=2,
            trunk_hidden_size=64,
            n_trunk_layers=2,
            activation_fn=torch.nn.ReLU,
            initial_lr=lr,
            lr_schedule_steps=None,
            lr_schedule_gammas=None,
        ).to(torch_device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"[•] Overfit check: {model_type} on {n_samples} sample(s), mode={mode}, steps={steps}")
    model.train_model(dataset=wrapper, epochs=steps, progress_bar=True)

    model.eval()
    total_rel = 0.0
    total_mse = 0.0
    with torch.no_grad():
        for i in range(n_samples):
            sample = subset[i]
            source_points = torch.tensor(np.array(sample["source_points"]), device=torch_device, dtype=torch.float32)
            target_points = torch.tensor(np.array(sample["target_points"]), device=torch_device, dtype=torch.float32)
            velocity_field = torch.tensor(np.array(sample["velocity_field"]), device=torch_device, dtype=torch.float32)

            source_coords = source_points.unsqueeze(0)
            source_weights = torch.ones(1, source_points.shape[0], 1, device=torch_device, dtype=torch.float32)

            if mode == "transport_map":
                pred = model(source_coords, source_weights, source_coords)
                target = (target_points - source_points).unsqueeze(0)
            else:
                grid_coords = torch.tensor(_grid_coords_from_sample(sample), device=torch_device, dtype=torch.float32)
                pred = model(source_coords, source_weights, grid_coords.unsqueeze(0))
                target = velocity_field.reshape(1, -1, 2)

            total_mse += torch.nn.functional.mse_loss(pred, target).item()
            total_rel += calculate_l2_relative_error(pred.reshape(1, -1), target.reshape(1, -1)).item()

    print(f"[•] Overfit results: MSE={total_mse / n_samples:.6e}, Rel L2={total_rel / n_samples:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostics for transport dataset quality and difficulty.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/transport_data/transport_dataset",
        help="Path to transport dataset saved with datasets.save_to_disk",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to sample")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of dataset samples to evaluate")
    parser.add_argument("--max_points", type=int, default=None, help="Subsample points per sample for speed")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--overfit_model", type=str, default="none", choices=["none", "setonet", "vidon"])
    parser.add_argument("--overfit_samples", type=int, default=1, help="Number of samples to overfit")
    parser.add_argument("--overfit_steps", type=int, default=2000, help="Training steps for overfit check")
    parser.add_argument("--overfit_batch_size", type=int, default=1, help="Batch size for overfit check")
    parser.add_argument("--overfit_device", type=str, default="cpu", help="Device for overfit check")
    parser.add_argument("--overfit_mode", type=str, default="transport_map", choices=["transport_map", "velocity_field"])
    parser.add_argument("--overfit_lr", type=float, default=5e-4, help="Learning rate for overfit check")
    args = parser.parse_args()

    dataset = load_from_disk(args.data_path)
    samples = _load_samples(dataset, args.split, args.n_samples, args.seed)

    print(f"[•] Dataset: {args.data_path} ({args.split}, n={len(samples)})")
    if args.max_points is not None:
        print(f"[•] Point subsample per sample: {args.max_points}")

    consistency = check_consistency(samples, args.max_points, args.seed)
    linear_baseline = check_linear_baseline(samples, args.max_points, args.seed)
    magnitude_stats = check_magnitude_stats(samples, args.max_points, args.seed)

    print("\n[1] Consistency: transport vectors vs interpolated velocity field")
    print(
        "    rel L2 (mean/median/p90/p95/max): "
        f"{consistency.mean:.4f} / {consistency.median:.4f} / {consistency.p90:.4f} / "
        f"{consistency.p95:.4f} / {consistency.max:.4f}"
    )

    print("\n[2] Linear baseline: per-sample affine fit of transport vectors")
    print(
        "    rel L2 (mean/median/p90/p95/max): "
        f"{linear_baseline.mean:.4f} / {linear_baseline.median:.4f} / {linear_baseline.p90:.4f} / "
        f"{linear_baseline.p95:.4f} / {linear_baseline.max:.4f}"
    )

    print("\n[3] Transport vector magnitude stats (per point)")
    print(
        "    ||delta|| (mean/median/p90/p95/max): "
        f"{magnitude_stats.mean:.4f} / {magnitude_stats.median:.4f} / {magnitude_stats.p90:.4f} / "
        f"{magnitude_stats.p95:.4f} / {magnitude_stats.max:.4f}"
    )

    if args.overfit_model != "none":
        run_overfit_check(
            dataset=dataset,
            split=args.split,
            n_samples=args.overfit_samples,
            seed=args.seed,
            mode=args.overfit_mode,
            model_type=args.overfit_model,
            steps=args.overfit_steps,
            batch_size=args.overfit_batch_size,
            device=args.overfit_device,
            lr=args.overfit_lr,
        )


if __name__ == "__main__":
    main()
