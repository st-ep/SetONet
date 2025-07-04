#!/usr/bin/env python
"""generate_pcb_heat.py
----------------------------------
Synthetic steady‑state temperature fields for a PCB with many heat sources.
"""
from __future__ import annotations

import argparse
import numpy as np
from datasets import Array3D, Dataset, Features, Sequence, Value
from tqdm import tqdm

def green_temperature(
    xs: np.ndarray,
    ys: np.ndarray,
    src_xy: np.ndarray,  # shape (N, 2)
    src_q: np.ndarray,   # shape (N,)
    eps: float = 1e-2,
) -> np.ndarray:
    """Analytic steady‑state solution with free boundaries.

    T(x, y) = Σ (Q_i / 2πk) · log‖(x, y) − r_i‖,  with k≡1, plus ε‑shift.
    """
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    field = np.zeros_like(X, dtype=np.float32)

    for (x0, y0), q in zip(src_xy, src_q):
        r = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2) + eps
        field += (q / (2 * np.pi)).astype(np.float32) * np.log(r)

    return field

def make_record(
    *,
    rng: np.random.Generator,
    grid_n: int,
    n_min: int,
    n_max: int,
    constant_power: bool,
    power_low: float,
    power_high: float,
    eps: float,
) -> dict[str, object]:
    """Create one (sources, field, T_min, T_max) record."""
    n_src = rng.integers(n_min, n_max + 1)

    # (x, y) ∈ [0, 1]^2
    src_xy = rng.random(size=(n_src, 2), dtype=np.float32)

    # Powers Q_i
    if constant_power:
        src_q = np.ones(n_src, dtype=np.float32)
    else:
        # log‑uniform in [power_low, power_high]
        log_low, log_high = np.log10(power_low), np.log10(power_high)
        src_q = 10 ** rng.uniform(log_low, log_high, size=n_src).astype(np.float32)

    # Build field on uniform grid
    xs = np.linspace(0, 1, grid_n, dtype=np.float32)
    ys = xs
    field = green_temperature(xs, ys, src_xy, src_q, eps)

    # No normalization - keep raw temperature field for operator learning
    field = field[..., None]  # add channel dimension

    sources = np.column_stack([src_xy, src_q]).astype(np.float32)
    return {
        "sources": sources.tolist(),  # variable‑length list of 3‑vectors
        "field": field,  # Raw temperature field (no normalization)
    }

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Stream‑based builder to keep memory usage low."""

    def _gen():
        rng = np.random.default_rng(kwargs.pop("seed", None))
        for _ in tqdm(range(num_samples), desc="samples"):
            yield make_record(rng=rng, **kwargs)

    features = Features(
        {
            "sources": Sequence(feature=Sequence(Value("float32"), length=3)),
            "field": Array3D(shape=(kwargs["grid_n"], kwargs["grid_n"], 1), dtype="float32"),
        }
    )
    return Dataset.from_generator(_gen, features=features)

def main():
    parser = argparse.ArgumentParser(description="Generate PCB‑heat dataset (steady‑state).")
    parser.add_argument("--train", type=int, default=10_000, help="# training samples")
    parser.add_argument("--test", type=int, default=1_000, help="# test samples")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution (N×N)")

    # source distribution
    parser.add_argument("--n_min", type=int, default=10, help="Min # sources")
    parser.add_argument("--n_max", type=int, default=10, help="Max # sources")

    # power distribution
    parser.add_argument("--constant_power", action="store_true", help="Set all Q_i = 1")
    parser.add_argument("--power_low", type=float, default=1e-2, help="Lower bound of Q_i (ignored if constant)")
    parser.add_argument("--power_high", type=float, default=1.0, help="Upper bound of Q_i (ignored if constant)")

    parser.add_argument("--eps", type=float, default=1e-2, help="Softening radius ε in Green function")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")

    args = parser.parse_args()

    params = dict(
        grid_n=args.grid,
        n_min=args.n_min,
        n_max=args.n_max,
        constant_power=args.constant_power,
        power_low=args.power_low,
        power_high=args.power_high,
        eps=args.eps,
        seed=args.seed,
    )

    # Use hardcoded dataset path like chladni_plate_generator.py
    dataset_path = "Data/heat_data/pcb_heat_dataset"

    # Calculate total samples needed
    total_samples = args.train + args.test

    print("[•] Generating full dataset …")
    # Generate single dataset with all samples
    full_ds = build_dataset(total_samples, **params)
    
    print("[•] Splitting into train/test sets …")
    # Split dataset using train_test_split like chladni_plate_generator.py
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    print("[•] Saving dataset …")
    ds.save_to_disk(dataset_path)

    print(f"✅ Done. Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset stored in {dataset_path}")


if __name__ == "__main__":
    main()
