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

def generate_adaptive_grid(
    src_xy: np.ndarray,
    src_q: np.ndarray,
    n_points: int,
    spike_focus: float,
    eps: float,
    rng: np.random.Generator,
    min_coverage: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate adaptive grid points focused on temperature spikes.
    
    Args:
        src_xy: Source locations (N, 2)
        src_q: Source powers (N,)
        n_points: Target number of grid points
        spike_focus: Focus strength on spikes (0=uniform, higher=more focused)
        eps: Softening parameter
        rng: Random number generator
        
    Returns:
        tuple of (grid_coords, field_values) where:
        - grid_coords: (n_points, 2) adaptive grid coordinates
        - field_values: (n_points,) temperature field at grid points
    """
    if spike_focus <= 0:
        # Uniform grid fallback
        grid_1d = int(np.sqrt(n_points))
        xs = np.linspace(0, 1, grid_1d, dtype=np.float32)
        ys = xs
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        grid_coords = np.column_stack([X.flatten(), Y.flatten()])
        # Trim to exact n_points
        grid_coords = grid_coords[:n_points]
        
        # Compute field values (VECTORIZED)
        grid_expanded = grid_coords[:, np.newaxis, :]  # (n_points, 1, 2)
        src_expanded = src_xy[np.newaxis, :, :]        # (1, n_sources, 2)
        distances = np.sqrt(np.sum((grid_expanded - src_expanded) ** 2, axis=2)) + eps
        log_distances = np.log(distances)
        powers_expanded = src_q[np.newaxis, :]
        contributions = (powers_expanded / (2 * np.pi)) * log_distances
        field_values = np.sum(contributions, axis=1).astype(np.float32)
        
        return grid_coords, field_values
    
    # Step 1: Create a coarse reference grid to estimate temperature distribution
    coarse_res = 128  # Higher resolution for better spike detection
    xs_coarse = np.linspace(0, 1, coarse_res, dtype=np.float32)
    ys_coarse = xs_coarse
    field_coarse = green_temperature(xs_coarse, ys_coarse, src_xy, src_q, eps)
    
    # Step 2: Create sampling probabilities based on temperature magnitude
    # Focus on most negative values (spikes) with exponential weighting
    temp_magnitude = np.abs(field_coarse)
    # Apply spike focus: higher values = more focus on extreme temperatures
    normalized_magnitude = (temp_magnitude - temp_magnitude.min()) / (temp_magnitude.max() - temp_magnitude.min() + 1e-8)
    spike_weights = np.exp(spike_focus * normalized_magnitude)
    
    # Ensure minimum coverage by adding a constant minimum weight
    min_weight = min_coverage * np.max(spike_weights)
    sampling_weights = spike_weights + min_weight
    
    # Normalize to probabilities
    sampling_probs = sampling_weights / np.sum(sampling_weights)
    
    # Step 3: Sample grid points according to temperature-based probabilities
    flat_indices = rng.choice(
        coarse_res * coarse_res, 
        size=n_points, 
        p=sampling_probs.flatten(), 
        replace=True
    )
    
    # Convert flat indices to 2D coordinates
    row_indices = flat_indices // coarse_res
    col_indices = flat_indices % coarse_res
    
    # Get exact coordinates with small random perturbation for diversity
    grid_spacing = 1.0 / (coarse_res - 1)
    perturbation_scale = grid_spacing * 0.3  # Small perturbation within grid cell
    
    grid_coords = np.zeros((n_points, 2), dtype=np.float32)
    grid_coords[:, 0] = xs_coarse[col_indices] + rng.normal(0, perturbation_scale, n_points)
    grid_coords[:, 1] = ys_coarse[row_indices] + rng.normal(0, perturbation_scale, n_points)
    
    # Clamp to domain [0, 1]
    grid_coords = np.clip(grid_coords, 0, 1)
    
    # Step 4: Compute exact field values at adaptive grid points (VECTORIZED)
    # Use broadcasting to compute all distances at once
    # grid_coords shape: (n_points, 2)
    # src_xy shape: (n_sources, 2)
    # We want: (n_points, n_sources) distance matrix
    
    # Expand dimensions for broadcasting: (n_points, 1, 2) - (1, n_sources, 2)
    grid_expanded = grid_coords[:, np.newaxis, :]  # (n_points, 1, 2)
    src_expanded = src_xy[np.newaxis, :, :]        # (1, n_sources, 2)
    
    # Compute all pairwise distances at once: (n_points, n_sources)
    distances = np.sqrt(np.sum((grid_expanded - src_expanded) ** 2, axis=2)) + eps
    
    # Compute log distances: (n_points, n_sources)
    log_distances = np.log(distances)
    
    # Apply source powers and sum: (n_points,)
    # src_q shape: (n_sources,) -> broadcast to (n_points, n_sources)
    powers_expanded = src_q[np.newaxis, :]  # (1, n_sources)
    contributions = (powers_expanded / (2 * np.pi)) * log_distances  # (n_points, n_sources)
    field_values = np.sum(contributions, axis=1).astype(np.float32)  # (n_points,)
    
    return grid_coords, field_values

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
    adaptive_mesh: bool = False,
    spike_focus: float = 2.0,
    n_adaptive_points: int = 4096,
    min_coverage: float = 0.1,
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

    sources = np.column_stack([src_xy, src_q]).astype(np.float32)
    
    if adaptive_mesh:
        # Generate adaptive grid focused on temperature spikes
        grid_coords, field_values = generate_adaptive_grid(
            src_xy, src_q, n_adaptive_points, spike_focus, eps, rng, min_coverage
        )
        
        return {
            "sources": sources.tolist(),  # variable‑length list of 3‑vectors
            "grid_coords": grid_coords.tolist(),  # (n_points, 2) adaptive coordinates
            "field_values": field_values.tolist(),  # (n_points,) temperature values
        }
    else:
        # Original uniform grid approach
        xs = np.linspace(0, 1, grid_n, dtype=np.float32)
        ys = xs
        field = green_temperature(xs, ys, src_xy, src_q, eps)

        # No normalization - keep raw temperature field for operator learning
        field = field[..., None]  # add channel dimension

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

    # Define features based on whether adaptive mesh is used
    if kwargs.get("adaptive_mesh", False):
        features = Features(
            {
                "sources": Sequence(feature=Sequence(Value("float32"), length=3)),
                "grid_coords": Sequence(feature=Sequence(Value("float32"), length=2)),
                "field_values": Sequence(feature=Value("float32")),
            }
        )
    else:
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

    parser.add_argument("--eps", type=float, default=1e-1, help="Softening radius ε in Green function")
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    
    # Adaptive mesh parameters
    parser.add_argument("--adaptive_mesh", action="store_true", help="Use adaptive mesh focused on temperature spikes")
    parser.add_argument("--spike_focus", type=float, default=9.0, help="Focus strength on spikes (0=uniform, higher=more focused)")
    parser.add_argument("--n_adaptive_points", type=int, default=16384, help="Number of adaptive grid points")
    parser.add_argument("--min_coverage", type=float, default=0.01, help="Minimum coverage ratio (0=pure spike focus, 1=uniform)")

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
        adaptive_mesh=args.adaptive_mesh,
        spike_focus=args.spike_focus,
        n_adaptive_points=args.n_adaptive_points,
        min_coverage=args.min_coverage,
    )

    # Use different dataset paths for adaptive vs uniform mesh
    if args.adaptive_mesh:
        dataset_path = f"Data/heat_data/pcb_heat_adaptive_dataset{args.spike_focus}_n{args.n_adaptive_points}"
    else:
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
