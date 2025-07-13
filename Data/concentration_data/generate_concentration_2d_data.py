#!/usr/bin/env python
"""generate_concentration_2d_data.py
----------------------------------
Synthetic steady-state concentration fields from chemical sources using exact 2D advection-diffusion Green's function.
"""
from __future__ import annotations

import argparse
import numpy as np
from datasets import Array3D, Dataset, Features, Sequence, Value
from tqdm import tqdm
from scipy.special import k0

def advection_diffusion_concentration(
    xs: np.ndarray,
    ys: np.ndarray,
    src_xy: np.ndarray,  # shape (N, 2)
    src_s: np.ndarray,   # shape (N,)
    wind_angle: float,   # wind direction in radians
) -> np.ndarray:
    """Exact 2D steady-state advection-diffusion Green's function.

    C(x,y) = s_i * (exp(v⃗ · r⃗ / (2D)) / (2πD)) * K_0(|v⃗| r / (2D))
    
    Args:
        xs, ys: coordinate arrays
        src_xy: source positions (N, 2)
        src_s: source rates (N,)
        wind_angle: wind direction in radians (fixed at 0 for +x direction)
    """
    D = 0.1  # diffusion coefficient (tunable for smoothness)
    v_mag = 1.0  # velocity magnitude
    
    # Wind velocity components
    vx = v_mag * np.cos(wind_angle)
    vy = v_mag * np.sin(wind_angle)
    
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    field = np.zeros_like(X, dtype=np.float32)

    for (x0, y0), s in zip(src_xy, src_s):
        # Distance vector r⃗ = (x-x0, y-y0)
        rx = X - x0
        ry = Y - y0
        r = np.sqrt(rx**2 + ry**2)
        
        # Handle r=0 case (source location) with small epsilon
        r = np.maximum(r, 1e-2)
        # v⃗ · r⃗ = vx*(x-x0) + vy*(y-y0)
        v_dot_r = vx * rx + vy * ry
        
        # Exponential term: exp(v⃗ · r⃗ / (2D))
        exp_term = np.exp(v_dot_r / (2 * D))
        
        # Modified Bessel function: K_0(|v⃗| r / (2D))
        bessel_arg = v_mag * r / (2 * D)
        bessel_term = k0(bessel_arg)
        
        # Full Green's function contribution
        contrib = s * (exp_term / (2 * np.pi * D)) * bessel_term
        field += contrib.astype(np.float32)

    return field

def generate_adaptive_grid(
    src_xy: np.ndarray,
    src_s: np.ndarray,
    wind_angle: float,
    n_points: int,
    spike_focus: float,
    rng: np.random.Generator,
    initial_grid_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate adaptive grid points focused on concentration peaks.
    
    Args:
        src_xy: Source locations (N, 2)
        src_s: Source rates (N,)
        wind_angle: Wind direction in radians
        n_points: Target number of grid points
        spike_focus: Focus strength on peaks (0=uniform, higher=more focused)
        rng: Random number generator
        initial_grid_size: Size of initial uniform grid (default 5x5)
        
    Returns:
        tuple of (grid_coords, field_values) where:
        - grid_coords: (n_points, 2) adaptive grid coordinates
        - field_values: (n_points,) concentration field at grid points
    """
    D = 0.1  # diffusion coefficient
    v_mag = 1.0  # velocity magnitude
    
    # Wind velocity components
    vx = v_mag * np.cos(wind_angle)
    vy = v_mag * np.sin(wind_angle)
    
    if spike_focus <= 0:
        # Uniform grid fallback
        grid_1d = int(np.sqrt(n_points))
        xs = np.linspace(0, 1, grid_1d, dtype=np.float32)
        ys = xs
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        grid_coords = np.column_stack([X.flatten(), Y.flatten()])
        # Trim to exact n_points
        grid_coords = grid_coords[:n_points]
        
        # Compute field values using advection-diffusion Green's function (VECTORIZED)
        grid_x_exp = grid_coords[:, 0][:, np.newaxis]  # (M, 1)
        grid_y_exp = grid_coords[:, 1][:, np.newaxis]  # (M, 1)
        src_x_exp = src_xy[:, 0][np.newaxis, :]        # (1, N)
        src_y_exp = src_xy[:, 1][np.newaxis, :]        # (1, N)
        
        # Distance components r⃗ = (x-x0, y-y0)
        rx = grid_x_exp - src_x_exp                    # (M, N)
        ry = grid_y_exp - src_y_exp                    # (M, N)
        r = np.sqrt(rx**2 + ry**2)                     # (M, N)
        r = np.maximum(r, 1e-6)                        # Avoid singularity
        
        # v⃗ · r⃗ = vx*rx + vy*ry
        v_dot_r = vx * rx + vy * ry                    # (M, N)
        
        # Exponential term: exp(v⃗ · r⃗ / (2D))
        exp_term = np.exp(v_dot_r / (2 * D))           # (M, N)
        
        # Modified Bessel function: K_0(|v⃗| r / (2D))
        bessel_arg = v_mag * r / (2 * D)               # (M, N)
        bessel_term = k0(bessel_arg)                   # (M, N)
        
        # Full Green's function contribution
        contrib = (src_s[np.newaxis, :] / (2 * np.pi * D)) * exp_term * bessel_term  # (M, N)
        field_values = np.sum(contrib, axis=1).astype(np.float32)  # (M,)
        
        return grid_coords, field_values
    
    # Step 0: Start with initial coarse grid
    xs_initial = np.linspace(0, 1, initial_grid_size, dtype=np.float32)
    ys_initial = xs_initial
    X_initial, Y_initial = np.meshgrid(xs_initial, ys_initial, indexing="xy")
    initial_grid_coords = np.column_stack([X_initial.flatten(), Y_initial.flatten()])  # (initial_grid_size^2, 2)
    
    # Compute field values for initial grid using advection-diffusion Green's function (VECTORIZED)
    grid_x_exp = initial_grid_coords[:, 0][:, np.newaxis]  # (M, 1)
    grid_y_exp = initial_grid_coords[:, 1][:, np.newaxis]  # (M, 1)
    src_x_exp = src_xy[:, 0][np.newaxis, :]                # (1, N)
    src_y_exp = src_xy[:, 1][np.newaxis, :]                # (1, N)
    
    # Distance components r⃗ = (x-x0, y-y0)
    rx = grid_x_exp - src_x_exp                            # (M, N)
    ry = grid_y_exp - src_y_exp                            # (M, N)
    r = np.sqrt(rx**2 + ry**2)                             # (M, N)
    r = np.maximum(r, 1e-6)                                # Avoid singularity
    
    # v⃗ · r⃗ = vx*rx + vy*ry
    v_dot_r = vx * rx + vy * ry                            # (M, N)
    
    # Exponential term: exp(v⃗ · r⃗ / (2D))
    exp_term = np.exp(v_dot_r / (2 * D))                   # (M, N)
    
    # Modified Bessel function: K_0(|v⃗| r / (2D))
    bessel_arg = v_mag * r / (2 * D)                       # (M, N)
    bessel_term = k0(bessel_arg)                           # (M, N)
    
    # Full Green's function contribution
    contrib = (src_s[np.newaxis, :] / (2 * np.pi * D)) * exp_term * bessel_term  # (M, N)
    initial_field_values = np.sum(contrib, axis=1).astype(np.float32)  # (initial_grid_size^2,)
    
    # If we only need initial_grid_size^2 or fewer points, return the initial grid
    initial_grid_points = initial_grid_size * initial_grid_size
    if n_points <= initial_grid_points:
        return initial_grid_coords[:n_points], initial_field_values[:n_points]
    
    # Step 1: Create a coarse reference grid to estimate concentration distribution
    coarse_res = 128  
    xs_coarse = np.linspace(0, 1, coarse_res, dtype=np.float32)
    ys_coarse = xs_coarse
    field_coarse = advection_diffusion_concentration(xs_coarse, ys_coarse, src_xy, src_s, wind_angle)
    
    # Step 2: Create sampling probabilities based on concentration magnitude
    # Focus on high values with exponential weighting
    conc_magnitude = field_coarse
    normalized_magnitude = (conc_magnitude - conc_magnitude.min()) / (conc_magnitude.max() - conc_magnitude.min() + 1e-8)
    spike_weights = np.exp(spike_focus * normalized_magnitude)
    
    # Normalize to probabilities
    sampling_probs = spike_weights / np.sum(spike_weights)
    
    # Step 3: Sample additional adaptive grid points according to concentration-based probabilities
    n_adaptive_points = n_points - initial_grid_points  # Remaining points after initial grid
    flat_indices = rng.choice(
        coarse_res * coarse_res, 
        size=n_adaptive_points, 
        p=sampling_probs.flatten(), 
        replace=True
    )
    
    # Convert flat indices to 2D coordinates
    row_indices = flat_indices // coarse_res
    col_indices = flat_indices % coarse_res
    
    # Get exact coordinates with small random perturbation for diversity
    grid_spacing = 1.0 / (coarse_res - 1)
    perturbation_scale = grid_spacing * 0.3  # Small perturbation within grid cell
    
    adaptive_grid_coords = np.zeros((n_adaptive_points, 2), dtype=np.float32)
    adaptive_grid_coords[:, 0] = xs_coarse[col_indices] + rng.normal(0, perturbation_scale, n_adaptive_points)
    adaptive_grid_coords[:, 1] = ys_coarse[row_indices] + rng.normal(0, perturbation_scale, n_adaptive_points)
    
    # Clamp to domain [0, 1]
    adaptive_grid_coords = np.clip(adaptive_grid_coords, 0, 1)
    
    # Step 4: Compute exact field values at adaptive grid points using advection-diffusion Green's function (VECTORIZED)
    grid_x_exp = adaptive_grid_coords[:, 0][:, np.newaxis]  # (M, 1)
    grid_y_exp = adaptive_grid_coords[:, 1][:, np.newaxis]  # (M, 1)
    src_x_exp = src_xy[:, 0][np.newaxis, :]                 # (1, N)
    src_y_exp = src_xy[:, 1][np.newaxis, :]                 # (1, N)
    
    # Distance components r⃗ = (x-x0, y-y0)
    rx = grid_x_exp - src_x_exp                             # (M, N)
    ry = grid_y_exp - src_y_exp                             # (M, N)
    r = np.sqrt(rx**2 + ry**2)                              # (M, N)
    r = np.maximum(r, 1e-6)                                 # Avoid singularity
    
    # v⃗ · r⃗ = vx*rx + vy*ry
    v_dot_r = vx * rx + vy * ry                             # (M, N)
    
    # Exponential term: exp(v⃗ · r⃗ / (2D))
    exp_term = np.exp(v_dot_r / (2 * D))                    # (M, N)
    
    # Modified Bessel function: K_0(|v⃗| r / (2D))
    bessel_arg = v_mag * r / (2 * D)                        # (M, N)
    bessel_term = k0(bessel_arg)                            # (M, N)
    
    # Full Green's function contribution
    contrib = (src_s[np.newaxis, :] / (2 * np.pi * D)) * exp_term * bessel_term  # (M, N)
    adaptive_field_values = np.sum(contrib, axis=1).astype(np.float32)  # (n_adaptive_points,)
    
    # Step 5: Combine initial grid with adaptive points
    final_grid_coords = np.concatenate([initial_grid_coords, adaptive_grid_coords], axis=0)
    final_field_values = np.concatenate([initial_field_values, adaptive_field_values], axis=0)
    
    return final_grid_coords, final_field_values

def make_record(
    *,
    rng: np.random.Generator,
    grid_n: int,
    n_min: int,
    n_max: int,
    constant_rate: bool,
    rate_low: float,
    rate_high: float,
    adaptive_mesh: bool = False,
    spike_focus: float = 2.0,
    n_adaptive_points: int = 4096,
    initial_grid_size: int = 5,
) -> dict[str, object]:
    """Create one (sources, field) record."""
    n_src = rng.integers(n_min, n_max + 1)

    # (x, y) ∈ [0, 1]^2
    src_xy = rng.random(size=(n_src, 2), dtype=np.float32)

    # Rates S_i
    if constant_rate:
        src_s = np.ones(n_src, dtype=np.float32)
    else:
        # log-uniform in [rate_low, rate_high]
        log_low, log_high = np.log10(rate_low), np.log10(rate_high)
        src_s = 10 ** rng.uniform(log_low, log_high, size=n_src).astype(np.float32)

    # Fixed wind direction in +x direction (0 radians)
    wind_angle = 0.0

    sources = np.column_stack([src_xy, src_s]).astype(np.float32)
    
    if adaptive_mesh:
        # Generate adaptive grid focused on concentration peaks
        grid_coords, field_values = generate_adaptive_grid(
            src_xy, src_s, wind_angle, n_adaptive_points, spike_focus, rng, initial_grid_size
        )
        
        return {
            "sources": sources.tolist(),  # variable-length list of 3-vectors
            "grid_coords": grid_coords.tolist(),  # (n_points, 2) adaptive coordinates
            "field_values": field_values.tolist(),  # (n_points,) concentration values
            "wind_angle": wind_angle,  # wind direction in radians
        }
    else:
        # Original uniform grid approach
        xs = np.linspace(0, 1, grid_n, dtype=np.float32)
        ys = xs
        field = advection_diffusion_concentration(xs, ys, src_xy, src_s, wind_angle)

        # No normalization - keep raw concentration field for operator learning
        field = field[..., None]  # add channel dimension

        return {
            "sources": sources.tolist(),  # variable-length list of 3-vectors
            "field": field,  # Raw concentration field (no normalization)
            "wind_angle": wind_angle,  # wind direction in radians
        }

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Stream-based builder to keep memory usage low."""

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
                "wind_angle": Value("float32"),
            }
        )
    else:
        features = Features(
            {
                "sources": Sequence(feature=Sequence(Value("float32"), length=3)),
                "field": Array3D(shape=(kwargs["grid_n"], kwargs["grid_n"], 1), dtype="float32"),
                "wind_angle": Value("float32"),
            }
        )
    return Dataset.from_generator(_gen, features=features)

def main():
    parser = argparse.ArgumentParser(description="Generate chemical concentration dataset (2D advection-diffusion Green's function).")
    parser.add_argument("--train", type=int, default=10_000, help="# training samples")
    parser.add_argument("--test", type=int, default=1_000, help="# test samples")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution (N×N)")

    # source distribution
    parser.add_argument("--n_min", type=int, default=30, help="Min # sources")
    parser.add_argument("--n_max", type=int, default=30, help="Max # sources")

    # rate distribution
    parser.add_argument("--constant_rate", action="store_true", help="Set all S_i = 1")
    parser.add_argument("--rate_low", type=float, default=0.1, help="Lower bound of S_i (ignored if constant)")
    parser.add_argument("--rate_high", type=float, default=1.0, help="Upper bound of S_i (ignored if constant)")

    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    
    # Adaptive mesh parameters
    parser.add_argument("--adaptive_mesh", action="store_true", help="Use adaptive mesh focused on concentration peaks")
    parser.add_argument("--spike_focus", type=float, default=10.0, help="Focus strength on peaks (0=uniform, higher=more focused)")
    parser.add_argument("--n_adaptive_points", type=int, default=8192, help="Number of adaptive grid points")
    parser.add_argument("--initial_grid_size", type=int, default=25, help="Size of initial uniform grid (NxN)")

    args = parser.parse_args()

    params = dict(
        grid_n=args.grid,
        n_min=args.n_min,
        n_max=args.n_max,
        constant_rate=args.constant_rate,
        rate_low=args.rate_low,
        rate_high=args.rate_high,
        seed=args.seed,
        adaptive_mesh=args.adaptive_mesh,
        spike_focus=args.spike_focus,
        n_adaptive_points=args.n_adaptive_points,
        initial_grid_size=args.initial_grid_size,
    )

    # Use different dataset paths for adaptive vs uniform mesh
    if args.adaptive_mesh:
        dataset_path = f"Data/concentration_data/chem_plume_adaptive_dataset{args.spike_focus}_n{args.n_adaptive_points}_N{args.initial_grid_size}_P{args.n_min}"
    else:
        dataset_path = "Data/concentration_data/chem_plume_dataset"

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
