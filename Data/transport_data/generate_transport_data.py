#!/usr/bin/env python
"""generate_transport_data.py
----------------------------------
Generate optimal transport dataset using OTT/JAX with HuggingFace datasets format.
"""
from __future__ import annotations

import argparse
import numpy as np
import jax.numpy as jnp
from datasets import Array3D, Dataset, Features, Sequence, Value
from tqdm import tqdm
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.interpolate import griddata

def make_record(
    *,
    rng: np.random.Generator,
    grid_n: int,
    n_samples: int,
    domain_size: float,
    reg_param: float,
    max_iterations: int,
) -> dict[str, object]:
    """Create one optimal transport record."""
    
    # Setup grid
    xs = np.linspace(-domain_size, domain_size, grid_n)
    ys = np.linspace(-domain_size, domain_size, grid_n)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    # Generate source distribution parameters
    mean1 = rng.uniform(-2, 2, size=2)
    mean2 = rng.uniform(-2, 2, size=2)
    diag1 = rng.uniform(0.1, 1.0, size=2)
    diag2 = rng.uniform(0.1, 1.0, size=2)
    cov1 = np.diag(diag1)
    cov2 = np.diag(diag2)

    # Source density (mixture of Gaussians)
    rho_src = (0.5 * multivariate_normal(mean1, cov1).pdf(grid_pts) +
               0.5 * multivariate_normal(mean2, cov2).pdf(grid_pts))
    rho_src /= rho_src.sum()
    
    # Target density (Gaussian at origin)
    rho_tar = multivariate_normal([0, 0], 0.5 * np.eye(2)).pdf(grid_pts)
    rho_tar /= rho_tar.sum()

    # Convert to JAX arrays for OTT
    x = jnp.array(grid_pts)
    y = jnp.array(grid_pts)
    a = jnp.array(rho_src)
    b = jnp.array(rho_tar)

    # Solve optimal transport using OTT
    geom = pointcloud.PointCloud(x, y)
    problem = linear_problem.LinearProblem(geom, a, b)
    solver = sinkhorn.Sinkhorn(max_iterations=max_iterations)
    out = solver(problem)

    # Extract transport map
    P = np.array(out.matrix)
    T_weighted = (P[..., None] * grid_pts[None, ...]).sum(axis=1)
    T_grid = (T_weighted / rho_src[:, None]).reshape((grid_n, grid_n, 2))
    V_bary = T_grid - grid_pts.reshape(grid_n, grid_n, 2)  # velocity field

    # Sample points from source distribution
    X = np.vstack([
        rng.multivariate_normal(mean1, cov1, n_samples // 2),
        rng.multivariate_normal(mean2, cov2, n_samples // 2)
    ])
    
    # Interpolate transport map to sample points
    Ti = griddata(grid_pts, T_grid.reshape(-1, 2), X, method='cubic')
    # Handle NaN values from cubic interpolation
    nan_mask = np.isnan(Ti).any(axis=1)
    if nan_mask.any():
        Ti_linear = griddata(grid_pts, T_grid.reshape(-1, 2), X[nan_mask], method='linear')
        Ti[nan_mask] = Ti_linear
    
    V = Ti - X
    Xf = X + V

    return {
        "source_points": X.astype(np.float32).tolist(),  # (n_samples, 2)
        "target_points": Xf.astype(np.float32).tolist(),  # (n_samples, 2) 
        "velocity_field": V_bary.astype(np.float32),  # (grid_n, grid_n, 2)
        "source_params": {
            "mean1": mean1.astype(np.float32).tolist(),
            "mean2": mean2.astype(np.float32).tolist(),
            "cov1": cov1.astype(np.float32).tolist(),
            "cov2": cov2.astype(np.float32).tolist(),
        },
        "grid_coords": grid_pts.astype(np.float32).tolist(),  # (grid_n*grid_n, 2)
        "domain_size": domain_size,
    }

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Stream-based builder to keep memory usage low."""

    def _gen():
        rng = np.random.default_rng(kwargs.pop("seed", None))
        for _ in tqdm(range(num_samples), desc="samples"):
            yield make_record(rng=rng, **kwargs)

    # Define features for HuggingFace dataset
    features = Features(
        {
            "source_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "target_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "velocity_field": Array3D(shape=(kwargs["grid_n"], kwargs["grid_n"], 2), dtype="float32"),
            "source_params": {
                "mean1": Sequence(feature=Value("float32"), length=2),
                "mean2": Sequence(feature=Value("float32"), length=2),
                "cov1": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
                "cov2": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
            },
            "grid_coords": Sequence(feature=Sequence(Value("float32"), length=2)),
            "domain_size": Value("float32"),
        }
    )
    return Dataset.from_generator(_gen, features=features)

def main():
    parser = argparse.ArgumentParser(description="Generate optimal transport dataset using OTT/JAX.")
    parser.add_argument("--train", type=int, default=400, help="# training samples")
    parser.add_argument("--test", type=int, default=100, help="# test samples")
    parser.add_argument("--grid", type=int, default=80, help="Grid resolution (N×N)")

    # Transport parameters
    parser.add_argument("--n_samples", type=int, default=256, help="# sample points per trial")
    parser.add_argument("--domain_size", type=float, default=5.0, help="Domain size [-domain_size, domain_size]")
    parser.add_argument("--reg_param", type=float, default=0.05, help="Sinkhorn regularization parameter")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum Sinkhorn iterations")

    parser.add_argument("--seed", type=int, default=2, help="Global RNG seed")

    args = parser.parse_args()

    params = dict(
        grid_n=args.grid,
        n_samples=args.n_samples,
        domain_size=args.domain_size,
        reg_param=args.reg_param,
        max_iterations=args.max_iterations,
        seed=args.seed,
    )

    dataset_path = "Data/transport_data/transport_dataset"

    # Calculate total samples needed
    total_samples = args.train + args.test

    print("[•] Generating full dataset …")
    # Generate single dataset with all samples
    full_ds = build_dataset(total_samples, **params)
    
    print("[•] Splitting into train/test sets …")
    # Split dataset using train_test_split
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    print("[•] Saving dataset …")
    ds.save_to_disk(dataset_path)

    print(f"✅ Done. Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset stored in {dataset_path}")


if __name__ == "__main__":
    main()