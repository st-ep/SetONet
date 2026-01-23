#!/usr/bin/env python
"""generate_transport_data.py
----------------------------------
Generate optimal transport dataset using OTT/JAX with HuggingFace datasets format.
GPU-optimized version with JIT compilation and early stopping.
"""
from __future__ import annotations

import argparse
import gc
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from datasets import Array3D, Dataset, Features, Sequence, Value
from jax.scipy.ndimage import map_coordinates as jax_map_coordinates
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from scipy.ndimage import map_coordinates as scipy_map_coordinates
from scipy.stats import multivariate_normal
from tqdm import tqdm


# ============================================================================
# OTT regularization wiring
# ============================================================================

_REG_PARAM_TARGET = "pointcloud"
_REG_PARAM_KW = "epsilon"


# ============================================================================
# JIT-compiled core functions (defined at module level for reuse)
# ============================================================================

@partial(jax.jit, static_argnames=("max_iterations", "threshold", "reg_param", "reg_param_target", "reg_param_kw"))
def solve_ot(
    x: jnp.ndarray,
    y: jnp.ndarray, 
    a: jnp.ndarray,
    b: jnp.ndarray,
    max_iterations: int,
    threshold: float,
    reg_param: float,
    reg_param_target: str | None,
    reg_param_kw: str | None,
) -> jnp.ndarray:
    """JIT-compiled Sinkhorn solver. Returns transport matrix."""
    geom_kwargs = {}
    problem_kwargs = {}
    solver_kwargs = {"max_iterations": max_iterations, "threshold": threshold}

    if reg_param_target and reg_param_kw:
        if reg_param_target == "pointcloud":
            geom_kwargs[reg_param_kw] = reg_param
        elif reg_param_target == "linear_problem":
            problem_kwargs[reg_param_kw] = reg_param
        elif reg_param_target == "sinkhorn":
            solver_kwargs[reg_param_kw] = reg_param

    geom = pointcloud.PointCloud(x, y, **geom_kwargs)
    problem = linear_problem.LinearProblem(geom, a, b, **problem_kwargs)
    solver = sinkhorn.Sinkhorn(**solver_kwargs)
    out = solver(problem)
    return out.matrix


@partial(jax.jit, static_argnames=("grid_n",))
def compute_transport_map_and_velocity(
    P: jnp.ndarray,
    grid_pts: jnp.ndarray,
    rho_src: jnp.ndarray,
    grid_n: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute barycentric transport map and velocity field on GPU."""
    # Barycentric projection: T(x) = sum_y P(x,y) * y / rho(x)
    T_weighted = jnp.einsum('ij,jk->ik', P, grid_pts)  # (n_grid, 2)
    
    # Add small regularization to prevent division by very small densities
    eps = 1e-8
    rho_src_reg = rho_src + eps
    T_grid = T_weighted / rho_src_reg[:, None]
    T_grid = T_grid.reshape((grid_n, grid_n, 2))
    
    # Velocity field = T(x) - x
    grid_pts_reshaped = grid_pts.reshape(grid_n, grid_n, 2)
    V_bary = T_grid - grid_pts_reshaped
    
    return T_grid, V_bary


@partial(jax.jit, static_argnames=("domain_size", "grid_n"))
def interpolate_transport_jax(
    T_grid: jnp.ndarray,
    sample_points: jnp.ndarray,
    domain_size: float,
    grid_n: int,
) -> jnp.ndarray:
    """Interpolate transport map at sample points using JAX (GPU)."""
    # Clip sample points to grid bounds
    X_clipped = jnp.clip(sample_points, -domain_size, domain_size)
    
    # Convert coordinates to grid indices (fractional)
    grid_indices = (X_clipped + domain_size) * (grid_n - 1) / (2 * domain_size)
    
    # map_coordinates expects (ndim, npoints) format
    coords = grid_indices.T  # Shape: (2, n_samples)
    
    # Interpolate both components using linear interpolation (order=1)
    Ti_x = jax_map_coordinates(T_grid[:, :, 0], coords, order=1, mode='nearest')
    Ti_y = jax_map_coordinates(T_grid[:, :, 1], coords, order=1, mode='nearest')
    Ti = jnp.stack([Ti_x, Ti_y], axis=-1)
    
    return Ti


# ============================================================================
# Record generation
# ============================================================================

def make_record(
    *,
    rng: np.random.Generator,
    grid_pts_np: np.ndarray,
    grid_pts_jax: jnp.ndarray,
    grid_n: int,
    n_samples: int,
    domain_size: float,
    reg_param: float,
    max_iterations: int,
    threshold: float,
    reg_param_target: str | None,
    reg_param_kw: str | None,
) -> dict[str, object]:
    """Create one optimal transport record using JAX (GPU-accelerated)."""
    
    # Generate source distribution parameters (CPU - fast)
    mean1 = rng.uniform(-2, 2, size=2).astype(np.float32)
    mean2 = rng.uniform(-2, 2, size=2).astype(np.float32)
    diag1 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
    diag2 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
    cov1 = np.diag(diag1)
    cov2 = np.diag(diag2)

    # Source density (mixture of Gaussians) - CPU is fine, this is fast
    rho_src = (0.5 * multivariate_normal(mean1, cov1).pdf(grid_pts_np) +
               0.5 * multivariate_normal(mean2, cov2).pdf(grid_pts_np))
    rho_src = rho_src.astype(np.float32)
    rho_src /= rho_src.sum()
    
    # Target density (Gaussian at origin)
    rho_tar = multivariate_normal([0, 0], 0.5 * np.eye(2)).pdf(grid_pts_np)
    rho_tar = rho_tar.astype(np.float32)
    rho_tar /= rho_tar.sum()

    # Convert to JAX arrays
    a = jnp.array(rho_src, dtype=jnp.float32)
    b = jnp.array(rho_tar, dtype=jnp.float32)

    # Solve optimal transport (JIT-compiled, stays on GPU)
    P = solve_ot(
        grid_pts_jax,
        grid_pts_jax,
        a,
        b,
        max_iterations,
        threshold,
        reg_param,
        reg_param_target,
        reg_param_kw,
    )
    
    # Compute transport map and velocity field (on GPU)
    T_grid, V_bary = compute_transport_map_and_velocity(P, grid_pts_jax, a, grid_n)
    
    # Sample points from source distribution (CPU)
    X = np.vstack([
        rng.multivariate_normal(mean1, cov1, n_samples // 2),
        rng.multivariate_normal(mean2, cov2, n_samples // 2)
    ]).astype(np.float32)
    X_jax = jnp.array(X)
    
    # Interpolate transport map at sample points (on GPU)
    Ti = interpolate_transport_jax(T_grid, X_jax, domain_size, grid_n)
    
    # Compute target points
    Xf = Ti  # Ti is the transported position
    
    # Convert back to numpy only at the end
    V_bary_np = np.array(V_bary)
    Xf_np = np.array(Xf)

    return {
        "source_points": X.tolist(),
        "target_points": Xf_np.tolist(),
        "velocity_field": V_bary_np,
        "source_params": {
            "mean1": mean1.tolist(),
            "mean2": mean2.tolist(),
            "cov1": cov1.tolist(),
            "cov2": cov2.tolist(),
        },
        "domain_size": domain_size,
    }


# ============================================================================
# Dataset building (dict-of-lists)
# ============================================================================

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Build dataset directly by accumulating records in memory."""
    grid_n = kwargs["grid_n"]
    domain_size = kwargs["domain_size"]
    reg_param = kwargs["reg_param"]
    use_reg_param = kwargs.get("use_reg_param", False)
    max_iterations = kwargs["max_iterations"]
    threshold = kwargs.get("threshold", 1e-3)
    n_samples_per_record = kwargs["n_samples"]
    seed = kwargs.get("seed", None)

    xs = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    grid_pts_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    grid_pts_jax = jnp.array(grid_pts_np)

    rng = np.random.default_rng(seed)

    all_records = {
        "source_points": [],
        "target_points": [],
        "velocity_field": [],
        "source_params": [],
        "domain_size": [],
    }

    for i in tqdm(range(num_samples), desc="samples"):
        if i % 100 == 0 and i > 0:
            gc.collect()

        record = make_record(
            rng=rng,
            grid_pts_np=grid_pts_np,
            grid_pts_jax=grid_pts_jax,
            grid_n=grid_n,
            n_samples=n_samples_per_record,
            domain_size=domain_size,
            reg_param=reg_param,
            max_iterations=max_iterations,
            threshold=threshold,
            reg_param_target=_REG_PARAM_TARGET if use_reg_param else None,
            reg_param_kw=_REG_PARAM_KW if use_reg_param else None,
        )

        all_records["source_points"].append(record["source_points"])
        all_records["target_points"].append(record["target_points"])
        all_records["velocity_field"].append(record["velocity_field"])
        all_records["source_params"].append(record["source_params"])
        all_records["domain_size"].append(record["domain_size"])

    # Define features for HuggingFace dataset
    features = Features(
        {
            "source_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "target_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "velocity_field": Array3D(shape=(grid_n, grid_n, 2), dtype="float32"),
            "source_params": {
                "mean1": Sequence(feature=Value("float32"), length=2),
                "mean2": Sequence(feature=Value("float32"), length=2),
                "cov1": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
                "cov2": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
            },
            "domain_size": Value("float32"),
        }
    )
    
    return Dataset.from_dict(all_records, features=features)


# ============================================================================
# Quality regression check
# ============================================================================

def run_quality_check(num_check: int = 10, seed: int = 42):
    """Compare JAX interpolation vs SciPy interpolation for quality assurance."""
    print(f"[•] Running quality check on {num_check} samples...")
    
    rng = np.random.default_rng(seed)
    grid_n = 80
    domain_size = 5.0
    n_samples = 256
    reg_param = 0.05
    max_iterations = 5000
    threshold = 1e-3
    
    # Precompute grid
    xs = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    grid_pts_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    grid_pts_jax = jnp.array(grid_pts_np)
    
    max_rel_error = 0.0
    
    for i in range(num_check):
        # Generate random source distribution
        mean1 = rng.uniform(-2, 2, size=2).astype(np.float32)
        mean2 = rng.uniform(-2, 2, size=2).astype(np.float32)
        diag1 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
        diag2 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
        cov1 = np.diag(diag1)
        cov2 = np.diag(diag2)
        
        rho_src = (0.5 * multivariate_normal(mean1, cov1).pdf(grid_pts_np) +
                   0.5 * multivariate_normal(mean2, cov2).pdf(grid_pts_np))
        rho_src = rho_src.astype(np.float32)
        rho_src /= rho_src.sum()
        
        rho_tar = multivariate_normal([0, 0], 0.5 * np.eye(2)).pdf(grid_pts_np)
        rho_tar = rho_tar.astype(np.float32)
        rho_tar /= rho_tar.sum()
        
        a = jnp.array(rho_src)
        b = jnp.array(rho_tar)
        
        # Solve OT
        P = solve_ot(
            grid_pts_jax,
            grid_pts_jax,
            a,
            b,
            max_iterations,
            threshold,
            reg_param,
            _REG_PARAM_TARGET,
            _REG_PARAM_KW,
        )
        T_grid, V_bary = compute_transport_map_and_velocity(P, grid_pts_jax, a, grid_n)
        T_grid_np = np.array(T_grid)
        
        # Sample points
        X = np.vstack([
            rng.multivariate_normal(mean1, cov1, n_samples // 2),
            rng.multivariate_normal(mean2, cov2, n_samples // 2)
        ]).astype(np.float32)
        
        # JAX interpolation
        Ti_jax = np.array(interpolate_transport_jax(T_grid, jnp.array(X), domain_size, grid_n))
        
        # SciPy interpolation (reference)
        X_clipped = np.clip(X, -domain_size, domain_size)
        grid_indices = (X_clipped + domain_size) * (grid_n - 1) / (2 * domain_size)
        coords = grid_indices.T
        Ti_x_scipy = scipy_map_coordinates(T_grid_np[:, :, 0], coords, order=1, mode='nearest')
        Ti_y_scipy = scipy_map_coordinates(T_grid_np[:, :, 1], coords, order=1, mode='nearest')
        Ti_scipy = np.column_stack([Ti_x_scipy, Ti_y_scipy])
        
        # Compute relative L2 error
        diff = Ti_jax - Ti_scipy
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(Ti_scipy) + 1e-8)
        max_rel_error = max(max_rel_error, rel_error)
    
    print(f"    Max relative L2 error (JAX vs SciPy): {max_rel_error:.2e}")
    
    if max_rel_error < 1e-5:
        print("    ✅ Quality check PASSED")
        return True
    else:
        print("    ⚠️  Quality check: difference detected but within acceptable range")
        return max_rel_error < 1e-3


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate optimal transport dataset using OTT/JAX (GPU-optimized).")
    parser.add_argument("--train", type=int, default=20000, help="# training samples")
    parser.add_argument("--test", type=int, default=1000, help="# test samples")
    parser.add_argument("--grid", type=int, default=80, help="Grid resolution (N×N)")

    # Transport parameters
    parser.add_argument("--n_samples", type=int, default=256, help="# sample points per trial")
    parser.add_argument("--domain_size", type=float, default=5.0, help="Domain size [-domain_size, domain_size]")
    parser.add_argument("--reg_param", type=float, default=0.05, help="Sinkhorn regularization parameter")
    parser.add_argument("--use_reg_param", action="store_true", default=True, help="Use reg_param instead of OTT defaults")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum Sinkhorn iterations")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Sinkhorn convergence threshold (early stopping)")
    
    # Quality check option
    parser.add_argument("--safe_check", action="store_true", help="Run quality regression check before generating")

    parser.add_argument("--seed", type=int, default=2, help="Global RNG seed")

    args = parser.parse_args()

    # Report devices
    devices = jax.devices()
    print(f"[•] JAX devices: {devices}")

    if args.use_reg_param:
        print(f"[•] Using reg_param via {_REG_PARAM_TARGET}.{_REG_PARAM_KW} = {args.reg_param}")
    else:
        print("[•] Using OTT default regularization (reg_param ignored)")
    
    # Run quality check if requested
    if args.safe_check:
        if not run_quality_check():
            print("Quality check failed. Aborting.")
            return

    params = dict(
        grid_n=args.grid,
        n_samples=args.n_samples,
        domain_size=args.domain_size,
        reg_param=args.reg_param,
        use_reg_param=args.use_reg_param,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        seed=args.seed,
    )

    dataset_path = "Data/transport_data/transport_dataset"

    # Calculate total samples needed
    total_samples = args.train + args.test

    print(f"[•] Generating {total_samples} samples...")
    full_ds = build_dataset(total_samples, **params)
    
    print("[•] Splitting into train/test sets...")
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    print("[•] Saving dataset...")
    ds.save_to_disk(dataset_path)

    print(f"✅ Done. Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset stored in {dataset_path}")


if __name__ == "__main__":
    main()
