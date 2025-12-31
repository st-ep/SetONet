#!/usr/bin/env python
"""generate_transport_q_data.py
----------------------------------
Generate optimal transport dataset with *decoupled query points* (Strategy 1).

This addresses the "ys = xs" shortcut in the original transport benchmark by
introducing independent query points for trunk evaluation.

- Source density rho0: mixture of two Gaussians with random means/covariances
- Target density rho1: fixed centered Gaussian N(0, 0.5 I)
- Solve entropic OT (Sinkhorn) on a fixed grid
- Compute barycentric transport map on the grid
- Supervision:
  - source_points: samples from rho0 (branch sensors)
  - target_points: transported positions at source_points (legacy / visualization)
  - query_points: independent query locations (mix of rho0-samples + uniform)
  - query_vectors: displacement T(query_points) - query_points (primary target)
  - velocity_field: grid displacement (legacy)
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
# JIT-compiled core functions
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
    """Compute barycentric transport map and velocity field."""
    # Barycentric projection: T(x) = sum_y P(x,y) * y / rho(x)
    T_weighted = jnp.einsum("ij,jk->ik", P, grid_pts)  # (n_grid, 2)

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
    """Interpolate transport map at sample points using JAX (linear interpolation)."""
    X_clipped = jnp.clip(sample_points, -domain_size, domain_size)
    grid_indices = (X_clipped + domain_size) * (grid_n - 1) / (2 * domain_size)
    coords = grid_indices.T  # (2, n_points)

    Ti_x = jax_map_coordinates(T_grid[:, :, 0], coords, order=1, mode="nearest")
    Ti_y = jax_map_coordinates(T_grid[:, :, 1], coords, order=1, mode="nearest")
    Ti = jnp.stack([Ti_x, Ti_y], axis=-1)
    return Ti


# ============================================================================
# Helper functions
# ============================================================================

def _sample_from_gmm2(
    rng: np.random.Generator,
    mean1: np.ndarray,
    mean2: np.ndarray,
    cov1: np.ndarray,
    cov2: np.ndarray,
    n: int,
) -> np.ndarray:
    """Sample n points from a balanced 2-component GMM (Gaussian mixture)."""
    n1 = n // 2
    n2 = n - n1
    X = np.vstack(
        [
            rng.multivariate_normal(mean1, cov1, n1),
            rng.multivariate_normal(mean2, cov2, n2),
        ]
    ).astype(np.float32)
    return X


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
    n_queries: int,
    query_uniform_frac: float,
    domain_size: float,
    reg_param: float,
    max_iterations: int,
    threshold: float,
    reg_param_target: str | None,
    reg_param_kw: str | None,
) -> dict[str, object]:
    """Create one OT record with decoupled query points."""
    # --- Source distribution parameters ---
    mean1 = rng.uniform(-2, 2, size=2).astype(np.float32)
    mean2 = rng.uniform(-2, 2, size=2).astype(np.float32)
    diag1 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
    diag2 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
    cov1 = np.diag(diag1)
    cov2 = np.diag(diag2)

    # --- Grid densities (rho0 on grid, rho1 fixed on grid) ---
    rho_src = (
        0.5 * multivariate_normal(mean1, cov1).pdf(grid_pts_np)
        + 0.5 * multivariate_normal(mean2, cov2).pdf(grid_pts_np)
    ).astype(np.float32)
    rho_src /= rho_src.sum()

    rho_tar = multivariate_normal([0, 0], 0.5 * np.eye(2)).pdf(grid_pts_np).astype(np.float32)
    rho_tar /= rho_tar.sum()

    a = jnp.array(rho_src, dtype=jnp.float32)
    b = jnp.array(rho_tar, dtype=jnp.float32)

    # --- Solve OT on grid ---
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

    # --- Barycentric map and velocity field on grid ---
    T_grid, V_bary = compute_transport_map_and_velocity(P, grid_pts_jax, a, grid_n)

    # --- Source samples (branch sensors) ---
    X = _sample_from_gmm2(rng, mean1, mean2, cov1, cov2, n_samples)
    X_jax = jnp.array(X)

    # Transport map evaluated at source points (legacy pointwise supervision / visualization)
    Ti_src = interpolate_transport_jax(T_grid, X_jax, domain_size, grid_n)
    Xf = Ti_src  # transported positions

    # --- Decoupled query points (trunk queries) ---
    n_u = int(round(float(n_queries) * float(query_uniform_frac)))
    n_u = max(0, min(n_queries, n_u))
    n_s = n_queries - n_u

    # Sample from source distribution (where the map matters most)
    Q_from_src = _sample_from_gmm2(rng, mean1, mean2, cov1, cov2, n_s) if n_s > 0 else np.zeros((0, 2), dtype=np.float32)

    # Sample uniformly over domain (forces global consistency)
    Q_uniform = rng.uniform(-domain_size, domain_size, size=(n_u, 2)).astype(np.float32) if n_u > 0 else np.zeros((0, 2), dtype=np.float32)

    Q = np.vstack([Q_from_src, Q_uniform]).astype(np.float32)
    if Q.shape[0] != n_queries:
        raise RuntimeError(f"Query sampler produced {Q.shape[0]} points, expected {n_queries}")

    # Shuffle queries to avoid any ordering artifacts
    perm = rng.permutation(n_queries)
    Q = Q[perm]

    Q_jax = jnp.array(Q)
    Tq = interpolate_transport_jax(T_grid, Q_jax, domain_size, grid_n)
    Vq = Tq - Q_jax  # displacement at query points

    # Convert to numpy
    V_bary_np = np.array(V_bary)
    Xf_np = np.array(Xf)
    Vq_np = np.array(Vq)

    return {
        "source_points": X.tolist(),
        "target_points": Xf_np.tolist(),
        "query_points": Q.tolist(),
        "query_vectors": Vq_np.tolist(),
        "velocity_field": V_bary_np,
        "source_params": {
            "mean1": mean1.tolist(),
            "mean2": mean2.tolist(),
            "cov1": cov1.tolist(),
            "cov2": cov2.tolist(),
        },
        "domain_size": float(domain_size),
    }


# ============================================================================
# Dataset building
# ============================================================================

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Build dataset directly by accumulating records in memory."""
    grid_n = kwargs["grid_n"]
    domain_size = kwargs["domain_size"]
    reg_param = kwargs["reg_param"]
    use_reg_param = kwargs.get("use_reg_param", True)
    max_iterations = kwargs["max_iterations"]
    threshold = kwargs.get("threshold", 1e-3)
    n_samples_per_record = kwargs["n_samples"]
    n_queries = kwargs["n_queries"]
    query_uniform_frac = kwargs.get("query_uniform_frac", 0.5)
    seed = kwargs.get("seed", None)

    xs = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    grid_pts_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    grid_pts_jax = jnp.array(grid_pts_np)

    rng = np.random.default_rng(seed)

    all_records = {
        "source_points": [],
        "target_points": [],
        "query_points": [],
        "query_vectors": [],
        "velocity_field": [],
        "source_params": [],
        "domain_size": [],
    }

    for i in tqdm(range(num_samples), desc="Generating samples"):
        if i % 100 == 0 and i > 0:
            gc.collect()

        record = make_record(
            rng=rng,
            grid_pts_np=grid_pts_np,
            grid_pts_jax=grid_pts_jax,
            grid_n=grid_n,
            n_samples=n_samples_per_record,
            n_queries=n_queries,
            query_uniform_frac=query_uniform_frac,
            domain_size=domain_size,
            reg_param=reg_param,
            max_iterations=max_iterations,
            threshold=threshold,
            reg_param_target=_REG_PARAM_TARGET if use_reg_param else None,
            reg_param_kw=_REG_PARAM_KW if use_reg_param else None,
        )

        all_records["source_points"].append(record["source_points"])
        all_records["target_points"].append(record["target_points"])
        all_records["query_points"].append(record["query_points"])
        all_records["query_vectors"].append(record["query_vectors"])
        all_records["velocity_field"].append(record["velocity_field"])
        all_records["source_params"].append(record["source_params"])
        all_records["domain_size"].append(record["domain_size"])

    features = Features(
        {
            "source_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "target_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "query_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "query_vectors": Sequence(feature=Sequence(Value("float32"), length=2)),
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
# Quality regression check (optional)
# ============================================================================

def run_quality_check(num_check: int = 10, seed: int = 42):
    """Compare JAX interpolation vs SciPy interpolation for QA."""
    print(f"[*] Running quality check on {num_check} samples...")

    rng = np.random.default_rng(seed)
    grid_n = 80
    domain_size = 5.0
    n_samples = 256
    reg_param = 0.05
    max_iterations = 5000
    threshold = 1e-3

    xs = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    grid_pts_np = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    grid_pts_jax = jnp.array(grid_pts_np)

    max_rel_error = 0.0

    for _ in range(num_check):
        mean1 = rng.uniform(-2, 2, size=2).astype(np.float32)
        mean2 = rng.uniform(-2, 2, size=2).astype(np.float32)
        diag1 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
        diag2 = rng.uniform(0.1, 1.0, size=2).astype(np.float32)
        cov1 = np.diag(diag1)
        cov2 = np.diag(diag2)

        rho_src = (
            0.5 * multivariate_normal(mean1, cov1).pdf(grid_pts_np)
            + 0.5 * multivariate_normal(mean2, cov2).pdf(grid_pts_np)
        ).astype(np.float32)
        rho_src /= rho_src.sum()

        rho_tar = multivariate_normal([0, 0], 0.5 * np.eye(2)).pdf(grid_pts_np).astype(np.float32)
        rho_tar /= rho_tar.sum()

        a = jnp.array(rho_src)
        b = jnp.array(rho_tar)

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
        T_grid, _ = compute_transport_map_and_velocity(P, grid_pts_jax, a, grid_n)
        T_grid_np = np.array(T_grid)

        X = _sample_from_gmm2(rng, mean1, mean2, cov1, cov2, n_samples)

        Ti_jax = np.array(interpolate_transport_jax(T_grid, jnp.array(X), domain_size, grid_n))

        X_clipped = np.clip(X, -domain_size, domain_size)
        grid_indices = (X_clipped + domain_size) * (grid_n - 1) / (2 * domain_size)
        coords = grid_indices.T
        Ti_x_scipy = scipy_map_coordinates(T_grid_np[:, :, 0], coords, order=1, mode="nearest")
        Ti_y_scipy = scipy_map_coordinates(T_grid_np[:, :, 1], coords, order=1, mode="nearest")
        Ti_scipy = np.column_stack([Ti_x_scipy, Ti_y_scipy])

        diff = Ti_jax - Ti_scipy
        rel_error = np.linalg.norm(diff) / (np.linalg.norm(Ti_scipy) + 1e-8)
        max_rel_error = max(max_rel_error, rel_error)

    print(f"    Max relative L2 error (JAX vs SciPy): {max_rel_error:.2e}")
    if max_rel_error < 1e-5:
        print("    Quality check PASSED")
        return True
    print("    Quality check: difference detected but within acceptable range")
    return max_rel_error < 1e-3


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate OT dataset with decoupled queries (Strategy 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", type=int, default=20000, help="Number of training samples")
    parser.add_argument("--test", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--grid", type=int, default=80, help="Grid resolution (N x N)")

    # Source samples (branch sensors)
    parser.add_argument("--n_samples", type=int, default=512, help="Number of source sample points per record")

    # Decoupled queries (trunk queries)
    parser.add_argument("--n_queries", type=int, default=1024, help="Number of query points per record")
    parser.add_argument(
        "--query_uniform_frac",
        type=float,
        default=0.0,
        help="Fraction of query points drawn uniformly over the domain (rest drawn from rho0)",
    )

    # Domain + OT params
    parser.add_argument("--domain_size", type=float, default=5.0, help="Domain size [-domain_size, domain_size]")
    parser.add_argument("--reg_param", type=float, default=0.075, help="Sinkhorn regularization parameter (epsilon)")
    parser.add_argument(
        "--use_reg_param",
        action="store_true",
        default=True,
        help="Use reg_param instead of OTT defaults",
    )
    parser.add_argument("--max_iterations", type=int, default=5000, help="Maximum Sinkhorn iterations")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Sinkhorn convergence threshold")

    parser.add_argument("--safe_check", action="store_true", help="Run quality regression check before generating")
    parser.add_argument("--seed", type=int, default=2, help="Global RNG seed")

    # Output path
    parser.add_argument(
        "--output_path",
        type=str,
        default="Data/transport_q_data/transport_dataset_reg_0.075",
        help="Output path for the dataset",
    )

    args = parser.parse_args()

    devices = jax.devices()
    print(f"[*] JAX devices: {devices}")

    if args.use_reg_param:
        print(f"[*] Using reg_param via {_REG_PARAM_TARGET}.{_REG_PARAM_KW} = {args.reg_param}")
    else:
        print("[*] Using OTT default regularization (reg_param ignored)")

    print(f"[*] Query points: {args.n_queries} ({args.query_uniform_frac*100:.0f}% uniform, {(1-args.query_uniform_frac)*100:.0f}% from rho0)")

    if args.safe_check:
        if not run_quality_check():
            print("Quality check failed. Aborting.")
            return

    params = dict(
        grid_n=args.grid,
        n_samples=args.n_samples,
        n_queries=args.n_queries,
        query_uniform_frac=args.query_uniform_frac,
        domain_size=args.domain_size,
        reg_param=args.reg_param,
        use_reg_param=args.use_reg_param,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        seed=args.seed,
    )

    total_samples = args.train + args.test

    print(f"[*] Generating {total_samples} samples ({args.train} train, {args.test} test)...")
    full_ds = build_dataset(total_samples, **params)

    print("[*] Splitting into train/test sets...")
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    print(f"[*] Saving dataset to {args.output_path}...")
    ds.save_to_disk(args.output_path)

    print(f"Done! Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset stored in {args.output_path}")


if __name__ == "__main__":
    main()
