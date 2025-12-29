#!/usr/bin/env python
"""
generate_transport_plan_data.py
----------------------------------
Generate an Optimal Transport *transport-plan* dataset for operator learning.

Option 1 benchmark (amortized OT plan):
- Input:  X = {x_i}_{i=1..M} point cloud sampled from a source distribution rho0
          (mixture of two Gaussians, random parameters per sample)
- Target: fixed point cloud Y_fixed = {y_j}_{j=1..N} sampled once from rho1
          (Gaussian N(0, 0.5 I2)), reused for the whole dataset
- Labels: W* in R^{M x N}, the row-conditional assignment distribution
          W*_{ij} = P*_{ij} / a_i, where P* is the entropic OT coupling
          between empirical measures mu_X and nu_Y_fixed with uniform weights.

Saved fields per sample (HuggingFace dataset):
- source_points: (M,2) float32
- target_points: (N,2) float32  (duplicated across samples for convenience)
- transport_plan: (M,N) float32  (row-stochastic, rows sum to 1)
- source_params: dict(mean1, mean2, cov1, cov2)
- domain_size: float32
- epsilon: float32 (the entropic regularization used if --use_reg_param)
- ot_cost: float32 (objective value <P,C> with P = (1/M) * W)
- marginal_error: float32 (max marginal deviation for sanity/debug)

Notes:
- This script solves *discrete* OT between particles (no 80x80 grid).
- It uses OTT/JAX Sinkhorn. See:
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn
"""

from __future__ import annotations

import argparse
import gc
from functools import partial
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from datasets import Array2D, Dataset, Features, Sequence, Value
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from tqdm import tqdm


# -----------------------------------------------------------------------------
# OTT regularization wiring (kept similar to your old script style)
# -----------------------------------------------------------------------------

_REG_PARAM_TARGET = "pointcloud"
_REG_PARAM_KW = "epsilon"


# -----------------------------------------------------------------------------
# JIT-compiled Sinkhorn solver
# -----------------------------------------------------------------------------

@partial(
    jax.jit,
    static_argnames=(
        "max_iterations",
        "threshold",
        "reg_param",
        "reg_param_target",
        "reg_param_kw",
    ),
)
def solve_ot_coupling(
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
    """
    Solve entropic OT via Sinkhorn and return the coupling matrix P (shape MxN).

    x: (M,2), y: (N,2)
    a: (M,), b: (N,) (probability weights, sum to 1)
    """
    geom_kwargs: Dict[str, Any] = {}
    problem_kwargs: Dict[str, Any] = {}
    solver_kwargs: Dict[str, Any] = {
        "max_iterations": max_iterations,
        "threshold": threshold,
    }

    # Wire epsilon to the desired OTT component (default: PointCloud.epsilon)
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


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------

def _sample_source_mixture(
    rng: np.random.Generator,
    n_source: int,
    mean_range: Tuple[float, float] = (-2.0, 2.0),
    var_range: Tuple[float, float] = (0.1, 1.0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample X ~ 0.5 N(mean1, diag(var1)) + 0.5 N(mean2, diag(var2))."""
    mean1 = rng.uniform(mean_range[0], mean_range[1], size=(2,)).astype(np.float32)
    mean2 = rng.uniform(mean_range[0], mean_range[1], size=(2,)).astype(np.float32)

    diag1 = rng.uniform(var_range[0], var_range[1], size=(2,)).astype(np.float32)
    diag2 = rng.uniform(var_range[0], var_range[1], size=(2,)).astype(np.float32)

    cov1 = np.diag(diag1).astype(np.float32)
    cov2 = np.diag(diag2).astype(np.float32)

    n1 = n_source // 2
    n2 = n_source - n1

    x1 = rng.multivariate_normal(mean1, cov1, size=n1).astype(np.float32)
    x2 = rng.multivariate_normal(mean2, cov2, size=n2).astype(np.float32)
    X = np.concatenate([x1, x2], axis=0)

    rng.shuffle(X, axis=0)

    params = {
        "mean1": mean1.tolist(),
        "mean2": mean2.tolist(),
        "cov1": cov1.tolist(),
        "cov2": cov2.tolist(),
    }
    return X, params


def _sample_fixed_target_cloud(
    rng: np.random.Generator,
    n_target: int,
    cov_scale: float = 0.5,
    domain_size: float = 5.0,
) -> np.ndarray:
    """Sample Y_fixed ~ N(0, cov_scale * I2), then clip to domain bounds."""
    mean = np.array([0.0, 0.0], dtype=np.float32)
    cov = (cov_scale * np.eye(2)).astype(np.float32)
    Y = rng.multivariate_normal(mean, cov, size=n_target).astype(np.float32)
    Y = np.clip(Y, -domain_size, domain_size)
    return Y


def _compute_ot_cost_np(X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> float:
    """Compute <P, C> with C_ij = ||x_i - y_j||^2."""
    # (M,1,2) - (1,N,2) -> (M,N,2) -> (M,N)
    C = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1).astype(np.float32)
    return float(np.sum(P * C))


def make_record(
    *,
    rng_source: np.random.Generator,
    y_fixed_np: np.ndarray,
    y_fixed_jax: jnp.ndarray,
    n_source: int,
    domain_size: float,
    reg_param: float,
    max_iterations: int,
    threshold: float,
    use_reg_param: bool,
) -> Dict[str, Any]:
    """
    Create one sample:
      X (source points), fixed Y, and row-conditional plan W = P / a.
    """
    X_np, src_params = _sample_source_mixture(rng_source, n_source=n_source)
    X_np = np.clip(X_np, -domain_size, domain_size).astype(np.float32)

    M = X_np.shape[0]
    N = y_fixed_np.shape[0]

    # Uniform weights (empirical measures)
    a = jnp.full((M,), 1.0 / M, dtype=jnp.float32)
    b = jnp.full((N,), 1.0 / N, dtype=jnp.float32)

    P = solve_ot_coupling(
        jnp.array(X_np),
        y_fixed_jax,
        a,
        b,
        max_iterations=max_iterations,
        threshold=threshold,
        reg_param=reg_param,
        reg_param_target=_REG_PARAM_TARGET if use_reg_param else None,
        reg_param_kw=_REG_PARAM_KW if use_reg_param else None,
    )

    # Convert coupling -> row-conditional assignment W (rows sum to 1)
    # W_ij = P_ij / a_i
    W = P / a[:, None]

    # Bring back to numpy
    W_np = np.array(W, dtype=np.float32)

    # For sanity/debug: reconstruct P from W (P = diag(a) W)
    # Since a is uniform: P = (1/M) * W
    P_np = (W_np / float(M)).astype(np.float32)

    ot_cost = _compute_ot_cost_np(X_np, y_fixed_np, P_np)

    # Marginal error checks (should be small when Sinkhorn converged)
    marg_a = P_np.sum(axis=1)  # (M,)
    marg_b = P_np.sum(axis=0)  # (N,)
    marginal_error = float(
        max(
            np.max(np.abs(marg_a - (1.0 / M))),
            np.max(np.abs(marg_b - (1.0 / N))),
        )
    )

    return {
        "source_points": X_np,
        "target_points": y_fixed_np.astype(np.float32),
        "transport_plan": W_np,
        "source_params": src_params,
        "domain_size": np.float32(domain_size),
        "epsilon": np.float32(reg_param) if use_reg_param else np.float32(-1.0),
        "ot_cost": np.float32(ot_cost),
        "marginal_error": np.float32(marginal_error),
    }


# -----------------------------------------------------------------------------
# Dataset building
# -----------------------------------------------------------------------------

def build_dataset(
    *,
    num_samples: int,
    y_fixed_np: np.ndarray,
    n_source: int,
    domain_size: float,
    reg_param: float,
    use_reg_param: bool,
    max_iterations: int,
    threshold: float,
    seed: int,
) -> Dataset:
    """
    Build a HuggingFace Dataset in-memory (dict-of-lists).

    NOTE: Full W is (M,N). For M=N=256, each sample is ~256 KB float32.
    Keep dataset size reasonable unless you intentionally want multi-GB.
    """
    rng_source = np.random.default_rng(seed)
    y_fixed_jax = jnp.array(y_fixed_np, dtype=jnp.float32)

    all_records: Dict[str, list] = {
        "source_points": [],
        "target_points": [],
        "transport_plan": [],
        "source_params": [],
        "domain_size": [],
        "epsilon": [],
        "ot_cost": [],
        "marginal_error": [],
    }

    # Trigger JIT compilation once up front (compile cost paid once)
    _ = solve_ot_coupling(
        jnp.zeros((n_source, 2), dtype=jnp.float32),
        y_fixed_jax,
        jnp.full((n_source,), 1.0 / n_source, dtype=jnp.float32),
        jnp.full((y_fixed_np.shape[0],), 1.0 / y_fixed_np.shape[0], dtype=jnp.float32),
        max_iterations=max_iterations,
        threshold=threshold,
        reg_param=reg_param,
        reg_param_target=_REG_PARAM_TARGET if use_reg_param else None,
        reg_param_kw=_REG_PARAM_KW if use_reg_param else None,
    )

    for i in tqdm(range(num_samples), desc="samples"):
        if i > 0 and (i % 100 == 0):
            gc.collect()

        rec = make_record(
            rng_source=rng_source,
            y_fixed_np=y_fixed_np,
            y_fixed_jax=y_fixed_jax,
            n_source=n_source,
            domain_size=domain_size,
            reg_param=reg_param,
            max_iterations=max_iterations,
            threshold=threshold,
            use_reg_param=use_reg_param,
        )

        for k in all_records.keys():
            all_records[k].append(rec[k])

    n_target = int(y_fixed_np.shape[0])

    features = Features(
        {
            "source_points": Array2D(shape=(n_source, 2), dtype="float32"),
            "target_points": Array2D(shape=(n_target, 2), dtype="float32"),
            "transport_plan": Array2D(shape=(n_source, n_target), dtype="float32"),
            "source_params": {
                "mean1": Sequence(feature=Value("float32"), length=2),
                "mean2": Sequence(feature=Value("float32"), length=2),
                "cov1": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
                "cov2": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
            },
            "domain_size": Value("float32"),
            "epsilon": Value("float32"),
            "ot_cost": Value("float32"),
            "marginal_error": Value("float32"),
        }
    )

    return Dataset.from_dict(all_records, features=features)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OT transport-plan dataset (Option 1) using OTT/JAX."
    )

    # Defaults chosen to keep plan dataset size reasonable by default
    parser.add_argument("--train", type=int, default=10000, help="# training samples")
    parser.add_argument("--test", type=int, default=1000, help="# test samples")

    # Point-cloud sizes
    parser.add_argument("--n_source", type=int, default=256, help="# source points per sample (M)")
    parser.add_argument("--n_target", type=int, default=256, help="# fixed target points (N)")

    # Domain / distributions
    parser.add_argument("--domain_size", type=float, default=5.0, help="Domain [-domain_size, domain_size]")
    parser.add_argument("--target_cov_scale", type=float, default=0.5, help="rho1 = N(0, target_cov_scale I)")

    # Sinkhorn parameters
    parser.add_argument("--reg_param", type=float, default=0.05, help="Entropic reg (epsilon)")
    parser.add_argument("--use_reg_param", action="store_true", help="Wire reg_param into OTT (PointCloud.epsilon)")
    parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum Sinkhorn iterations")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Sinkhorn convergence threshold")

    # Seeds
    parser.add_argument("--seed", type=int, default=2, help="RNG seed for source sampling")
    parser.add_argument("--target_seed", type=int, default=123, help="RNG seed for Y_fixed sampling")

    # Output path
    parser.add_argument(
        "--out",
        type=str,
        default="Data/transport_data/transport_plan_dataset",
        help="Output directory for HuggingFace dataset (save_to_disk)",
    )

    args = parser.parse_args()

    # Devices
    devices = jax.devices()
    print(f"[•] JAX devices: {devices}")

    # Sample Y_fixed once
    rng_target = np.random.default_rng(args.target_seed)
    y_fixed_np = _sample_fixed_target_cloud(
        rng_target,
        n_target=args.n_target,
        cov_scale=args.target_cov_scale,
        domain_size=args.domain_size,
    )
    print(f"[•] Y_fixed sampled once: shape={y_fixed_np.shape}, seed={args.target_seed}")

    if args.use_reg_param:
        print(f"[•] Using epsilon via {_REG_PARAM_TARGET}.{_REG_PARAM_KW} = {args.reg_param}")
    else:
        print("[•] Using OTT default epsilon (reg_param ignored; epsilon stored as -1 in dataset)")

    total = args.train + args.test
    print(f"[•] Generating {total} samples...")

    full_ds = build_dataset(
        num_samples=total,
        y_fixed_np=y_fixed_np,
        n_source=args.n_source,
        domain_size=args.domain_size,
        reg_param=args.reg_param,
        use_reg_param=args.use_reg_param,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        seed=args.seed,
    )

    print("[•] Splitting train/test...")
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[•] Saving dataset to: {out_path}")
    ds.save_to_disk(str(out_path))

    print(f"✅ Done. Saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset directory: {out_path}")


if __name__ == "__main__":
    main()
