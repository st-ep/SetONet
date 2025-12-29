#!/usr/bin/env python
"""
generate_transport_bar_data.py
----------------------------------
Generate an Optimal Transport benchmark dataset (Option 1):

Input:
  - Source point cloud X = {x_i}_{i=1..M} sampled from a mixture of 2 Gaussians.

Fixed across the whole dataset:
  - Target point cloud Y_fixed = {y_j}_{j=1..N} sampled once from rho_1 = N(0, 0.5 I_2).

Labels (supervision only where mass is):
  - Barycentric OT map evaluated at source points:
        u*(x_i) = sum_j P_ij y_j / sum_j P_ij
    where P is the entropically-regularized OT coupling between
        mu = sum_i a_i delta_{x_i}  and  nu = sum_j b_j delta_{y_j},
    with uniform weights a_i=1/M, b_j=1/N.

We store per-sample:
  - source_points: X                               (M, 2)
  - transported_points: U* = {u*(x_i)}            (M, 2)
  - displacement: D* = U* - X                     (M, 2)
  - source_params: mixture parameters (means/covs)
  - domain_size, epsilon, etc (for reproducibility)

We also store globally in the dataset directory:
  - y_fixed.npy  (N, 2)
  - meta.json

Saved using HuggingFace `datasets` (DatasetDict.save_to_disk).
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from datasets import Dataset, Features, Sequence, Value

from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from tqdm import tqdm


# ---------------------------
# JAX / OTT core (JIT)
# ---------------------------

@partial(jax.jit, static_argnames=("max_iterations", "threshold", "epsilon"))
def solve_ot_coupling(
    x: jnp.ndarray,          # (M,2)
    y: jnp.ndarray,          # (N,2)
    a: jnp.ndarray,          # (M,)
    b: jnp.ndarray,          # (N,)
    *,
    epsilon: float,
    max_iterations: int,
    threshold: float,
) -> jnp.ndarray:
    """
    Solve entropic OT between two point clouds and return dense coupling matrix P (M,N).
    """
    geom = pointcloud.PointCloud(x, y, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom, a, b)
    solver = sinkhorn.Sinkhorn(max_iterations=max_iterations, threshold=threshold)
    out = solver(prob)
    return out.matrix


@jax.jit
def barycentric_map_from_coupling(
    P: jnp.ndarray,  # (M,N)
    y: jnp.ndarray,  # (N,2)
) -> jnp.ndarray:
    """
    Barycentric projection of coupling to a per-source map:
        u_i = (sum_j P_ij y_j) / (sum_j P_ij)
    """
    # (M,2)
    weighted_sum = P @ y
    # (M,1)
    row_sum = jnp.sum(P, axis=1, keepdims=True)
    return weighted_sum / (row_sum + 1e-8)


# ---------------------------
# Sampling utilities
# ---------------------------

@dataclass(frozen=True)
class SourceParams:
    mean1: list[float]
    mean2: list[float]
    cov1: list[list[float]]
    cov2: list[list[float]]

    @staticmethod
    def sample(rng: np.random.Generator) -> "SourceParams":
        mean1 = rng.uniform(-2.0, 2.0, size=(2,)).astype(np.float32)
        mean2 = rng.uniform(-2.0, 2.0, size=(2,)).astype(np.float32)

        diag1 = rng.uniform(0.1, 1.0, size=(2,)).astype(np.float32)
        diag2 = rng.uniform(0.1, 1.0, size=(2,)).astype(np.float32)

        cov1 = np.diag(diag1).astype(np.float32)
        cov2 = np.diag(diag2).astype(np.float32)

        return SourceParams(
            mean1=mean1.tolist(),
            mean2=mean2.tolist(),
            cov1=cov1.tolist(),
            cov2=cov2.tolist(),
        )

    def sample_points(
        self,
        rng: np.random.Generator,
        n: int,
        *,
        clip_to_domain: bool,
        domain_size: float,
    ) -> np.ndarray:
        mean1 = np.array(self.mean1, dtype=np.float32)
        mean2 = np.array(self.mean2, dtype=np.float32)
        cov1 = np.array(self.cov1, dtype=np.float32)
        cov2 = np.array(self.cov2, dtype=np.float32)

        n1 = n // 2
        n2 = n - n1

        x1 = rng.multivariate_normal(mean1, cov1, size=n1).astype(np.float32)
        x2 = rng.multivariate_normal(mean2, cov2, size=n2).astype(np.float32)
        X = np.vstack([x1, x2]).astype(np.float32)

        if clip_to_domain:
            X = np.clip(X, -domain_size, domain_size)

        return X


def sample_y_fixed(
    rng: np.random.Generator,
    n_target: int,
    *,
    clip_to_domain: bool,
    domain_size: float,
) -> np.ndarray:
    """
    Sample a fixed target cloud from N(0, 0.5 I2).
    """
    mean = np.zeros((2,), dtype=np.float32)
    cov = (0.5 * np.eye(2)).astype(np.float32)
    Y = rng.multivariate_normal(mean, cov, size=n_target).astype(np.float32)

    if clip_to_domain:
        Y = np.clip(Y, -domain_size, domain_size)

    return Y


# ---------------------------
# Record generation
# ---------------------------

def make_record(
    *,
    rng: np.random.Generator,
    y_fixed_jax: jnp.ndarray,   # (N,2)
    n_source: int,
    n_target: int,
    domain_size: float,
    clip_to_domain: bool,
    epsilon: float,
    max_iterations: int,
    threshold: float,
) -> dict[str, Any]:
    """
    Create a single sample:
      X ~ rho0, solve OT to Y_fixed, produce barycentric transported points U and displacement D.
    """
    src_params = SourceParams.sample(rng)
    X = src_params.sample_points(
        rng, n_source, clip_to_domain=clip_to_domain, domain_size=domain_size
    )

    x_jax = jnp.array(X, dtype=jnp.float32)

    # Uniform weights
    a = jnp.full((n_source,), 1.0 / float(n_source), dtype=jnp.float32)
    b = jnp.full((n_target,), 1.0 / float(n_target), dtype=jnp.float32)

    P = solve_ot_coupling(
        x_jax,
        y_fixed_jax,
        a,
        b,
        epsilon=epsilon,
        max_iterations=max_iterations,
        threshold=threshold,
    )

    U = barycentric_map_from_coupling(P, y_fixed_jax)  # (M,2)
    D = U - x_jax                                      # (M,2)

    U_np = np.array(U, dtype=np.float32)
    D_np = np.array(D, dtype=np.float32)

    return {
        "source_points": X.tolist(),
        "transported_points": U_np.tolist(),
        "displacement": D_np.tolist(),
        "source_params": asdict(src_params),
        "domain_size": float(domain_size),
        "epsilon": float(epsilon),
    }


# ---------------------------
# Dataset building
# ---------------------------

def build_dataset(
    num_samples: int,
    *,
    y_fixed: np.ndarray,
    n_source: int,
    domain_size: float,
    clip_to_domain: bool,
    epsilon: float,
    max_iterations: int,
    threshold: float,
    seed: int,
) -> Dataset:
    """
    Build a HuggingFace Dataset (single split) as dict-of-lists.
    """
    rng = np.random.default_rng(seed)

    y_fixed_jax = jnp.array(y_fixed, dtype=jnp.float32)
    n_target = int(y_fixed.shape[0])

    all_records: dict[str, list[Any]] = {
        "source_points": [],
        "transported_points": [],
        "displacement": [],
        "source_params": [],
        "domain_size": [],
        "epsilon": [],
    }

    for i in tqdm(range(num_samples), desc="samples"):
        if i % 200 == 0 and i > 0:
            gc.collect()

        rec = make_record(
            rng=rng,
            y_fixed_jax=y_fixed_jax,
            n_source=n_source,
            n_target=n_target,
            domain_size=domain_size,
            clip_to_domain=clip_to_domain,
            epsilon=epsilon,
            max_iterations=max_iterations,
            threshold=threshold,
        )

        for k in all_records.keys():
            all_records[k].append(rec[k])

    # Features: variable-length sequences allowed (outer Sequence has no fixed length).
    features = Features(
        {
            "source_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "transported_points": Sequence(feature=Sequence(Value("float32"), length=2)),
            "displacement": Sequence(feature=Sequence(Value("float32"), length=2)),
            "source_params": {
                "mean1": Sequence(Value("float32"), length=2),
                "mean2": Sequence(Value("float32"), length=2),
                "cov1": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
                "cov2": Sequence(feature=Sequence(Value("float32"), length=2), length=2),
            },
            "domain_size": Value("float32"),
            "epsilon": Value("float32"),
        }
    )

    return Dataset.from_dict(all_records, features=features)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate OT dataset (Option 1): source point cloud -> barycentric transport on source points."
    )

    # splits
    parser.add_argument("--train", type=int, default=20000, help="# training samples")
    parser.add_argument("--test", type=int, default=1000, help="# test samples")

    # point counts
    parser.add_argument("--n_source", type=int, default=256, help="# source points per sample (M)")
    parser.add_argument("--n_target", type=int, default=256, help="# fixed target points (N)")

    # domain / sampling
    parser.add_argument("--domain_size", type=float, default=5.0, help="Domain bound for optional clipping [-D, D]")
    parser.add_argument("--clip_to_domain", action="store_true", help="Clip sampled points to [-domain_size, domain_size]")

    # OT / Sinkhorn
    parser.add_argument("--epsilon", type=float, default=0.05, help="Entropic regularization epsilon for PointCloud")
    parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum Sinkhorn iterations")
    parser.add_argument("--threshold", type=float, default=1e-3, help="Sinkhorn convergence threshold")

    # reproducibility
    parser.add_argument("--seed", type=int, default=2, help="Global RNG seed")
    parser.add_argument("--y_fixed_seed", type=int, default=12345, help="Seed for Y_fixed sampling")

    # I/O
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="Data/transport_data/transport_plan_dataset",
        help="Where to save the HuggingFace dataset",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite dataset_path if it exists")

    args = parser.parse_args()

    # Report devices (helpful to confirm GPU use)
    print(f"[•] JAX devices: {jax.devices()}")

    dataset_path = Path(args.dataset_path)
    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset path already exists: {dataset_path}\n"
                f"Use --overwrite to remove it."
            )
        print(f"[•] Removing existing dataset directory: {dataset_path}")
        shutil.rmtree(dataset_path)

    # Sample fixed target cloud once
    rng_y = np.random.default_rng(args.y_fixed_seed)
    y_fixed = sample_y_fixed(
        rng_y,
        args.n_target,
        clip_to_domain=args.clip_to_domain,
        domain_size=args.domain_size,
    )

    # Build full dataset then split
    total = args.train + args.test
    print(f"[•] Generating {total} samples (train={args.train}, test={args.test}) ...")

    full_ds = build_dataset(
        total,
        y_fixed=y_fixed,
        n_source=args.n_source,
        domain_size=args.domain_size,
        clip_to_domain=args.clip_to_domain,
        epsilon=args.epsilon,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        seed=args.seed,
    )

    print("[•] Splitting into train/test sets (no shuffle)...")
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    print(f"[•] Saving HuggingFace dataset to: {dataset_path}")
    ds.save_to_disk(str(dataset_path))

    # Save Y_fixed and metadata alongside the dataset
    np.save(str(dataset_path / "y_fixed.npy"), y_fixed.astype(np.float32))

    meta = {
        "task": "ot_barycentric_map_on_source_points",
        "n_source": int(args.n_source),
        "n_target": int(args.n_target),
        "domain_size": float(args.domain_size),
        "clip_to_domain": bool(args.clip_to_domain),
        "epsilon": float(args.epsilon),
        "max_iterations": int(args.max_iterations),
        "threshold": float(args.threshold),
        "seed": int(args.seed),
        "y_fixed_seed": int(args.y_fixed_seed),
        "source_distribution": "balanced mixture of 2 Gaussians (means ~ Unif([-2,2]^2), diag cov entries ~ Unif([0.1,1.0]))",
        "target_distribution": "Gaussian N(0, 0.5 I2) sampled once as fixed target point cloud",
    }
    with open(dataset_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Done. Saved: {len(ds['train'])} train, {len(ds['test'])} test")
    print(f"    Dataset directory: {dataset_path}")
    print(f"    Fixed target cloud: {dataset_path / 'y_fixed.npy'}")
    print(f"    Metadata: {dataset_path / 'meta.json'}")


if __name__ == "__main__":
    main()
