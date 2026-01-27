"""
Benchmark: Phase-Screen Diffraction (object set -> complex field at t0)

Dataset output:
- Uniform grid output: store full (grid_n, grid_n, 2) field [Re, Im]

Saved as a HuggingFace dataset (Dataset.from_generator), then split train/test and saved to disk.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
from datasets import Array3D, Dataset, Features, Sequence, Value
from tqdm import tqdm


def _loguniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    """Sample log-uniform in [low, high]."""
    low = float(low)
    high = float(high)
    if low <= 0 or high <= 0:
        raise ValueError("loguniform bounds must be > 0")
    log_low = np.log(low)
    log_high = np.log(high)
    return np.exp(rng.uniform(log_low, log_high, size=size)).astype(np.float32)


def build_phase_field(
    X: np.ndarray,
    Y: np.ndarray,
    bump_xy: np.ndarray,   # (N,2) float32
    bump_alpha: np.ndarray,  # (N,) float32
    bump_ell: np.ndarray,    # (N,) float32
    chunk_size: int = 128,
) -> np.ndarray:
    """Build periodic minimal-image Gaussian phase field Φ(X,Y) on a grid.

    Φ(X,Y) = Σ_i alpha_i * exp(-(dx^2+dy^2)/(2*ell_i^2)),
    where dx,dy are minimal-image distances on the unit torus.

    Notes:
    - Uses float64 accumulation for stability.
    - Chunking over bumps keeps peak memory modest.
    """
    if bump_xy.shape[0] == 0:
        return np.zeros_like(X, dtype=np.float64)

    bx = bump_xy[:, 0].astype(np.float64)
    by = bump_xy[:, 1].astype(np.float64)
    alpha = bump_alpha.astype(np.float64)
    ell = bump_ell.astype(np.float64)

    phi = np.zeros_like(X, dtype=np.float64)

    # Broadcast X,Y to float64 once
    X64 = X.astype(np.float64, copy=False)
    Y64 = Y.astype(np.float64, copy=False)

    N = bx.shape[0]
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        bxs = bx[s:e][None, None, :]        # (1,1,C)
        bys = by[s:e][None, None, :]        # (1,1,C)
        alphas = alpha[s:e][None, None, :]  # (1,1,C)
        ells = ell[s:e][None, None, :]      # (1,1,C)

        # minimal image on [0,1): dx in [-0.5, 0.5)
        dx = np.mod(X64[:, :, None] - bxs + 0.5, 1.0) - 0.5
        dy = np.mod(Y64[:, :, None] - bys + 0.5, 1.0) - 0.5
        r2 = dx * dx + dy * dy
        denom = 2.0 * (ells * ells) + 1e-18
        phi += np.sum(alphas * np.exp(-r2 / denom), axis=-1)

    return phi


def propagate_free_schrodinger_fft(
    u0: np.ndarray,  # (M,M) complex
    t0: float,
) -> np.ndarray:
    """Free Schrödinger propagation on the unit torus via FFT.

    PDE: i ∂_t u = - (1/2) Δ u
    Spectral: û(k,t) = exp(- i * 0.5 * |k|^2 * t) û(k,0)

    Returns:
        u_t: (M,M) complex128 (numpy FFT defaults); caller may cast.
    """
    M = u0.shape[0]
    freq = np.fft.fftfreq(M, d=1.0 / M)  # cycles per unit length
    k = 2.0 * np.pi * freq               # radians per unit length
    KX, KY = np.meshgrid(k, k, indexing="ij")
    prop = np.exp(-0.5j * (KX * KX + KY * KY) * float(t0))

    u_hat = np.fft.fft2(u0)
    u_t = np.fft.ifft2(u_hat * prop)
    return u_t


def make_record(
    *,
    rng: np.random.Generator,
    grid_n: int,
    n_min: int,
    n_max: int,
    alpha_low: float,
    alpha_high: float,
    ell_low: float,
    ell_high: float,
    sigma_env: float,
    t0: float,
    k_x: float,
    k_y: float,
) -> Dict[str, object]:
    """Create one (bumps, field) record."""
    n_bumps = int(rng.integers(n_min, n_max + 1))

    # Bump parameters
    bump_xy = rng.random(size=(n_bumps, 2), dtype=np.float32)  # (N,2) in [0,1)
    bump_alpha = rng.uniform(alpha_low, alpha_high, size=n_bumps).astype(np.float32)
    bump_ell = _loguniform(rng, ell_low, ell_high, size=n_bumps)

    # Grid on [0,1)
    xs = np.linspace(0.0, 1.0, grid_n, endpoint=False, dtype=np.float32)
    ys = xs
    X, Y = np.meshgrid(xs, ys, indexing="ij")  # (grid_n, grid_n)

    # Phase field Φ in float64
    phi = build_phase_field(X, Y, bump_xy, bump_alpha, bump_ell, chunk_size=128)

    # Modulo 2π for stability before exp(iΦ)
    two_pi = 2.0 * np.pi
    phi_mod = np.mod(phi + np.pi, two_pi) - np.pi

    # Envelope A(X,Y) in float64 -> float32
    Xc = X.astype(np.float64) - 0.5
    Yc = Y.astype(np.float64) - 0.5
    env = np.exp(-(Xc * Xc + Yc * Yc) / (2.0 * float(sigma_env) * float(sigma_env) + 1e-18)).astype(np.float32)

    # Carrier phase (centered at 0.5)
    carrier = (float(k_x) * (X.astype(np.float64) - 0.5) + float(k_y) * (Y.astype(np.float64) - 0.5))

    # Initial field u0 = A * exp(i*(carrier + Φ))
    phase = (carrier + phi_mod).astype(np.float32)
    # exp(i*phase) via cos/sin to keep complex64
    exp_i_phase = (np.cos(phase).astype(np.float32) + 1j * np.sin(phase).astype(np.float32)).astype(np.complex64)
    u0 = (env.astype(np.complex64) * exp_i_phase).astype(np.complex64)

    # Propagate to t0
    u_t = propagate_free_schrodinger_fft(u0, t0=t0)  # complex128 typically
    u_t = u_t.astype(np.complex64, copy=False)

    bumps = np.column_stack([bump_xy, bump_alpha[:, None], bump_ell[:, None]]).astype(np.float32)

    field = np.stack([u_t.real, u_t.imag], axis=-1).astype(np.float32)  # (grid_n,grid_n,2)
    return {
        "bumps": bumps.tolist(),  # variable-length list of 4-vectors
        "field": field,
        "t0": float(t0),
        "sigma_env": float(sigma_env),
        "k_x": float(k_x),
        "k_y": float(k_y),
    }


def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Stream-based builder to keep memory usage low."""

    def _gen():
        rng = np.random.default_rng(kwargs.pop("seed", None))
        for _ in tqdm(range(num_samples), desc="samples"):
            yield make_record(rng=rng, **kwargs)

    features = Features(
        {
            "bumps": Sequence(feature=Sequence(Value("float32"), length=4)),
            "field": Array3D(shape=(kwargs["grid_n"], kwargs["grid_n"], 2), dtype="float32"),
            "t0": Value("float32"),
            "sigma_env": Value("float32"),
            "k_x": Value("float32"),
            "k_y": Value("float32"),
        }
    )
    return Dataset.from_generator(_gen, features=features)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase-Screen Diffraction dataset (free 2D Schrödinger, FFT propagation)."
    )
    parser.add_argument("--train", type=int, default=20000, help="# training samples")
    parser.add_argument("--test", type=int, default=1000, help="# test samples")
    parser.add_argument("--grid", type=int, default=128, help="Grid resolution (N×N), on [0,1) with endpoint=False")

    # bump distribution
    parser.add_argument("--n_min", type=int, default=10, help="Min # phase bumps")
    parser.add_argument("--n_max", type=int, default=10, help="Max # phase bumps")
    parser.add_argument("--alpha_low", type=float, default=-np.pi/2, help="Lower bound of alpha (radians)")
    parser.add_argument("--alpha_high", type=float, default=np.pi/2, help="Upper bound of alpha (radians)")
    parser.add_argument("--ell_low", type=float, default=0.4, help="Lower bound of ell (log-uniform)")
    parser.add_argument("--ell_high", type=float, default=0.4, help="Upper bound of ell (log-uniform)")

    # physics constants
    parser.add_argument("--sigma_env", type=float, default=0.2, help="Envelope width sigma_env")
    parser.add_argument("--t0", type=float, default=0.1, help="Fixed time t0 for propagation")
    parser.add_argument("--k_x", type=float, default=0.0, help="Carrier wave k_x (radians per unit length)")
    parser.add_argument("--k_y", type=float, default=0.0, help="Carrier wave k_y (radians per unit length)")

    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")

    args = parser.parse_args()

    if args.grid <= 1:
        raise ValueError("--grid must be > 1")
    if args.n_min <= 0 or args.n_max < args.n_min:
        raise ValueError("--n_min must be > 0 and --n_max >= --n_min")
    if args.ell_low <= 0 or args.ell_high <= 0 or args.ell_high < args.ell_low:
        raise ValueError("--ell_low, --ell_high must be > 0 and ell_high >= ell_low")

    params = dict(
        grid_n=args.grid,
        n_min=args.n_min,
        n_max=args.n_max,
        alpha_low=args.alpha_low,
        alpha_high=args.alpha_high,
        ell_low=args.ell_low,
        ell_high=args.ell_high,
        sigma_env=args.sigma_env,
        t0=args.t0,
        k_x=args.k_x,
        k_y=args.k_y,
        seed=args.seed,
    )

    dataset_path = "Data/diffraction_data/phase_screen_dataset"

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    total_samples = int(args.train + args.test)

    print("[•] Generating full dataset …")
    full_ds = build_dataset(total_samples, **params)

    print("[•] Splitting into train/test sets …")
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)

    print("[•] Saving dataset …")
    ds.save_to_disk(dataset_path)

    params_path = os.path.join(dataset_path, "dataset_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"✅ Done. Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset stored in {dataset_path}")
    print(f"Dataset parameters saved to {params_path}")


if __name__ == "__main__":
    main()
