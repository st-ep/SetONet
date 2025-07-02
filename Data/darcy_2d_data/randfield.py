"""Random log‑normal permeability sampler.

A single public helper:
    sample_log_k(rng) -> np.ndarray   # shape (nx, ny)

Uses FFT filtering of white noise to obtain a Gaussian Random Field
with exponential‑squared covariance and then exponentiates to get k(x,y).
"""
from __future__ import annotations

import numpy as np
import config as cfg

__all__ = ["sample_log_k"]

def _fft_freq(n: int) -> np.ndarray:
    """Normalized FFT frequencies (0, 1/n, 2/n, …, -1/n, …)."""
    return np.fft.fftfreq(n)


def sample_log_k(
    rng: np.random.Generator,
    *,
    nx: int = cfg.RESOLUTION + 1,
    ny: int = cfg.RESOLUTION + 1,
    corr_len: float = cfg.CORR_LEN,
    log_std: float = cfg.LOG_STD,
) -> np.ndarray:
    """Return one log‑normal permeability field k(x,y).

    Parameters
    ----------
    rng       : NumPy Generator for reproducibility.
    nx, ny    : Grid size (defaults follow cfg.RESOLUTION).
    corr_len  : Correlation length as fraction of domain.
    log_std   : Standard deviation of the underlying Gaussian field.

    Notes
    -----
    • We use an RBF kernel in spectral form:\n        exp(‑0.5 * (2πL)^2 |k|^2)\n      where L is the correlation length.\n    • The output is `float32` to keep dataset size modest.
    """
    # 1. frequency grid & filter
    kx = _fft_freq(nx).reshape(-1, 1)  # column
    ky = _fft_freq(ny).reshape(1, -1)  # row
    k_squared = kx ** 2 + ky ** 2
    filt = np.exp(-0.5 * (2 * np.pi * corr_len) ** 2 * k_squared)

    # 2. complex white noise (Hermitian symmetry not needed for real ifft)
    noise = rng.normal(size=(nx, ny)) + 1j * rng.normal(size=(nx, ny))

    # 3. filtered field in Fourier domain → inverse FFT → real field
    g_hat = noise * filt
    g = np.fft.ifft2(g_hat).real

    # 4. normalise & exponentiate to get log‑normal k(x,y)
    g = (log_std / g.std()) * g
    return np.exp(g).astype(np.float32)
