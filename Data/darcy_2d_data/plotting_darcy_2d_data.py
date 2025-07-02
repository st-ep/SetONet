"""Generate overview plots of the Darcy 2D dataset.

Usage:
    python plotting_darcy_2d_data.py

Features
--------
* Loads both train and test datasets created by `make_dataset.py`.
* Plots 3 random samples from train and 3 from test.
* Shows permeability k(x,y) and pressure p(x,y) side-by-side.
* Validates data quality with discrete residual checks.
* Saves high-quality plots to darcy_2d_data_plots/ directory.
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk

import config as cfg
from solve import _build_matrix, IDX, NX, NY, N, h

def discrete_residual(k: np.ndarray, p: np.ndarray) -> float:
    """Return max |A*p - b| where A is the same matrix used by the solver."""
    # Use the same matrix assembly as the solver
    A = _build_matrix(k)
    
    # Create the same RHS as the solver (double precision)
    b = np.zeros(A.shape[0], dtype=np.float64)
    
    # Boundary conditions
    left_nodes = IDX[0, :]
    right_nodes = IDX[N, :]
    
    b[left_nodes] = 1.0
    b[right_nodes] = 0.0
    
    # Add boundary contributions with same precision as solver
    for iy in range(NY):
        if 1 < N:
            idx_adj = IDX[1, iy]
            k1, k2 = k[1, iy], k[0, iy]
            eps = 1e-15  # Match solver precision
            k_w = 2.0 * k1 * k2 / (k1 + k2 + eps)
            coeff_w = k_w / (h * h)
            b[idx_adj] += coeff_w * 1.0
    
    # Compute residual using double precision
    p_vec = p.astype(np.float64).flatten()
    residual_vec = A.dot(p_vec) - b
    
    return float(np.abs(residual_vec).max())

def main():
    # Create output directory
    output_dir = "/home/titanv/Stepan/setprojects/SetONet/Data/darcy_2d_data/darcy_2d_data_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load both train and test datasets
    train_path = Path("darcy64/train")
    test_path = Path("darcy64/test")
    
    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found at '{train_path}'. Run make_dataset.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at '{test_path}'. Run make_dataset.py first.")

    ds_train = load_from_disk(str(train_path))
    ds_test = load_from_disk(str(test_path))
    
    print(f"Loaded {len(ds_train)} train samples and {len(ds_test)} test samples.")

    # Select 3 random samples from train and test
    train_indices = np.random.choice(len(ds_train), 3, replace=False)
    test_indices = np.random.choice(len(ds_test), 3, replace=False)
    
    # Create individual plots for train samples
    for i, idx in enumerate(train_indices):
        sample = ds_train[int(idx)]
        k = np.array(sample["k"])
        p = np.array(sample["p"])
        
        # Verify this sample
        res = discrete_residual(k, p)
        print(f"Train sample {idx}: k.range=[{k.min():.3f}, {k.max():.3f}], p.range=[{p.min():.3f}, {p.max():.3f}], residual={res:.2e}")
        
        # Create individual plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Physical domain extent (0,1) x (0,1)
        extent = [0, 1, 0, 1]
        
        # Plot k
        im_k = axs[0].imshow(k, cmap='viridis', extent=extent, origin='lower')
        axs[0].set_title(f"Permeability k(x,y)")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        plt.colorbar(im_k, ax=axs[0], fraction=0.046)
        
        # Plot p
        im_p = axs[1].imshow(p, cmap='coolwarm', extent=extent, origin='lower')
        axs[1].set_title(f"Pressure p(x,y)")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        plt.colorbar(im_p, ax=axs[1], fraction=0.046)
        
        fig.suptitle(f"Train Sample {idx} (Residual: {res:.2e})", fontsize=14)
        fig.tight_layout()
        
        # Save individual train plot
        output_file = os.path.join(output_dir, f"train_sample_{idx:03d}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Train plot saved to: {output_file}")
        plt.close()
    
    # Create individual plots for test samples
    for i, idx in enumerate(test_indices):
        sample = ds_test[int(idx)]
        k = np.array(sample["k"])
        p = np.array(sample["p"])
        
        # Verify this sample
        res = discrete_residual(k, p)
        print(f"Test sample {idx}: k.range=[{k.min():.3f}, {k.max():.3f}], p.range=[{p.min():.3f}, {p.max():.3f}], residual={res:.2e}")
        
        # Create individual plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Physical domain extent (0,1) x (0,1)
        extent = [0, 1, 0, 1]
        
        # Plot k
        im_k = axs[0].imshow(k, cmap='viridis', extent=extent, origin='lower')
        axs[0].set_title(f"Permeability k(x,y)")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        plt.colorbar(im_k, ax=axs[0], fraction=0.046)
        
        # Plot p
        im_p = axs[1].imshow(p, cmap='coolwarm', extent=extent, origin='lower')
        axs[1].set_title(f"Pressure p(x,y)")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        plt.colorbar(im_p, ax=axs[1], fraction=0.046)
        
        fig.suptitle(f"Test Sample {idx} (Residual: {res:.2e})", fontsize=14)
        fig.tight_layout()
        
        # Save individual test plot
        output_file = os.path.join(output_dir, f"test_sample_{idx:03d}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Test plot saved to: {output_file}")
        plt.close()


if __name__ == "__main__":
    main()
