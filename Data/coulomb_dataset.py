#!/usr/bin/env python
"""
coulomb_dataset.py
------------------
Generate a benchmark dataset for the proof-of-concept SetONet study:
    unordered set of 2-D point charges  →  electric potential field V(y).

Outputs
-------
* <out_file>.npz  – compressed NumPy archive with n_samples dicts:
      positions   : (M, 2)  float32
      charges     : (M,)    float32
      queries     : (K, 2)  float32
      potentials  : (K,)    float32
* A matplotlib window visualising the first sample on a dense grid.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ------------------------------------------------------------
# 1.  Physics helpers
# ------------------------------------------------------------
def sample_charges(
    M: int,
    box_size: float = 1.0,
    q_std: float = 2.0,
    rng: np.random.Generator | None = None,
):
    """Return positions (M,2) ∈ [0,box_size] and charge values (M,)."""
    if rng is None:
        rng = np.random.default_rng()
    positions = rng.uniform(0.0, box_size, size=(M, 2))
    
    # Mix of different charge distributions for more interesting fields
    if rng.random() < 0.3:  # 30% chance of strong dipoles
        charges = rng.choice([-3.0, -2.0, 2.0, 3.0], size=M)
    elif rng.random() < 0.5:  # 20% chance of uniform random charges
        charges = rng.uniform(-2.5, 2.5, size=M)
    else:  # 50% chance of normal distribution
        charges = rng.normal(0.0, q_std, size=M)
    
    return positions, charges


def coulomb_potential(
    positions: np.ndarray,
    charges:   np.ndarray,
    queries:   np.ndarray,
    eps0: float = 1.0,
    min_distance: float = 0.02,  # Minimum distance to avoid singularities
    use_soft_core: bool = True,  # Use soft-core potential for stability
):
    """
    Free-space 2-D Coulomb potential at each query point with regularization.

    V(y) = (1 / 2πϵ₀) Σ_i q_i / sqrt(||y – x_i||² + r_c²)
    
    Parameters:
    -----------
    min_distance : float
        Minimum distance to enforce between queries and particles
    use_soft_core : bool
        If True, uses soft-core potential: 1/sqrt(r² + r_c²)
        If False, uses hard cutoff with min_distance
    """
    diff = queries[:, None, :] - positions[None, :, :]    # shape (K, M, 2)
    r_squared = np.sum(diff**2, axis=-1)                  # squared distances
    
    if use_soft_core:
        # Soft-core potential: smoother and more stable
        r_core_squared = min_distance**2
        r_regularized = np.sqrt(r_squared + r_core_squared)
        V = np.sum(charges / r_regularized, axis=1) / (2.0 * np.pi * eps0)
    else:
        # Hard cutoff: enforce minimum distance
        r = np.sqrt(r_squared)
        r_regularized = np.maximum(r, min_distance)
        V = np.sum(charges / r_regularized, axis=1) / (2.0 * np.pi * eps0)
    
    return V                                              # shape (K,)


# ------------------------------------------------------------
# 2.  Single-sample & bulk generators
# ------------------------------------------------------------
def generate_safe_queries(
    positions: np.ndarray,
    K: int,
    box_size: float = 1.0,
    min_distance: float = 0.02,
    max_attempts: int = 10000,
    rng: np.random.Generator | None = None,
):
    """
    Generate query points that maintain minimum distance from all particles.
    
    Parameters:
    -----------
    positions : array [M, 2]
        Particle positions
    K : int
        Number of query points to generate
    min_distance : float
        Minimum allowed distance from any particle
    max_attempts : int
        Maximum attempts to find valid query points
    """
    if rng is None:
        rng = np.random.default_rng()
    
    queries = []
    attempts = 0
    
    while len(queries) < K and attempts < max_attempts:
        # Generate candidate query point
        candidate = rng.uniform(0.0, box_size, size=(2,))
        
        # Check distances to all particles
        distances = np.linalg.norm(positions - candidate[None, :], axis=1)
        min_dist_to_particles = np.min(distances)
        
        if min_dist_to_particles >= min_distance:
            queries.append(candidate)
        
        attempts += 1
    
    # If we couldn't find enough valid points, fill remaining with uniform random
    # (this shouldn't happen often with reasonable min_distance)
    while len(queries) < K:
        candidate = rng.uniform(0.0, box_size, size=(2,))
        queries.append(candidate)
    
    return np.array(queries)


def generate_sample(
    M: int = 100,
    K: int = 256,
    rng: np.random.Generator | None = None,
    fixed_positions: np.ndarray | None = None,
    use_safe_queries: bool = True,  # NEW: Use distance-constrained queries
    min_distance: float = 0.02,    # NEW: Minimum distance for safe queries
):
    """Return one dict sample compatible with a SetONet data loader."""
    if rng is None:
        rng = np.random.default_rng()
    
    if fixed_positions is not None:
        # Use provided fixed positions
        positions = fixed_positions.copy()
        # Generate only charges (keeping position-charge relationship varied)
        if rng.random() < 0.3:  # 30% chance of strong dipoles
            charges = rng.choice([-3.0, -2.0, 2.0, 3.0], size=M)
        elif rng.random() < 0.5:  # 20% chance of uniform random charges
            charges = rng.uniform(-2.5, 2.5, size=M)
        else:  # 50% chance of normal distribution
            charges = rng.normal(0.0, 2.0, size=M)
    else:
        # Original behavior: variable positions and charges
        positions, charges = sample_charges(M, rng=rng)
    
    if use_safe_queries:
        queries = generate_safe_queries(positions, K, min_distance=min_distance)
    else:
        queries = rng.uniform(0.0, 1.0, size=(K, 2))  # Always variable
    
    potentials     = coulomb_potential(positions, charges, queries)
    return dict(
        positions  = positions.astype(np.float32),
        charges    = charges.astype(np.float32),
        queries    = queries.astype(np.float32),
        potentials = potentials.astype(np.float32),
    )


def generate_dataset(
    n_samples: int = 10_000,
    M: int = 100,
    K: int = 256,
    out_file: str | Path | None = "coulomb_2D_train.npz",
    seed: int | None = 42,
    use_fixed_positions: bool = False,
    use_safe_queries: bool = True,     # NEW: Use distance-constrained queries
    min_distance: float = 0.02,       # NEW: Minimum distance for stability
):
    rng   = np.random.default_rng(seed)
    
    # Generate fixed particle positions if requested
    fixed_positions = None
    if use_fixed_positions:
        print(f"🔧 Generating FIXED particle positions for all {n_samples} samples")
        # Create a well-distributed set of particle positions
        fixed_positions = rng.uniform(0.0, 1.0, size=(M, 2))
        print(f"   Fixed positions shape: {fixed_positions.shape}")
    else:
        print(f"🔧 Using VARIABLE particle positions (original behavior)")
    
    # Safety information
    safety_mode = "SAFE (min_dist={:.3f})".format(min_distance) if use_safe_queries else "UNSAFE (no distance constraints)"
    print(f"🔧 Query generation mode: {safety_mode}")
    
    data  = [generate_sample(M, K, rng, fixed_positions, use_safe_queries, min_distance) for _ in range(n_samples)]
    if out_file is not None:
        out_file = Path(out_file)
        np.savez_compressed(out_file, **{f"s{i}": d for i, d in enumerate(data)})
        position_type = "FIXED" if use_fixed_positions else "VARIABLE"
        print(f"✓ Saved {n_samples:,} samples with {position_type} positions and {safety_mode} queries → {out_file.resolve()}")
    return data


# ------------------------------------------------------------
# 3.  Plotting helper
# ------------------------------------------------------------
def potential_on_grid(
    positions: np.ndarray,
    charges:   np.ndarray,
    n: int = 120,
):
    """Evaluate V(x,y) on an n×n grid – handy for diagnostics."""
    xv = np.linspace(0.0, 1.0, n)
    yv = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(xv, yv, indexing="xy")
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)     # (n², 2)
    V = coulomb_potential(positions, charges, grid_pts).reshape(n, n)
    return xx, yy, V


def plot_sample(sample: dict, grid_res: int = 120, save_path: str | Path | None = None):
    """Visualise potential field + charge locations for quick sanity check."""
    positions, charges = sample["positions"], sample["charges"]
    xx, yy, V = potential_on_grid(positions, charges, n=grid_res)

    plt.figure(figsize=(8, 6))
    
    # Use a more dynamic colormap range based on the actual data
    vmin, vmax = np.percentile(V, [5, 95])  # Use 5th-95th percentile for better contrast
    im = plt.imshow(
        V,
        origin="lower",
        extent=[0, 1, 0, 1],
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
        cmap="RdBu_r"  # Red-blue colormap: red=positive, blue=negative
    )
    
    # Plot charges as small dots with color indicating sign/magnitude
    sizes = 8 + 4 * np.abs(charges)  # Much smaller: base size + magnitude scaling
    colors = charges  # Use actual charge values for coloring
    scatter = plt.scatter(
        positions[:, 0],
        positions[:, 1],
        s=sizes,
        c=colors,
        cmap="RdBu_r",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
        vmin=-3, vmax=3  # Fixed color range for charges
    )
    
    plt.title(f"Electric Potential Field ({len(charges)} charges)")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Colorbar for the potential field
    cb = plt.colorbar(im, label="Electric Potential V", shrink=0.8)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2))
    
    # Add charge magnitude info
    plt.text(0.02, 0.98, f"Charge range: [{charges.min():.1f}, {charges.max():.1f}]", 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved → {save_path.resolve()}")
    else:
        plt.show()
    plt.close()


# ------------------------------------------------------------
# 4.  Main – create dataset *and* preview first sample
# ------------------------------------------------------------
if __name__ == "__main__":
    DATASET = generate_dataset(
        n_samples = 1_000,
        M         = 50,  # Start with 10 particles for testing
        K         = 64,  # Fewer query points too
        out_file  = "coulomb_2D_10particles_safe.npz",
        seed      = 123,
        use_fixed_positions = True,  # Fixed particle positions
        use_safe_queries = True,     # NEW: Safe query generation
        min_distance = 0.03,         # NEW: Conservative minimum distance
    )

    # Visualise several samples to confirm everything looks right
    for i in range(3):
        plot_sample(DATASET[i], save_path=f"Data/coulomb_plots/sample_10particles_safe_{i}.png")
