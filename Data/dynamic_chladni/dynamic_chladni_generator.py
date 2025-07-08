#!/usr/bin/env python
"""
Generate Dynamic Chladni Plate Dataset (Point Cloud Version)
- Generate forces with fixed magnitudes but dynamic locations.
- Each sample has multiple force points: [x, y, F] where F is force magnitude.
- Compute displacement field at uniform grid points.
- Split into training and testing samples.
- Save in point cloud format similar to heat dataset.
"""

import numpy as np
from scipy.integrate import quad
from datasets import Dataset, Features, Sequence, Value, Array3D
from tqdm import tqdm
import argparse

def chladni_green_function(
    target_coords: np.ndarray,
    force_coords: np.ndarray,
    force_magnitudes: np.ndarray,
    L: float, M: float, omega: float, t_fixed: float, 
    gamma: float, v: float, n_max: int = 6, m_max: int = 6,
    eps: float = 1e-3
) -> np.ndarray:
    """
    Compute displacement field at target coordinates due to point forces.
    
    Uses modal expansion similar to original Chladni generator but with point forces
    instead of distributed modal coefficients.
    
    Args:
        target_coords: (N_target, 2) coordinates where displacement is computed
        force_coords: (N_forces, 2) coordinates where forces are applied
        force_magnitudes: (N_forces,) magnitude of each force
        L, M: plate dimensions
        omega: driving frequency
        t_fixed: time at which solution is evaluated
        gamma, v: damping parameters
        n_max, m_max: maximum mode numbers
        eps: regularization parameter
        
    Returns:
        displacement: (N_target,) displacement values at target coordinates
    """
    
    # Precompute modal parameters
    n_vals = np.arange(1, n_max + 1)
    m_vals = np.arange(1, m_max + 1)
    
    mu_vals = n_vals * np.pi / L  # shape: (n_max,)
    lam_vals = m_vals * np.pi / M  # shape: (m_max,)
    
    # Compute time integrals and mode factors for each (n,m) mode
    displacement = np.zeros(len(target_coords), dtype=np.float32)
    
    for n_idx, n in enumerate(n_vals):
        for m_idx, m in enumerate(m_vals):
            mu_n = mu_vals[n_idx]
            lam_m = lam_vals[m_idx]
            
            # Modal frequency
            beta_nm = np.sqrt(mu_n**2 + lam_m**2 + 3*v**2 - gamma**4)
            
            # Time integral (same as original)
            def integrand(tau):
                return (np.sin(omega * (tau - t_fixed)) * 
                       np.exp(-gamma**2 + v**2 * tau) * 
                       np.sin(beta_nm * tau))
            
            time_integral, _ = quad(integrand, 0, t_fixed)
            
            # Mode factor
            mode_factor = (v**2 / beta_nm) * time_integral * (4/(L*M))
            
            # Compute modal contribution from all forces
            modal_amplitude = 0.0
            
            for force_x, force_y, force_mag in zip(force_coords[:, 0], force_coords[:, 1], force_magnitudes):
                # Force contribution to this mode (assuming point force representation)
                # This replaces the modal coefficient alpha(n,m) with actual force projection
                force_projection = (force_mag * 
                                  np.cos(mu_n * force_x) * np.cos(lam_m * force_y) *
                                  np.cos(mu_n * (L/2)) * np.cos(lam_m * (M/2)))
                modal_amplitude += force_projection
            
            # Add this mode's contribution to displacement at all target points
            for i, (target_x, target_y) in enumerate(target_coords):
                mode_shape = np.cos(mu_n * target_x) * np.cos(lam_m * target_y)
                displacement[i] += modal_amplitude * mode_factor * mode_shape
    
    return displacement

def make_record(
    *,
    rng: np.random.Generator,
    grid_n: int,
    n_forces_min: int,
    n_forces_max: int,
    force_magnitude_min: float,
    force_magnitude_max: float,
    L: float, M: float, omega: float, t_fixed: float,
    gamma: float, v: float,
    constant_force: bool = False,
    n_max: int = 6, m_max: int = 6
) -> dict[str, object]:
    """Create one record with dynamic force locations and uniform grid output."""
    
    # Generate fixed number of forces (or random if min != max)
    if n_forces_min == n_forces_max:
        n_forces = n_forces_min
    else:
        n_forces = rng.integers(n_forces_min, n_forces_max + 1)
    
    # Generate force locations in normalized coordinates [0, 1]^2
    force_coords_norm = rng.random(size=(n_forces, 2), dtype=np.float32)
    
    # Convert to physical coordinates
    force_coords = force_coords_norm.copy()
    force_coords[:, 0] *= L  # x coordinates
    force_coords[:, 1] *= M  # y coordinates
    
    # Generate force magnitudes
    if constant_force:
        force_magnitudes = np.ones(n_forces, dtype=np.float32)
    else:
        # Log-uniform distribution
        log_low = np.log10(force_magnitude_min)
        log_high = np.log10(force_magnitude_max)
        force_magnitudes = 10 ** rng.uniform(log_low, log_high, size=n_forces).astype(np.float32)
    
    # Create forces array: [x_norm, y_norm, force_magnitude]
    forces = np.column_stack([force_coords_norm, force_magnitudes]).astype(np.float32)
    
    # Create uniform target grid (normalized coordinates)
    target_x_norm = np.linspace(0, 1, grid_n, dtype=np.float32)
    target_y_norm = np.linspace(0, 1, grid_n, dtype=np.float32)
    target_X_norm, target_Y_norm = np.meshgrid(target_x_norm, target_y_norm, indexing='ij')
    
    # Convert to physical coordinates for computation
    target_x = target_x_norm * L
    target_y = target_y_norm * M
    target_X, target_Y = np.meshgrid(target_x, target_y, indexing='ij')
    target_coords = np.column_stack([target_X.flatten(), target_Y.flatten()])
    
    # Compute displacement field using Green's function approach
    displacement_field = chladni_green_function(
        target_coords, force_coords, force_magnitudes,
        L, M, omega, t_fixed, gamma, v, n_max, m_max
    )
    
    # Reshape displacement to grid format and add channel dimension
    displacement_grid = displacement_field.reshape(grid_n, grid_n, 1).astype(np.float32)
    
    return {
        "sources": forces.tolist(),  # variable-length list of [x_norm, y_norm, force] triplets
        "field": displacement_grid,  # displacement field (grid_n, grid_n, 1)
    }

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Stream-based builder to keep memory usage low."""
    
    def _gen():
        rng = np.random.default_rng(kwargs.pop("seed", None))
        for _ in tqdm(range(num_samples), desc="samples"):
            yield make_record(rng=rng, **kwargs)
    
    # Define features
    features = Features(
        {
            "sources": Sequence(feature=Sequence(Value("float32"), length=3)),
            "field": Array3D(shape=(kwargs["grid_n"], kwargs["grid_n"], 1), dtype="float32"),
        }
    )
    return Dataset.from_generator(_gen, features=features)

def main():
    parser = argparse.ArgumentParser(description="Generate Dynamic Chladni dataset with point cloud forces.")
    parser.add_argument("--train", type=int, default=10_000, help="# training samples")
    parser.add_argument("--test", type=int, default=1_000, help="# test samples")
    parser.add_argument("--grid", type=int, default=64, help="Grid resolution (N×N)")
    
    # Force distribution
    parser.add_argument("--n_forces_min", type=int, default=5, help="Min # forces")
    parser.add_argument("--n_forces_max", type=int, default=15, help="Max # forces")
    parser.add_argument("--constant_force", action="store_true", help="Set all force magnitudes = 1")
    parser.add_argument("--force_min", type=float, default=0.01, help="Min force magnitude")
    parser.add_argument("--force_max", type=float, default=0.1, help="Max force magnitude")
    
    # Physical parameters
    parser.add_argument("--L", type=float, default=8.75 * 0.0254, help="Plate length (m)")
    parser.add_argument("--M", type=float, default=8.75 * 0.0254, help="Plate width (m)")
    parser.add_argument("--omega", type=float, default=None, help="Driving frequency (auto-computed if None)")
    parser.add_argument("--t_fixed", type=float, default=6.0, help="Time at which solution is evaluated")
    parser.add_argument("--gamma", type=float, default=0.02, help="Damping parameter")
    parser.add_argument("--v", type=float, default=0.5, help="Velocity parameter")
    
    # Modal parameters
    parser.add_argument("--n_max", type=int, default=6, help="Maximum n mode number")
    parser.add_argument("--m_max", type=int, default=6, help="Maximum m mode number")
    
    parser.add_argument("--seed", type=int, default=0, help="Global RNG seed")
    
    args = parser.parse_args()
    
    # Compute omega if not provided
    if args.omega is None:
        args.omega = 50 * np.pi / args.M
    
    params = dict(
        grid_n=args.grid,
        n_forces_min=args.n_forces_min,
        n_forces_max=args.n_forces_max,
        force_magnitude_min=args.force_min,
        force_magnitude_max=args.force_max,
        constant_force=args.constant_force,
        L=args.L,
        M=args.M,
        omega=args.omega,
        t_fixed=args.t_fixed,
        gamma=args.gamma,
        v=args.v,
        n_max=args.n_max,
        m_max=args.m_max,
        seed=args.seed,
    )
    
    dataset_path = "Data/dynamic_chladni/dynamic_chladni_dataset"
    
    # Calculate total samples needed
    total_samples = args.train + args.test
    
    print(f"[•] Generating {total_samples} samples with dynamic force locations...")
    print(f"    - Forces per sample: {args.n_forces_min} to {args.n_forces_max}")
    print(f"    - Force magnitude range: {args.force_min} to {args.force_max}")
    print(f"    - Grid resolution: {args.grid}×{args.grid}")
    print(f"    - Physical dimensions: {args.L:.4f}m × {args.M:.4f}m")
    print(f"    - Frequency: {args.omega:.2f} rad/s")
    
    # Generate full dataset
    full_ds = build_dataset(total_samples, **params)
    
    print("[•] Splitting into train/test sets...")
    # Split dataset
    ds = full_ds.train_test_split(test_size=args.test, shuffle=False)
    
    print("[•] Saving dataset...")
    ds.save_to_disk(dataset_path)
    
    print(f"✅ Done. Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Dataset stored in {dataset_path}")

if __name__ == "__main__":
    main() 