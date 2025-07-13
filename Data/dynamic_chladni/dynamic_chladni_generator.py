#!/usr/bin/env python
"""
Generate Dynamic Chladni Plate Dataset (Point Cloud Version)
- Generate forces with dynamic locations and magnitudes.
- Each sample has multiple force points: [x, y, F] where F is force magnitude.
- Compute displacement field using modal superposition for thin plate vibration.
- Split into training and testing samples.
- Save in point cloud format similar to heat dataset.

Physics: Solves the forced vibration problem for a thin plate using modal expansion.
The displacement field is expressed as a sum of normal modes, each excited by
the applied point forces through modal projection.
"""

import numpy as np
from scipy.integrate import quad
from datasets import Dataset, Features, Sequence, Value, Array3D
from tqdm import tqdm
import argparse

def precompute_chladni_modal_params(
    L: float, M: float, omega: float, t_fixed: float, 
    gamma: float, v: float, n_max: int = 6, m_max: int = 6
) -> dict:
    """
    Precompute modal parameters for Chladni plate vibration analysis.
    """
    print("Precomputing modal parameters...")
    
    # Modal parameters
    n_vals = np.arange(1, n_max + 1)
    m_vals = np.arange(1, m_max + 1)
    
    mu_vals = n_vals * np.pi / L  # shape: (n_max,)
    lam_vals = m_vals * np.pi / M  # shape: (m_max,)
    
    # Precompute center factors
    centerFactor = np.cos(mu_vals[:, None] * (L/2)) * np.cos(lam_vals[None, :] * (M/2))
    
    # Precompute beta values
    mu_squared = mu_vals[:, None]**2  # shape: (n_max, 1)
    lam_squared = lam_vals[None, :]**2  # shape: (1, m_max)
    beta_nm = np.sqrt(mu_squared + lam_squared + 3*v**2 - gamma**4)
    
    # Precompute time integrals (this is the expensive part)
    timeInt = np.zeros((n_max, m_max))
    
    for n in range(n_max):
        for m in range(m_max):
            current_beta = beta_nm[n, m]
            
            def integrand(tau):
                return (np.sin(omega * (tau - t_fixed)) * 
                       np.exp(-gamma**2 + v**2 * tau) * 
                       np.sin(current_beta * tau))
            
            timeInt[n, m], _ = quad(integrand, 0, t_fixed)
    
    # Precompute mode factors
    modeFactor = (v**2 / beta_nm) * timeInt * (4/(L*M)) * centerFactor
    
    return {
        'mu_vals': mu_vals,
        'lam_vals': lam_vals,
        'modeFactor': modeFactor,
        'n_max': n_max,
        'm_max': m_max,
        'L': L,
        'M': M
    }

def compute_chladni_displacement_field(
    target_coords: np.ndarray,
    force_coords: np.ndarray,
    force_magnitudes: np.ndarray,
    modal_params: dict
) -> np.ndarray:
    """
    Compute displacement field at target coordinates due to point forces.
    
    Args:
        target_coords: (N_target, 2) coordinates where displacement is computed
        force_coords: (N_forces, 2) coordinates where forces are applied  
        force_magnitudes: (N_forces,) magnitude of each force
        modal_params: precomputed modal parameters from precompute_chladni_modal_params
        
    Returns:
        displacement: (N_target,) displacement values at target coordinates
    """
    mu_vals = modal_params['mu_vals']
    lam_vals = modal_params['lam_vals']
    modeFactor = modal_params['modeFactor']
    n_max = modal_params['n_max']
    m_max = modal_params['m_max']
    L = modal_params['L']
    M = modal_params['M']
    
    # Vectorized computation of mode shapes at all target points
    # target_coords: (N_target, 2)
    target_x = target_coords[:, 0]  # (N_target,)
    target_y = target_coords[:, 1]  # (N_target,)
    
    # Compute cos(mu_n * target_x) for all n and all target points
    # Shape: (n_max, N_target)
    cos_mu_x = np.cos(mu_vals[:, None] * target_x[None, :])
    
    # Compute cos(lam_m * target_y) for all m and all target points
    # Shape: (m_max, N_target)
    cos_lam_y = np.cos(lam_vals[:, None] * target_y[None, :])
    
    # Initialize displacement
    displacement = np.zeros(len(target_coords), dtype=np.float32)
    
    # Loop over modes (much faster now since time integrals are precomputed)
    for n in range(n_max):
        for m in range(m_max):
            mu_n = mu_vals[n]
            lam_m = lam_vals[m]
            
            # Compute modal contribution from all forces
            modal_amplitude = 0.0
            
            for force_x, force_y, force_mag in zip(force_coords[:, 0], force_coords[:, 1], force_magnitudes):
                # Force contribution to this mode
                force_projection = (force_mag * 
                                  np.cos(mu_n * force_x) * np.cos(lam_m * force_y))
                modal_amplitude += force_projection
            
            # Add this mode's contribution to displacement at all target points (vectorized)
            mode_shape = cos_mu_x[n, :] * cos_lam_y[m, :]  # (N_target,)
            displacement += modal_amplitude * modeFactor[n, m] * mode_shape
    
    return displacement

def make_record(
    *,
    rng: np.random.Generator,
    grid_n: int,
    n_forces: int,
    force_min: float,
    force_max: float,
    L: float, M: float,
    modal_params: dict,
    constant_force: bool = False
) -> dict[str, object]:
    """Create one record with dynamic force locations and uniform grid output."""
    
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
        # Log-uniform distribution from force_min to force_max
        log_low = np.log10(force_min)
        log_high = np.log10(force_max)
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
    
    # Compute displacement field using modal superposition
    displacement_field = compute_chladni_displacement_field(
        target_coords, force_coords, force_magnitudes, modal_params
    )
    
    # Reshape displacement to grid format and add channel dimension
    displacement_grid = displacement_field.reshape(grid_n, grid_n, 1).astype(np.float32)
    
    return {
        "sources": forces.tolist(),  # variable-length list of [x_norm, y_norm, force] triplets
        "field": displacement_grid,  # displacement field (grid_n, grid_n, 1)
    }

def build_dataset(num_samples: int, **kwargs) -> Dataset:
    """Stream-based builder to keep memory usage low."""
    
    # Precompute modal parameters once for all samples
    modal_params = precompute_chladni_modal_params(
        L=kwargs['L'], M=kwargs['M'], omega=kwargs['omega'], t_fixed=kwargs['t_fixed'],
        gamma=kwargs['gamma'], v=kwargs['v'], n_max=kwargs['n_max'], m_max=kwargs['m_max']
    )
    
    def _gen():
        rng = np.random.default_rng(kwargs.pop("seed", None))
        for _ in tqdm(range(num_samples), desc="samples"):
            # Pass modal_params to make_record
            record_kwargs = {k: v for k, v in kwargs.items() if k not in ['omega', 't_fixed', 'gamma', 'v', 'n_max', 'm_max']}
            record_kwargs['modal_params'] = modal_params
            yield make_record(rng=rng, **record_kwargs)
    
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
    parser.add_argument("--n_forces", type=int, default=10, help="Number of forces (constant)")
    parser.add_argument("--constant_force", action="store_true", help="Set all force magnitudes = 1")
    parser.add_argument("--force_min", type=float, default=0.01, help="Min force magnitude")
    parser.add_argument("--force_max", type=float, default=0.05, help="Max force magnitude")
    
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
        n_forces=args.n_forces,
        force_min=args.force_min,
        force_max=args.force_max,
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
    print(f"    - Forces per sample: {args.n_forces} (constant)")
    if args.constant_force:
        print(f"    - Force magnitude: 1.0 (constant)")
    else:
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