"""
Generate 11000 Forcing Samples for Chladni Plate (Python Version)
- Precompute everything that does NOT depend on alpha(n,m).
- Plot the first few samples immediately.
- Split into 10000 Training Samples + 1000 Testing Samples.
- Save to ChladniData.npz, including S(x,y) arrays.
- Optimized using NumPy vectorization for better performance.
- Includes full preprocessing and normalization for SetONet training.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from datasets import Dataset, load_from_disk
import tqdm
import time
import os
import json

# Get paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

def normalize_data(data, axis=None):
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: numpy array to normalize
        axis: axis along which to compute statistics (None for global stats)
    
    Returns:
        normalized_data: normalized array
        mean: mean values used for normalization
        std: standard deviation values used for normalization
    """
    if axis is None:
        mean = np.mean(data)
        std = np.std(data)
    else:
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std < 1e-8, 1.0, std)
    normalized_data = (data - mean) / std
    
    return normalized_data, mean.squeeze() if hasattr(mean, 'squeeze') else mean, std.squeeze() if hasattr(std, 'squeeze') else std

def generate_chladni_data():
    """Generate Chladni plate simulation data with full preprocessing."""
    
    print("Starting Chladni plate data generation...")
    
    # 1) Basic Setup
    L = 8.75 * 0.0254   # Dimensions in meters
    M = 8.75 * 0.0254
    omega = 50 * np.pi / M  # Frequency
    t_fixed = 6             # Time at which to evaluate the solution
    
    gamma = 0.02  # damping_adjustment
    v = 0.5
    
    numPoints = 32
    x = np.linspace(0, L, numPoints)
    y = np.linspace(0, M, numPoints)
    
    n_range = 6
    m_range = 6
    
    N_total = 11000   # total number of samples
    N_train = 10000   # number of training samples
    N_test = 1000     # number of testing samples
    
    # Initialize storage arrays
    alpha_full = np.zeros((n_range, m_range, N_total))
    S_full = np.zeros((numPoints, numPoints, N_total))
    Z_full = np.zeros((numPoints, numPoints, N_total))
    
    print("Setup complete. Starting precomputation...")
    
    # 2) Precompute Terms That Don't Depend on alpha
    # ------------------------------------------------
    
    # (A) Wave numbers mu(n), lambda(m)
    mu_vals = np.arange(1, n_range + 1) * np.pi / L  # shape: (n_range,)
    lam_vals = np.arange(1, m_range + 1) * np.pi / M  # shape: (m_range,)
    
    # (B) cosX(n,i) = cos(mu_vals(n) * x(i))
    # Using broadcasting: mu_vals[:, None] * x[None, :] creates (n_range, numPoints)
    cosX = np.cos(mu_vals[:, None] * x[None, :])  # shape: (n_range, numPoints)
    
    # (C) cosY(m,j) = cos(lam_vals(m) * y(j))
    cosY = np.cos(lam_vals[:, None] * y[None, :])  # shape: (m_range, numPoints)
    
    # (D) centerFactor(n,m) = cos(mu_n*(L/2)) * cos(lam_m*(M/2))
    centerFactor = np.cos(mu_vals[:, None] * (L/2)) * np.cos(lam_vals[None, :] * (M/2))
    # shape: (n_range, m_range)
    
    # (E) beta(n,m) = sqrt(mu^2 + lam^2 + 3*v^2 - gamma^4)
    mu_squared = mu_vals[:, None]**2  # shape: (n_range, 1)
    lam_squared = lam_vals[None, :]**2  # shape: (1, m_range)
    beta_nm = np.sqrt(mu_squared + lam_squared + 3*v**2 - gamma**4)
    # shape: (n_range, m_range)
    
    print("Computing time integrals...")
    
    # (F) timeInt(n,m) = integral of integrand from 0 to t_fixed
    timeInt = np.zeros((n_range, m_range))
    
    for n in range(n_range):
        for m in range(m_range):
            current_beta = beta_nm[n, m]
            
            def integrand(tau):
                return (np.sin(omega * (tau - t_fixed)) * 
                       np.exp(-gamma**2 + v**2 * tau) * 
                       np.sin(current_beta * tau))
            
            timeInt[n, m], _ = quad(integrand, 0, t_fixed)
    
    # (G) modeFactor(n,m)
    modeFactor = (v**2 / beta_nm) * timeInt * (4/(L*M)) * centerFactor
    
    print("Precomputation complete. Generating samples...")
    
    # 3) Main Loop: Generate Data
    # Using vectorized operations for significant speedup
    # ------------------------------------------------------------------------
    
    # Create meshgrids for vectorized computation
    X, Y = np.meshgrid(x, y, indexing='ij')  # shape: (numPoints, numPoints)
    
    for k in range(N_total):
        if k % 1000 == 0:
            print(f"Generated {k}/{N_total} samples")
        
        # (A) Generate random alpha-coefficients
        alpha_k = 0.01 * np.random.randn(n_range, m_range)
        alpha_full[:, :, k] = alpha_k
        
        # (B) Compute S(i,j) vectorized
        # S_k = sum_{n,m} alpha_k(n,m) * cosX(n,i) * cosY(m,j)
        # Using einsum for efficient tensor contraction
        S_k = np.einsum('nm,ni,mj->ij', alpha_k, cosX, cosY)
        S_full[:, :, k] = S_k
        
        # (C) Compute Z(i,j) vectorized  
        # Z_k = sum_{n,m} alpha_k(n,m) * cosX(n,i) * cosY(m,j) * modeFactor(n,m)
        alpha_weighted = alpha_k * modeFactor
        Z_k = np.einsum('nm,ni,mj->ij', alpha_weighted, cosX, cosY)
        Z_full[:, :, k] = Z_k
        
        # (D) Plot the first few samples
        if k < 4:
            plt.figure(figsize=(8, 6))
            plt.style.use('dark_background')
            
            contours = plt.contour(x, y, Z_k.T, levels=[0], 
                                 colors=[(0.85, 0.65, 0.13)], linewidths=3)
            plt.xlabel('X axis', color='white')
            plt.ylabel('Y axis', color='white')
            plt.title(f'Sample #{k+1}, ω = {omega:.2f} Hz', color='white')
            plt.gca().set_facecolor('black')
            
            # Save the plot
            plot_save_path = os.path.join(current_dir, f'chladni_sample_{k+1}.png')
            plt.savefig(plot_save_path, facecolor='black', edgecolor='white', dpi=150)
            plt.show()
    
    print("Sample generation complete. Starting data preprocessing...")
    
    # 4) Data Preprocessing and Normalization
    # ---------------------------------------
    print("Preparing data for SetONet format...")
    
    # Create coordinate meshgrid
    X_coords, Y_coords = np.meshgrid(x, y, indexing='ij')
    coords_flat = np.stack([X_coords.flatten(), Y_coords.flatten()], axis=1)
    n_points = numPoints * numPoints
    
    # Prepare raw data arrays
    raw_data = {
        "X": np.zeros((N_total, n_points, 2), dtype=np.float32),
        "u": np.zeros((N_total, n_points), dtype=np.float32),
        "Y": np.zeros((N_total, n_points, 2), dtype=np.float32),
        "s": np.zeros((N_total, n_points), dtype=np.float32),
    }
    
    print("Structuring data...")
    for k in tqdm.tqdm(range(N_total), desc="Processing samples"):
        raw_data["X"][k] = coords_flat
        raw_data["Y"][k] = coords_flat
        raw_data["u"][k] = S_full[:, :, k].flatten()
        raw_data["s"][k] = Z_full[:, :, k].flatten()
    
    # 5) Apply Normalization
    # ----------------------
    print("Computing normalization statistics...")
    
    # Calculate normalization statistics
    u_normalized, u_mean, u_std = normalize_data(raw_data["u"])
    s_normalized, s_mean, s_std = normalize_data(raw_data["s"])
    
    # Coordinate normalization
    X_reshaped = raw_data["X"].reshape(-1, 2)
    xy_mean = np.mean(X_reshaped, axis=0)
    xy_std = np.std(X_reshaped, axis=0)
    xy_std = np.where(xy_std < 1e-8, 1.0, xy_std)
    
    X_normalized = (raw_data["X"] - xy_mean) / xy_std
    Y_normalized = (raw_data["Y"] - xy_mean) / xy_std
    
    # Create dataset with both normalized and original data
    full_dataset = {
        "X": X_normalized.tolist(),
        "u": u_normalized.tolist(),
        "Y": Y_normalized.tolist(),
        "s": s_normalized.tolist(),
        "X_orig": raw_data["X"].tolist(),
        "u_orig": raw_data["u"].tolist(),
        "Y_orig": raw_data["Y"].tolist(),
        "s_orig": raw_data["s"].tolist(),
    }
    
    # Save normalization statistics
    normalization_stats = {
        "u_mean": float(u_mean),
        "u_std": float(u_std),
        "s_mean": float(s_mean),
        "s_std": float(s_std),
        "xy_mean": xy_mean.tolist(),
        "xy_std": xy_std.tolist(),
    }
    
    # Create and save dataset
    ds = Dataset.from_dict(full_dataset)
    ds = ds.train_test_split(test_size=N_test, shuffle=False)
    
    dataset_path = os.path.join(current_dir, "chladni_dataset")
    ds.save_to_disk(dataset_path)
    
    normalization_stats_path = os.path.join(current_dir, 'chladni_normalization_stats.json')
    with open(normalization_stats_path, 'w') as f:
        json.dump(normalization_stats, f, indent=2)
    
    print(f"Dataset saved: {len(ds['train'])} train, {len(ds['test'])} test samples")
    print(f"Normalization stats saved to: {normalization_stats_path}")
    
    # Save original arrays for reference
    original_data_path = os.path.join(current_dir, 'ChladniData_original.npz')
    np.savez_compressed(original_data_path,
                       alpha_full=alpha_full, S_full=S_full, Z_full=Z_full,
                       x=x, y=y, L=L, M=M, omega=omega, t_fixed=t_fixed,
                       gamma=gamma, v=v, numPoints=numPoints,
                       n_range=n_range, m_range=m_range, **normalization_stats)
    
    return ds

def load_chladni_data():
    """Load the generated Chladni data in SetONet format."""
    try:
        dataset_path = os.path.join(current_dir, 'chladni_dataset')
        ds = load_from_disk(dataset_path)
        return ds
    except:
        # Fallback to original format if SetONet format not available
        original_data_path = os.path.join(current_dir, 'ChladniData_original.npz')
        data = np.load(original_data_path)
        return {key: data[key] for key in data.keys()}

def load_chladni_normalization_stats():
    """Load the normalization statistics for the Chladni dataset."""
    normalization_stats_path = os.path.join(current_dir, 'chladni_normalization_stats.json')
    with open(normalization_stats_path, 'r') as f:
        return json.load(f)

def load_chladni_original():
    """Load the original Chladni data arrays."""
    original_data_path = os.path.join(current_dir, 'ChladniData_original.npz')
    data = np.load(original_data_path)
    return {key: data[key] for key in data.keys()}

if __name__ == "__main__":
    start_time = time.time()
    
    # Generate the data
    print("Starting Chladni plate data generation...")
    ds = generate_chladni_data()
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.1f} seconds")
    print("Ready for SetONet training!") 