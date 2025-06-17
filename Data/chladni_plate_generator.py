"""
Generate 11000 Forcing Samples for Chladni Plate (Python Version)
- Precompute everything that does NOT depend on alpha(n,m).
- Plot the first few samples immediately.
- Split into 10000 Training Samples + 1000 Testing Samples.
- Save to ChladniData.npz, including S(x,y) arrays.
- Optimized using NumPy vectorization for better performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from datasets import Dataset
import tqdm
import time
import os


def generate_chladni_data():
    """Generate Chladni plate simulation data."""
    
    print("Starting Chladni plate data generation...")
    
    # 1) Basic Setup
    L = 8.75 * 0.0254   # Dimensions in meters
    M = 8.75 * 0.0254
    omega = 55 * np.pi / M  # Frequency
    t_fixed = 4             # Time at which to evaluate the solution
    
    gamma = 0.02  # damping_adjustment
    v = 0.5
    
    numPoints = 25
    x = np.linspace(0, L, numPoints)
    y = np.linspace(0, M, numPoints)
    
    n_range = 10
    m_range = 10
    
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
            plt.savefig(f'Data/chladni_sample_{k+1}.png', 
                       facecolor='black', edgecolor='white', dpi=150)
            plt.show()
    
    print("Sample generation complete. Splitting and saving data...")
    
    # 4) Prepare data for SetONet training
    # ------------------------------------
    print("Preparing data for SetONet format...")
    
    # Create coordinate meshgrid
    X_coords, Y_coords = np.meshgrid(x, y, indexing='ij')
    # Flatten coordinates for each sample
    coords_flat = np.stack([X_coords.flatten(), Y_coords.flatten()], axis=1)  # Shape: (numPoints^2, 2)
    n_points = numPoints * numPoints
    
    # Prepare data arrays for SetONet format
    # For SetONet: Input coordinates (X), Input function values (u), Output coordinates (Y), Output function values (s)
    # In our case: X=coordinates, u=S_values, Y=coordinates, s=Z_values
    
    setONet_data = {
        "X": np.zeros((N_total, n_points, 2), dtype=np.float32),  # Input coordinates
        "u": np.zeros((N_total, n_points), dtype=np.float32),     # Input function (S forces)
        "Y": np.zeros((N_total, n_points, 2), dtype=np.float32),  # Output coordinates
        "s": np.zeros((N_total, n_points), dtype=np.float32),     # Output function (Z displacements)
    }
    
    print("Flattening and structuring data...")
    for k in tqdm.tqdm(range(N_total), desc="Processing samples"):
        # Input and output coordinates are the same for our problem
        setONet_data["X"][k] = coords_flat
        setONet_data["Y"][k] = coords_flat
        
        # Flatten the 2D force and displacement fields
        setONet_data["u"][k] = S_full[:, :, k].flatten()  # S forces (input)
        setONet_data["s"][k] = Z_full[:, :, k].flatten()  # Z displacements (output)
    
    # Convert to Hugging Face dataset format (list of lists)
    print("Converting to Hugging Face format...")
    hf_ready = {k: v.tolist() for k, v in setONet_data.items()}
    ds = Dataset.from_dict(hf_ready)
    
    # Create train/test split
    ds = ds.train_test_split(test_size=N_test, shuffle=False)
    
    # Save dataset
    dataset_path = "Data/chladni_dataset"
    ds.save_to_disk(dataset_path)
    
    print("Data saved in SetONet format!")
    print(f"Training samples: {len(ds['train'])}")
    print(f"Testing samples: {len(ds['test'])}")
    print(f"Grid size: {numPoints}x{numPoints} = {n_points} points")
    print(f"Coordinate dimensions: 2D (x, y)")
    print(f"Dataset saved to: {dataset_path}")
    
    # Also save the original arrays and parameters for reference
    np.savez_compressed('Data/ChladniData_original.npz',
                       alpha_full=alpha_full,
                       S_full=S_full,
                       Z_full=Z_full,
                       x=x, y=y,
                       # Parameters for reference
                       L=L, M=M, omega=omega, t_fixed=t_fixed,
                       gamma=gamma, v=v, numPoints=numPoints,
                       n_range=n_range, m_range=m_range)
    
    return ds


def load_chladni_data():
    """Load the generated Chladni data in SetONet format."""
    from datasets import load_from_disk
    try:
        ds = load_from_disk('Data/chladni_dataset')
        return ds
    except:
        # Fallback to original format if SetONet format not available
        data = np.load('Data/ChladniData_original.npz')
        return {key: data[key] for key in data.keys()}


def load_chladni_original():
    """Load the original Chladni data arrays."""
    data = np.load('Data/ChladniData_original.npz')
    return {key: data[key] for key in data.keys()}


def visualize_samples(n_samples=4):
    """Visualize some samples from the generated data."""
    try:
        # Try to load SetONet format data
        ds = load_chladni_data()
        if hasattr(ds, 'keys'):  # Original format
            data = ds
            Z_train = data['Z_full'][:, :, :n_samples]
            x = data['x']
            y = data['y']
            omega = data['omega']
        else:  # SetONet format
            # Load original data for visualization
            data = load_chladni_original()
            Z_train = data['Z_full'][:, :, :n_samples]
            x = data['x']
            y = data['y']
            omega = data['omega']
    except:
        print("No data found. Please generate data first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    plt.style.use('dark_background')
    
    for i in range(min(n_samples, 4)):
        ax = axes[i]
        contours = ax.contour(x, y, Z_train[:, :, i].T, levels=[0], 
                            colors=[(0.85, 0.65, 0.13)], linewidths=2)
        ax.set_xlabel('X axis', color='white')
        ax.set_ylabel('Y axis', color='white')
        ax.set_title(f'Training Sample #{i+1}', color='white')
        ax.set_facecolor('black')
    
    plt.tight_layout()
    plt.savefig('Data/chladni_samples_overview.png', 
               facecolor='black', edgecolor='white', dpi=150)
    plt.show()


def visualize_setONet_sample(sample_idx=0):
    """Visualize a sample in SetONet format showing input forces and output displacements."""
    ds = load_chladni_data()
    if hasattr(ds, 'keys'):  # Original format
        print("Data not in SetONet format. Use visualize_samples() instead.")
        return
    
    # Get original parameters for grid reconstruction
    data_orig = load_chladni_original()
    x = data_orig['x']
    y = data_orig['y']
    numPoints = len(x)
    omega = data_orig['omega']
    
    # Get training sample
    train_sample = ds['train'][sample_idx]
    
    # Reshape flattened data back to 2D grid
    S_2d = np.array(train_sample['u']).reshape(numPoints, numPoints)
    Z_2d = np.array(train_sample['s']).reshape(numPoints, numPoints)
    
    # Plot input forces and output displacements
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use('dark_background')
    
    # Input forces
    ax1 = axes[0]
    im1 = ax1.contourf(x, y, S_2d.T, levels=20, cmap='viridis')
    ax1.set_xlabel('X axis', color='white')
    ax1.set_ylabel('Y axis', color='white')
    ax1.set_title(f'Input Forces S(x,y) - Sample #{sample_idx}', color='white')
    ax1.set_facecolor('black')
    plt.colorbar(im1, ax=ax1)
    
    # Output displacements
    ax2 = axes[1]
    contours = ax2.contour(x, y, Z_2d.T, levels=[0], 
                          colors=[(0.85, 0.65, 0.13)], linewidths=3)
    ax2.set_xlabel('X axis', color='white')
    ax2.set_ylabel('Y axis', color='white')
    ax2.set_title(f'Output Displacements Z(x,y) - Sample #{sample_idx}', color='white')
    ax2.set_facecolor('black')
    
    plt.tight_layout()
    plt.savefig(f'Data/chladni_setONet_sample_{sample_idx}.png', 
               facecolor='black', edgecolor='white', dpi=150)
    plt.show()
    
    print(f"Sample {sample_idx} info:")
    print(f"Input coordinates shape: {np.array(train_sample['X']).shape}")
    print(f"Input forces shape: {np.array(train_sample['u']).shape}")
    print(f"Output coordinates shape: {np.array(train_sample['Y']).shape}")
    print(f"Output displacements shape: {np.array(train_sample['s']).shape}")
    print(f"Forces range: [{np.array(train_sample['u']).min():.4f}, {np.array(train_sample['u']).max():.4f}]")
    print(f"Displacements range: [{np.array(train_sample['s']).min():.4f}, {np.array(train_sample['s']).max():.4f}]")


if __name__ == "__main__":
    start_time = time.time()
    
    # Generate the data
    ds = generate_chladni_data()
    
    # Visualize some samples
    print("Creating sample visualization...")
    visualize_samples()
    
    # Also create SetONet format visualization
    print("Creating SetONet format visualization...")
    visualize_setONet_sample(sample_idx=0)
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Print dataset info
    print("\nDataset Summary:")
    print(f"Training samples: {len(ds['train'])}")
    print(f"Testing samples: {len(ds['test'])}")
    print(f"Input/Output dimensions: 2D coordinates")
    print("Ready for SetONet training!") 