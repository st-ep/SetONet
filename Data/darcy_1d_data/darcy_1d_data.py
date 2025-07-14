import numpy as np
from scipy.optimize import fsolve
from sklearn.metrics.pairwise import rbf_kernel
from functools import partial
from datasets import Dataset
import tqdm
import os
import multiprocessing


class Darcy1DSolver:
    def __init__(self, nx=100):
        self.nx = nx
        self.x = np.linspace(0, 1, nx + 1)  # Grid points (including boundaries)
        self.dx = 1.0 / nx
        
    def permeability(self, s):
        """Nonlinear permeability function k(s) = 0.2 + s^2"""
        return 0.2 + s**2
    
    def sample_gp_prior(self, kernel, X, n_samples=1):
        """Sample from Gaussian Process prior"""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        K = kernel(X, X)
        prior = np.random.multivariate_normal(
            mean=np.zeros(X.shape[0]),
            cov=K,
            size=n_samples,
        )
        return prior
    
    def generate_source_term(self, n_samples=1):
        """Generate random source term from GP prior"""
        sigma = 0.08
        gamma = 1 / (2 * sigma**2)
        
        random_function = self.sample_gp_prior(
            kernel=partial(rbf_kernel, gamma=gamma),
            X=self.x,
            n_samples=n_samples,
        )
        return random_function
    
    def residual(self, s, u):
        """
        Compute residual for nonlinear Darcy equation:
        -d/dx(k(s) * ds/dx) = u
        with boundary conditions s[0] = s[-1] = 0
        """
        res = np.zeros_like(s)
        
        interior = slice(1, -1)
        s_left = s[:-2]
        s_mid = s[1:-1]
        s_right = s[2:]
        
        k_left = self.permeability(0.5 * (s_left + s_mid))
        k_right = self.permeability(0.5 * (s_mid + s_right))
        
        flux_left = k_left * (s_mid - s_left) / self.dx
        flux_right = k_right * (s_right - s_mid) / self.dx
        
        res[interior] = -(flux_right - flux_left) / self.dx - u[interior]
        
        res[0] = s[0]
        res[-1] = s[-1]
        
        return res
    
    def solve_darcy(self, u):
        """Solve the nonlinear Darcy equation using Newton's method"""
        # Initial guess
        s0 = np.zeros_like(u)
        
        # Solve nonlinear system
        solution = fsolve(lambda s: self.residual(s, u), s0, xtol=1e-8)
        
        return solution
    
    def generate_sample(self):
        """Generate one sample (source term + solution)"""
        u = self.generate_source_term()
        s = self.solve_darcy(u)
        
        return {
            "X": self.x,
            "u": u,
            "Y": self.x,
            "s": s,
        }
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.generate_sample()


if __name__ == "__main__":
    train_size = 10000
    test_size = 1000
    ds_size = train_size + test_size
    np.random.seed(42)
    
    # Define save directory - updated to match new structure
    save_dir = "/home/titanv/Stepan/setprojects/SetONet/Data/darcy_1d_data/"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    
    solver = Darcy1DSolver(nx=500)  # Changed from nx=100 to nx=300 for 301 grid points
    
    nx = len(solver.x)
    
    print("Generating dataset...")
    
    U = solver.generate_source_term(ds_size)
    
    def solve_single(u):
        return fsolve(lambda s: solver.residual(s, u), np.zeros_like(u), xtol=1e-8)
    
    with multiprocessing.Pool() as p:
        S_list = list(tqdm.tqdm(p.imap(solve_single, U), total=ds_size))
    
    S = np.array(S_list)
    
    data = {
        "X": np.tile(solver.x, (ds_size, 1)).astype(np.float32),
        "u": U.astype(np.float32),
        "Y": np.tile(solver.x, (ds_size, 1)).astype(np.float32),
        "s": S.astype(np.float32),
    }
    
    # Hugging-Face expects (list-of-list) rather than raw ndarrays
    hf_ready = {k: v.tolist() for k, v in data.items()}
    ds = Dataset.from_dict(hf_ready)
    ds = ds.train_test_split(test_size=test_size, shuffle=False)
    
    # Save dataset with updated name to distinguish from 101-point version
    dataset_path = os.path.join(save_dir, "darcy_1d_dataset_501")
    ds.save_to_disk(dataset_path)
    
    print(f"Dataset created successfully!")
    print(f"Train size: {len(ds['train'])}")
    print(f"Test size: {len(ds['test'])}")
    print(f"Grid points: {nx}")
    print(f"Saved to: {dataset_path}")
    
    # Quick visualization of a sample
    import matplotlib.pyplot as plt
    
    sample_idx = 0
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(data["X"][sample_idx], data["u"][sample_idx], 'b-', label='Source term u(x)')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Source Term')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(data["Y"][sample_idx], data["s"][sample_idx], 'r-', label='Solution s(x)')
    plt.xlabel('x')
    plt.ylabel('s(x)')
    plt.title('Darcy Solution')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'darcy_sample_501.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Sample plot saved to: {plot_path}") 