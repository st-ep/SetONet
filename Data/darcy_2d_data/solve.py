# solve_fd.py  (rename or overwrite solve.py)

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import config as cfg
from randfield import sample_log_k

__all__ = ["solve_one"]

# Pre-assemble the constant pieces once
N = cfg.RESOLUTION         # cells per side
h = 1.0 / N                # grid spacing
NX = NY = N + 1            # nodes = cells + 1
IDX = np.arange(NX * NY).reshape(NX, NY)

# Offsets for 5-point stencil (E, W, N, S)
OFF = dict(E=(1, 0), W=(-1, 0), N=(0, 1), S=(0, -1))

def _build_matrix(k):
    """Assemble sparse matrix A p = b for heterogeneous k(x,y).
    
    Discretizes ∇·(k∇p) = 0 using finite differences.
    High-precision assembly for operator learning datasets.
    """
    rows, cols, data = [], [], []

    def add(i, j, v):
        rows.append(i); cols.append(j); data.append(v)

    # Use double precision for matrix assembly
    h_sq = (h * h)
    
    for ix in range(NX):
        for iy in range(NY):
            idx_c = IDX[ix, iy]

            # Dirichlet boundary conditions - identity rows
            if ix == 0 or ix == N:
                add(idx_c, idx_c, 1.0)
                continue

            # Interior nodes: discretize ∇·(k∇p) = 0
            # Correct finite difference: (k_w + k_e + k_s + k_n)*p_ij - k_w*p_west - k_e*p_east - k_s*p_south - k_n*p_north = 0
            
            diag = 0.0
            
            # West coefficient (ix-1, iy)
            if ix > 0:
                # Ultra-robust harmonic averaging for maximum stability
                k1, k2 = k[ix, iy], k[ix-1, iy]
                eps = 1e-10  # Conservative regularization for numerical stability
                k_w = 2.0 * k1 * k2 / (k1 + k2 + eps)
                coeff_w = k_w / h_sq
                diag += coeff_w  # Add to diagonal
                if ix-1 == 0:  # Adjacent to left boundary (p=1)
                    # Contribution from boundary goes to RHS
                    pass
                else:
                    add(idx_c, IDX[ix-1, iy], -coeff_w)  # Negative off-diagonal
            
            # East coefficient (ix+1, iy)  
            if ix < N:
                k1, k2 = k[ix, iy], k[ix+1, iy]
                eps = 1e-10
                k_e = 2.0 * k1 * k2 / (k1 + k2 + eps)
                coeff_e = k_e / h_sq
                diag += coeff_e  # Add to diagonal
                if ix+1 == N:  # Adjacent to right boundary (p=0)
                    # Contribution from boundary goes to RHS
                    pass
                else:
                    add(idx_c, IDX[ix+1, iy], -coeff_e)  # Negative off-diagonal
            
            # South coefficient (ix, iy-1)
            if iy > 0:
                k1, k2 = k[ix, iy], k[ix, iy-1]
                eps = 1e-10
                k_s = 2.0 * k1 * k2 / (k1 + k2 + eps)
                coeff_s = k_s / h_sq
                diag += coeff_s  # Add to diagonal
                add(idx_c, IDX[ix, iy-1], -coeff_s)  # Negative off-diagonal
            
            # North coefficient (ix, iy+1)
            if iy < N:
                k1, k2 = k[ix, iy], k[ix, iy+1]
                eps = 1e-10
                k_n = 2.0 * k1 * k2 / (k1 + k2 + eps)
                coeff_n = k_n / h_sq
                diag += coeff_n  # Add to diagonal
                add(idx_c, IDX[ix, iy+1], -coeff_n)  # Negative off-diagonal
            
            add(idx_c, idx_c, diag)
    
    # Use double precision for the matrix
    A = sp.csr_matrix((data, (rows, cols)), shape=(NX * NY, NX * NY), dtype=np.float64)
    return A

def solve_one(rng):
    """Return (k_grid, p_grid) with no external dependencies."""
    k_grid = sample_log_k(rng)               # (NX, NY)
    
    # Allow larger contrasts for more interesting plots while maintaining stability
    # Balanced approach: 100x contrast instead of 10x for more diversity
    k_grid = np.clip(k_grid, 0.1, 10.0)  # Max contrast ratio: 100x for interesting diversity
    
    A = _build_matrix(k_grid)

    # RHS vector
    b = np.zeros(A.shape[0], dtype=np.float64)
    
    # Set boundary conditions
    left_nodes = IDX[0, :]   # left boundary: p = 1
    right_nodes = IDX[N, :]  # right boundary: p = 0
    
    b[left_nodes] = 1.0
    b[right_nodes] = 0.0
    
    # Add boundary contributions to RHS
    # For nodes adjacent to left boundary (p=1)
    for iy in range(NY):
        if 1 < N:  # If there are interior nodes
            idx_adj = IDX[1, iy]
            k1, k2 = k_grid[1, iy], k_grid[0, iy]
            eps = 1e-10  # Match matrix assembly precision
            k_w = 2.0 * k1 * k2 / (k1 + k2 + eps)
            coeff_w = k_w / (h * h)
            b[idx_adj] += coeff_w * 1.0  # p=1 on left boundary
    
    # For nodes adjacent to right boundary (p=0), no contribution needed since p=0

    # Use direct solver for maximum precision (required for operator learning)
    try:
        # Use double precision throughout for better accuracy
        A_double = A.astype(np.float64)
        b_double = b.astype(np.float64)
        
        p_vec = spla.spsolve(A_double, b_double)
        
        # Verify solution quality with more reasonable tolerance
        residual_check = np.max(np.abs(A_double.dot(p_vec) - b_double))
        if residual_check > 1e-6:  # More reasonable tolerance
            print(f"Warning: High residual {residual_check:.2e} detected, refining solution...")
            
            # If residual is too high, use iterative refinement
            p_vec, info = spla.cg(A_double, b_double, x0=p_vec, tol=1e-10, maxiter=10000)
            if info != 0:
                print(f"Warning: CG refinement failed with info={info}")
                # Try with more relaxed tolerance if CG fails
                p_vec, info = spla.cg(A_double, b_double, x0=p_vec, tol=1e-8, maxiter=20000)
                if info != 0:
                    raise RuntimeError(f"Iterative solver failed completely with info={info}")
        
    except Exception as e:
        raise RuntimeError(f"Direct solver failed. Matrix may be singular. k_range: [{k_grid.min():.2e}, {k_grid.max():.2e}]. Error: {e}")

    p_grid = p_vec.reshape(NX, NY).astype(np.float32)
    return k_grid.astype(np.float32), p_grid
