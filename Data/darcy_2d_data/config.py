"""
Configuration for 2‑D Darcy data generation.
Edit these constants as you explore new setups.
"""

# Mesh
RESOLUTION = 64          # grid nodes per axis
DOMAIN      = (0.0, 1.0) # unit square

# Dataset size
N_SAMPLES = 1000       # how many (k, p) samples

# Random log‑normal permeability k(x,y)
CORR_LEN = 5.0         # correlation length (fraction of domain) - SMALLER for more variation
LOG_STD  = 1.5          # std‑dev before exp() - LARGER for more contrast

# Boundary conditions
#   'dirichlet' – p=1 on x=0, p=0 on x=1, no‑flux elsewhere
#   'mixed'     – customise later in solve.py
BC_TYPE = "dirichlet"

# Reproducibility
SEED = 42

# Output (HuggingFace datasets format)
DATASET_DIR = "darcy64_hfds"  # arrow shards + dataset script will be saved here
