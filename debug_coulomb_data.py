import numpy as np
import torch

# Load the dataset
data = np.load("coulomb_2D_train.npz", allow_pickle=True)

# Get first few samples
sample_0 = data['s0'].item()
sample_1 = data['s1'].item()

print("=== Coulomb Dataset Analysis ===")
print(f"Sample 0 structure:")
for key, value in sample_0.items():
    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    print(f"    range: [{value.min():.6f}, {value.max():.6f}]")
    print(f"    mean: {value.mean():.6f}, std: {value.std():.6f}")
    print()

print("=== Potential Distribution ===")
all_potentials = []
all_charges = []
all_positions = []

for i in range(10):  # Check first 10 samples
    sample = data[f's{i}'].item()
    all_potentials.extend(sample['potentials'])
    all_charges.extend(sample['charges'])
    all_positions.extend(sample['positions'].flatten())

all_potentials = np.array(all_potentials)
all_charges = np.array(all_charges)
all_positions = np.array(all_positions)

print(f"Potential stats across samples:")
print(f"  Range: [{all_potentials.min():.6f}, {all_potentials.max():.6f}]")
print(f"  Mean: {all_potentials.mean():.6f}, Std: {all_potentials.std():.6f}")
print(f"  Median: {np.median(all_potentials):.6f}")
print(f"  95% range: [{np.percentile(all_potentials, 2.5):.6f}, {np.percentile(all_potentials, 97.5):.6f}]")

print(f"\nCharge stats:")
print(f"  Range: [{all_charges.min():.6f}, {all_charges.max():.6f}]")
print(f"  Mean: {all_charges.mean():.6f}, Std: {all_charges.std():.6f}")

print(f"\nPosition stats:")
print(f"  Range: [{all_positions.min():.6f}, {all_positions.max():.6f}]")
print(f"  Mean: {all_positions.mean():.6f}, Std: {all_positions.std():.6f}")

# Check for any extreme values that might cause training issues
print("\n=== Potential Issues ===")
if np.any(np.isnan(all_potentials)):
    print("❌ Found NaN values in potentials!")
if np.any(np.isinf(all_potentials)):
    print("❌ Found infinite values in potentials!")
if np.abs(all_potentials).max() > 1000:
    print(f"⚠️  Very large potential values found: max = {np.abs(all_potentials).max():.2f}")
if np.abs(all_potentials).min() < 1e-6:
    print(f"⚠️  Very small potential values found: min = {np.abs(all_potentials).min():.2e}")

# Test a simple forward pass
print("\n=== Simple Model Test ===")
try:
    positions = torch.tensor(sample_0['positions'], dtype=torch.float32).unsqueeze(0)  # [1, 100, 2]
    charges = torch.tensor(sample_0['charges'], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 100, 1]
    queries = torch.tensor(sample_0['queries'], dtype=torch.float32).unsqueeze(0)  # [1, 256, 2]
    
    print(f"Input shapes:")
    print(f"  positions (xs): {positions.shape}")
    print(f"  charges (us): {charges.shape}")
    print(f"  queries (ys): {queries.shape}")
    
    print(f"Input ranges:")
    print(f"  positions: [{positions.min():.6f}, {positions.max():.6f}]")
    print(f"  charges: [{charges.min():.6f}, {charges.max():.6f}]")
    print(f"  queries: [{queries.min():.6f}, {queries.max():.6f}]")
    
except Exception as e:
    print(f"❌ Error in tensor conversion: {e}") 