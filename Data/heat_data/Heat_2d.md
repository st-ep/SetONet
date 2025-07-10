# 2D Heat Transfer Dataset

## Overview

This dataset contains synthetic steady-state temperature fields for 2D heat transfer problems, specifically designed for training neural operator models. The dataset simulates heat conduction on a PCB (Printed Circuit Board) with multiple point heat sources using an analytical Green's function solution.

## Physical Model

### Mathematical Foundation

The dataset is based on the 2D steady-state heat equation with free boundaries:

```
∇²T(x,y) = -∑ᵢ Qᵢ δ(x-xᵢ, y-yᵢ)
```

Where:
- `T(x,y)` is the temperature field
- `Qᵢ` is the power of the i-th heat source
- `(xᵢ, yᵢ)` are the coordinates of the i-th heat source
- `δ` is the Dirac delta function

### Analytical Solution

The temperature field is computed using the Green's function solution:

```
T(x,y) = ∑ᵢ (Qᵢ / 2πk) · log(‖(x,y) - (xᵢ,yᵢ)‖ + ε)
```

Where:
- `k = 1` (thermal conductivity, normalized)
- `ε` is a softening parameter to avoid singularities (default: 0.1)

## Dataset Generation

### Source Configuration

- **Number of sources**: Variable per sample, ranging from `n_min` to `n_max` (default: 10-10)
- **Source locations**: Randomly distributed in the unit square `[0,1]²`
- **Source powers**: Either constant (`Qᵢ = 1`) or log-uniformly distributed in `[power_low, power_high]` (default: [0.1, 1.0])

### Grid Types

The dataset supports two types of spatial discretization:

#### 1. Uniform Grid (Default)
- Regular grid with `grid_n × grid_n` points (default: 5×5)
- Evenly spaced points in `[0,1]²`
- Same grid structure for all samples

#### 2. Adaptive Grid (Optional)
- Intelligently places more grid points near temperature extrema
- Total number of points: `n_adaptive_points` (default: 8192)
- Combines initial uniform grid (`initial_grid_size × initial_grid_size`) with adaptively sampled points
- Uses temperature magnitude-based probability sampling with exponential weighting
- Focus strength controlled by `spike_focus` parameter (default: 9.0)

## Dataset Structure

### Data Format

The dataset is stored using HuggingFace's `datasets` library with train/test splits.

#### Uniform Grid Format
Each sample contains:
- `sources`: List of `[x, y, power]` triplets for each heat source
- `field`: 3D array of shape `(grid_n, grid_n, 1)` containing temperature values

#### Adaptive Grid Format
Each sample contains:
- `sources`: List of `[x, y, power]` triplets for each heat source
- `grid_coords`: List of `[x, y]` coordinates for adaptive grid points
- `field_values`: List of temperature values at corresponding grid points

### File Organization

- **Uniform grid datasets**: Saved as `Data/heat_data/pcb_heat_dataset`
- **Adaptive grid datasets**: Saved as `Data/heat_data/pcb_heat_adaptive_dataset{spike_focus}_n{n_adaptive_points}_N{initial_grid_size}_P{n_min}`

## Usage

### Dataset Generation

```bash
# Generate uniform grid dataset (default)
python Data/heat_data/generate_heat_2d_data.py --train 10000 --test 1000 --grid 64

# Generate adaptive grid dataset
python Data/heat_data/generate_heat_2d_data.py \
    --adaptive_mesh \
    --spike_focus 9.0 \
    --n_adaptive_points 8192 \
    --initial_grid_size 25 \
    --train 10000 \
    --test 1000
```

### Loading and Using the Dataset

```python
from Data.heat_data.heat_2d_dataset import load_heat_dataset

# Load dataset
dataset, heat_dataset = load_heat_dataset(
    data_path="Data/heat_data/pcb_heat_dataset",
    batch_size=64,
    device='cuda'
)

# Sample a batch for training
xs, us, ys, G_u_ys, _ = heat_dataset.sample()
# xs: source coordinates (batch_size, max_sources, 2)
# us: source powers (batch_size, max_sources, 1)
# ys: target coordinates (batch_size, n_points, 2)
# G_u_ys: temperature values (batch_size, n_points, 1)
```

## Configuration Parameters

### Basic Parameters
- `--train`: Number of training samples (default: 10,000)
- `--test`: Number of test samples (default: 1,000)
- `--grid`: Grid resolution for uniform grid (default: 5)
- `--seed`: Random seed for reproducibility (default: 0)

### Source Parameters
- `--n_min`: Minimum number of heat sources per sample (default: 10)
- `--n_max`: Maximum number of heat sources per sample (default: 10)
- `--constant_power`: Use constant power `Qᵢ = 1` for all sources
- `--power_low`: Lower bound for log-uniform power distribution (default: 0.1)
- `--power_high`: Upper bound for log-uniform power distribution (default: 1.0)

### Physical Parameters
- `--eps`: Softening radius to avoid singularities (default: 0.1)

### Adaptive Grid Parameters
- `--adaptive_mesh`: Enable adaptive grid generation
- `--spike_focus`: Focus strength on temperature spikes (default: 9.0)
- `--n_adaptive_points`: Total number of adaptive grid points (default: 8192)
- `--initial_grid_size`: Size of initial uniform grid (default: 25)

## Adaptive Grid Algorithm

The adaptive grid generation follows these steps:

1. **Initial Grid**: Start with a uniform `initial_grid_size × initial_grid_size` grid
2. **Coarse Reference**: Compute temperature on a 128×128 reference grid
3. **Probability Sampling**: Create sampling probabilities based on temperature magnitude using exponential weighting: `exp(spike_focus × normalized_magnitude)`
4. **Adaptive Sampling**: Sample remaining points according to these probabilities
5. **Perturbation**: Add small random perturbations for diversity
6. **Field Computation**: Compute exact temperature values at all grid points using vectorized operations

## Applications

This dataset is designed for:

- **Neural Operator Learning**: Training models like DeepONet, FNO, or Transformer-based operators
- **Physics-Informed Neural Networks**: Benchmarking PINN performance on heat transfer problems
- **Mesh-Free Methods**: Evaluating methods that work with irregular point distributions
- **Multi-Scale Learning**: Testing models on problems with sharp temperature gradients

## Key Features

- **Analytical Ground Truth**: Exact solutions without numerical discretization errors
- **Variable Complexity**: Adjustable number of heat sources per sample
- **Adaptive Resolution**: Optional intelligent grid placement for better spike resolution
- **Efficient Implementation**: Vectorized computations for fast dataset generation
- **Flexible Format**: Support for both regular and irregular grid structures
- **Reproducible**: Deterministic generation with configurable random seeds

## Performance Characteristics

- **Generation Speed**: ~1000 samples/second on modern hardware
- **Memory Efficiency**: Stream-based generation to handle large datasets
- **Numerical Stability**: Softening parameter prevents singularities near sources
- **Batch Processing**: Efficient PyTorch data loading with padding for variable-length sequences

## Visualization

The dataset includes a comprehensive visualization script `plot_heat_2d_data.py` for exploring and analyzing the generated heat transfer data.

### Usage

```bash
# Visualize the default dataset
python Data/heat_data/plot_heat_2d_data.py
```

The script will automatically:
1. Load the dataset from `Data/heat_data/pcb_heat_dataset`
2. Generate 6 individual sample plots (3 from train, 3 from test)
3. Save all plots to `Data/heat_data/plots/` directory

### Generated Plots

- **Individual Sample Plots**: Show temperature field as colored contours with black contour lines, heat sources as colored scatter points (by power), and dual colorbars for temperature and source power
- **Output Files**: 
  - `train_sample_1.png`, `train_sample_2.png`, `train_sample_3.png`
  - `test_sample_1.png`, `test_sample_2.png`, `test_sample_3.png`
