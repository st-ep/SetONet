# SetONet: A Set-Based Operator Network for PDEs with Variable-Input Sampling

SetONet is a DeepONet-class neural operator that learns mappings between function spaces while removing key limitations of standard DeepONet. Beyond the need for fixed sensor locations, standard DeepONet cannot natively accept unordered, variable-length, unstructured inputs such as point-cloud measurements. SetONet modifies DeepONet’s branch network to process inputs as an unordered set of location–value pairs and incorporates Deep Sets principles to guarantee permutation invariance. It preserves the same trunk network and overall synthesis as DeepONet and, importantly, maintains the same number of trainable parameters—acting as a drop-in replacement without increasing model complexity.

This repository provides SetONet and baseline DeepONet implementations, dataset generators/loaders, benchmark run scripts, evaluation and plotting utilities, and logging tooling. For more details, see the paper: https://arxiv.org/abs/2505.04738


## Highlights
- Deep Sets branch with attention pooling for expressive, permutation-invariant aggregation
- Standard DeepONet-style trunk
- Sinusoidal positional encoding for spatial awareness at multiple scales
- Lightweight, parameter-matched design vs. DeepONet baseline
- Benchmarks: 1D calculus operators, 1D Darcy flow, 2D elastic plate, 2D heat with point sources, 2D advection–diffusion (chemical plume), 2D optimal transport


## Table of Contents
- Quickstart
- Datasets
  - Heat 2D (adaptive/uniform)
  - Elastic Plate 2D
  - Optimal Transport 2D
  - Concentration 2D (advection–diffusion)
  - Darcy 1D
  - Chladni 2D
  - Dynamic Chladni
  - 1D Calculus Operators
- Training & Evaluation
- Plotting
- Configuration Reference
- Repository Layout
- Contributing
- License & Citation


## Quickstart

Requirements: Python 3.9+ and a CUDA-enabled GPU (CPU works but is slow).

1) Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Verify PyTorch can see your GPU (optional)

```
python - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

4) Run a minimal SetONet training (Heat 2D) — see Datasets below to generate/load data first

```
python Benchmarks/run_SetONet/run_heat_2d.py \
  --device cuda:0 \
  --data_path Data/heat_data/pcb_heat_adaptive_dataset9.0_n8192_N25_P10
```

TensorBoard logs and figures are written under `logs/`.


## Datasets

SetONet uses HuggingFace `datasets`-style on-disk datasets. Each benchmark has a generator and loader.

| Benchmark | Generator | Loader | Run Script |
|-----------|-----------|--------|------------|
| Heat 2D (point sources) | `Data/heat_data/generate_heat_2d_data.py` | `Data/heat_data/heat_2d_dataset.py` | `Benchmarks/run_SetONet/run_heat_2d.py` |
| Elastic Plate 2D | `Data/elastic_2d_data/get_elastic_data.py` | `Data/elastic_2d_data/elastic_2d_dataset.py` | `Benchmarks/run_SetONet/run_elastic_2d.py` |
| Optimal Transport 2D | `Data/transport_data/generate_transport_data.py` | `Data/transport_data/transport_dataset.py` | `Benchmarks/run_SetONet/run_transoprt.py` |
| Concentration 2D (advection–diffusion) | `Data/concentration_data/generate_concentration_2d_data.py` | `Data/concentration_data/concentration_2d_dataset.py` | `Benchmarks/run_SetONet/run_consantration_2d.py` |
| Darcy 1D | `Data/darcy_1d_data/darcy_1d_data.py` | `Data/darcy_1d_data/darcy_1d_dataset.py` | `Benchmarks/run_SetONet/run_darcy_1d.py` |
| Chladni 2D | `Data/chladni_data/chladni_plate_generator.py` | `Data/chladni_data/chladni_2d_dataset.py` | `Benchmarks/run_SetONet/run_chladni_2d.py` |
| Dynamic Chladni | `Data/dynamic_chladni/dynamic_chladni_generator.py` | `Data/dynamic_chladni/dynamic_chladni_dataset.py` | `Benchmarks/run_SetONet/run_dynamic_chladni.py` |
| 1D Calculus Operators | `Data/synthetic_1d_data.py` | (inline sampling) | `Benchmarks/run_SetONet/run_1d.py` |

Quick tips:
- Heat 2D datasets can be uniform-grid or adaptive (auto-detected).
- Elastic 2D loader preloads pre-normalized tensors to GPU; `--train_sensor_dropoff` and `--replace_with_nearest` improve robustness.
- Optimal transport loader supports `velocity_field`, `transport_map`, and `density_transport` modes.
- The concentration (chemical plume) generator mirrors the heat setup but uses advection–diffusion physics.
- Plotting utilities for each benchmark reside in `Plotting/` (e.g., `plot_heat_2d_utils.py`, `plot_consentration_2d_utils.py`).


## Training & Evaluation

Each SetONet run script follows the same pattern:

```
python Benchmarks/run_SetONet/<script>.py [--flags]
```

Examples:

- Heat 2D:
```
python Benchmarks/run_SetONet/run_heat_2d.py \
  --device cuda:0 \
  --data_path Data/heat_data/pcb_heat_adaptive_dataset9.0_n8192_N25_P10 \
  --son_p_dim 128 --son_phi_hidden 256 --son_rho_hidden 256 \
  --son_trunk_hidden 256 --son_n_trunk_layers 4 \
  --son_phi_output_size 32 --son_aggregation attention \
  --pos_encoding_type sinusoidal --pos_encoding_dim 64 --pos_encoding_max_freq 0.01 \
  --son_lr 5e-4 --son_epochs 50000 --batch_size 32
```

- Elastic 2D:
```
python Benchmarks/run_SetONet/run_elastic_2d.py \
  --device cuda:0 \
  --son_p_dim 128 --son_phi_hidden 256 --son_rho_hidden 256 \
  --son_trunk_hidden 256 --son_n_trunk_layers 4 \
  --son_phi_output_size 32 --son_aggregation attention \
  --pos_encoding_type sinusoidal --pos_encoding_dim 64 --pos_encoding_max_freq 0.1 \
  --son_lr 5e-4 --son_epochs 125000 --batch_size 64 \
  --train_sensor_dropoff 0.0 --eval_sensor_dropoff 0.0
```

- Optimal Transport (velocity field):
```
python Benchmarks/run_SetONet/run_transoprt.py \
  --device cuda:0 \
  --data_path Data/transport_data/transport_dataset \
  --mode velocity_field \
  --son_p_dim 128 --son_phi_hidden 256 --son_rho_hidden 256 \
  --son_trunk_hidden 256 --son_n_trunk_layers 4 \
  --son_phi_output_size 32 --son_aggregation attention \
  --pos_encoding_type sinusoidal --pos_encoding_dim 64 --pos_encoding_max_freq 0.01 \
  --son_lr 5e-4 --son_epochs 50000 --batch_size 32
```

DeepONet baselines:

```
python Benchmarks/run_DeepONet/run_elastic_2d_don.py --device cuda:0
python Benchmarks/run_DeepONet/run_1d_don.py --device cuda:0
```

Evaluation is performed automatically at the end of training; periodic eval is logged via TensorBoard if enabled. Results (MSE, relative L2) and plots are stored in `logs/<model>/<timestamp>/`.


## Plotting

Each run script can save sample figures. You can also directly call plotting utilities:

- Heat 2D: `Plotting/plot_heat_2d_utils.py`
- Elastic 2D: `Plotting/plot_elastic_2d_utils.py`
- Darcy 1D: `Plotting/plot_darcy_1d_utils.py`
- Transport: `Plotting/plot_transport_utils.py`

TensorBoard:

```
tensorboard --logdir logs
```


## Configuration Reference

Key SetONet hyperparameters (also CLI flags):

- Latent dimension `p` (`--son_p_dim`): 32 (1D) or 128 (2D) typically
- φ MLP: `--son_phi_hidden` (default 256), `--son_phi_output_size` (default 32)
- Aggregation: `--son_aggregation` = `attention` (default) or `mean`
  - Attention pooling uses multihead attention over set elements with learnable query token(s)
- ρ MLP: `--son_rho_hidden` (default 256)
- Trunk MLP: `--son_trunk_hidden` (default 256), `--son_n_trunk_layers`
- Positional encoding: `--pos_encoding_type` (`sinusoidal`|`skip`), `--pos_encoding_dim` (64), `--pos_encoding_max_freq` (e.g., 0.01–0.1)
- Training: `--son_lr` (5e-4), `--son_epochs`, `--batch_size`
- LR schedule: `--lr_schedule_steps`, `--lr_schedule_gammas` (milestone-style; see `Models/SetONet.py`)
- Device: `--device cuda:0` (or `cpu`)

Elastic-only robustness flags (also work for some 1D/2D problems with 2D sensors):

- `--train_sensor_dropoff`, `--eval_sensor_dropoff` (0.0–1.0)
- `--replace_with_nearest` (duplicate nearest kept sensors to maintain size)


## Repository Layout

- Benchmarks (run scripts)
  - SetONet: `Benchmarks/run_SetONet/*.py` (heat, elastic, darcy, chladni, dynamic chladni, concentration, transport, 1D operators)
  - DeepONet: `Benchmarks/run_DeepONet/*.py`
- Models
  - `Models/SetONet.py` — core SetONet model (Deep Sets branch, trunk, training loop)
  - `Models/DeepONet.py` — baseline DeepONet wrapper
  - `Models/utils/attention_pool.py` — attention pooling (PMA) aggregator
  - `Models/utils/tensorboard_callback.py` — periodic eval + TB logging
  - `Models/utils/config_utils.py` — experiment metadata snapshot to JSON
  - `Models/utils/helper_utils.py` — metrics and input prep helpers
- Data
  - Heat 2D: `Data/heat_data/`
  - Concentration (advection–diffusion) 2D: `Data/concentration_data/`
  - Elastic 2D: `Data/elastic_2d_data/`
  - Darcy 1D: `Data/darcy_1d_data/`
  - Chladni 2D: `Data/chladni_data/`
  - Dynamic Chladni: `Data/dynamic_chladni/`
  - Synthetic 1D operators: `Data/synthetic_1d_data.py`
  - Transport 2D: `Data/transport_data/`
  - Utilities: `Data/data_utils.py`
- Plotting utilities: `Plotting/*.py` (heat, elastic, darcy, chladni, dynamic chladni, concentration, transport, etc.)
- Logs: `logs/<model>/<timestamp>/`


## Contributing

- Keep changes minimal and focused. Follow patterns in existing benchmarks and loaders.
- To add a new dataset: mirror an existing generator/loader pair and expose `.sample()` that returns `(xs, us, ys, G_u_ys, aux)` on GPU.
- To add a benchmark: add a run script under `Benchmarks/run_SetONet/` (and optionally a DeepONet baseline) and a plotting utility under `Plotting/`.

Issues/PRs that include: repro steps, commands, dataset parameters, and environment details help a lot.


## License & Citation

- License: see `LICENSE` (MIT)

If you find this repository useful in your research, please cite our paper:

```bibtex
@article{tretiakov2025setonet,
  title         = {SetONet: A Set-Based Operator Network for Solving PDEs with Variable-Input Sampling},
  author        = {Tretiakov, Stepan and Li, Xingjian and Kumar, Krishna},
  year          = {2025},
  eprint        = {2505.04738},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2505.04738},
  url           = {https://arxiv.org/abs/2505.04738}
}
```
