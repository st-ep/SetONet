#!/usr/bin/env python
"""paper_plot_utils.py - Model and dataset loading utilities for paper figures."""
from __future__ import annotations

import json
from pathlib import Path

import torch

# Project root for imports
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent

from Models.SetONet import SetONet
from Models.DeepONet import DeepONetWrapper
from Models.VIDON import VIDON

from Plotting.paper_plot_config import BENCHMARKS, CHECKPOINT_PATTERNS


# =============================================================================
# Dimension Inference from Checkpoints
# =============================================================================

def infer_vidon_dims(state_dict: dict) -> tuple[int, int]:
    """Infer (p, output_size_tgt) from VIDON checkpoint.
    
    VIDON: combiner_out = p * d_out, trunk_out = (p+1) * d_out
    """
    combiner_biases = sorted([k for k in state_dict if 'combiner.net' in k and 'bias' in k])
    trunk_biases = sorted([k for k in state_dict if 'trunk.net' in k and 'bias' in k])
    
    if combiner_biases and trunk_biases:
        c_out = int(state_dict[combiner_biases[-1]].shape[0])
        t_out = int(state_dict[trunk_biases[-1]].shape[0])
        d_out = t_out - c_out
        if d_out > 0:
            return c_out // d_out, d_out
    return 128, 1


def infer_output_dim(state_dict: dict, model_type: str) -> int:
    """Infer output_size_tgt from checkpoint."""
    if model_type == 'vidon':
        return infer_vidon_dims(state_dict)[1]
    return int(state_dict.get('bias', torch.tensor([0])).shape[0]) or 1


def infer_vidon_output_size_src(state_dict: dict) -> int | None:
    """Infer VIDON output_size_src (du) from checkpoint."""
    key = "value_encoder.net.0.weight"
    if key in state_dict:
        return int(state_dict[key].shape[1])
    return None


def infer_setonet_output_size_src(state_dict: dict, arch: dict) -> int | None:
    """Infer output_size_src (du) for SetONet from checkpoint weights."""
    use_pe = arch.get("use_positional_encoding", True)
    pe_type = arch.get("pos_encoding_type", "sinusoidal")
    if pe_type == "skip":
        use_pe = False
    dx_enc = arch.get("pos_encoding_dim", 0) if use_pe else arch.get("input_size_src", 2)

    # Prefer heads with explicit (x,u) value nets
    value_keys = [
        "quadrature_head.value_net.0.weight",
        "adaptive_quadrature_head.value_net.0.weight",
        "galerkin_head.value_net.0.weight",
        "pg_head.value_net.0.weight",
    ]
    for key in value_keys:
        if key in state_dict:
            in_features = int(state_dict[key].shape[1])
            du = in_features - int(dx_enc)
            if du > 0:
                return du

    # Fallback: standard phi network (concatenated [x, u] or [pe(x), u])
    if "phi.0.weight" in state_dict:
        in_features = int(state_dict["phi.0.weight"].shape[1])
        if use_pe:
            du = in_features - int(dx_enc)
        else:
            du = in_features - int(arch.get("input_size_src", 2))
        if du > 0:
            return du

    return None


# =============================================================================
# Model Creation
# =============================================================================

def create_setonet(arch: dict, device: str) -> SetONet:
    """Create SetONet from architecture config."""
    return SetONet(
        input_size_src=arch.get("input_size_src", 2),
        output_size_src=arch.get("output_size_src", 1),
        input_size_tgt=arch.get("input_size_tgt", 2),
        output_size_tgt=arch.get("output_size_tgt", 1),
        p=arch.get("son_p_dim", 128),
        phi_hidden_size=arch.get("son_phi_hidden", 256),
        rho_hidden_size=arch.get("son_rho_hidden", 256),
        trunk_hidden_size=arch.get("son_trunk_hidden", 256),
        n_trunk_layers=arch.get("son_n_trunk_layers", 4),
        phi_output_size=arch.get("son_phi_output_size", 32),
        use_deeponet_bias=arch.get("use_deeponet_bias", True),
        pos_encoding_type=arch.get("pos_encoding_type", "sinusoidal"),
        pos_encoding_dim=arch.get("pos_encoding_dim", 64),
        pos_encoding_max_freq=arch.get("pos_encoding_max_freq", 0.01),
        use_positional_encoding=arch.get("use_positional_encoding", True),
        aggregation_type=arch.get("son_aggregation", "attention"),
        attention_n_tokens=arch.get("attention_n_tokens", 1),
        branch_head_type=arch.get("son_branch_head_type", "quadrature"),
    ).to(device)


def create_vidon(arch: dict, device: str) -> VIDON:
    """Create VIDON from architecture config."""
    return VIDON(
        input_size_src=arch.get("input_size_src", 2),
        output_size_src=arch.get("output_size_src", 1),
        input_size_tgt=arch.get("input_size_tgt", 2),
        output_size_tgt=arch.get("output_size_tgt", 1),
        p=arch.get("vidon_p_dim", 128),
        n_heads=arch.get("vidon_n_heads", 4),
        d_enc=arch.get("vidon_d_enc", 40),
        head_output_size=arch.get("vidon_head_output_size", 64),
        enc_hidden_size=arch.get("vidon_enc_hidden", 40),
        enc_n_layers=arch.get("vidon_enc_n_layers", 4),
        head_hidden_size=arch.get("vidon_head_hidden", 128),
        head_n_layers=arch.get("vidon_head_n_layers", 4),
        combine_hidden_size=arch.get("vidon_combine_hidden", 256),
        combine_n_layers=arch.get("vidon_combine_n_layers", 4),
        trunk_hidden_size=arch.get("vidon_trunk_hidden", 256),
        n_trunk_layers=arch.get("vidon_n_trunk_layers", 4),
    ).to(device)


def create_deeponet(arch: dict, data_cfg: dict, device: str) -> DeepONetWrapper:
    """Create DeepONet from architecture config."""
    return DeepONetWrapper(
        branch_input_dim=data_cfg.get("n_force_points", 301),
        trunk_input_dim=arch.get("input_size_tgt", 2),
        p=arch.get("don_p_dim", 128),
        trunk_hidden_size=arch.get("don_trunk_hidden", 256),
        n_trunk_layers=arch.get("don_n_trunk_layers", 4),
        branch_hidden_size=arch.get("don_branch_hidden", 128),
        n_branch_layers=arch.get("don_n_branch_layers", 3),
        use_deeponet_bias=arch.get("use_deeponet_bias", True),
    ).to(device)


# =============================================================================
# Model Loading
# =============================================================================

def load_model(benchmark: str, model_name: str, seed: int, logs_dir: Path, device: str):
    """Load a trained model from checkpoint."""
    log_dir = logs_dir / benchmark / model_name / f"seed_{seed}"
    if not log_dir.exists():
        print(f"  Warning: {log_dir} not found")
        return None
    
    # Load config
    config_path = log_dir / "experiment_config.json"
    if not config_path.exists():
        print(f"  Warning: Config not found in {log_dir}")
        return None
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Find checkpoint
    ckpt_name = CHECKPOINT_PATTERNS.get(benchmark, {}).get(model_name)
    ckpt_path = log_dir / ckpt_name if ckpt_name else None
    
    if not ckpt_path or not ckpt_path.exists():
        pth_files = list(log_dir.glob("*.pth"))
        if not pth_files:
            print(f"  Warning: No checkpoint in {log_dir}")
            return None
        ckpt_path = pth_files[0]
    
    # Load state dict and infer dimensions
    state_dict = torch.load(ckpt_path, map_location=device)
    arch = config.get("model_architecture", {})
    
    # Determine model type and fix dimensions from checkpoint
    mtype = "setonet" if "setonet" in model_name else "vidon" if "vidon" in model_name else "deeponet"
    
    arch["output_size_tgt"] = infer_output_dim(state_dict, mtype)
    if mtype == "vidon":
        arch["vidon_p_dim"] = infer_vidon_dims(state_dict)[0]
        du = infer_vidon_output_size_src(state_dict)
        if du is not None:
            arch["output_size_src"] = du
    elif mtype == "setonet":
        du = infer_setonet_output_size_src(state_dict, arch)
        if du is not None:
            arch["output_size_src"] = du
    
    # Create and load model
    if mtype == "setonet":
        model = create_setonet(arch, device)
    elif mtype == "vidon":
        model = create_vidon(arch, device)
    else:
        model = create_deeponet(arch, config.get("dataset_structure", {}), device)
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded {model_name} from {ckpt_path.name}")
    return model


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset(benchmark: str, device: str):
    """Load dataset for a benchmark. Returns (hf_dataset, wrapper)."""
    cfg = BENCHMARKS[benchmark]
    data_path = str(_project_root / cfg['data_path'])
    plot_type = cfg['plot_type']
    
    if plot_type == 'heat':
        from Data.heat_data.heat_2d_dataset import load_heat_dataset
        return load_heat_dataset(data_path, batch_size=1, device=device)
    
    elif plot_type == 'elastic':
        from Data.elastic_2d_data.elastic_2d_dataset import load_elastic_dataset
        return load_elastic_dataset(data_path, batch_size=1, device=device)
    
    elif plot_type == 'concentration':
        from Data.concentration_data.concentration_2d_dataset import load_concentration_dataset
        return load_concentration_dataset(data_path, batch_size=1, device=device)
    
    elif plot_type == 'diffraction':
        from Data.diffraction_2d_data.diffraction_2d_dataset import load_diffraction_dataset
        return load_diffraction_dataset(data_path, batch_size=1, device=device)
    
    elif plot_type == 'transport_q':
        from Data.transport_q_data.transport_dataset import load_transport_dataset
        return load_transport_dataset(data_path, batch_size=1, device=device, mode='transport_map')
    
    raise ValueError(f"Unknown plot type: {plot_type}")
