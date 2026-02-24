#!/usr/bin/env python
"""paper_plot_utils_1d.py - Utilities for 1D paper figures."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from Data.burgers_1d_data.burgers_1d_dataset import load_burgers_dataset
from Data.darcy_1d_data.darcy_1d_dataset import load_darcy_dataset
from Models.SetONet import SetONet
from Models.DeepONet import DeepONetWrapper
from Models.VIDON import VIDON
from Benchmarks.benchmark_utils import (
    MODEL_VARIANTS,
    get_benchmark_overrides,
    load_benchmark_config,
)
from Plotting.paper_plot_config_1d import BENCHMARK_CONFIGS, ROW_CONFIGS_BY_BENCHMARK

_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent
_benchmarks_dir = _project_root / "Benchmarks"
_default_benchmark_config_path = _benchmarks_dir / "benchmark_config.yaml"


def _get_activation(name: str) -> type[nn.Module]:
    if not name:
        return nn.ReLU
    name = name.lower()
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
    }.get(name, nn.ReLU)


def _default_quad_phi_activation(benchmark: str) -> str:
    if benchmark.startswith(("1d_", "darcy_", "burgers_")):
        return "softplus"
    return "tanh"


@lru_cache(maxsize=16)
def _load_user_overrides(logs_dir_str: str) -> dict:
    logs_dir = Path(logs_dir_str)
    candidates = [
        logs_dir / "_results" / "benchmark_config.yaml",
        _default_benchmark_config_path,
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r") as handle:
            cfg = yaml.safe_load(handle) or {}
        overrides = cfg.get("overrides", {})
        if isinstance(overrides, dict):
            return overrides
    return {}


def _infer_setonet_variant_defaults(model_name: str) -> dict:
    defaults = {}
    if model_name == "setonet_sum":
        defaults["son_aggregation"] = "sum"
    elif model_name == "setonet_mean":
        defaults["son_aggregation"] = "mean"
    elif model_name == "setonet_attention":
        defaults["son_aggregation"] = "attention"

    if model_name == "setonet_petrov":
        defaults["son_branch_head_type"] = "petrov_attention"
    elif model_name == "setonet_galerkin":
        defaults["son_branch_head_type"] = "galerkin_pou"
    elif model_name == "setonet_quadrature":
        defaults["son_branch_head_type"] = "quadrature"
    elif model_name == "setonet_adaptive":
        defaults["son_branch_head_type"] = "adaptive_quadrature"
    elif model_name.startswith("setonet"):
        defaults["son_branch_head_type"] = "standard"
    return defaults


def _resolve_arch_from_benchmarks(benchmark: str, model_name: str, logs_root: Path) -> dict:
    variant = MODEL_VARIANTS.get(model_name, {"base": model_name, "overrides": {}})
    base_model = variant.get("base", model_name)
    arch = {}

    if base_model in {"setonet", "deeponet", "vidon"}:
        arch.update(load_benchmark_config(_benchmarks_dir, base_model, benchmark))
        arch.update(variant.get("overrides", {}))
        arch.update(get_benchmark_overrides(variant, benchmark))

    user_overrides = _load_user_overrides(str(logs_root.resolve()))
    bench_overrides = user_overrides.get(benchmark, {})
    model_bench_overrides = user_overrides.get(f"{model_name}_{benchmark}", {})
    if isinstance(bench_overrides, dict):
        arch.update(bench_overrides)
    if isinstance(model_bench_overrides, dict):
        arch.update(model_bench_overrides)

    if model_name.startswith("setonet"):
        arch.setdefault("son_quad_phi_activation", _default_quad_phi_activation(benchmark))
        arch.setdefault("son_quad_value_mode", "linear_u")
    return arch


def infer_setonet_branch_head_type(state_dict: dict) -> str:
    if any(k.startswith("adaptive_quadrature_head.") for k in state_dict):
        return "adaptive_quadrature"
    if any(k.startswith("quadrature_head.") for k in state_dict):
        return "quadrature"
    if any(k.startswith("galerkin_head.") for k in state_dict):
        return "galerkin_pou"
    if any(k.startswith("pg_head.") for k in state_dict):
        return "petrov_attention"
    return "standard"


def _load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def _create_setonet(arch: dict, device: str) -> SetONet:
    activation_fn = _get_activation(arch.get("activation_fn", "relu"))
    use_positional_encoding = arch.get("use_positional_encoding")
    if use_positional_encoding is None:
        use_positional_encoding = arch.get("pos_encoding_type", "sinusoidal") != "skip"

    return SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=arch.get("son_p_dim", 32),
        phi_hidden_size=arch.get("son_phi_hidden", 256),
        rho_hidden_size=arch.get("son_rho_hidden", 256),
        trunk_hidden_size=arch.get("son_trunk_hidden", 256),
        n_trunk_layers=arch.get("son_n_trunk_layers", 4),
        activation_fn=activation_fn,
        use_deeponet_bias=arch.get("use_deeponet_bias", True),
        phi_output_size=arch.get("son_phi_output_size", 32),
        initial_lr=arch.get("son_lr", 5e-4),
        lr_schedule_steps=arch.get("lr_schedule_steps"),
        lr_schedule_gammas=arch.get("lr_schedule_gammas"),
        pos_encoding_type=arch.get("pos_encoding_type", "sinusoidal"),
        pos_encoding_dim=arch.get("pos_encoding_dim", 64),
        pos_encoding_max_freq=arch.get("pos_encoding_max_freq", 0.1),
        use_positional_encoding=use_positional_encoding,
        aggregation_type=arch.get("son_aggregation", "attention"),
        attention_n_tokens=arch.get("attention_n_tokens", 1),
        branch_head_type=arch.get("son_branch_head_type", "standard"),
        pg_dk=arch.get("son_pg_dk"),
        pg_dv=arch.get("son_pg_dv"),
        pg_use_logw=(not arch.get("son_pg_no_logw", False)),
        galerkin_dk=arch.get("son_galerkin_dk"),
        galerkin_dv=arch.get("son_galerkin_dv"),
        galerkin_normalize=arch.get("son_galerkin_normalize", "total"),
        galerkin_learn_temperature=arch.get("son_galerkin_learn_temperature", False),
        quad_dk=arch.get("son_quad_dk", 64),
        quad_dv=arch.get("son_quad_dv"),
        quad_key_hidden=arch.get("son_quad_key_hidden"),
        quad_key_layers=arch.get("son_quad_key_layers", 3),
        quad_phi_activation=arch.get("son_quad_phi_activation", "softplus"),
        quad_value_mode=arch.get("son_quad_value_mode", "linear_u"),
        quad_normalize=arch.get("son_quad_normalize", "total"),
        quad_learn_temperature=arch.get("son_quad_learn_temperature", False),
        adapt_quad_rank=arch.get("son_adapt_quad_rank", 4),
        adapt_quad_hidden=arch.get("son_adapt_quad_hidden", 64),
        adapt_quad_scale=arch.get("son_adapt_quad_scale", 0.1),
        adapt_quad_use_value_context=arch.get("son_adapt_quad_use_value_context", True),
    ).to(device)


def _create_deeponet(arch: dict, data_cfg: dict, device: str) -> DeepONetWrapper:
    activation_fn = _get_activation(arch.get("activation_fn", "relu"))
    return DeepONetWrapper(
        branch_input_dim=data_cfg.get("n_force_points", 300),
        trunk_input_dim=1,
        p=arch.get("don_p_dim", 32),
        trunk_hidden_size=arch.get("don_trunk_hidden", 256),
        n_trunk_layers=arch.get("don_n_trunk_layers", 4),
        branch_hidden_size=arch.get("don_branch_hidden", 128),
        n_branch_layers=arch.get("don_n_branch_layers", 3),
        activation_fn=activation_fn,
        initial_lr=arch.get("don_lr", 5e-4),
        lr_schedule_steps=arch.get("lr_schedule_steps"),
        lr_schedule_gammas=arch.get("lr_schedule_gammas"),
        use_deeponet_bias=arch.get("use_deeponet_bias", True),
    ).to(device)


def _create_vidon(arch: dict, device: str) -> VIDON:
    activation_fn = _get_activation(arch.get("activation_fn", "relu"))
    return VIDON(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=arch.get("vidon_p_dim", 32),
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
        activation_fn=activation_fn,
        initial_lr=arch.get("vidon_lr", 5e-4),
        lr_schedule_steps=arch.get("lr_schedule_steps"),
        lr_schedule_gammas=arch.get("lr_schedule_gammas"),
    ).to(device)


def _resolve_ckpt_name(bench_cfg: dict, model_dir: str) -> str | None:
    ckpt_names = bench_cfg.get("ckpt_names", {})
    if model_dir.startswith("setonet"):
        return ckpt_names.get("setonet")
    if model_dir == "deeponet":
        return ckpt_names.get("deeponet")
    if model_dir == "vidon":
        return ckpt_names.get("vidon")
    return None


def load_model(logs_root: Path, run_dir: str, model_dir: str, device: str, bench_cfg: dict):
    log_dir = logs_root / run_dir / model_dir / "seed_0"
    if not log_dir.exists():
        print(f"  Warning: {log_dir} not found")
        return None

    config_path = log_dir / "experiment_config.json"
    if not config_path.exists():
        print(f"  Warning: {config_path} missing")
        return None

    config = _load_json(config_path)
    arch = config.get("model_architecture", {})
    data_cfg = config.get("dataset_structure", {})

    resolved_arch = _resolve_arch_from_benchmarks(run_dir, model_dir, logs_root)
    arch = {**resolved_arch, **arch}

    if model_dir.startswith("setonet"):
        variant_defaults = _infer_setonet_variant_defaults(model_dir)
        for key, value in variant_defaults.items():
            arch.setdefault(key, value)
        arch.setdefault("son_quad_phi_activation", _default_quad_phi_activation(run_dir))
        arch.setdefault("son_quad_value_mode", "linear_u")
        if "use_positional_encoding" not in arch:
            arch["use_positional_encoding"] = arch.get("pos_encoding_type", "sinusoidal") != "skip"

    ckpt_name = _resolve_ckpt_name(bench_cfg, model_dir)
    if not ckpt_name:
        print(f"  Warning: No checkpoint name for {model_dir}")
        return None

    ckpt_path = log_dir / ckpt_name
    if not ckpt_path.exists():
        pth_files = list(log_dir.glob("*.pth"))
        if not pth_files:
            print(f"  Warning: No checkpoint in {log_dir}")
            return None
        ckpt_path = pth_files[0]

    state_dict = torch.load(ckpt_path, map_location=device)
    if model_dir.startswith("setonet"):
        arch.setdefault("son_branch_head_type", infer_setonet_branch_head_type(state_dict))
        model = _create_setonet(arch, device)
    elif model_dir == "deeponet":
        model = _create_deeponet(arch, data_cfg, device)
    elif model_dir == "vidon":
        model = _create_vidon(arch, device)
    else:
        print(f"  Warning: Unknown model dir {model_dir}")
        return None

    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded {model_dir} from {ckpt_path.name}")
    return {
        "model": model,
        "config": config,
        "n_force_points": data_cfg.get("n_force_points"),
        "n_mesh_points": data_cfg.get("n_mesh_points"),
    }


def infer_sensor_and_query_counts(logs_root: Path, benchmark: str) -> tuple[int, int]:
    row_configs = ROW_CONFIGS_BY_BENCHMARK[benchmark]
    base_dir = row_configs[0]["log_dir"]
    config_path = logs_root / base_dir / "setonet_quadrature" / "seed_0" / "experiment_config.json"
    if config_path.exists():
        config = _load_json(config_path)
        data_cfg = config.get("dataset_structure", {})
        sensor_size = data_cfg.get("n_force_points")
        query_size = data_cfg.get("n_mesh_points")
        if sensor_size and query_size:
            return int(sensor_size), int(query_size)
    defaults = BENCHMARK_CONFIGS[benchmark]
    return defaults["default_sensor_size"], defaults["default_query_points"]


def load_dataset_for_benchmark(benchmark: str, device: str, darcy_data_path: str):
    if benchmark == "burgers_1d":
        dataset, stats = load_burgers_dataset(device=device)
        return dataset, stats
    if benchmark == "darcy_1d":
        dataset = load_darcy_dataset(darcy_data_path)
        return dataset, None
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_models_for_rows(
    row_configs: list[dict],
    bench_cfg: dict,
    logs_root: Path,
    device: str,
    grid_points: torch.Tensor | None = None,
) -> list[dict]:
    row_models = []
    for row_cfg in row_configs:
        models = {}
        for key, model_dir in row_cfg["model_dirs"].items():
            model_info = load_model(logs_root, row_cfg["log_dir"], model_dir, device, bench_cfg)
            if model_info and bench_cfg["data_kind"] == "dataset":
                n_force_points = model_info.get("n_force_points") or bench_cfg["default_sensor_size"]
                sensor_x, sensor_indices = bench_cfg["sensor_points_fn"](
                    {"sensor_size": n_force_points}, device, grid_points
                )
                model_info["sensor_x"] = sensor_x
                model_info["sensor_indices"] = sensor_indices
            models[key] = model_info
        row_models.append(models)
    return row_models


def sample_sensor_points(sensor_size: int, input_range: tuple[float, float], device: str, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    lo, hi = input_range
    sensor_x = torch.rand(sensor_size, generator=gen, device=device) * (hi - lo) + lo
    sensor_x = sensor_x.sort()[0]
    return sensor_x.view(-1, 1)


def sample_synthetic_coeffs(seed: int, scale: float, device: str) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    a = (torch.rand(1, generator=gen, device=device) * 2 - 1) * scale
    b = (torch.rand(1, generator=gen, device=device) * 2 - 1) * scale
    c = (torch.rand(1, generator=gen, device=device) * 2 - 1) * scale
    e = (torch.rand(1, generator=gen, device=device) * 2 - 1) * scale
    d = torch.zeros(1, device=device)
    return {"a": a, "b": b, "c": c, "d": d, "e": e}


def eval_synthetic_function(x: torch.Tensor, coeffs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    a = coeffs["a"]
    b = coeffs["b"]
    c = coeffs["c"]
    d = coeffs["d"]
    e = coeffs["e"]
    f_val = a * x**3 + b * x**2 + c * x + d + e * torch.sin(x)
    f_prime = 3 * a * x**2 + 2 * b * x + c + e * torch.cos(x)
    return f_val, f_prime
