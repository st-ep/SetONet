#!/usr/bin/env python3
"""
Training Speed Comparison Script for SetONet, DeepONet, and VIDON.

Measures training throughput (samples/second) and time per iteration
across all model variants on elastic_2d benchmarks.

Usage:
    python Benchmarks/run_timing_comparison.py
    python Benchmarks/run_timing_comparison.py --device cuda:0
    python Benchmarks/run_timing_comparison.py --seeds 0,1,2
    python Benchmarks/run_timing_comparison.py --models vidon,setonet_attention --benchmark-config Benchmarks/benchmark_config.yaml
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.SetONet import SetONet
from Models.DeepONet import DeepONetWrapper
from Models.VIDON import VIDON
from Benchmarks.benchmark_utils import (
    MODEL_VARIANTS as BENCHMARK_MODEL_VARIANTS,
    apply_user_overrides as apply_user_overrides_from_utils,
    get_benchmark_overrides as get_benchmark_overrides_from_utils,
    load_benchmark_config as load_benchmark_config_from_utils,
)


# =============================================================================
# Configuration
# =============================================================================

WARMUP_EPOCHS = 100
MEASURED_EPOCHS = 1000
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BENCHMARK_CONFIG = SCRIPT_DIR / "benchmark_config.yaml"
DEFAULT_TIMING_MODELS = [
    "deeponet",
    "vidon",
    "setonet_sum",
    "setonet_mean",
    "setonet_attention",
    "setonet_quadrature",
]

# Source of truth for model variants
MODEL_VARIANTS = BENCHMARK_MODEL_VARIANTS

# Benchmark configurations
BENCHMARKS = {
    "elastic_2d": {
        "default_data_path": "Data/elastic_2d_data/elastic_dataset",
        "input_size_src": 2,
        "output_size_src": 1,
        "input_size_tgt": 2,
        "output_size_tgt": 1,
        "sensor_size": None,  # Will be set from dataset (n_force_points)
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TimingResult:
    """Result of a single timing run."""
    model: str
    benchmark: str
    seed: int
    warmup_epochs: int
    measured_epochs: int
    batch_size: int
    elapsed_seconds: float
    time_per_iter_ms: float


@dataclass
class AggregatedResult:
    """Aggregated result across seeds."""
    model: str
    benchmark: str
    n_seeds: int
    batch_size: int
    time_per_iter_ms_mean: float
    time_per_iter_ms_std: float


# =============================================================================
# Config Loading
# =============================================================================

def load_benchmark_runner_config(config_path: Path) -> Dict[str, Any]:
    """Load benchmark runner config and return parsed YAML (or empty dict)."""
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_model_config(
    model_name: str,
    benchmark: str,
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Get merged config for a model/benchmark pair using benchmark_utils resolution order."""
    variant = MODEL_VARIANTS[model_name]
    base_model = variant["base"]
    config = load_benchmark_config_from_utils(SCRIPT_DIR, base_model, benchmark)

    # Apply variant overrides (e.g., son_aggregation, son_branch_head_type)
    config.update(variant.get("overrides", {}))

    # Apply benchmark-specific overrides (e.g., son_rho_hidden for darcy_1d)
    bench_specific = get_benchmark_overrides_from_utils(variant, benchmark)
    config.update(bench_specific)

    # Apply user overrides from benchmark_config.yaml
    config = apply_user_overrides_from_utils(config, user_overrides, model_name, benchmark)

    return config


# =============================================================================
# Model Creation
# =============================================================================

def create_setonet(config: Dict[str, Any], bench_info: Dict[str, Any], device: torch.device) -> SetONet:
    """Create SetONet model from config."""
    model = SetONet(
        input_size_src=bench_info["input_size_src"],
        output_size_src=bench_info["output_size_src"],
        input_size_tgt=bench_info["input_size_tgt"],
        output_size_tgt=bench_info["output_size_tgt"],
        p=config.get("son_p_dim", 32),
        phi_hidden_size=config.get("son_phi_hidden", 256),
        rho_hidden_size=config.get("son_rho_hidden", 256),
        trunk_hidden_size=config.get("son_trunk_hidden", 256),
        n_trunk_layers=config.get("son_n_trunk_layers", 4),
        phi_output_size=config.get("son_phi_output_size", 32),
        activation_fn=nn.ReLU,
        use_deeponet_bias=True,
        initial_lr=config.get("son_lr", 5e-4),
        lr_schedule_steps=config.get("lr_schedule_steps"),
        lr_schedule_gammas=config.get("lr_schedule_gammas"),
        pos_encoding_type=config.get("pos_encoding_type", "sinusoidal"),
        pos_encoding_dim=config.get("pos_encoding_dim", 64),
        pos_encoding_max_freq=config.get("pos_encoding_max_freq", 0.1),
        aggregation_type=config.get("son_aggregation", "attention"),
        use_positional_encoding=(config.get("pos_encoding_type", "sinusoidal") != "skip"),
        attention_n_tokens=1,
        branch_head_type=config.get("son_branch_head_type", "standard"),
        pg_n_refine_iters=config.get("son_pg_n_refine_iters", 0),
        pg_dk=config.get("son_pg_dk"),
        pg_dv=config.get("son_pg_dv"),
        pg_use_logw=not config.get("son_pg_no_logw", False),
        galerkin_dk=config.get("son_galerkin_dk"),
        galerkin_dv=config.get("son_galerkin_dv"),
        galerkin_normalize=config.get("son_galerkin_normalize", "total"),
        galerkin_learn_temperature=config.get("son_galerkin_learn_temperature", False),
        quad_dk=config.get("son_quad_dk"),
        quad_dv=config.get("son_quad_dv"),
        quad_key_hidden=config.get("son_quad_key_hidden"),
        quad_key_layers=config.get("son_quad_key_layers", 3),
        quad_phi_activation=config.get("son_quad_phi_activation", "tanh"),
        quad_value_mode=config.get("son_quad_value_mode", "linear_u"),
        quad_normalize=config.get("son_quad_normalize", "total"),
        quad_learn_temperature=config.get("son_quad_learn_temperature", False),
        adapt_quad_rank=config.get("son_adapt_quad_rank", 4),
        adapt_quad_hidden=config.get("son_adapt_quad_hidden", 64),
        adapt_quad_scale=config.get("son_adapt_quad_scale", 0.1),
        adapt_quad_use_value_context=config.get("son_adapt_quad_use_value_context", True),
    ).to(device)
    return model


def create_vidon(config: Dict[str, Any], bench_info: Dict[str, Any], device: torch.device) -> VIDON:
    """Create VIDON model from config."""
    model = VIDON(
        input_size_src=bench_info["input_size_src"],
        output_size_src=bench_info["output_size_src"],
        input_size_tgt=bench_info["input_size_tgt"],
        output_size_tgt=bench_info["output_size_tgt"],
        p=config.get("vidon_p_dim", 32),
        n_heads=config.get("vidon_n_heads", 4),
        d_enc=config.get("vidon_d_enc", 40),
        head_output_size=config.get("vidon_head_output_size", 64),
        enc_hidden_size=config.get("vidon_enc_hidden", 40),
        enc_n_layers=config.get("vidon_enc_n_layers", 4),
        head_hidden_size=config.get("vidon_head_hidden", 128),
        head_n_layers=config.get("vidon_head_n_layers", 4),
        combine_hidden_size=config.get("vidon_combine_hidden", 256),
        combine_n_layers=config.get("vidon_combine_n_layers", 4),
        trunk_hidden_size=config.get("vidon_trunk_hidden", 256),
        n_trunk_layers=config.get("vidon_n_trunk_layers", 4),
        activation_fn=nn.ReLU,
        initial_lr=config.get("vidon_lr", 5e-4),
        lr_schedule_steps=config.get("lr_schedule_steps"),
        lr_schedule_gammas=config.get("lr_schedule_gammas"),
    ).to(device)
    return model


def create_deeponet(config: Dict[str, Any], bench_info: Dict[str, Any], device: torch.device) -> DeepONetWrapper:
    """Create DeepONet model from config."""
    sensor_size = bench_info.get("sensor_size", 300)
    model = DeepONetWrapper(
        branch_input_dim=sensor_size,
        trunk_input_dim=bench_info["input_size_tgt"],
        p=config.get("don_p_dim", 32),
        trunk_hidden_size=config.get("don_trunk_hidden", 256),
        n_trunk_layers=config.get("don_n_trunk_layers", 4),
        branch_hidden_size=config.get("don_branch_hidden", 128),
        n_branch_layers=config.get("don_n_branch_layers", 3),
        activation_fn=nn.ReLU,
        initial_lr=config.get("don_lr", 5e-4),
        lr_schedule_steps=config.get("lr_schedule_steps"),
        lr_schedule_gammas=config.get("lr_schedule_gammas"),
    ).to(device)
    return model


def create_model(
    model_name: str,
    benchmark: str,
    device: torch.device,
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[nn.Module]:
    """Create model for given variant and benchmark."""
    config = get_model_config(model_name, benchmark, user_overrides=user_overrides)
    if config is None:
        return None

    variant = MODEL_VARIANTS[model_name]
    base_model = variant["base"]
    bench_info = BENCHMARKS[benchmark]

    if base_model == "setonet":
        return create_setonet(config, bench_info, device)
    elif base_model == "vidon":
        return create_vidon(config, bench_info, device)
    elif base_model == "deeponet":
        return create_deeponet(config, bench_info, device)
    else:
        raise ValueError(f"Unknown base model: {base_model}")


# =============================================================================
# Dataset Loading
# =============================================================================

def load_elastic_dataset(device: torch.device, batch_size: int, config: Dict[str, Any]):
    """Load Elastic 2D dataset and return data generator."""
    from Data.elastic_2d_data.elastic_2d_dataset import load_elastic_dataset as _load_elastic

    data_path = config.get("data_path", BENCHMARKS["elastic_2d"]["default_data_path"])
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    _, elastic_dataset = _load_elastic(
        str(data_path),
        batch_size=batch_size,
        device=device,
        train_sensor_dropoff=config.get("train_sensor_dropoff", 0.0),
        replace_with_nearest=config.get("replace_with_nearest", False),
    )

    # Update sensor_size in BENCHMARKS for DeepONet (n_force_points is fixed)
    BENCHMARKS["elastic_2d"]["sensor_size"] = elastic_dataset.n_force_points

    return elastic_dataset, batch_size


def get_dataset_loader(benchmark: str):
    """Get dataset loader function for benchmark."""
    if benchmark == "elastic_2d":
        return load_elastic_dataset
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# =============================================================================
# Timing Measurement
# =============================================================================

def run_training_epochs(model: nn.Module, dataset, n_epochs: int, device: torch.device, use_sensor_mask: bool = True) -> None:
    """Run training for n_epochs without timing."""
    model.train()
    for _ in range(n_epochs):
        # Keep LR scheduling behavior aligned with model.train_model()
        if hasattr(model, "_update_lr"):
            model._update_lr()

        xs, us, ys, G_u_ys, sensor_mask = dataset.sample(device=device)
        if use_sensor_mask:
            pred = model(xs, us, ys, sensor_mask=sensor_mask)
        else:
            pred = model(xs, us, ys)
        loss = nn.MSELoss()(pred, G_u_ys)
        model.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.opt.step()

        if hasattr(model, "total_steps"):
            model.total_steps += 1


def measure_training_time(
    model: nn.Module,
    dataset,
    warmup_epochs: int,
    measured_epochs: int,
    batch_size: int,
    device: torch.device,
    use_sensor_mask: bool = True,
) -> Tuple[float, int]:
    """
    Run warmup + measured epochs and return elapsed time and total samples.

    Returns:
        elapsed_seconds: Time for measured_epochs only
        total_samples: batch_size * measured_epochs
    """
    model.train()

    # Warmup phase (not timed)
    run_training_epochs(model, dataset, warmup_epochs, device, use_sensor_mask)

    # Synchronize GPU before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Measured phase
    start_time = time.perf_counter()
    run_training_epochs(model, dataset, measured_epochs, device, use_sensor_mask)

    # Synchronize GPU after timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time
    total_samples = batch_size * measured_epochs

    return elapsed, total_samples


def run_single_timing(
    model_name: str,
    benchmark: str,
    seed: int,
    device: torch.device,
    user_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[TimingResult]:
    """Run timing for a single (model, benchmark, seed) combination."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Get config and batch size
    config = get_model_config(model_name, benchmark, user_overrides=user_overrides)
    if config is None:
        return None

    batch_size = config.get("batch_size", 64)

    # Load dataset FIRST (this sets sensor_size for elastic_2d, needed by DeepONet)
    loader_fn = get_dataset_loader(benchmark)
    dataset, _ = loader_fn(device, batch_size, config)

    # Create model (after dataset so sensor_size is available)
    model = create_model(model_name, benchmark, device, user_overrides=user_overrides)
    if model is None:
        return None

    # Determine if sensor_mask should be used (DeepONet doesn't support it)
    variant = MODEL_VARIANTS[model_name]
    use_sensor_mask = variant["base"] != "deeponet"

    # Run timing
    elapsed, total_samples = measure_training_time(
        model=model,
        dataset=dataset,
        warmup_epochs=WARMUP_EPOCHS,
        measured_epochs=MEASURED_EPOCHS,
        batch_size=batch_size,
        device=device,
        use_sensor_mask=use_sensor_mask,
    )

    time_per_iter_ms = (elapsed / MEASURED_EPOCHS) * 1000

    return TimingResult(
        model=model_name,
        benchmark=benchmark,
        seed=seed,
        warmup_epochs=WARMUP_EPOCHS,
        measured_epochs=MEASURED_EPOCHS,
        batch_size=batch_size,
        elapsed_seconds=elapsed,
        time_per_iter_ms=time_per_iter_ms,
    )


# =============================================================================
# Results Aggregation
# =============================================================================

def aggregate_results(results: List[TimingResult]) -> List[AggregatedResult]:
    """Aggregate results by (benchmark, model) across seeds."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        grouped[(r.benchmark, r.model)].append(r)

    aggregated = []
    # Sort by (benchmark, model) so all models for one benchmark come first
    for (benchmark, model), runs in sorted(grouped.items()):
        tpi = [r.time_per_iter_ms for r in runs]
        batch_size = runs[0].batch_size

        aggregated.append(AggregatedResult(
            model=model,
            benchmark=benchmark,
            n_seeds=len(runs),
            batch_size=batch_size,
            time_per_iter_ms_mean=mean(tpi),
            time_per_iter_ms_std=stdev(tpi) if len(tpi) > 1 else 0.0,
        ))

    return aggregated


def save_results(
    aggregated_results: List[AggregatedResult],
    output_dir: Path,
    model_names: List[str],
    benchmark_config_path: Path,
) -> None:
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pivot table (benchmark × model) - models as columns
    pivot_path = output_dir / "timing_pivot.csv"
    benchmarks = sorted(set(r.benchmark for r in aggregated_results))
    models = sorted(set(r.model for r in aggregated_results))

    # Create lookup
    lookup = {(r.model, r.benchmark): r for r in aggregated_results}

    with open(pivot_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: benchmark, then all models
        header = ["benchmark"] + models
        writer.writerow(header)
        # Each row is a benchmark with time_per_iter_ms for each model
        for bench in benchmarks:
            row = [bench]
            for model in models:
                r = lookup.get((model, bench))
                if r:
                    row.append(f"{r.time_per_iter_ms_mean:.3f} +/- {r.time_per_iter_ms_std:.6f}")
                else:
                    row.append("N/A")
            writer.writerow(row)

    # Save metadata
    metadata_path = output_dir / "timing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "warmup_epochs": WARMUP_EPOCHS,
            "measured_epochs": MEASURED_EPOCHS,
            "models": model_names,
            "benchmarks": list(BENCHMARKS.keys()),
            "benchmark_config_path": str(benchmark_config_path),
            "n_aggregated_results": len(aggregated_results),
        }, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - {pivot_path.name}")
    print(f"  - {metadata_path.name}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training speed comparison for neural operators")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="CUDA device (default: cuda:0)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (default: 0,1,2,3,4)")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model variants to run (default: built-in timing model set)")
    parser.add_argument("--use-config-models", action="store_true", help="Use 'models' list from benchmark_config.yaml when --models is not provided")
    parser.add_argument("--benchmark-config", type=str, default=str(DEFAULT_BENCHMARK_CONFIG), help="Path to benchmark_config.yaml for user overrides/model selection")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory (default: logs_timing/)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load benchmark runner config for user overrides and default model selection
    benchmark_config_path = Path(args.benchmark_config)
    if not benchmark_config_path.is_absolute():
        benchmark_config_path = PROJECT_ROOT / benchmark_config_path
    benchmark_runner_config = load_benchmark_runner_config(benchmark_config_path)
    user_overrides = benchmark_runner_config.get("overrides") or {}

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = DEFAULT_SEEDS

    # Parse model list:
    # 1) explicit --models
    # 2) if requested, benchmark_config.yaml "models"
    # 3) script defaults (includes DeepONet)
    if args.models:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    elif args.use_config_models:
        model_names = benchmark_runner_config.get("models") or list(DEFAULT_TIMING_MODELS)
    else:
        model_names = list(DEFAULT_TIMING_MODELS)

    unknown_models = [m for m in model_names if m not in MODEL_VARIANTS]
    if unknown_models:
        raise ValueError(f"Unknown model variants: {unknown_models}. Available: {sorted(MODEL_VARIANTS.keys())}")

    # Setup device
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Warmup epochs: {WARMUP_EPOCHS}")
    print(f"Measured epochs: {MEASURED_EPOCHS}")
    print(f"Models: {model_names}")
    print(f"Benchmarks: {list(BENCHMARKS.keys())}")
    print(f"Benchmark config: {benchmark_config_path}")
    print(f"User overrides active: {'yes' if user_overrides else 'no'}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "logs_timing"

    # Calculate total runs
    total_runs = len(model_names) * len(BENCHMARKS) * len(seeds)
    print(f"\nTotal runs: {total_runs}")
    print("=" * 60)

    # Run timing for all combinations
    all_results = []
    run_idx = 0

    for benchmark in BENCHMARKS:
        for model_name in model_names:
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] Running {model_name} on {benchmark} (seed={seed})...", end=" ", flush=True)

                try:
                    result = run_single_timing(
                        model_name,
                        benchmark,
                        seed,
                        device,
                        user_overrides=user_overrides,
                    )
                    if result:
                        all_results.append(result)
                        print(f"Done: {result.time_per_iter_ms:.3f} ms/iter")
                    else:
                        print("Skipped (unsupported)")
                except Exception as e:
                    print(f"Error: {e}")

                # Clear GPU memory
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    print("=" * 60)
    print(f"Completed {len(all_results)} timing runs")

    # Aggregate and save results
    if all_results:
        aggregated = aggregate_results(all_results)
        save_results(
            aggregated,
            output_dir,
            model_names=model_names,
            benchmark_config_path=benchmark_config_path,
        )

        # Print summary table
        print("\n" + "=" * 100)
        print("SUMMARY (time_per_iter_ms, mean +/- std)")
        print("=" * 100)
        # Header: models as columns
        header = f"{'Benchmark':<15}"
        for model in model_names:
            header += f"{model:>12}"
        print(header)
        print("-" * 100)

        lookup = {(r.model, r.benchmark): r for r in aggregated}
        for bench in BENCHMARKS:
            row = f"{bench:<15}"
            for model in model_names:
                r = lookup.get((model, bench))
                if r:
                    row += f"{r.time_per_iter_ms_mean:>12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
        print("=" * 100)


if __name__ == "__main__":
    main()
