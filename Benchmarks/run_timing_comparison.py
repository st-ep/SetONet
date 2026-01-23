#!/usr/bin/env python3
"""
Training Speed Comparison Script for SetONet, DeepONet, and VIDON.

Measures training throughput (samples/second) and time per iteration
across all model variants on darcy_1d and elastic_2d benchmarks.

Usage:
    python Benchmarks/run_timing_comparison.py
    python Benchmarks/run_timing_comparison.py --device cuda:0
    python Benchmarks/run_timing_comparison.py --seeds 0,1,2
"""
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
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


# =============================================================================
# Configuration
# =============================================================================

WARMUP_EPOCHS = 100
MEASURED_EPOCHS = 1000
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_DEVICE = "cuda:0"

# Model variants to benchmark (matches benchmark_utils.py exactly)
MODEL_VARIANTS = {
    "deeponet": {"base": "deeponet", "overrides": {}},
    "vidon": {"base": "vidon", "overrides": {}},
    "setonet_sum": {"base": "setonet", "overrides": {"son_aggregation": "sum"},
        "benchmark_overrides": {
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
    "setonet_mean": {"base": "setonet", "overrides": {"son_aggregation": "mean"},
        "benchmark_overrides": {
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
    "setonet_attention": {"base": "setonet", "overrides": {},
        "benchmark_overrides": {
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
    "setonet_quadrature": {"base": "setonet", "overrides": {"son_branch_head_type": "quadrature"},
        "benchmark_overrides": {
            ("1d_", "elastic_", "darcy_", "burgers_"): {"son_rho_hidden": 200},
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
}

# Benchmark configurations
BENCHMARKS = {
    "elastic_2d": {
        "data_path": "Data/elastic_2d_data/elastic_dataset",
        "input_size_src": 2,
        "output_size_src": 1,
        "input_size_tgt": 2,
        "output_size_tgt": 1,
        "config_file": "setonet_elastic2d.yaml",
        "vidon_config_file": "vidon_elastic2d.yaml",
        "deeponet_config_file": "deeponet_elastic2d.yaml",
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

def load_yaml_config(config_file: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = SCRIPT_DIR / "configs" / config_file
    if not config_path.exists():
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_benchmark_overrides(variant: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    """Get benchmark-specific overrides for a model variant (matches benchmark_utils.py)."""
    bench_overrides = variant.get("benchmark_overrides", {})
    for patterns, overrides in bench_overrides.items():
        # patterns is a tuple of prefixes/names to match
        for pattern in patterns:
            if benchmark.startswith(pattern) or benchmark == pattern:
                return overrides
    return {}


def get_model_config(model_name: str, benchmark: str) -> Dict[str, Any]:
    """Get merged config for a model/benchmark pair (matches benchmark_utils.py logic)."""
    variant = MODEL_VARIANTS[model_name]
    base_model = variant["base"]
    bench_info = BENCHMARKS[benchmark]

    # Load base config from YAML
    if base_model == "setonet":
        config = load_yaml_config(bench_info["config_file"])
    elif base_model == "vidon":
        config = load_yaml_config(bench_info["vidon_config_file"])
    elif base_model == "deeponet":
        if bench_info["deeponet_config_file"] is None:
            return None  # DeepONet not supported for this benchmark
        config = load_yaml_config(bench_info["deeponet_config_file"])
    else:
        config = {}

    # Apply variant overrides (e.g., son_aggregation, son_branch_head_type)
    config.update(variant.get("overrides", {}))

    # Apply benchmark-specific overrides (e.g., son_rho_hidden for darcy_1d)
    bench_specific = get_benchmark_overrides(variant, benchmark)
    config.update(bench_specific)

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
        pos_encoding_type=config.get("pos_encoding_type", "sinusoidal"),
        pos_encoding_dim=config.get("pos_encoding_dim", 64),
        pos_encoding_max_freq=config.get("pos_encoding_max_freq", 0.1),
        aggregation_type=config.get("son_aggregation", "attention"),
        use_positional_encoding=(config.get("pos_encoding_type", "sinusoidal") != "skip"),
        attention_n_tokens=1,
        branch_head_type=config.get("son_branch_head_type", "standard"),
        pg_dk=config.get("son_pg_dk"),
        pg_dv=config.get("son_pg_dv"),
        pg_use_logw=not config.get("son_pg_no_logw", False),
        galerkin_dk=config.get("son_galerkin_dk"),
        galerkin_dv=config.get("son_galerkin_dv"),
        galerkin_normalize=config.get("son_galerkin_normalize", "total"),
        galerkin_learn_temperature=config.get("son_galerkin_learn_temperature", False),
        quad_dk=config.get("son_quad_dk"),
        quad_dv=config.get("son_quad_dv"),
        adapt_quad_rank=config.get("son_adapt_quad_rank", 4),
        adapt_quad_hidden=config.get("son_adapt_quad_hidden", 64),
        adapt_quad_scale=config.get("son_adapt_quad_scale", 0.1),
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
    ).to(device)
    return model


def create_model(model_name: str, benchmark: str, device: torch.device) -> Optional[nn.Module]:
    """Create model for given variant and benchmark."""
    config = get_model_config(model_name, benchmark)
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

def load_elastic_dataset(device: torch.device, batch_size: int):
    """Load Elastic 2D dataset and return data generator."""
    from Data.elastic_2d_data.elastic_2d_dataset import load_elastic_dataset as _load_elastic

    data_path = PROJECT_ROOT / BENCHMARKS["elastic_2d"]["data_path"]
    _, elastic_dataset = _load_elastic(str(data_path), batch_size=batch_size, device=device)

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
) -> Optional[TimingResult]:
    """Run timing for a single (model, benchmark, seed) combination."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Get config and batch size
    config = get_model_config(model_name, benchmark)
    if config is None:
        return None

    batch_size = config.get("batch_size", 64)

    # Load dataset FIRST (this sets sensor_size for elastic_2d, needed by DeepONet)
    loader_fn = get_dataset_loader(benchmark)
    dataset, _ = loader_fn(device, batch_size)

    # Create model (after dataset so sensor_size is available)
    model = create_model(model_name, benchmark, device)
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
            "models": list(MODEL_VARIANTS.keys()),
            "benchmarks": list(BENCHMARKS.keys()),
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
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="CUDA device (default: cuda:1)")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds (default: 0,1,2,3,4)")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory (default: logs_timing/)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seeds = DEFAULT_SEEDS

    # Setup device
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Warmup epochs: {WARMUP_EPOCHS}")
    print(f"Measured epochs: {MEASURED_EPOCHS}")
    print(f"Models: {list(MODEL_VARIANTS.keys())}")
    print(f"Benchmarks: {list(BENCHMARKS.keys())}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "logs_timing"

    # Calculate total runs
    total_runs = len(MODEL_VARIANTS) * len(BENCHMARKS) * len(seeds)
    print(f"\nTotal runs: {total_runs}")
    print("=" * 60)

    # Run timing for all combinations
    all_results = []
    run_idx = 0

    for benchmark in BENCHMARKS:
        for model_name in MODEL_VARIANTS:
            for seed in seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] Running {model_name} on {benchmark} (seed={seed})...", end=" ", flush=True)

                try:
                    result = run_single_timing(model_name, benchmark, seed, device)
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
        save_results(aggregated, output_dir)

        # Print summary table
        print("\n" + "=" * 100)
        print("SUMMARY (time_per_iter_ms, mean +/- std)")
        print("=" * 100)
        # Header: models as columns
        header = f"{'Benchmark':<15}"
        for model in MODEL_VARIANTS:
            header += f"{model:>12}"
        print(header)
        print("-" * 100)

        lookup = {(r.model, r.benchmark): r for r in aggregated}
        for bench in BENCHMARKS:
            row = f"{bench:<15}"
            for model in MODEL_VARIANTS:
                r = lookup.get((model, bench))
                if r:
                    row += f"{r.time_per_iter_ms_mean:>12.3f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
        print("=" * 100)


if __name__ == "__main__":
    main()
