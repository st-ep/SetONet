#!/usr/bin/env python3
"""
Multi-Seed Benchmark Runner for SetONet and DeepONet.

Usage:
    python Benchmarks/run_benchmarks.py --config Benchmarks/benchmark_config.yaml
    python Benchmarks/run_benchmarks.py --dry-run
    python Benchmarks/run_benchmarks.py --seeds 0,1,2 --benchmarks heat_2d,darcy_1d
"""
import argparse
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from benchmark_utils import (
    BENCHMARK_CONFIG_MAP, DEEPONET_SCRIPTS, MODEL_VARIANTS, SETONET_SCRIPTS,
    Job, aggregate_results, run_jobs_parallel,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration with defaults."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    defaults = {'seeds': [0,1,2,3,4], 'devices': ['cuda:0'], 'models': [], 'benchmarks': [],
                'overrides': {}, 'continue_on_failure': True, 'log_level': 'INFO'}
    for k, v in defaults.items():
        config.setdefault(k, v)
    return config


def load_benchmark_config(benchmarks_dir: Path, base_model: str, benchmark: str) -> Dict[str, Any]:
    """Load hyperparameter config for a (base_model, benchmark) pair."""
    config_filename = BENCHMARK_CONFIG_MAP.get(base_model, {}).get(benchmark)
    if not config_filename:
        return {}
    config_path = benchmarks_dir / "configs" / config_filename
    if not config_path.exists():
        logging.warning(f"Config not found: {config_path}")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def build_job_queue(config: Dict[str, Any], benchmarks_dir: Path) -> List[Job]:
    """Build job queue from configuration."""
    jobs = []
    seeds, devices = config['seeds'], config['devices']
    user_overrides = config.get('overrides') or {}
    device_idx = 0

    for model_name in config['models']:
        variant = MODEL_VARIANTS.get(model_name, {"base": model_name, "overrides": {}})
        base_model, variant_overrides = variant["base"], variant.get("overrides", {})
        scripts = SETONET_SCRIPTS if base_model == "setonet" else DEEPONET_SCRIPTS

        for benchmark in config['benchmarks']:
            if benchmark not in scripts:
                logging.warning(f"Skipping {model_name}/{benchmark}: no script for '{base_model}'")
                continue

            bench_config = load_benchmark_config(benchmarks_dir, base_model, benchmark)
            for seed in seeds:
                overrides = {**bench_config, **variant_overrides}
                if benchmark in user_overrides:
                    overrides.update(user_overrides[benchmark])
                if f"{model_name}_{benchmark}" in user_overrides:
                    overrides.update(user_overrides[f"{model_name}_{benchmark}"])

                jobs.append(Job(model_name, benchmark, seed, devices[device_idx % len(devices)], overrides))
                device_idx += 1
    return jobs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Multi-seed benchmark runner")
    p.add_argument("--config", "-c", default="Benchmarks/benchmark_config.yaml", help="YAML config path")
    p.add_argument("--dry-run", "-n", action="store_true", help="Show jobs without executing")
    p.add_argument("--seeds", help="Override seeds (comma-separated)")
    p.add_argument("--benchmarks", help="Override benchmarks (comma-separated)")
    p.add_argument("--models", choices=["setonet", "deeponet", "both"], help="Filter models")
    p.add_argument("--devices", help="Override devices (comma-separated)")
    p.add_argument("--output-dir", "-o", help="Override output directory")
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Apply CLI overrides
    if args.seeds:
        config['seeds'] = [int(s.strip()) for s in args.seeds.split(',')]
    if args.devices:
        config['devices'] = [d.strip() for d in args.devices.split(',')]
    if args.benchmarks:
        config['benchmarks'] = [b.strip() for b in args.benchmarks.split(',')]
    if args.models:
        base_filter = args.models if args.models != "both" else None
        if base_filter:
            config['models'] = [m for m in config['models']
                               if MODEL_VARIANTS.get(m, {"base": m}).get("base") == base_filter]

    logging.basicConfig(level=getattr(logging, config.get('log_level', 'INFO').upper()),
                       format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    jobs = build_job_queue(config, script_dir)
    if not jobs:
        print("No jobs to run.")
        sys.exit(0)

    run_output_dir = Path(args.output_dir) if args.output_dir else \
                     project_root / "logs" / "benchmark_runs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_output_dir / "benchmark_config.yaml")

    if args.dry_run:
        print(f"\nDRY RUN - {len(jobs)} jobs:\n")
        print(f"{'#':<4} {'Job ID':<40} {'Device':<12} Overrides")
        print("-" * 80)
        for i, job in enumerate(jobs, 1):
            print(f"{i:<4} {job.job_id:<40} {job.device:<12} {job.overrides or '(defaults)'}")
        print(f"\nOutput: {run_output_dir}")
        return

    logging.info(f"Starting {len(jobs)} jobs, output: {run_output_dir}")
    start = time.time()
    results = run_jobs_parallel(jobs, config, script_dir, project_root, run_output_dir)
    aggregate_results(results, run_output_dir)
    logging.info(f"Total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
