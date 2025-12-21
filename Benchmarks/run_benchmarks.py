#!/usr/bin/env python3
"""
Multi-Seed Benchmark Runner for SetONet and DeepONet.

Automatically generates all_configs.json and param_table.csv before running.

Usage:
    python Benchmarks/run_benchmarks.py --config Benchmarks/benchmark_config.yaml
    python Benchmarks/run_benchmarks.py --dry-run
    python Benchmarks/run_benchmarks.py --seeds 0,1,2 --benchmarks heat_2d,darcy_1d
"""
import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from benchmark_utils import (
    DEEPONET_SCRIPTS, MODEL_VARIANTS, SETONET_SCRIPTS,
    Job, aggregate_results, generate_all_configs, generate_param_table,
    load_benchmark_config, run_jobs_parallel,
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


def save_configs_and_param_table(script_dir: Path, results_dir: Path) -> None:
    """Generate and save all_configs.json and param_table.csv."""
    # Save all configs
    configs = generate_all_configs(script_dir)
    config_path = results_dir / "all_configs.json"
    with open(config_path, 'w') as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                  "description": "Complete configs for all (model, benchmark) pairs",
                  "model_variants": list(MODEL_VARIANTS.keys()), "configs": configs}, f, indent=2)
    logging.info(f"Saved configs to: {config_path}")

    # Save param table
    param_table = generate_param_table(script_dir)
    all_benchmarks = sorted(set(b for m in param_table.values() for b in m.keys()))
    csv_path = results_dir / "param_table.csv"
    with open(csv_path, 'w') as f:
        f.write("model," + ",".join(all_benchmarks) + "\n")
        for model_name in MODEL_VARIANTS.keys():
            if model_name not in param_table:
                continue
            f.write(",".join([model_name] + [str(param_table[model_name].get(b, "")) for b in all_benchmarks]) + "\n")
    logging.info(f"Saved param table to: {csv_path}")


def main():
    args = parse_args()
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    logs_all_dir = project_root / "logs_all"
    results_dir = logs_all_dir / "_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and validate config
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
    if args.models and args.models != "both":
        config['models'] = [m for m in config['models']
                           if MODEL_VARIANTS.get(m, {"base": m}).get("base") == args.models]

    logging.basicConfig(level=getattr(logging, config.get('log_level', 'INFO').upper()),
                       format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    jobs = build_job_queue(config, script_dir)
    if not jobs:
        print("No jobs to run.")
        sys.exit(0)

    # Use custom output dir or default logs_all
    if args.output_dir:
        logs_all_dir = Path(args.output_dir)
        results_dir = logs_all_dir / "_results"
        results_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, results_dir / "benchmark_config.yaml")
    save_configs_and_param_table(script_dir, results_dir)

    if args.dry_run:
        print(f"\nDRY RUN - {len(jobs)} jobs:\n")
        print(f"{'#':<4} {'Job ID':<40} {'Device':<12} Log Dir")
        print("-" * 90)
        for i, job in enumerate(jobs, 1):
            job_log = f"logs_all/{job.benchmark}/{job.model}/seed_{job.seed}/"
            print(f"{i:<4} {job.job_id:<40} {job.device:<12} {job_log}")
        print(f"\nOutput structure: logs_all/<benchmark>/<model>/seed_<X>/")
        print(f"Results: {results_dir}")
        return

    logging.info(f"Starting {len(jobs)} jobs, output: {logs_all_dir}")
    start = time.time()
    results = run_jobs_parallel(jobs, config, script_dir, project_root, logs_all_dir)
    aggregate_results(results, results_dir)
    logging.info(f"Total time: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
