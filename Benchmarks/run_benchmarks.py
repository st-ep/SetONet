#!/usr/bin/env python3
"""
Multi-Seed Benchmark Runner for SetONet and DeepONet.

This script orchestrates running multiple experiments across different seeds,
models, and benchmarks with parallel GPU execution.

Usage:
    python Benchmarks/run_benchmarks.py --config Benchmarks/benchmark_config.yaml
    python Benchmarks/run_benchmarks.py --dry-run  # Show jobs without executing
    python Benchmarks/run_benchmarks.py --seeds 0,1 --benchmarks heat_2d,darcy_1d
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# -----------------------------------------------------------------------------
# Script Mapping
# -----------------------------------------------------------------------------
# Maps benchmark keys to their respective script filenames

SETONET_SCRIPTS = {
    "heat_2d": "run_heat_2d.py",
    "elastic_2d": "run_elastic_2d.py",
    "darcy_1d": "run_darcy_1d.py",
    "chladni_2d": "run_chladni_2d.py",
    "1d": "run_1d.py",
    "concentration_2d": "run_consantration_2d.py",  # Note: typo in original filename
    "transport": "run_transoprt.py",  # Note: typo in original filename
    "dynamic_chladni": "run_dynamic_chladni.py",
}

DEEPONET_SCRIPTS = {
    "elastic_2d": "run_elastic_2d_don.py",
    "darcy_1d": "run_darcy_1d_don.py",
    "chladni_2d": "run_chladni_2d_don.py",
    "1d": "run_1d_don.py",
}

# -----------------------------------------------------------------------------
# Model Variants
# -----------------------------------------------------------------------------
# Maps model variant names to their base type and CLI overrides
# All setonet_* variants use SetONet scripts with different args

MODEL_VARIANTS = {
    # DeepONet baseline
    "deeponet": {
        "base": "deeponet",
        "overrides": {},
    },
    # SetONet with different aggregation types (standard branch head)
    "setonet_sum": {
        "base": "setonet",
        "overrides": {"son_aggregation": "sum"},
    },
    "setonet_mean": {
        "base": "setonet",
        "overrides": {"son_aggregation": "mean"},
    },
    "setonet_attention": {
        "base": "setonet",
        "overrides": {},  # Uses script defaults (attention aggregation)
    },
    # SetONet with different branch head types
    "setonet_petrov": {
        "base": "setonet",
        "overrides": {"son_branch_head_type": "petrov_attention"},
    },
    "setonet_galerkin": {
        "base": "setonet",
        "overrides": {"son_branch_head_type": "galerkin_pou"},
    },
    "setonet_quadrature": {
        "base": "setonet",
        "overrides": {"son_branch_head_type": "quadrature"},
    },
    "setonet_adaptive": {
        "base": "setonet",
        "overrides": {"son_branch_head_type": "adaptive_quadrature"},
    },
}

# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class Job:
    """Represents a single benchmark job to execute."""
    model: str  # Model variant name (e.g., "setonet_attention", "deeponet")
    benchmark: str  # e.g., "heat_2d", "darcy_1d"
    seed: int
    device: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def job_id(self) -> str:
        return f"{self.model}_{self.benchmark}_seed{self.seed}"
    
    @property
    def base_model(self) -> str:
        """Get the base model type ('setonet' or 'deeponet')."""
        variant = MODEL_VARIANTS.get(self.model, {})
        return variant.get("base", self.model)
    
    def get_script_path(self, benchmarks_dir: Path) -> Path:
        """Get the full path to the script for this job."""
        base = self.base_model
        if base == "setonet":
            script_name = SETONET_SCRIPTS.get(self.benchmark)
            script_dir = benchmarks_dir / "run_SetONet"
        else:  # deeponet
            script_name = DEEPONET_SCRIPTS.get(self.benchmark)
            script_dir = benchmarks_dir / "run_DeepONet"
        
        if script_name is None:
            raise ValueError(f"No script found for {self.model}/{self.benchmark}")
        
        return script_dir / script_name


@dataclass
class JobResult:
    """Result of a completed job."""
    job: Job
    success: bool
    log_dir: Optional[str] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set defaults
    config.setdefault('seeds', [0, 1, 2, 3, 4])
    config.setdefault('devices', ['cuda:0'])
    config.setdefault('models', {})
    config.setdefault('overrides', {})
    config.setdefault('continue_on_failure', True)
    config.setdefault('log_level', 'INFO')
    
    return config


def build_job_queue(config: Dict[str, Any]) -> List[Job]:
    """Build the queue of jobs from the configuration."""
    jobs = []
    seeds = config['seeds']
    devices = config['devices']
    overrides = config.get('overrides') or {}
    
    device_idx = 0
    
    for model_name, model_config in config['models'].items():
        if not model_config.get('enabled', False):
            continue
        
        # Get variant info (or default to treating model_name as base type)
        variant_info = MODEL_VARIANTS.get(model_name, {"base": model_name, "overrides": {}})
        base_model = variant_info["base"]
        variant_overrides = variant_info.get("overrides", {})
        
        benchmarks = model_config.get('benchmarks', [])
        
        for benchmark in benchmarks:
            # Validate benchmark exists for this base model type
            scripts = SETONET_SCRIPTS if base_model == "setonet" else DEEPONET_SCRIPTS
            if benchmark not in scripts:
                logging.warning(f"Skipping {model_name}/{benchmark}: no script available for base model '{base_model}'")
                continue
            
            for seed in seeds:
                # Collect overrides in order of priority:
                # 1. Variant-specific overrides (from MODEL_VARIANTS)
                # 2. General benchmark overrides (from config)
                # 3. Model-specific overrides (e.g., "setonet_attention_heat_2d")
                job_overrides = {}
                
                # 1. Variant-specific overrides
                job_overrides.update(variant_overrides)
                
                # 2. General benchmark overrides
                if benchmark in overrides:
                    job_overrides.update(overrides[benchmark])
                
                # 3. Model-specific overrides (e.g., "setonet_attention_heat_2d")
                model_benchmark_key = f"{model_name}_{benchmark}"
                if model_benchmark_key in overrides:
                    job_overrides.update(overrides[model_benchmark_key])
                
                # Assign device round-robin
                device = devices[device_idx % len(devices)]
                device_idx += 1
                
                jobs.append(Job(
                    model=model_name,
                    benchmark=benchmark,
                    seed=seed,
                    device=device,
                    overrides=job_overrides
                ))
    
    return jobs


# -----------------------------------------------------------------------------
# Job Execution
# -----------------------------------------------------------------------------

def run_single_job(job: Job, benchmarks_dir: Path, project_root: Path, 
                   run_output_dir: Path) -> JobResult:
    """Execute a single benchmark job."""
    start_time = time.time()
    
    try:
        script_path = job.get_script_path(benchmarks_dir)
        
        if not script_path.exists():
            return JobResult(
                job=job,
                success=False,
                error_message=f"Script not found: {script_path}",
                duration_seconds=time.time() - start_time
            )
        
        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--seed", str(job.seed),
            "--device", job.device,
        ]
        
        # Add overrides
        for key, value in job.overrides.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.append(f"--{key}")
                cmd.extend(str(v) for v in value)
            else:
                cmd.extend([f"--{key}", str(value)])
        
        # Log file for this job
        job_log_file = run_output_dir / f"{job.job_id}.log"
        
        logging.info(f"Starting: {job.job_id} on {job.device}")
        logging.debug(f"Command: {' '.join(cmd)}")
        
        # Execute
        with open(job_log_file, 'w') as log_f:
            log_f.write(f"Job: {job.job_id}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Started: {datetime.now().isoformat()}\n")
            log_f.write("=" * 80 + "\n\n")
            log_f.flush()
            
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            return JobResult(
                job=job,
                success=False,
                error_message=f"Process exited with code {result.returncode}",
                duration_seconds=duration
            )
        
        # Try to find the log directory and extract metrics
        log_dir, metrics = find_job_results(job, project_root)
        
        logging.info(f"Completed: {job.job_id} in {duration:.1f}s")
        
        return JobResult(
            job=job,
            success=True,
            log_dir=log_dir,
            duration_seconds=duration,
            metrics=metrics
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"Error in {job.job_id}: {e}")
        return JobResult(
            job=job,
            success=False,
            error_message=str(e),
            duration_seconds=duration
        )


def find_job_results(job: Job, project_root: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """Find the log directory and extract metrics for a completed job."""
    # Determine the log folder name pattern based on base model type
    base_model = job.base_model
    if base_model == "setonet":
        folder_patterns = [
            f"SetONet_{job.benchmark}",
            f"SetONet_{job.benchmark.replace('_', '')}",
        ]
    else:
        folder_patterns = [
            f"DeepONet_{job.benchmark}",
            f"DeepONet_{job.benchmark.replace('_', '')}",
        ]
    
    logs_dir = project_root / "logs"
    
    # Find the most recent matching log directory
    latest_dir = None
    latest_time = 0
    
    for pattern in folder_patterns:
        pattern_dir = logs_dir / pattern
        if pattern_dir.exists():
            for subdir in pattern_dir.iterdir():
                if subdir.is_dir():
                    # Check if this run has the right seed
                    config_file = subdir / "experiment_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file) as f:
                                config = json.load(f)
                            if config.get("seed") == job.seed:
                                mtime = config_file.stat().st_mtime
                                if mtime > latest_time:
                                    latest_time = mtime
                                    latest_dir = subdir
                        except (json.JSONDecodeError, KeyError):
                            continue
    
    if latest_dir is None:
        return None, {}
    
    # Extract metrics from experiment_config.json
    metrics = {}
    config_file = latest_dir / "experiment_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            test_results = config.get("test_results", {})
            if "relative_l2_error" in test_results:
                metrics["rel_l2_error"] = test_results["relative_l2_error"]
            if "mse_loss" in test_results:
                metrics["mse_loss"] = test_results["mse_loss"]
        except (json.JSONDecodeError, KeyError):
            pass
    
    return str(latest_dir), metrics


# -----------------------------------------------------------------------------
# Parallel Execution
# -----------------------------------------------------------------------------

def run_jobs_parallel(jobs: List[Job], config: Dict[str, Any], 
                      benchmarks_dir: Path, project_root: Path,
                      run_output_dir: Path) -> List[JobResult]:
    """Run jobs in parallel, one worker per GPU device."""
    devices = config['devices']
    n_workers = len(devices)
    continue_on_failure = config.get('continue_on_failure', True)
    
    results = []
    
    # Group jobs by device
    device_jobs: Dict[str, List[Job]] = {d: [] for d in devices}
    for job in jobs:
        device_jobs[job.device].append(job)
    
    logging.info(f"Running {len(jobs)} jobs across {n_workers} GPU(s)")
    for device, device_job_list in device_jobs.items():
        logging.info(f"  {device}: {len(device_job_list)} jobs")
    
    # Use ProcessPoolExecutor with n_workers
    # Each worker processes jobs assigned to its device sequentially
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_job = {}
        for job in jobs:
            future = executor.submit(
                run_single_job, job, benchmarks_dir, project_root, run_output_dir
            )
            future_to_job[future] = job
        
        # Collect results as they complete
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                
                if not result.success and not continue_on_failure:
                    logging.error(f"Job {result.job.job_id} failed and continue_on_failure=False. Stopping.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                    
            except Exception as e:
                logging.error(f"Job {job.job_id} raised exception: {e}")
                results.append(JobResult(
                    job=job,
                    success=False,
                    error_message=str(e)
                ))
    
    return results


# -----------------------------------------------------------------------------
# Result Aggregation
# -----------------------------------------------------------------------------

def aggregate_results(results: List[JobResult], output_dir: Path) -> None:
    """Aggregate results and save to CSV and JSON files."""
    import csv
    import statistics
    
    # Filter successful results with metrics
    successful = [r for r in results if r.success and r.metrics]
    failed = [r for r in results if not r.success]
    
    logging.info(f"Aggregating {len(successful)} successful results, {len(failed)} failed")
    
    # Write individual results CSV
    csv_path = output_dir / "results_individual.csv"
    fieldnames = ["model", "benchmark", "seed", "device", "success", 
                  "rel_l2_error", "mse_loss", "duration_seconds", "log_dir", "error"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = {
                "model": r.job.model,
                "benchmark": r.job.benchmark,
                "seed": r.job.seed,
                "device": r.job.device,
                "success": r.success,
                "rel_l2_error": r.metrics.get("rel_l2_error", ""),
                "mse_loss": r.metrics.get("mse_loss", ""),
                "duration_seconds": f"{r.duration_seconds:.1f}",
                "log_dir": r.log_dir or "",
                "error": r.error_message or ""
            }
            writer.writerow(row)
    
    logging.info(f"Individual results saved to: {csv_path}")
    
    # Compute aggregated statistics
    # Group by (model, benchmark)
    from collections import defaultdict
    grouped: Dict[Tuple[str, str], List[JobResult]] = defaultdict(list)
    for r in successful:
        key = (r.job.model, r.job.benchmark)
        grouped[key].append(r)
    
    # Write aggregated results CSV
    agg_csv_path = output_dir / "results_summary.csv"
    agg_fieldnames = ["model", "benchmark", "n_seeds", 
                      "rel_l2_mean", "rel_l2_std", "mse_mean", "mse_std"]
    
    aggregated_data = []
    
    with open(agg_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        
        for (model, benchmark), group_results in sorted(grouped.items()):
            rel_l2_values = [r.metrics["rel_l2_error"] for r in group_results 
                             if "rel_l2_error" in r.metrics]
            mse_values = [r.metrics["mse_loss"] for r in group_results 
                          if "mse_loss" in r.metrics]
            
            row = {
                "model": model,
                "benchmark": benchmark,
                "n_seeds": len(group_results),
                "rel_l2_mean": f"{statistics.mean(rel_l2_values):.6f}" if rel_l2_values else "",
                "rel_l2_std": f"{statistics.stdev(rel_l2_values):.6f}" if len(rel_l2_values) > 1 else "0.0",
                "mse_mean": f"{statistics.mean(mse_values):.6e}" if mse_values else "",
                "mse_std": f"{statistics.stdev(mse_values):.6e}" if len(mse_values) > 1 else "0.0",
            }
            writer.writerow(row)
            
            aggregated_data.append({
                "model": model,
                "benchmark": benchmark,
                "n_seeds": len(group_results),
                "seeds": [r.job.seed for r in group_results],
                "rel_l2": {
                    "mean": statistics.mean(rel_l2_values) if rel_l2_values else None,
                    "std": statistics.stdev(rel_l2_values) if len(rel_l2_values) > 1 else 0.0,
                    "values": rel_l2_values
                },
                "mse": {
                    "mean": statistics.mean(mse_values) if mse_values else None,
                    "std": statistics.stdev(mse_values) if len(mse_values) > 1 else 0.0,
                    "values": mse_values
                }
            })
    
    logging.info(f"Aggregated results saved to: {agg_csv_path}")
    
    # Write JSON summary
    json_path = output_dir / "results_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_jobs": len(results),
        "successful_jobs": len(successful),
        "failed_jobs": len(failed),
        "aggregated": aggregated_data,
        "individual": [
            {
                "model": r.job.model,
                "benchmark": r.job.benchmark,
                "seed": r.job.seed,
                "device": r.job.device,
                "success": r.success,
                "metrics": r.metrics,
                "duration_seconds": r.duration_seconds,
                "log_dir": r.log_dir,
                "error": r.error_message
            }
            for r in results
        ]
    }
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"JSON summary saved to: {json_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total jobs: {len(results)} | Successful: {len(successful)} | Failed: {len(failed)}")
    print("-" * 80)
    print(f"{'Model':<12} {'Benchmark':<18} {'Seeds':<8} {'Rel L2 Error':<25} {'MSE Loss':<25}")
    print("-" * 80)
    
    for data in aggregated_data:
        rel_l2_str = ""
        if data["rel_l2"]["mean"] is not None:
            rel_l2_str = f"{data['rel_l2']['mean']:.6f} ± {data['rel_l2']['std']:.6f}"
        
        mse_str = ""
        if data["mse"]["mean"] is not None:
            mse_str = f"{data['mse']['mean']:.2e} ± {data['mse']['std']:.2e}"
        
        print(f"{data['model']:<12} {data['benchmark']:<18} {data['n_seeds']:<8} {rel_l2_str:<25} {mse_str:<25}")
    
    print("=" * 80)
    
    if failed:
        print("\nFailed jobs:")
        for r in failed:
            print(f"  - {r.job.job_id}: {r.error_message}")


# -----------------------------------------------------------------------------
# CLI Argument Parsing
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed benchmark runner for SetONet and DeepONet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Benchmarks/run_benchmarks.py --config Benchmarks/benchmark_config.yaml
  python Benchmarks/run_benchmarks.py --dry-run
  python Benchmarks/run_benchmarks.py --seeds 0,1,2 --benchmarks heat_2d,darcy_1d
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="Benchmarks/benchmark_config.yaml",
        help="Path to YAML configuration file (default: Benchmarks/benchmark_config.yaml)"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show jobs that would be run without executing them"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        help="Override seeds (comma-separated, e.g., '0,1,2')"
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        help="Override benchmarks (comma-separated, e.g., 'heat_2d,darcy_1d')"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        choices=["setonet", "deeponet", "both"],
        help="Override which models to run"
    )
    
    parser.add_argument(
        "--devices",
        type=str,
        help="Override devices (comma-separated, e.g., 'cuda:0,cuda:1')"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory for results"
    )
    
    parser.add_argument(
        "--generate-configs",
        action="store_true",
        help="Generate benchmark_configs.json with full configurations for all (model, benchmark) combinations"
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to the configuration."""
    if args.seeds:
        config['seeds'] = [int(s.strip()) for s in args.seeds.split(',')]
    
    if args.devices:
        config['devices'] = [d.strip() for d in args.devices.split(',')]
    
    # Apply --models filter FIRST (so --benchmarks applies to newly enabled models)
    if args.models:
        # Handle model filtering by base type or specific variant
        for model_name in config['models']:
            variant_info = MODEL_VARIANTS.get(model_name, {"base": model_name})
            base_model = variant_info.get("base", model_name)
            
            if args.models == "setonet":
                # Enable all setonet variants, disable deeponet
                config['models'][model_name]['enabled'] = (base_model == "setonet")
            elif args.models == "deeponet":
                # Enable deeponet, disable all setonet variants
                config['models'][model_name]['enabled'] = (base_model == "deeponet")
            # "both" keeps existing enabled states
    
    # Apply --benchmarks filter AFTER --models (to affect newly enabled models)
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(',')]
        # Apply to all enabled models
        for model_name in config['models']:
            if config['models'][model_name].get('enabled', False):
                config['models'][model_name]['benchmarks'] = benchmarks


# -----------------------------------------------------------------------------
# Config Generation for Reproducibility
# -----------------------------------------------------------------------------

def extract_script_defaults(script_path: Path) -> Dict[str, Any]:
    """
    Extract default argument values from a benchmark script's parse_arguments() function.
    
    Args:
        script_path: Path to the benchmark script (e.g., run_heat_2d.py)
    
    Returns:
        Dict of argument names to their default values
    """
    import importlib.util
    import io
    
    if not script_path.exists():
        return {}
    
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("benchmark_script", script_path)
        if spec is None or spec.loader is None:
            return {}
        
        module = importlib.util.module_from_spec(spec)
        
        # We need to handle the sys.path manipulation that scripts do
        import sys
        original_argv = sys.argv
        original_stderr = sys.stderr
        sys.argv = [str(script_path)]  # Fake argv to prevent argparse errors
        sys.stderr = io.StringIO()  # Suppress stderr output
        
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass  # Some scripts might call sys.exit on import
        finally:
            sys.argv = original_argv
            sys.stderr = original_stderr
        
        # Get the parse_arguments function
        if not hasattr(module, 'parse_arguments'):
            return {}
        
        # Create a parser and extract defaults
        # We need to intercept the parser before parse_args() is called
        import argparse
        
        # Monkey-patch parse_args to capture the parser
        original_parse_args = argparse.ArgumentParser.parse_args
        captured_parser = None
        
        def capture_parser(self, args=None, namespace=None):
            nonlocal captured_parser
            captured_parser = self
            # Return defaults without actually parsing
            return self.parse_args.__wrapped__(self, args=[], namespace=namespace)
        
        capture_parser.__wrapped__ = original_parse_args
        argparse.ArgumentParser.parse_args = capture_parser
        
        # Suppress stderr during parse_arguments call
        sys.stderr = io.StringIO()
        try:
            # Call parse_arguments to trigger parser creation
            module.parse_arguments()
        except (SystemExit, Exception):
            pass
        finally:
            argparse.ArgumentParser.parse_args = original_parse_args
            sys.stderr = original_stderr
        
        if captured_parser is None:
            return {}
        
        # Extract defaults from the parser
        defaults = {}
        for action in captured_parser._actions:
            if action.dest != 'help' and action.default is not None:
                # Convert non-serializable types
                value = action.default
                if isinstance(value, type):
                    value = str(value.__name__)
                elif hasattr(value, '__name__'):
                    value = value.__name__
                defaults[action.dest] = value
        
        return defaults
        
    except Exception as e:
        logging.warning(f"Could not extract defaults from {script_path}: {e}")
        return {}


def generate_all_configs(benchmarks_dir: Path) -> Dict[str, Any]:
    """
    Generate complete configurations for all (model variant, benchmark) combinations.
    
    Returns:
        Dict with structure: {model_variant: {benchmark: {param: value, ...}, ...}, ...}
    """
    configs = {}
    
    # Process all model variants
    for model_name, variant_info in MODEL_VARIANTS.items():
        base_model = variant_info["base"]
        variant_overrides = variant_info.get("overrides", {})
        
        # Get the appropriate script mapping
        scripts = SETONET_SCRIPTS if base_model == "setonet" else DEEPONET_SCRIPTS
        script_dir = benchmarks_dir / ("run_SetONet" if base_model == "setonet" else "run_DeepONet")
        
        model_configs = {}
        
        for benchmark, script_name in scripts.items():
            script_path = script_dir / script_name
            
            # Extract defaults from the script
            defaults = extract_script_defaults(script_path)
            
            if not defaults:
                logging.warning(f"No defaults extracted for {model_name}/{benchmark}")
                continue
            
            # Apply variant-specific overrides
            config = defaults.copy()
            config.update(variant_overrides)
            
            # Add metadata
            config["_script"] = script_name
            config["_base_model"] = base_model
            
            model_configs[benchmark] = config
        
        if model_configs:
            configs[model_name] = model_configs
    
    return configs


def save_all_configs(benchmarks_dir: Path, output_path: Optional[Path] = None) -> Path:
    """
    Generate and save all configurations to a JSON file.
    
    Args:
        benchmarks_dir: Path to the Benchmarks directory
        output_path: Optional custom output path (defaults to benchmarks_dir/benchmark_configs.json)
    
    Returns:
        Path to the saved JSON file
    """
    configs = generate_all_configs(benchmarks_dir)
    
    output = {
        "generated_at": datetime.now().isoformat(),
        "description": "Complete configurations for all (model variant, benchmark) combinations. "
                       "These are the effective parameters that will be used when running benchmarks.",
        "model_variants": list(MODEL_VARIANTS.keys()),
        "configs": configs
    }
    
    if output_path is None:
        output_path = benchmarks_dir / "benchmark_configs.json"
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    return output_path


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine project root and paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    benchmarks_dir = script_dir
    
    # Handle --generate-configs mode
    if args.generate_configs:
        print("Generating complete benchmark configurations...")
        output_path = save_all_configs(benchmarks_dir)
        print(f"\nSaved configurations to: {output_path}")
        print("\nThis file contains the full effective configuration for every")
        print("(model variant, benchmark) combination for reproducibility.")
        return
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(str(config_path))
    
    # Apply CLI overrides
    apply_cli_overrides(config, args)
    
    # Setup logging
    log_level = getattr(logging, config.get('log_level', 'INFO').upper())
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Build job queue
    jobs = build_job_queue(config)
    
    if not jobs:
        print("No jobs to run. Check your configuration.")
        sys.exit(0)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        run_output_dir = Path(args.output_dir)
    else:
        run_output_dir = project_root / "logs" / "benchmark_runs" / timestamp
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config to output dir
    import shutil
    shutil.copy(config_path, run_output_dir / "benchmark_config.yaml")
    
    # Dry run mode
    if args.dry_run:
        print(f"\nDRY RUN - Would execute {len(jobs)} jobs:\n")
        print(f"{'#':<4} {'Job ID':<40} {'Device':<12} {'Overrides'}")
        print("-" * 80)
        for i, job in enumerate(jobs, 1):
            overrides_str = str(job.overrides) if job.overrides else "(defaults)"
            print(f"{i:<4} {job.job_id:<40} {job.device:<12} {overrides_str}")
        print()
        print(f"Output would be saved to: {run_output_dir}")
        return
    
    # Run jobs
    logging.info(f"Starting benchmark run: {len(jobs)} jobs")
    logging.info(f"Output directory: {run_output_dir}")
    logging.info(f"Configuration: {config_path}")
    
    start_time = time.time()
    results = run_jobs_parallel(jobs, config, benchmarks_dir, project_root, run_output_dir)
    total_time = time.time() - start_time
    
    # Aggregate and save results
    aggregate_results(results, run_output_dir)
    
    logging.info(f"Total execution time: {total_time / 60:.1f} minutes")
    logging.info(f"Results saved to: {run_output_dir}")


if __name__ == "__main__":
    main()

