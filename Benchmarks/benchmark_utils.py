"""Utility classes and functions for benchmark runner."""
import csv
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Script mappings
SETONET_SCRIPTS = {
    "heat_2d_P30": "run_heat_2d.py", "heat_2d_P10": "run_heat_2d.py",
    "concentration_2d": "run_consantration_2d.py", "transport": "run_transport.py",
    "diffraction_2d": "run_diffraction_2d.py",
    "elastic_2d": "run_elastic_2d.py", "elastic_2d_robust_train": "run_elastic_2d.py", "elastic_2d_robust_eval": "run_elastic_2d.py",
    "darcy_1d": "run_darcy_1d.py", "darcy_1d_robust_train": "run_darcy_1d.py", "darcy_1d_robust_eval": "run_darcy_1d.py",
    "burgers_1d": "run_burgers_1d.py", "burgers_1d_robust_train": "run_burgers_1d.py", "burgers_1d_robust_eval": "run_burgers_1d.py",
    "1d_integral": "run_1d.py", "1d_integral_varsens": "run_1d.py", "1d_integral_robust": "run_1d.py",
    "1d_derivative": "run_1d.py", "1d_derivative_varsens": "run_1d.py", "1d_derivative_robust": "run_1d.py",
}
DEEPONET_SCRIPTS = {
    "elastic_2d": "run_elastic_2d_don.py", "elastic_2d_robust_train": "run_elastic_2d_don.py", "elastic_2d_robust_eval": "run_elastic_2d_don.py",
    "darcy_1d": "run_darcy_1d_don.py", "darcy_1d_robust_train": "run_darcy_1d_don.py", "darcy_1d_robust_eval": "run_darcy_1d_don.py",
    "burgers_1d": "run_burgers_1d_don.py", "burgers_1d_robust_train": "run_burgers_1d_don.py", "burgers_1d_robust_eval": "run_burgers_1d_don.py",
    "1d_integral": "run_1d_don.py", "1d_integral_varsens": "run_1d_don.py", "1d_integral_robust": "run_1d_don.py",
    "1d_derivative": "run_1d_don.py", "1d_derivative_varsens": "run_1d_don.py", "1d_derivative_robust": "run_1d_don.py",
}
VIDON_SCRIPTS = {
    "heat_2d_P30": "run_heat_2d_vidon.py", "heat_2d_P10": "run_heat_2d_vidon.py",
    "concentration_2d": "run_consantration_2d_vidon.py", "transport": "run_transport_vidon.py",
    "diffraction_2d": "run_diffraction_2d_vidon.py",
    "elastic_2d": "run_elastic_2d_vidon.py", "elastic_2d_robust_train": "run_elastic_2d_vidon.py", "elastic_2d_robust_eval": "run_elastic_2d_vidon.py",
    "darcy_1d": "run_darcy_1d_vidon.py", "darcy_1d_robust_train": "run_darcy_1d_vidon.py", "darcy_1d_robust_eval": "run_darcy_1d_vidon.py",
    "burgers_1d": "run_burgers_1d_vidon.py", "burgers_1d_robust_train": "run_burgers_1d_vidon.py", "burgers_1d_robust_eval": "run_burgers_1d_vidon.py",
    "1d_integral": "run_1d_vidon.py", "1d_integral_varsens": "run_1d_vidon.py", "1d_integral_robust": "run_1d_vidon.py",
    "1d_derivative": "run_1d_vidon.py", "1d_derivative_varsens": "run_1d_vidon.py", "1d_derivative_robust": "run_1d_vidon.py",
}

# Maps (base_model, benchmark) to config file
BENCHMARK_CONFIG_MAP = {
    "setonet": {
        "1d_integral": "setonet_1d.yaml", "1d_integral_varsens": "setonet_1d.yaml", "1d_integral_robust": "setonet_1d.yaml",
        "1d_derivative": "setonet_1d.yaml", "1d_derivative_varsens": "setonet_1d.yaml", "1d_derivative_robust": "setonet_1d.yaml",
        "darcy_1d": "setonet_1d.yaml", "darcy_1d_robust_train": "setonet_1d.yaml", "darcy_1d_robust_eval": "setonet_1d.yaml",
        "burgers_1d": "setonet_1d.yaml", "burgers_1d_robust_train": "setonet_1d.yaml", "burgers_1d_robust_eval": "setonet_1d.yaml",
        "heat_2d_P30": "setonet_heat2d.yaml", "heat_2d_P10": "setonet_heat2d.yaml", "diffraction_2d": "setonet_heat2d.yaml",
        "concentration_2d": "setonet_heat2d.yaml", "transport": "setonet_heat2d.yaml",
        "elastic_2d": "setonet_elastic2d.yaml", "elastic_2d_robust_train": "setonet_elastic2d.yaml", "elastic_2d_robust_eval": "setonet_elastic2d.yaml",
    },
    "deeponet": {
        "1d_integral": "deeponet_1d.yaml", "1d_integral_varsens": "deeponet_1d.yaml", "1d_integral_robust": "deeponet_1d.yaml",
        "1d_derivative": "deeponet_1d.yaml", "1d_derivative_varsens": "deeponet_1d.yaml", "1d_derivative_robust": "deeponet_1d.yaml",
        "darcy_1d": "deeponet_1d.yaml", "darcy_1d_robust_train": "deeponet_1d.yaml", "darcy_1d_robust_eval": "deeponet_1d.yaml",
        "burgers_1d": "deeponet_1d.yaml", "burgers_1d_robust_train": "deeponet_1d.yaml", "burgers_1d_robust_eval": "deeponet_1d.yaml",
        "elastic_2d": "deeponet_elastic2d.yaml", "elastic_2d_robust_train": "deeponet_elastic2d.yaml", "elastic_2d_robust_eval": "deeponet_elastic2d.yaml",
    },
    "vidon": {
        "1d_integral": "vidon_1d.yaml", "1d_integral_varsens": "vidon_1d.yaml", "1d_integral_robust": "vidon_1d.yaml",
        "1d_derivative": "vidon_1d.yaml", "1d_derivative_varsens": "vidon_1d.yaml", "1d_derivative_robust": "vidon_1d.yaml",
        "darcy_1d": "vidon_1d.yaml", "darcy_1d_robust_train": "vidon_1d.yaml", "darcy_1d_robust_eval": "vidon_1d.yaml",
        "burgers_1d": "vidon_1d.yaml", "burgers_1d_robust_train": "vidon_1d.yaml", "burgers_1d_robust_eval": "vidon_1d.yaml",
        "heat_2d_P30": "vidon_heat2d.yaml", "heat_2d_P10": "vidon_heat2d.yaml", "diffraction_2d": "vidon_heat2d.yaml",
        "concentration_2d": "vidon_heat2d.yaml", "transport": "vidon_heat2d.yaml",
        "elastic_2d": "vidon_elastic2d.yaml", "elastic_2d_robust_train": "vidon_elastic2d.yaml", "elastic_2d_robust_eval": "vidon_elastic2d.yaml",
    },
}

# Model variant definitions
# benchmark_overrides: dict mapping benchmark pattern tuples to additional overrides
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
    "setonet_petrov": {"base": "setonet", "overrides": {"son_branch_head_type": "petrov_attention"},
        "benchmark_overrides": {
            ("1d_", "elastic_", "darcy_", "burgers_"): {"son_rho_hidden": 200},
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
    "setonet_galerkin": {"base": "setonet", "overrides": {"son_branch_head_type": "galerkin_pou"},
        "benchmark_overrides": {
            ("1d_", "elastic_", "darcy_", "burgers_"): {"son_rho_hidden": 200},
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
    "setonet_quadrature": {"base": "setonet", "overrides": {"son_branch_head_type": "quadrature"},
        "benchmark_overrides": {
            ("1d_", "elastic_", "darcy_", "burgers_"): {"son_rho_hidden": 200},
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
    "setonet_adaptive": {"base": "setonet", "overrides": {"son_branch_head_type": "adaptive_quadrature"},
        "benchmark_overrides": {
            ("1d_", "elastic_", "darcy_", "burgers_"): {"son_rho_hidden": 185},
            ("transport",): {"pos_encoding_max_freq": 0.1},
        }},
}

# Benchmark dimensions: (input_size_src, output_size_src, input_size_tgt, output_size_tgt)
BENCHMARK_DIMS = {
    "1d_integral": (1, 1, 1, 1), "1d_integral_varsens": (1, 1, 1, 1), "1d_integral_robust": (1, 1, 1, 1),
    "1d_derivative": (1, 1, 1, 1), "1d_derivative_varsens": (1, 1, 1, 1), "1d_derivative_robust": (1, 1, 1, 1),
    "darcy_1d": (1, 1, 1, 1), "darcy_1d_robust_train": (1, 1, 1, 1), "darcy_1d_robust_eval": (1, 1, 1, 1),
    "burgers_1d": (1, 1, 1, 1), "burgers_1d_robust_train": (1, 1, 1, 1), "burgers_1d_robust_eval": (1, 1, 1, 1),
    "heat_2d_P30": (2, 1, 2, 1), "heat_2d_P10": (2, 1, 2, 1), "diffraction_2d": (2, 2, 2, 2),
    "concentration_2d": (2, 1, 2, 1), "transport": (2, 1, 2, 2),
    "elastic_2d": (2, 1, 2, 1), "elastic_2d_robust_train": (2, 1, 2, 1), "elastic_2d_robust_eval": (2, 1, 2, 1),
}


@dataclass
class Job:
    """Represents a single benchmark job."""
    model: str
    benchmark: str
    seed: int
    device: str
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        return f"{self.model}_{self.benchmark}_seed{self.seed}"

    @property
    def base_model(self) -> str:
        return MODEL_VARIANTS.get(self.model, {}).get("base", self.model)

    def get_script_path(self, benchmarks_dir: Path) -> Path:
        base = self.base_model
        if base == "setonet":
            scripts, script_dir = SETONET_SCRIPTS, benchmarks_dir / "run_SetONet"
        elif base == "vidon":
            scripts, script_dir = VIDON_SCRIPTS, benchmarks_dir / "run_VIDON"
        else:
            scripts, script_dir = DEEPONET_SCRIPTS, benchmarks_dir / "run_DeepONet"
        script_name = scripts.get(self.benchmark)
        if script_name is None:
            raise ValueError(f"No script for {self.model}/{self.benchmark}")
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


def run_single_job(job: Job, benchmarks_dir: Path, project_root: Path, logs_all_dir: Path) -> JobResult:
    """Execute a single benchmark job."""
    start_time = time.time()
    try:
        script_path = job.get_script_path(benchmarks_dir)
        if not script_path.exists():
            return JobResult(job, False, error_message=f"Script not found: {script_path}",
                           duration_seconds=time.time() - start_time)

        # Build log directory: logs_all/<benchmark>/<model>/seed_<X>
        job_log_dir = logs_all_dir / job.benchmark / job.model / f"seed_{job.seed}"
        job_log_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, str(script_path), "--seed", str(job.seed), "--device", job.device,
               "--log_dir", str(job_log_dir)]
        
        # Track arguments set by benchmark-specific handling to prevent config overrides
        protected_args = set()

        # Handle 1d benchmarks which require --benchmark argument and variant-specific flags
        if job.benchmark.startswith("1d_"):
            # Parse: 1d_integral, 1d_integral_varsens, 1d_integral_robust, etc.
            parts = job.benchmark.split("_")
            benchmark_type = parts[1]  # "integral" or "derivative"
            cmd.extend(["--benchmark", benchmark_type])

            # Handle variants
            if len(parts) > 2:
                variant = parts[2]
                if variant == "varsens":
                    cmd.append("--variable_sensors")
                    protected_args.add("variable_sensors")
                elif variant == "robust":
                    # Robust evaluation uses variable sensors AND sensor dropout
                    cmd.append("--variable_sensors")
                    cmd.extend(["--eval_sensor_dropoff", "0.2", "--replace_with_nearest"])
                    protected_args.update(["variable_sensors", "eval_sensor_dropoff", "replace_with_nearest"])

                    # Load checkpoint from varsens benchmark (not base) to skip training
                    clean_benchmark = f"1d_{benchmark_type}_varsens"  # e.g., "1d_integral_varsens"
                    clean_checkpoint_dir = logs_all_dir / clean_benchmark / job.model / f"seed_{job.seed}"

                    # Find checkpoint file (naming varies by benchmark/model)
                    checkpoint_patterns = [
                        f"1d_{benchmark_type}*model.pth",
                        f"integral*model.pth" if benchmark_type == "integral" else f"derivative*model.pth",
                        "*model.pth",
                        "best_model.pth",
                    ]
                    checkpoint_path = None
                    for pattern in checkpoint_patterns:
                        matches = list(clean_checkpoint_dir.glob(pattern))
                        if matches:
                            checkpoint_path = matches[0]
                            break

                    if checkpoint_path and checkpoint_path.exists():
                        cmd.extend(["--load_model_path", str(checkpoint_path)])
                        protected_args.add("load_model_path")
                        logging.info(f"Will load checkpoint from: {checkpoint_path}")
                    else:
                        logging.warning(f"No checkpoint found for {job.benchmark} in {clean_checkpoint_dir} - will train from scratch")

        # Handle robust variants for elastic_2d and darcy_1d
        if job.benchmark.endswith("_robust_train"):
            cmd.extend(["--train_sensor_dropoff", "0.2", "--eval_sensor_dropoff", "0.2", "--replace_with_nearest"])
            protected_args.update(["train_sensor_dropoff", "eval_sensor_dropoff", "replace_with_nearest"])
        elif job.benchmark.endswith("_robust_eval"):
            cmd.extend(["--eval_sensor_dropoff", "0.2", "--replace_with_nearest"])
            protected_args.update(["eval_sensor_dropoff", "replace_with_nearest"])

            # Load checkpoint from clean (non-robust) benchmark to skip training
            clean_benchmark = job.benchmark.replace("_robust_eval", "")
            clean_checkpoint_dir = logs_all_dir / clean_benchmark / job.model / f"seed_{job.seed}"

            # Find checkpoint file (naming varies by benchmark/model)
            checkpoint_patterns = [
                f"{clean_benchmark.replace('_', '')}*model.pth",  # e.g., darcy1d_setonet_model.pth
                f"{clean_benchmark}*model.pth",
                "*model.pth",
                "best_model.pth",
            ]
            checkpoint_path = None
            for pattern in checkpoint_patterns:
                matches = list(clean_checkpoint_dir.glob(pattern))
                if matches:
                    checkpoint_path = matches[0]
                    break

            if checkpoint_path and checkpoint_path.exists():
                cmd.extend(["--load_model_path", str(checkpoint_path)])
                protected_args.add("load_model_path")
                logging.info(f"Will load checkpoint from: {checkpoint_path}")
            else:
                logging.warning(f"No checkpoint found for {job.benchmark} in {clean_checkpoint_dir} - will train from scratch")

        # Handle heat_2d dataset variants
        if job.benchmark == "heat_2d_P30":
            cmd.extend(["--data_path", str(project_root / "Data" / "heat_data" / "pcb_heat_adaptive_dataset8.0_n8192_N25_P30")])
            protected_args.add("data_path")
        elif job.benchmark == "heat_2d_P10":
            cmd.extend(["--data_path", str(project_root / "Data" / "heat_data" / "pcb_heat_adaptive_dataset9.0_n8192_N25_P10")])
            protected_args.add("data_path")

        # Apply config overrides, but skip protected arguments to preserve benchmark-specific settings
        for key, value in job.overrides.items():
            if key in protected_args:
                continue  # Skip - already set by benchmark-specific handling above
            if isinstance(value, bool):
                if value: cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.append(f"--{key}")
                cmd.extend(str(v) for v in value)
            else:
                cmd.extend([f"--{key}", str(value)])

        console_log = logs_all_dir / "_results" / "console_logs" / f"{job.job_id}.log"
        console_log.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting: {job.job_id} on {job.device}")

        with open(console_log, 'w') as log_f:
            log_f.write(f"Job: {job.job_id}\nCommand: {' '.join(cmd)}\nStarted: {datetime.now().isoformat()}\n{'='*80}\n\n")
            log_f.flush()
            result = subprocess.run(cmd, cwd=str(project_root), stdout=log_f, stderr=subprocess.STDOUT, text=True)

        duration = time.time() - start_time
        if result.returncode != 0:
            return JobResult(job, False, error_message=f"Exit code {result.returncode}", duration_seconds=duration)

        metrics = extract_metrics_from_log_dir(job_log_dir)
        logging.info(f"Completed: {job.job_id} in {duration:.1f}s")
        return JobResult(job, True, log_dir=str(job_log_dir), duration_seconds=duration, metrics=metrics)

    except Exception as e:
        logging.error(f"Error in {job.job_id}: {e}")
        return JobResult(job, False, error_message=str(e), duration_seconds=time.time() - start_time)


def extract_metrics_from_log_dir(log_dir: Path) -> Dict[str, float]:
    """Extract metrics from experiment_config.json in the log directory."""
    metrics = {}
    config_file = log_dir / "experiment_config.json"
    if not config_file.exists():
        return metrics
    try:
        with open(config_file) as f:
            data = json.load(f)
        # Check evaluation_results first (DeepONet/SetONet format)
        eval_results = data.get("evaluation_results", {})
        if "test_relative_l2_error" in eval_results:
            metrics["rel_l2_error"] = eval_results["test_relative_l2_error"]
        if "test_mse_loss" in eval_results:
            metrics["mse_loss"] = eval_results["test_mse_loss"]
        # Fallback to test_results format if eval_results empty
        if not metrics:
            test_results = data.get("test_results", {})
            if "relative_l2_error" in test_results:
                metrics["rel_l2_error"] = test_results["relative_l2_error"]
            if "mse_loss" in test_results:
                metrics["mse_loss"] = test_results["mse_loss"]
    except (json.JSONDecodeError, KeyError):
        pass
    return metrics


def scan_all_results(logs_all_dir: Path) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    """Scan all existing experiment results in logs_all directory.
    
    Returns dict mapping (model, benchmark, seed) -> metrics
    """
    all_results = {}
    
    # Walk through logs_all/<benchmark>/<model>/seed_<X>/experiment_config.json
    for benchmark_dir in logs_all_dir.iterdir():
        if not benchmark_dir.is_dir() or benchmark_dir.name.startswith("_"):
            continue  # Skip _results and non-directories
        benchmark = benchmark_dir.name
        
        for model_dir in benchmark_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            for seed_dir in model_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                
                metrics = extract_metrics_from_log_dir(seed_dir)
                if metrics:
                    all_results[(model, benchmark, seed)] = metrics
    
    return all_results


def run_jobs_parallel(jobs: List[Job], config: Dict[str, Any], benchmarks_dir: Path,
                      project_root: Path, logs_all_dir: Path) -> List[JobResult]:
    """Run jobs in parallel across GPUs with strict FIFO ordering."""
    devices = config['devices']
    continue_on_failure = config.get('continue_on_failure', True)
    max_workers = len(devices)
    results = []

    logging.info(f"Running {len(jobs)} jobs across {max_workers} GPU(s)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Track pending futures and their jobs
        pending_futures = {}
        job_queue = list(jobs)
        next_job_idx = 0

        # Submit initial batch (up to max_workers)
        while next_job_idx < len(job_queue) and len(pending_futures) < max_workers:
            job = job_queue[next_job_idx]
            future = executor.submit(run_single_job, job, benchmarks_dir, project_root, logs_all_dir)
            pending_futures[future] = job
            next_job_idx += 1

        # Process completions and submit new jobs in FIFO order
        while pending_futures:
            # Wait for any job to complete
            done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                job = pending_futures.pop(future)
                try:
                    result = future.result()
                    results.append(result)
                    if not result.success and not continue_on_failure:
                        executor.shutdown(wait=False, cancel_futures=True)
                        return results
                except Exception as e:
                    logging.error(f"Job {job.job_id} exception: {e}")
                    results.append(JobResult(job, False, error_message=str(e)))

                # Submit next job in queue if available
                if next_job_idx < len(job_queue):
                    next_job = job_queue[next_job_idx]
                    next_future = executor.submit(run_single_job, next_job, benchmarks_dir, project_root, logs_all_dir)
                    pending_futures[next_future] = next_job
                    next_job_idx += 1

    return results


def aggregate_results(results: List[JobResult], output_dir: Path, logs_all_dir: Path = None) -> None:
    """Aggregate results and save to CSV and JSON.

    If logs_all_dir is provided, scans all existing results and merges with current run.
    Current run results take precedence over historical results.
    """
    successful = [r for r in results if r.success and r.metrics]
    failed = [r for r in results if not r.success]
    logging.info(f"Current run: {len(successful)} successful, {len(failed)} failed")

    # Save individual results from current run
    with open(output_dir / "results_individual.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, ["model", "benchmark", "seed", "device", "success", "rel_l2_error", "mse_loss", "duration_seconds", "log_dir", "error"])
        w.writeheader()
        for r in results:
            w.writerow({"model": r.job.model, "benchmark": r.job.benchmark, "seed": r.job.seed,
                       "device": r.job.device, "success": r.success,
                       "rel_l2_error": r.metrics.get("rel_l2_error", ""),
                       "mse_loss": r.metrics.get("mse_loss", ""),
                       "duration_seconds": f"{r.duration_seconds:.1f}",
                       "log_dir": r.log_dir or "", "error": r.error_message or ""})

    # Build combined metrics: start with historical, then overlay current (current takes precedence)
    combined_metrics: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    
    if logs_all_dir and logs_all_dir.exists():
        # Scan all historical results
        historical = scan_all_results(logs_all_dir)
        combined_metrics.update(historical)
        logging.info(f"Found {len(historical)} historical results in {logs_all_dir}")
    
    # Overlay current run results (takes precedence)
    for r in successful:
        combined_metrics[(r.job.model, r.job.benchmark, r.job.seed)] = r.metrics
    
    logging.info(f"Total combined results: {len(combined_metrics)}")

    # Group by (model, benchmark) for aggregation
    grouped: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for (model, bench, seed), metrics in combined_metrics.items():
        grouped[(model, bench)].append(metrics)

    agg_data = []
    with open(output_dir / "results_summary.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, ["model", "benchmark", "n_seeds", "rel_l2_mean", "rel_l2_std", "mse_mean", "mse_std"])
        w.writeheader()
        for (model, bench), metrics_list in sorted(grouped.items()):
            l2 = [m["rel_l2_error"] for m in metrics_list if m.get("rel_l2_error") is not None]
            mse = [m["mse_loss"] for m in metrics_list if m.get("mse_loss") is not None]
            w.writerow({"model": model, "benchmark": bench, "n_seeds": len(metrics_list),
                       "rel_l2_mean": f"{mean(l2):.6f}" if l2 else "",
                       "rel_l2_std": f"{stdev(l2):.6f}" if len(l2) > 1 else "0.0",
                       "mse_mean": f"{mean(mse):.6e}" if mse else "",
                       "mse_std": f"{stdev(mse):.6e}" if len(mse) > 1 else "0.0"})
            agg_data.append({"model": model, "benchmark": bench, "n_seeds": len(metrics_list),
                            "rel_l2": {"mean": mean(l2) if l2 else None, "std": stdev(l2) if len(l2) > 1 else 0.0},
                            "mse": {"mean": mean(mse) if mse else None, "std": stdev(mse) if len(mse) > 1 else 0.0}})

    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "total_combined": len(combined_metrics),
                   "current_run_successful": len(successful), "current_run_failed": len(failed), 
                   "aggregated": agg_data}, f, indent=2)

    # Generate pivot tables (models × benchmarks) from combined data
    _generate_pivot_tables_from_metrics(grouped, output_dir)
    logging.info(f"Results saved to: {output_dir}")


def _generate_pivot_tables_from_metrics(grouped: Dict[Tuple[str, str], List[Dict[str, float]]], output_dir: Path) -> None:
    """Generate pivot-style result tables in CSV and LaTeX formats from metrics dicts."""
    # Collect all models and benchmarks
    models = sorted(set(m for m, _ in grouped.keys()))
    benchmarks = sorted(set(b for _, b in grouped.keys()))

    # Build data matrices
    mse_data, l2_data = {}, {}
    for (model, bench), metrics_list in grouped.items():
        l2_vals = [m["rel_l2_error"] for m in metrics_list if m.get("rel_l2_error") is not None]
        mse_vals = [m["mse_loss"] for m in metrics_list if m.get("mse_loss") is not None]

        if l2_vals:
            l2_mean, l2_std = mean(l2_vals), stdev(l2_vals) if len(l2_vals) > 1 else 0.0
            l2_data[(model, bench)] = (l2_mean, l2_std)
        if mse_vals:
            mse_mean, mse_std = mean(mse_vals), stdev(mse_vals) if len(mse_vals) > 1 else 0.0
            mse_data[(model, bench)] = (mse_mean, mse_std)

    # Write CSV pivot tables
    _write_pivot_csv(output_dir / "results_l2.csv", models, benchmarks, l2_data, "{:.6f}")
    _write_pivot_csv(output_dir / "results_mse.csv", models, benchmarks, mse_data, "{:.2e}")

    # Write LaTeX tables
    _write_pivot_latex(output_dir / "results_l2.tex", models, benchmarks, l2_data, "{:.6f}", "Relative L2 Error")
    _write_pivot_latex(output_dir / "results_mse.tex", models, benchmarks, mse_data, "{:.2e}", "MSE Loss")


def _write_pivot_csv(path: Path, models: List[str], benchmarks: List[str], 
                     data: Dict[Tuple[str, str], Tuple[float, float]], fmt: str) -> None:
    """Write pivot table to CSV."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["model"] + benchmarks)
        for model in models:
            row = [model]
            for bench in benchmarks:
                if (model, bench) in data:
                    m, s = data[(model, bench)]
                    row.append(f"{fmt.format(m)} ± {fmt.format(s)}")
                else:
                    row.append("N/A")
            w.writerow(row)


def _write_pivot_latex(path: Path, models: List[str], benchmarks: List[str],
                       data: Dict[Tuple[str, str], Tuple[float, float]], fmt: str, caption: str) -> None:
    """Write pivot table to LaTeX."""
    with open(path, 'w') as f:
        ncols = len(benchmarks) + 1
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\begin{tabular}{l" + "c" * len(benchmarks) + "}\n\\toprule\n")
        f.write("Model & " + " & ".join(b.replace("_", "\\_") for b in benchmarks) + " \\\\\n\\midrule\n")
        for model in models:
            row = [model.replace("_", "\\_")]
            for bench in benchmarks:
                if (model, bench) in data:
                    m, s = data[(model, bench)]
                    row.append(f"${fmt.format(m)} \\pm {fmt.format(s)}$")
                else:
                    row.append("N/A")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def get_benchmark_overrides(variant: Dict[str, Any], benchmark: str) -> Dict[str, Any]:
    """Get benchmark-specific overrides for a model variant."""
    bench_overrides = variant.get("benchmark_overrides", {})
    for patterns, overrides in bench_overrides.items():
        # patterns is a tuple of prefixes/names to match
        for pattern in patterns:
            if benchmark.startswith(pattern) or benchmark == pattern:
                return overrides
    return {}


def generate_all_configs(benchmarks_dir: Path) -> Dict[str, Any]:
    """Generate complete configs for all (model_variant, benchmark) pairs."""
    configs = {}
    for model_name, variant in MODEL_VARIANTS.items():
        base_model, variant_overrides = variant["base"], variant.get("overrides", {})
        if base_model == "setonet":
            scripts = SETONET_SCRIPTS
        elif base_model == "vidon":
            scripts = VIDON_SCRIPTS
        else:
            scripts = DEEPONET_SCRIPTS
        model_configs = {}
        for benchmark, script_name in scripts.items():
            bench_config = load_benchmark_config(benchmarks_dir, base_model, benchmark)
            bench_specific = get_benchmark_overrides(variant, benchmark)
            config = {**bench_config, **variant_overrides, **bench_specific, "_script": script_name, "_base_model": base_model}
            model_configs[benchmark] = config
        if model_configs:
            configs[model_name] = model_configs
    return configs


def count_parameters(model) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_param_table(benchmarks_dir: Path) -> Dict[str, Dict[str, int]]:
    """Generate parameter counts for all (model, benchmark) pairs."""
    project_root = benchmarks_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from Models.SetONet import SetONet
    from Models.DeepONet import DeepONetWrapper
    from Models.VIDON import VIDON

    param_table = {}
    for model_name, variant in MODEL_VARIANTS.items():
        base_model = variant["base"]
        variant_overrides = variant.get("overrides", {})
        if base_model == "setonet":
            scripts = SETONET_SCRIPTS
        elif base_model == "vidon":
            scripts = VIDON_SCRIPTS
        else:
            scripts = DEEPONET_SCRIPTS
        model_params = {}
        
        for benchmark in scripts.keys():
            if benchmark not in BENCHMARK_DIMS:
                continue
            dims = BENCHMARK_DIMS[benchmark]
            cfg = load_benchmark_config(benchmarks_dir, base_model, benchmark)
            cfg.update(variant_overrides)
            cfg.update(get_benchmark_overrides(variant, benchmark))
            
            try:
                if base_model == "setonet":
                    model = SetONet(
                        input_size_src=dims[0], output_size_src=dims[1],
                        input_size_tgt=dims[2], output_size_tgt=dims[3],
                        p=cfg.get("son_p_dim", 32),
                        phi_hidden_size=cfg.get("son_phi_hidden", 256),
                        rho_hidden_size=cfg.get("son_rho_hidden", 256),
                        trunk_hidden_size=cfg.get("son_trunk_hidden", 256),
                        n_trunk_layers=cfg.get("son_n_trunk_layers", 4),
                        phi_output_size=cfg.get("son_phi_output_size", 32),
                        pos_encoding_type=cfg.get("pos_encoding_type", "sinusoidal"),
                        pos_encoding_dim=cfg.get("pos_encoding_dim", 64),
                        aggregation_type=cfg.get("son_aggregation", "attention"),
                        branch_head_type=cfg.get("son_branch_head_type", "standard"),
                        adapt_quad_rank=cfg.get("son_adapt_quad_rank", 4),
                        adapt_quad_hidden=cfg.get("son_adapt_quad_hidden", 64),
                    )
                elif base_model == "vidon":
                    model = VIDON(
                        input_size_src=dims[0], output_size_src=dims[1],
                        input_size_tgt=dims[2], output_size_tgt=dims[3],
                        p=cfg.get("vidon_p_dim", 32),
                        n_heads=cfg.get("vidon_n_heads", 4),
                        d_enc=cfg.get("vidon_d_enc", 40),
                        head_output_size=cfg.get("vidon_head_output_size", 64),
                        enc_hidden_size=cfg.get("vidon_enc_hidden", 40),
                        enc_n_layers=cfg.get("vidon_enc_n_layers", 4),
                        head_hidden_size=cfg.get("vidon_head_hidden", 128),
                        head_n_layers=cfg.get("vidon_head_n_layers", 4),
                        combine_hidden_size=cfg.get("vidon_combine_hidden", 256),
                        combine_n_layers=cfg.get("vidon_combine_n_layers", 4),
                        trunk_hidden_size=cfg.get("vidon_trunk_hidden", 256),
                        n_trunk_layers=cfg.get("vidon_n_trunk_layers", 4),
                    )
                else:
                    sensor_size = cfg.get("sensor_size", 300)
                    model = DeepONetWrapper(
                        branch_input_dim=sensor_size, trunk_input_dim=dims[2],
                        p=cfg.get("don_p_dim", 32),
                        trunk_hidden_size=cfg.get("don_trunk_hidden", 256),
                        n_trunk_layers=cfg.get("don_n_trunk_layers", 4),
                        branch_hidden_size=cfg.get("don_branch_hidden", 128),
                        n_branch_layers=cfg.get("don_n_branch_layers", 3),
                    )
                model_params[benchmark] = count_parameters(model)
            except Exception as e:
                print(f"Warning: Could not create {model_name}/{benchmark}: {e}")
                model_params[benchmark] = -1
        
        if model_params:
            param_table[model_name] = model_params
    return param_table
