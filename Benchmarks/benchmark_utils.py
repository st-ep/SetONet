"""Utility classes and functions for benchmark runner."""
import csv
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

# Script mappings
SETONET_SCRIPTS = {
    "heat_2d": "run_heat_2d.py", "elastic_2d": "run_elastic_2d.py",
    "darcy_1d": "run_darcy_1d.py", "1d": "run_1d.py",
    "concentration_2d": "run_consantration_2d.py", "transport": "run_transoprt.py",
}
DEEPONET_SCRIPTS = {
    "elastic_2d": "run_elastic_2d_don.py", "darcy_1d": "run_darcy_1d_don.py", "1d": "run_1d_don.py",
}

# Maps (base_model, benchmark) to config file
BENCHMARK_CONFIG_MAP = {
    "setonet": {
        "1d": "setonet_1d.yaml", "darcy_1d": "setonet_1d.yaml",
        "heat_2d": "setonet_heat2d.yaml", "concentration_2d": "setonet_heat2d.yaml",
        "transport": "setonet_heat2d.yaml", "elastic_2d": "setonet_elastic2d.yaml",
    },
    "deeponet": {"1d": "deeponet_1d.yaml", "darcy_1d": "deeponet_1d.yaml", "elastic_2d": "deeponet_elastic2d.yaml"},
}

# Model variant definitions
MODEL_VARIANTS = {
    "deeponet": {"base": "deeponet", "overrides": {}},
    "setonet_sum": {"base": "setonet", "overrides": {"son_aggregation": "sum"}},
    "setonet_mean": {"base": "setonet", "overrides": {"son_aggregation": "mean"}},
    "setonet_attention": {"base": "setonet", "overrides": {}},
    "setonet_petrov": {"base": "setonet", "overrides": {"son_branch_head_type": "petrov_attention"}},
    "setonet_galerkin": {"base": "setonet", "overrides": {"son_branch_head_type": "galerkin_pou"}},
    "setonet_quadrature": {"base": "setonet", "overrides": {"son_branch_head_type": "quadrature"}},
    "setonet_adaptive": {"base": "setonet", "overrides": {"son_branch_head_type": "adaptive_quadrature"}},
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
        scripts = SETONET_SCRIPTS if base == "setonet" else DEEPONET_SCRIPTS
        script_dir = benchmarks_dir / ("run_SetONet" if base == "setonet" else "run_DeepONet")
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


def run_single_job(job: Job, benchmarks_dir: Path, project_root: Path, run_output_dir: Path) -> JobResult:
    """Execute a single benchmark job."""
    start_time = time.time()
    try:
        script_path = job.get_script_path(benchmarks_dir)
        if not script_path.exists():
            return JobResult(job, False, error_message=f"Script not found: {script_path}",
                           duration_seconds=time.time() - start_time)

        cmd = [sys.executable, str(script_path), "--seed", str(job.seed), "--device", job.device]
        for key, value in job.overrides.items():
            if isinstance(value, bool):
                if value: cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.append(f"--{key}")
                cmd.extend(str(v) for v in value)
            else:
                cmd.extend([f"--{key}", str(value)])

        job_log_file = run_output_dir / f"{job.job_id}.log"
        logging.info(f"Starting: {job.job_id} on {job.device}")

        with open(job_log_file, 'w') as log_f:
            log_f.write(f"Job: {job.job_id}\nCommand: {' '.join(cmd)}\nStarted: {datetime.now().isoformat()}\n{'='*80}\n\n")
            log_f.flush()
            result = subprocess.run(cmd, cwd=str(project_root), stdout=log_f, stderr=subprocess.STDOUT, text=True)

        duration = time.time() - start_time
        if result.returncode != 0:
            return JobResult(job, False, error_message=f"Exit code {result.returncode}", duration_seconds=duration)

        log_dir, metrics = find_job_results(job, project_root)
        logging.info(f"Completed: {job.job_id} in {duration:.1f}s")
        return JobResult(job, True, log_dir=log_dir, duration_seconds=duration, metrics=metrics)

    except Exception as e:
        logging.error(f"Error in {job.job_id}: {e}")
        return JobResult(job, False, error_message=str(e), duration_seconds=time.time() - start_time)


def find_job_results(job: Job, project_root: Path) -> Tuple[Optional[str], Dict[str, float]]:
    """Find log directory and extract metrics for a completed job."""
    prefix = "SetONet" if job.base_model == "setonet" else "DeepONet"
    patterns = [f"{prefix}_{job.benchmark}", f"{prefix}_{job.benchmark.replace('_', '')}"]
    logs_dir = project_root / "logs"
    
    latest_dir, latest_time = None, 0
    for pattern in patterns:
        pattern_dir = logs_dir / pattern
        if not pattern_dir.exists(): continue
        for subdir in pattern_dir.iterdir():
            config_file = subdir / "experiment_config.json"
            if not config_file.exists(): continue
            try:
                with open(config_file) as f:
                    cfg = json.load(f)
                if cfg.get("seed") == job.seed and (mtime := config_file.stat().st_mtime) > latest_time:
                    latest_time, latest_dir = mtime, subdir
            except (json.JSONDecodeError, KeyError):
                continue

    if latest_dir is None:
        return None, {}

    metrics = {}
    try:
        with open(latest_dir / "experiment_config.json") as f:
            test_results = json.load(f).get("test_results", {})
        if "relative_l2_error" in test_results: metrics["rel_l2_error"] = test_results["relative_l2_error"]
        if "mse_loss" in test_results: metrics["mse_loss"] = test_results["mse_loss"]
    except: pass
    return str(latest_dir), metrics


def run_jobs_parallel(jobs: List[Job], config: Dict[str, Any], benchmarks_dir: Path,
                      project_root: Path, run_output_dir: Path) -> List[JobResult]:
    """Run jobs in parallel across GPUs."""
    devices = config['devices']
    continue_on_failure = config.get('continue_on_failure', True)
    results = []

    logging.info(f"Running {len(jobs)} jobs across {len(devices)} GPU(s)")
    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(run_single_job, job, benchmarks_dir, project_root, run_output_dir): job
                   for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                if not result.success and not continue_on_failure:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
            except Exception as e:
                logging.error(f"Job {futures[future].job_id} exception: {e}")
                results.append(JobResult(futures[future], False, error_message=str(e)))
    return results


def aggregate_results(results: List[JobResult], output_dir: Path) -> None:
    """Aggregate results and save to CSV and JSON."""
    successful = [r for r in results if r.success and r.metrics]
    failed = [r for r in results if not r.success]
    logging.info(f"Aggregating {len(successful)} successful, {len(failed)} failed")

    # Individual results CSV
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

    # Summary by (model, benchmark)
    grouped: Dict[Tuple[str, str], List[JobResult]] = defaultdict(list)
    for r in successful:
        grouped[(r.job.model, r.job.benchmark)].append(r)

    agg_data = []
    with open(output_dir / "results_summary.csv", 'w', newline='') as f:
        w = csv.DictWriter(f, ["model", "benchmark", "n_seeds", "rel_l2_mean", "rel_l2_std", "mse_mean", "mse_std"])
        w.writeheader()
        for (model, bench), group in sorted(grouped.items()):
            l2 = [r.metrics["rel_l2_error"] for r in group if "rel_l2_error" in r.metrics]
            mse = [r.metrics["mse_loss"] for r in group if "mse_loss" in r.metrics]
            w.writerow({"model": model, "benchmark": bench, "n_seeds": len(group),
                       "rel_l2_mean": f"{mean(l2):.6f}" if l2 else "",
                       "rel_l2_std": f"{stdev(l2):.6f}" if len(l2) > 1 else "0.0",
                       "mse_mean": f"{mean(mse):.6e}" if mse else "",
                       "mse_std": f"{stdev(mse):.6e}" if len(mse) > 1 else "0.0"})
            agg_data.append({"model": model, "benchmark": bench, "n_seeds": len(group),
                            "rel_l2": {"mean": mean(l2) if l2 else None, "std": stdev(l2) if len(l2) > 1 else 0.0},
                            "mse": {"mean": mean(mse) if mse else None, "std": stdev(mse) if len(mse) > 1 else 0.0}})

    # JSON summary
    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "total": len(results),
                   "successful": len(successful), "failed": len(failed), "aggregated": agg_data}, f, indent=2)

    logging.info(f"Results saved to: {output_dir}")

