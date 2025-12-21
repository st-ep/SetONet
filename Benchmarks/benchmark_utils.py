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

import yaml

# Script mappings
SETONET_SCRIPTS = {
    "heat_2d": "run_heat_2d.py", "concentration_2d": "run_consantration_2d.py", "transport": "run_transoprt.py",
    "elastic_2d": "run_elastic_2d.py", "elastic_2d_robust_train": "run_elastic_2d.py", "elastic_2d_robust_eval": "run_elastic_2d.py",
    "darcy_1d": "run_darcy_1d.py", "darcy_1d_robust_train": "run_darcy_1d.py", "darcy_1d_robust_eval": "run_darcy_1d.py",
    "1d_integral": "run_1d.py", "1d_integral_varsens": "run_1d.py", "1d_integral_robust": "run_1d.py",
    "1d_derivative": "run_1d.py", "1d_derivative_varsens": "run_1d.py", "1d_derivative_robust": "run_1d.py",
}
DEEPONET_SCRIPTS = {
    "elastic_2d": "run_elastic_2d_don.py", "elastic_2d_robust_train": "run_elastic_2d_don.py", "elastic_2d_robust_eval": "run_elastic_2d_don.py",
    "darcy_1d": "run_darcy_1d_don.py", "darcy_1d_robust_train": "run_darcy_1d_don.py", "darcy_1d_robust_eval": "run_darcy_1d_don.py",
    "1d_integral": "run_1d_don.py", "1d_integral_varsens": "run_1d_don.py", "1d_integral_robust": "run_1d_don.py",
    "1d_derivative": "run_1d_don.py", "1d_derivative_varsens": "run_1d_don.py", "1d_derivative_robust": "run_1d_don.py",
}

# Maps (base_model, benchmark) to config file
BENCHMARK_CONFIG_MAP = {
    "setonet": {
        "1d_integral": "setonet_1d.yaml", "1d_integral_varsens": "setonet_1d.yaml", "1d_integral_robust": "setonet_1d.yaml",
        "1d_derivative": "setonet_1d.yaml", "1d_derivative_varsens": "setonet_1d.yaml", "1d_derivative_robust": "setonet_1d.yaml",
        "darcy_1d": "setonet_1d.yaml", "darcy_1d_robust_train": "setonet_1d.yaml", "darcy_1d_robust_eval": "setonet_1d.yaml",
        "heat_2d": "setonet_heat2d.yaml", "concentration_2d": "setonet_heat2d.yaml", "transport": "setonet_heat2d.yaml",
        "elastic_2d": "setonet_elastic2d.yaml", "elastic_2d_robust_train": "setonet_elastic2d.yaml", "elastic_2d_robust_eval": "setonet_elastic2d.yaml",
    },
    "deeponet": {
        "1d_integral": "deeponet_1d.yaml", "1d_integral_varsens": "deeponet_1d.yaml", "1d_integral_robust": "deeponet_1d.yaml",
        "1d_derivative": "deeponet_1d.yaml", "1d_derivative_varsens": "deeponet_1d.yaml", "1d_derivative_robust": "deeponet_1d.yaml",
        "darcy_1d": "deeponet_1d.yaml", "darcy_1d_robust_train": "deeponet_1d.yaml", "darcy_1d_robust_eval": "deeponet_1d.yaml",
        "elastic_2d": "deeponet_elastic2d.yaml", "elastic_2d_robust_train": "deeponet_elastic2d.yaml", "elastic_2d_robust_eval": "deeponet_elastic2d.yaml",
    },
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

# Benchmark dimensions: (input_size_src, output_size_src, input_size_tgt, output_size_tgt)
BENCHMARK_DIMS = {
    "1d_integral": (1, 1, 1, 1), "1d_integral_varsens": (1, 1, 1, 1), "1d_integral_robust": (1, 1, 1, 1),
    "1d_derivative": (1, 1, 1, 1), "1d_derivative_varsens": (1, 1, 1, 1), "1d_derivative_robust": (1, 1, 1, 1),
    "darcy_1d": (1, 1, 1, 1), "darcy_1d_robust_train": (1, 1, 1, 1), "darcy_1d_robust_eval": (1, 1, 1, 1),
    "heat_2d": (2, 1, 2, 1), "concentration_2d": (2, 1, 2, 1), "transport": (2, 1, 2, 2),
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
                elif variant == "robust":
                    cmd.extend(["--eval_sensor_dropoff", "0.2", "--replace_with_nearest"])
        
        # Handle robust variants for elastic_2d and darcy_1d
        if job.benchmark.endswith("_robust_train"):
            cmd.extend(["--train_sensor_dropoff", "0.2", "--eval_sensor_dropoff", "0.2", "--replace_with_nearest"])
        elif job.benchmark.endswith("_robust_eval"):
            cmd.extend(["--eval_sensor_dropoff", "0.2", "--replace_with_nearest"])
        
        for key, value in job.overrides.items():
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
            test_results = json.load(f).get("test_results", {})
        if "relative_l2_error" in test_results:
            metrics["rel_l2_error"] = test_results["relative_l2_error"]
        if "mse_loss" in test_results:
            metrics["mse_loss"] = test_results["mse_loss"]
    except (json.JSONDecodeError, KeyError):
        pass
    return metrics


def run_jobs_parallel(jobs: List[Job], config: Dict[str, Any], benchmarks_dir: Path,
                      project_root: Path, logs_all_dir: Path) -> List[JobResult]:
    """Run jobs in parallel across GPUs."""
    devices = config['devices']
    continue_on_failure = config.get('continue_on_failure', True)
    results = []

    logging.info(f"Running {len(jobs)} jobs across {len(devices)} GPU(s)")
    with ProcessPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(run_single_job, job, benchmarks_dir, project_root, logs_all_dir): job
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

    with open(output_dir / "results_summary.json", 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "total": len(results),
                   "successful": len(successful), "failed": len(failed), "aggregated": agg_data}, f, indent=2)

    # Generate pivot tables (models × benchmarks)
    _generate_pivot_tables(grouped, output_dir)
    logging.info(f"Results saved to: {output_dir}")


def _generate_pivot_tables(grouped: Dict[Tuple[str, str], List], output_dir: Path) -> None:
    """Generate pivot-style result tables in CSV and LaTeX formats."""
    # Collect all models and benchmarks
    models = sorted(set(m for m, _ in grouped.keys()))
    benchmarks = sorted(set(b for _, b in grouped.keys()))
    
    # Build data matrices
    mse_data, l2_data = {}, {}
    for (model, bench), group in grouped.items():
        l2_vals = [r.metrics["rel_l2_error"] for r in group if "rel_l2_error" in r.metrics]
        mse_vals = [r.metrics["mse_loss"] for r in group if "mse_loss" in r.metrics]
        
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


def generate_all_configs(benchmarks_dir: Path) -> Dict[str, Any]:
    """Generate complete configs for all (model_variant, benchmark) pairs."""
    configs = {}
    for model_name, variant in MODEL_VARIANTS.items():
        base_model, variant_overrides = variant["base"], variant.get("overrides", {})
        scripts = SETONET_SCRIPTS if base_model == "setonet" else DEEPONET_SCRIPTS
        model_configs = {}
        for benchmark, script_name in scripts.items():
            bench_config = load_benchmark_config(benchmarks_dir, base_model, benchmark)
            config = {**bench_config, **variant_overrides, "_script": script_name, "_base_model": base_model}
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

    param_table = {}
    for model_name, variant in MODEL_VARIANTS.items():
        base_model = variant["base"]
        variant_overrides = variant.get("overrides", {})
        scripts = SETONET_SCRIPTS if base_model == "setonet" else DEEPONET_SCRIPTS
        model_params = {}
        
        for benchmark in scripts.keys():
            if benchmark not in BENCHMARK_DIMS:
                continue
            dims = BENCHMARK_DIMS[benchmark]
            cfg = load_benchmark_config(benchmarks_dir, base_model, benchmark)
            cfg.update(variant_overrides)
            
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
