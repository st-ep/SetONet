#!/usr/bin/env python3
"""
Export TensorBoard eval MSE scalars to merged CSV files (one per benchmark/model).

Walks logs_all/ (or --logs-dir), reads all seed tensorboard directories for each
(benchmark, model) pair, deduplicates by step (latest run wins), and writes a single
eval_mse.csv with columns: step, seed_0, seed_1, ..., seed_4.

Output structure:
    logs_csv/<benchmark>/<model>/eval_mse.csv

Usage:
    python Benchmarks/export_tensorboard_csv.py
    python Benchmarks/export_tensorboard_csv.py --logs-dir /path/to/logs
    python Benchmarks/export_tensorboard_csv.py --output-dir /path/to/csvs
"""
import argparse
import csv
import multiprocessing
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAG = "Evaluation/Test_MSE_Loss"
OUT_FILENAME = "eval_mse.csv"
SKIP_MODELS = frozenset({"setonet_adaptive", "setonet_petrov", "setonet_galerkin"})


def _load_seed(tb_dir: Path) -> dict[int, float] | None:
    """
    Load eval MSE from a single seed's tensorboard directory.

    Returns dict mapping step -> value (deduplicated, latest run wins),
    or None if no data is available.
    """
    event_files = list(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    try:
        ea = EventAccumulator(str(tb_dir))
        ea.Reload()
    except Exception:
        return None

    if TAG not in ea.Tags().get("scalars", []):
        return None
    raw_events = ea.Scalars(TAG)
    if not raw_events:
        return None

    by_step = {}
    for e in raw_events:
        by_step[e.step] = e.value
    return by_step


def main():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard eval MSE to merged CSV (one per benchmark/model)."
    )
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    default_logs = project_root / "logs_all"
    default_output = project_root / "logs_csv"
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=default_logs,
        help=f"Root logs directory (default: {default_logs})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Output directory for CSVs (default: {default_output})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(multiprocessing.cpu_count(), 16),
        help="Number of parallel workers for reading event files (default: min(cpu_count, 16))",
    )
    args = parser.parse_args()
    logs_dir = args.logs_dir if args.logs_dir.is_absolute() else project_root / args.logs_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    if not logs_dir.is_dir():
        print(f"Error: logs directory not found: {logs_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover (benchmark, model) -> list of (seed_name, tb_dir)
    groups: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    for bench_dir in sorted(logs_dir.iterdir()):
        if not bench_dir.is_dir() or bench_dir.name.startswith("_") or "robust" in bench_dir.name:
            continue
        for model_dir in sorted(bench_dir.iterdir()):
            if not model_dir.is_dir() or model_dir.name in SKIP_MODELS:
                continue
            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                tb_path = seed_dir / "tensorboard"
                if tb_path.is_dir():
                    groups[(bench_dir.name, model_dir.name)].append((seed_dir.name, tb_path))

    total_groups = len(groups)
    total_seeds = sum(len(v) for v in groups.values())
    if total_groups == 0:
        print("No tensorboard directories found.")
        return

    print(f"Found {total_groups} (benchmark, model) groups with {total_seeds} seed dirs.")
    print(f"Output: {output_dir}")
    print(f"Using {args.workers} workers.")

    # Load all seeds in parallel (the bottleneck is EventAccumulator.Reload)
    all_tb_paths = []
    path_to_key: dict[str, tuple[str, str, str]] = {}
    for (benchmark, model_name), seed_entries in groups.items():
        for seed_name, tb_path in seed_entries:
            all_tb_paths.append(tb_path)
            path_to_key[str(tb_path)] = (benchmark, model_name, seed_name)

    print(f"Loading {len(all_tb_paths)} tensorboard directories...")
    loaded: dict[str, dict[int, float] | None] = {}

    def _load_wrapper(tb_path: Path) -> tuple[str, dict[int, float] | None]:
        return (str(tb_path), _load_seed(tb_path))

    if args.workers <= 1:
        for i, tb_path in enumerate(all_tb_paths, 1):
            loaded[str(tb_path)] = _load_seed(tb_path)
            if i % 10 == 0 or i == len(all_tb_paths):
                print(f"  Loaded: {i}/{len(all_tb_paths)}")
    else:
        try:
            with multiprocessing.Pool(processes=args.workers) as pool:
                for i, (key, data) in enumerate(
                    pool.imap_unordered(_load_wrapper, all_tb_paths, chunksize=1), start=1
                ):
                    loaded[key] = data
                    if i % 10 == 0 or i == len(all_tb_paths):
                        print(f"  Loaded: {i}/{len(all_tb_paths)}")
        except (OSError, ImportError) as e:
            print(f"Warning: Pool failed ({e}), falling back to sequential.", file=sys.stderr)
            for i, tb_path in enumerate(all_tb_paths, 1):
                loaded[str(tb_path)] = _load_seed(tb_path)
                if i % 10 == 0 or i == len(all_tb_paths):
                    print(f"  Loaded: {i}/{len(all_tb_paths)}")

    ok_count = 0
    skip_no_data = []
    partial = []
    errors = []

    for (benchmark, model), seed_entries in sorted(groups.items()):
        seed_data: dict[str, dict[int, float]] = {}
        for seed_name, tb_path in seed_entries:
            data = loaded.get(str(tb_path))
            if data is not None:
                seed_data[seed_name] = data

        if not seed_data:
            skip_no_data.append(f"{benchmark}/{model}")
            continue

        # Collect all steps across seeds
        all_steps = sorted(set(s for d in seed_data.values() for s in d))
        seed_names = sorted(seed_data.keys())

        missing_seeds = [s for s, _ in seed_entries if s not in seed_data]
        if missing_seeds:
            partial.append(f"{benchmark}/{model} (missing: {', '.join(missing_seeds)})")

        csv_path = output_dir / benchmark / model / OUT_FILENAME
        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step"] + seed_names)
                for step in all_steps:
                    row = [step] + [seed_data[s].get(step, "") for s in seed_names]
                    w.writerow(row)
            ok_count += 1
        except Exception as e:
            errors.append(f"{benchmark}/{model}: {e}")

    print()
    print("Summary:")
    print(f"  CSVs written: {ok_count}")
    if skip_no_data:
        print(f"  Skipped (no eval data for any seed): {len(skip_no_data)}")
        for s in skip_no_data:
            print(f"    {s}")
    if partial:
        print(f"  Partial (some seeds missing eval data): {len(partial)}")
        for p in partial:
            print(f"    {p}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()
