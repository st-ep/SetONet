import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate transport-plan baselines (uniform and kernel-only)."
    )
    default_data_path = os.path.join(project_root, "Data", "transport_data", "transport_plan_dataset")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to transport-plan dataset")
    parser.add_argument("--mode", type=str, default="plan", choices=["plan", "coupling", "barycentric"],
                        help="Learning target: row-conditional plan, coupling, or barycentric map")
    parser.add_argument("--n_test_samples", type=int, default=100, help="Number of test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to use")
    parser.add_argument("--baseline_epsilon", type=float, default=-1.0,
                        help="Epsilon for kernel-only baseline if dataset epsilon is missing (<=0 disables)")
    parser.add_argument("--eval_sinkhorn_iters", type=int, default=0,
                        help="Eval-only Sinkhorn-Knopp iterations to project onto marginals (0 disables)")
    parser.add_argument("--eval_sinkhorn_eps", type=float, default=1e-9,
                        help="Stability epsilon for eval Sinkhorn-Knopp projection")
    return parser.parse_args()


def _normalize_rows(target, eps=1e-12):
    target = target.clamp_min(0.0)
    row_sums = target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return target / row_sums


def compute_plan_metrics(pred, target, eps=1e-12):
    target = _normalize_rows(target, eps=eps)
    log_probs = F.log_softmax(pred, dim=-1)
    ce = -(target * log_probs).sum(dim=-1).mean()
    log_target = torch.log(target.clamp_min(eps))
    entropy = -(target * log_target).sum(dim=-1).mean()
    kl = ce - entropy
    log_n = math.log(float(target.shape[-1]))
    kl_norm = kl / log_n if log_n > 0.0 else kl * 0.0
    return ce, entropy, kl, kl_norm


def compute_column_marginal_errors(w_hat):
    if w_hat.dim() == 3:
        col_marg = w_hat.mean(dim=1)
    elif w_hat.dim() == 2:
        col_marg = w_hat.mean(dim=0)
    else:
        raise ValueError(f"Unexpected w_hat shape: {tuple(w_hat.shape)}")
    n_target = w_hat.shape[-1]
    target = 1.0 / float(n_target)
    diff = col_marg - target
    l1 = diff.abs().mean()
    linf = diff.abs().max()
    return l1, linf


def sinkhorn_project(p_hat, n_iters=10, eps=1e-9):
    if n_iters <= 0:
        return p_hat
    m, n = p_hat.shape
    a = torch.full((m,), 1.0 / float(m), device=p_hat.device, dtype=p_hat.dtype)
    b = torch.full((n,), 1.0 / float(n), device=p_hat.device, dtype=p_hat.dtype)
    k = p_hat.clamp_min(0.0) + eps
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(n_iters):
        u = a / (k @ v).clamp_min(eps)
        v = b / (k.t() @ u).clamp_min(eps)
    return (u[:, None] * k) * v[None, :]


def _get_baseline_epsilon(sample, args):
    eps = float(sample.get("epsilon", -1.0))
    if eps > 0.0:
        return eps
    override = float(getattr(args, "baseline_epsilon", -1.0))
    return override if override > 0.0 else None


def _init_totals():
    return {
        "count": 0,
        "loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "kl_norm": 0.0,
        "marg_l1": 0.0,
        "marg_linf": 0.0,
        "cost_gap": 0.0,
        "abs_cost_gap": 0.0,
        "rel_cost_gap": 0.0,
        "n_cost": 0,
        "target_val": None,
        "proj_marg_l1": 0.0,
        "proj_marg_linf": 0.0,
        "proj_cost_gap": 0.0,
        "proj_abs_cost_gap": 0.0,
        "proj_rel_cost_gap": 0.0,
        "proj_n_cost": 0,
    }


def _accumulate(totals, logits, target, cost_matrix, ot_cost, eval_sinkhorn_iters, eval_sinkhorn_eps):
    ce, entropy, kl, kl_norm = compute_plan_metrics(logits.unsqueeze(0), target.unsqueeze(0))
    w_hat = torch.softmax(logits, dim=-1)
    marg_l1, marg_linf = compute_column_marginal_errors(w_hat)

    totals["count"] += 1
    totals["loss"] += ce.item()
    totals["entropy"] += entropy.item()
    totals["kl"] += kl.item()
    totals["kl_norm"] += kl_norm.item()
    totals["marg_l1"] += marg_l1.item()
    totals["marg_linf"] += marg_linf.item()
    totals["target_val"] = 1.0 / float(w_hat.shape[-1])

    if not np.isnan(ot_cost) and ot_cost != 0.0:
        p_hat = w_hat / float(w_hat.shape[0])
        cost_pred = torch.sum(p_hat * cost_matrix).item()
        cost_gap = cost_pred - ot_cost
        totals["cost_gap"] += cost_gap
        totals["abs_cost_gap"] += abs(cost_gap)
        totals["rel_cost_gap"] += (cost_gap / ot_cost)
        totals["n_cost"] += 1

    if eval_sinkhorn_iters > 0:
        p_hat = w_hat / float(w_hat.shape[0])
        p_proj = sinkhorn_project(p_hat, n_iters=eval_sinkhorn_iters, eps=eval_sinkhorn_eps)
        w_proj = p_proj * float(w_hat.shape[0])
        proj_l1, proj_linf = compute_column_marginal_errors(w_proj)
        totals["proj_marg_l1"] += proj_l1.item()
        totals["proj_marg_linf"] += proj_linf.item()
        if not np.isnan(ot_cost) and ot_cost != 0.0:
            cost_pred = torch.sum(p_proj * cost_matrix).item()
            cost_gap = cost_pred - ot_cost
            totals["proj_cost_gap"] += cost_gap
            totals["proj_abs_cost_gap"] += abs(cost_gap)
            totals["proj_rel_cost_gap"] += (cost_gap / ot_cost)
            totals["proj_n_cost"] += 1


def _report(name, totals, eval_sinkhorn_iters):
    if totals["count"] == 0:
        return

    avg_loss = totals["loss"] / totals["count"]
    avg_entropy = totals["entropy"] / totals["count"]
    avg_kl = totals["kl"] / totals["count"]
    avg_kl_norm = totals["kl_norm"] / totals["count"]
    avg_marg_l1 = totals["marg_l1"] / totals["count"]
    avg_marg_linf = totals["marg_linf"] / totals["count"]
    target_val = totals["target_val"]
    rel_marg_l1 = avg_marg_l1 / target_val if (target_val and target_val > 0.0) else float("nan")
    rel_marg_linf = avg_marg_linf / target_val if (target_val and target_val > 0.0) else float("nan")
    avg_cost_gap = (totals["cost_gap"] / totals["n_cost"]) if totals["n_cost"] > 0 else float("nan")
    avg_abs_cost_gap = (totals["abs_cost_gap"] / totals["n_cost"]) if totals["n_cost"] > 0 else float("nan")
    avg_rel_cost_gap = (totals["rel_cost_gap"] / totals["n_cost"]) if totals["n_cost"] > 0 else float("nan")

    print(f"\nBaseline: {name}")
    print(f"Baseline CE ({totals['count']} samples): {avg_loss:.6e}")
    print(f"Baseline Target Entropy: {avg_entropy:.6e}")
    print(f"Baseline KL (CE - H): {avg_kl:.6e}")
    print(f"Baseline KL/logN: {avg_kl_norm:.6e}")
    print(f"Baseline Column Marginal L1: {avg_marg_l1:.6e}")
    print(f"Baseline Column Marginal Linf: {avg_marg_linf:.6e}")
    print(f"Baseline Column Marginal L1 (rel): {rel_marg_l1:.6e}")
    print(f"Baseline Column Marginal Linf (rel): {rel_marg_linf:.6e}")
    if totals["n_cost"] > 0:
        print(f"Baseline Cost Gap (pred - true): {avg_cost_gap:.6e}")
        print(f"Baseline |Cost Gap|: {avg_abs_cost_gap:.6e}")
        print(f"Baseline Relative Cost Gap: {avg_rel_cost_gap:.6e}")

    if eval_sinkhorn_iters > 0:
        avg_proj_marg_l1 = totals["proj_marg_l1"] / totals["count"]
        avg_proj_marg_linf = totals["proj_marg_linf"] / totals["count"]
        rel_proj_l1 = avg_proj_marg_l1 / target_val if (target_val and target_val > 0.0) else float("nan")
        rel_proj_linf = avg_proj_marg_linf / target_val if (target_val and target_val > 0.0) else float("nan")
        avg_proj_cost_gap = (totals["proj_cost_gap"] / totals["proj_n_cost"]) if totals["proj_n_cost"] > 0 else float("nan")
        avg_proj_abs_cost_gap = (totals["proj_abs_cost_gap"] / totals["proj_n_cost"]) if totals["proj_n_cost"] > 0 else float("nan")
        avg_proj_rel_cost_gap = (totals["proj_rel_cost_gap"] / totals["proj_n_cost"]) if totals["proj_n_cost"] > 0 else float("nan")
        print(f"Projected Column Marginal L1: {avg_proj_marg_l1:.6e}")
        print(f"Projected Column Marginal Linf: {avg_proj_marg_linf:.6e}")
        print(f"Projected Column Marginal L1 (rel): {rel_proj_l1:.6e}")
        print(f"Projected Column Marginal Linf (rel): {rel_proj_linf:.6e}")
        if totals["proj_n_cost"] > 0:
            print(f"Projected Cost Gap (pred - true): {avg_proj_cost_gap:.6e}")
            print(f"Projected |Cost Gap|: {avg_proj_abs_cost_gap:.6e}")
            print(f"Projected Relative Cost Gap: {avg_proj_rel_cost_gap:.6e}")


def evaluate_baselines(dataset, device, mode, n_test_samples=100, args=None):
    if mode == "barycentric":
        print("Baselines skipped for barycentric mode.")
        return

    test_data = dataset["test"]
    n_test = min(n_test_samples, len(test_data))
    eval_sinkhorn_iters = int(getattr(args, "eval_sinkhorn_iters", 0))
    eval_sinkhorn_eps = float(getattr(args, "eval_sinkhorn_eps", 1e-9))

    uniform_totals = _init_totals()
    kernel_totals = _init_totals()
    kernel_skipped = 0

    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            source_points = torch.tensor(np.array(sample["source_points"]), device=device, dtype=torch.float32)
            target_points = torch.tensor(np.array(sample["target_points"]), device=device, dtype=torch.float32)
            plan = torch.tensor(np.array(sample["transport_plan"]), device=device, dtype=torch.float32)
            M, _ = plan.shape

            if mode == "plan":
                target = plan
            elif mode == "coupling":
                target = plan / float(M)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            ot_cost = float(sample.get("ot_cost", float("nan")))
            cost_matrix = torch.sum((source_points[:, None, :] - target_points[None, :, :]) ** 2, dim=-1)

            uniform_logits = torch.zeros_like(plan)
            _accumulate(
                uniform_totals,
                uniform_logits,
                target,
                cost_matrix,
                ot_cost,
                eval_sinkhorn_iters,
                eval_sinkhorn_eps,
            )

            eps = _get_baseline_epsilon(sample, args)
            if eps is None or eps <= 0.0:
                kernel_skipped += 1
                continue
            kernel_logits = -cost_matrix / float(eps)
            _accumulate(
                kernel_totals,
                kernel_logits,
                target,
                cost_matrix,
                ot_cost,
                eval_sinkhorn_iters,
                eval_sinkhorn_eps,
            )

    _report("Uniform (row)", uniform_totals, eval_sinkhorn_iters)
    if kernel_totals["count"] > 0:
        _report("Kernel-only (row)", kernel_totals, eval_sinkhorn_iters)
    else:
        print("\nBaseline: Kernel-only (row) skipped (epsilon unavailable).")
        if kernel_skipped > 0:
            print("Provide --baseline_epsilon to enable this baseline.")


def main():
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if not os.path.exists(args.data_path):
        print(f"Dataset directory not found: {args.data_path}")
        print("Run: python Data/transport_plan_data/generate_transport_plan_data.py")
        return

    dataset = load_from_disk(args.data_path)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    evaluate_baselines(
        dataset,
        device,
        mode=args.mode,
        n_test_samples=args.n_test_samples,
        args=args,
    )


if __name__ == "__main__":
    main()
