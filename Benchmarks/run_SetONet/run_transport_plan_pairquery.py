import argparse
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.SetONet import SetONet
from Models.utils.config_utils import save_experiment_configuration


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SetONet with pair queries for OT transport-plan prediction."
    )

    # Data parameters
    default_data_path = os.path.join(project_root, "Data", "transport_data", "transport_plan_dataset")
    parser.add_argument("--data_path", type=str, default=default_data_path, help="Path to transport-plan dataset")

    # Pair-query parameters
    parser.add_argument("--pair_k_pos", type=int, default=4, help="Positive target samples per source")
    parser.add_argument("--pair_k_neg", type=int, default=8, help="Negative target samples per source")
    parser.add_argument(
        "--neg_sampling",
        type=str,
        default="uniform",
        choices=["uniform"],
        help="Negative sampling strategy (currently only uniform without replacement).",
    )
    parser.add_argument("--disable_cost_prior", action="store_true", help="Disable quadratic OT cost prior in logits")
    parser.add_argument("--epsilon_override", type=float, default=-1.0, help="Override epsilon if dataset epsilon <= 0")

    # Evaluation chunking (prevents OOM if you later increase M/N)
    parser.add_argument(
        "--eval_chunk_n_target",
        type=int,
        default=64,
        help="Chunk size over target points for full evaluation (>=1).",
    )

    # Model architecture
    parser.add_argument("--son_p_dim", type=int, default=128, help="Latent dimension p for SetONet")
    parser.add_argument("--son_phi_hidden", type=int, default=256, help="Hidden size for SetONet phi network")
    parser.add_argument("--son_rho_hidden", type=int, default=256, help="Hidden size for SetONet rho network")
    parser.add_argument("--son_trunk_hidden", type=int, default=256, help="Hidden size for SetONet trunk network")
    parser.add_argument("--son_n_trunk_layers", type=int, default=4, help="Number of layers in SetONet trunk network")
    parser.add_argument("--son_phi_output_size", type=int, default=32, help="Output size of SetONet phi network before aggregation")
    parser.add_argument(
        "--son_aggregation",
        type=str,
        default="attention",
        choices=["mean", "attention", "sum"],
        help="Aggregation type for SetONet",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu", "swish"],
        help="Activation function for SetONet networks",
    )
    parser.add_argument(
        "--son_branch_head_type",
        type=str,
        default="standard",
        choices=["standard", "petrov_attention", "galerkin_pou", "quadrature", "adaptive_quadrature"],
        help="Branch head type for SetONet.",
    )
    parser.add_argument("--son_pg_dk", type=int, default=None, help="PG attention key/query dim (default: son_phi_output_size)")
    parser.add_argument("--son_pg_dv", type=int, default=None, help="PG attention value dim (default: son_phi_output_size)")
    parser.add_argument("--son_pg_no_logw", action="store_true", help="Disable log(sensor_weights) term for PG attention.")
    parser.add_argument("--son_galerkin_dk", type=int, default=None, help="Galerkin PoU key/query dim (default: son_phi_output_size)")
    parser.add_argument("--son_galerkin_dv", type=int, default=None, help="Galerkin PoU value dim (default: son_phi_output_size)")
    parser.add_argument("--son_quad_dk", type=int, default=None, help="Quadrature key/query dim (default: son_phi_output_size)")
    parser.add_argument("--son_quad_dv", type=int, default=None, help="Quadrature value dim (default: son_phi_output_size)")
    parser.add_argument(
        "--son_galerkin_normalize",
        type=str,
        default="total",
        choices=["none", "total", "token"],
        help="Galerkin PoU normalization.",
    )
    parser.add_argument(
        "--son_galerkin_learn_temperature",
        action="store_true",
        help="Learn temperature for Galerkin PoU softmax sharpness.",
    )
    parser.add_argument("--son_adapt_quad_rank", type=int, default=4, help="Adaptive quadrature low-rank update rank R")
    parser.add_argument("--son_adapt_quad_hidden", type=int, default=64, help="Adaptive quadrature adapter MLP hidden dimension")
    parser.add_argument("--son_adapt_quad_scale", type=float, default=0.1, help="Adaptive quadrature tanh-bounded update scale")

    # Training parameters
    parser.add_argument("--son_lr", type=float, default=5e-4, help="Learning rate for SetONet")
    parser.add_argument("--son_epochs", type=int, default=50000, help="Number of steps for SetONet")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (number of dataset samples per step)")
    parser.add_argument(
        "--pos_encoding_type",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "skip"],
        help="Positional encoding type for SetONet",
    )
    parser.add_argument("--pos_encoding_dim", type=int, default=64, help="Dimension for positional encoding")
    parser.add_argument("--pos_encoding_max_freq", type=float, default=0.01, help="Max frequency for sinusoidal positional encoding")
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        nargs="+",
        default=[15000, 30000, 125000, 175000, 1250000, 1500000],
        help="List of steps for LR decay milestones.",
    )
    parser.add_argument(
        "--lr_schedule_gammas",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5],
        help="List of multiplicative factors for LR decay.",
    )

    # Model loading
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to pre-trained SetONet model")

    # Random seed and device
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device to use.")

    # TensorBoard logging
    # Keep your original behavior (TB enabled by default), but finally allow disabling.
    parser.add_argument(
        "--enable_tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging (enabled by default; use --disable_tensorboard to turn off).",
    )
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument("--tb_eval_frequency", type=int, default=2000, help="How often to run full evaluation (in steps)")
    parser.add_argument("--tb_test_samples", type=int, default=10, help="Number of test samples for full evaluation")

    # Logging directory (overrides default if provided)
    parser.add_argument("--log_dir", type=str, default=None, help="Custom log directory (overrides default timestamped dir)")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def setup_logging(project_root, custom_log_dir=None):
    """Setup logging directory or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join(project_root, "logs")
        model_folder_name = "SetONet_transport_plan_pairquery"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir


def get_activation_function(activation_name):
    """Get activation function by name."""
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
    }
    return activation_map.get(activation_name.lower(), nn.ReLU)


def create_model(args, device):
    """Create SetONet model for pair-query transport-plan prediction."""
    activation_fn = get_activation_function(args.activation_fn)
    model = SetONet(
        input_size_src=2,
        output_size_src=1,
        input_size_tgt=4,   # pair query: (x,y) in R^4
        output_size_tgt=1,  # scalar logit per pair
        p=args.son_p_dim,
        phi_hidden_size=args.son_phi_hidden,
        rho_hidden_size=args.son_rho_hidden,
        trunk_hidden_size=args.son_trunk_hidden,
        n_trunk_layers=args.son_n_trunk_layers,
        activation_fn=activation_fn,
        use_deeponet_bias=True,
        phi_output_size=args.son_phi_output_size,
        initial_lr=args.son_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
        pos_encoding_type=args.pos_encoding_type,
        pos_encoding_dim=args.pos_encoding_dim,
        pos_encoding_max_freq=args.pos_encoding_max_freq,
        aggregation_type=args.son_aggregation,
        use_positional_encoding=(args.pos_encoding_type != "skip"),
        attention_n_tokens=1,
        branch_head_type=args.son_branch_head_type,
        pg_dk=args.son_pg_dk,
        pg_dv=args.son_pg_dv,
        pg_use_logw=(not args.son_pg_no_logw),
        galerkin_dk=args.son_galerkin_dk,
        galerkin_dv=args.son_galerkin_dv,
        quad_dk=args.son_quad_dk,
        quad_dv=args.son_quad_dv,
        galerkin_normalize=args.son_galerkin_normalize,
        galerkin_learn_temperature=args.son_galerkin_learn_temperature,
        adapt_quad_rank=args.son_adapt_quad_rank,
        adapt_quad_hidden=args.son_adapt_quad_hidden,
        adapt_quad_scale=args.son_adapt_quad_scale,
    ).to(device)
    return model


def _normalize_rows(target, eps=1e-12):
    target = target.clamp_min(0.0)
    row_sums = target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return target / row_sums


def compute_plan_metrics(pred_logits, target_W, eps=1e-12):
    """
    pred_logits: (B,M,N) logits
    target_W:    (B,M,N) row-stochastic target (or close; we renormalize)
    Returns CE, entropy, KL, KL/logN
    """
    target_W = _normalize_rows(target_W, eps=eps)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    ce = -(target_W * log_probs).sum(dim=-1).mean()
    log_target = torch.log(target_W.clamp_min(eps))
    entropy = -(target_W * log_target).sum(dim=-1).mean()
    kl = ce - entropy
    log_n = math.log(float(target_W.shape[-1]))
    kl_norm = kl / log_n if log_n > 0.0 else kl * 0.0
    return ce, entropy, kl, kl_norm


def compute_column_marginal_errors(w_hat):
    """
    w_hat: (M,N) row-stochastic plan W
    Target coupling marginal is b = 1/N, where coupling is P = W/M.
    P's column marginal is mean_i W_ij.
    """
    col_marg = w_hat.mean(dim=0)  # = sum_i W_ij / M
    n_target = w_hat.shape[-1]
    target = 1.0 / float(n_target)
    diff = col_marg - target
    l1 = diff.abs().mean()
    linf = diff.abs().max()
    return l1, linf


def _effective_epsilon(sample, args):
    eps = float(sample.get("epsilon", -1.0))
    if eps > 0.0:
        return eps
    if args.epsilon_override > 0.0:
        return float(args.epsilon_override)
    return None


# -----------------------------------------------------------------------------
# Candidate sampling (FIXED: unique + pos/neg disjoint)
# -----------------------------------------------------------------------------

def sample_candidate_indices(W, k_pos, k_neg, eps=1e-12):
    """
    W: (M,N) row-stochastic (or close).
    Returns cand_idx: (M, K) with K=k_pos+k_neg, unique per row, and neg disjoint from pos.
    """
    if W.dim() != 2:
        raise ValueError(f"W must be (M,N), got {W.shape}")

    M, N = W.shape
    if k_pos <= 0:
        raise ValueError("pair_k_pos must be > 0.")
    if k_neg < 0:
        raise ValueError("pair_k_neg must be >= 0.")
    if k_pos + k_neg > N:
        raise ValueError(f"k_pos + k_neg must be <= N. Got {k_pos}+{k_neg} > {N}")

    # Safe renormalize
    W = W.clamp_min(0.0)
    row_sums = W.sum(dim=1, keepdim=True)
    W = torch.where(row_sums > eps, W / row_sums.clamp_min(eps), torch.full_like(W, 1.0 / N))

    # Positives: WITHOUT replacement
    pos_idx = torch.multinomial(W, k_pos, replacement=False)  # (M,k_pos)

    if k_neg == 0:
        return pos_idx

    # Uniform negatives WITHOUT replacement and disjoint from pos:
    # sample random scores and take top-k among indices not in pos.
    scores = torch.rand((M, N), device=W.device)
    scores.scatter_(1, pos_idx, -1.0)  # exclude positives
    neg_idx = scores.topk(k=k_neg, dim=1).indices  # (M,k_neg)

    cand_idx = torch.cat([pos_idx, neg_idx], dim=1)  # (M,K)
    return cand_idx


# -----------------------------------------------------------------------------
# Batch builder (FIXED: cache Y_fixed; safe shapes)
# -----------------------------------------------------------------------------

def _build_pair_batch(samples, device, Y_fixed, k_pos, k_neg, eps=1e-12):
    """
    Build a batch for pair-query training.

    Returns:
      xs:          (B,M,2)
      us:          (B,M,1)
      ys:          (B,M*K,4)  pair queries flattened
      targets:     (B,M,K)    candidate-restricted target row distributions
      x_sources:   (B,M,2)
      y_candidates:(B,M,K,2)
      eps_tensor:  (B,)       epsilon per sample (may be <=0)
    """
    xs_list, us_list, ys_list = [], [], []
    target_list, x_list, y_list, eps_list = [], [], [], []

    Y = Y_fixed  # (N,2) already on device
    N = int(Y.shape[0])

    for sample in samples:
        X = torch.tensor(np.array(sample["source_points"]), device=device, dtype=torch.float32)
        W = torch.tensor(np.array(sample["transport_plan"]), device=device, dtype=torch.float32)

        W = W.clamp_min(0.0)
        W = W / W.sum(dim=1, keepdim=True).clamp_min(eps)

        M = int(X.shape[0])
        if int(W.shape[0]) != M or int(W.shape[1]) != N:
            raise ValueError(f"Shape mismatch: X={X.shape}, W={W.shape}, Y_fixed={Y.shape}")

        cand_idx = sample_candidate_indices(W, k_pos=k_pos, k_neg=k_neg, eps=eps)  # (M,K)
        K = int(cand_idx.shape[1])

        cand_w = W.gather(1, cand_idx)  # (M,K)
        target = cand_w / cand_w.sum(dim=1, keepdim=True).clamp_min(eps)  # (M,K)

        cand_y = Y[cand_idx]  # (M,K,2)

        x_rep = X[:, None, :].expand(M, K, 2)            # (M,K,2)
        pair_queries = torch.cat([x_rep, cand_y], dim=-1)  # (M,K,4)
        pair_queries = pair_queries.reshape(M * K, 4)      # (M*K,4)

        xs_list.append(X)
        us_list.append(torch.ones(M, 1, device=device, dtype=torch.float32))
        ys_list.append(pair_queries)

        target_list.append(target)
        x_list.append(X)
        y_list.append(cand_y)
        eps_list.append(float(sample.get("epsilon", -1.0)))

    xs = torch.stack(xs_list, dim=0)           # (B,M,2)
    us = torch.stack(us_list, dim=0)           # (B,M,1)
    ys = torch.stack(ys_list, dim=0)           # (B,M*K,4)
    targets = torch.stack(target_list, dim=0)  # (B,M,K)
    x_sources = torch.stack(x_list, dim=0)     # (B,M,2)
    y_candidates = torch.stack(y_list, dim=0)  # (B,M,K,2)
    eps_tensor = torch.tensor(eps_list, device=device, dtype=torch.float32)  # (B,)

    return xs, us, ys, targets, x_sources, y_candidates, eps_tensor


# -----------------------------------------------------------------------------
# Cost prior (logit <- logit - ||x-y||^2 / eps)
# -----------------------------------------------------------------------------

def _apply_cost_prior(logits, x_sources, y_candidates, eps_tensor):
    """
    logits:      (B,M,K)
    x_sources:   (B,M,2)
    y_candidates:(B,M,K,2)
    eps_tensor:  (B,)
    """
    dist2 = torch.sum((x_sources.unsqueeze(2) - y_candidates) ** 2, dim=-1)  # (B,M,K)
    valid = eps_tensor > 0.0
    if not torch.any(valid):
        return logits
    eps_safe = torch.where(valid, eps_tensor, torch.ones_like(eps_tensor))
    prior = dist2 / eps_safe[:, None, None]
    return logits - prior * valid[:, None, None]


# -----------------------------------------------------------------------------
# Full evaluation (FIXED: cache Y_fixed + chunked prediction)
# -----------------------------------------------------------------------------

def predict_full_logits_chunked(model, X, Y_fixed, device, eps=None, chunk_n_target=64):
    """
    Compute full logits matrix (M,N) via chunking over target points.
    Applies cost prior if eps is provided and > 0.
    """
    if chunk_n_target < 1:
        raise ValueError("chunk_n_target must be >= 1")

    X = X.to(device)
    Y = Y_fixed.to(device)
    M = int(X.shape[0])
    N = int(Y.shape[0])

    xs = X.unsqueeze(0)  # (1,M,2)
    us = torch.ones(1, M, 1, device=device, dtype=torch.float32)

    logits_full = torch.empty((M, N), device=device, dtype=torch.float32)

    with torch.inference_mode():
        for j0 in range(0, N, chunk_n_target):
            j1 = min(N, j0 + chunk_n_target)
            Yc = Y[j0:j1]  # (Nc,2)
            Nc = int(Yc.shape[0])

            x_rep = X[:, None, :].expand(M, Nc, 2)     # (M,Nc,2)
            y_rep = Yc[None, :, :].expand(M, Nc, 2)    # (M,Nc,2)
            pair = torch.cat([x_rep, y_rep], dim=-1).reshape(1, M * Nc, 4)  # (1,M*Nc,4)

            out = model(xs, us, pair).squeeze(0).squeeze(-1).reshape(M, Nc)  # (M,Nc)

            if eps is not None and float(eps) > 0.0:
                cost = torch.sum((X[:, None, :] - Yc[None, :, :]) ** 2, dim=-1)  # (M,Nc)
                out = out - cost / float(eps)

            logits_full[:, j0:j1] = out

    return logits_full


def evaluate_full(model, dataset, device, args, Y_fixed, n_test_samples=10):
    """
    Evaluate model by computing full W_hat for each sample (exact, but chunked).
    """
    model.eval()
    test_data = dataset["test"]
    n_test = min(n_test_samples, len(test_data))

    total_ce = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_kl_norm = 0.0
    total_marg_l1 = 0.0
    total_marg_linf = 0.0
    total_cost_gap = 0.0
    total_abs_cost_gap = 0.0
    total_rel_cost_gap = 0.0
    n_cost = 0

    use_cost_prior = not args.disable_cost_prior
    chunk_n_target = int(args.eval_chunk_n_target)

    with torch.inference_mode():
        for i in range(n_test):
            sample = test_data[i]
            X = torch.tensor(np.array(sample["source_points"]), device=device, dtype=torch.float32)
            W_true = torch.tensor(np.array(sample["transport_plan"]), device=device, dtype=torch.float32)

            eps = _effective_epsilon(sample, args) if use_cost_prior else None

            logits = predict_full_logits_chunked(
                model,
                X=X,
                Y_fixed=Y_fixed,
                device=device,
                eps=eps,
                chunk_n_target=chunk_n_target,
            )  # (M,N)

            # CE/KL against full W_true
            ce, entropy, kl, kl_norm = compute_plan_metrics(logits.unsqueeze(0), W_true.unsqueeze(0))
            total_ce += ce.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
            total_kl_norm += kl_norm.item()

            # Predicted plan and marginal errors
            w_hat = torch.softmax(logits, dim=-1)  # (M,N)
            marg_l1, marg_linf = compute_column_marginal_errors(w_hat)
            total_marg_l1 += marg_l1.item()
            total_marg_linf += marg_linf.item()

            # Cost gap vs stored ot_cost (computed on coupling P = W/M)
            ot_cost = float(sample.get("ot_cost", float("nan")))
            if (not np.isnan(ot_cost)) and (ot_cost != 0.0):
                Y = Y_fixed  # cached
                cost_matrix = torch.sum((X[:, None, :] - Y[None, :, :]) ** 2, dim=-1)  # (M,N)
                p_hat = w_hat / float(w_hat.shape[0])  # P = W/M
                cost_pred = torch.sum(p_hat * cost_matrix).item()
                cost_gap = cost_pred - ot_cost

                total_cost_gap += cost_gap
                total_abs_cost_gap += abs(cost_gap)
                total_rel_cost_gap += (cost_gap / ot_cost)
                n_cost += 1

    avg_ce = total_ce / n_test
    avg_entropy = total_entropy / n_test
    avg_kl = total_kl / n_test
    avg_kl_norm = total_kl_norm / n_test
    avg_marg_l1 = total_marg_l1 / n_test
    avg_marg_linf = total_marg_linf / n_test
    avg_cost_gap = (total_cost_gap / n_cost) if n_cost > 0 else float("nan")
    avg_abs_cost_gap = (total_abs_cost_gap / n_cost) if n_cost > 0 else float("nan")
    avg_rel_cost_gap = (total_rel_cost_gap / n_cost) if n_cost > 0 else float("nan")

    print(f"Test CE ({n_test} samples): {avg_ce:.6e}")
    print(f"Test Target Entropy: {avg_entropy:.6e}")
    print(f"Test KL (CE - H): {avg_kl:.6e}")
    print(f"Test KL/logN: {avg_kl_norm:.6e}")
    print(f"Test Column Marginal L1: {avg_marg_l1:.6e}")
    print(f"Test Column Marginal Linf: {avg_marg_linf:.6e}")
    if n_cost > 0:
        print(f"Test Cost Gap (pred - true): {avg_cost_gap:.6e}")
        print(f"Test |Cost Gap|: {avg_abs_cost_gap:.6e}")
        print(f"Test Relative Cost Gap: {avg_rel_cost_gap:.6e}")

    model.train()
    return {
        "loss": avg_ce,
        "entropy": avg_entropy,
        "kl": avg_kl,
        "kl_norm": avg_kl_norm,
        "marg_l1": avg_marg_l1,
        "marg_linf": avg_marg_linf,
        "cost_gap": avg_cost_gap,
        "abs_cost_gap": avg_abs_cost_gap,
        "rel_cost_gap": avg_rel_cost_gap,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_dir = setup_logging(project_root, args.log_dir)

    if not os.path.exists(args.data_path):
        print(f"Dataset directory not found: {args.data_path}")
        print("Run: python Data/transport_plan_data/generate_transport_plan_data.py")
        return

    dataset = load_from_disk(args.data_path)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")

    # Cache Y_fixed once (dataset duplicates it across samples)
    y0 = np.array(dataset["train"][0]["target_points"], dtype=np.float32)
    Y_fixed = torch.tensor(y0, device=device, dtype=torch.float32)
    print(f"Cached Y_fixed: {tuple(Y_fixed.shape)} (fixed across dataset)")

    # Create model
    model = create_model(args, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    if args.load_model_path:
        print(f"Loading pre-trained model from: {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))

    # TensorBoard
    enable_tb = bool(args.enable_tensorboard) and (not bool(args.disable_tensorboard))
    writer = None
    if enable_tb:
        print("Setting up TensorBoard logging...")
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        print(f"To view logs, run: tensorboard --logdir {tb_log_dir}")
    else:
        print("TensorBoard logging disabled.")

    train_data = dataset["train"]
    use_cost_prior = not args.disable_cost_prior

    # One-time epsilon warning
    warned_no_eps = False

    # Sanity checks for k_pos/k_neg
    N = int(Y_fixed.shape[0])
    if args.pair_k_pos <= 0:
        raise ValueError("--pair_k_pos must be > 0")
    if args.pair_k_neg < 0:
        raise ValueError("--pair_k_neg must be >= 0")
    if args.pair_k_pos + args.pair_k_neg > N:
        raise ValueError(f"--pair_k_pos + --pair_k_neg must be <= N={N}")

    print(f"\nStarting training for {args.son_epochs} steps...")
    bar = trange(args.son_epochs)
    for _ in bar:
        current_lr = model._update_lr()

        # Sample dataset items
        batch_indices = np.random.randint(0, len(train_data), size=args.batch_size).tolist()
        batch_samples = [train_data[int(i)] for i in batch_indices]

        xs, us, ys, targets, x_sources, y_candidates, eps_tensor = _build_pair_batch(
            batch_samples,
            device=device,
            Y_fixed=Y_fixed,
            k_pos=args.pair_k_pos,
            k_neg=args.pair_k_neg,
        )

        # epsilon override for missing epsilon
        if args.epsilon_override > 0.0:
            override = torch.tensor(args.epsilon_override, device=device, dtype=eps_tensor.dtype)
            eps_tensor = torch.where(eps_tensor > 0.0, eps_tensor, override)

        # Apply cost prior if enabled and available
        preds = model(xs, us, ys).squeeze(-1)  # (B, M*K)
        B, M, K = targets.shape
        preds = preds.reshape(B, M, K)

        if use_cost_prior:
            if (not torch.any(eps_tensor > 0.0)) and (args.epsilon_override <= 0.0) and (not warned_no_eps):
                print("Warning: epsilon not available in dataset and no --epsilon_override given; cost prior will be skipped.")
                warned_no_eps = True
            preds = _apply_cost_prior(preds, x_sources, y_candidates, eps_tensor)

        # Sampled-softmax CE over candidate set
        log_probs = F.log_softmax(preds, dim=-1)
        ce_loss = -(targets * log_probs).sum(dim=-1).mean()
        loss = ce_loss

        model.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.opt.step()

        model.total_steps += 1

        # Logging
        if writer is not None:
            writer.add_scalar("Training/CE_Loss", ce_loss.item(), model.total_steps)

            if model.total_steps % args.tb_eval_frequency == 0:
                eval_metrics = evaluate_full(
                    model,
                    dataset,
                    device,
                    args=args,
                    Y_fixed=Y_fixed,
                    n_test_samples=args.tb_test_samples,
                )
                writer.add_scalar("Evaluation/CE_Loss", eval_metrics["loss"], model.total_steps)
                writer.add_scalar("Evaluation/Target_Entropy", eval_metrics["entropy"], model.total_steps)
                writer.add_scalar("Evaluation/KL", eval_metrics["kl"], model.total_steps)
                writer.add_scalar("Evaluation/KL_logN", eval_metrics["kl_norm"], model.total_steps)
                writer.add_scalar("Evaluation/Column_Marginal_L1", eval_metrics["marg_l1"], model.total_steps)
                writer.add_scalar("Evaluation/Column_Marginal_Linf", eval_metrics["marg_linf"], model.total_steps)
                if not np.isnan(eval_metrics["cost_gap"]):
                    writer.add_scalar("Evaluation/Cost_Gap", eval_metrics["cost_gap"], model.total_steps)
                    writer.add_scalar("Evaluation/Abs_Cost_Gap", eval_metrics["abs_cost_gap"], model.total_steps)
                    writer.add_scalar("Evaluation/Rel_Cost_Gap", eval_metrics["rel_cost_gap"], model.total_steps)

        bar.set_description(
            f"Step {model.total_steps} | Loss: {loss.item():.4e} | Grad Norm: {float(norm):.2f} | LR: {current_lr:.2e}"
        )

    print("\nEvaluating model...")
    eval_metrics = evaluate_full(model, dataset, device, args=args, Y_fixed=Y_fixed, n_test_samples=args.tb_test_samples)

    save_experiment_configuration(
        args,
        model,
        dataset,
        dataset_wrapper=None,
        device=device,
        log_dir=log_dir,
        dataset_type="transport_plan_pairquery",
        test_results={
            "mse_loss": None,
            "n_test_samples": args.tb_test_samples,
            "ce_loss": eval_metrics["loss"],
            "kl_norm": eval_metrics["kl_norm"],
            "marg_linf": eval_metrics["marg_linf"],
            "rel_cost_gap": eval_metrics["rel_cost_gap"],
        },
    )

    model_save_path = os.path.join(log_dir, "transport_plan_pairquery_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
