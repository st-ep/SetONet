import torch
import numpy as np
import sys
import os
from datetime import datetime
import argparse
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from Models.VIDON import VIDON
from Models.utils.config_utils_vidon import save_experiment_configuration
from Data.transport_plan_data.transport_plan_dataset import load_transport_plan_dataset
from Plotting.plot_transport_plan_utils import (
    plot_transport_barycentric_results,
    plot_transport_row_distribution_results,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VIDON for OT transport-plan prediction.')

    # Data parameters
    default_data_path = os.path.join(project_root, 'Data', 'transport_data', 'transport_plan_dataset')
    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help='Path to transport-plan dataset')
    parser.add_argument('--mode', type=str, default='plan',
                        choices=['plan', 'coupling', 'barycentric'],
                        help='Learning target: row-conditional plan, coupling, or barycentric map')
    parser.add_argument('--marginal_penalty_weight', type=float, default=1.0,
                        help='Weight for target marginal L1 penalty (plan/coupling only)')
    parser.add_argument('--eval_sinkhorn_iters', type=int, default=0,
                        help='Eval-only Sinkhorn-Knopp iterations to project onto marginals (0 disables)')
    parser.add_argument('--eval_sinkhorn_eps', type=float, default=1e-9,
                        help='Stability epsilon for eval Sinkhorn-Knopp projection')

    # Model architecture - VIDON specific
    parser.add_argument('--vidon_p_dim', type=int, default=128, help='Number of trunk basis functions (excluding τ0)')
    parser.add_argument('--vidon_n_heads', type=int, default=4, help='Number of attention heads (H)')
    parser.add_argument('--vidon_d_enc', type=int, default=40, help='Encoding dimension (d_enc)')
    parser.add_argument('--vidon_head_output_size', type=int, default=64, help='Output dimension of each head')

    # Encoder networks (Ψc, Ψv)
    parser.add_argument('--vidon_enc_hidden', type=int, default=40, help='Hidden size for encoder networks')
    parser.add_argument('--vidon_enc_n_layers', type=int, default=4, help='Number of layers in encoder networks')

    # Head MLPs (ωe, νe)
    parser.add_argument('--vidon_head_hidden', type=int, default=128, help='Hidden size for head MLPs')
    parser.add_argument('--vidon_head_n_layers', type=int, default=4, help='Number of layers in head MLPs')

    # Combiner Φ
    parser.add_argument('--vidon_combine_hidden', type=int, default=256, help='Hidden size for combiner network')
    parser.add_argument('--vidon_combine_n_layers', type=int, default=4, help='Number of layers in combiner network')

    # Trunk network τ
    parser.add_argument('--vidon_trunk_hidden', type=int, default=256, help='Hidden size for trunk network')
    parser.add_argument('--vidon_n_trunk_layers', type=int, default=4, help='Number of layers in trunk network')

    parser.add_argument('--activation_fn', type=str, default='relu', choices=['relu', 'tanh', 'gelu', 'swish'],
                        help='Activation function for networks')

    # Training parameters
    parser.add_argument('--vidon_lr', type=float, default=5e-4, help='Learning rate for VIDON')
    parser.add_argument('--vidon_epochs', type=int, default=50000, help='Number of steps for VIDON')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr_schedule_steps', type=int, nargs='+',
                        default=[15000, 30000, 125000, 175000, 1250000, 1500000],
                        help='List of steps for LR decay milestones.')
    parser.add_argument('--lr_schedule_gammas', type=float, nargs='+',
                        default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5],
                        help='List of multiplicative factors for LR decay.')

    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained VIDON model')

    # Random seed and device
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device to use.')

    # TensorBoard logging
    parser.add_argument('--enable_tensorboard', action='store_true', default=True,
                        help='Enable TensorBoard logging of training metrics')
    parser.add_argument('--tb_eval_frequency', type=int, default=1000,
                        help='How often to evaluate on test set for TensorBoard logging (in steps)')
    parser.add_argument('--tb_test_samples', type=int, default=1000,
                        help='Number of test samples to use for TensorBoard evaluation')

    # Logging directory (overrides default if provided)
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Custom log directory (overrides default timestamped dir)')

    return parser.parse_args()


def setup_logging(project_root, custom_log_dir=None):
    """Setup logging directory or use custom directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base_in_project = os.path.join(project_root, 'logs')
        model_folder_name = 'VIDON_transport_plan'
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f'Logging to: {log_dir}')
    return log_dir


def get_activation_function(activation_name):
    """Get activation function by name."""
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'swish': nn.SiLU  # SiLU is equivalent to Swish
    }
    return activation_map.get(activation_name.lower(), nn.ReLU)


def create_model(args, device, output_size_tgt):
    """Create VIDON model for transport-plan prediction."""
    activation_fn = get_activation_function(args.activation_fn)

    model = VIDON(
        input_size_src=2,
        output_size_src=1,
        input_size_tgt=2,
        output_size_tgt=output_size_tgt,
        p=args.vidon_p_dim,
        n_heads=args.vidon_n_heads,
        d_enc=args.vidon_d_enc,
        head_output_size=args.vidon_head_output_size,
        enc_hidden_size=args.vidon_enc_hidden,
        enc_n_layers=args.vidon_enc_n_layers,
        head_hidden_size=args.vidon_head_hidden,
        head_n_layers=args.vidon_head_n_layers,
        combine_hidden_size=args.vidon_combine_hidden,
        combine_n_layers=args.vidon_combine_n_layers,
        trunk_hidden_size=args.vidon_trunk_hidden,
        n_trunk_layers=args.vidon_n_trunk_layers,
        activation_fn=activation_fn,
        initial_lr=args.vidon_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
    ).to(device)

    return model


def _normalize_rows(target, eps=1e-12):
    target = target.clamp_min(0.0)
    row_sums = target.sum(dim=-1, keepdim=True).clamp_min(eps)
    return target / row_sums


def compute_plan_metrics(pred, target, eps=1e-12):
    """Compute CE, target entropy, KL, and KL/logN for plan/coupling targets."""
    target = _normalize_rows(target, eps=eps)
    log_probs = F.log_softmax(pred, dim=-1)
    ce = -(target * log_probs).sum(dim=-1).mean()
    log_target = torch.log(target.clamp_min(eps))
    entropy = -(target * log_target).sum(dim=-1).mean()
    kl = ce - entropy
    log_n = math.log(float(target.shape[-1]))
    if log_n <= 0.0:
        kl_norm = kl * 0.0
    else:
        kl_norm = kl / log_n
    return ce, entropy, kl, kl_norm


def compute_column_marginal_errors(w_hat):
    """Compute L1 and Linf errors for the target marginal of P = W/M."""
    if w_hat.dim() == 3:
        col_marg = w_hat.mean(dim=1)
    elif w_hat.dim() == 2:
        col_marg = w_hat.mean(dim=0)
    else:
        raise ValueError(f'Unexpected w_hat shape: {tuple(w_hat.shape)}')
    n_target = w_hat.shape[-1]
    target = 1.0 / float(n_target)
    diff = col_marg - target
    l1 = diff.abs().mean()
    linf = diff.abs().max()
    return l1, linf


def sinkhorn_project(p_hat, n_iters=10, eps=1e-9):
    """Project a nonnegative coupling onto uniform marginals via Sinkhorn-Knopp scaling."""
    if n_iters <= 0:
        return p_hat
    if p_hat.dim() == 2:
        return _sinkhorn_project_2d(p_hat, n_iters=n_iters, eps=eps)
    if p_hat.dim() == 3:
        projected = [
            _sinkhorn_project_2d(p_hat[i], n_iters=n_iters, eps=eps)
            for i in range(p_hat.shape[0])
        ]
        return torch.stack(projected, dim=0)
    raise ValueError(f'Unexpected p_hat shape: {tuple(p_hat.shape)}')


def _sinkhorn_project_2d(p_hat, n_iters=10, eps=1e-9):
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


def evaluate_model(model, dataset, device, mode, n_test_samples=100, args=None):
    """Evaluate the model on test data."""
    model.eval()
    eval_sinkhorn_iters = args.eval_sinkhorn_iters if args is not None else 0
    eval_sinkhorn_eps = args.eval_sinkhorn_eps if args is not None else 1e-9
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))

    total_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_kl_norm = 0.0
    total_marg_l1 = 0.0
    total_marg_linf = 0.0
    total_cost_gap = 0.0
    total_abs_cost_gap = 0.0
    total_rel_cost_gap = 0.0
    n_cost = 0
    total_proj_marg_l1 = 0.0
    total_proj_marg_linf = 0.0
    total_proj_cost_gap = 0.0
    total_proj_abs_cost_gap = 0.0
    total_proj_rel_cost_gap = 0.0
    n_proj_cost = 0

    target_val = None
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]

            source_points = torch.tensor(np.array(sample['source_points']), device=device, dtype=torch.float32)
            target_points = torch.tensor(np.array(sample['target_points']), device=device, dtype=torch.float32)
            plan = torch.tensor(np.array(sample['transport_plan']), device=device, dtype=torch.float32)

            xs = source_points.unsqueeze(0)
            us = torch.ones(1, source_points.shape[0], 1, device=device, dtype=torch.float32)
            ys = xs

            if mode == 'plan':
                target = plan.unsqueeze(0)
            elif mode == 'coupling':
                target = (plan / float(plan.shape[0])).unsqueeze(0)
            elif mode == 'barycentric':
                target = (plan @ target_points).unsqueeze(0)
            else:
                raise ValueError(f'Unknown mode: {mode}')

            pred = model(xs, us, ys)
            if mode == 'barycentric':
                loss = F.mse_loss(pred, target)
                total_loss += loss.item()
            else:
                ce, entropy, kl, kl_norm = compute_plan_metrics(pred, target)
                total_loss += ce.item()
                total_entropy += entropy.item()
                total_kl += kl.item()
                total_kl_norm += kl_norm.item()
                w_hat = torch.softmax(pred, dim=-1)
                marg_l1, marg_linf = compute_column_marginal_errors(w_hat)
                total_marg_l1 += marg_l1.item()
                total_marg_linf += marg_linf.item()
                target_val = 1.0 / float(w_hat.shape[-1])
                ot_cost = float(sample.get('ot_cost', float('nan')))
                if not np.isnan(ot_cost) and ot_cost != 0.0:
                    p_hat = w_hat / float(plan.shape[0])
                    cost_matrix = torch.sum((source_points[:, None, :] - target_points[None, :, :]) ** 2, dim=-1)
                    cost_pred = torch.sum(p_hat * cost_matrix).item()
                    cost_gap = cost_pred - ot_cost
                    total_cost_gap += cost_gap
                    total_abs_cost_gap += abs(cost_gap)
                    total_rel_cost_gap += (cost_gap / ot_cost)
                    n_cost += 1
                if eval_sinkhorn_iters > 0:
                    p_hat = w_hat / float(plan.shape[0])
                    p_proj = sinkhorn_project(
                        p_hat,
                        n_iters=eval_sinkhorn_iters,
                        eps=eval_sinkhorn_eps,
                    )
                    w_proj = p_proj * float(plan.shape[0])
                    proj_l1, proj_linf = compute_column_marginal_errors(w_proj)
                    total_proj_marg_l1 += proj_l1.item()
                    total_proj_marg_linf += proj_linf.item()
                    if not np.isnan(ot_cost) and ot_cost != 0.0:
                        cost_pred = torch.sum(p_proj * cost_matrix).item()
                        cost_gap = cost_pred - ot_cost
                        total_proj_cost_gap += cost_gap
                        total_proj_abs_cost_gap += abs(cost_gap)
                        total_proj_rel_cost_gap += (cost_gap / ot_cost)
                        n_proj_cost += 1

    avg_loss = total_loss / n_test
    if mode == 'barycentric':
        print(f'Test Loss ({n_test} samples): {avg_loss:.6e}')
        metrics = {'loss': avg_loss, 'entropy': None, 'kl': None}
    else:
        avg_entropy = total_entropy / n_test
        avg_kl = total_kl / n_test
        avg_kl_norm = total_kl_norm / n_test
        avg_marg_l1 = total_marg_l1 / n_test
        avg_marg_linf = total_marg_linf / n_test
        avg_cost_gap = (total_cost_gap / n_cost) if n_cost > 0 else float('nan')
        avg_abs_cost_gap = (total_abs_cost_gap / n_cost) if n_cost > 0 else float('nan')
        avg_rel_cost_gap = (total_rel_cost_gap / n_cost) if n_cost > 0 else float('nan')
        rel_marg_l1 = avg_marg_l1 / target_val if (target_val and target_val > 0.0) else float('nan')
        rel_marg_linf = avg_marg_linf / target_val if (target_val and target_val > 0.0) else float('nan')
        print(f'Test CE ({n_test} samples): {avg_loss:.6e}')
        print(f'Test Target Entropy: {avg_entropy:.6e}')
        print(f'Test KL (CE - H): {avg_kl:.6e}')
        print(f'Test KL/logN: {avg_kl_norm:.6e}')
        print(f'Test Column Marginal L1: {avg_marg_l1:.6e}')
        print(f'Test Column Marginal Linf: {avg_marg_linf:.6e}')
        print(f'Test Column Marginal L1 (rel): {rel_marg_l1:.6e}')
        print(f'Test Column Marginal Linf (rel): {rel_marg_linf:.6e}')
        if n_cost > 0:
            print(f'Test Cost Gap (pred - true): {avg_cost_gap:.6e}')
            print(f'Test |Cost Gap|: {avg_abs_cost_gap:.6e}')
            print(f'Test Relative Cost Gap: {avg_rel_cost_gap:.6e}')
        metrics = {
            'loss': avg_loss,
            'entropy': avg_entropy,
            'kl': avg_kl,
            'kl_norm': avg_kl_norm,
            'marg_l1': avg_marg_l1,
            'marg_linf': avg_marg_linf,
            'marg_l1_rel': rel_marg_l1,
            'marg_linf_rel': rel_marg_linf,
            'cost_gap': avg_cost_gap,
            'abs_cost_gap': avg_abs_cost_gap,
            'rel_cost_gap': avg_rel_cost_gap,
        }
        if eval_sinkhorn_iters > 0:
            avg_proj_marg_l1 = total_proj_marg_l1 / n_test
            avg_proj_marg_linf = total_proj_marg_linf / n_test
            rel_proj_l1 = avg_proj_marg_l1 / target_val if (target_val and target_val > 0.0) else float('nan')
            rel_proj_linf = avg_proj_marg_linf / target_val if (target_val and target_val > 0.0) else float('nan')
            avg_proj_cost_gap = (total_proj_cost_gap / n_proj_cost) if n_proj_cost > 0 else float('nan')
            avg_proj_abs_cost_gap = (total_proj_abs_cost_gap / n_proj_cost) if n_proj_cost > 0 else float('nan')
            avg_proj_rel_cost_gap = (total_proj_rel_cost_gap / n_proj_cost) if n_proj_cost > 0 else float('nan')
            print(f'Projected Column Marginal L1: {avg_proj_marg_l1:.6e}')
            print(f'Projected Column Marginal Linf: {avg_proj_marg_linf:.6e}')
            print(f'Projected Column Marginal L1 (rel): {rel_proj_l1:.6e}')
            print(f'Projected Column Marginal Linf (rel): {rel_proj_linf:.6e}')
            if n_proj_cost > 0:
                print(f'Projected Cost Gap (pred - true): {avg_proj_cost_gap:.6e}')
                print(f'Projected |Cost Gap|: {avg_proj_abs_cost_gap:.6e}')
                print(f'Projected Relative Cost Gap: {avg_proj_rel_cost_gap:.6e}')
            metrics.update({
                'proj_marg_l1': avg_proj_marg_l1,
                'proj_marg_linf': avg_proj_marg_linf,
                'proj_marg_l1_rel': rel_proj_l1,
                'proj_marg_linf_rel': rel_proj_linf,
                'proj_cost_gap': avg_proj_cost_gap,
                'proj_abs_cost_gap': avg_proj_abs_cost_gap,
                'proj_rel_cost_gap': avg_proj_rel_cost_gap,
            })

    model.train()
    return metrics


def main():
    """Main training function."""
    args = parse_arguments()
    device = torch.device(args.device)
    print(f'Using device: {device}')

    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError('--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.')

    # Set random seed and ensure reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # For better reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_dir = setup_logging(project_root, args.log_dir)
    run_training(args, device, log_dir)


def run_training(args, device, log_dir):
    # Load dataset
    dataset, transport_dataset = load_transport_plan_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        mode=args.mode,
    )
    if dataset is None or transport_dataset is None:
        return

    if args.mode == 'barycentric':
        output_size_tgt = 2
    else:
        output_size_tgt = transport_dataset.n_target_points

    # Create model
    print('Creating VIDON model...')
    model = create_model(args, device, output_size_tgt)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {total_params:,} (trainable: {trainable_params:,})')

    # Load pre-trained model if specified
    model_was_loaded = False
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            print(f'Loading pre-trained model from: {args.load_model_path}')
            model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            model_was_loaded = True
        else:
            print(f'Warning: Model path not found: {args.load_model_path}')

    writer = None
    if args.enable_tensorboard:
        print('Setting up TensorBoard logging...')
        tb_log_dir = os.path.join(log_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_log_dir)
        print(f'TensorBoard logs will be saved to: {tb_log_dir}')
        print(f'To view logs, run: tensorboard --logdir {tb_log_dir}')

    # Train model
    if not model_was_loaded:
        print(f'\nStarting training for {args.vidon_epochs} steps...')

        bar = trange(args.vidon_epochs)
        for _ in bar:
            current_lr = model._update_lr()
            xs, us, ys, targets, _ = transport_dataset.sample(device=device)

            preds = model(xs, us, ys)
            if args.mode == 'barycentric':
                loss = F.mse_loss(preds, targets)
            else:
                ce, entropy, kl, kl_norm = compute_plan_metrics(preds, targets)
                w_hat = torch.softmax(preds, dim=-1)
                marg_l1, _ = compute_column_marginal_errors(w_hat)
                loss = ce + (args.marginal_penalty_weight * marg_l1)

            model.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.opt.step()

            model.total_steps += 1

            if writer is not None:
                if args.mode == 'barycentric':
                    writer.add_scalar('Training/MSE_Loss', loss.item(), model.total_steps)
                else:
                    writer.add_scalar('Training/CE_Loss', loss.item(), model.total_steps)
                    writer.add_scalar('Training/Target_Entropy', entropy.item(), model.total_steps)
                    writer.add_scalar('Training/KL', kl.item(), model.total_steps)
                    writer.add_scalar('Training/KL_logN', kl_norm.item(), model.total_steps)
                    writer.add_scalar('Training/Column_Marginal_L1', marg_l1.item(), model.total_steps)

                if model.total_steps % args.tb_eval_frequency == 0:
                    eval_metrics = evaluate_model(
                        model,
                        dataset,
                        device,
                        mode=args.mode,
                        n_test_samples=args.tb_test_samples,
                        args=args,
                    )
                    if args.mode == 'barycentric':
                        writer.add_scalar('Evaluation/MSE_Loss', eval_metrics['loss'], model.total_steps)
                    else:
                        writer.add_scalar('Evaluation/CE_Loss', eval_metrics['loss'], model.total_steps)
                        writer.add_scalar('Evaluation/Target_Entropy', eval_metrics['entropy'], model.total_steps)
                        writer.add_scalar('Evaluation/KL', eval_metrics['kl'], model.total_steps)
                        writer.add_scalar('Evaluation/KL_logN', eval_metrics['kl_norm'], model.total_steps)
                        writer.add_scalar('Evaluation/Column_Marginal_L1', eval_metrics['marg_l1'], model.total_steps)
                        writer.add_scalar('Evaluation/Column_Marginal_Linf', eval_metrics['marg_linf'], model.total_steps)
                        writer.add_scalar('Evaluation/Column_Marginal_L1_Rel', eval_metrics['marg_l1_rel'], model.total_steps)
                        writer.add_scalar('Evaluation/Column_Marginal_Linf_Rel', eval_metrics['marg_linf_rel'], model.total_steps)
                        if not np.isnan(eval_metrics['cost_gap']):
                            writer.add_scalar('Evaluation/Cost_Gap', eval_metrics['cost_gap'], model.total_steps)
                            writer.add_scalar('Evaluation/Abs_Cost_Gap', eval_metrics['abs_cost_gap'], model.total_steps)
                            writer.add_scalar('Evaluation/Rel_Cost_Gap', eval_metrics['rel_cost_gap'], model.total_steps)
                        if args.eval_sinkhorn_iters > 0:
                            writer.add_scalar('Evaluation/Proj_Column_Marginal_L1', eval_metrics['proj_marg_l1'], model.total_steps)
                            writer.add_scalar('Evaluation/Proj_Column_Marginal_Linf', eval_metrics['proj_marg_linf'], model.total_steps)
                            writer.add_scalar('Evaluation/Proj_Column_Marginal_L1_Rel', eval_metrics['proj_marg_l1_rel'], model.total_steps)
                            writer.add_scalar('Evaluation/Proj_Column_Marginal_Linf_Rel', eval_metrics['proj_marg_linf_rel'], model.total_steps)
                            if not np.isnan(eval_metrics['proj_cost_gap']):
                                writer.add_scalar('Evaluation/Proj_Cost_Gap', eval_metrics['proj_cost_gap'], model.total_steps)
                                writer.add_scalar('Evaluation/Proj_Abs_Cost_Gap', eval_metrics['proj_abs_cost_gap'], model.total_steps)
                                writer.add_scalar('Evaluation/Proj_Rel_Cost_Gap', eval_metrics['proj_rel_cost_gap'], model.total_steps)

            bar.set_description(
                f'Step {model.total_steps} | Loss: {loss.item():.4e} | Grad Norm: {float(norm):.2f} | LR: {current_lr:.2e}'
            )
    else:
        print('\nVIDON transport-plan model loaded. Skipping training.')

    # Evaluate model
    print('\nEvaluating model...')
    eval_metrics = evaluate_model(model, dataset, device, mode=args.mode, n_test_samples=args.tb_test_samples, args=args)

    test_results = {
        'relative_l2_error': None,
        'mse_loss': eval_metrics['loss'] if args.mode == 'barycentric' else None,
        'n_test_samples': args.tb_test_samples
    }
    if args.mode != 'barycentric':
        test_results['transport_plan_metrics'] = {
            'rel_cost_gap': eval_metrics.get('rel_cost_gap'),
            'marg_l1_rel': eval_metrics.get('marg_l1_rel'),
            'marg_linf_rel': eval_metrics.get('marg_linf_rel'),
            'kl_logN': eval_metrics.get('kl_norm'),
        }

    # Plot results
    print('Generating plots...')
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f'transport_plan_barycentric_test_sample_{i+1}.png')
        plot_transport_barycentric_results(
            model,
            dataset,
            device,
            sample_idx=i,
            save_path=plot_save_path,
            dataset_split='test',
            mode=args.mode,
        )

    for i in range(3):
        plot_save_path = os.path.join(log_dir, f'transport_plan_barycentric_train_sample_{i+1}.png')
        plot_transport_barycentric_results(
            model,
            dataset,
            device,
            sample_idx=i,
            save_path=plot_save_path,
            dataset_split='train',
            mode=args.mode,
        )

    for i in range(3):
        plot_save_path = os.path.join(log_dir, f'transport_plan_rowdist_test_sample_{i+1}.png')
        plot_transport_row_distribution_results(
            model,
            dataset,
            device,
            sample_idx=i,
            save_path=plot_save_path,
            dataset_split='test',
            mode=args.mode,
        )

    for i in range(3):
        plot_save_path = os.path.join(log_dir, f'transport_plan_rowdist_train_sample_{i+1}.png')
        plot_transport_row_distribution_results(
            model,
            dataset,
            device,
            sample_idx=i,
            save_path=plot_save_path,
            dataset_split='train',
            mode=args.mode,
        )

    # Save experiment configuration with test results
    save_experiment_configuration(
        args,
        model,
        dataset,
        transport_dataset,
        device,
        log_dir,
        dataset_type='transport_plan',
        test_results=test_results,
    )

    # Save model
    if not model_was_loaded:
        model_save_path = os.path.join(log_dir, 'transport_plan_vidon_model.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to: {model_save_path}')

    print('Training completed!')

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
