import argparse
import os
import sys

import numpy as np
import torch

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.SetONet import SetONet
from Models.utils.helper_utils import calculate_l2_relative_error
from Data.data_utils import generate_batch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Minimal basis-transfer experiment (shared branch+trunk).")
    parser.add_argument("--pretrain_task", type=str, default="integral", choices=["integral", "derivative", "identity"])
    parser.add_argument("--transfer_task", type=str, default="derivative", choices=["integral", "derivative", "identity"])
    parser.add_argument("--pretrain_samples", type=int, default=2000)
    parser.add_argument("--transfer_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=1000)
    parser.add_argument("--pretrain_epochs", type=int, default=5000)
    parser.add_argument("--transfer_epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sensor_size", type=int, default=100)
    parser.add_argument("--n_trunk_points", type=int, default=200)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip random-branch and scratch baselines")
    return parser.parse_args()


class FixedSyntheticDataset:
    def __init__(self, benchmark, n_samples, sensor_x, n_trunk_points, scale, input_range, device, seed):
        torch.manual_seed(seed)
        batch = generate_batch(
            batch_size=n_samples,
            n_trunk_points=n_trunk_points,
            sensor_x=sensor_x,
            scale=scale,
            input_range=input_range,
            device=device,
            constant_zero=True,
            variable_sensors=False,
        )
        f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval = batch

        if benchmark == "derivative":
            us = f_at_sensors
            target = f_prime_at_trunk.T
        elif benchmark == "integral":
            us = f_prime_at_sensors
            target = f_at_trunk.T
        elif benchmark == "identity":
            us = f_at_sensors
            target = f_at_trunk.T
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        self.us = us
        self.target = target
        self.sensor_x = sensor_x
        self.trunk_x = x_eval
        self.n_samples = n_samples
        self.device = device

    def sample(self, batch_size):
        idx = torch.randint(0, self.n_samples, (batch_size,), device=self.device)
        us = self.us[idx].unsqueeze(-1)
        target = self.target[idx].unsqueeze(-1)
        xs = self.sensor_x.unsqueeze(0).expand(batch_size, -1, -1)
        ys = self.trunk_x.unsqueeze(0).expand(batch_size, -1, -1)
        return xs, us, ys, target

    def evaluate(self, forward_fn, batch_size):
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, self.n_samples, batch_size):
                end = min(start + batch_size, self.n_samples)
                xs = self.sensor_x.unsqueeze(0).expand(end - start, -1, -1)
                ys = self.trunk_x.unsqueeze(0).expand(end - start, -1, -1)
                us = self.us[start:end].unsqueeze(-1)
                target = self.target[start:end].unsqueeze(-1)
                pred = forward_fn(xs, us, ys)
                rel = calculate_l2_relative_error(pred.squeeze(-1), target.squeeze(-1)).item()
                total += rel * (end - start)
                count += (end - start)
        return total / max(1, count)


def build_backbone(device):
    model = SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=32,
        phi_hidden_size=256,
        rho_hidden_size=256,
        trunk_hidden_size=256,
        n_trunk_layers=4,
        activation_fn=torch.nn.ReLU,
        use_deeponet_bias=False,
        phi_output_size=32,
        initial_lr=5e-4,
        lr_schedule_steps=None,
        lr_schedule_gammas=None,
        pos_encoding_type="sinusoidal",
        pos_encoding_dim=32,
        pos_encoding_max_freq=0.1,
        aggregation_type="mean",
        use_positional_encoding=True,
        attention_n_tokens=1,
        branch_head_type="quadrature",
        quad_dk=64,
        quad_dv=32,
        quad_key_hidden=256,
        quad_key_layers=3,
        quad_phi_activation="softplus",
        quad_value_mode="linear_u",
        quad_normalize="total",
        quad_learn_temperature=False,
    ).to(device)
    return model


def init_mixing(p, device):
    return torch.nn.Parameter(torch.eye(p, device=device))


def forward_with_mixing(backbone, M, xs, us, ys):
    b = backbone.forward_branch(xs, us, ys=ys)  # (B, p, 1)
    t = backbone.forward_trunk(ys)  # (B, n, p, 1)
    b = b.squeeze(-1)  # (B, p)
    t = t.squeeze(-1)  # (B, n, p)
    bM = torch.einsum("bp,pq->bq", b, M)
    out = torch.einsum("bq,bnq->bn", bM, t)
    return out.unsqueeze(-1)


def train(backbone, M, dataset, epochs, batch_size, train_backbone):
    params = [M]
    for p in backbone.parameters():
        p.requires_grad = train_backbone
        if train_backbone:
            params.append(p)
    opt = torch.optim.Adam(params, lr=5e-4)
    for _ in range(epochs):
        xs, us, ys, target = dataset.sample(batch_size)
        pred = forward_with_mixing(backbone, M, xs, us, ys)
        loss = torch.nn.MSELoss()(pred, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


def copy_backbone(src, dst):
    dst.load_state_dict(src.state_dict())


def main():
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    input_range = [-1, 1]
    sensor_x = torch.linspace(input_range[0], input_range[1], args.sensor_size, device=device).view(-1, 1)

    pretrain_dataset = FixedSyntheticDataset(
        args.pretrain_task,
        args.pretrain_samples,
        sensor_x,
        args.n_trunk_points,
        args.scale,
        input_range,
        device,
        seed=args.seed + 1,
    )
    transfer_dataset = FixedSyntheticDataset(
        args.transfer_task,
        args.transfer_samples,
        sensor_x,
        args.n_trunk_points,
        args.scale,
        input_range,
        device,
        seed=args.seed + 2,
    )
    test_dataset = FixedSyntheticDataset(
        args.transfer_task,
        args.test_samples,
        sensor_x,
        args.n_trunk_points,
        args.scale,
        input_range,
        device,
        seed=args.seed + 3,
    )

    print(f"Pretrain task: {args.pretrain_task} | Transfer task: {args.transfer_task}")
    print(f"Pretrain samples: {args.pretrain_samples} | Transfer samples: {args.transfer_samples}")

    # Pretrain backbone + mixing
    backbone = build_backbone(device)
    M_pre = init_mixing(backbone.p, device)
    print("\n[1] Pretraining shared branch+trunk...")
    train(backbone, M_pre, pretrain_dataset, args.pretrain_epochs, args.batch_size, train_backbone=True)

    # Transfer: freeze backbone, train only mixing
    transfer_backbone = build_backbone(device)
    copy_backbone(backbone, transfer_backbone)
    M_transfer = init_mixing(transfer_backbone.p, device)
    print("[2] Transfer: frozen branch+trunk, train mixing only...")
    train(transfer_backbone, M_transfer, transfer_dataset, args.transfer_epochs, args.batch_size, train_backbone=False)
    transfer_rel = test_dataset.evaluate(
        lambda xs, us, ys: forward_with_mixing(transfer_backbone, M_transfer, xs, us, ys),
        args.batch_size,
    )
    print(f"Transfer (frozen backbone) Rel L2: {transfer_rel:.6f}")

    if not args.skip_baselines:
        # Baseline A: random frozen backbone, train mixing only
        rand_backbone = build_backbone(device)
        M_rand = init_mixing(rand_backbone.p, device)
        print("[3] Baseline: random frozen backbone, train mixing only...")
        train(rand_backbone, M_rand, transfer_dataset, args.transfer_epochs, args.batch_size, train_backbone=False)
        rand_rel = test_dataset.evaluate(
            lambda xs, us, ys: forward_with_mixing(rand_backbone, M_rand, xs, us, ys),
            args.batch_size,
        )
        print(f"Random-backbone (frozen) Rel L2: {rand_rel:.6f}")

        # Baseline B: scratch (train backbone + mixing on transfer task)
        scratch_backbone = build_backbone(device)
        M_scratch = init_mixing(scratch_backbone.p, device)
        print("[4] Baseline: scratch (train backbone + mixing on transfer task)...")
        train(scratch_backbone, M_scratch, transfer_dataset, args.transfer_epochs, args.batch_size, train_backbone=True)
        scratch_rel = test_dataset.evaluate(
            lambda xs, us, ys: forward_with_mixing(scratch_backbone, M_scratch, xs, us, ys),
            args.batch_size,
        )
        print(f"Scratch (full) Rel L2: {scratch_rel:.6f}")


if __name__ == "__main__":
    main()
