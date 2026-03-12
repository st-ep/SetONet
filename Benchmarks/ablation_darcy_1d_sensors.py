#!/usr/bin/env python
"""
Darcy 1D sensor count ablation: MSE on full test set vs number of sensors.
Loads 5 pretrained models (SetONet Quadrature/Attention/Mean/Sum, VIDON),
evaluates at equally-spaced sensor counts, and plots results (MSE in log scale).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Project root
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Data.darcy_1d_data.darcy_1d_dataset import load_darcy_dataset
from Models.SetONet import SetONet
from Models.VIDON import VIDON

# Fixed seed
SEED = 0
SENSOR_COUNTS = [5, 15, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
N_GRID = 501
N_QUERY = 300
MODEL_BASE = "logs_all/darcy_1d"


def _get_activation(name):
    from torch import nn
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "swish": nn.SiLU}.get(
        (name or "relu").lower(), nn.ReLU
    )


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def build_setonet_quadrature(device):
    cfg_path = os.path.join(project_root, MODEL_BASE, "setonet_quadrature", "seed_0", "experiment_config.json")
    cfg = load_config(cfg_path)["model_architecture"]
    act = _get_activation(cfg.get("activation_fn"))
    model = SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=cfg["son_p_dim"],
        phi_hidden_size=cfg["son_phi_hidden"],
        rho_hidden_size=cfg["son_rho_hidden"],
        trunk_hidden_size=cfg["son_trunk_hidden"],
        n_trunk_layers=cfg["son_n_trunk_layers"],
        activation_fn=act,
        use_deeponet_bias=True,
        phi_output_size=cfg["son_phi_output_size"],
        initial_lr=5e-4,
        use_positional_encoding=cfg["use_positional_encoding"],
        pos_encoding_type=cfg["pos_encoding_type"],
        pos_encoding_dim=cfg["pos_encoding_dim"],
        pos_encoding_max_freq=cfg["pos_encoding_max_freq"],
        aggregation_type="attention",
        attention_n_tokens=cfg.get("attention_n_tokens", 1),
        branch_head_type="quadrature",
        quad_dk=64,
        quad_phi_activation="softplus",
        quad_value_mode="linear_u",
    ).to(device)
    ckpt = os.path.join(project_root, MODEL_BASE, "setonet_quadrature", "seed_0", "darcy1d_setonet_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def build_setonet_quadrature_uniform(device):
    cfg_path = os.path.join(project_root, MODEL_BASE, "setonet_quadrature_uniform", "seed_0", "experiment_config.json")
    cfg = load_config(cfg_path)["model_architecture"]
    act = _get_activation(cfg.get("activation_fn"))
    model = SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=cfg["son_p_dim"],
        phi_hidden_size=cfg["son_phi_hidden"],
        rho_hidden_size=cfg["son_rho_hidden"],
        trunk_hidden_size=cfg["son_trunk_hidden"],
        n_trunk_layers=cfg["son_n_trunk_layers"],
        activation_fn=act,
        use_deeponet_bias=True,
        phi_output_size=cfg["son_phi_output_size"],
        initial_lr=5e-4,
        use_positional_encoding=cfg["use_positional_encoding"],
        pos_encoding_type=cfg["pos_encoding_type"],
        pos_encoding_dim=cfg["pos_encoding_dim"],
        pos_encoding_max_freq=cfg["pos_encoding_max_freq"],
        aggregation_type="attention",
        attention_n_tokens=cfg.get("attention_n_tokens", 1),
        branch_head_type="quadrature",
        quad_dk=64,
        quad_phi_activation="softplus",
        quad_value_mode="linear_u",
        quad_uniform_weights=True,
    ).to(device)
    ckpt = os.path.join(project_root, MODEL_BASE, "setonet_quadrature_uniform", "seed_0", "darcy1d_setonet_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def build_setonet_standard(aggregation, device):
    subdir = {"attention": "setonet_attention", "mean": "setonet_mean", "sum": "setonet_sum"}[aggregation]
    cfg_path = os.path.join(project_root, MODEL_BASE, subdir, "seed_0", "experiment_config.json")
    cfg = load_config(cfg_path)["model_architecture"]
    act = _get_activation(cfg.get("activation_fn"))
    model = SetONet(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=cfg["son_p_dim"],
        phi_hidden_size=cfg["son_phi_hidden"],
        rho_hidden_size=cfg["son_rho_hidden"],
        trunk_hidden_size=cfg["son_trunk_hidden"],
        n_trunk_layers=cfg["son_n_trunk_layers"],
        activation_fn=act,
        use_deeponet_bias=True,
        phi_output_size=cfg["son_phi_output_size"],
        initial_lr=5e-4,
        use_positional_encoding=cfg["use_positional_encoding"],
        pos_encoding_type=cfg["pos_encoding_type"],
        pos_encoding_dim=cfg["pos_encoding_dim"],
        pos_encoding_max_freq=cfg["pos_encoding_max_freq"],
        aggregation_type=aggregation,
        attention_n_tokens=cfg.get("attention_n_tokens", 1),
        branch_head_type="standard",
    ).to(device)
    ckpt = os.path.join(project_root, MODEL_BASE, subdir, "seed_0", "darcy1d_setonet_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def build_vidon(device):
    cfg_path = os.path.join(project_root, MODEL_BASE, "vidon", "seed_0", "experiment_config.json")
    cfg = load_config(cfg_path)["model_architecture"]
    act = _get_activation(cfg.get("activation_fn"))
    model = VIDON(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=cfg["vidon_p_dim"],
        n_heads=cfg["vidon_n_heads"],
        d_enc=cfg["vidon_d_enc"],
        head_output_size=cfg["vidon_head_output_size"],
        enc_hidden_size=cfg["vidon_enc_hidden"],
        enc_n_layers=cfg["vidon_enc_n_layers"],
        head_hidden_size=cfg["vidon_head_hidden"],
        head_n_layers=cfg["vidon_head_n_layers"],
        combine_hidden_size=cfg["vidon_combine_hidden"],
        combine_n_layers=cfg["vidon_combine_n_layers"],
        trunk_hidden_size=cfg["vidon_trunk_hidden"],
        n_trunk_layers=cfg["vidon_n_trunk_layers"],
        activation_fn=act,
        initial_lr=5e-4,
    ).to(device)
    ckpt = os.path.join(project_root, MODEL_BASE, "vidon", "seed_0", "darcy1d_vidon_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def run_ablation(data_path, device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = load_darcy_dataset(data_path)
    test_data = dataset["test"]
    n_test = len(test_data)
    grid = torch.tensor(dataset["train"][0]["X"], dtype=torch.float32, device=device)
    grid_points = grid.view(-1, 1)  # (501, 1)
    query_indices = torch.linspace(0, N_GRID - 1, N_QUERY, dtype=torch.long, device=device)
    query_x = grid_points[query_indices]  # (300, 1)

    models = {
        "SetONet-Key": build_setonet_quadrature(device),
        "SetONet-Key (Uniform)": build_setonet_quadrature_uniform(device),
        "SetONet (Attention)": build_setonet_standard("attention", device),
        "SetONet (Mean)": build_setonet_standard("mean", device),
        "SetONet (Sum)": build_setonet_standard("sum", device),
        "VIDON": build_vidon(device),
    }
    for m in models.values():
        m.eval()

    results = {name: {} for name in models}

    for n_sensors in SENSOR_COUNTS:
        print(f"Sensor count: {n_sensors}")
        sensor_indices = torch.linspace(0, N_GRID - 1, n_sensors, dtype=torch.long, device=device)
        sensor_x_n = grid_points[sensor_indices]  # (n_sensors, 1)

        for model_name, model in models.items():
            total_mse = 0.0
            with torch.no_grad():
                for i in range(n_test):
                    sample = test_data[i]
                    u_full = torch.tensor(sample["u"], dtype=torch.float32, device=device)
                    s_full = torch.tensor(sample["s"], dtype=torch.float32, device=device)
                    target = s_full[query_indices].view(1, -1, 1)  # (1, 300, 1)

                    u_at_n = u_full[sensor_indices].view(1, -1, 1)  # (1, n_sensors, 1)
                    xs = sensor_x_n.unsqueeze(0)  # (1, n_sensors, 1)
                    us = u_at_n
                    ys = query_x.unsqueeze(0)  # (1, 300, 1)

                    pred = model(xs, us, ys)  # (1, 300, 1)
                    mse = torch.nn.functional.mse_loss(pred, target).item()
                    total_mse += mse

            results[model_name][n_sensors] = total_mse / n_test
            print(f"  {model_name} @ {n_sensors} sensors: MSE = {results[model_name][n_sensors]:.6e}")

    return results


def plot_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    model_colors = {
        "SetONet-Key": "#1b9e77",
        "SetONet-Key (Uniform)": "#66c2a5",
        "SetONet (Attention)": "#d95f02",
        "SetONet (Mean)": "#7570b3",
        "SetONet (Sum)": "#1f78b4",
        "VIDON": "#e7298a",
    }
    markers = ["o", "x", "s", "^", "D", "v"]
    for idx, (name, mse_by_n) in enumerate(results.items()):
        x = list(mse_by_n.keys())
        y = [mse_by_n[n] for n in x]
        ax.plot(
            x,
            y,
            marker=markers[idx % len(markers)],
            color=model_colors.get(name, "#333333"),
            label=name,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Number of sensors ($M$)")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.set_ylim(top=1e-4)
    ax.legend(frameon=False, loc="upper center")
    out_png = os.path.join(output_dir, "darcy_1d_sensor_ablation.png")
    out_pdf = os.path.join(output_dir, "darcy_1d_sensor_ablation.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {out_png}")
    print(f"Plot saved to {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Darcy 1D sensor count ablation: MSE vs sensors.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for evaluation.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/darcy_1d_data/darcy_1d_dataset_501",
        help="Path to Darcy 1D dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/ablation_darcy_1d_sensors",
        help="Directory to save the plot.",
    )
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(project_root, args.data_path)

    print("Running Darcy 1D sensor ablation (seed 0, equally spaced sensors).")
    results = run_ablation(data_path, device)
    plot_results(results, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
