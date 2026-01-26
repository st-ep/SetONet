#!/usr/bin/env python
"""paper_plot_config_1d.py - Config for 1D paper figures."""
from __future__ import annotations

from Data.burgers_1d_data.burgers_1d_dataset import (
    create_sensor_points as burgers_create_sensor_points,
    create_query_points as burgers_create_query_points,
)
from Data.darcy_1d_data.darcy_1d_dataset import (
    create_sensor_points as darcy_create_sensor_points,
    create_query_points as darcy_create_query_points,
)

DEFAULT_N_SAMPLES = 10
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = "paper_figures_1d"
DEFAULT_LOGS_DIR = "logs_all"

MODEL_STYLES = {
    "setonet_quad": {"color": "#1b9e77", "label": "SetONet (Quadrature)"},
    "setonet_attn": {"color": "#d95f02", "label": "SetONet (Attention)"},
    "deeponet": {"color": "#7570b3", "label": "DeepONet"},
    "vidon": {"color": "#e7298a", "label": "VIDON"},
}

ROW_CONFIGS_BY_BENCHMARK = {
    "burgers_1d": [
        {
            "title": "Fixed",
            "log_dir": "burgers_1d",
            "pred_keys": ["setonet_quad", "deeponet", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "deeponet": "deeponet",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "DeepONet",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.0,
            "replace_with_nearest": False,
        },
        {
            "title": "Variable",
            "log_dir": "burgers_1d_robust_train",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.2,
            "replace_with_nearest": True,
        },
        {
            "title": "Drop-off",
            "log_dir": "burgers_1d",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.2,
            "replace_with_nearest": True,
        },
    ],
    "darcy_1d": [
        {
            "title": "Fixed",
            "log_dir": "darcy_1d",
            "pred_keys": ["setonet_quad", "deeponet", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "deeponet": "deeponet",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "DeepONet",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.0,
            "replace_with_nearest": False,
        },
        {
            "title": "Variable",
            "log_dir": "darcy_1d_robust_train",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.2,
            "replace_with_nearest": True,
        },
        {
            "title": "Drop-off",
            "log_dir": "darcy_1d",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.2,
            "replace_with_nearest": True,
        },
    ],
    "1d_derivative": [
        {
            "title": "Fixed",
            "log_dir": "1d_derivative",
            "pred_keys": ["setonet_quad", "deeponet", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "deeponet": "deeponet",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "DeepONet",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.0,
            "replace_with_nearest": False,
            "variable_sensors": False,
        },
        {
            "title": "Variable",
            "log_dir": "1d_derivative_varsens",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.0,
            "replace_with_nearest": False,
            "variable_sensors": True,
        },
        {
            "title": "Drop-off",
            "log_dir": "1d_derivative_varsens",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.2,
            "replace_with_nearest": True,
            "variable_sensors": True,
        },
    ],
    "1d_integral": [
        {
            "title": "Fixed",
            "log_dir": "1d_integral",
            "pred_keys": ["setonet_quad", "deeponet", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "deeponet": "deeponet",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "DeepONet",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.0,
            "replace_with_nearest": False,
            "variable_sensors": False,
        },
        {
            "title": "Variable",
            "log_dir": "1d_integral_varsens",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.0,
            "replace_with_nearest": False,
            "variable_sensors": True,
        },
        {
            "title": "Drop-off",
            "log_dir": "1d_integral_varsens",
            "pred_keys": ["setonet_quad", "setonet_attn", "vidon"],
            "model_dirs": {
                "setonet_quad": "setonet_quadrature",
                "setonet_attn": "setonet_attention",
                "vidon": "vidon",
            },
            "col_labels": [
                "Input",
                "SetONet (Quadrature)",
                "SetONet (Attention)",
                "VIDON",
                "Residuals",
            ],
            "eval_dropoff": 0.2,
            "replace_with_nearest": True,
            "variable_sensors": True,
        },
    ],
}

BENCHMARK_CONFIGS = {
    "burgers_1d": {
        "data_kind": "dataset",
        "ckpt_names": {
            "setonet": "burgers1d_setonet_model.pth",
            "deeponet": "burgers1d_deeponet_model.pth",
            "vidon": "burgers1d_vidon_model.pth",
        },
        "input_label": "u(x,0)",
        "output_label": "u(x,T)",
        "default_sensor_size": 300,
        "default_query_points": 128,
        "data_path": None,
        "sensor_points_fn": burgers_create_sensor_points,
        "query_points_fn": burgers_create_query_points,
    },
    "darcy_1d": {
        "data_kind": "dataset",
        "ckpt_names": {
            "setonet": "darcy1d_setonet_model.pth",
            "deeponet": "darcy1d_deeponet_model.pth",
            "vidon": "darcy1d_vidon_model.pth",
        },
        "input_label": "f(x)",
        "output_label": "u(x)",
        "default_sensor_size": 300,
        "default_query_points": 300,
        "data_path": "Data/darcy_1d_data/darcy_1d_dataset_501",
        "sensor_points_fn": darcy_create_sensor_points,
        "query_points_fn": darcy_create_query_points,
    },
    "1d_derivative": {
        "data_kind": "synthetic",
        "ckpt_names": {
            "setonet": "setonet_derivative_model.pth",
            "deeponet": "deeponet_derivative_model.pth",
            "vidon": "vidon_derivative_model.pth",
        },
        "input_label": "f(x)",
        "output_label": "f'(x)",
        "default_sensor_size": 100,
        "default_query_points": 200,
        "input_range": (-1.0, 1.0),
        "scale": 0.1,
    },
    "1d_integral": {
        "data_kind": "synthetic",
        "ckpt_names": {
            "setonet": "setonet_integral_model.pth",
            "deeponet": "deeponet_integral_model.pth",
            "vidon": "vidon_integral_model.pth",
        },
        "input_label": "f'(x)",
        "output_label": "f(x)",
        "default_sensor_size": 100,
        "default_query_points": 200,
        "input_range": (-1.0, 1.0),
        "scale": 0.1,
    },
}
