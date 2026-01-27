#!/usr/bin/env python
"""paper_plot_config.py
----------------------------------
Configuration for paper-quality figure generation.
Defines styling, model ordering, and benchmark configurations.
"""
from __future__ import annotations

# Paper-quality matplotlib style settings
# Width: 6.5 inches (double-column), 12pt labels
PAPER_STYLE = {
    'figure.figsize': (6.5, 2.5),  # Default per-row height, adjusted per benchmark
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
}

# Model order: SetONet first, DeepONet (if exists), VIDON last
MODEL_ORDER = ['setonet_quadrature', 'deeponet', 'vidon']

# Model display names (for reference, not used in plots per user request)
MODEL_DISPLAY_NAMES = {
    'setonet_quadrature': 'SetONet',
    'deeponet': 'DeepONet',
    'vidon': 'VIDON',
}

# Benchmark configurations
# n_cols: number of subplots per row
# row_height: height of each row in inches
# data_path_key: key in experiment_config.json for data path
BENCHMARKS = {
    'heat_2d_P10': {
        'n_cols': 3,
        'row_height': 2.2,
        'plot_type': 'heat',
        'data_path': 'Data/heat_data/pcb_heat_adaptive_dataset9.0_n8192_N25_P10',
        'available_models': ['setonet_quadrature', 'vidon'],
    },
    'heat_2d_P30': {
        'n_cols': 3,
        'row_height': 2.2,
        'plot_type': 'heat',
        'data_path': 'Data/heat_data/pcb_heat_adaptive_dataset8.0_n8192_N25_P30',
        'available_models': ['setonet_quadrature', 'vidon'],
    },
    'elastic_2d': {
        'n_cols': 4,
        'row_height': 1.8,
        'plot_type': 'elastic',
        'data_path': 'Data/elastic_2d_data/elastic_dataset',
        'available_models': ['setonet_quadrature', 'deeponet', 'vidon'],
    },
    'concentration_2d': {
        'n_cols': 3,
        'row_height': 2.2,
        'plot_type': 'concentration',
        'data_path': 'Data/concentration_data/chem_plume_adaptive_dataset4.0_n4096_N25_P30',
        'available_models': ['setonet_quadrature', 'vidon'],
    },
    'diffraction_2d': {
        'n_cols': 4,
        'row_height': 2.2,
        'plot_type': 'diffraction',
        'data_path': 'Data/diffraction_data/phase_screen_dataset',
        'available_models': ['setonet_quadrature', 'vidon'],
    },
    'transport': {
        'n_cols': 4,
        'row_height': 2.0,
        'plot_type': 'transport_q',
        'data_path': 'Data/transport_q_data/transport_dataset',
        'available_models': ['setonet_quadrature', 'vidon'],
    },
}

# Model checkpoint file patterns per benchmark/model
CHECKPOINT_PATTERNS = {
    'heat_2d_P10': {
        'setonet_quadrature': 'heat2d_setonet_model.pth',
        'vidon': 'heat2d_vidon_model.pth',
    },
    'heat_2d_P30': {
        'setonet_quadrature': 'heat2d_setonet_model.pth',
        'vidon': 'heat2d_vidon_model.pth',
    },
    'elastic_2d': {
        'setonet_quadrature': 'elastic2d_setonet_model.pth',
        'deeponet': 'elastic2d_deeponet_model.pth',
        'vidon': 'elastic2d_vidon_model.pth',
    },
    'concentration_2d': {
        'setonet_quadrature': 'concentration2d_setonet_model.pth',
        'vidon': 'concentration2d_vidon_model.pth',
    },
    'diffraction_2d': {
        'setonet_quadrature': 'diffraction2d_setonet_model.pth',
        'vidon': 'diffraction2d_vidon_model.pth',
    },
    'transport': {
        'setonet_quadrature': 'transport_setonet_model.pth',
        'vidon': 'transport_vidon_model.pth',
    },
}

# Figure width fixed at 6.5 inches
FIGURE_WIDTH = 6.5

# Output formats
OUTPUT_FORMATS = ['pdf', 'png']

# DPI for raster output
PNG_DPI = 300

# Default seed to use
DEFAULT_SEED = 0

# Number of test samples to generate
DEFAULT_N_SAMPLES = 10
