#!/usr/bin/env python
"""
Test suite for AdaptiveQuadratureHead implementation.

Tests:
1. Permutation invariance (head + SetONet integration)
2. Mask-all-false produces finite output (no NaNs)
"""

import os
import sys

import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from Models.SetONet import SetONet
from Models.utils.adaptive_quadrature_head import AdaptiveQuadratureHead


def _build_head(device: torch.device) -> AdaptiveQuadratureHead:
    return AdaptiveQuadratureHead(
        p=16,
        dx_enc=2,
        du=1,
        dout=1,
        dk=32,
        dv=1,  # dv == dout -> identity rho_token for cleaner invariance testing
        hidden=64,
        activation_fn=nn.ReLU,
        adapt_rank=4,
        adapt_hidden=64,
        adapt_scale=0.1,
        use_value_context=True,
        normalize="total",
        learn_temperature=False,
    ).to(device)


def _build_model(device: torch.device) -> SetONet:
    return SetONet(
        input_size_src=2,
        output_size_src=1,
        input_size_tgt=2,
        output_size_tgt=1,
        p=16,
        phi_hidden_size=64,
        rho_hidden_size=64,
        trunk_hidden_size=64,
        n_trunk_layers=2,
        activation_fn=nn.ReLU,
        use_deeponet_bias=False,
        phi_output_size=32,
        pos_encoding_type="skip",
        use_positional_encoding=False,
        aggregation_type="mean",
        branch_head_type="adaptive_quadrature",
        quad_dk=32,
        quad_dv=1,
        quad_normalize="total",
        quad_learn_temperature=False,
        adapt_quad_rank=4,
        adapt_quad_hidden=64,
        adapt_quad_scale=0.1,
        adapt_quad_use_value_context=True,
    ).to(device)


def test_adaptive_quadrature_head_permutation_invariance():
    device = torch.device("cpu")
    torch.manual_seed(0)

    head = _build_head(device)
    head.eval()

    B, N, dx, du = 2, 40, 2, 1
    x_enc = torch.randn(B, N, dx, device=device)
    u = torch.randn(B, N, du, device=device)
    w = torch.rand(B, N, device=device) + 0.1

    sensor_mask = (torch.rand(B, N, device=device) > 0.2)
    for b in range(B):
        if not bool(sensor_mask[b].any()):
            sensor_mask[b, 0] = True

    perm = torch.randperm(N, device=device)

    with torch.no_grad():
        out1 = head(x_enc, u, sensor_mask=sensor_mask, sensor_weights=w)
        out2 = head(
            x_enc[:, perm, :],
            u[:, perm, :],
            sensor_mask=sensor_mask[:, perm],
            sensor_weights=w[:, perm],
        )

    max_diff = torch.max(torch.abs(out1 - out2)).item()
    assert max_diff < 1e-5, f"Permutation invariance failed: {max_diff=}"


def test_adaptive_quadrature_setonet_permutation_invariance():
    device = torch.device("cpu")
    torch.manual_seed(1)

    model = _build_model(device)
    model.eval()

    B, N, dx, du = 2, 32, 2, 1
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    w = torch.rand(B, N, device=device) + 0.1
    sensor_mask = (torch.rand(B, N, device=device) > 0.1)
    for b in range(B):
        if not bool(sensor_mask[b].any()):
            sensor_mask[b, 0] = True

    perm = torch.randperm(N, device=device)

    with torch.no_grad():
        out1 = model.forward_branch(xs, us, sensor_mask=sensor_mask, sensor_weights=w)
        out2 = model.forward_branch(
            xs[:, perm, :],
            us[:, perm, :],
            sensor_mask=sensor_mask[:, perm],
            sensor_weights=w[:, perm],
        )

    max_diff = torch.max(torch.abs(out1 - out2)).item()
    assert max_diff < 1e-5, f"SetONet permutation invariance failed: {max_diff=}"


def test_adaptive_quadrature_mask_all_false_no_nans():
    device = torch.device("cpu")
    torch.manual_seed(2)

    model = _build_model(device)
    model.eval()

    B, N, dx, du = 2, 16, 2, 1
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    w = torch.rand(B, N, device=device) + 0.1

    sensor_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model.forward_branch(xs, us, sensor_mask=sensor_mask, sensor_weights=w)

    assert torch.isfinite(out).all(), "Output contains NaNs/Infs for all-false sensor_mask"

