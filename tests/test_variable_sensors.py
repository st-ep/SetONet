#!/usr/bin/env python
"""
Test to diagnose why Galerkin PoU fails on variable sensor configurations.

This script compares attention pattern entropy between PG and Galerkin PoU
with fixed vs variable sensor locations.
"""

import torch
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from Models.SetONet import SetONet
import torch.nn as nn


def compute_attention_entropy(model, xs, us):
    """
    Extract and analyze attention patterns from the head.
    Returns entropy of attention distribution (higher = more diffuse).
    """
    model.eval()

    with torch.no_grad():
        # Access the head directly
        if hasattr(model, 'galerkin_head') and model.galerkin_head is not None:
            head = model.galerkin_head
            x_enc = model._sinusoidal_encoding(xs) if model.use_positional_encoding else xs

            # Compute attention scores (same as in forward)
            K = head.key_net(x_enc)
            Q = head.query_tokens.expand(xs.shape[0], -1, -1)
            scores = torch.einsum("bpk,bnk->bpn", Q, K) / torch.sqrt(torch.tensor(head.dk))

            # Galerkin PoU: softmax over tokens (dim=1)
            Phi = torch.softmax(scores, dim=1)  # (B, p, N)

        elif hasattr(model, 'pg_head') and model.pg_head is not None:
            head = model.pg_head
            x_enc = model._sinusoidal_encoding(xs) if model.use_positional_encoding else xs

            # Compute attention scores
            K = head.key_net(x_enc)
            Q = head.query_tokens.expand(xs.shape[0], -1, -1)
            scores = torch.einsum("bpk,bnk->bpn", Q, K) / torch.sqrt(torch.tensor(head.dk))

            # PG: softmax over sensors (dim=2)
            Phi = torch.softmax(scores, dim=2)  # (B, p, N)
        else:
            raise ValueError("Model doesn't have pg_head or galerkin_head")

    # Compute entropy of attention distribution
    # Higher entropy = more diffuse (uniform), lower entropy = sharper (focused)
    epsilon = 1e-10
    entropy = -torch.sum(Phi * torch.log(Phi + epsilon), dim=[1, 2])  # Sum over tokens and sensors

    return Phi, entropy.mean().item()


def test_fixed_vs_variable_sensors():
    """
    Compare attention patterns with fixed vs variable sensor locations.
    """
    print("=" * 80)
    print("TESTING: Fixed vs Variable Sensor Configurations")
    print("=" * 80)

    device = torch.device("cpu")
    torch.manual_seed(42)

    # Model configuration (1D problem)
    config = {
        'input_size_src': 1,
        'output_size_src': 1,
        'input_size_tgt': 1,
        'output_size_tgt': 1,
        'p': 32,
        'phi_hidden_size': 256,
        'rho_hidden_size': 256,
        'trunk_hidden_size': 256,
        'n_trunk_layers': 4,
        'phi_output_size': 32,
        'activation_fn': nn.ReLU,
        'pos_encoding_type': 'sinusoidal',
        'pos_encoding_dim': 64,
    }

    B, N = 8, 100  # batch size, n_sensors

    # Test 1: Galerkin PoU with fixed sensors
    print("\n1. GALERKIN PoU - Fixed Sensor Locations")
    print("-" * 80)

    model_gal = SetONet(**config, branch_head_type="galerkin_pou",
                        galerkin_normalize="total").to(device)

    # Fixed sensors (same for all batches)
    xs_fixed = torch.linspace(0, 1, N).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
    us_fixed = torch.randn(B, N, 1, device=device)

    Phi_gal_fixed, entropy_gal_fixed = compute_attention_entropy(model_gal, xs_fixed, us_fixed)
    print(f"Attention entropy (fixed): {entropy_gal_fixed:.4f}")
    print(f"Attention pattern sparsity: {(Phi_gal_fixed < 0.01).float().mean().item():.2%} of weights < 0.01")

    # Test 2: Galerkin PoU with variable sensors
    print("\n2. GALERKIN PoU - Variable Sensor Locations")
    print("-" * 80)

    # Variable sensors (different for each batch)
    xs_variable = torch.rand(B, N, 1, device=device)
    us_variable = torch.randn(B, N, 1, device=device)

    Phi_gal_var, entropy_gal_var = compute_attention_entropy(model_gal, xs_variable, us_variable)
    print(f"Attention entropy (variable): {entropy_gal_var:.4f}")
    print(f"Attention pattern sparsity: {(Phi_gal_var < 0.01).float().mean().item():.2%} of weights < 0.01")
    print(f"Entropy increase: {entropy_gal_var - entropy_gal_fixed:.4f} ({100*(entropy_gal_var/entropy_gal_fixed - 1):.1f}%)")

    # Test 3: Petrov-Galerkin with fixed sensors
    print("\n3. PETROV-GALERKIN - Fixed Sensor Locations")
    print("-" * 80)

    model_pg = SetONet(**config, branch_head_type="petrov_attention").to(device)

    Phi_pg_fixed, entropy_pg_fixed = compute_attention_entropy(model_pg, xs_fixed, us_fixed)
    print(f"Attention entropy (fixed): {entropy_pg_fixed:.4f}")
    print(f"Attention pattern sparsity: {(Phi_pg_fixed < 0.01).float().mean().item():.2%} of weights < 0.01")

    # Test 4: Petrov-Galerkin with variable sensors
    print("\n4. PETROV-GALERKIN - Variable Sensor Locations")
    print("-" * 80)

    Phi_pg_var, entropy_pg_var = compute_attention_entropy(model_pg, xs_variable, us_variable)
    print(f"Attention entropy (variable): {entropy_pg_var:.4f}")
    print(f"Attention pattern sparsity: {(Phi_pg_var < 0.01).float().mean().item():.2%} of weights < 0.01")
    print(f"Entropy increase: {entropy_pg_var - entropy_pg_fixed:.4f} ({100*(entropy_pg_var/entropy_pg_fixed - 1):.1f}%)")

    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY: Sensitivity to Variable Sensors")
    print("=" * 80)

    print(f"\nGalerkin PoU entropy increase: {entropy_gal_var - entropy_gal_fixed:.4f} "
          f"({100*(entropy_gal_var/entropy_gal_fixed - 1):.1f}%)")
    print(f"Petrov-Galerkin entropy increase: {entropy_pg_var - entropy_pg_fixed:.4f} "
          f"({100*(entropy_pg_var/entropy_pg_fixed - 1):.1f}%)")

    print("\nInterpretation:")
    print("- Higher entropy increase = more sensitive to variable sensors (worse)")
    print("- Lower entropy = sharper attention (better for learning)")

    if entropy_gal_var > entropy_pg_var:
        print(f"\n⚠️  Galerkin PoU has {entropy_gal_var/entropy_pg_var:.2f}x higher entropy with variable sensors!")
        print("    → PoU constraint forces diffuse attention when sensors vary")
    else:
        print(f"\n✓ Galerkin PoU maintains sharp attention with variable sensors")


if __name__ == "__main__":
    test_fixed_vs_variable_sensors()
