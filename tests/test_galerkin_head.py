#!/usr/bin/env python
"""
Test suite for GalerkinPoUHead implementation.

Tests:
1. Shape checks
2. Permutation invariance
3. Mask handling
4. Weight handling
5. Normalization modes
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from Models.SetONet import SetONet
import torch.nn as nn


def test_shape_check():
    """Test that output shapes are correct."""
    print("=" * 70)
    print("TEST 1: Shape Check")
    print("=" * 70)

    device = torch.device("cpu")
    B, N, M = 4, 50, 100  # batch, n_sensors, n_query_points
    dx = 2  # 2D problem
    du = 1  # scalar sensor values
    dy = 2  # 2D query
    dout = 1  # scalar output
    p = 32

    # Create model with galerkin_pou head
    model = SetONet(
        input_size_src=dx,
        output_size_src=du,
        input_size_tgt=dy,
        output_size_tgt=dout,
        p=p,
        phi_hidden_size=128,
        rho_hidden_size=128,
        trunk_hidden_size=128,
        n_trunk_layers=3,
        activation_fn=nn.ReLU,
        branch_head_type="galerkin_pou",
        galerkin_dk=64,
        galerkin_dv=64,
        galerkin_normalize="total",
    ).to(device)

    # Create random input data
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    ys = torch.randn(B, M, dy, device=device)

    # Forward pass
    output = model(xs, us, ys)

    # Check shapes
    expected_shape = (B, M, dout)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    print(f"✓ Input shapes: xs={xs.shape}, us={us.shape}, ys={ys.shape}")
    print(f"✓ Output shape: {output.shape} (expected: {expected_shape})")
    print(f"✓ Shape check PASSED\n")

    return True


def test_permutation_invariance():
    """Test that permuting sensors doesn't change output."""
    print("=" * 70)
    print("TEST 2: Permutation Invariance")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(42)

    B, N, M = 2, 30, 50
    dx, du, dy, dout = 2, 1, 2, 1

    model = SetONet(
        input_size_src=dx,
        output_size_src=du,
        input_size_tgt=dy,
        output_size_tgt=dout,
        p=16,
        phi_hidden_size=64,
        rho_hidden_size=64,
        trunk_hidden_size=64,
        n_trunk_layers=2,
        activation_fn=nn.ReLU,
        branch_head_type="galerkin_pou",
        galerkin_normalize="total",
    ).to(device)

    model.eval()  # Set to eval mode

    # Create input data
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    ys = torch.randn(B, M, dy, device=device)

    # Forward pass with original order
    with torch.no_grad():
        output1 = model(xs, us, ys)

    # Create random permutation
    perm = torch.randperm(N)
    xs_perm = xs[:, perm, :]
    us_perm = us[:, perm, :]

    # Forward pass with permuted sensors
    with torch.no_grad():
        output2 = model(xs_perm, us_perm, ys)

    # Check difference
    max_diff = torch.max(torch.abs(output1 - output2)).item()
    mean_diff = torch.mean(torch.abs(output1 - output2)).item()

    print(f"✓ Permutation applied to {N} sensors")
    print(f"✓ Max absolute difference: {max_diff:.2e}")
    print(f"✓ Mean absolute difference: {mean_diff:.2e}")

    tolerance = 1e-5
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

    print(f"✓ Permutation invariance PASSED (tolerance: {tolerance})\n")

    return True


def test_mask_handling():
    """Test that sensor masking works correctly."""
    print("=" * 70)
    print("TEST 3: Mask Handling")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(123)

    B, N, M = 2, 40, 30
    dx, du, dy, dout = 2, 1, 2, 1

    model = SetONet(
        input_size_src=dx,
        output_size_src=du,
        input_size_tgt=dy,
        output_size_tgt=dout,
        p=16,
        phi_hidden_size=64,
        rho_hidden_size=64,
        trunk_hidden_size=64,
        n_trunk_layers=2,
        activation_fn=nn.ReLU,
        branch_head_type="galerkin_pou",
        galerkin_normalize="total",
    ).to(device)

    model.eval()

    # Create input data
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    ys = torch.randn(B, M, dy, device=device)

    # Create mask that marks last k sensors as invalid
    k = 10
    sensor_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    sensor_mask[:, -k:] = False  # Mark last k sensors as invalid

    # Forward with mask
    with torch.no_grad():
        output_masked = model(xs, us, ys, sensor_mask=sensor_mask)

    # Forward with physically removed sensors (should be approximately equal)
    xs_truncated = xs[:, :-k, :]
    us_truncated = us[:, :-k, :]
    with torch.no_grad():
        output_truncated = model(xs_truncated, us_truncated, ys)

    # Check difference
    max_diff = torch.max(torch.abs(output_masked - output_truncated)).item()
    mean_diff = torch.mean(torch.abs(output_masked - output_truncated)).item()

    print(f"✓ Masked out {k}/{N} sensors")
    print(f"✓ Max difference (masked vs truncated): {max_diff:.2e}")
    print(f"✓ Mean difference: {mean_diff:.2e}")

    tolerance = 1e-5
    assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

    print(f"✓ Mask handling PASSED (tolerance: {tolerance})\n")

    return True


def test_weight_handling():
    """Test that quadrature weights are properly applied."""
    print("=" * 70)
    print("TEST 4: Weight Handling")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(456)

    B, N, M = 2, 30, 20
    dx, du, dy, dout = 1, 1, 1, 1  # 1D for simplicity

    model = SetONet(
        input_size_src=dx,
        output_size_src=du,
        input_size_tgt=dy,
        output_size_tgt=dout,
        p=16,
        phi_hidden_size=64,
        rho_hidden_size=64,
        trunk_hidden_size=64,
        n_trunk_layers=2,
        activation_fn=nn.ReLU,
        branch_head_type="galerkin_pou",
        galerkin_normalize="total",
    ).to(device)

    model.eval()

    # Create input data
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    ys = torch.randn(B, M, dy, device=device)

    # Test 1: Uniform weights should give same result as no weights
    weights_uniform = torch.ones(B, N, device=device)
    with torch.no_grad():
        output_with_uniform = model(xs, us, ys, sensor_weights=weights_uniform)
        output_no_weights = model(xs, us, ys, sensor_weights=None)

    diff_uniform = torch.max(torch.abs(output_with_uniform - output_no_weights)).item()
    print(f"✓ Difference with uniform weights vs no weights: {diff_uniform:.2e}")
    assert diff_uniform < 1e-5, f"Uniform weights differ from no weights: {diff_uniform}"

    # Test 2: Zero weights should eliminate contribution
    # Set half the sensors to zero weight
    weights_half_zero = torch.ones(B, N, device=device)
    weights_half_zero[:, N//2:] = 0.0

    with torch.no_grad():
        output_half_weighted = model(xs, us, ys, sensor_weights=weights_half_zero)

    # Compare with actually removing those sensors
    xs_half = xs[:, :N//2, :]
    us_half = us[:, :N//2, :]

    with torch.no_grad():
        output_half_removed = model(xs_half, us_half, ys)

    diff_removal = torch.max(torch.abs(output_half_weighted - output_half_removed)).item()
    print(f"✓ Zero weights vs physical removal: {diff_removal:.2e}")
    assert diff_removal < 1e-5, f"Zero weights don't match physical removal: {diff_removal}"

    # Test 3: Different weights should produce different outputs
    weights_random = torch.rand(B, N, device=device) + 0.1  # Avoid zeros

    with torch.no_grad():
        output_random_weights = model(xs, us, ys, sensor_weights=weights_random)

    diff_random = torch.max(torch.abs(output_random_weights - output_no_weights)).item()
    print(f"✓ Random weights vs no weights: {diff_random:.2e}")
    # Random weights should produce different output
    assert diff_random > 1e-6, f"Random weights don't change output significantly: {diff_random}"

    print(f"✓ Weight handling PASSED\n")

    return True


def test_normalization_modes():
    """Test different normalization modes."""
    print("=" * 70)
    print("TEST 5: Normalization Modes")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(789)

    B, N, M = 2, 30, 20
    dx, du, dy, dout = 2, 1, 2, 1

    results = {}

    for normalize_mode in ["none", "total", "token"]:
        model = SetONet(
            input_size_src=dx,
            output_size_src=du,
            input_size_tgt=dy,
            output_size_tgt=dout,
            p=16,
            phi_hidden_size=64,
            rho_hidden_size=64,
            trunk_hidden_size=64,
            n_trunk_layers=2,
            activation_fn=nn.ReLU,
            branch_head_type="galerkin_pou",
            galerkin_normalize=normalize_mode,
        ).to(device)

        model.eval()

        # Create input data
        xs = torch.randn(B, N, dx, device=device)
        us = torch.randn(B, N, du, device=device)
        ys = torch.randn(B, M, dy, device=device)

        with torch.no_grad():
            output = model(xs, us, ys)

        results[normalize_mode] = {
            "mean": output.mean().item(),
            "std": output.std().item(),
            "min": output.min().item(),
            "max": output.max().item(),
        }

        print(f"✓ Mode '{normalize_mode}':")
        print(f"  Mean: {results[normalize_mode]['mean']:.4f}, Std: {results[normalize_mode]['std']:.4f}")

    # All modes should produce valid outputs (no NaN or Inf)
    for mode, stats in results.items():
        assert not np.isnan(stats['mean']), f"NaN detected in mode '{mode}'"
        assert not np.isinf(stats['mean']), f"Inf detected in mode '{mode}'"

    print(f"✓ All normalization modes produce valid outputs")
    print(f"✓ Normalization modes PASSED\n")

    return True


def test_comparison_with_standard_head():
    """Compare galerkin_pou with standard head on same task."""
    print("=" * 70)
    print("TEST 6: Comparison with Standard Head")
    print("=" * 70)

    device = torch.device("cpu")
    torch.manual_seed(999)

    B, N, M = 2, 40, 30
    dx, du, dy, dout = 2, 1, 2, 1

    # Create models with both heads
    model_galerkin = SetONet(
        input_size_src=dx,
        output_size_src=du,
        input_size_tgt=dy,
        output_size_tgt=dout,
        p=16,
        phi_hidden_size=64,
        rho_hidden_size=64,
        trunk_hidden_size=64,
        n_trunk_layers=2,
        activation_fn=nn.ReLU,
        branch_head_type="galerkin_pou",
        galerkin_normalize="total",
    ).to(device)

    model_standard = SetONet(
        input_size_src=dx,
        output_size_src=du,
        input_size_tgt=dy,
        output_size_tgt=dout,
        p=16,
        phi_hidden_size=64,
        rho_hidden_size=64,
        trunk_hidden_size=64,
        n_trunk_layers=2,
        activation_fn=nn.ReLU,
        branch_head_type="standard",
        aggregation_type="attention",
    ).to(device)

    # Count parameters
    params_galerkin = sum(p.numel() for p in model_galerkin.parameters())
    params_standard = sum(p.numel() for p in model_standard.parameters())

    print(f"✓ Galerkin PoU head parameters: {params_galerkin:,}")
    print(f"✓ Standard head parameters: {params_standard:,}")
    print(f"✓ Parameter ratio (Galerkin/Standard): {params_galerkin/params_standard:.2f}")

    # Both models should run without errors
    xs = torch.randn(B, N, dx, device=device)
    us = torch.randn(B, N, du, device=device)
    ys = torch.randn(B, M, dy, device=device)

    with torch.no_grad():
        output_galerkin = model_galerkin(xs, us, ys)
        output_standard = model_standard(xs, us, ys)

    print(f"✓ Galerkin output range: [{output_galerkin.min():.4f}, {output_galerkin.max():.4f}]")
    print(f"✓ Standard output range: [{output_standard.min():.4f}, {output_standard.max():.4f}]")
    print(f"✓ Both models run successfully")
    print(f"✓ Comparison PASSED\n")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GALERKIN PARTITION-OF-UNITY HEAD TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        ("Shape Check", test_shape_check),
        ("Permutation Invariance", test_permutation_invariance),
        ("Mask Handling", test_mask_handling),
        ("Weight Handling", test_weight_handling),
        ("Normalization Modes", test_normalization_modes),
        ("Comparison with Standard", test_comparison_with_standard_head),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}\n")
            failed += 1
            import traceback
            traceback.print_exc()

    print("=" * 70)
    print(f"TEST SUMMARY: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"WARNING: {failed} test(s) failed")
    else:
        print("ALL TESTS PASSED ✓")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
