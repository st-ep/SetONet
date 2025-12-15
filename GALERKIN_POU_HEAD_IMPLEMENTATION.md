# Galerkin Partition-of-Unity (PoU) Head Implementation

## Overview

A new branch head type `galerkin_pou` has been successfully implemented for SetONet. This head uses a learned partition-of-unity over tokens with Galerkin-style quadrature integration.

## Key Architectural Difference

**Old heads (Standard/PetrovGalerkin)**: Softmax over sensors → "each token selects sensors then averages" (conditional expectation pooling)

**New GalerkinPoU head**: Softmax over tokens → "each sensor assigns its mass to tokens via PoU basis evaluation", then coefficients are quadrature sums over sensors

## Mathematical Interpretation

- **Tokens k** represent basis/test functions φ_k
- **Sensors i** represent quadrature points x_i with weights w_i
- **Partition-of-unity**: Φ_{k,i} = softmax_k(Q_k · K_i / √d_k) enforces Σ_k Φ_{k,i} = 1 for each sensor i
- **Galerkin coefficient**: c_k = Σ_i w_i Φ_{k,i} V_i (quadrature sum)

This makes the key/value split principled:
- **Keys** depend only on sensor position x (basis evaluation depends only on spatial location)
- **Values** depend on (x, u) (the integrand carries both position and sensor value)
- **Weights** w_i multiply the integrand in the sum (standard quadrature, NOT added to logits)

## Files Created/Modified

### New Files
1. **[Models/utils/galerkin_head.py](Models/utils/galerkin_head.py)** - GalerkinPoUHead implementation
2. **[tests/test_galerkin_head.py](tests/test_galerkin_head.py)** - Comprehensive test suite

### Modified Files
1. **[Models/SetONet.py](Models/SetONet.py)** - Integrated new head as third branch option
2. **Benchmark scripts** (all 8 scripts updated):
   - [Benchmarks/run_SetONet/run_1d.py](Benchmarks/run_SetONet/run_1d.py)
   - [Benchmarks/run_SetONet/run_consantration_2d.py](Benchmarks/run_SetONet/run_consantration_2d.py)
   - [Benchmarks/run_SetONet/run_darcy_1d.py](Benchmarks/run_SetONet/run_darcy_1d.py)
   - [Benchmarks/run_SetONet/run_chladni_2d.py](Benchmarks/run_SetONet/run_chladni_2d.py)
   - [Benchmarks/run_SetONet/run_elastic_2d.py](Benchmarks/run_SetONet/run_elastic_2d.py)
   - [Benchmarks/run_SetONet/run_dynamic_chladni.py](Benchmarks/run_SetONet/run_dynamic_chladni.py)
   - [Benchmarks/run_SetONet/run_heat_2d.py](Benchmarks/run_SetONet/run_heat_2d.py)
   - [Benchmarks/run_SetONet/run_transoprt.py](Benchmarks/run_SetONet/run_transoprt.py)

## Usage

### Python API

```python
from Models.SetONet import SetONet
import torch.nn as nn

model = SetONet(
    input_size_src=2,
    output_size_src=1,
    input_size_tgt=2,
    output_size_tgt=1,
    p=128,
    phi_hidden_size=256,
    rho_hidden_size=256,
    trunk_hidden_size=256,
    n_trunk_layers=4,
    activation_fn=nn.ReLU,

    # Galerkin PoU head configuration
    branch_head_type="galerkin_pou",
    galerkin_dk=64,                          # Key/query dimension (default: phi_output_size)
    galerkin_dv=64,                          # Value dimension (default: phi_output_size)
    galerkin_normalize="total",              # Normalization: "none" | "total" | "token"
    galerkin_learn_temperature=False,        # Learn softmax temperature
)
```

### Command Line (Benchmark Scripts)

```bash
# Example: Run 1D benchmark with Galerkin PoU head
python Benchmarks/run_SetONet/run_1d.py \
    --benchmark derivative \
    --son_branch_head_type galerkin_pou \
    --son_galerkin_dk 64 \
    --son_galerkin_dv 64 \
    --son_galerkin_normalize total \
    --son_galerkin_learn_temperature \
    --son_epochs 50000
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `branch_head_type` | str | "standard" | Set to "galerkin_pou" to use this head |
| `galerkin_dk` | int | phi_output_size | Key/query embedding dimension |
| `galerkin_dv` | int | phi_output_size | Value embedding dimension |
| `galerkin_normalize` | str | "total" | Normalization strategy (see below) |
| `galerkin_learn_temperature` | bool | False | Learn temperature parameter τ |

### Normalization Modes

- **"none"**: c_k = Σ_i w_i Φ_{k,i} V_i (raw quadrature sum)
- **"total"** (recommended): c_k = (Σ_i w_i Φ_{k,i} V_i) / (Σ_i w_i)
  - Scale-invariant w.r.t. number of sensors M
  - Recommended for variable-size sensor sets
- **"token"**: c_k = (Σ_i w_i Φ_{k,i} V_i) / (Σ_i w_i Φ_{k,i})
  - Per-token mass normalization

## Test Results

All tests passed ✓ (6/6):

1. **Shape Check**: Correct output shapes (B, M, dout)
2. **Permutation Invariance**: Max difference < 2.24e-08 after sensor permutation
3. **Mask Handling**: Perfect masking (difference 0.00e+00)
4. **Weight Handling**: Correct quadrature weight application
5. **Normalization Modes**: All modes produce valid outputs (no NaN/Inf)
6. **Comparison**: 41% parameter count vs standard head (37K vs 89K)

### Run Tests

```bash
python tests/test_galerkin_head.py
```

## Backward Compatibility

✓ **Fully backward compatible**
- Default `branch_head_type="standard"` unchanged
- Existing `"petrov_attention"` head works identically
- No breaking changes to existing APIs
- All existing training scripts work without modification

## Performance Characteristics

- **Parameter efficiency**: ~41% of standard head parameter count
- **Permutation invariant**: Exact invariance to sensor ordering
- **Supports variable M**: Handles different numbers of sensors per batch
- **Mask/weight aware**: Proper handling of invalid sensors and quadrature weights

## Key Implementation Details

### GalerkinPoUHead Class

Located in [Models/utils/galerkin_head.py](Models/utils/galerkin_head.py)

```python
class GalerkinPoUHead(nn.Module):
    def forward(self, x_enc, u, sensor_mask=None, sensor_weights=None):
        # 1. Compute keys and values
        K = self.key_net(x_enc)                    # (B, N, dk)
        V = self.value_net(concat([x_enc, u]))     # (B, N, dv)
        Q = self.query_tokens.expand(B, -1, -1)    # (B, p, dk)

        # 2. Attention scores
        scores = einsum("bpk,bnk->bpn", Q, K) / sqrt(dk)

        # 3. CRITICAL: Partition-of-unity over TOKENS
        Phi = softmax(scores, dim=1)  # (B, p, N) - softmax over token dim!

        # 4. Quadrature sum
        pooled = einsum("bpn,bn,bnd->bpd", Phi, w, V)

        # 5. Optional normalization
        if normalize == "total":
            pooled = pooled / w.sum(dim=1).clamp_min(eps)

        # 6. Output projection
        return self.rho_token(pooled)  # (B, p, dout)
```

### Integration in SetONet

The head is integrated as a third branch option alongside "standard" and "petrov_attention":

```python
# In SetONet.__init__:
if branch_head_type == "galerkin_pou":
    from .utils.galerkin_head import GalerkinPoUHead
    self.galerkin_head = GalerkinPoUHead(...)

# In forward_branch:
if self.branch_head_type == "galerkin_pou":
    x_enc = self._sinusoidal_encoding(xs) if self.use_positional_encoding else xs
    return self.galerkin_head(x_enc, us, sensor_mask=sensor_mask,
                             sensor_weights=sensor_weights)
```

## Future Work / Extensions

Potential enhancements:
1. Multi-head Galerkin attention (like multi-head self-attention)
2. Learnable basis functions (beyond queries)
3. Hierarchical quadrature schemes
4. Adaptive quadrature weight learning

## References

- DeepONet paper: Lu et al., "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators"
- Set Transformer: Lee et al., "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
- Petrov-Galerkin methods: Classic numerical analysis for weak formulations of PDEs

## Contact

For questions or issues, please create an issue in the repository.
