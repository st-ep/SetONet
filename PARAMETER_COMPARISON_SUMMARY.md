# SetONet Branch Head Parameter Comparison

## Executive Summary

**Key Finding**: Petrov-Galerkin and Galerkin PoU heads have **identical parameter counts** and use **1.29-1.42x more parameters** than the standard head (depending on latent dimension p).

## Detailed Comparison Tables

### 1D Problems (p=32, 1D coordinates)

| Head Type | Total Params | Relative to Standard | Branch Params | Trunk Params |
|-----------|--------------|---------------------|---------------|--------------|
| **Standard** (baseline) | 252,161 | 1.00x (100%) | 111,840 | 140,320 |
| **Petrov-Galerkin** | 357,282 | 1.42x (142%) | 216,961 | 140,320 |
| **Galerkin PoU** | 357,282 | 1.42x (142%) | 216,961 | 140,320 |

**Standard head breakdown**:
- phi_network: 90,912 (36.1%)
- attention_pool: 4,256 (1.7%)
- rho_network: 16,672 (6.6%)
- trunk_network: 140,320 (55.6%)

**PG/Galerkin head breakdown** (identical):
- key_net + value_net + query_tokens + rho_token: 216,961 (60.7%)
- trunk_network: 140,320 (39.3%)

---

### 2D Problems (p=128, 2D coordinates)

| Head Type | Total Params | Relative to Standard | Branch Params | Trunk Params |
|-----------|--------------|---------------------|---------------|--------------|
| **Standard** (baseline) | 301,761 | 1.00x (100%) | 136,512 | 165,248 |
| **Petrov-Galerkin** | 388,354 | 1.29x (129%) | 223,105 | 165,248 |
| **Galerkin PoU** | 388,354 | 1.29x (129%) | 223,105 | 165,248 |

**Standard head breakdown**:
- phi_network: 90,912 (30.1%)
- attention_pool: 4,256 (1.4%)
- rho_network: 41,344 (13.7%)
- trunk_network: 165,248 (54.8%)

**PG/Galerkin head breakdown** (identical):
- key_net + value_net + query_tokens + rho_token: 223,105 (57.4%)
- trunk_network: 165,248 (42.6%)

---

### Scaling with Latent Dimension p

| Latent Dim (p) | Standard | Petrov-Galerkin | Galerkin PoU | PG/Gal Ratio |
|----------------|----------|-----------------|--------------|--------------|
| 32 | 252,417 | 357,538 | 357,538 | **1.42x** |
| 128 | 301,761 | 388,354 | 388,354 | **1.29x** |
| 256 | 367,553 | 429,442 | 429,442 | **1.17x** |

**Observation**: As p increases, the parameter count ratio decreases because trunk network parameters (which scale with p) become more dominant.

---

## Why Petrov-Galerkin and Galerkin PoU Have the Same Count?

Both heads use nearly identical architectures:

### Petrov-Galerkin Head
```
key_net:    x_enc → dk        (learns spatial encoding)
value_net:  (x_enc, u) → dv   (learns integrand)
query_tokens: [p, dk]         (learnable test functions)
rho_token:  dv → dout         (output projection)

Computation: Softmax over SENSORS
  Φ_{k,i} = softmax_i(Q_k · K_i)  ← softmax over sensor dimension
  c_k = Σ_i Φ_{k,i} V_i
```

### Galerkin PoU Head
```
key_net:    x_enc → dk        (learns spatial encoding)
value_net:  (x_enc, u) → dv   (learns integrand)
query_tokens: [p, dk]         (learnable basis functions)
rho_token:  dv → dout         (output projection)

Computation: Softmax over TOKENS (partition-of-unity)
  Φ_{k,i} = softmax_k(Q_k · K_i)  ← softmax over token dimension
  c_k = Σ_i w_i Φ_{k,i} V_i       ← multiply by quadrature weights
```

**Key architectural difference**: Only the softmax dimension and weight handling differ, but the network components are identical.

---

## Why Standard Head Has Fewer Parameters?

The standard head uses a different architecture optimized for the Deep Sets framework:

### Standard Head
```
phi:    (x_enc, u) → phi_output_size    (processes sensor pairs)
pool:   attention pooling                (aggregates to k tokens)
rho:    (k × phi_output_size) → p×dout  (maps to branch output)
```

The phi network is smaller (outputs to phi_output_size=32 instead of dk=64 or dv=64), and attention pooling is lightweight.

---

## Which Head to Choose?

| Criterion | Standard | Petrov-Galerkin | Galerkin PoU |
|-----------|----------|-----------------|--------------|
| **Parameter Count** | ✓ Smallest | ✗ 1.29-1.42x | ✗ 1.29-1.42x |
| **Mathematical Principle** | Deep Sets | Conditional expectation | Galerkin quadrature |
| **Weight Handling** | Not designed for | Log-space addition | Multiplicative (principled) |
| **Softmax Over** | - | Sensors | Tokens (PoU) |
| **Quadrature Interpretation** | ✗ | ✗ | ✓ Yes |
| **Production Use** | ✓ Default | ✓ Tested | ✓ New option |

### Recommendations

1. **Standard head**: Best for general use, proven architecture, smallest parameter count
2. **Petrov-Galerkin**: Use when you have sensor weights and want log-space integration
3. **Galerkin PoU**: Use when:
   - You have quadrature weights (w_i) that should multiply the integrand
   - You want mathematically principled partition-of-unity basis functions
   - You prefer tokens as basis functions rather than sensor selectors

---

## Performance vs Parameters Trade-off

**Important**: More parameters ≠ Better performance necessarily

- Standard head may be sufficient for many problems
- PG/Galerkin heads offer **different inductive biases** more than parameter efficiency
- The ~30-40% parameter increase is modest and may be worthwhile for problems where:
  - Quadrature interpretation matters
  - You have meaningful sensor weights
  - Spatial basis functions are conceptually important

---

## Configuration Tips

For **fair comparisons** between heads:

```python
# Keep these CONSTANT across heads:
p = 128                      # Latent dimension
phi_output_size = 32         # For standard head
phi_hidden_size = 256
rho_hidden_size = 256
trunk_hidden_size = 256

# Set these for PG/Galerkin heads:
pg_dk = 64                   # Or try matching phi_output_size
pg_dv = 64
galerkin_dk = 64
galerkin_dv = 64

# This gives PG/Galerkin ~1.3x parameters of standard
```

To match parameter counts more closely, reduce dk/dv:
```python
# Smaller embeddings (closer to standard head params)
pg_dk = 32
pg_dv = 32
galerkin_dk = 32
galerkin_dv = 32
```

---

## Conclusion

- **Petrov-Galerkin and Galerkin PoU are parameter-equivalent** (same internal architecture)
- **Standard head is most parameter-efficient** (1.00x baseline)
- **Choose based on mathematical interpretation**, not parameter count
- The difference is in **computation pattern** (softmax dimension, weight handling), not network capacity

All three heads are production-ready and backward-compatible.
