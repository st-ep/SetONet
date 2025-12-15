# Galerkin Partition-of-Unity Head Architecture

## Visual Architecture Diagram

```
INPUT DATA (per batch)
═══════════════════════════════════════════════════════════════════

x_enc: (B, N, dx_enc=64)  ← Encoded sensor positions (sinusoidal)
u:     (B, N, du=1)       ← Sensor values

═══════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────────┐
                    │   GALERKIN PoU BRANCH HEAD      │
                    └─────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  KEY NETWORK (learns spatial encoding)                           │
│  Input: x_enc (B, N, 64)                                         │
│                                                                   │
│  Layer 1: Linear(64 → 256)     [Parameters: 64×256 + 256]       │
│           ↓ (B, N, 256)        = 16,640                          │
│  Layer 2: ReLU()               [Parameters: 0]                   │
│           ↓ (B, N, 256)                                          │
│  Layer 3: Linear(256 → 256)    [Parameters: 256×256 + 256]      │
│           ↓ (B, N, 256)        = 65,792                          │
│  Layer 4: ReLU()               [Parameters: 0]                   │
│           ↓ (B, N, 256)                                          │
│  Layer 5: Linear(256 → 64)     [Parameters: 256×64 + 64]        │
│           ↓ (B, N, 64)         = 16,448                          │
│                                                                   │
│  Output: K (B, N, dk=64)       [Total: 98,880 params]           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  VALUE NETWORK (learns integrand)                                │
│  Input: concat([x_enc, u], dim=-1) = (B, N, 65)                 │
│                                                                   │
│  Layer 1: Linear(65 → 256)     [Parameters: 65×256 + 256]       │
│           ↓ (B, N, 256)        = 16,896                          │
│  Layer 2: ReLU()               [Parameters: 0]                   │
│           ↓ (B, N, 256)                                          │
│  Layer 3: Linear(256 → 256)    [Parameters: 256×256 + 256]      │
│           ↓ (B, N, 256)        = 65,792                          │
│  Layer 4: ReLU()               [Parameters: 0]                   │
│           ↓ (B, N, 256)                                          │
│  Layer 5: Linear(256 → 64)     [Parameters: 256×64 + 64]        │
│           ↓ (B, N, 64)         = 16,448                          │
│                                                                   │
│  Output: V (B, N, dv=64)       [Total: 99,136 params]           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  LEARNABLE QUERY TOKENS (basis function parameters)              │
│                                                                   │
│  Shape: (1, p=128, dk=64)      [Parameters: 128×64 = 8,192]     │
│                                                                   │
│  Expanded to: Q (B, 128, 64)   [Total: 8,192 params]            │
└──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
                    ATTENTION COMPUTATION
═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Compute attention scores                                │
│                                                                   │
│  scores = einsum("bpk,bnk->bpn", Q, K) / sqrt(dk)               │
│         = (B, 128, 64) @ (B, N, 64)^T / sqrt(64)                │
│         → (B, 128, N)                                            │
│                                                                   │
│  Optional: scores = scores / tau  (if learn_temperature=True)   │
│            [Additional parameters: 1]                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: Partition-of-Unity Softmax (CRITICAL!)                  │
│                                                                   │
│  Phi = softmax(scores, dim=1)  ← Softmax over TOKEN dimension!  │
│      = (B, 128, N)                                               │
│                                                                   │
│  Ensures: Σ_k Phi[b, k, i] = 1  for each sensor i               │
│  (Partition-of-unity constraint)                                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Galerkin Quadrature Sum                                 │
│                                                                   │
│  pooled = einsum("bpn,bn,bnd->bpd", Phi, w, V)                  │
│         = Σ_i w[b,i] × Phi[b,p,i] × V[b,i,d]                   │
│         → (B, 128, 64)                                           │
│                                                                   │
│  If normalize="total":                                           │
│    pooled = pooled / (Σ_i w[b,i])                               │
│    (Scale-invariant w.r.t. number of sensors)                   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  RHO_TOKEN (output projection, only if dv ≠ dout)                │
│  Input: pooled (B, 128, 64)                                      │
│                                                                   │
│  IF dv == dout:                                                  │
│    Layer: Identity()           [Parameters: 0]                   │
│                                                                   │
│  IF dv ≠ dout (our case: dv=64, dout=1):                        │
│    Layer 1: Linear(64 → 256)   [Parameters: 64×256 + 256]       │
│             ↓ (B, 128, 256)    = 16,640                          │
│    Layer 2: ReLU()             [Parameters: 0]                   │
│             ↓ (B, 128, 256)                                      │
│    Layer 3: Linear(256 → 1)    [Parameters: 256×1 + 1]          │
│             ↓ (B, 128, 1)      = 257                             │
│                                                                   │
│  Output: b (B, p=128, dout=1)  [Total: 16,897 params]           │
└──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════

b: (B, p=128, dout=1)  ← Branch coefficients for DeepONet

═══════════════════════════════════════════════════════════════════
```

## Parameter Count Summary

### 2D Problem Configuration (p=128, dk=64, dv=64, hidden=256)

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| **Key Network** | 98,880 | (64→256): 16,640<br>(256→256): 65,792<br>(256→64): 16,448 |
| **Value Network** | 99,136 | (65→256): 16,896<br>(256→256): 65,792<br>(256→64): 16,448 |
| **Query Tokens** | 8,192 | 128 × 64 |
| **Rho Token** | 16,897 | (64→256): 16,640<br>(256→1): 257 |
| **Learn Temp** | 0 or 1 | Optional |
| **TOTAL** | **223,105** | |

---

## 1D Problem Configuration (p=32, dk=64, dv=64, hidden=256)

```
INPUT:
x_enc: (B, N, 64)  ← Sinusoidal encoding (1D positions → 64D)
u:     (B, N, 1)   ← Scalar sensor values

KEY NETWORK:
  Linear(64 → 256):    16,640 params
  ReLU
  Linear(256 → 256):   65,792 params
  ReLU
  Linear(256 → 64):    16,448 params
  → K (B, N, 64)
  Total: 98,880 params

VALUE NETWORK:
  Linear(65 → 256):    16,896 params  ← Note: 64+1=65 input
  ReLU
  Linear(256 → 256):   65,792 params
  ReLU
  Linear(256 → 64):    16,448 params
  → V (B, N, 64)
  Total: 99,136 params

QUERY TOKENS:
  Shape: (1, 32, 64)
  Total: 2,048 params  ← Smaller p = fewer params

RHO TOKEN:
  Linear(64 → 256):    16,640 params
  ReLU
  Linear(256 → 1):     257 params
  → b (B, 32, 1)
  Total: 16,897 params

TOTAL: 216,961 parameters
```

---

## Comparison: Standard vs Galerkin PoU

### 2D Problem (p=128)

**Standard Head:**
```
PHI NETWORK:
  Input: concat([x_enc, u]) = (B, N, 65)
  Linear(65 → 256):      16,896 params
  ReLU
  Linear(256 → 256):     65,792 params
  ReLU
  Linear(256 → 32):      8,224 params
  → (B, N, 32)
  Total: 90,912 params

ATTENTION POOL:
  query_tokens: (1, 1, 32)         32 params
  MultiheadAttention(32, 4 heads): 4,224 params
  → (B, 1, 32) = (B, 32)
  Total: 4,256 params

RHO NETWORK:
  Linear(32 → 256):      8,448 params
  ReLU
  Linear(256 → 128):     32,896 params
  → (B, 128)
  Total: 41,344 params

BRANCH TOTAL: 136,512 params
```

**Galerkin PoU:**
```
BRANCH TOTAL: 223,105 params
```

**Ratio: 223,105 / 136,512 = 1.63x more parameters**

---

## Key Architectural Insights

### 1. Separate Key/Value Networks
- **Key network**: Only sees positions (x_enc)
  - Learns spatial basis function structure
  - 98,880 parameters

- **Value network**: Sees positions + values (x_enc, u)
  - Learns the integrand at each point
  - 99,136 parameters

### 2. Learnable Tokens as Basis Functions
- `query_tokens` of shape (p, dk)
- Represents p basis/test functions
- For p=128: only 8,192 params (efficient!)

### 3. Softmax Over Tokens (Not Sensors!)
```python
Phi = softmax(scores, dim=1)  # dim=1 = token axis
```
- Enforces partition-of-unity: Σ_k Φ_{k,i} = 1
- Each sensor distributes its mass across all tokens

### 4. Quadrature Structure
```python
c_k = Σ_i w_i Φ_{k,i} V_i / (Σ_i w_i)
```
- Weights multiply the integrand (standard quadrature)
- Normalization ensures scale invariance

---

## Memory Footprint (Forward Pass)

### Intermediate Activations (B=32, N=100, p=128)

| Tensor | Shape | Memory |
|--------|-------|--------|
| x_enc | (32, 100, 64) | 819 KB |
| u | (32, 100, 1) | 13 KB |
| K | (32, 100, 64) | 819 KB |
| V | (32, 100, 64) | 819 KB |
| Q | (32, 128, 64) | 1,049 KB |
| scores | (32, 128, 100) | 1,638 KB |
| Phi | (32, 128, 100) | 1,638 KB |
| pooled | (32, 128, 64) | 1,049 KB |
| b | (32, 128, 1) | 16 KB |
| **Total** | | **~8 MB** |

Compare to standard head: ~3 MB (less intermediate tensors)

---

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Key network forward | O(N × dk × hidden) | Linear in N |
| Value network forward | O(N × dv × hidden) | Linear in N |
| Score computation | O(p × N × dk) | Attention scores |
| Softmax | O(p × N) | Over token dim |
| Quadrature sum | O(p × N × dv) | einsum |
| **Total** | **O(N × p × max(dk, dv, hidden))** | Linear in sensors |

Efficient for variable N because of linear scaling!

---

## Variants & Extensions

### Smaller Model (Fewer Params)
```python
# Reduce dk, dv to match phi_output_size
galerkin_dk=32  # Instead of 64
galerkin_dv=32  # Instead of 64

# Result: ~120K params (closer to standard)
```

### Larger Model (More Capacity)
```python
# Increase embedding dimensions
galerkin_dk=128
galerkin_dv=128

# Result: ~400K params (2x standard)
```

### With Learnable Temperature
```python
galerkin_learn_temperature=True

# Adds 1 parameter: log_tau
# Allows model to learn softmax sharpness
```

---

## Summary

**Galerkin PoU Head (2D, p=128):**
- **Total Parameters**: 223,105
- **Main Networks**: key_net (99K) + value_net (99K) + tokens (8K) + rho (17K)
- **Ratio vs Standard**: 1.63x more parameters
- **Advantage**: Principled Galerkin structure, scale-invariant, compositional basis

The extra parameters are concentrated in the key/value networks, which learn richer spatial and integrand representations!
