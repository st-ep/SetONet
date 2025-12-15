# Iterative Query Refinement for Petrov-Galerkin Attention

## Overview

Add optional iterative refinement to PetrovGalerkinHead where queries are updated based on attended sensor information across multiple iterations.

**Current flow (single-pass):**
```
Q₀ (learned) → Attend(Q₀, K, V) → pooled → rho_token → output
```

**Proposed flow (iterative):**
```
Q₀ (learned) → Attend(Q₀, K, V) → update Q → Q₁ → Attend(Q₁, K, V) → ... → final pooled → rho_token → output
```

## Design Decisions

### 1. Number of Iterations
- New parameter: `n_refine_iters: int = 0` (0 = disabled, behaves exactly as current)
- Default to 0 for backward compatibility
- Typical values: 1-3 iterations

### 2. Query Update Mechanism
**Chosen approach: Residual MLP update with LayerNorm**

```python
# After getting pooled values from attention:
pooled = Attend(Q, K, V)  # (B, p, dv)
# Project pooled to query space and add residually:
delta_Q = self.refine_mlp(pooled)  # (B, p, dk)
Q = self.refine_ln(Q + delta_Q)  # LayerNorm for stability
```

Rationale:
- Simple and proven effective (similar to transformer decoder refinement)
- Residual connection ensures gradients flow easily
- LayerNorm stabilizes iterative updates
- MLP allows flexible transformation from value space (dv) to query space (dk)

### 3. Weight Sharing
**Chosen: Shared weights across iterations**

Rationale:
- Fewer parameters (~3x fewer than unshared for 3 iterations)
- Works well in practice (DETR uses this)
- Can always add unshared option later if needed

### 4. Parameter Cost Estimate
New parameters per PetrovGalerkinHead (when enabled):
- `refine_mlp`: Linear(dv, hidden) + Linear(hidden, dk) ≈ dv×hidden + hidden×dk
- `refine_ln`: LayerNorm(dk) ≈ 2×dk

For typical values (dv=32, dk=32, hidden=256):
- refine_mlp: 32×256 + 256×32 = 16,384 params
- refine_ln: 64 params
- **Total: ~16.5k new params** (only when n_refine_iters > 0)

## Implementation Plan

### Step 1: Modify PetrovGalerkinHead.__init__

Add new parameter and initialize refinement layers:

```python
def __init__(
    self,
    *,
    p: int,
    dx_enc: int,
    du: int,
    dout: int,
    dk: int,
    dv: int,
    hidden: int,
    activation_fn,
    eps: float = 1e-8,
    n_refine_iters: int = 0,  # NEW: 0 = disabled
) -> None:
    # ... existing init code ...

    self.n_refine_iters = n_refine_iters

    # Refinement layers (only created if n_refine_iters > 0)
    self.refine_mlp = None
    self.refine_ln = None
    if self.n_refine_iters > 0:
        self.refine_mlp = nn.Sequential(
            nn.Linear(dv, hidden),
            activation_fn(),
            nn.Linear(hidden, dk),
        )
        self.refine_ln = nn.LayerNorm(dk)
```

### Step 2: Modify PetrovGalerkinHead.forward

Add iteration loop:

```python
def forward(self, x_enc, u, sensor_mask=None, sensor_weights=None):
    # ... existing validation and K, V computation ...

    K = self.key_net(x_enc)  # (B, N, dk)
    V = self.value_net(torch.cat([x_enc, u], dim=-1))  # (B, N, dv)

    # Initialize queries
    Q = self.query_tokens.expand(batch_size, -1, -1)  # (B, p, dk)

    # Iterative refinement loop
    for _ in range(self.n_refine_iters + 1):  # +1 for the final pass
        # Compute attention scores
        scores = torch.einsum("bpk,bnk->bpn", Q, K) / math.sqrt(self.dk)

        if sensor_weights is not None:
            scores = scores + torch.log(sensor_weights.clamp_min(self.eps)).unsqueeze(1)
        if sensor_mask is not None:
            scores = scores.masked_fill(~sensor_mask.unsqueeze(1), -float("inf"))

        A = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bpn,bnd->bpd", A, V)  # (B, p, dv)

        # Update queries (except on final iteration)
        if self.refine_mlp is not None and _ < self.n_refine_iters:
            delta_Q = self.refine_mlp(pooled)
            Q = self.refine_ln(Q + delta_Q)

    return self.rho_token(pooled)
```

### Step 3: Update SetONet to pass new parameter

In SetONet.__init__, add `pg_n_refine_iters` parameter:

```python
def __init__(
    self,
    # ... existing params ...
    pg_n_refine_iters: int = 0,  # NEW
):
    # ... when creating PetrovGalerkinHead ...
    self.pg_head = PetrovGalerkinHead(
        # ... existing params ...
        n_refine_iters=pg_n_refine_iters,
    )
```

### Step 4: Update synthetic_1d_data.py

Same pattern - pass the parameter through during model creation.

## Files to Modify

1. **Models/utils/petrov_galerkin_head.py** - Core implementation
2. **Models/SetONet.py** - Pass through parameter
3. **Data/synthetic_1d_data.py** - Model creation

## Testing Strategy

1. **Backward compatibility**: Set `pg_n_refine_iters=0` (default) - should behave identically to current
2. **Sanity check**: Try `pg_n_refine_iters=2` on a simple benchmark (e.g., derivative task)
3. **Compare**: Test whether refinement improves accuracy on a benchmark

## Expected Benefits

- Queries can adapt to the specific sensor configuration in each sample
- Multiple passes allow the model to "refine" its understanding
- Similar to how DETR iteratively refines object queries

## Potential Concerns

- Increased compute per forward pass (linear in n_refine_iters)
- May need more training iterations to see benefits
- Could overfit if dataset is small
