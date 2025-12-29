#!/usr/bin/env python
"""
transport_plan_dataset.py
----------------------------------
Dataset wrapper/loader for the OT *transport-plan* dataset (Option 1).

It mirrors the idea of your previous wrapper: provide a `sample()` method that returns
(xs, us, ys, G_u_ys, extra).

For this benchmark:
- xs: source points X (batch, M, 2)
- us: source weights (batch, M, 1)  (uniform ones; your model can normalize if desired)
- ys: query locations (batch, M, 2) = source points (we supervise only where mass exists)
- G_u_ys:
    mode="plan" (default): row-conditional transport plan W (batch, M, N)
    mode="coupling": coupling P = (1/M) W (batch, M, N)
    mode="barycentric": barycentric map U = W @ Y_fixed (batch, M, 2)

- extra: dict containing Y_fixed and (optionally) metadata.

Important practical detail:
- We DO NOT preload the full dataset onto GPU by default.
  Transport plans are large (M*N), so we store them on CPU and move only sampled batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk


class TransportPlanDataset:
    """Dataset wrapper for OT transport-plan learning."""

    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        device: str = "cuda",
        mode: str = "plan",
        preload_to_gpu: bool = False,
    ) -> None:
        """
        Args:
            dataset: HuggingFace dataset dict with 'train'/'test'
            batch_size: batch size for sampling
            device: device for returned tensors
            mode:
                - "plan": return row-conditional plan W (rows sum to 1)
                - "coupling": return coupling P = (1/M) * W
                - "barycentric": return barycentric map U = W @ Y_fixed
            preload_to_gpu: if True, move all tensors to GPU on init (not recommended for large datasets)
        """
        self.batch_size = int(batch_size)
        self.device = device
        self.mode = mode

        train_data = dataset["train"]
        self.n_samples = len(train_data)

        if self.n_samples == 0:
            raise ValueError("Empty training split in dataset.")

        # Inspect shapes from first sample
        s0 = train_data[0]
        y_fixed_np = np.array(s0["target_points"], dtype=np.float32)
        self._y_fixed_cpu = torch.tensor(y_fixed_np, dtype=torch.float32, device="cpu")
        self.n_target_points = int(self._y_fixed_cpu.shape[0])

        x0_np = np.array(s0["source_points"], dtype=np.float32)
        w0_np = np.array(s0["transport_plan"], dtype=np.float32)
        self.n_source_points = int(x0_np.shape[0])

        # Store all samples on CPU (safe for large W); move batch to GPU at sample()
        self.sources_cpu = []
        self.plans_cpu = []
        self.meta_cpu = []

        for i in range(self.n_samples):
            s = train_data[i]
            X = torch.tensor(np.array(s["source_points"], dtype=np.float32), dtype=torch.float32, device="cpu")
            W = torch.tensor(np.array(s["transport_plan"], dtype=np.float32), dtype=torch.float32, device="cpu")

            self.sources_cpu.append(X)
            self.plans_cpu.append(W)

            # Optional metadata useful for debugging/analysis
            meta = {
                "epsilon": float(s.get("epsilon", -1.0)),
                "ot_cost": float(s.get("ot_cost", np.nan)),
                "marginal_error": float(s.get("marginal_error", np.nan)),
                "domain_size": float(s.get("domain_size", 5.0)),
                "source_params": s.get("source_params", None),
            }
            self.meta_cpu.append(meta)

        # Optional: preload everything to GPU (only if dataset is small)
        self._preload_to_gpu = bool(preload_to_gpu)
        self.sources = self.sources_cpu
        self.plans = self.plans_cpu
        self._y_fixed_cache: Dict[str, torch.Tensor] = {}

        if self._preload_to_gpu:
            self.sources = [t.to(self.device, non_blocking=True) for t in self.sources_cpu]
            self.plans = [t.to(self.device, non_blocking=True) for t in self.plans_cpu]
            self._y_fixed_cache[self.device] = self._y_fixed_cpu.to(self.device, non_blocking=True)

        print("Loaded TransportPlanDataset:")
        print(f"  - #train samples: {self.n_samples}")
        print(f"  - Source points (M): {self.n_source_points}")
        print(f"  - Target points (N): {self.n_target_points} (fixed across samples)")
        print(f"  - Mode: {self.mode}")
        print(f"  - Preload to GPU: {self._preload_to_gpu}")

    def _get_y_fixed(self, device: Optional[str] = None) -> torch.Tensor:
        device = device or self.device
        if device in self._y_fixed_cache:
            return self._y_fixed_cache[device]
        y = self._y_fixed_cpu.to(device, non_blocking=True)
        self._y_fixed_cache[device] = y
        return y

    def sample(self, device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Sample a training batch.

        Returns:
            xs: (B, M, 2) source points
            us: (B, M, 1) source weights (ones)
            ys: (B, M, 2) query locations (source points)
            G_u_ys:
                - plan: (B, M, N) row-conditional W
                - coupling: (B, M, N) coupling P = W / M
                - barycentric: (B, M, 2) barycentric transported points U = W @ Y_fixed
            extra: dict with Y_fixed and per-sample metadata
        """
        device = device or self.device
        idx = torch.randint(0, self.n_samples, (self.batch_size,), device="cpu").tolist()

        xs_batch = []
        us_batch = []
        ys_batch = []
        out_batch = []
        meta_batch = []

        Y_fixed = self._get_y_fixed(device)  # (N,2)

        for i in idx:
            X = self.sources[i]
            W = self.plans[i]

            if not self._preload_to_gpu:
                X = X.to(device, non_blocking=True)
                W = W.to(device, non_blocking=True)

            # Input "values" for SetONet-style branch: uniform weights (you can normalize in model)
            u = torch.ones((X.shape[0], 1), dtype=torch.float32, device=device)

            # Query only where mass exists (source support)
            ys = X

            if self.mode == "plan":
                out = W  # (M,N), rows sum to 1
            elif self.mode == "coupling":
                out = W / float(X.shape[0])  # P = a*W, with a=1/M
            elif self.mode == "barycentric":
                # U_i = sum_j W_ij * y_j
                out = W @ Y_fixed  # (M,N) @ (N,2) -> (M,2)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            xs_batch.append(X)
            us_batch.append(u)
            ys_batch.append(ys)
            out_batch.append(out)
            meta_batch.append(self.meta_cpu[i])

        xs = torch.stack(xs_batch, dim=0)
        us = torch.stack(us_batch, dim=0)
        ys = torch.stack(ys_batch, dim=0)
        G_u_ys = torch.stack(out_batch, dim=0)

        extra = {
            "Y_fixed": Y_fixed,         # (N,2)
            "meta": meta_batch,         # list of dicts length B
        }
        return xs, us, ys, G_u_ys, extra

    def get_sample_for_visualization(self, idx: int = 0, device: str = "cpu") -> Dict:
        """Return a single sample for plotting/debug."""
        idx = int(idx) % self.n_samples
        X = self.sources_cpu[idx].to(device)
        W = self.plans_cpu[idx].to(device)
        Y = self._y_fixed_cpu.to(device)
        U = W @ Y  # barycentric map

        return {
            "X": X.detach().cpu().numpy(),
            "Y_fixed": Y.detach().cpu().numpy(),
            "W": W.detach().cpu().numpy(),
            "U_bary": U.detach().cpu().numpy(),
            "meta": self.meta_cpu[idx],
        }


def load_transport_plan_dataset(
    data_path: str = "Data/transport_data/transport_plan_dataset",
    batch_size: int = 32,
    device: str = "cuda",
    mode: str = "plan",
    preload_to_gpu: bool = False,
):
    """
    Load HuggingFace dataset from disk and wrap it.

    Returns:
        dataset: raw HF dataset dict
        wrapper: TransportPlanDataset
    """
    print(f"Loading transport-plan dataset from: {data_path}")

    p = Path(data_path)
    if not p.exists():
        print(f"Dataset directory not found: {p}")
        print("Run: python Data/transport_data/generate_transport_plan_data.py")
        return None, None

    try:
        dataset = load_from_disk(str(p))
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")

        wrapper = TransportPlanDataset(
            dataset,
            batch_size=batch_size,
            device=device,
            mode=mode,
            preload_to_gpu=preload_to_gpu,
        )
        return dataset, wrapper
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the dataset directory is valid and was generated correctly.")
        return None, None
