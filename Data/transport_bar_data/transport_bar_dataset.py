#!/usr/bin/env python
"""
transport_bar_dataset.py
----------------------------------
Dataset wrapper + loader for the OT "transport plan / barycentric map" dataset.

This is designed for operator learning with point-cloud inputs:

- Input set (branch): source point cloud X with uniform values (ones or 1/M).
- Query locations (trunk): source point cloud X (evaluate map where mass exists).
- Supervision: either displacement D = u(x)-x or map values u(x).

Also exposes the fixed target cloud Y_fixed (same for all samples) so training can add
a pushforward / distribution-matching loss if desired.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from datasets import load_from_disk


Mode = Literal["displacement", "map"]          # what supervision to return
InputValues = Literal["ones", "mass"]          # what "u" values to feed as branch values


class TransportPlanDataset:
    """
    Wrapper that preloads the train split into torch tensors for fast random batching.
    """

    def __init__(
        self,
        dataset,
        *,
        data_path: Optional[str] = None,
        split: str = "train",
        batch_size: int = 64,
        device: str = "cuda",
        mode: Mode = "displacement",
        input_values: InputValues = "ones",
        preload_to_device: bool = True,
    ):
        self.batch_size = int(batch_size)
        self.device = device
        self.mode = mode
        self.input_values = input_values
        self.preload_to_device = bool(preload_to_device)

        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in dataset. Available: {list(dataset.keys())}")

        self.data = dataset[split]
        self.n_samples = len(self.data)

        # Load fixed target cloud (Y_fixed) saved alongside dataset.
        y_fixed = None
        meta = None
        if data_path is not None:
            p = Path(data_path)
            y_path = p / "y_fixed.npy"
            meta_path = p / "meta.json"
            if y_path.exists():
                y_fixed = np.load(y_path).astype(np.float32)
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

        # Fallback: if y_fixed wasn't found, try to infer from dataset (not stored by default).
        if y_fixed is None:
            raise FileNotFoundError(
                "Could not find 'y_fixed.npy'.\n"
                "Expected it next to the saved dataset directory.\n"
                "Re-generate the dataset or pass the correct data_path."
            )

        self.meta = meta or {}
        self.n_target_points = int(y_fixed.shape[0])
        self.y_fixed = torch.tensor(y_fixed, dtype=torch.float32)
        if self.preload_to_device:
            self.y_fixed = self.y_fixed.to(self.device)

        # Preload all samples
        self.sources = []
        self.maps = []
        self.displacements = []

        # Load first sample to infer M
        s0 = self.data[0]
        x0 = np.array(s0["source_points"], dtype=np.float32)
        self.n_source_points = int(x0.shape[0])

        # A single template for branch "values" (uniform weights or ones)
        if self.input_values == "mass":
            val = 1.0 / float(self.n_source_points)
        elif self.input_values == "ones":
            val = 1.0
        else:
            raise ValueError(f"Unknown input_values: {self.input_values}")

        self._branch_values_template = torch.full(
            (self.n_source_points, 1), fill_value=val, dtype=torch.float32
        )
        if self.preload_to_device:
            self._branch_values_template = self._branch_values_template.to(self.device)

        # Actually load samples
        for i in range(self.n_samples):
            sample = self.data[i]
            X = torch.tensor(np.array(sample["source_points"], dtype=np.float32))
            U = torch.tensor(np.array(sample["transported_points"], dtype=np.float32))
            D = torch.tensor(np.array(sample["displacement"], dtype=np.float32))

            if self.preload_to_device:
                X = X.to(self.device)
                U = U.to(self.device)
                D = D.to(self.device)

            self.sources.append(X)
            self.maps.append(U)
            self.displacements.append(D)

        print("Loaded OT TransportPlanDataset")
        print(f"  - split: {split}")
        print(f"  - n_samples: {self.n_samples}")
        print(f"  - n_source_points (M): {self.n_source_points}")
        print(f"  - n_target_points (N): {self.n_target_points}")
        print(f"  - mode: {self.mode}")
        print(f"  - input_values: {self.input_values}")
        if meta:
            print(f"  - epsilon: {meta.get('epsilon', 'unknown')}")
            print(f"  - max_iterations: {meta.get('max_iterations', 'unknown')}")
            print(f"  - threshold: {meta.get('threshold', 'unknown')}")

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch.

        Returns:
            xs: (B, M, 2) source points
            us: (B, M, 1) branch values (ones or mass weights)
            ys: (B, M, 2) query locations (source points)
            G_u_ys: (B, M, 2) supervision (displacement or transported points)
            y_fixed: (B, N, 2) fixed target cloud (repeated for convenience)
        """
        # Choose indices on CPU to avoid tiny GPU kernel launches in randint for some setups.
        idx = torch.randint(0, self.n_samples, (self.batch_size,), device="cpu").tolist()

        xs = torch.stack([self.sources[i] for i in idx], dim=0)  # (B,M,2)
        ys = xs  # query only where mass exists

        us = self._branch_values_template.unsqueeze(0).repeat(self.batch_size, 1, 1)  # (B,M,1)

        if self.mode == "displacement":
            G = torch.stack([self.displacements[i] for i in idx], dim=0)  # (B,M,2)
        elif self.mode == "map":
            G = torch.stack([self.maps[i] for i in idx], dim=0)  # (B,M,2)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Repeat fixed target cloud across batch for convenience in loss code.
        y_fixed = self.y_fixed
        if y_fixed.ndim != 2 or y_fixed.shape[1] != 2:
            raise RuntimeError(f"y_fixed has unexpected shape: {tuple(y_fixed.shape)}")
        y_fixed_batched = y_fixed.unsqueeze(0).repeat(self.batch_size, 1, 1)  # (B,N,2)

        # If we're not preloading to device, move here
        if not self.preload_to_device:
            xs = xs.to(self.device)
            ys = ys.to(self.device)
            us = us.to(self.device)
            G = G.to(self.device)
            y_fixed_batched = y_fixed_batched.to(self.device)

        return xs, us, ys, G, y_fixed_batched

    def get_sample_for_visualization(self, idx: int = 0):
        """
        Return a single sample as CPU numpy arrays for plotting.
        """
        if idx < 0 or idx >= self.n_samples:
            idx = 0

        X = self.sources[idx].detach().cpu().numpy()
        U = self.maps[idx].detach().cpu().numpy()
        D = self.displacements[idx].detach().cpu().numpy()
        Y = self.y_fixed.detach().cpu().numpy()

        return {
            "X": X,        # source points
            "U": U,        # transported points (barycentric)
            "D": D,        # displacement vectors
            "Y_fixed": Y,  # fixed target cloud
        }


def load_transport_plan_dataset(
    data_path: str = "Data/transport_data/transport_plan_dataset",
    *,
    batch_size: int = 64,
    device: str = "cuda",
    mode: Mode = "displacement",
    input_values: InputValues = "ones",
    split: str = "train",
    preload_to_device: bool = True,
):
    """
    Load dataset from disk and return (hf_dataset, wrapper).
    """
    p = Path(data_path)
    if not p.exists():
        print(f"Dataset directory not found: {data_path}")
        print("Run: python Data/transport_data/generate_transport_plan_data.py")
        return None, None

    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

        wrapper = TransportPlanDataset(
            dataset,
            data_path=data_path,
            split=split,
            batch_size=batch_size,
            device=device,
            mode=mode,
            input_values=input_values,
            preload_to_device=preload_to_device,
        )
        return dataset, wrapper
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None
