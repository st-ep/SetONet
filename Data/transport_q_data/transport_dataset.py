#!/usr/bin/env python
"""transport_dataset.py
----------------------------------
Dataset wrapper and loader for OT data with decoupled query points (Strategy 1).

Key change vs the original transport_dataset.py:
- If query_points/query_vectors are present, mode='transport_map' returns:
    xs = source_points (sensors)
    us = uniform weights
    ys = query_points (independent trunk queries)
    G_u_ys = query_vectors (T(y)-y displacement at those queries)
- mode='transport_map_at_source' keeps the legacy behavior (ys = xs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk


def _compute_grid_coords(sample):
    """Compute grid coordinates from sample metadata."""
    if "grid_coords" in sample:
        return np.array(sample["grid_coords"], dtype=np.float32)

    domain_size = float(sample.get("domain_size", 5.0))
    velocity_field = np.array(sample["velocity_field"])
    grid_h, grid_w = velocity_field.shape[0], velocity_field.shape[1]
    xs = np.linspace(-domain_size, domain_size, grid_h, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_w, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)


class TransportDataset:
    """Dataset wrapper for OT data with decoupled query points."""

    def __init__(self, dataset, batch_size=64, device="cuda", mode="transport_map"):
        """
        Initialize the transport dataset wrapper.

        Args:
            dataset: HuggingFace dataset dict with 'train' and 'test' splits
            batch_size: Batch size for sampling
            device: Torch device for tensors
            mode: How to structure data for operator learning
                - 'velocity_field': Learn velocity field on grid from source points
                - 'transport_map': Learn displacement at *decoupled query points* (Strategy 1)
                - 'transport_map_at_source': Learn displacement at source points (legacy: ys=xs)
                - 'density_transport': Legacy placeholder
        """
        print("Loading Optimal Transport-Q dataset...")

        self.batch_size = batch_size
        self.device = device
        self.mode = mode

        train_data = dataset["train"]
        self._hf_train = train_data  # keep reference for lazy access
        self.n_samples = len(train_data)

        sample_0 = train_data[0]

        # Grid info (for velocity_field mode / visualization / evaluation)
        velocity_field_0 = np.array(sample_0["velocity_field"])
        grid_coords = _compute_grid_coords(sample_0)
        self.grid_h, self.grid_w = velocity_field_0.shape[0], velocity_field_0.shape[1]
        self.n_grid_points = self.grid_h * self.grid_w
        self.grid_pts = torch.tensor(grid_coords, device=device, dtype=torch.float32)

        # Storage for source/target data
        self.sources_data = []
        self.targets_data = []
        self.transport_vectors = []

        # Query storage (Strategy 1 - decoupled queries)
        self.has_queries = ("query_points" in sample_0) and ("query_vectors" in sample_0)
        self.query_points_data = []
        self.query_vectors_data = []

        # Only load full velocity fields if needed by current mode
        self.load_velocity_fields = (mode == "velocity_field")
        self.velocity_fields = []

        # Load all training data to GPU
        for i in range(self.n_samples):
            sample = train_data[i]

            source_points = np.array(sample["source_points"], dtype=np.float32)
            target_points = np.array(sample["target_points"], dtype=np.float32)

            src_t = torch.tensor(source_points, device=device, dtype=torch.float32)
            tgt_t = torch.tensor(target_points, device=device, dtype=torch.float32)

            self.sources_data.append(src_t)
            self.targets_data.append(tgt_t)
            self.transport_vectors.append(tgt_t - src_t)

            if self.has_queries:
                qp = torch.tensor(
                    np.array(sample["query_points"], dtype=np.float32),
                    device=device,
                    dtype=torch.float32,
                )
                qv = torch.tensor(
                    np.array(sample["query_vectors"], dtype=np.float32),
                    device=device,
                    dtype=torch.float32,
                )
                self.query_points_data.append(qp)
                self.query_vectors_data.append(qv)

            if self.load_velocity_fields:
                vf = torch.tensor(
                    np.array(sample["velocity_field"], dtype=np.float32),
                    device=device,
                    dtype=torch.float32,
                )
                self.velocity_fields.append(vf)

        # Record dimensions
        self.n_source_points = int(self.sources_data[0].shape[0])
        self.n_target_points = int(self.targets_data[0].shape[0])
        self.n_query_points = int(self.query_points_data[0].shape[0]) if self.has_queries else 0

        # Print dataset info
        print("Dataset loaded successfully:")
        print(f"  - Number of samples: {self.n_samples}")
        print(f"  - Source points per sample: {self.n_source_points}")
        print(f"  - Target points per sample: {self.n_target_points}")
        print(f"  - Grid: {self.grid_h} x {self.grid_w}")
        print(f"  - Has decoupled queries: {self.has_queries}")
        if self.has_queries:
            print(f"  - Query points per sample: {self.n_query_points}")
        print(f"  - Mode: {mode}")
        if not self.load_velocity_fields:
            print("  - Note: velocity_field tensors are NOT preloaded (mode != 'velocity_field')")

    def sample(self, device=None):
        """
        Sample a batch for training.

        Returns:
            xs: (B, n_sensors, 2) source point coordinates
            us: (B, n_sensors, 1) uniform weights
            ys: (B, n_queries, 2) query point coordinates
            G_u_ys: (B, n_queries, 2) ground truth displacement at query points
            mask: None (no masking for this dataset)
        """
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)

        if self.mode == "velocity_field":
            return self._sample_velocity_field(indices)
        elif self.mode == "transport_map":
            return self._sample_transport_map(indices, prefer_queries=True)
        elif self.mode == "transport_map_at_source":
            return self._sample_transport_map(indices, prefer_queries=False)
        elif self.mode == "density_transport":
            return self._sample_density_transport(indices)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _sample_velocity_field(self, indices):
        """
        Sample for velocity field learning on grid.
        Input: source points with uniform weights
        Output: velocity field at grid points
        """
        if not self.load_velocity_fields or len(self.velocity_fields) == 0:
            raise RuntimeError("velocity_field mode requested, but velocity fields were not preloaded.")

        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []

        for idx in indices:
            source_points = self.sources_data[idx.item()]  # (n_source_points, 2)
            source_weights = torch.ones(
                self.n_source_points, 1, device=self.device, dtype=torch.float32
            )

            target_locations = self.grid_pts  # (n_grid_points, 2)
            velocity_field = self.velocity_fields[idx.item()].reshape(-1, 2)  # (n_grid_points, 2)

            xs_batch.append(source_points)
            us_batch.append(source_weights)
            ys_batch.append(target_locations)
            G_u_ys_batch.append(velocity_field)

        xs = torch.stack(xs_batch, dim=0)
        us = torch.stack(us_batch, dim=0)
        ys = torch.stack(ys_batch, dim=0)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)
        return xs, us, ys, G_u_ys, None

    def _sample_transport_map(self, indices, prefer_queries: bool = True):
        """
        Sample for transport map (displacement field) learning.

        If prefer_queries=True and query_points/query_vectors exist:
          - ys = query_points (decoupled from sensors)
          - G_u_ys = query_vectors (= T(ys) - ys)

        Else (legacy behavior):
          - ys = source_points (same as sensors)
          - G_u_ys = targets - sources
        """
        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []

        use_queries = bool(prefer_queries and self.has_queries and len(self.query_points_data) > 0)

        for idx in indices:
            source_points = self.sources_data[idx.item()]  # (n_source_points, 2)
            source_weights = torch.ones(
                self.n_source_points, 1, device=self.device, dtype=torch.float32
            )

            if use_queries:
                target_locations = self.query_points_data[idx.item()]  # (n_queries, 2)
                transport_vectors = self.query_vectors_data[idx.item()]  # (n_queries, 2)
            else:
                target_locations = source_points  # (n_source_points, 2)
                transport_vectors = self.transport_vectors[idx.item()]  # (n_source_points, 2)

            xs_batch.append(source_points)
            us_batch.append(source_weights)
            ys_batch.append(target_locations)
            G_u_ys_batch.append(transport_vectors)

        xs = torch.stack(xs_batch, dim=0)
        us = torch.stack(us_batch, dim=0)
        ys = torch.stack(ys_batch, dim=0)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)
        return xs, us, ys, G_u_ys, None

    def _sample_density_transport(self, indices):
        """
        Legacy placeholder: returns uniform density at target points.
        """
        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []

        for idx in indices:
            source_points = self.sources_data[idx.item()]
            source_weights = torch.ones(
                self.n_source_points, 1, device=self.device, dtype=torch.float32
            )

            target_points = self.targets_data[idx.item()]
            target_density = torch.ones(
                target_points.shape[0], 1, device=self.device, dtype=torch.float32
            )

            xs_batch.append(source_points)
            us_batch.append(source_weights)
            ys_batch.append(target_points)
            G_u_ys_batch.append(target_density)

        xs = torch.stack(xs_batch, dim=0)
        us = torch.stack(us_batch, dim=0)
        ys = torch.stack(ys_batch, dim=0)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)
        return xs, us, ys, G_u_ys, None

    def get_sample_for_visualization(self, idx=0):
        """Get a single sample for visualization purposes."""
        if idx >= self.n_samples:
            idx = 0

        # If velocity fields not preloaded, load from HF dataset on demand
        if self.load_velocity_fields and len(self.velocity_fields) > 0:
            V = self.velocity_fields[idx].detach().cpu().numpy()
        else:
            V = np.array(self._hf_train[idx]["velocity_field"], dtype=np.float32)

        out = {
            "X": self.sources_data[idx].detach().cpu().numpy(),
            "Y": self.targets_data[idx].detach().cpu().numpy(),
            "V": V,
            "transport_vectors": self.transport_vectors[idx].detach().cpu().numpy(),
            "grid_pts": self.grid_pts.detach().cpu().numpy(),
        }

        if self.has_queries:
            out["Q"] = self.query_points_data[idx].detach().cpu().numpy()
            out["Q_vectors"] = self.query_vectors_data[idx].detach().cpu().numpy()

        return out


def load_transport_dataset(
    data_path="Data/transport_q_data/transport_dataset",
    batch_size=64,
    device="cuda",
    mode="transport_map",
):
    """
    Load the Transport-Q dataset and return (hf_dataset, wrapper).

    Args:
        data_path: Path to saved HuggingFace dataset directory
        batch_size: Training batch size
        device: Torch device
        mode: Data mode
            - 'transport_map': Learn displacement at decoupled query points (Strategy 1)
            - 'transport_map_at_source': Legacy behavior (ys = xs)
            - 'velocity_field': Learn velocity field on grid
            - 'density_transport': Legacy placeholder

    Returns:
        dataset: HuggingFace dataset dict with 'train' and 'test' splits
        transport_dataset: TransportDataset wrapper for training
    """
    print(f"Loading dataset from: {data_path}")

    if not Path(data_path).exists():
        print(f"Dataset directory not found: {data_path}")
        print("Run: python Data/transport_q_data/generate_transport_q_data.py")
        return None, None

    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")

        transport_dataset = TransportDataset(
            dataset, batch_size=batch_size, device=device, mode=mode
        )
        return dataset, transport_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the dataset directory is valid and was generated correctly.")
        return None, None
