"""diffraction_2d_dataset.py
---------------------------
Dataset wrapper and loader for 2D Phase-Screen Diffraction data.

- Loads HuggingFace dataset from disk
- Stores all samples in memory on the chosen device
- Provides .sample() that returns padded tensors and masks:

    xs_padded, us_padded, ys, G_u_ys, object_mask

Where:
- xs: bump coordinates (n_bumps, 2) = [x, y]
- us: bump features   (n_bumps, 2) = [alpha, ell]
- ys: query coords (shared global grid coords, grid_n^2 x 2)
- G_u_ys: target complex values at ys (grid_n^2 x 2)
- object_mask: boolean mask for bumps (like sensor_mask)
"""

from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk


class DiffractionDataset:
    """Dataset wrapper for 2D diffraction data."""

    def __init__(self, dataset, batch_size: int = 64, device: str = "cuda"):
        print("Loading Diffraction 2D dataset...")

        self.batch_size = int(batch_size)
        self.device = device

        train_data = dataset["train"]
        self.n_samples = len(train_data)

        # Detect adaptive mesh dataset (no longer supported)
        sample_0 = train_data[0]
        if "grid_coords" in sample_0 or "field_values" in sample_0:
            raise ValueError("Adaptive mesh diffraction datasets are no longer supported. Regenerate uniform-grid data.")
        self.is_adaptive = False

        # Store bump data (variable-length per sample)
        self.bumps_data = []

        field0 = np.array(sample_0["field"], dtype=np.float32)  # (grid_n,grid_n,2)
        self.grid_size = int(field0.shape[0])
        self.n_grid_points = int(self.grid_size * self.grid_size)
        print(f"Uniform grid: {self.grid_size}x{self.grid_size}, Total grid points: {self.n_grid_points}")

        # Create coordinate grid (same for all samples), on [0,1) endpoint=False to match generator
        x = np.linspace(0.0, 1.0, self.grid_size, endpoint=False, dtype=np.float32)
        y = x
        xx, yy = np.meshgrid(x, y, indexing="ij")
        coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)  # (grid_n^2,2)
        self.grid_coords = torch.tensor(coords, device=device, dtype=torch.float32)

        # Store all fields (flattened) in a single tensor for efficiency
        self.fields = torch.zeros(self.n_samples, self.n_grid_points, 2, device=device, dtype=torch.float32)

        for i in range(self.n_samples):
            sample = train_data[i]

            bumps = np.array(sample["bumps"], dtype=np.float32)  # (n_bumps,4)
            self.bumps_data.append(torch.tensor(bumps, device=device, dtype=torch.float32))

            field = np.array(sample["field"], dtype=np.float32)  # (grid_n,grid_n,2)
            self.fields[i] = torch.tensor(field.reshape(-1, 2), device=device, dtype=torch.float32)

        print(f"Dataset loaded: {self.n_samples} samples")

    def sample(self, device=None):
        """Sample a batch for training.

        Returns:
            xs_padded: (B, max_bumps, 2)
            us_padded: (B, max_bumps, 2) where features=[alpha, ell]
            ys:        (B, n_grid_points, 2)
            G_u_ys:    (B, n_grid_points, 2)
            object_mask: (B, max_bumps) bool
        """
        if device is None:
            device = self.device

        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)

        xs_batch = []
        us_batch = []
        for idx in indices:
            bumps = self.bumps_data[int(idx.item())]  # (n_bumps,4)
            bump_coords = bumps[:, 0:2]  # (n_bumps,2)
            bump_feats = bumps[:, 2:4]   # (n_bumps,2) = [alpha, ell]

            xs_batch.append(bump_coords)
            us_batch.append(bump_feats)

        # Pad bumps to max length in batch
        max_bumps = max(x.shape[0] for x in xs_batch)
        xs_padded = torch.zeros(self.batch_size, max_bumps, 2, device=device, dtype=torch.float32)
        us_padded = torch.zeros(self.batch_size, max_bumps, 2, device=device, dtype=torch.float32)
        object_mask = torch.zeros(self.batch_size, max_bumps, device=device, dtype=torch.bool)

        for i, (xs, us) in enumerate(zip(xs_batch, us_batch)):
            n_bumps = xs.shape[0]
            xs_padded[i, :n_bumps] = xs.to(device=device)
            us_padded[i, :n_bumps] = us.to(device=device)
            object_mask[i, :n_bumps] = True

        # Uniform grid: all samples share same target coords/size
        ys = self.grid_coords.unsqueeze(0).expand(self.batch_size, -1, -1).to(device=device)
        G_u_ys = self.fields[indices].to(device=device)

        return xs_padded, us_padded, ys, G_u_ys, object_mask


def load_diffraction_dataset(
    data_path: str = "Data/diffraction_data/phase_screen_dataset",
    batch_size: int = 64,
    device: str = "cuda",
):
    """Load diffraction dataset and return (raw_hf_dataset, DiffractionDataset wrapper)."""
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")

        diffraction_dataset = DiffractionDataset(dataset, batch_size=batch_size, device=device)
        return dataset, diffraction_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/diffraction_data/generate_diffraction_2d_data.py")
        return None, None
