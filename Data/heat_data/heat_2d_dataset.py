#!/usr/bin/env python
"""heat_2d_dataset.py
----------------------------------
Dataset wrapper and loader for 2D heat data.
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk

class HeatDataset:
    """Dataset wrapper for 2D heat data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("Loading Heat 2D dataset...")
        
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        temp_field = np.array(sample_0['field'])  # temperature field (grid_n, grid_n, 1)
        
        self.grid_size = temp_field.shape[0]  # Should be 64 for 64x64 grid
        self.n_grid_points = self.grid_size * self.grid_size  # Total number of grid points
        
        print(f"Grid size: {self.grid_size}x{self.grid_size}, Total grid points: {self.n_grid_points}")
        
        # Create coordinate grid (same for all samples)
        x = np.linspace(0.0, 1.0, self.grid_size)
        y = np.linspace(0.0, 1.0, self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
        self.grid_coords = torch.tensor(coords, device=device, dtype=torch.float32)
        
        # Store all data for efficient sampling
        self.sources_data = []
        self.temp_fields = torch.zeros(self.n_samples, self.n_grid_points, device=device, dtype=torch.float32)
        
        # Load data
        for i in range(self.n_samples):
            sample = train_data[i]
            
            # Sources: list of [x, y, power] triplets
            sources = np.array(sample['sources'])  # shape: (n_sources, 3)
            self.sources_data.append(torch.tensor(sources, device=device, dtype=torch.float32))
            
            # Temperature field: (grid_n, grid_n, 1) -> flatten to (grid_n*grid_n,)
            temp_field = np.array(sample['field'])
            self.temp_fields[i] = torch.tensor(temp_field[:, :, 0].flatten(), device=device, dtype=torch.float32)
        
        print(f"Dataset loaded: {self.n_samples} samples")
        
    def sample(self, device=None):
        """Sample a batch for training."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        # Prepare batch data
        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []
        
        for i, idx in enumerate(indices):
            # Source data for this sample
            sources = self.sources_data[idx.item()]  # (n_sources, 3)
            source_coords = sources[:, :2]  # (n_sources, 2) - x, y coordinates
            source_powers = sources[:, 2:3]  # (n_sources, 1) - power values
            
            # Target data (grid points and temperature field)
            target_coords = self.grid_coords  # (n_grid_points, 2)
            target_temps = self.temp_fields[idx].unsqueeze(-1)  # (n_grid_points, 1)
            
            xs_batch.append(source_coords)
            us_batch.append(source_powers)
            ys_batch.append(target_coords)
            G_u_ys_batch.append(target_temps)
        
        # Convert to tensors - note: xs and us have variable length per sample
        # For now, we'll handle this by padding to max length in the batch
        max_sources = max(x.shape[0] for x in xs_batch)
        
        xs_padded = torch.zeros(self.batch_size, max_sources, 2, device=self.device)
        us_padded = torch.zeros(self.batch_size, max_sources, 1, device=self.device)
        
        for i, (xs, us) in enumerate(zip(xs_batch, us_batch)):
            n_sources = xs.shape[0]
            xs_padded[i, :n_sources] = xs
            us_padded[i, :n_sources] = us
        
        # ys and G_u_ys are same size for all samples
        ys = torch.stack(ys_batch, dim=0)  # (batch_size, n_grid_points, 2)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)  # (batch_size, n_grid_points, 1)
        
        return xs_padded, us_padded, ys, G_u_ys, None

def load_heat_dataset(data_path="Data/heat_data/pcb_heat_dataset", batch_size=64, device='cuda'):
    """Load heat dataset and return dataset wrapper."""
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        # Create dataset wrapper
        heat_dataset = HeatDataset(dataset, batch_size=batch_size, device=device)
        
        return dataset, heat_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/heat_data/generate_heat_2d_data.py")
        return None, None 