#!/usr/bin/env python
"""dynamic_chladni_dataset.py
----------------------------------
Dataset wrapper and loader for Dynamic Chladni plate data with point cloud forces.
"""
from __future__ import annotations

import numpy as np
import torch
from datasets import load_from_disk

class DynamicChladniDataset:
    """Dataset wrapper for Dynamic Chladni plate data with point cloud forces."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("Loading Dynamic Chladni dataset...")
        
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get grid size from first sample
        sample_0 = train_data[0]
        displacement_field = np.array(sample_0['field'])  # shape: (grid_n, grid_n, 1)
        self.grid_size = displacement_field.shape[0]
        self.n_grid_points = self.grid_size * self.grid_size
        
        print(f"Grid resolution: {self.grid_size}×{self.grid_size}, Total grid points: {self.n_grid_points}")
        
        # Create coordinate grid (same for all samples, normalized [0,1])
        x = np.linspace(0.0, 1.0, self.grid_size)
        y = np.linspace(0.0, 1.0, self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
        self.grid_coords = torch.tensor(coords, device=device, dtype=torch.float32)
        
        # Store source data and displacement fields
        self.sources_data = []
        self.displacement_fields = torch.zeros(self.n_samples, self.n_grid_points, device=device, dtype=torch.float32)
        
        # Load data
        for i in range(self.n_samples):
            sample = train_data[i]
            
            # Sources: list of [x_norm, y_norm, force_magnitude] triplets
            sources = np.array(sample['sources'])  # shape: (n_forces, 3)
            self.sources_data.append(torch.tensor(sources, device=device, dtype=torch.float32))
            
            # Displacement field: (grid_n, grid_n, 1) -> flatten to (grid_n*grid_n,)
            displacement_field = np.array(sample['field'])
            self.displacement_fields[i] = torch.tensor(displacement_field[:, :, 0].flatten(), device=device, dtype=torch.float32)
        
        print(f"Dataset loaded: {self.n_samples} samples")
        
        # Analyze force statistics
        all_force_counts = [len(sources) for sources in self.sources_data]
        all_force_mags = []
        for sources in self.sources_data:
            if len(sources) > 0:
                all_force_mags.extend(sources[:, 2].cpu().numpy().tolist())
        
        print(f"Force statistics:")
        print(f"  - Force count range: {min(all_force_counts)} to {max(all_force_counts)}")
        if all_force_mags:
            print(f"  - Force magnitude range: {min(all_force_mags):.4f} to {max(all_force_mags):.4f}")
    
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
            sources = self.sources_data[idx.item()]  # (n_forces, 3)
            
            if len(sources) > 0:
                source_coords = sources[:, :2]  # (n_forces, 2) - x, y coordinates (normalized)
                source_forces = sources[:, 2:3]  # (n_forces, 1) - force magnitude values
            else:
                # Handle case with no forces (shouldn't happen, but good to be safe)
                source_coords = torch.zeros((1, 2), device=self.device, dtype=torch.float32)
                source_forces = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
            
            # Target coordinates (same for all samples - uniform grid)
            target_coords = self.grid_coords  # (n_grid_points, 2)
            target_displacements = self.displacement_fields[idx].unsqueeze(-1)  # (n_grid_points, 1)
            
            xs_batch.append(source_coords)
            us_batch.append(source_forces)
            ys_batch.append(target_coords)
            G_u_ys_batch.append(target_displacements)
        
        # Convert to tensors - handle variable length sources by padding
        max_forces = max(x.shape[0] for x in xs_batch)
        
        xs_padded = torch.zeros(self.batch_size, max_forces, 2, device=self.device)
        us_padded = torch.zeros(self.batch_size, max_forces, 1, device=self.device)
        
        for i, (xs, us) in enumerate(zip(xs_batch, us_batch)):
            n_forces = xs.shape[0]
            xs_padded[i, :n_forces] = xs
            us_padded[i, :n_forces] = us
        
        # For uniform grid, all samples have same number of target points
        ys = torch.stack(ys_batch, dim=0)  # (batch_size, n_grid_points, 2)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)  # (batch_size, n_grid_points, 1)
        
        return xs_padded, us_padded, ys, G_u_ys, None
    
    def get_sample(self, idx):
        """Get a specific sample for visualization or analysis."""
        if idx >= self.n_samples:
            raise IndexError(f"Sample index {idx} out of range (max: {self.n_samples-1})")
        
        sources = self.sources_data[idx].cpu().numpy()
        displacement_field = self.displacement_fields[idx].cpu().numpy().reshape(self.grid_size, self.grid_size)
        grid_coords = self.grid_coords.cpu().numpy().reshape(self.grid_size, self.grid_size, 2)
        
        return {
            'sources': sources,  # (n_forces, 3) - [x_norm, y_norm, force_mag]
            'displacement_field': displacement_field,  # (grid_size, grid_size)
            'grid_coords': grid_coords,  # (grid_size, grid_size, 2)
        }
    
    def get_statistics(self):
        """Get dataset statistics."""
        force_counts = [len(sources) for sources in self.sources_data]
        all_force_mags = []
        all_displacements = []
        
        for i in range(self.n_samples):
            sources = self.sources_data[i]
            if len(sources) > 0:
                all_force_mags.extend(sources[:, 2].cpu().numpy().tolist())
            all_displacements.extend(self.displacement_fields[i].cpu().numpy().tolist())
        
        stats = {
            'n_samples': self.n_samples,
            'grid_size': self.grid_size,
            'force_count_range': (min(force_counts), max(force_counts)),
            'force_count_mean': np.mean(force_counts),
            'displacement_range': (min(all_displacements), max(all_displacements)),
            'displacement_mean': np.mean(all_displacements),
            'displacement_std': np.std(all_displacements),
        }
        
        if all_force_mags:
            stats.update({
                'force_magnitude_range': (min(all_force_mags), max(all_force_mags)),
                'force_magnitude_mean': np.mean(all_force_mags),
                'force_magnitude_std': np.std(all_force_mags),
            })
        
        return stats

def load_dynamic_chladni_dataset(data_path="Data/dynamic_chladni/dynamic_chladni_dataset", batch_size=64, device='cuda'):
    """Load dynamic chladni dataset and return dataset wrapper."""
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        # Create dataset wrapper
        chladni_dataset = DynamicChladniDataset(dataset, batch_size=batch_size, device=device)
        
        return dataset, chladni_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/dynamic_chladni/dynamic_chladni_generator.py")
        return None, None 