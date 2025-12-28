#!/usr/bin/env python
"""transport_dataset.py
----------------------------------
Dataset wrapper and loader for optimal transport data.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from datasets import load_from_disk

def _compute_grid_coords(sample):
    if "grid_coords" in sample:
        return np.array(sample["grid_coords"], dtype=np.float32)

    domain_size = float(sample.get("domain_size", 5.0))
    velocity_field = np.array(sample["velocity_field"])
    grid_h, grid_w = velocity_field.shape[0], velocity_field.shape[1]
    xs = np.linspace(-domain_size, domain_size, grid_h, dtype=np.float32)
    ys = np.linspace(-domain_size, domain_size, grid_w, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

class TransportDataset:
    """Dataset wrapper for optimal transport data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda', mode='velocity_field'):
        """
        Initialize transport dataset.
        
        Args:
            dataset: HuggingFace dataset containing transport data
            batch_size: Batch size for sampling
            device: Device to load tensors on
            mode: How to structure the data for operator learning
                - 'velocity_field': Learn velocity field from source points
                - 'transport_map': Learn transport map from source to target points
                - 'density_transport': Learn target density from source points
        """
        print("Loading Optimal Transport dataset...")
        
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        
        # Load HuggingFace dataset
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get sample to determine dimensions
        sample_0 = train_data[0]
        
        # Store all data for efficient sampling
        self.sources_data = []
        self.targets_data = []
        self.velocity_fields = []
        
        # Get grid information from first sample
        velocity_field = np.array(sample_0['velocity_field'])
        grid_coords = _compute_grid_coords(sample_0)
        
        self.grid_h, self.grid_w = velocity_field.shape[0], velocity_field.shape[1]
        self.n_grid_points = self.grid_h * self.grid_w
        self.grid_pts = torch.tensor(grid_coords, device=device, dtype=torch.float32)
        
        # Load all data
        for i in range(self.n_samples):
            sample = train_data[i]
            
            # Source and target points
            source_points = np.array(sample['source_points'])
            target_points = np.array(sample['target_points'])
            velocity_field = np.array(sample['velocity_field'])
            
            self.sources_data.append(torch.tensor(source_points, device=device, dtype=torch.float32))
            self.targets_data.append(torch.tensor(target_points, device=device, dtype=torch.float32))
            self.velocity_fields.append(torch.tensor(velocity_field, device=device, dtype=torch.float32))
        
        # Get typical number of source/target points
        self.n_source_points = len(self.sources_data[0])
        self.n_target_points = len(self.targets_data[0])
        
        print(f"Dataset loaded successfully:")
        print(f"  - Number of samples: {self.n_samples}")
        print(f"  - Source points per sample: {self.n_source_points}")
        print(f"  - Target points per sample: {self.n_target_points}")
        print(f"  - Velocity field grid: {self.grid_h}×{self.grid_w}")
        print(f"  - Mode: {mode}")
        
        # Precompute transport vectors (Y - X)
        self.transport_vectors = []
        for i in range(self.n_samples):
            transport_vec = self.targets_data[i] - self.sources_data[i]
            self.transport_vectors.append(transport_vec)
    
    def sample(self, device=None):
        """Sample a batch for training."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        if self.mode == 'velocity_field':
            return self._sample_velocity_field(indices)
        elif self.mode == 'transport_map':
            return self._sample_transport_map(indices)
        elif self.mode == 'density_transport':
            return self._sample_density_transport(indices)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _sample_velocity_field(self, indices):
        """
        Sample for velocity field learning.
        Input: Source points with uniform weights
        Output: Velocity field at grid points
        """
        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []
        
        for idx in indices:
            # Input locations: source points
            source_points = self.sources_data[idx.item()]  # (n_source_points, 2)
            
            # Input functions: uniform weights (could be density estimates)
            source_weights = torch.ones(self.n_source_points, 1, device=self.device, dtype=torch.float32)
            
            # Target locations: grid points
            target_locations = self.grid_pts  # (n_grid_points, 2)
            
            # Target values: velocity field at grid points
            velocity_field = self.velocity_fields[idx.item()].reshape(-1, 2)  # (n_grid_points, 2)
            
            xs_batch.append(source_points)
            us_batch.append(source_weights)
            ys_batch.append(target_locations)
            G_u_ys_batch.append(velocity_field)
        
        # Stack tensors
        xs = torch.stack(xs_batch, dim=0)  # (batch_size, n_source_points, 2)
        us = torch.stack(us_batch, dim=0)  # (batch_size, n_source_points, 1)
        ys = torch.stack(ys_batch, dim=0)  # (batch_size, n_grid_points, 2)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)  # (batch_size, n_grid_points, 2)
        
        return xs, us, ys, G_u_ys, None
    
    def _sample_transport_map(self, indices):
        """
        Sample for transport map learning.
        Input: Source points with uniform weights
        Output: Transport vectors at source points
        """
        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []
        
        for idx in indices:
            # Input locations: source points
            source_points = self.sources_data[idx.item()]  # (n_source_points, 2)
            
            # Input functions: uniform weights
            source_weights = torch.ones(self.n_source_points, 1, device=self.device, dtype=torch.float32)
            
            # Target locations: same as source points (evaluating transport at source)
            target_locations = source_points  # (n_source_points, 2)
            
            # Target values: transport vectors (Y - X)
            transport_vectors = self.transport_vectors[idx.item()]  # (n_source_points, 2)
            
            xs_batch.append(source_points)
            us_batch.append(source_weights)
            ys_batch.append(target_locations)
            G_u_ys_batch.append(transport_vectors)
        
        # Stack tensors
        xs = torch.stack(xs_batch, dim=0)  # (batch_size, n_source_points, 2)
        us = torch.stack(us_batch, dim=0)  # (batch_size, n_source_points, 1)
        ys = torch.stack(ys_batch, dim=0)  # (batch_size, n_source_points, 2)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)  # (batch_size, n_source_points, 2)
        
        return xs, us, ys, G_u_ys, None
    
    def _sample_density_transport(self, indices):
        """
        Sample for density transport learning.
        Input: Source points with uniform weights
        Output: Target point density (using target points as evaluation locations)
        """
        xs_batch = []
        us_batch = []
        ys_batch = []
        G_u_ys_batch = []
        
        for idx in indices:
            # Input locations: source points
            source_points = self.sources_data[idx.item()]  # (n_source_points, 2)
            
            # Input functions: uniform weights
            source_weights = torch.ones(self.n_source_points, 1, device=self.device, dtype=torch.float32)
            
            # Target locations: target points
            target_points = self.targets_data[idx.item()]  # (n_target_points, 2)
            
            # Target values: uniform density at target points (simplified)
            target_density = torch.ones(self.n_target_points, 1, device=self.device, dtype=torch.float32)
            
            xs_batch.append(source_points)
            us_batch.append(source_weights)
            ys_batch.append(target_points)
            G_u_ys_batch.append(target_density)
        
        # Stack tensors
        xs = torch.stack(xs_batch, dim=0)  # (batch_size, n_source_points, 2)
        us = torch.stack(us_batch, dim=0)  # (batch_size, n_source_points, 1)
        ys = torch.stack(ys_batch, dim=0)  # (batch_size, n_target_points, 2)
        G_u_ys = torch.stack(G_u_ys_batch, dim=0)  # (batch_size, n_target_points, 1)
        
        return xs, us, ys, G_u_ys, None
    
    def get_sample_for_visualization(self, idx=0):
        """Get a single sample for visualization purposes."""
        if idx >= self.n_samples:
            idx = 0
        
        return {
            'X': self.sources_data[idx].cpu().numpy(),  # source points
            'Y': self.targets_data[idx].cpu().numpy(),  # target points  
            'V': self.velocity_fields[idx].cpu().numpy(),  # velocity field
            'transport_vectors': self.transport_vectors[idx].cpu().numpy(),  # Y - X
            'grid_pts': self.grid_pts.cpu().numpy(),  # grid coordinates
        }

def load_transport_dataset(data_path="Data/transport_data/transport_dataset", 
                          batch_size=64, device='cuda', mode='velocity_field'):
    """
    Load transport dataset and return dataset wrapper.
    
    Args:
        data_path: Path to HuggingFace dataset directory containing transport data
        batch_size: Batch size for training
        device: Device to load tensors on  
        mode: How to structure data for operator learning
            - 'velocity_field': Learn velocity field from source points
            - 'transport_map': Learn transport map from source to target points
            - 'density_transport': Learn target density from source points
    """
    print(f"Loading dataset from: {data_path}")
    
    # Check if directory exists
    if not Path(data_path).exists():
        print(f"Dataset directory not found: {data_path}")
        print("Run: python Data/transport_data/generate_transport_data.py")
        return None, None
    
    try:
        # Load HuggingFace dataset
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        # Create dataset wrapper
        transport_dataset = TransportDataset(dataset, batch_size=batch_size, device=device, mode=mode)
        
        return dataset, transport_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the dataset directory is valid and was generated correctly.")
        return None, None 
