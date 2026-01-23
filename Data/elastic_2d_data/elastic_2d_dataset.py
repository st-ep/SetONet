#!/usr/bin/env python
"""elastic_2d_dataset.py
----------------------------------
Dataset wrapper and loader for 2D Elastic plate data.
"""
from __future__ import annotations

import numpy as np
import torch
import json
import os
from datasets import load_from_disk
from Data.data_utils import apply_sensor_dropoff

# Get paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

class ElasticDataset:
    """Dataset wrapper for Elastic plate data that uses pre-normalized data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda', train_sensor_dropoff=0.0, replace_with_nearest=False):
        print("Loading pre-normalized Elastic dataset...")
        
        self.batch_size = batch_size
        self.device = device
        self.train_sensor_dropoff = train_sensor_dropoff
        self.replace_with_nearest = replace_with_nearest
        
        # Print sensor dropout configuration for training
        if self.train_sensor_dropoff > 0.0:
            replacement_mode = "nearest neighbor replacement" if self.replace_with_nearest else "removal"
            print(f"Training dataset configured with sensor drop-off rate: {self.train_sensor_dropoff:.1%} ({replacement_mode})")
        
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        self.n_force_points = len(sample_0['X'])
        self.n_mesh_points = len(sample_0['Y'])
        self.input_dim = len(sample_0['X'][0])
        
        print(f"Dataset structure: {self.n_samples} samples, {self.n_force_points} force points, {self.n_mesh_points} mesh points")
        
        # Pre-allocate tensors on GPU for ALL data
        self.X_data = torch.zeros(self.n_samples, self.n_force_points, self.input_dim, device=device, dtype=torch.float32)
        self.u_data = torch.zeros(self.n_samples, self.n_force_points, device=device, dtype=torch.float32)
        self.Y_data = torch.zeros(self.n_samples, self.n_mesh_points, self.input_dim, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(self.n_samples, self.n_mesh_points, device=device, dtype=torch.float32)
        
        # Load pre-normalized data to GPU
        for i in range(self.n_samples):
            sample = train_data[i]
            self.X_data[i] = torch.tensor(sample['X'], device=device, dtype=torch.float32)
            self.u_data[i] = torch.tensor(sample['u'], device=device, dtype=torch.float32)
            self.Y_data[i] = torch.tensor(sample['Y'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(sample['s'], device=device, dtype=torch.float32)
        
        print(f"Dataset loaded: {self.n_samples} samples")
        print(f"  - Force points: {self.n_force_points}")
        print(f"  - Mesh points: {self.n_mesh_points}")
        
        # Load normalization statistics
        stats_path = os.path.join(current_dir, 'elastic_normalization_stats.json')
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.u_mean = torch.tensor(stats['u_mean'], device=device, dtype=torch.float32)
        self.u_std = torch.tensor(stats['u_std'], device=device, dtype=torch.float32)
        self.s_mean = torch.tensor(stats['s_mean'], device=device, dtype=torch.float32)
        self.s_std = torch.tensor(stats['s_std'], device=device, dtype=torch.float32)
        
    def sample(self, device=None):
        """Sample a batch using pre-normalized GPU tensors."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        xs = self.X_data[indices]
        us = self.u_data[indices].unsqueeze(-1)
        ys = self.Y_data[indices]
        G_u_ys = self.s_data[indices].unsqueeze(-1)
        
        # Apply sensor dropout during training if specified
        if self.train_sensor_dropoff > 0.0:
            xs_dropped_list = []
            us_dropped_list = []
            
            # Apply sensor dropout to each sample in the batch
            for i in range(self.batch_size):
                # Remove batch dimension for dropout function
                xs_single = xs[i]  # Shape: (n_sensors, 2)
                us_single = us[i].squeeze(-1)  # Shape: (n_sensors,)
                
                # Apply sensor dropout
                xs_dropped, us_dropped = apply_sensor_dropoff(
                    xs_single, 
                    us_single, 
                    self.train_sensor_dropoff, 
                    self.replace_with_nearest
                )
                
                xs_dropped_list.append(xs_dropped)
                us_dropped_list.append(us_dropped)
            
            # Stack the results back into batches
            xs = torch.stack(xs_dropped_list, dim=0)
            us = torch.stack(us_dropped_list, dim=0).unsqueeze(-1)
        
        return xs, us, ys, G_u_ys, None
    
    def denormalize_displacement(self, s_norm):
        """Denormalize displacement predictions."""
        return s_norm * (self.s_std + 1e-8) + self.s_mean
    
    def denormalize_force(self, u_norm):
        """Denormalize force values."""
        return u_norm * (self.u_std + 1e-8) + self.u_mean
    
    def denormalize_coordinates(self, coords_norm):
        """Coordinates are not normalized, so return as-is."""
        return coords_norm

def load_elastic_dataset(data_path=None, 
                        batch_size=64, device='cuda', train_sensor_dropoff=0.0, replace_with_nearest=False):
    """Load elastic dataset and return dataset wrapper."""
    if data_path is None:
        data_path = os.path.join(current_dir, "elastic_dataset")
    
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        # Create dataset wrapper
        elastic_dataset = ElasticDataset(
            dataset, 
            batch_size=batch_size, 
            device=device,
            train_sensor_dropoff=train_sensor_dropoff,
            replace_with_nearest=replace_with_nearest
        )
        
        return dataset, elastic_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/elastic_2d_data/get_elastic_data.py")
        return None, None
