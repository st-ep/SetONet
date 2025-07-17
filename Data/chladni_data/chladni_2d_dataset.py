#!/usr/bin/env python
"""chladni_2d_dataset.py
----------------------------------
Dataset wrapper and loader for 2D Chladni plate data.
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

class ChladniDataset:
    """Dataset wrapper for Chladni plate data that uses pre-normalized data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda', normalization_stats_path=None, 
                 train_sensor_dropoff=0.0, replace_with_nearest=False):
        print("Loading pre-normalized Chladni dataset...")
        
        self.batch_size = batch_size
        self.device = device
        self.train_sensor_dropoff = train_sensor_dropoff
        self.replace_with_nearest = replace_with_nearest
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        self.n_points = len(sample_0['X'])
        self.input_dim = len(sample_0['X'][0])
        
        # Pre-allocate tensors on GPU for ALL data
        self.X_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.u_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        self.Y_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        
        # Load pre-normalized data to GPU
        for i in range(self.n_samples):
            sample = train_data[i]
            self.X_data[i] = torch.tensor(sample['X'], device=device, dtype=torch.float32)
            self.u_data[i] = torch.tensor(sample['u'], device=device, dtype=torch.float32)
            self.Y_data[i] = torch.tensor(sample['Y'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(sample['s'], device=device, dtype=torch.float32)
        
        print(f"Dataset loaded: {self.n_samples} samples, {self.n_points} points")
        
        # Load normalization statistics
        if normalization_stats_path is None:
            normalization_stats_path = os.path.join(current_dir, 'chladni_normalization_stats.json')
        
        with open(normalization_stats_path, 'r') as f:
            stats = json.load(f)
        
        self.u_mean = torch.tensor(stats['u_mean'], device=device, dtype=torch.float32)
        self.u_std = torch.tensor(stats['u_std'], device=device, dtype=torch.float32)
        self.s_mean = torch.tensor(stats['s_mean'], device=device, dtype=torch.float32)
        self.s_std = torch.tensor(stats['s_std'], device=device, dtype=torch.float32)
        self.xy_mean = torch.tensor(stats['xy_mean'], device=device, dtype=torch.float32)
        self.xy_std = torch.tensor(stats['xy_std'], device=device, dtype=torch.float32)
        
        if self.train_sensor_dropoff > 0.0:
            replacement_mode = "nearest replacement" if self.replace_with_nearest else "removal"
            print(f"⚠️ Training with {self.train_sensor_dropoff:.1%} sensor dropout ({replacement_mode}) in ChladniDataset")
        
    def sample(self, device=None):
        """Sample a batch using pre-normalized GPU tensors."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        xs = self.X_data[indices]  # [batch_size, n_points, 2]
        us = self.u_data[indices].unsqueeze(-1)  # [batch_size, n_points, 1]
        ys = self.Y_data[indices]  # [batch_size, n_points, 2]
        G_u_ys = self.s_data[indices].unsqueeze(-1)  # [batch_size, n_points, 1]
        
        # Apply sensor dropout during training if specified
        if self.train_sensor_dropoff > 0.0:
            xs_list = []
            us_list = []
            
            for sample_idx in range(self.batch_size):
                # Apply dropout to this sample's sensor data
                # xs[sample_idx]: [n_points, 2], us[sample_idx]: [n_points, 1]
                xs_dropped, us_dropped = apply_sensor_dropoff(
                    xs[sample_idx],  # [n_points, 2]
                    us[sample_idx].squeeze(-1),  # [n_points] (remove last dimension for apply_sensor_dropoff)
                    self.train_sensor_dropoff,
                    self.replace_with_nearest
                )
                
                xs_list.append(xs_dropped)
                us_list.append(us_dropped.unsqueeze(-1))  # Add back the last dimension
            
            # Stack the results back into batch format
            xs = torch.stack(xs_list, dim=0)
            us = torch.stack(us_list, dim=0)
        
        return xs, us, ys, G_u_ys, None
    
    def denormalize_displacement(self, s_norm):
        """Denormalize displacement predictions."""
        return s_norm * (self.s_std + 1e-8) + self.s_mean
    
    def denormalize_force(self, u_norm):
        """Denormalize force values."""
        return u_norm * (self.u_std + 1e-8) + self.u_mean
    
    def denormalize_coordinates(self, coords_norm):
        """Denormalize coordinates."""
        return coords_norm * self.xy_std + self.xy_mean

def load_chladni_dataset(data_path=None, batch_size=64, device='cuda'):
    """Load chladni dataset and return dataset wrapper."""
    
    # Default to current directory if no path provided
    if data_path is None:
        data_path = os.path.join(current_dir, "chladni_dataset")
    
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        # Create dataset wrapper with corresponding normalization stats path
        normalization_stats_path = os.path.join(os.path.dirname(data_path), 'chladni_normalization_stats.json')
        chladni_dataset = ChladniDataset(dataset, batch_size=batch_size, device=device, 
                                       normalization_stats_path=normalization_stats_path)
        
        return dataset, chladni_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/chladni_data/chladni_plate_generator.py")
        return None, None 