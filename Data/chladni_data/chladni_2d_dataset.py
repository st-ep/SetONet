#!/usr/bin/env python
"""chladni_2d_dataset.py
----------------------------------
Dataset wrapper and loader for 2D Chladni plate data.
"""
from __future__ import annotations

import numpy as np
import torch
import json
from datasets import load_from_disk

class ChladniDataset:
    """Dataset wrapper for Chladni plate data that uses pre-normalized data."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("Loading pre-normalized Chladni dataset...")
        
        self.batch_size = batch_size
        self.device = device
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
        with open('/home/titanv/Stepan/setprojects/SetONet/Data/chladni_data/chladni_normalization_stats.json', 'r') as f:
            stats = json.load(f)
        
        self.u_mean = torch.tensor(stats['u_mean'], device=device, dtype=torch.float32)
        self.u_std = torch.tensor(stats['u_std'], device=device, dtype=torch.float32)
        self.s_mean = torch.tensor(stats['s_mean'], device=device, dtype=torch.float32)
        self.s_std = torch.tensor(stats['s_std'], device=device, dtype=torch.float32)
        self.xy_mean = torch.tensor(stats['xy_mean'], device=device, dtype=torch.float32)
        self.xy_std = torch.tensor(stats['xy_std'], device=device, dtype=torch.float32)
        
    def sample(self, device=None):
        """Sample a batch using pre-normalized GPU tensors."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        xs = self.X_data[indices]
        us = self.u_data[indices].unsqueeze(-1)
        ys = self.Y_data[indices]
        G_u_ys = self.s_data[indices].unsqueeze(-1)
        
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

def load_chladni_dataset(data_path="/home/titanv/Stepan/setprojects/SetONet/Data/chladni_data/chladni_dataset", batch_size=64, device='cuda'):
    """Load chladni dataset and return dataset wrapper."""
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
        
        # Create dataset wrapper
        chladni_dataset = ChladniDataset(dataset, batch_size=batch_size, device=device)
        
        return dataset, chladni_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/chladni_data/chladni_plate_generator.py")
        return None, None 