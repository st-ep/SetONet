#!/usr/bin/env python
"""darcy_1d_dataset.py
----------------------------------
Dataset wrapper and loader for 1D Darcy flow data.
"""
from __future__ import annotations

import torch
import numpy as np
from datasets import load_from_disk

from Data.data_utils import sample_variable_sensor_points, apply_sensor_dropoff

def load_darcy_dataset(data_path):
    """Load and return the Darcy 1D dataset."""
    try:
        dataset = load_from_disk(data_path)
        print(f"Loaded Darcy 1D dataset from: {data_path}")
        print(f"Train samples: {len(dataset['train'])}")
        print(f"Test samples: {len(dataset['test'])}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {data_path}: {e}")
        raise

class DarcyDataGenerator:
    """Dataset wrapper for Darcy 1D data that's compatible with SetONet training loop."""
    
    def __init__(self, dataset, sensor_indices, query_indices, device, params, grid_points):
        print("📊 Pre-loading and optimizing dataset...")
        
        # Store basic info
        self.device = device
        self.params = params
        self.batch_size = params['batch_size_train']
        
        # PRE-LOAD all data to GPU for maximum efficiency
        train_data = dataset['train']
        n_train = len(train_data)
        n_grid = len(train_data[0]['u'])
        
        # Pre-allocate tensors on GPU
        self.u_data = torch.zeros(n_train, n_grid, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(n_train, n_grid, device=device, dtype=torch.float32)
        
        # Load all data to GPU once (much faster than per-batch loading)
        for i in range(n_train):
            self.u_data[i] = torch.tensor(train_data[i]['u'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(train_data[i]['s'], device=device, dtype=torch.float32)
        
        self.query_indices = query_indices.to(device)
        self.n_train = n_train
        self.grid_points = grid_points.to(device)
        self.n_grid = n_grid
        
        # Store query points for TensorBoard callback compatibility
        self.query_x = self.grid_points[self.query_indices].view(-1, 1)
        
        # Dataset structure info (like elastic dataset)
        self.n_force_points = len(sensor_indices)  # Number of sensor points
        self.n_mesh_points = len(query_indices)    # Number of query points
        self.input_dim = 1  # 1D coordinates
        
        # Pre-extract query data (always fixed)
        self.s_queries = self.s_data[:, self.query_indices]   # [n_train, n_queries]
        
        self.train_sensor_dropoff = params.get('train_sensor_dropoff', 0.0)
        self.replace_with_nearest = params.get('replace_with_nearest', False)
        
        # Fixed sensors: pre-extract sensor data for efficiency
        self.sensor_indices = sensor_indices.to(device)
        self.sensor_x = self.grid_points[self.sensor_indices].view(-1, 1)
        self.u_sensors = self.u_data[:, self.sensor_indices]  # [n_train, n_sensors]
        print(f"✅ Dataset optimized: {n_train} samples pre-loaded to GPU (FIXED sensors)")
        
        if self.train_sensor_dropoff > 0.0:
            replacement_mode = "nearest replacement" if self.replace_with_nearest else "removal"
            print(f"⚠️ Training with {self.train_sensor_dropoff:.1%} sensor dropout ({replacement_mode})")
    
    def sample(self, device=None):
        """Sample a batch using pre-loaded GPU tensors (compatible with SetONet training loop)."""
        # Random sampling directly on GPU (much faster)
        indices = torch.randint(0, self.n_train, (self.batch_size,), device=self.device)
        
        # Fixed sensors: use pre-extracted sensor data
        u_at_sensors = self.u_sensors[indices]  # [batch_size, n_sensors]
        batch_sensor_x = self.sensor_x
        
        # Query data is always the same
        s_at_queries = self.s_queries[indices]  # [batch_size, n_queries]
        
        # Prepare data in SetONet format
        xs = batch_sensor_x.unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size, n_sensors, 1]
        us = u_at_sensors.unsqueeze(-1)  # [batch_size, n_sensors, 1]
        ys = self.grid_points[self.query_indices].view(-1, 1).unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size, n_queries, 1]
        G_u_ys = s_at_queries.unsqueeze(-1)  # [batch_size, n_queries, 1]
        
        # Apply sensor dropout during training if specified
        if self.train_sensor_dropoff > 0.0:
            xs_dropped_list = []
            us_dropped_list = []
            
            # Apply sensor dropout to each sample in the batch
            for i in range(self.batch_size):
                # Remove batch dimension for dropout function
                xs_single = xs[i]  # Shape: (n_sensors, 1)
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
    
    def generate_batch(self, batch_size):
        """Legacy method for backward compatibility."""
        # Temporarily override batch size for this call
        original_batch_size = self.batch_size
        self.batch_size = batch_size
        
        # Use the sample method and extract the raw data
        xs, us, ys, G_u_ys, _ = self.sample()
        
        # Restore original batch size
        self.batch_size = original_batch_size
        
        # Convert back to the original format
        u_at_sensors = us.squeeze(-1)  # [batch_size, n_sensors]
        s_at_queries = G_u_ys.squeeze(-1)  # [batch_size, n_queries]
        batch_sensor_x = xs[0]  # [n_sensors, 1] - same for all samples
        
        return u_at_sensors, s_at_queries, batch_sensor_x

def create_sensor_points(params, device, grid_points):
    """Create fixed sensor points from the grid."""
    # Use fixed sensor locations - evenly spaced subset of grid points
    sensor_indices = torch.linspace(0, len(grid_points)-1, params['sensor_size'], dtype=torch.long)
    sensor_x = grid_points[sensor_indices].to(device).view(-1, 1)
    print(f"Using {params['sensor_size']} FIXED sensor locations (evenly spaced)")
    return sensor_x, sensor_indices

def create_query_points(params, device, grid_points, n_query_points):
    """Create query points for evaluation."""
    # Use evenly spaced query points
    query_indices = torch.linspace(0, len(grid_points)-1, n_query_points, dtype=torch.long)
    query_x = grid_points[query_indices].to(device).view(-1, 1)
    return query_x, query_indices

def setup_parameters(args):
    """Setup problem parameters."""
    return {
        'input_range': [0, 1],  # Darcy problem is on [0,1] (keep domain-specific)
        'scale': 0.1,  # Same as run_1d.py
        'sensor_size': args.sensor_size,
        'batch_size_train': args.batch_size,
        'n_trunk_points_train': args.n_query_points,  # Query points are trunk points for Darcy
        'n_test_samples_eval': args.n_test_samples_eval,
        'sensor_seed': 42,
        'variable_sensors': False,
        'eval_sensor_dropoff': args.eval_sensor_dropoff,
        'replace_with_nearest': getattr(args, 'replace_with_nearest', False),  # Default to False for DeepONet scripts
        'train_sensor_dropoff': args.train_sensor_dropoff,
    } 