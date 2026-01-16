#!/usr/bin/env python
"""burgers_1d_dataset.py
----------------------------------
Dataset wrapper and loader for 1D Burgers equation data from HuggingFace.
"""
from __future__ import annotations

import torch
import numpy as np
import os
import json
from datasets import load_dataset

from Data.data_utils import apply_sensor_dropoff

# Get paths relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
stats_path = os.path.join(current_dir, 'burgers_1d_normalization_stats.json')


def compute_stats(dataset):
    """Compute mean and std for u and s fields in the dataset."""
    u_data = np.array(dataset["u_initial"])
    # u_trajectory is [samples, time_steps, space], we need the last time step for s
    s_data = np.array(dataset["u_trajectory"])[:, -1]
    
    return {
        "u_mean": float(np.mean(u_data)),
        "u_std": float(np.std(u_data)),
        "s_mean": float(np.mean(s_data)),
        "s_std": float(np.std(s_data))
    }


def load_burgers_dataset(device="cpu"):
    """Load and return the Burgers 1D dataset from HuggingFace.
    
    Args:
        device: Device to load tensors to (not used for initial load, but kept for API compatibility)
    
    Returns:
        dict with 'train' and 'test' splits, each containing lists of samples with 'X', 'u', 'Y', 's' keys
        Also returns the normalization stats
    """
    print("Loading Burgers 1D dataset from HuggingFace (ajthor/burgers-fenics)...")
    
    # Load or compute normalization stats
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded normalization stats from {stats_path}")
    else:
        print(f"Normalization stats not found. Computing from training set...")
        train_ds = load_dataset("ajthor/burgers-fenics", split="train")
        stats = compute_stats(train_ds)
        
        # Save stats
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")
    
    # Load both splits
    train_hf = load_dataset("ajthor/burgers-fenics", split="train")
    test_hf = load_dataset("ajthor/burgers-fenics", split="test")
    
    def normalize(data, mean, std):
        """Normalize data using mean and std."""
        return (data - mean) / std
    
    def convert_split(hf_split, stats):
        """Convert HuggingFace split to our format with normalization."""
        samples = []
        for i in range(len(hf_split)):
            item = hf_split[i]
            
            # Get spatial coordinates (same for input and output)
            X = np.array(item["spatial_coordinates"], dtype=np.float32)
            
            # Get input function (initial condition) and normalize
            u = np.array(item["u_initial"], dtype=np.float32)
            u_normalized = normalize(u, stats["u_mean"], stats["u_std"])
            
            # Get output function (final time step of trajectory) and normalize
            trajectory = np.array(item["u_trajectory"], dtype=np.float32)
            s = trajectory[-1]  # Last time step
            s_normalized = normalize(s, stats["s_mean"], stats["s_std"])
            
            samples.append({
                'X': X,
                'u': u_normalized,
                'Y': X,  # Same grid for output
                's': s_normalized
            })
        return samples
    
    dataset = {
        'train': convert_split(train_hf, stats),
        'test': convert_split(test_hf, stats)
    }
    
    print(f"Loaded Burgers 1D dataset:")
    print(f"  Train samples: {len(dataset['train'])}")
    print(f"  Test samples: {len(dataset['test'])}")
    print(f"  Grid points: {len(dataset['train'][0]['X'])}")
    
    return dataset, stats


class BurgersDataGenerator:
    """Dataset wrapper for Burgers 1D data that's compatible with SetONet training loop."""

    def __init__(self, dataset, sensor_x, sensor_indices, query_indices, device, params, grid_points, stats=None):
        """
        Initialize the Burgers data generator.

        Args:
            dataset: Dictionary with 'train' and 'test' splits
            sensor_x: Tensor of sensor coordinates (n_sensors, 1)
            sensor_indices: Indices for sensor locations in the grid (or None if interpolated)
            query_indices: Indices for query locations in the grid
            device: PyTorch device
            params: Dictionary with training parameters
            grid_points: 1D tensor of spatial coordinates
            stats: Normalization statistics (optional, for denormalization)
        """
        print("📊 Pre-loading and optimizing Burgers dataset...")
        
        # Store basic info
        self.device = device
        self.params = params
        self.batch_size = params['batch_size_train']
        self.stats = stats
        
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
        self.n_mesh_points = len(query_indices)    # Number of query points
        self.input_dim = 1  # 1D coordinates
        
        # Pre-extract query data (always fixed)
        self.s_queries = self.s_data[:, self.query_indices]   # [n_train, n_queries]
        
        self.train_sensor_dropoff = params.get('train_sensor_dropoff', 0.0)
        self.replace_with_nearest = params.get('replace_with_nearest', False)

        # Fixed sensors: pre-extract sensor data for efficiency
        # Store sensor locations (passed from create_sensor_points)
        self.sensor_x = sensor_x

        # Extract sensor values (direct indexing or interpolation)
        if sensor_indices is not None:
            # Direct indexing from grid (sensor_size <= grid_size)
            self.sensor_indices = sensor_indices.to(device)
            self.u_sensors = self.u_data[:, self.sensor_indices]  # [n_train, n_sensors]
            self.n_force_points = len(sensor_indices)
        else:
            # Interpolation (sensor_size > grid_size)
            self.sensor_indices = None
            self.u_sensors = self._interpolate_sensors(self.u_data, self.grid_points, sensor_x)
            self.n_force_points = sensor_x.shape[0]

        print(f"✅ Burgers dataset optimized: {n_train} samples pre-loaded to GPU (FIXED sensors)")

        if self.train_sensor_dropoff > 0.0:
            replacement_mode = "nearest replacement" if self.replace_with_nearest else "removal"
            print(f"⚠️ Training with {self.train_sensor_dropoff:.1%} sensor dropout ({replacement_mode})")

    def _interpolate_sensors(self, data_grid, grid_points, sensor_x):
        """
        Interpolate sensor values from grid data using linear interpolation.

        Args:
            data_grid: (n_samples, n_grid) grid values for all samples
            grid_points: (n_grid,) grid coordinates in [0, 1]
            sensor_x: (n_sensors, 1) target sensor locations

        Returns:
            interpolated_values: (n_samples, n_sensors) interpolated sensor values
        """
        n_grid = data_grid.shape[1]

        # Flatten sensor_x to 1D for searchsorted
        sensor_x_flat = sensor_x.squeeze(-1)  # (n_sensors,)

        # Find indices of surrounding grid points using binary search
        # grid_points is sorted, so searchsorted gives us the right index
        indices = torch.searchsorted(grid_points, sensor_x_flat)  # (n_sensors,)

        # Clamp indices to valid range [1, n_grid-1]
        indices = torch.clamp(indices, 1, n_grid - 1)

        # Get left and right grid indices
        idx_left = indices - 1  # (n_sensors,)
        idx_right = indices      # (n_sensors,)

        # Get x coordinates of surrounding points
        x_left = grid_points[idx_left]   # (n_sensors,)
        x_right = grid_points[idx_right]  # (n_sensors,)

        # Compute interpolation weights
        weights = (sensor_x_flat - x_left) / (x_right - x_left + 1e-10)  # (n_sensors,)
        weights = weights.clamp(0.0, 1.0)  # Safety clamp

        # Vectorized interpolation for all samples
        y_left = data_grid[:, idx_left]   # (n_samples, n_sensors)
        y_right = data_grid[:, idx_right]  # (n_samples, n_sensors)
        weights_broadcast = weights.unsqueeze(0)  # (1, n_sensors)
        interpolated = y_left + weights_broadcast * (y_right - y_left)

        return interpolated

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
    
    def denormalize_output(self, s_normalized):
        """Denormalize output values using stored stats."""
        if self.stats is None:
            return s_normalized
        return s_normalized * self.stats['s_std'] + self.stats['s_mean']
    
    def denormalize_input(self, u_normalized):
        """Denormalize input values using stored stats."""
        if self.stats is None:
            return u_normalized
        return u_normalized * self.stats['u_std'] + self.stats['u_mean']


def create_sensor_points(params, device, grid_points):
    """Create fixed sensor points from the grid (with interpolation if sensor_size > grid_size)."""
    n_grid = len(grid_points)
    sensor_size = params['sensor_size']

    if sensor_size <= n_grid:
        # Use direct indexing (current approach)
        sensor_indices = torch.linspace(0, n_grid-1, sensor_size, dtype=torch.long)
        sensor_x = grid_points[sensor_indices].to(device).view(-1, 1)
        print(f"Using {sensor_size} FIXED sensor locations (evenly spaced, direct indexing)")
        return sensor_x, sensor_indices
    else:
        # Use interpolation to create more sensors than grid points
        # IMPORTANT: Match the range of the grid points, not hardcode [0, 1]
        grid_min = grid_points.min().item()
        grid_max = grid_points.max().item()
        sensor_x = torch.linspace(grid_min, grid_max, sensor_size, dtype=torch.float32, device=device).view(-1, 1)
        print(f"Using {sensor_size} FIXED sensor locations (evenly spaced in [{grid_min:.4f}, {grid_max:.4f}], interpolated from {n_grid} grid points)")
        return sensor_x, None  # None indicates interpolation mode


def create_query_points(params, device, grid_points, n_query_points):
    """Create query points for evaluation."""
    # Use evenly spaced query points
    query_indices = torch.linspace(0, len(grid_points)-1, n_query_points, dtype=torch.long)
    query_x = grid_points[query_indices].to(device).view(-1, 1)
    return query_x, query_indices


def setup_parameters(args):
    """Setup problem parameters."""
    return {
        'input_range': [0, 1],  # Burgers problem spatial domain
        'scale': 0.1,
        'sensor_size': args.sensor_size,
        'batch_size_train': args.batch_size,
        'n_trunk_points_train': args.n_query_points,
        'n_test_samples_eval': args.n_test_samples_eval,
        'sensor_seed': 42,
        'variable_sensors': False,
        'eval_sensor_dropoff': args.eval_sensor_dropoff,
        'replace_with_nearest': getattr(args, 'replace_with_nearest', False),
        'train_sensor_dropoff': args.train_sensor_dropoff,
    }

