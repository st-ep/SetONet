#!/usr/bin/env python
"""darcy_1d_dataset.py
----------------------------------
Dataset wrapper and loader for 1D Darcy flow data.
"""
from __future__ import annotations

import torch
import numpy as np
from datasets import load_from_disk

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

def interpolate_sensor_values(u_data, grid_points, sensor_locations, device):
    """
    Efficiently interpolate sensor values at arbitrary continuous locations using linear interpolation.
    
    Args:
        u_data: [n_samples, n_grid] - Function values on the grid
        grid_points: [n_grid] - Grid point locations (assumed to be sorted)
        sensor_locations: [n_sensors] - Arbitrary sensor locations to interpolate at
        device: PyTorch device
        
    Returns:
        u_interpolated: [n_samples, n_sensors] - Interpolated values at sensor locations
    """
    # Ensure all tensors are on the same device
    grid_points = grid_points.to(device)
    sensor_locations = sensor_locations.to(device)
    u_data = u_data.to(device)
    
    # Use torch.searchsorted to find interpolation indices efficiently
    # This finds the right insertion point for each sensor location in the sorted grid
    indices = torch.searchsorted(grid_points, sensor_locations, right=False)
    
    # Clamp indices to valid range [1, n_grid-1] to avoid boundary issues
    n_grid = len(grid_points)
    indices = torch.clamp(indices, 1, n_grid - 1)
    
    # Get left and right grid points for interpolation
    left_indices = indices - 1  # [n_sensors]
    right_indices = indices     # [n_sensors]
    
    # Get grid coordinates for interpolation
    x_left = grid_points[left_indices]   # [n_sensors]
    x_right = grid_points[right_indices] # [n_sensors]
    
    # Compute interpolation weights
    # weight = (sensor_loc - x_left) / (x_right - x_left)
    dx = x_right - x_left
    # Handle potential division by zero (though shouldn't happen with proper grid)
    dx = torch.where(dx > 1e-10, dx, torch.ones_like(dx))
    weights = (sensor_locations - x_left) / dx  # [n_sensors]
    
    # Get function values at left and right grid points
    # u_data: [n_samples, n_grid], left_indices: [n_sensors]
    # Result: [n_samples, n_sensors]
    u_left = u_data[:, left_indices]   # [n_samples, n_sensors]
    u_right = u_data[:, right_indices] # [n_samples, n_sensors]
    
    # Linear interpolation: u_interp = u_left + weight * (u_right - u_left)
    u_interpolated = u_left + weights.unsqueeze(0) * (u_right - u_left)
    
    return u_interpolated

class DarcyDataGenerator:
    """OPTIMIZED Data generator for Darcy 1D dataset with support for variable sensors."""
    
    def __init__(self, dataset, sensor_indices, query_indices, device, params, grid_points):
        print("📊 Pre-loading and optimizing dataset...")
        
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
        self.device = device
        self.n_train = n_train
        self.params = params
        self.grid_points = grid_points.to(device)
        self.n_grid = n_grid
        self.input_range = params.get('input_range', [0, 1])  # Domain range for continuous sampling
        
        # Pre-extract query data (always fixed)
        self.s_queries = self.s_data[:, self.query_indices]   # [n_train, n_queries]
        
        if not params['variable_sensors']:
            # Fixed sensors: pre-extract sensor data for efficiency
            self.sensor_indices = sensor_indices.to(device)
            self.u_sensors = self.u_data[:, self.sensor_indices]  # [n_train, n_sensors]
            print(f"✅ Dataset optimized: {n_train} samples pre-loaded to GPU (FIXED sensors)")
        else:
            # Variable sensors: don't pre-extract sensor data (will interpolate dynamically)
            self.sensor_indices = None
            self.u_sensors = None
            print(f"✅ Dataset optimized: {n_train} samples pre-loaded to GPU (VARIABLE sensors with interpolation)")
        
    def generate_batch(self, batch_size):
        """OPTIMIZED: Generate a batch using pre-loaded GPU data."""
        # Random sampling directly on GPU (much faster)
        indices = torch.randint(0, self.n_train, (batch_size,), device=self.device)
        
        if not self.params['variable_sensors']:
            # Fixed sensors: use pre-extracted sensor data
            u_at_sensors = self.u_sensors[indices]  # [batch_size, n_sensors]
        else:
            # Variable sensors: generate new CONTINUOUS random sensor locations for this batch
            from Data.data_utils import sample_variable_sensor_points
            
            # Generate truly random sensor locations from continuous domain [0,1]
            current_sensor_x = sample_variable_sensor_points(
                self.params['sensor_size'], 
                self.input_range, 
                self.device
            ).squeeze(-1)  # [n_sensors] - remove the last dimension for interpolation
            
            # Interpolate sensor values at these arbitrary continuous locations
            # This is the key improvement: we're not limited to grid points anymore!
            u_batch = self.u_data[indices]  # [batch_size, n_grid]
            u_at_sensors = interpolate_sensor_values(
                u_batch, 
                self.grid_points, 
                current_sensor_x, 
                self.device
            )  # [batch_size, n_sensors]
            
            # Convert back to the expected format for return
            current_sensor_x = current_sensor_x.view(-1, 1)  # [n_sensors, 1]
        
        # Query data is always the same
        s_at_queries = self.s_queries[indices]  # [batch_size, n_queries]
        
        # Return sensor locations along with data for variable sensors
        if self.params['variable_sensors']:
            return u_at_sensors, s_at_queries, current_sensor_x
        else:
            return u_at_sensors, s_at_queries

def create_sensor_points(params, device, grid_points):
    """Create sensor points from the grid - fixed or variable based on params."""
    if params['variable_sensors']:
        # For variable sensors, we'll generate them dynamically during training
        # But we still need to return initial sensor locations for setup
        print(f"Using VARIABLE sensor locations (random {params['sensor_size']} CONTINUOUS locations per batch)")
        print("Note: Sensor locations will be sampled from continuous domain [0,1] and interpolated")
        # Return initial random sensor locations just for setup
        sensor_indices = torch.randperm(len(grid_points))[:params['sensor_size']].sort()[0]
        sensor_x = grid_points[sensor_indices].to(device).view(-1, 1)
        return sensor_x, sensor_indices
    else:
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
        'variable_sensors': args.variable_sensors,
        'eval_sensor_dropoff': args.eval_sensor_dropoff,
        'replace_with_nearest': args.replace_with_nearest,
    } 