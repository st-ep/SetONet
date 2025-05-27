import torch

def sample_trunk_points(n_points, input_range, device):
    """
    Generates linearly spaced points within the specified input_range on the given device.
    """
    return torch.linspace(input_range[0], input_range[1], n_points, device=device).view(-1, 1)

def sample_variable_sensor_points(sensor_size, input_range, device):
    """
    Generates random sensor points for variable sensor training.
    """
    sensor_x = torch.rand(sensor_size, device=device) * (input_range[1] - input_range[0]) + input_range[0]
    sensor_x = sensor_x.sort()[0]  # Sort for consistency
    return sensor_x.view(-1, 1)

def apply_sensor_dropoff(sensor_x, sensor_values, dropoff_rate, replace_with_nearest=False):
    """
    Randomly drops sensors based on the specified drop-off rate.
    
    Args:
        sensor_x: Sensor locations [n_sensors, 1]
        sensor_values: Sensor values [batch_size, n_sensors] or [n_sensors]
        dropoff_rate: Fraction of sensors to drop (0.0 to 1.0)
        replace_with_nearest: If True, replace dropped sensors with nearest remaining sensors
                             instead of removing them entirely (creates duplicate sensor pairs)
    
    Returns:
        Tuple of (processed_sensor_x, processed_sensor_values)
    """
    if dropoff_rate == 0.0:
        return sensor_x, sensor_values
    
    n_sensors = sensor_x.shape[0]
    n_keep = max(1, int(n_sensors * (1.0 - dropoff_rate)))  # Keep at least 1 sensor
    
    if not replace_with_nearest:
        # Original behavior: just remove dropped sensors entirely
        keep_indices = torch.randperm(n_sensors, device=sensor_x.device)[:n_keep]
        keep_indices = keep_indices.sort()[0]  # Sort to maintain order
        
        dropped_sensor_x = sensor_x[keep_indices]
        
        if sensor_values.dim() == 1:
            dropped_sensor_values = sensor_values[keep_indices]
        else:
            dropped_sensor_values = sensor_values[:, keep_indices]
        
        return dropped_sensor_x, dropped_sensor_values
    
    else:
        # Leverage permutation invariance: replace dropped sensors with nearest remaining sensors
        # This creates duplicate (location, value) pairs to maintain the same input size
        
        # Get indices efficiently
        all_indices = torch.randperm(n_sensors, device=sensor_x.device)
        keep_indices = all_indices[:n_keep]
        drop_indices = all_indices[n_keep:]
        
        if len(drop_indices) == 0:
            return sensor_x, sensor_values
        
        # Clone original tensors to avoid modifying inputs
        processed_sensor_x = sensor_x.clone()
        processed_sensor_values = sensor_values.clone()
        
        # Vectorized nearest neighbor replacement for (location, value) pairs
        keep_positions = sensor_x[keep_indices].squeeze()  # [n_keep]
        drop_positions = sensor_x[drop_indices].squeeze()  # [n_drop]
        
        # Calculate distances: [n_drop, n_keep]
        distances = torch.abs(drop_positions.unsqueeze(1) - keep_positions.unsqueeze(0))
        
        # Find nearest keep indices for each drop index: [n_drop]
        nearest_keep_local_indices = torch.argmin(distances, dim=1)
        nearest_keep_global_indices = keep_indices[nearest_keep_local_indices]
        
        # Debug: Print replacement info for first few sensors
        if len(drop_indices) > 0 and torch.rand(1).item() < 0.0:  # Set to 0.0 to disable
            print(f"\nSensor replacement debug:")
            print(f"Dropped sensor indices: {drop_indices.tolist()}")
            print(f"Nearest remaining indices: {nearest_keep_global_indices.tolist()}")
            for i, (drop_idx, nearest_idx) in enumerate(zip(drop_indices, nearest_keep_global_indices)):
                if i < 3:  # Only show first 3 for brevity
                    orig_pos = sensor_x[drop_idx].item()
                    nearest_pos = sensor_x[nearest_idx].item()
                    print(f"  Sensor {drop_idx.item()} at x={orig_pos:.3f} -> replaced with sensor {nearest_idx.item()} at x={nearest_pos:.3f}")
        
        # Replace BOTH positions and values as pairs (this leverages permutation invariance)
        # Example: if sensor 2 is dropped and sensor 3 is nearest:
        # Original: [(x1,u1), (x2,u2), (x3,u3)] -> Result: [(x1,u1), (x3,u3), (x3,u3)]
        processed_sensor_x[drop_indices] = sensor_x[nearest_keep_global_indices]
        
        if sensor_values.dim() == 1:
            # For 1D sensor values: [n_sensors]
            processed_sensor_values[drop_indices] = sensor_values[nearest_keep_global_indices]
        else:
            # For batched sensor values: [batch_size, n_sensors]
            processed_sensor_values[:, drop_indices] = sensor_values[:, nearest_keep_global_indices]
        
        return processed_sensor_x, processed_sensor_values

def generate_batch(batch_size, n_trunk_points, sensor_x, scale, input_range, device, constant_zero=True, variable_sensors=False, sensor_size=None):
    """
    Generates a batch of polynomial functions and their derivatives.
    
    Args:
        batch_size: Number of samples in the batch
        n_trunk_points: Number of trunk evaluation points
        sensor_x: Fixed sensor locations (None if variable_sensors=True)
        scale: Scale for polynomial coefficients
        input_range: Input domain range
        device: PyTorch device
        constant_zero: Whether to set integration constant to zero
        variable_sensors: Whether to use different sensor locations per batch (not per sample)
        sensor_size: Number of sensor points (required if variable_sensors=True)
    
    Returns:
        f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval, [sensor_x_batch]
    """
    # Generate trunk points
    x_eval = sample_trunk_points(n_trunk_points, input_range, device)
    
    # Handle sensor locations
    if variable_sensors:
        if sensor_size is None:
            raise ValueError("sensor_size must be provided when variable_sensors=True")
        # Generate ONE set of sensor locations for the entire batch
        sensor_x_to_use = sample_variable_sensor_points(sensor_size, input_range, device)
    else:
        # Use fixed sensor locations for all samples
        if sensor_x is None:
            raise ValueError("sensor_x must be provided when variable_sensors=False")
        sensor_x_to_use = sensor_x
    
    # Generate polynomial coefficients
    a = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    if constant_zero:
        d = torch.zeros(batch_size, device=device)
    else:
        d = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    
    # Vectorized evaluation at sensor points (much faster!)
    # sensor_x_to_use: [sensor_size, 1]
    # a, b, c, d: [batch_size]
    # Result: [batch_size, sensor_size]
    sensor_x_expanded = sensor_x_to_use.T  # [1, sensor_size]
    
    f_at_sensors = (a.unsqueeze(1) * sensor_x_expanded**3 + 
                   b.unsqueeze(1) * sensor_x_expanded**2 + 
                   c.unsqueeze(1) * sensor_x_expanded + 
                   d.unsqueeze(1))  # [batch_size, sensor_size]
    
    f_prime_at_sensors = (3 * a.unsqueeze(1) * sensor_x_expanded**2 + 
                         2 * b.unsqueeze(1) * sensor_x_expanded + 
                         c.unsqueeze(1))  # [batch_size, sensor_size]
    
    # Vectorized evaluation at trunk points (same as before)
    f_at_trunk = a.unsqueeze(1) * x_eval.T**3 + b.unsqueeze(1) * x_eval.T**2 + c.unsqueeze(1) * x_eval.T + d.unsqueeze(1)
    f_prime_at_trunk = 3 * a.unsqueeze(1) * x_eval.T**2 + 2 * b.unsqueeze(1) * x_eval.T + c.unsqueeze(1)
    
    f_at_trunk = f_at_trunk.T  # [n_trunk_points, batch_size]
    f_prime_at_trunk = f_prime_at_trunk.T  # [n_trunk_points, batch_size]
    
    if variable_sensors:
        return f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval, sensor_x_to_use
    else:
        return f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval 