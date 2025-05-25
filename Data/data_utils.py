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