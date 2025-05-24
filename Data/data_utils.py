import torch

def sample_trunk_points(n_points, input_range, device):
    """
    Generates linearly spaced points within the specified input_range on the given device.
    """
    return torch.linspace(input_range[0], input_range[1], n_points, device=device).view(-1, 1)

def generate_batch(batch_size, n_trunk_points, sensor_x, scale, input_range, device, *, constant_zero=False):
    """
    Generates a batch of cubic polynomials, their values, and their derivatives
    at specified sensor and trunk locations.
    
    Args:
        batch_size (int): Number of samples in the batch.
        n_trunk_points (int): Number of points to evaluate the derivative at (trunk points).
        sensor_x (torch.Tensor): Fixed sensor locations for evaluating the function and its derivative.
                                 Shape: [num_sensors, 1] (should be on device).
        scale (float): Scaling factor for polynomial coefficients.
        input_range (list or tuple): The [min, max] range for trunk points.
        device (torch.device): The device to create tensors on.
        constant_zero (bool): If True, sets the integration constant d=0 so that
                              f′(x) uniquely determines f(x).  Default False.
        
    Returns:
        f_at_sensors (torch.Tensor): Function values f(x_i) at sensor_x locations.
                                     Shape: [batch_size, num_sensors]
        f_prime_at_sensors (torch.Tensor): Derivative values f'(x_i) at sensor_x locations.
                                           Shape: [batch_size, num_sensors]
        f_at_trunk (torch.Tensor): Function values f(y_j) at x_eval (trunk) locations.
                                   Shape: [batch_size, n_trunk_points]
        f_prime_at_trunk (torch.Tensor): True derivative values f'(y_j) at x_eval (trunk) points.
                                         Shape: [batch_size, n_trunk_points]
        x_eval (torch.Tensor): Trunk evaluation points y_j. Shape: [n_trunk_points, 1]
    """
    a = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale

    if constant_zero:
        d = torch.zeros(batch_size, 1, device=device)
    else:
        d = (torch.rand(batch_size, 1, device=device) * 2 - 1) * scale

    # Function values at sensor locations
    # sensor_x is [num_sensors, 1]. After broadcasting with a,b,c,d [B,1], sensor_x**3 becomes [B, num_sensors]
    f_at_sensors = a * sensor_x.T**3 + b * sensor_x.T**2 + c * sensor_x.T + d

    # Derivative values at sensor locations
    f_prime_at_sensors = 3 * a * sensor_x.T**2 + 2 * b * sensor_x.T + c

    # Trunk input points (evaluation y_j for derivative and function)
    x_eval = sample_trunk_points(n_trunk_points, input_range, device) # Shape [n_trunk_points, 1]

    # Function values at trunk locations
    # x_eval.T is [1, n_trunk_points]. After broadcasting, f_at_trunk is [B, n_trunk_points]
    f_at_trunk = a * x_eval.T**3 + b * x_eval.T**2 + c * x_eval.T + d
    
    # True derivative values at trunk (x_eval) points
    f_prime_at_trunk = 3 * a * x_eval.T**2 + 2 * b * x_eval.T + c
    
    return f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval 