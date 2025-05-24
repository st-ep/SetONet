import torch
import sys
import os

def calculate_l2_relative_error(y_pred, y_true):
    """
    Calculates the mean L2 relative error.
    Args:
        y_pred (torch.Tensor): Predicted values, shape (batch_size, num_points).
        y_true (torch.Tensor): True values, shape (batch_size, num_points).
    Returns:
        torch.Tensor: Mean L2 relative error.
    """
    error_norm = torch.norm(y_pred - y_true, dim=1)
    true_norm = torch.norm(y_true, dim=1)
    # Add a small epsilon to prevent division by zero if true_norm is zero for some samples
    relative_error = error_norm / (true_norm + 1e-8) 
    return relative_error.mean()

def prepare_setonet_inputs(sensor_x_global, current_batch_size, batch_f_values_norm_expanded, batch_x_eval_norm, global_sensor_size):
    # sensor_x_global: [S, 1] (normalized sensor locations, S = global_sensor_size)
    # current_batch_size: int (batch size)
    # batch_f_values_norm_expanded: [B, S, 1] (normalized sensor values)
    # batch_x_eval_norm: [T, 1] (normalized trunk query locations)
    # global_sensor_size: int (number of sensor points, S)

    # Ensure current_batch_size is an int for expand()
    _current_batch_size_int = current_batch_size
    if isinstance(_current_batch_size_int, torch.Tensor):
        _current_batch_size_int = _current_batch_size_int.item() # Convert 0-dim tensor to Python number
    _current_batch_size_int = int(_current_batch_size_int) # Ensure it's an integer

    # Ensure global_sensor_size is an int for view()
    _global_sensor_size_int = global_sensor_size
    if isinstance(_global_sensor_size_int, torch.Tensor):
        _global_sensor_size_int = _global_sensor_size_int.item()
    _global_sensor_size_int = int(_global_sensor_size_int)


    # Prepare sensor locations for SetONet branch: xs_setonet [B, S, 1]
    # sensor_x_global is [global_sensor_size, 1]
    # We need to expand it to [current_batch_size, global_sensor_size, 1]
    xs_setonet = sensor_x_global.view(1, _global_sensor_size_int, 1).expand(_current_batch_size_int, -1, -1)

    # Sensor values for SetONet branch: us_setonet [B, S, 1]
    us_setonet = batch_f_values_norm_expanded # This is already [B, S, 1]

    # Trunk query locations for SetONet: ys_setonet [B, T, 1]
    # batch_x_eval_norm is [T, 1]
    # We need to expand it to [current_batch_size, T, 1]
    num_trunk_points = batch_x_eval_norm.shape[0] # shape[0] gives an int
    ys_setonet = batch_x_eval_norm.view(1, num_trunk_points, 1).expand(_current_batch_size_int, -1, -1)
    
    return xs_setonet, us_setonet, ys_setonet