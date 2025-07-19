import torch
import numpy as np
from tqdm import tqdm
from Data.data_utils import generate_batch, apply_sensor_dropoff
from .utils.helper_utils import calculate_l2_relative_error

def evaluate_deeponet_model(deeponet_model, eval_params, device):
    """
    Evaluates a DeepONet model on test data.
    
    Args:
        deeponet_model: Trained DeepONet model
        eval_params: Dictionary with evaluation parameters
        device: PyTorch device
    
    Returns:
        dict: Evaluation results
    """
    deeponet_model.eval()
    
    # Extract evaluation parameters
    sensor_x_original = eval_params['sensor_x_original']
    scale = eval_params['scale']
    input_range = eval_params['input_range']
    batch_size_train = eval_params['batch_size_train']
    n_trunk_points_eval = eval_params['n_trunk_points_eval']
    sensor_size = eval_params['sensor_size']
    n_test_samples_eval = eval_params['n_test_samples_eval']
    benchmark = eval_params['benchmark']
    variable_sensors = eval_params.get('variable_sensors', False)
    sensor_dropoff = eval_params.get('sensor_dropoff', 0.0)
    replace_with_nearest = eval_params.get('replace_with_nearest', False)
    
    total_l2_error = 0.0
    n_batches = (n_test_samples_eval + batch_size_train - 1) // batch_size_train
    
    print(f"\n--- Evaluating {benchmark.title()} Model (DeepONet) ---")
    if sensor_dropoff > 0:
        replacement_mode = "nearest neighbor replacement" if replace_with_nearest else "removal"
        print(f"Using sensor drop-off rate during evaluation: {sensor_dropoff:.1%} with {replacement_mode}")
    
    # Flag to print input sizes only once
    printed_input_sizes = False
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Evaluating"):
            current_batch_size = min(batch_size_train, n_test_samples_eval - batch_idx * batch_size_train)
            
            # Generate test batch with same sensor configuration as training
            if variable_sensors:
                batch_data = generate_batch(
                    batch_size=current_batch_size,
                    n_trunk_points=n_trunk_points_eval,
                    sensor_x=None,
                    scale=scale,
                    input_range=input_range,
                    device=device,
                    constant_zero=True,
                    variable_sensors=True,
                    sensor_size=sensor_size
                )
            else:
                batch_data = generate_batch(
                    batch_size=current_batch_size,
                    n_trunk_points=n_trunk_points_eval,
                    sensor_x=sensor_x_original,
                    scale=scale,
                    input_range=input_range,
                    device=device,
                    constant_zero=True
                )
            f_at_sensors_test = batch_data[0]
            _ = batch_data[1]  # f_prime_at_sensors not used
            f_at_trunk_test = batch_data[2]
            f_prime_at_trunk_test = batch_data[3]
            batch_x_eval_points_test = batch_data[4]
            if variable_sensors:
                sensor_x_to_use = batch_data[5]
            else:
                sensor_x_to_use = sensor_x_original
            
            # Apply sensor drop-off if specified
            if sensor_dropoff > 0.0:
                if benchmark == 'derivative':
                    sensor_x_dropped, f_at_sensors_dropped = apply_sensor_dropoff(
                        sensor_x_to_use, f_at_sensors_test, sensor_dropoff, replace_with_nearest
                    )
                elif benchmark == 'integral':
                    batch_x_eval_points_dropped, f_prime_at_trunk_dropped = apply_sensor_dropoff(
                        batch_x_eval_points_test, f_prime_at_trunk_test.T, sensor_dropoff, replace_with_nearest
                    )
                    f_prime_at_trunk_test = f_prime_at_trunk_dropped.T
                    batch_x_eval_points_test = batch_x_eval_points_dropped
                
                actual_sensor_size = f_at_sensors_dropped.shape[1] if benchmark == 'derivative' else f_prime_at_trunk_test.shape[1]
            else:
                f_at_sensors_dropped = f_at_sensors_test
                actual_sensor_size = sensor_size
            
            # Prepare model inputs and get predictions
            if benchmark == 'derivative':
                branch_input = f_at_sensors_dropped  # [current_batch_size, actual_sensor_size]
                trunk_input = batch_x_eval_points_test.view(-1, 1) if not variable_sensors else batch_x_eval_points_test.unsqueeze(0).expand(current_batch_size, -1, 1)
                
                # Print input sizes - only once
                if not printed_input_sizes:
                    print(f"\n--- Branch Input Sizes (Derivative Task) ---")
                    print(f"Branch input shape: {branch_input.shape}")  # [batch_size, n_sensors]
                    print(f"Trunk input shape: {trunk_input.shape}")   # [n_trunk_points, 1] or [batch_size, n_trunk_points, 1]
                    print(f"Effective sensors per sample: {actual_sensor_size}")
                    print(f"Trunk evaluation points per sample: {trunk_input.shape[-2] if trunk_input.dim() == 3 else trunk_input.shape[0]}")
                    printed_input_sizes = True
                
                pred = deeponet_model(branch_input, trunk_input)
                target = f_prime_at_trunk_test.T  # [current_batch_size, n_trunk_points_eval]
                
            elif benchmark == 'integral':
                branch_input = f_prime_at_trunk_test.T  # [current_batch_size, n_trunk_points_eval]
                trunk_input = sensor_x_to_use.view(-1, 1) if not variable_sensors else sensor_x_to_use.unsqueeze(0).expand(current_batch_size, -1, 1)
                
                # Print input sizes - only once
                if not printed_input_sizes:
                    print(f"\n--- Branch Input Sizes (Integral Task) ---")
                    print(f"Branch input shape: {branch_input.shape}")  # [batch_size, n_trunk_points]
                    print(f"Trunk input shape: {trunk_input.shape}")   # [n_query_points, 1] or [batch_size, n_query_points, 1]
                    print(f"Branch points per sample: {branch_input.shape[1]}")
                    print(f"Query points per sample: {trunk_input.shape[-2] if trunk_input.dim() == 3 else trunk_input.shape[0]}")
                    printed_input_sizes = True
                
                pred = deeponet_model(branch_input, trunk_input)
                target = f_at_sensors_test  # Note: Using original since code doesn't update for integral
            
            # Calculate L2 relative error for this batch
            batch_l2_error = calculate_l2_relative_error(pred, target)
            total_l2_error += batch_l2_error.item() * current_batch_size
    
    # Calculate average L2 relative error
    avg_l2_error = total_l2_error / n_test_samples_eval
    
    print(f"Final L2 relative error ({benchmark}): {avg_l2_error:.6f}")
    
    return {
        'final_l2_relative_error': avg_l2_error,
        'benchmark_task': benchmark,
        'n_test_samples': n_test_samples_eval,
        'variable_sensors_used': variable_sensors,
        'sensor_dropoff_used': sensor_dropoff,
        'replace_with_nearest_used': replace_with_nearest
    } 