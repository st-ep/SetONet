import torch
import numpy as np
from tqdm import tqdm
from Data.data_utils import generate_batch, apply_sensor_dropoff
from .utils.helper_utils import calculate_l2_relative_error, prepare_setonet_inputs

def evaluate_setonet_model(setonet_model, eval_params, device):
    """
    Evaluates a SetONet model on test data.
    
    Args:
        setonet_model: Trained SetONet model
        eval_params: Dictionary with evaluation parameters
        device: PyTorch device
    
    Returns:
        dict: Evaluation results
    """
    setonet_model.eval()
    
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
    
    print(f"\n--- Evaluating {benchmark.title()} Model ---")
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
                f_at_sensors_test, _, f_at_trunk_test, f_prime_at_trunk_test, batch_x_eval_points_test, sensor_x_to_use = batch_data
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
                f_at_sensors_test, _, f_at_trunk_test, f_prime_at_trunk_test, batch_x_eval_points_test = batch_data
                sensor_x_to_use = sensor_x_original
            
            # Apply sensor drop-off if specified
            sensor_x_dropped = sensor_x_to_use
            f_at_sensors_dropped = f_at_sensors_test
            actual_sensor_size = len(sensor_x_to_use)
            
            if sensor_dropoff > 0.0:
                if benchmark == 'derivative':
                    sensor_x_dropped, f_at_sensors_dropped = apply_sensor_dropoff(
                        sensor_x_to_use, f_at_sensors_test, sensor_dropoff, replace_with_nearest
                    )
                elif benchmark == 'integral':
                    # For integral task, we need to drop from trunk points and values
                    batch_x_eval_points_dropped, f_prime_at_trunk_dropped = apply_sensor_dropoff(
                        batch_x_eval_points_test, f_prime_at_trunk_test.T, sensor_dropoff, replace_with_nearest
                    )
                    f_prime_at_trunk_test = f_prime_at_trunk_dropped.T
                    batch_x_eval_points_test = batch_x_eval_points_dropped
                
                actual_sensor_size = len(sensor_x_dropped)
            
            # Prepare model inputs and get predictions
            if benchmark == 'derivative':
                # Use the sensor locations for this batch (possibly dropped)
                xs, us, ys = prepare_setonet_inputs(
                    sensor_x_dropped,
                    current_batch_size,
                    f_at_sensors_dropped.unsqueeze(-1),
                    batch_x_eval_points_test,
                    actual_sensor_size
                )
                
                # Print input sizes to phi network (sensor data) - only once
                if not printed_input_sizes:
                    print(f"\n--- Phi Network Input Sizes (Derivative Task) ---")
                    print(f"Sensor locations (xs) shape: {xs.shape}")  # [batch_size, n_sensors, 1]
                    print(f"Sensor values (us) shape: {us.shape}")     # [batch_size, n_sensors, 1] 
                    print(f"Trunk evaluation points (ys) shape: {ys.shape}")  # [batch_size, n_trunk_points, 1]
                    print(f"Effective sensors per sample: {actual_sensor_size}")
                    print(f"Trunk evaluation points per sample: {ys.shape[1]}")
                    printed_input_sizes = True
                
                pred = setonet_model(xs, us, ys)
                pred = pred.squeeze(-1)  # [current_batch_size, n_trunk_points_eval]
                target = f_prime_at_trunk_test.T  # [current_batch_size, n_trunk_points_eval]
                
            elif benchmark == 'integral':
                # Use the sensor locations for this batch (possibly dropped)
                xs, us, ys = prepare_setonet_inputs(
                    batch_x_eval_points_test,
                    current_batch_size,
                    f_prime_at_trunk_test.T.unsqueeze(-1),
                    sensor_x_dropped,
                    n_trunk_points_eval
                )
                
                # Print input sizes to phi network (trunk data) - only once
                if not printed_input_sizes:
                    print(f"\n--- Phi Network Input Sizes (Integral Task) ---")
                    print(f"Trunk locations (xs) shape: {xs.shape}")     # [batch_size, n_trunk_points, 1]
                    print(f"Trunk values (us) shape: {us.shape}")       # [batch_size, n_trunk_points, 1]
                    print(f"Query locations (ys) shape: {ys.shape}")    # [batch_size, n_query_points, 1]
                    print(f"Trunk points per sample: {n_trunk_points_eval}")
                    print(f"Query points per sample: {ys.shape[1]}")
                    printed_input_sizes = True
                
                pred = setonet_model(xs, us, ys)
                pred = pred.squeeze(-1)  # [current_batch_size, actual_sensor_size]
                target = f_at_sensors_dropped  # [current_batch_size, actual_sensor_size]
            
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

# Keep the old function for backward compatibility
def evaluate_setonet_models(setonet_model_T, setonet_model_T_inv, eval_params, device):
    """
    Legacy function for backward compatibility.
    Evaluates both SetONet models on test data.
    """
    print("\n--- Evaluating Models ---")
    
    # Extract evaluation parameters
    sensor_x_original = eval_params['sensor_x_original']
    scale = eval_params['scale']
    input_range = eval_params['input_range']
    batch_size_train = eval_params['batch_size_train']
    n_trunk_points_eval = eval_params['n_trunk_points_eval']
    sensor_size = eval_params['sensor_size']
    n_test_samples_eval = eval_params['n_test_samples_eval']
    
    setonet_model_T.eval()
    setonet_model_T_inv.eval()
    
    all_rel_errors_T = []
    all_rel_errors_T_inv = []

    with torch.no_grad():
        for _ in tqdm(range(n_test_samples_eval // batch_size_train), desc="Evaluating"):
            f_at_sensors_test, _, _, f_prime_at_trunk_test, batch_x_eval_points_test = generate_batch(
                batch_size=batch_size_train,
                n_trunk_points=n_trunk_points_eval,
                sensor_x=sensor_x_original,
                scale=scale,
                input_range=input_range,
                device=device,
                constant_zero=True
            )
            current_eval_batch_size = f_at_sensors_test.shape[0]

            # Evaluate T
            xs_T_test, us_T_test, ys_T_test = prepare_setonet_inputs(
                sensor_x_original,
                current_eval_batch_size,
                f_at_sensors_test.unsqueeze(-1),
                batch_x_eval_points_test,
                sensor_size
            )
            pred_T_test = setonet_model_T(xs_T_test, us_T_test, ys_T_test)
            rel_err_T = calculate_l2_relative_error(pred_T_test.squeeze(-1), f_prime_at_trunk_test)
            all_rel_errors_T.append(rel_err_T.item())

            # Evaluate T_inv
            xs_T_inv_test, us_T_inv_test, ys_T_inv_test = prepare_setonet_inputs(
                batch_x_eval_points_test,
                current_eval_batch_size,
                f_prime_at_trunk_test.unsqueeze(-1),
                sensor_x_original,
                n_trunk_points_eval
            )
            pred_T_inv_test = setonet_model_T_inv(xs_T_inv_test, us_T_inv_test, ys_T_inv_test)
            rel_err_T_inv = calculate_l2_relative_error(pred_T_inv_test.squeeze(-1), f_at_sensors_test)
            all_rel_errors_T_inv.append(rel_err_T_inv.item())

    avg_rel_error_T = np.mean(all_rel_errors_T) if all_rel_errors_T else float('nan')
    avg_rel_error_T_inv = np.mean(all_rel_errors_T_inv) if all_rel_errors_T_inv else float('nan')

    print(f"SetONet T (f->f'): Average L2 Relative Error: {avg_rel_error_T:.6f}")
    print(f"SetONet T_inv (f'->f): Average L2 Relative Error: {avg_rel_error_T_inv:.6f}")
    
    return avg_rel_error_T, avg_rel_error_T_inv 