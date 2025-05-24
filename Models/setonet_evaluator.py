import torch
import numpy as np
from tqdm import tqdm
from Data.data_utils import generate_batch
from .utils.helper_utils import calculate_l2_relative_error, prepare_setonet_inputs

def evaluate_setonet_models(setonet_model_T, setonet_model_T_inv, eval_params, device):
    """
    Evaluates both SetONet models on test data.
    
    Returns:
        tuple: (avg_rel_error_T, avg_rel_error_T_inv)
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