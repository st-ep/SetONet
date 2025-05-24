import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Data.data_utils import generate_batch
from .utils.helper_utils import calculate_l2_relative_error, prepare_setonet_inputs

def train_setonet_models(setonet_model_T, setonet_model_T_inv, args, training_params, device, log_dir=None):
    """
    Trains both SetONet models with cycle consistency.
    
    Args:
        setonet_model_T: Forward model (f -> f')
        setonet_model_T_inv: Inverse model (f' -> f)
        args: Command line arguments
        training_params: Dictionary with training parameters
        device: PyTorch device
        log_dir: Directory for logging (optional, for TensorBoard)
    """
    print("\n--- Training Models ---")
    
    # Extract training parameters
    sensor_x_original = training_params['sensor_x_original']
    scale = training_params['scale']
    input_range = training_params['input_range']
    batch_size_train = training_params['batch_size_train']
    n_trunk_points_train = training_params['n_trunk_points_train']
    sensor_size = training_params['sensor_size']
    
    # Initialize TensorBoard writer if log_dir is provided
    writer = None
    if log_dir:
        tensorboard_dir = f"{log_dir}/tensorboard"
        writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logging to: {tensorboard_dir}")
    
    # Combined Optimizer for both models
    optimizer = optim.Adam(
        list(setonet_model_T.parameters()) + list(setonet_model_T_inv.parameters()),
        lr=args.son_lr
    )
    
    loss_fn = torch.nn.MSELoss()

    print("\nTraining SetONet (Forward and Inverse with Cycle Consistency)...")
    epoch_pbar = tqdm(range(args.son_epochs), desc="Training Bi-Directional SetONet")
    
    for epoch in epoch_pbar:
        setonet_model_T.train()
        setonet_model_T_inv.train()

        # Generate data
        f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = generate_batch(
            batch_size=batch_size_train,
            n_trunk_points=n_trunk_points_train,
            sensor_x=sensor_x_original,
            scale=scale,
            input_range=input_range,
            device=device,
            constant_zero=True
        )
        current_batch_size = f_at_sensors.shape[0]

        # --- Forward pass for T (f -> f') ---
        xs_T, us_T, ys_T = prepare_setonet_inputs(
            sensor_x_global=sensor_x_original,
            current_batch_size=current_batch_size,
            batch_f_values_norm_expanded=f_at_sensors.unsqueeze(-1),
            batch_x_eval_norm=batch_x_eval_points,
            global_sensor_size=sensor_size
        )
        pred_T = setonet_model_T(xs_T, us_T, ys_T)
        target_T = f_prime_at_trunk.unsqueeze(-1)
        loss_F = loss_fn(pred_T, target_T)

        # --- Forward pass for T_inv (f' -> f) ---
        xs_T_inv, us_T_inv, ys_T_inv = prepare_setonet_inputs(
            sensor_x_global=batch_x_eval_points,
            current_batch_size=current_batch_size,
            batch_f_values_norm_expanded=f_prime_at_trunk.unsqueeze(-1),
            batch_x_eval_norm=sensor_x_original,
            global_sensor_size=n_trunk_points_train
        )
        pred_T_inv = setonet_model_T_inv(xs_T_inv, us_T_inv, ys_T_inv)
        target_T_inv = f_at_sensors.unsqueeze(-1)
        loss_I = loss_fn(pred_T_inv, target_T_inv)

        # --- Cycle Consistency Losses ---
        # Cycle 1: f -> T -> f_hat_prime -> T_inv -> f_hat_hat
        pred_T_detached_for_cycle1 = pred_T.detach()
        xs_cycle1, us_cycle1, ys_cycle1 = prepare_setonet_inputs(
            sensor_x_global=batch_x_eval_points,
            current_batch_size=current_batch_size,
            batch_f_values_norm_expanded=pred_T_detached_for_cycle1,
            batch_x_eval_norm=sensor_x_original,
            global_sensor_size=n_trunk_points_train
        )
        pred_cycle1 = setonet_model_T_inv(xs_cycle1, us_cycle1, ys_cycle1)
        loss_cycle1 = loss_fn(pred_cycle1, target_T_inv)

        # Cycle 2: f' -> T_inv -> f_hat -> T -> f_hat_hat_prime
        pred_T_inv_detached_for_cycle2 = pred_T_inv.detach()
        xs_cycle2, us_cycle2, ys_cycle2 = prepare_setonet_inputs(
            sensor_x_global=sensor_x_original,
            current_batch_size=current_batch_size,
            batch_f_values_norm_expanded=pred_T_inv_detached_for_cycle2,
            batch_x_eval_norm=batch_x_eval_points,
            global_sensor_size=sensor_size
        )
        pred_cycle2 = setonet_model_T(xs_cycle2, us_cycle2, ys_cycle2)
        loss_cycle2 = loss_fn(pred_cycle2, target_T)

        # --- Total Loss ---
        total_loss = loss_F + loss_I + args.lambda_cycle * (loss_cycle1 + loss_cycle2)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Learning rate scheduling
        current_iteration = epoch + 1
        if current_iteration in args.lr_schedule_steps:
            milestone_idx = args.lr_schedule_steps.index(current_iteration)
            gamma = args.lr_schedule_gammas[milestone_idx]
            
            old_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
            new_lr = optimizer.param_groups[0]['lr']
            print(f"\nIteration {current_iteration}: LR decayed from {old_lr:.2e} to {new_lr:.2e} (factor {gamma}).")

        # Calculate L2 errors for progress bar and logging
        rel_l2_error_T_batch = calculate_l2_relative_error(pred_T.squeeze(-1), f_prime_at_trunk)
        rel_l2_error_T_inv_batch = calculate_l2_relative_error(pred_T_inv.squeeze(-1), f_at_sensors)
        rel_l2_error_cycle1_batch = calculate_l2_relative_error(pred_cycle1.squeeze(-1), f_at_sensors)
        rel_l2_error_cycle2_batch = calculate_l2_relative_error(pred_cycle2.squeeze(-1), f_prime_at_trunk)

        # TensorBoard logging
        if writer and epoch % 100 == 0:  # Log every 100 epochs to avoid too much data
            # Loss components
            writer.add_scalar('Loss/Forward_T', loss_F.item(), epoch)
            writer.add_scalar('Loss/Inverse_T_inv', loss_I.item(), epoch)
            writer.add_scalar('Loss/Cycle_1', loss_cycle1.item(), epoch)
            writer.add_scalar('Loss/Cycle_2', loss_cycle2.item(), epoch)
            writer.add_scalar('Loss/Total', total_loss.item(), epoch)
            
            # L2 relative errors
            writer.add_scalar('L2_Error/Forward_T', rel_l2_error_T_batch.item(), epoch)
            writer.add_scalar('L2_Error/Inverse_T_inv', rel_l2_error_T_inv_batch.item(), epoch)
            writer.add_scalar('L2_Error/Cycle_1', rel_l2_error_cycle1_batch.item(), epoch)
            writer.add_scalar('L2_Error/Cycle_2', rel_l2_error_cycle2_batch.item(), epoch)
            
            # Combined metrics
            total_l2_error = rel_l2_error_T_batch + rel_l2_error_T_inv_batch + rel_l2_error_cycle1_batch + rel_l2_error_cycle2_batch
            writer.add_scalar('L2_Error/Total', total_l2_error.item(), epoch)

        epoch_pbar.set_postfix(
            L2_T=f"{rel_l2_error_T_batch.item():.3e}",
            L2_Tinv=f"{rel_l2_error_T_inv_batch.item():.3e}",
            L2_Cyc1=f"{rel_l2_error_cycle1_batch.item():.3e}",
            L2_Cyc2=f"{rel_l2_error_cycle2_batch.item():.3e}",
            L2_Tot=f"{(rel_l2_error_T_batch + rel_l2_error_T_inv_batch + rel_l2_error_cycle1_batch + rel_l2_error_cycle2_batch).item():.3e}"
        )
    
    # Close TensorBoard writer
    if writer:
        writer.close()
        print(f"\nTensorBoard logs saved. To view, run: tensorboard --logdir={tensorboard_dir}") 