import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Data.data_utils import generate_batch, apply_sensor_dropoff
from .utils.helper_utils import calculate_l2_relative_error, prepare_setonet_inputs, prepare_setonet_inputs_variable, prepare_setonet_inputs_variable_integral
from .utils.model_utils import print_model_summary, count_parameters


def _evaluate_test_data(model, benchmark, sensor_x_original, scale, input_range, 
                       batch_size, n_trunk_points, sensor_size, device, variable_sensors,
                       eval_sensor_dropoff=0.0, replace_with_nearest=False):
    """Evaluate model on test data and return average relative L2 error."""
    model.eval()
    
    n_test_batches = 10  # Evaluate on 10 batches for quick assessment
    total_rel_error = 0.0
    
    with torch.no_grad():
        for _ in range(n_test_batches):
            # Generate test batch - handle tuple unpacking safely
            batch_data = generate_batch(
                batch_size=batch_size,
                n_trunk_points=n_trunk_points,
                sensor_x=None if variable_sensors else sensor_x_original,
                scale=scale,
                input_range=input_range,
                device=device,
                constant_zero=True,
                variable_sensors=variable_sensors,
                sensor_size=sensor_size if variable_sensors else None
            )
            
            # Safely unpack based on variable_sensors flag
            if variable_sensors:
                # Variable sensors: expect 6 return values
                f_at_sensors, _, f_at_trunk, f_prime_at_trunk, batch_x_eval_points, sensor_x_batch = batch_data
                sensor_x_to_use = sensor_x_batch
            else:
                # Fixed sensors: expect 5 return values
                f_at_sensors, _, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = batch_data
                sensor_x_to_use = sensor_x_original
            
            # Apply sensor dropout if specified (same as final evaluation)
            sensor_x_used = sensor_x_to_use
            f_at_sensors_used = f_at_sensors
            actual_sensor_size = len(sensor_x_to_use)
            
            if eval_sensor_dropoff > 0.0:
                if benchmark == 'derivative':
                    sensor_x_dropped, f_at_sensors_dropped = apply_sensor_dropoff(
                        sensor_x_to_use, f_at_sensors, eval_sensor_dropoff, replace_with_nearest
                    )
                    sensor_x_used = sensor_x_dropped
                    f_at_sensors_used = f_at_sensors_dropped
                    actual_sensor_size = len(sensor_x_dropped)
                elif benchmark == 'integral':
                    # For integral task, we need to drop from trunk points and values
                    batch_x_eval_points_dropped, f_prime_at_trunk_dropped = apply_sensor_dropoff(
                        batch_x_eval_points, f_prime_at_trunk.T, eval_sensor_dropoff, replace_with_nearest
                    )
                    f_prime_at_trunk = f_prime_at_trunk_dropped.T
                    batch_x_eval_points = batch_x_eval_points_dropped
            
            # Prepare inputs and get predictions based on benchmark
            if benchmark == 'derivative':
                from .utils.helper_utils import prepare_setonet_inputs
                xs, us, ys = prepare_setonet_inputs(
                    sensor_x_used,
                    batch_size,
                    f_at_sensors_used.unsqueeze(-1),
                    batch_x_eval_points,
                    actual_sensor_size
                )
                pred = model(xs, us, ys).squeeze(-1)
                target = f_prime_at_trunk.T
            else:  # integral
                from .utils.helper_utils import prepare_setonet_inputs
                xs, us, ys = prepare_setonet_inputs(
                    batch_x_eval_points,
                    batch_size,
                    f_prime_at_trunk.T.unsqueeze(-1),
                    sensor_x_used,
                    n_trunk_points
                )
                pred = model(xs, us, ys).squeeze(-1)
                target = f_at_sensors_used
            
            # Calculate relative error
            rel_error = calculate_l2_relative_error(pred, target)
            total_rel_error += rel_error.item()
    
    model.train()
    return total_rel_error / n_test_batches


def train_setonet_model(setonet_model, args, training_params, device, log_dir=None):
    """
    Trains a single SetONet model for the specified benchmark.
    
    Args:
        setonet_model: The SetONet model to train
        args: Command line arguments
        training_params: Dictionary with training parameters including 'benchmark'
        device: PyTorch device
        log_dir: Directory for logging (optional, for TensorBoard)
    """
    print(f"\n--- Training {training_params['benchmark'].title()} Model ---")
    
    # Print model summary with parameter counts
    print_model_summary(setonet_model, f"SetONet ({training_params['benchmark']})")
    
    # Extract training parameters
    sensor_x_original = training_params['sensor_x_original']
    scale = training_params['scale']
    input_range = training_params['input_range']
    batch_size_train = training_params['batch_size_train']
    n_trunk_points_train = training_params['n_trunk_points_train']
    sensor_size = training_params['sensor_size']
    benchmark = training_params['benchmark']
    variable_sensors = training_params.get('variable_sensors', False)
    
    # Initialize TensorBoard writer if log_dir is provided
    writer = None
    if log_dir:
        tensorboard_dir = f"{log_dir}/tensorboard"
        writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard logging to: {tensorboard_dir}")
    
    # Optimizer
    optimizer = optim.Adam(setonet_model.parameters(), lr=args.son_lr)
    loss_fn = torch.nn.MSELoss()

    print(f"\nTraining SetONet for {benchmark} task...")
    print("Training with full sensor data (no drop-off)")
    epoch_pbar = tqdm(range(args.son_epochs), desc=f"Training SetONet ({benchmark})")
    
    for epoch in epoch_pbar:
        setonet_model.train()

        # Generate batch with variable or fixed sensors
        if variable_sensors:
            batch_data = generate_batch(
                batch_size=batch_size_train,
                n_trunk_points=n_trunk_points_train,
                sensor_x=None,
                scale=scale,
                input_range=input_range,
                device=device,
                constant_zero=True,
                variable_sensors=True,
                sensor_size=sensor_size
            )
            # Variable sensors return 6 values
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points, sensor_x_batch = batch_data
            sensor_x_to_use = sensor_x_batch
        else:
            batch_data = generate_batch(
                batch_size=batch_size_train,
                n_trunk_points=n_trunk_points_train,
                sensor_x=sensor_x_original,
                scale=scale,
                input_range=input_range,
                device=device,
                constant_zero=True
            )
            # Fixed sensors return 5 values
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = batch_data
            sensor_x_to_use = sensor_x_original
        
        # NO sensor drop-off during training - use full sensor data
        sensor_x_used = sensor_x_to_use
        f_at_sensors_used = f_at_sensors
        actual_sensor_size = sensor_size
        
        # Prepare inputs and get predictions based on benchmark
        if benchmark == 'derivative':
            # Use individual processing for derivative task if needed
            xs, us, ys = prepare_setonet_inputs(
                sensor_x_used,
                batch_size_train,
                f_at_sensors_used.unsqueeze(-1),
                batch_x_eval_points,
                actual_sensor_size
            )
            pred = setonet_model(xs, us, ys)
            pred = pred.squeeze(-1)  # [batch_size, n_trunk_points]
            
            target = f_prime_at_trunk.T  # [batch_size, n_trunk_points]
            
        elif benchmark == 'integral':
            # Use regular batched processing since all samples in batch have same sensors
            xs, us, ys = prepare_setonet_inputs(
                batch_x_eval_points,
                batch_size_train,
                f_prime_at_trunk.T.unsqueeze(-1),
                sensor_x_used,
                n_trunk_points_train
            )
            pred = setonet_model(xs, us, ys)
            pred = pred.squeeze(-1)  # [batch_size, actual_sensor_size]
            
            target = f_at_sensors_used  # [batch_size, actual_sensor_size]
        
        # Compute loss
        loss = loss_fn(pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
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

        # Calculate L2 error for progress bar and logging
        if benchmark == 'derivative':
            rel_l2_error = calculate_l2_relative_error(pred, f_prime_at_trunk.T)
        else:  # integral
            rel_l2_error = calculate_l2_relative_error(pred, f_at_sensors_used)

        # TensorBoard logging - enhanced for better visualization
        if writer:
            # Log training metrics more frequently for smooth curves
            if epoch % 10 == 0:  # Log every 10 epochs for smoother curves
                writer.add_scalar('Training/MSE_Loss', loss.item(), epoch)
                writer.add_scalar('Training/Relative_L2_Error', rel_l2_error.item(), epoch)
                writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                
                # Log gradient norm for training monitoring
                total_norm = 0
                for p in setonet_model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                writer.add_scalar('Training/Gradient_Norm', total_norm, epoch)
                
                # Log additional training statistics
                writer.add_scalar('Training/Effective_Sensors', actual_sensor_size, epoch)
                
                # Log target and prediction statistics for monitoring
                if benchmark == 'derivative':
                    target_stats = f_prime_at_trunk.T
                else:
                    target_stats = f_at_sensors_used
                
                writer.add_scalar('Training/Target_Mean', target_stats.mean().item(), epoch)
                writer.add_scalar('Training/Target_Std', target_stats.std().item(), epoch)
                writer.add_scalar('Training/Prediction_Mean', pred.mean().item(), epoch)
                writer.add_scalar('Training/Prediction_Std', pred.std().item(), epoch)
            
            # Periodic evaluation on test data (every 1000 epochs, similar to TensorBoard callback)
            if epoch % 1000 == 0 and epoch > 0:
                try:
                    # Extract evaluation parameters from training params
                    eval_sensor_dropoff = training_params.get('eval_sensor_dropoff', 0.0)
                    replace_with_nearest = training_params.get('replace_with_nearest', False)
                    
                    test_rel_l2 = _evaluate_test_data(
                        setonet_model, benchmark, sensor_x_original, scale, input_range,
                        batch_size_train, n_trunk_points_train, sensor_size, device, variable_sensors,
                        eval_sensor_dropoff, replace_with_nearest
                    )
                    writer.add_scalar('Evaluation/Test_Relative_L2_Error', test_rel_l2, epoch)
                    # Silent logging to avoid interfering with progress bars
                except Exception:
                    # Silent failure - continue training without test evaluation
                    pass

        progress_info = {
            "Loss": f"{loss.item():.3e}",
            "L2_Error": f"{rel_l2_error.item():.3e}",
            "Sensors": f"{actual_sensor_size}"
        }
        
        epoch_pbar.set_postfix(progress_info)
    
    # Close TensorBoard writer
    if writer:
        writer.close()
        print(f"\nTensorBoard logs saved. To view, run: tensorboard --logdir={tensorboard_dir}") 