import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Data.data_utils import generate_batch, apply_sensor_dropoff
from .utils.helper_utils import calculate_l2_relative_error, prepare_setonet_inputs, prepare_setonet_inputs_variable, prepare_setonet_inputs_variable_integral
from .utils.model_utils import print_model_summary, count_parameters

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
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points, sensor_x_batch = batch_data
            sensor_x_to_use = sensor_x_batch
        else:
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = generate_batch(
                batch_size=batch_size_train,
                n_trunk_points=n_trunk_points_train,
                sensor_x=sensor_x_original,
                scale=scale,
                input_range=input_range,
                device=device,
                constant_zero=True
            )
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

        # TensorBoard logging
        if writer and epoch % 100 == 0:  # Log every 100 epochs to avoid too much data
            writer.add_scalar('Loss/Training', loss.item(), epoch)
            writer.add_scalar('L2_Error/Training', rel_l2_error.item(), epoch)
            writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

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