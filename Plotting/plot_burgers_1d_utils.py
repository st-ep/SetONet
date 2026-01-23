import torch
import matplotlib.pyplot as plt
import os
import numpy as np


def interpolate_to_sensors(data_grid, grid_points, sensor_x):
    """
    Interpolate values from grid to sensor locations using linear interpolation.

    Args:
        data_grid: (n_samples, n_grid) or (n_grid,) grid values
        grid_points: (n_grid,) grid coordinates
        sensor_x: (n_sensors, 1) sensor locations

    Returns:
        interpolated: (n_samples, n_sensors) or (n_sensors,) interpolated values
    """
    # Handle 1D case
    if data_grid.dim() == 1:
        data_grid = data_grid.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    device = data_grid.device
    grid_points = grid_points.to(device)
    sensor_x = sensor_x.to(device)

    n_grid = data_grid.shape[1]
    sensor_x_flat = sensor_x.squeeze(-1)

    # Find surrounding grid points using binary search
    indices = torch.searchsorted(grid_points, sensor_x_flat)
    indices = torch.clamp(indices, 1, n_grid - 1)

    # Get left and right indices
    idx_left = indices - 1
    idx_right = indices

    # Get x coordinates
    x_left = grid_points[idx_left]
    x_right = grid_points[idx_right]

    # Compute interpolation weights
    weights = (sensor_x_flat - x_left) / (x_right - x_left + 1e-10)
    weights = weights.clamp(0.0, 1.0)

    # Interpolate
    y_left = data_grid[:, idx_left]
    y_right = data_grid[:, idx_right]
    weights_broadcast = weights.unsqueeze(0)
    interpolated = y_left + weights_broadcast * (y_right - y_left)

    if squeeze_output:
        interpolated = interpolated.squeeze(0)

    return interpolated


def plot_burgers_comparison(
    model_to_use,
    dataset,
    sensor_x,
    query_x,
    sensor_indices,
    query_indices,
    log_dir,
    num_samples_to_plot=3,
    plot_filename_prefix="burgers_comparison",
    sensor_dropoff=0.0,
    replace_with_nearest=False,
    dataset_split="test",
    batch_size=32,
    variable_sensors=False,
    grid_points=None,
    sensors_to_plot_fraction=0.1,
    stats=None,
    start_index=0
):
    """
    Plot comparison between true and predicted solutions for Burgers 1D problem.
    
    Args:
        model_to_use: Trained model (SetONet, DeepONet, or VIDON)
        dataset: Burgers dataset containing initial conditions and solutions
        sensor_x: Sensor locations [n_sensors, 1]
        query_x: Query locations [n_queries, 1] 
        sensor_indices: Indices of sensor locations in grid
        query_indices: Indices of query locations in grid
        log_dir: Directory to save plots
        num_samples_to_plot: Number of samples to plot
        plot_filename_prefix: Prefix for saved plot files
        sensor_dropoff: Sensor dropout rate for evaluation
        replace_with_nearest: Whether to replace dropped sensors with nearest
        dataset_split: Which dataset split to use ("train" or "test")
        batch_size: Batch size (not used, kept for compatibility)
        variable_sensors: Whether using variable sensors
        grid_points: Grid points tensor
        sensors_to_plot_fraction: Fraction of sensors to show as markers
        stats: Normalization statistics for denormalization
        start_index: Starting index in the dataset split (default 0)
    """
    print(f"Generating Burgers 1D comparison plots...")
    
    if not model_to_use:
        print("No model provided. Skipping plots.")
        return
        
    model_to_use.eval()
    device = next(model_to_use.parameters()).device
    
    # Get data split
    data_split = dataset[dataset_split]
    
    # Set larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'axes.titlesize': 18
    })
    
    def denormalize(data, mean, std):
        """Denormalize data for plotting."""
        if stats is None:
            return data
        return data * std + mean
    
    for i in range(num_samples_to_plot):
        sample_idx = start_index + i
        if sample_idx >= len(data_split):
            break
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Get sample data
        sample = data_split[sample_idx]

        # Extract initial condition (normalized) and solution (normalized)
        u_full = torch.tensor(sample['u'], device=device, dtype=torch.float32)  # Normalized initial condition
        s_full = torch.tensor(sample['s'], device=device, dtype=torch.float32)  # Normalized solution

        # Get sensor data - handle both direct indexing and interpolation
        if sensor_indices is not None:
            # Direct indexing (sensor_size <= grid_size)
            u_sensors = u_full[sensor_indices]  # Initial condition at sensor locations
        else:
            # Interpolation (sensor_size > grid_size)
            u_sensors = interpolate_to_sensors(u_full.unsqueeze(0), grid_points, sensor_x).squeeze(0)

        # Get query data
        s_queries_true = s_full[query_indices]  # True solution at query points
        
        # Convert grid points to CPU for plotting
        if grid_points is not None:
            x_plot = grid_points.cpu().numpy()
        else:
            x_plot = np.linspace(0, 1, len(u_full))
        
        # Denormalize for plotting
        if stats is not None:
            u_full_denorm = denormalize(u_full, stats['u_mean'], stats['u_std']).cpu().numpy()
            s_full_denorm = denormalize(s_full, stats['s_mean'], stats['s_std']).cpu().numpy()
        else:
            u_full_denorm = u_full.cpu().numpy()
            s_full_denorm = s_full.cpu().numpy()
            
        # Apply sensor dropout if specified
        if sensor_dropoff > 0.0:
            from Data.data_utils import apply_sensor_dropoff
            
            # Apply dropout to sensor data
            sensor_x_dropped, u_sensors_dropped = apply_sensor_dropoff(
                sensor_x, u_sensors, sensor_dropoff, replace_with_nearest
            )
            
            sensor_x_plot = sensor_x_dropped.squeeze().cpu()
            u_sensors_plot = u_sensors_dropped.cpu()
            if stats is not None:
                u_sensors_plot_denorm = denormalize(u_sensors_plot, stats['u_mean'], stats['u_std']).numpy()
            else:
                u_sensors_plot_denorm = u_sensors_plot.numpy()
            sensor_x_model = sensor_x_dropped.unsqueeze(0)  # Add batch dim
            u_sensors_model = u_sensors_dropped.unsqueeze(0).unsqueeze(-1)  # [1, n_sensors, 1]
            
        else:
            sensor_x_plot = sensor_x.squeeze().cpu() 
            u_sensors_plot = u_sensors.cpu()
            if stats is not None:
                u_sensors_plot_denorm = denormalize(u_sensors_plot, stats['u_mean'], stats['u_std']).numpy()
            else:
                u_sensors_plot_denorm = u_sensors_plot.numpy()
            sensor_x_model = sensor_x.unsqueeze(0)  # Add batch dim
            u_sensors_model = u_sensors.unsqueeze(0).unsqueeze(-1)  # [1, n_sensors, 1]
        
        # Left plot: Initial condition with sensor locations
        axs[0].plot(x_plot, u_full_denorm, 'b-', linewidth=2, label='Initial condition u(x,0)')
        
        # Show sensor locations (subsample for clarity)
        n_sensors_to_show = max(1, int(len(sensor_x_plot) * sensors_to_plot_fraction))
        sensor_step = max(1, len(sensor_x_plot) // n_sensors_to_show)
        sensor_indices_to_show = range(0, len(sensor_x_plot), sensor_step)
        
        axs[0].scatter(
            sensor_x_plot[sensor_indices_to_show], 
            u_sensors_plot_denorm[sensor_indices_to_show],
            c='red', s=50, zorder=5,
            label=f'Sensors ({len(sensor_indices_to_show)} shown of {len(sensor_x_plot)})'
        )
        
        # Add sensor dropout info if applicable
        if sensor_dropoff > 0:
            # Use sensor_x length instead of sensor_indices (which can be None for interpolation)
            original_count = len(sensor_x)
            dropout_info = f'Dropout: {sensor_dropoff:.1%} ({original_count}→{len(sensor_x_plot)})'
            if replace_with_nearest:
                dropout_info += ' (nearest replacement)'
            axs[0].text(0.02, 0.98, dropout_info, transform=axs[0].transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('u(x,0)')
        axs[0].set_title('Initial Condition')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
        
        # Get model prediction
        with torch.no_grad():
            query_x_model = query_x.unsqueeze(0)  # Add batch dim [1, n_queries, 1]
            
            # Model forward pass
            s_pred = model_to_use(sensor_x_model, u_sensors_model, query_x_model)
            s_pred = s_pred.squeeze().cpu()  # Remove batch and feature dims
            
            # Denormalize prediction for plotting
            if stats is not None:
                s_pred_denorm = denormalize(s_pred, stats['s_mean'], stats['s_std']).numpy()
            else:
                s_pred_denorm = s_pred.numpy()
        
        # Right plot: True vs predicted solution
        query_x_plot = query_x.squeeze().cpu().numpy()
        s_true_plot = s_queries_true.cpu()
        if stats is not None:
            s_true_plot_denorm = denormalize(s_true_plot, stats['s_mean'], stats['s_std']).numpy()
        else:
            s_true_plot_denorm = s_true_plot.numpy()
        
        axs[1].plot(query_x_plot, s_true_plot_denorm, 'g-', linewidth=2, label='True solution u(x,T)')
        axs[1].plot(query_x_plot, s_pred_denorm, 'r--', linewidth=2, label='Model prediction')
        
        # Calculate and display error (on denormalized values)
        mse = np.mean((s_true_plot_denorm - s_pred_denorm)**2)
        rel_error = np.linalg.norm(s_true_plot_denorm - s_pred_denorm) / np.linalg.norm(s_true_plot_denorm)
        
        error_text = f'MSE: {mse:.2e}\nRel Error: {rel_error:.4f}'
        axs[1].text(0.02, 0.98, error_text, transform=axs[1].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('u(x,T)')
        axs[1].set_title('Solution at Final Time')
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
        
        plt.tight_layout()
        
        # Save plot
        dropoff_suffix = f"_dropoff_{sensor_dropoff:.1f}" if sensor_dropoff > 0 else ""
        replacement_suffix = "_nearest" if replace_with_nearest and sensor_dropoff > 0 else ""
        save_path = os.path.join(log_dir, f"{plot_filename_prefix}_{i+1}{dropoff_suffix}{replacement_suffix}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Burgers comparison plot for sample {i+1} to {save_path}")
        plt.close(fig)
    
    # Reset font parameters to default
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10,
        'axes.titlesize': 12
    })
    
    print(f"Generated {min(num_samples_to_plot, len(data_split))} Burgers comparison plots.")

