import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_chladni_results(model, dataset, chladni_dataset, device, sample_idx=0, save_path=None, eval_sensor_dropoff=0.0, replace_with_nearest=False):
    """Plot input forces, predicted displacements, and ground truth for Chladni plate."""
    model.eval()
    test_data = dataset['test']
    sample = test_data[sample_idx]
    
    # Load pre-normalized data
    xs_norm = torch.tensor(sample['X'], dtype=torch.float32, device=device)
    xs = xs_norm.unsqueeze(0)
    
    us_norm = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)
    us = us_norm.unsqueeze(-1)
    
    ys_norm = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
    ys = ys_norm.unsqueeze(0)
    
    target_norm = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)
    
    # Apply sensor dropout if specified
    xs_used = xs
    us_used = us
    n_sensors_orig = xs.shape[1]
    dropout_info = ""
    
    if eval_sensor_dropoff > 0.0:
        from Data.data_utils import apply_sensor_dropoff
        
        # Apply dropout to sensor data (remove batch dimension for dropout function)
        xs_dropped, us_dropped = apply_sensor_dropoff(
            xs.squeeze(0),  # Remove batch dimension: (n_sensors, 2)
            us.squeeze(0).squeeze(-1),  # Remove batch and feature dimensions: (n_sensors,)
            eval_sensor_dropoff,
            replace_with_nearest
        )
        
        # Add batch dimension back
        xs_used = xs_dropped.unsqueeze(0)  # (1, n_remaining_sensors, 2)
        us_used = us_dropped.unsqueeze(0).unsqueeze(-1)  # (1, n_remaining_sensors, 1)
        
        n_sensors_remaining = xs_used.shape[1]
        replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
        dropout_info = f" (Dropout: {eval_sensor_dropoff:.1%}, {n_sensors_remaining}/{n_sensors_orig} sensors, {replacement_mode})"
    
    # Get prediction and denormalize
    with torch.no_grad():
        pred_norm = model(xs_used, us_used, ys)
        pred_orig = chladni_dataset.denormalize_displacement(pred_norm.squeeze(-1))
    
    # Use original data if available, otherwise denormalize
    if 'u_orig' in sample:
        forces_orig = torch.tensor(sample['u_orig'], dtype=torch.float32, device=device)
        target_orig = torch.tensor(sample['s_orig'], dtype=torch.float32, device=device)
        coords_orig = torch.tensor(sample['X_orig'], dtype=torch.float32, device=device)
    else:
        forces_orig = chladni_dataset.denormalize_force(us_norm.squeeze(0))
        target_orig = chladni_dataset.denormalize_displacement(target_norm.squeeze(0))
        coords_orig = chladni_dataset.denormalize_coordinates(xs_norm)
    
    # Reshape to 2D grid
    grid_size = int(np.sqrt(len(sample['X'])))
    coords = coords_orig.cpu().numpy()
    x_coords = coords[:, 0].reshape(grid_size, grid_size)
    y_coords = coords[:, 1].reshape(grid_size, grid_size)
    
    forces_2d = forces_orig.cpu().numpy().reshape(grid_size, grid_size)
    pred_2d = pred_orig.cpu().numpy().reshape(grid_size, grid_size)
    target_2d = target_orig.cpu().numpy().reshape(grid_size, grid_size)
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input forces
    im1 = axes[0].contourf(x_coords, y_coords, forces_2d, levels=20, cmap='RdBu_r')
    axes[0].set_title(f'Input Forces - Sample {sample_idx}{dropout_info}')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    plt.colorbar(im1, ax=axes[0])
    
    # Predicted displacements
    im2 = axes[1].contourf(x_coords, y_coords, pred_2d, levels=20, cmap='viridis')
    axes[1].set_title(f'Predicted Displacements{dropout_info}')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    plt.colorbar(im2, ax=axes[1])
    
    # Ground truth displacements
    im3 = axes[2].contourf(x_coords, y_coords, target_2d, levels=20, cmap='viridis')
    axes[2].set_title('Ground Truth Displacements')
    axes[2].set_xlabel('X (m)')
    axes[2].set_ylabel('Y (m)')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Error metrics
    mse_error = np.mean((pred_2d - target_2d)**2)
    rel_error = np.linalg.norm(pred_2d - target_2d) / np.linalg.norm(target_2d)
    
    if eval_sensor_dropoff > 0.0:
        replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
        print(f"Sample {sample_idx} with {eval_sensor_dropoff:.1%} sensor dropout ({replacement_mode}) - MSE: {mse_error:.6e}, Rel Error: {rel_error:.6f}")
    else:
        print(f"Sample {sample_idx} - MSE: {mse_error:.6e}, Rel Error: {rel_error:.6f}")
    
    model.train()
