import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_dynamic_chladni_results(model, dataset, chladni_dataset, device, sample_idx=0, save_path=None):
    """Plot input forces, predicted displacements, and ground truth for Dynamic Chladni plate."""
    model.eval()
    test_data = dataset['test']
    
    if sample_idx >= len(test_data):
        print(f"Warning: Sample index {sample_idx} out of range. Using sample 0.")
        sample_idx = 0
    
    sample = test_data[sample_idx]
    
    # Extract data from the raw sample
    sources = np.array(sample['sources'])  # (n_forces, 3) - [x_norm, y_norm, force_mag]
    displacement_field = np.array(sample['field'])  # (grid_size, grid_size, 1)
    
    # Get grid size
    grid_size = displacement_field.shape[0]
    
    # Create grid coordinates (normalized [0,1])
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    grid_coords = np.stack([xx, yy], axis=-1)  # (grid_size, grid_size, 2)
    
    # Convert to tensors and add batch dimension
    source_coords = torch.tensor(sources[:, :2], dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_forces, 2)
    source_forces = torch.tensor(sources[:, 2:3], dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_forces, 1)
    
    # Flatten grid coordinates
    target_coords = torch.tensor(grid_coords.reshape(-1, 2), dtype=torch.float32, device=device).unsqueeze(0)  # (1, n_grid_points, 2)
    
    # Get prediction
    with torch.no_grad():
        pred_displacements = model(source_coords, source_forces, target_coords)
        pred_2d = pred_displacements.squeeze(0).squeeze(-1).cpu().numpy().reshape(grid_size, grid_size)
    
    # Ground truth displacement field
    target_2d = displacement_field[:, :, 0]  # Remove channel dimension
    
    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input forces (scatter plot since forces are at specific locations)
    axes[0].scatter(sources[:, 0], sources[:, 1], c=sources[:, 2], cmap='RdBu_r', s=100, alpha=0.8)
    axes[0].set_title(f'Input Forces - Sample {sample_idx}')
    axes[0].set_xlabel('X (normalized)')
    axes[0].set_ylabel('Y (normalized)')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Add colorbar for force magnitudes
    scatter = axes[0].scatter(sources[:, 0], sources[:, 1], c=sources[:, 2], cmap='RdBu_r', s=100, alpha=0.8)
    plt.colorbar(scatter, ax=axes[0], label='Force Magnitude')
    
    # Predicted displacements
    im2 = axes[1].contourf(xx, yy, pred_2d, levels=20, cmap='viridis')
    axes[1].set_title(f'Predicted Displacements - Sample {sample_idx}')
    axes[1].set_xlabel('X (normalized)')
    axes[1].set_ylabel('Y (normalized)')
    plt.colorbar(im2, ax=axes[1], label='Displacement')
    
    # Ground truth displacements
    im3 = axes[2].contourf(xx, yy, target_2d, levels=20, cmap='viridis')
    axes[2].set_title('Ground Truth Displacements')
    axes[2].set_xlabel('X (normalized)')
    axes[2].set_ylabel('Y (normalized)')
    plt.colorbar(im3, ax=axes[2], label='Displacement')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Error metrics
    mse_error = np.mean((pred_2d - target_2d)**2)
    rel_error = np.linalg.norm(pred_2d - target_2d) / np.linalg.norm(target_2d)
    
    print(f"Sample {sample_idx} - MSE: {mse_error:.6e}, Rel Error: {rel_error:.6f}")
    print(f"Force locations: {len(sources)} forces")
    print(f"Force magnitude range: [{sources[:, 2].min():.4f}, {sources[:, 2].max():.4f}]")
    print(f"Displacement range: [{target_2d.min():.4f}, {target_2d.max():.4f}]")
    
    model.train()

def plot_multiple_dynamic_chladni_samples(model, dataset, chladni_dataset, device, n_samples=3, save_dir=None):
    """Plot multiple samples to show variety in the dynamic chladni dataset."""
    test_data = dataset['test']
    n_samples = min(n_samples, len(test_data))
    
    for i in range(n_samples):
        save_path = None
        if save_dir:
            save_path = f"{save_dir}/dynamic_chladni_sample_{i}.png"
        
        print(f"\n--- Sample {i} ---")
        plot_dynamic_chladni_results(model, dataset, chladni_dataset, device, 
                                   sample_idx=i, save_path=save_path)
