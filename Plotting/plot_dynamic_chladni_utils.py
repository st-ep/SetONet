import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def plot_dynamic_chladni_results(model, dataset, chladni_dataset, device, sample_idx=0, save_path=None):
    """Plot input forces, predicted displacements, and ground truth for Dynamic Chladni plate."""
    model.eval()
    test_data = dataset['test']
    
    if sample_idx >= len(test_data):
        print(f"Warning: Sample index {sample_idx} out of range. Using sample 0.")
        sample_idx = 0
    
    sample = test_data[sample_idx]
    
    # Extract data from the raw sample (normalized)
    sources = np.array(sample['sources'])  # (n_forces, 3) - [x_norm, y_norm, force_mag_norm]
    displacement_field = np.array(sample['field'])  # (grid_size, grid_size, 1) - normalized
    
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
    
    # Get prediction (normalized)
    with torch.no_grad():
        pred_displacements_norm = model(source_coords, source_forces, target_coords)
        pred_2d_norm = pred_displacements_norm.squeeze(0).squeeze(-1).reshape(grid_size, grid_size)
    
    # Ground truth displacement field (normalized)
    target_2d_norm = torch.tensor(displacement_field[:, :, 0], dtype=torch.float32, device=device)
    
    # Denormalize predictions and ground truth to physical units
    pred_2d = chladni_dataset.denormalize_displacement(pred_2d_norm).cpu().numpy()
    target_2d = chladni_dataset.denormalize_displacement(target_2d_norm).cpu().numpy()
    
    # Calculate absolute error (in physical units)
    abs_error = np.abs(pred_2d - target_2d)
    
    # Denormalize coordinates to physical space
    # Create coordinate tensors for denormalization
    coords_norm = torch.tensor(grid_coords.reshape(-1, 2), dtype=torch.float32, device=device)
    coords_phys = chladni_dataset.denormalize_coordinates(coords_norm).cpu().numpy()
    coords_phys = coords_phys.reshape(grid_size, grid_size, 2)
    xx_phys = coords_phys[:, :, 0]
    yy_phys = coords_phys[:, :, 1]
    
    # Denormalize source positions and forces to physical units
    source_coords_tensor = torch.tensor(sources[:, :2], dtype=torch.float32, device=device)
    sources_coords_phys = chladni_dataset.denormalize_coordinates(source_coords_tensor).cpu().numpy()
    sources_x_phys = sources_coords_phys[:, 0]
    sources_y_phys = sources_coords_phys[:, 1]
    
    # Denormalize force magnitudes
    forces_tensor = torch.tensor(sources[:, 2], dtype=torch.float32, device=device)
    forces_phys = chladni_dataset.denormalize_force(forces_tensor).cpu().numpy()
    
    # Compute shared vmin/vmax for prediction and ground truth displacements
    vmin_shared = min(pred_2d.min(), target_2d.min())
    vmax_shared = max(pred_2d.max(), target_2d.max())
    
    # Compute scaling orders for colorbars (using physical units now)
    forces_vabs = max(abs(forces_phys.min()), abs(forces_phys.max()))
    forces_order = int(np.floor(np.log10(forces_vabs))) if forces_vabs > 0 else 0
    
    shared_vabs = max(abs(vmin_shared), abs(vmax_shared))
    shared_order = int(np.floor(np.log10(shared_vabs))) if shared_vabs > 0 else 0
    
    error_vabs = np.max(abs_error)
    error_order = int(np.floor(np.log10(error_vabs))) if error_vabs > 0 else 0
    
    # Create figure with GridSpec for custom widths and variable spacing
    fig = plt.figure(figsize=(28, 6))
    plot_ratios = [1, 1, 1, 1]  # Equal ratios for all four plots
    has_colorbar = [True, False, True, True]  # Input Forces: yes, Predicted: no, Ground Truth: yes, Error: yes
    cb_ratio = 0.05  # Colorbar width ratio
    cb_spacer = 0.001  # Small spacer between plot and its colorbar
    wspaces = [0.2, 0.05, 0.2]  # Custom spaces: normal between 1-2, tight between 2-3, normal between 3-4
    
    # Build full ratio list including spaces and colorbars
    num_plots = len(plot_ratios)
    avg_width = sum(plot_ratios) / num_plots
    spacer_widths = [w * avg_width for w in wspaces]
    full_ratios = []
    col_starts = []
    current_col = 0
    
    for i in range(num_plots):
        if i > 0:
            full_ratios.append(spacer_widths[i-1])
            current_col += 1
        full_ratios.append(plot_ratios[i])
        col_starts.append(current_col)
        current_col += 1
        if has_colorbar[i]:
            # Add small spacer before colorbar
            full_ratios.append(cb_spacer)
            current_col += 1
            full_ratios.append(cb_ratio)
            current_col += 1
    
    gs = fig.add_gridspec(1, len(full_ratios), width_ratios=full_ratios, wspace=0)
    axes = [fig.add_subplot(gs[0, col_starts[j]]) for j in range(num_plots)]
    
    # Create colorbar axes where needed
    cb_axes = {}
    for i in range(num_plots):
        if has_colorbar[i]:
            cb_col = col_starts[i] + 2  # +2 because of the spacer
            cb_axes[i] = fig.add_subplot(gs[0, cb_col])
    
    # Apply scaling to data (now using physical units)
    forces_scaled = forces_phys * (10 ** -forces_order)
    pred_scaled = pred_2d * (10 ** -shared_order)
    target_scaled = target_2d * (10 ** -shared_order)
    error_scaled = abs_error * (10 ** -error_order)
    
    # Scale vmin/vmax for shared colorbar
    vmin_scaled = vmin_shared * (10 ** -shared_order)
    vmax_scaled = vmax_shared * (10 ** -shared_order)
    
    # Input forces - scatter plot for point forces (matching style of continuous field plot)
    # Create a background to show the plate boundaries
    # Get physical dimensions from the denormalized coordinates
    x_min, x_max = xx_phys.min(), xx_phys.max()
    y_min, y_max = yy_phys.min(), yy_phys.max()
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    
    # Plot the point forces as scatter
    scatter = axes[0].scatter(sources_x_phys, sources_y_phys, 
                             c=forces_scaled, cmap='Spectral_r', 
                             s=200, alpha=0.9, edgecolors='black', linewidths=0.5,
                             vmin=forces_scaled.min(), vmax=forces_scaled.max())
    axes[0].set_xlabel('X (m)', fontsize=16)
    axes[0].set_ylabel('Y (m)', fontsize=16)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Add grid for better visualization
    axes[0].grid(True, alpha=0.2, linestyle='--')
    
    cbar1 = plt.colorbar(scatter, cax=cb_axes[0], format='%.1f')
    cbar1.set_label('Force (N)', rotation=270, labelpad=20, fontsize=14)
    cbar1.ax.set_title(f'10^{{{forces_order}}}', fontsize=16, pad=20)
    cbar1.ax.tick_params(labelsize=16)
    
    # Predicted displacements (no colorbar, shared scale)
    im2 = axes[1].contourf(xx_phys, yy_phys, pred_scaled, levels=20, cmap='RdBu_r', 
                           vmin=vmin_scaled, vmax=vmax_scaled)
    # Add zero displacement contour lines (Chladni nodal lines)
    axes[1].contour(xx_phys, yy_phys, pred_scaled, levels=[0], colors='gold', 
                   linewidths=2, alpha=1.0, linestyles='-')
    axes[1].set_xlabel('X (m)', fontsize=16)
    axes[1].set_ylabel('Y (m)', fontsize=16)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Ground truth displacements (with shared colorbar)
    im3 = axes[2].contourf(xx_phys, yy_phys, target_scaled, levels=20, cmap='RdBu_r', 
                           vmin=vmin_scaled, vmax=vmax_scaled)
    # Add zero displacement contour lines (Chladni nodal lines)
    axes[2].contour(xx_phys, yy_phys, target_scaled, levels=[0], colors='gold', 
                   linewidths=2, alpha=1.0, linestyles='-')
    axes[2].set_xlabel('X (m)', fontsize=16)
    axes[2].set_ylabel('Y (m)', fontsize=16)
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    axes[2].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar3 = plt.colorbar(im3, cax=cb_axes[2], format='%.1f')
    cbar3.set_label('Displacement (m)', rotation=270, labelpad=20, fontsize=14)
    cbar3.ax.set_title(f'10^{{{shared_order}}}', fontsize=16, pad=20)
    cbar3.ax.tick_params(labelsize=16)
    
    # Absolute error (with its own colorbar)
    im4 = axes[3].contourf(xx_phys, yy_phys, error_scaled, levels=20, cmap='coolwarm')
    axes[3].set_xlabel('X (m)', fontsize=16)
    axes[3].set_ylabel('Y (m)', fontsize=16)
    axes[3].set_aspect('equal', adjustable='box')
    axes[3].tick_params(axis='both', which='major', labelsize=14)
    axes[3].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axes[3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar4 = plt.colorbar(im4, cax=cb_axes[3], format='%.1f')
    cbar4.set_label('Error (m)', rotation=270, labelpad=20, fontsize=14)
    cbar4.ax.set_title(f'10^{{{error_order}}}', fontsize=16, pad=20)
    cbar4.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Error metrics on physical units
    mse_error = np.mean((pred_2d - target_2d)**2)
    rel_error = np.linalg.norm(pred_2d - target_2d) / np.linalg.norm(target_2d)
    
    print(f"Sample {sample_idx} - MSE: {mse_error:.6e} m², Rel Error: {rel_error:.6f}")
    
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