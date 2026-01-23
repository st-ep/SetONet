import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_darcy_results(model, dataset, darcy_dataset, device, sample_idx=0, save_path=None, split='test'):
    """
    Plot Darcy 2D results showing input permeability, prediction, ground truth, and absolute error.
    
    Args:
        model: Trained SetONet model
        dataset: Dataset dict with 'train' and 'test' splits
        darcy_dataset: DarcyDataset wrapper with coordinate information
        device: PyTorch device
        sample_idx: Index of sample to plot
        save_path: Path to save the plot (optional)
        split: Which split to use ('train' or 'test')
    """
    model.eval()
    
    # Get sample from specified split
    data_split = dataset[split]
    if sample_idx >= len(data_split):
        sample_idx = 0
        print(f"Warning: sample_idx {sample_idx} out of bounds, using 0")
    
    sample = data_split[sample_idx]
    
    # Prepare input data
    k_field = np.array(sample['k'])  # Permeability field
    p_field_true = np.array(sample['p'])  # True pressure field
    
    # Convert to tensors and get prediction
    with torch.no_grad():
        # Use same coordinates as training
        xs = darcy_dataset.coords.unsqueeze(0)  # (1, n_points, 2)
        us = torch.tensor(k_field.flatten(), device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, n_points, 1)
        ys = xs.clone()  # Same coordinates for target
        
        # Get prediction
        pred = model(xs, us, ys)  # (1, n_points, 1)
        pred_field = pred.squeeze().cpu().numpy().reshape(darcy_dataset.grid_size, darcy_dataset.grid_size)
    
    # Calculate absolute error
    error_field = np.abs(pred_field - p_field_true)
    
    # Create the plot (all subplots in a row)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Physical domain extent (0,1) x (0,1) 
    extent = [0, 1, 0, 1]
    
    # Plot 1: Input permeability field
    im1 = axes[0].imshow(k_field, cmap='viridis', extent=extent, origin='lower')
    axes[0].set_title('Input: Permeability k(x,y)', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Plot 2: Predicted pressure field
    im2 = axes[1].imshow(pred_field, cmap='coolwarm', extent=extent, origin='lower')
    axes[1].set_title('Prediction: Pressure p(x,y)', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Plot 3: Ground truth pressure field
    im3 = axes[2].imshow(p_field_true, cmap='coolwarm', extent=extent, origin='lower')
    axes[2].set_title('Ground Truth: Pressure p(x,y)', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    # Plot 4: Absolute error
    im4 = axes[3].imshow(error_field, cmap='Reds', extent=extent, origin='lower')
    axes[3].set_title('Absolute Error: |Pred - Truth|', fontsize=12)
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    plt.colorbar(im4, ax=axes[3], fraction=0.046)
    
    # Calculate metrics for the title
    mse = np.mean(error_field**2)
    max_error = np.max(error_field)
    rel_l2 = np.linalg.norm(error_field) / np.linalg.norm(p_field_true)
    
    # Add overall title with metrics
    fig.suptitle(f'{split.capitalize()} Sample {sample_idx} | MSE: {mse:.4e} | Max Error: {max_error:.4f} | Rel L2: {rel_l2:.4f}', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.close()
    model.train()

def plot_multiple_darcy_results(model, dataset, darcy_dataset, device, log_dir, n_samples=3):
    """
    Plot multiple Darcy 2D results for both train and test sets.
    
    Args:
        model: Trained SetONet model
        dataset: Dataset dict with 'train' and 'test' splits
        darcy_dataset: DarcyDataset wrapper
        device: PyTorch device
        log_dir: Directory to save plots
        n_samples: Number of samples to plot for each split
    """
    print("Generating Darcy 2D result plots...")
    
    # Create plots directory
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate random sample indices
    np.random.seed(42)  # For reproducible results
    train_indices = np.random.choice(len(dataset['train']), min(n_samples, len(dataset['train'])), replace=False)
    test_indices = np.random.choice(len(dataset['test']), min(n_samples, len(dataset['test'])), replace=False)
    
    # Plot train samples
    for i, idx in enumerate(train_indices):
        save_path = os.path.join(plots_dir, f"train_sample_{idx:03d}.png")
        plot_darcy_results(model, dataset, darcy_dataset, device, 
                          sample_idx=int(idx), save_path=save_path, split='train')
    
    # Plot test samples  
    for i, idx in enumerate(test_indices):
        save_path = os.path.join(plots_dir, f"test_sample_{idx:03d}.png")
        plot_darcy_results(model, dataset, darcy_dataset, device,
                          sample_idx=int(idx), save_path=save_path, split='test')
    
    print(f"Generated {len(train_indices)} train plots and {len(test_indices)} test plots in {plots_dir}")

def plot_darcy_results_deeponet(model, dataset, darcy_dataset, device, sample_idx=0, save_path=None, split='test'):
    """
    Plot Darcy 2D results for DeepONet model showing input permeability, prediction, ground truth, and absolute error.
    
    Args:
        model: Trained DeepONet model
        dataset: Dataset dict with 'train' and 'test' splits
        darcy_dataset: DarcyDatasetDeepONet wrapper with coordinate information
        device: PyTorch device
        sample_idx: Index of sample to plot
        save_path: Path to save the plot (optional)
        split: Which split to use ('train' or 'test')
    """
    model.eval()
    
    # Get sample from specified split
    data_split = dataset[split]
    if sample_idx >= len(data_split):
        sample_idx = 0
        print(f"Warning: sample_idx {sample_idx} out of bounds, using 0")
    
    sample = data_split[sample_idx]
    
    # Prepare input data
    k_field = np.array(sample['k'])  # Permeability field
    p_field_true = np.array(sample['p'])  # True pressure field
    
    # Convert to tensors and get prediction
    with torch.no_grad():
        # Branch input: flattened permeability field
        branch_input = torch.tensor(k_field.flatten(), device=device, dtype=torch.float32).unsqueeze(0)  # (1, n_points)
        
        # Trunk input: coordinates
        trunk_input = darcy_dataset.coords.unsqueeze(0)  # (1, n_points, 2)
        
        # Get prediction
        pred = model(branch_input, trunk_input)  # (1, n_points)
        pred_field = pred.squeeze().cpu().numpy().reshape(darcy_dataset.grid_size, darcy_dataset.grid_size)
    
    # Calculate absolute error
    error_field = np.abs(pred_field - p_field_true)
    
    # Create the plot (all subplots in a row)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Physical domain extent (0,1) x (0,1) 
    extent = [0, 1, 0, 1]
    
    # Plot 1: Input permeability field
    im1 = axes[0].imshow(k_field, cmap='viridis', extent=extent, origin='lower')
    axes[0].set_title('Input: Permeability k(x,y)', fontsize=12)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    
    # Plot 2: Predicted pressure field
    im2 = axes[1].imshow(pred_field, cmap='coolwarm', extent=extent, origin='lower')
    axes[1].set_title('Prediction: Pressure p(x,y)', fontsize=12)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)
    
    # Plot 3: Ground truth pressure field
    im3 = axes[2].imshow(p_field_true, cmap='coolwarm', extent=extent, origin='lower')
    axes[2].set_title('Ground Truth: Pressure p(x,y)', fontsize=12)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)
    
    # Plot 4: Absolute error
    im4 = axes[3].imshow(error_field, cmap='Reds', extent=extent, origin='lower')
    axes[3].set_title('Absolute Error: |Pred - Truth|', fontsize=12)
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('y')
    plt.colorbar(im4, ax=axes[3], fraction=0.046)
    
    # Calculate metrics for the title
    mse = np.mean(error_field**2)
    max_error = np.max(error_field)
    rel_l2 = np.linalg.norm(error_field) / np.linalg.norm(p_field_true)
    
    # Add overall title with metrics
    fig.suptitle(f'{split.capitalize()} Sample {sample_idx} | MSE: {mse:.4e} | Max Error: {max_error:.4f} | Rel L2: {rel_l2:.4f}', 
                 fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.close()
    model.train()

def plot_multiple_darcy_results_deeponet(model, dataset, darcy_dataset, device, log_dir, n_samples=3):
    """
    Plot multiple Darcy 2D results for DeepONet model for both train and test sets.
    
    Args:
        model: Trained DeepONet model
        dataset: Dataset dict with 'train' and 'test' splits
        darcy_dataset: DarcyDatasetDeepONet wrapper
        device: PyTorch device
        log_dir: Directory to save plots
        n_samples: Number of samples to plot for each split
    """
    print("Generating Darcy 2D result plots for DeepONet...")
    
    # Create plots directory
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate random sample indices
    np.random.seed(42)  # For reproducible results
    train_indices = np.random.choice(len(dataset['train']), min(n_samples, len(dataset['train'])), replace=False)
    test_indices = np.random.choice(len(dataset['test']), min(n_samples, len(dataset['test'])), replace=False)
    
    # Plot train samples
    for i, idx in enumerate(train_indices):
        save_path = os.path.join(plots_dir, f"train_sample_{idx:03d}.png")
        plot_darcy_results_deeponet(model, dataset, darcy_dataset, device, 
                                   sample_idx=int(idx), save_path=save_path, split='train')
    
    # Plot test samples  
    for i, idx in enumerate(test_indices):
        save_path = os.path.join(plots_dir, f"test_sample_{idx:03d}.png")
        plot_darcy_results_deeponet(model, dataset, darcy_dataset, device,
                                   sample_idx=int(idx), save_path=save_path, split='test')
    
    print(f"Generated {len(train_indices)} train plots and {len(test_indices)} test plots in {plots_dir}")
