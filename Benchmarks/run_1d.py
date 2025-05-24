import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.setonet_factory import create_setonet_models, load_pretrained_models
from Models.setonet_trainer import train_setonet_models
from Models.setonet_evaluator import evaluate_setonet_models
from Models.utils.experiment_utils import save_experiment_config  # Import from utils
from Plotting.plotting_utils import plot_operator_comparison, plot_cycle_consistency_reconstruction

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for a derivative task.")
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=256, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    parser.add_argument('--son_aggregation', type=str, default="attention", choices=["mean", "attention"], help='Aggregation type for SetONet')
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=250000, help='Number of epochs for SetONet')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument('--lambda_cycle', type=float, default=0.0, help='Weight for cycle consistency loss')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[50000, 100000, 150000, 200000, 250000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2], help="List of multiplicative factors for LR decay.")
    parser.add_argument('--load_model_T_path', type=str, default=None, help='Path to pre-trained SetONet T model')
    parser.add_argument('--load_model_T_inv_path', type=str, default=None, help='Path to pre-trained SetONet T_inv model')
    
    return parser.parse_args()

def setup_logging(project_root):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = "SetONet"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def setup_parameters():
    """Setup problem parameters."""
    return {
        'input_range': [-1, 1],
        'scale': 0.1,
        'sensor_size': 200,
        'batch_size_train': 64,
        'n_trunk_points_train': 200,
        'n_test_samples_eval': 1000
    }

def save_models(setonet_model_T, setonet_model_T_inv, log_dir, models_were_loaded):
    """Save trained models."""
    if not models_were_loaded:
        setonet_T_model_path = os.path.join(log_dir, "setonet_model_T.pth")
        torch.save(setonet_model_T.state_dict(), setonet_T_model_path)
        print(f"SetONet_T model saved to {setonet_T_model_path}")

        setonet_T_inv_model_path = os.path.join(log_dir, "setonet_model_T_inv.pth")
        torch.save(setonet_model_T_inv.state_dict(), setonet_T_inv_model_path)
        print(f"SetONet_T_inv model saved to {setonet_T_inv_model_path}")
    else:
        print("\nSkipping model saving as pre-trained models were loaded.")

def generate_plots(setonet_model_T, setonet_model_T_inv, params, log_dir):
    """Generate all plots."""
    print("\n--- Generating Plots ---")
    
    device = next(setonet_model_T.parameters()).device
    sensor_x_original = torch.linspace(params['input_range'][0], params['input_range'][1], 
                                      params['sensor_size'], device=device).view(-1, 1)
    x_dense_plot = torch.linspace(params['input_range'][0], params['input_range'][1], 
                                 200, device=device).view(-1, 1)

    # Forward model plots
    if setonet_model_T:
        print("Plotting for Forward Model T (f -> f')")
        plot_operator_comparison(
            model_to_use=setonet_model_T,
            branch_input_locations=sensor_x_original,
            trunk_query_locations=x_dense_plot,
            input_range=params['input_range'],
            scale=params['scale'],
            log_dir=log_dir,
            num_samples_to_plot=3,
            plot_filename_prefix="forward_T_",
            is_inverse_task=False,
            use_zero_constant=True
        )

    # Inverse model plots
    if setonet_model_T_inv:
        print("Plotting for Inverse Model T_inv (f' -> f)")
        plot_operator_comparison(
            model_to_use=setonet_model_T_inv,
            branch_input_locations=x_dense_plot,
            trunk_query_locations=sensor_x_original,
            input_range=params['input_range'],
            scale=params['scale'],
            log_dir=log_dir,
            num_samples_to_plot=3,
            plot_filename_prefix="inverse_Tinv_",
            is_inverse_task=True,
            use_zero_constant=True
        )

    # Cycle consistency plots
    if setonet_model_T and setonet_model_T_inv:
        print("\n--- Generating Cycle Consistency Plots ---")
        
        print("Plotting for Cycle Consistency: T_inv(T(f)) approx f")
        plot_cycle_consistency_reconstruction(
            model_1=setonet_model_T,
            model_2=setonet_model_T_inv,
            locs_A_orig=sensor_x_original,
            locs_B_orig=x_dense_plot,
            input_range=params['input_range'],
            scale=params['scale'],
            log_dir=log_dir,
            num_samples_to_plot=3,
            plot_filename_prefix="cycle_",
            cycle_starts_with_f=True,
            use_zero_constant=True
        )

        print("Plotting for Cycle Consistency: T(T_inv(f')) approx f'")
        plot_cycle_consistency_reconstruction(
            model_1=setonet_model_T_inv,
            model_2=setonet_model_T,
            locs_A_orig=x_dense_plot,
            locs_B_orig=sensor_x_original,
            input_range=params['input_range'],
            scale=params['scale'],
            log_dir=log_dir,
            num_samples_to_plot=3,
            plot_filename_prefix="cycle_",
            cycle_starts_with_f=False,
            use_zero_constant=True
        )

def main():
    """Main execution function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args = parse_arguments()
    
    # Validate arguments
    if len(args.lr_schedule_steps) != len(args.lr_schedule_gammas):
        raise ValueError("--lr_schedule_steps and --lr_schedule_gammas must have the same number of elements.")
    
    log_dir = setup_logging(project_root)
    params = setup_parameters()
    
    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create sensor points
    sensor_x_original = torch.linspace(params['input_range'][0], params['input_range'][1], 
                                      params['sensor_size'], device=device).view(-1, 1)
    
    # Initialize models
    setonet_model_T, setonet_model_T_inv = create_setonet_models(args, device)
    
    # Load pre-trained models if available
    models_were_loaded = load_pretrained_models(setonet_model_T, setonet_model_T_inv, args, device)
    
    # Training
    if not models_were_loaded:
        training_params = {
            'sensor_x_original': sensor_x_original,
            'scale': params['scale'],
            'input_range': params['input_range'],
            'batch_size_train': params['batch_size_train'],
            'n_trunk_points_train': params['n_trunk_points_train'],
            'sensor_size': params['sensor_size']
        }
        train_setonet_models(setonet_model_T, setonet_model_T_inv, args, training_params, device, log_dir)
    else:
        print("\nBoth SetONet models loaded. Skipping training.")
    
    # Evaluation
    eval_params = {
        'sensor_x_original': sensor_x_original,
        'scale': params['scale'],
        'input_range': params['input_range'],
        'batch_size_train': params['batch_size_train'],
        'n_trunk_points_eval': params['n_trunk_points_train'],
        'sensor_size': params['sensor_size'],
        'n_test_samples_eval': params['n_test_samples_eval']
    }
    eval_results = evaluate_setonet_models(setonet_model_T, setonet_model_T_inv, eval_params, device)
    
    # Save models
    save_models(setonet_model_T, setonet_model_T_inv, log_dir, models_were_loaded)
    
    # Generate plots
    generate_plots(setonet_model_T, setonet_model_T_inv, params, log_dir)
    
    # Save experiment configuration with evaluation results
    save_experiment_config(args, params, log_dir, device, models_were_loaded, eval_results)
    
    print("\nScript finished.")

if __name__ == "__main__":
    main()