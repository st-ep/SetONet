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

from Models.setonet_factory import create_setonet_model, load_pretrained_model
from Models.setonet_trainer import train_setonet_model
from Models.setonet_evaluator import evaluate_setonet_model
from Models.utils.experiment_utils import save_experiment_config
from Plotting.plotting_utils import plot_operator_comparison
from Data.data_utils import generate_batch

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train SetONet for a specific benchmark task.")
    
    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True, choices=['integral', 'derivative'], 
                       help='Benchmark task: integral (f\' -> f) or derivative (f -> f\')')
    
    # Model architecture
    parser.add_argument('--son_p_dim', type=int, default=32, help='Latent dimension p for SetONet')
    parser.add_argument('--son_phi_hidden', type=int, default=256, help='Hidden size for SetONet phi network')
    parser.add_argument('--son_rho_hidden', type=int, default=512, help='Hidden size for SetONet rho network')
    parser.add_argument('--son_trunk_hidden', type=int, default=256, help='Hidden size for SetONet trunk network')
    parser.add_argument('--son_n_trunk_layers', type=int, default=4, help='Number of layers in SetONet trunk network')
    parser.add_argument('--son_phi_output_size', type=int, default=32, help='Output size of SetONet phi network before aggregation')
    
    # Aggregation options
    parser.add_argument('--son_aggregation', type=str, default="attention", 
                       choices=["mean", "attention", "statistical", "hybrid", "specialized"], 
                       help='Aggregation type for SetONet')
    parser.add_argument('--attention_n_tokens', type=int, default=1, help='Number of attention tokens')
    parser.add_argument('--statistical_features', type=str, nargs='+', 
                       choices=['mean', 'std', 'min', 'max', 'median', 'sum'],
                       default=None, help='Statistical features to use (default: mean,std,min,max)')
    parser.add_argument('--statistical_fusion', type=str, default='basic',
                       choices=['basic', 'structured', 'scale_aware'],
                       help='How to combine statistical features: basic=concat, structured=separate processing, scale_aware=normalized')
    parser.add_argument('--statistical_fusion_strategy', type=str, default='separate_then_combine',
                       choices=['separate_then_combine', 'attention_fusion', 'weighted_sum'],
                       help='Strategy for structured statistical fusion')
    parser.add_argument('--hybrid_combine_strategy', type=str, default='concat',
                       choices=['concat', 'learned', 'weighted'],
                       help='Strategy for combining attention and statistical features in hybrid mode')
    parser.add_argument('--attention_specializations', type=str, nargs='+',
                       choices=['general', 'extreme', 'variance', 'local'],
                       default=None, help='Specializations for attention tokens (default: general,extreme,variance,local)')
    parser.add_argument('--use_specialization_loss', action='store_true',
                       help='Use specialization loss to encourage query diversity')
    parser.add_argument('--specialization_loss_weight', type=float, default=0.01,
                       help='Weight for specialization loss')
    
    # Training parameters
    parser.add_argument('--son_lr', type=float, default=5e-4, help='Learning rate for SetONet')
    parser.add_argument('--son_epochs', type=int, default=175000, help='Number of epochs for SetONet')
    parser.add_argument('--pos_encoding_type', type=str, default='sinusoidal', choices=['sinusoidal', 'skip'], help='Positional encoding type for SetONet')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000, 1250000, 1500000], help="List of steps for LR decay milestones.")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5], help="List of multiplicative factors for LR decay.")
    
    # Data generation
    parser.add_argument('--variable_sensors', action='store_true', help='Use different random sensor locations for each sample (more challenging)')
    
    # Model loading
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained SetONet model')
    
    return parser.parse_args()

def setup_logging(project_root, benchmark):
    """Setup logging directory."""
    logs_base_in_project = os.path.join(project_root, "logs")
    model_folder_name = f"SetONet_{benchmark}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(logs_base_in_project, model_folder_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def setup_parameters(args):
    """Setup problem parameters."""
    return {
        'input_range': [-1, 1],
        'scale': 0.1,
        'sensor_size': 200,
        'batch_size_train': 64,
        'n_trunk_points_train': 200,
        'n_test_samples_eval': 1000,
        'sensor_seed': 42,
        'variable_sensors': args.variable_sensors
    }

def create_sensor_points(params, device):
    """Create sensor points - either fixed random or variable per batch."""
    if params.get('variable_sensors', False):
        # Sensor points will be generated per batch in data_utils.py
        print(f"Using variable random sensor locations (same within batch, different between batches)")
        return None  # Signal that sensors are variable
    else:
        # Use fixed random sensor locations (same across all samples)
        torch.manual_seed(params['sensor_seed'])  # Fixed seed for reproducible sensor locations
        sensor_x = torch.rand(params['sensor_size'], device=device) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
        sensor_x = sensor_x.sort()[0]  # Sort for better visualization
        sensor_x = sensor_x.view(-1, 1)
        print(f"Using {params['sensor_size']} fixed random sensor locations (sorted, seed={params['sensor_seed']})")
        return sensor_x

def save_model(setonet_model, log_dir, benchmark, model_was_loaded):
    """Save trained model."""
    if not model_was_loaded:
        model_path = os.path.join(log_dir, f"setonet_model_{benchmark}.pth")
        torch.save(setonet_model.state_dict(), model_path)
        print(f"SetONet {benchmark} model saved to {model_path}")
    else:
        print(f"\nSkipping model saving as pre-trained {benchmark} model was loaded.")

def generate_plots(setonet_model, params, log_dir, benchmark, sensor_x_original):
    """Generate plots for the specific benchmark."""
    print("\n--- Generating Plots ---")
    
    device = next(setonet_model.parameters()).device
    # Create dense evaluation points for plotting
    x_dense_plot = torch.linspace(params['input_range'][0], params['input_range'][1], 200, device=device).view(-1, 1)
    
    if benchmark == 'derivative':
        # For derivative: sensor locations -> dense plot locations
        print("Plotting for Derivative Model (f -> f')")
        
        if params.get('variable_sensors', False):
            # For variable sensor training, use actual sensor locations from evaluation batches
            print("Using actual sensor locations from evaluation batches")
            
            # Generate 3 different evaluation batches to show different sensor configurations
            
            for batch_idx in range(3):  # Show 3 different batches
                print(f"Generating plot for evaluation batch {batch_idx + 1}")
                
                # Generate the same way as in evaluation
                batch_data = generate_batch(
                    batch_size=1,  # Just need sensor locations, so batch_size=1
                    n_trunk_points=params['n_trunk_points_train'],
                    sensor_x=None,  # None for variable sensors
                    scale=params['scale'],
                    input_range=params['input_range'],
                    device=device,
                    constant_zero=True,
                    variable_sensors=True,
                    sensor_size=params['sensor_size']
                )
                
                # Extract the sensor locations used for this batch
                _, _, _, _, _, actual_sensor_locations = batch_data
                
                plot_operator_comparison(
                    model_to_use=setonet_model,
                    branch_input_locations=actual_sensor_locations,
                    trunk_query_locations=x_dense_plot,
                    input_range=params['input_range'],
                    scale=params['scale'],
                    log_dir=log_dir,
                    num_samples_to_plot=1,
                    plot_filename_prefix=f"{benchmark}_eval_batch_{batch_idx+1}_",
                    is_inverse_task=False,
                    use_zero_constant=True
                )
        else:
            # Fixed sensors - use the original sensor locations
            plot_operator_comparison(
                model_to_use=setonet_model,
                branch_input_locations=sensor_x_original,
                trunk_query_locations=x_dense_plot,
                input_range=params['input_range'],
                scale=params['scale'],
                log_dir=log_dir,
                num_samples_to_plot=3,
                plot_filename_prefix=f"{benchmark}_",
                is_inverse_task=False,
                use_zero_constant=True
            )
    elif benchmark == 'integral':
        # For integral: f' -> f (dense plot locations -> sensor locations)
        print("Plotting for Integral Model (f' -> f)")
        plot_operator_comparison(
            model_to_use=setonet_model,
            branch_input_locations=x_dense_plot,
            trunk_query_locations=sensor_x_original,  # Use actual random sensor locations
            input_range=params['input_range'],
            scale=params['scale'],
            log_dir=log_dir,
            num_samples_to_plot=3,
            plot_filename_prefix=f"{benchmark}_",
            is_inverse_task=True,
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
    
    print(f"Running benchmark: {args.benchmark}")
    
    log_dir = setup_logging(project_root, args.benchmark)
    params = setup_parameters(args)
    
    # Fix random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create sensor points (fixed random or None for variable)
    sensor_x_original = create_sensor_points(params, device)
    
    # Initialize model
    setonet_model = create_setonet_model(args, device)
    
    # Load pre-trained model if available
    model_was_loaded = load_pretrained_model(setonet_model, args, device)
    
    # Training
    if not model_was_loaded:
        training_params = {
            'sensor_x_original': sensor_x_original,
            'scale': params['scale'],
            'input_range': params['input_range'],
            'batch_size_train': params['batch_size_train'],
            'n_trunk_points_train': params['n_trunk_points_train'],
            'sensor_size': params['sensor_size'],
            'benchmark': args.benchmark,
            'variable_sensors': params['variable_sensors']
        }
        train_setonet_model(setonet_model, args, training_params, device, log_dir)
    else:
        print(f"\nSetONet {args.benchmark} model loaded. Skipping training.")
    
    # Evaluation
    eval_params = {
        'sensor_x_original': sensor_x_original,
        'scale': params['scale'],
        'input_range': params['input_range'],
        'batch_size_train': params['batch_size_train'],
        'n_trunk_points_eval': params['n_trunk_points_train'],
        'sensor_size': params['sensor_size'],
        'n_test_samples_eval': params['n_test_samples_eval'],
        'benchmark': args.benchmark,
        'variable_sensors': params['variable_sensors']
    }
    eval_result = evaluate_setonet_model(setonet_model, eval_params, device)
    
    # Save model
    save_model(setonet_model, log_dir, args.benchmark, model_was_loaded)
    
    # Generate plots (use fixed sensor locations for plotting even if training used variable)
    if sensor_x_original is None:
        # Create fixed sensor locations for plotting
        plot_sensor_x = torch.rand(params['sensor_size'], device=device) * (params['input_range'][1] - params['input_range'][0]) + params['input_range'][0]
        plot_sensor_x = plot_sensor_x.sort()[0].view(-1, 1)
    else:
        plot_sensor_x = sensor_x_original
    
    generate_plots(setonet_model, params, log_dir, args.benchmark, plot_sensor_x)
    
    # Save experiment configuration with evaluation results
    save_experiment_config(args, params, log_dir, device, model_was_loaded, eval_result, args.benchmark, setonet_model)
    
    print("\nScript finished.")

if __name__ == "__main__":
    main()