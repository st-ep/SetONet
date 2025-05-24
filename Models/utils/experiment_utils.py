import json
import os
from datetime import datetime

def save_experiment_config(args, params, log_dir, device, models_were_loaded, eval_results=None):
    """
    Save experiment configuration and results to JSON file.
    
    Args:
        args: Command line arguments
        params: Problem parameters
        log_dir: Directory to save the config
        device: PyTorch device used
        models_were_loaded: Whether pre-trained models were loaded
        eval_results: Tuple of (avg_rel_error_T, avg_rel_error_T_inv) if available
    
    Returns:
        str: Path to the saved configuration file
    """
    config = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "models_loaded_from_pretrained": models_were_loaded,
            "script_name": "Derivative.py",
            "description": "SetONet training for derivative operator learning with cycle consistency"
        },
        "model_architecture": {
            "son_p_dim": args.son_p_dim,
            "son_phi_hidden": args.son_phi_hidden,
            "son_rho_hidden": args.son_rho_hidden,
            "son_trunk_hidden": args.son_trunk_hidden,
            "son_n_trunk_layers": args.son_n_trunk_layers,
            "son_phi_output_size": args.son_phi_output_size,
            "son_aggregation": args.son_aggregation,
            "pos_encoding_type": args.pos_encoding_type,
            "activation_function": "Tanh",
            "use_deeponet_bias": True
        },
        "training_parameters": {
            "son_lr": args.son_lr,
            "son_epochs": args.son_epochs,
            "lambda_cycle": args.lambda_cycle,
            "lr_schedule_steps": args.lr_schedule_steps,
            "lr_schedule_gammas": args.lr_schedule_gammas,
            "optimizer": "Adam",
            "loss_function": "MSELoss"
        },
        "problem_parameters": {
            "input_range": params['input_range'],
            "scale": params['scale'],
            "sensor_size": params['sensor_size'],
            "batch_size_train": params['batch_size_train'],
            "n_trunk_points_train": params['n_trunk_points_train'],
            "n_test_samples_eval": params['n_test_samples_eval'],
            "constant_zero": True,
            "polynomial_type": "cubic",
            "task_description": "Forward: f(x) -> f'(x), Inverse: f'(x) -> f(x)"
        },
        "data_generation": {
            "coefficient_distribution": "uniform",
            "coefficient_range": f"[-{params['scale']}, {params['scale']}]",
            "integration_constant": "zero (constant_zero=True)",
            "sensor_point_distribution": "uniform_linspace",
            "trunk_point_distribution": "uniform_linspace"
        },
        "file_paths": {
            "load_model_T_path": args.load_model_T_path,
            "load_model_T_inv_path": args.load_model_T_inv_path,
            "log_directory": log_dir,
            "tensorboard_directory": f"{log_dir}/tensorboard"
        },
        "reproducibility": {
            "torch_manual_seed": 0,
            "numpy_random_seed": 0,
            "note": "Seeds are fixed in main() function"
        },
        "logging": {
            "tensorboard_enabled": True,
            "tensorboard_log_frequency": 100,
            "metrics_logged": [
                "Loss/Forward_T", "Loss/Inverse_T_inv", "Loss/Cycle_1", "Loss/Cycle_2", "Loss/Total",
                "L2_Error/Forward_T", "L2_Error/Inverse_T_inv", "L2_Error/Cycle_1", "L2_Error/Cycle_2", "L2_Error/Total",
                "Training/Learning_Rate"
            ]
        }
    }
    
    # Add evaluation results if available
    if eval_results is not None:
        config["evaluation_results"] = {
            "avg_rel_error_T_forward": float(eval_results[0]),
            "avg_rel_error_T_inv_inverse": float(eval_results[1]),
            "evaluation_metric": "L2_relative_error",
            "test_samples": params['n_test_samples_eval']
        }
    
    # Save to JSON file
    config_path = os.path.join(log_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)
    
    print(f"Experiment configuration saved to: {config_path}")
    return config_path

def load_experiment_config(config_path):
    """
    Load experiment configuration from JSON file.
    
    Args:
        config_path (str): Path to the configuration JSON file
    
    Returns:
        dict: Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_experiment_summary(config):
    """
    Create a human-readable summary of the experiment configuration.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        str: Formatted summary string
    """
    summary = []
    summary.append("=" * 60)
    summary.append("EXPERIMENT SUMMARY")
    summary.append("=" * 60)
    
    # Basic info
    exp_info = config.get("experiment_info", {})
    summary.append(f"Timestamp: {exp_info.get('timestamp', 'N/A')}")
    summary.append(f"Device: {exp_info.get('device', 'N/A')}")
    summary.append(f"Pre-trained models loaded: {exp_info.get('models_loaded_from_pretrained', 'N/A')}")
    summary.append("")
    
    # Model architecture
    arch = config.get("model_architecture", {})
    summary.append("Model Architecture:")
    summary.append(f"  - Latent dimension (p): {arch.get('son_p_dim', 'N/A')}")
    summary.append(f"  - Aggregation: {arch.get('son_aggregation', 'N/A')}")
    summary.append(f"  - Positional encoding: {arch.get('pos_encoding_type', 'N/A')}")
    summary.append(f"  - Trunk layers: {arch.get('son_n_trunk_layers', 'N/A')}")
    summary.append("")
    
    # Training parameters
    train = config.get("training_parameters", {})
    summary.append("Training Parameters:")
    summary.append(f"  - Learning rate: {train.get('son_lr', 'N/A')}")
    summary.append(f"  - Epochs: {train.get('son_epochs', 'N/A')}")
    summary.append(f"  - Cycle consistency weight: {train.get('lambda_cycle', 'N/A')}")
    summary.append("")
    
    # Problem setup
    prob = config.get("problem_parameters", {})
    summary.append("Problem Setup:")
    summary.append(f"  - Input range: {prob.get('input_range', 'N/A')}")
    summary.append(f"  - Scale: {prob.get('scale', 'N/A')}")
    summary.append(f"  - Sensor points: {prob.get('sensor_size', 'N/A')}")
    summary.append(f"  - Batch size: {prob.get('batch_size_train', 'N/A')}")
    summary.append("")
    
    # Results
    if "evaluation_results" in config:
        results = config["evaluation_results"]
        summary.append("Evaluation Results:")
        summary.append(f"  - Forward T error: {results.get('avg_rel_error_T_forward', 'N/A'):.6f}")
        summary.append(f"  - Inverse T_inv error: {results.get('avg_rel_error_T_inv_inverse', 'N/A'):.6f}")
        summary.append("")
    
    summary.append("=" * 60)
    
    return "\n".join(summary) 