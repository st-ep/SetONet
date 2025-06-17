import json
import os
from datetime import datetime
from .model_utils import count_parameters

def save_experiment_config(args, params, log_dir, device, model_was_loaded, eval_result, benchmark, model=None):
    """
    Save experiment configuration and results to JSON file.
    
    Args:
        args: Command line arguments
        params: Problem parameters
        log_dir: Directory to save the config
        device: PyTorch device used
        model_was_loaded: Whether pre-trained model was loaded
        eval_result: Evaluation result dictionary
        benchmark: Benchmark name ('integral' or 'derivative')
        model: The trained model (for parameter counting)
    
    Returns:
        str: Path to the saved configuration file
    """
    # Handle backward compatibility
    if benchmark is None:
        # Legacy mode - assume eval_result is a tuple
        description = "SetONet training for derivative operator learning with cycle consistency"
        task_desc = "Forward: f(x) -> f'(x), Inverse: f'(x) -> f(x)"
    else:
        # New single benchmark mode
        if benchmark == 'derivative':
            description = "SetONet training for derivative operator learning"
            task_desc = "Forward: f(x) -> f'(x)"
        elif benchmark == 'integral':
            description = "SetONet training for integral operator learning"
            task_desc = "Forward: f'(x) -> f(x)"
        else:
            description = f"SetONet training for {benchmark} operator learning"
            task_desc = f"Forward: {benchmark} operator"
    
    # Count model parameters if model is provided
    model_info = {}
    if model is not None:
        param_info = count_parameters(model)
        model_info = {
            "total_parameters": param_info['total_parameters'],
            "trainable_parameters": param_info['trainable_parameters'],
            "non_trainable_parameters": param_info['non_trainable_parameters']
        }
    
    config = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "model_loaded_from_pretrained": model_was_loaded,
            "script_name": "run_1d.py",
            "description": description,
            "benchmark": benchmark if benchmark else "legacy_bidirectional"
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
            "use_deeponet_bias": True,
            **model_info  # Add parameter counts
        },
        "training_parameters": {
            "son_lr": args.son_lr,
            "son_epochs": args.son_epochs,
            "lr_schedule_steps": args.lr_schedule_steps,
            "lr_schedule_gammas": args.lr_schedule_gammas,
            "optimizer": "Adam",
            "loss_function": "MSELoss"
        },
        "problem_parameters": {
            "input_range": params['input_range'],
            "sensor_size": params['sensor_size'],
            "batch_size_train": params['batch_size_train'],
            "n_trunk_points_train": params['n_trunk_points_train'],
            "n_test_samples_eval": params['n_test_samples_eval'],
            "constant_zero": True,
            "polynomial_type": "cubic",
            "task_description": task_desc,
            "sensor_distribution": "variable_per_batch" if params.get('variable_sensors', False) else "fixed_random_sorted",
            "sensor_seed": params.get('sensor_seed', 42),
            "variable_sensors": params.get('variable_sensors', False)
        },
        "file_paths": {
            "load_model_path": getattr(args, 'load_model_path', None),
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
                "Loss/Training", "L2_Error/Training", "Training/Learning_Rate"
            ]
        }
    }
    
    # Add scale parameter if it exists (for derivative/integral benchmarks)
    if 'scale' in params:
        config["problem_parameters"]["scale"] = params['scale']
        config["data_generation"] = {
            "coefficient_distribution": "uniform",
            "coefficient_range": f"[-{params['scale']}, {params['scale']}]",
            "integration_constant": "zero (constant_zero=True)",
            "sensor_point_distribution": "random_uniform_sorted",
            "trunk_point_distribution": "uniform_linspace"
        }
    
    # Handle legacy cycle consistency parameters
    if hasattr(args, 'lambda_cycle'):
        config["training_parameters"]["lambda_cycle"] = args.lambda_cycle
        config["logging"]["metrics_logged"].extend([
            "Loss/Forward_T", "Loss/Inverse_T_inv", "Loss/Cycle_1", "Loss/Cycle_2", "Loss/Total",
            "L2_Error/Forward_T", "L2_Error/Inverse_T_inv", "L2_Error/Cycle_1", "L2_Error/Cycle_2", "L2_Error/Total"
        ])
    
    # Handle evaluation results
    if eval_result is not None:
        if isinstance(eval_result, dict):
            # New single benchmark mode
            config["evaluation_results"] = eval_result
        else:
            # Legacy mode - assume it's a tuple
            config["evaluation_results"] = {
                "avg_rel_error_T_forward": eval_result[0] if len(eval_result) > 0 else None,
                "avg_rel_error_T_inv_inverse": eval_result[1] if len(eval_result) > 1 else None
            }
    
    # Add performance summary for easy access
    if eval_result is not None:
        if benchmark is None:
            # Legacy mode
            config["performance_summary"] = {
                "best_forward_error": float(eval_result[0]) if isinstance(eval_result, tuple) else None,
                "best_inverse_error": float(eval_result[1]) if isinstance(eval_result, tuple) else None,
                "training_completed": not model_was_loaded
            }
        else:
            # Single benchmark mode
            config["performance_summary"] = {
                "final_error": float(eval_result) if isinstance(eval_result, (float, int)) else None,
                "benchmark": benchmark,
                "training_completed": not model_was_loaded,
                "error_level": "excellent" if isinstance(eval_result, (float, int)) and eval_result < 1e-4 else "good" if isinstance(eval_result, (float, int)) and eval_result < 1e-3 else "moderate" if isinstance(eval_result, (float, int)) and eval_result < 1e-2 else "poor"
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

def generate_experiment_summary(config):
    """Generate a human-readable summary of the experiment configuration."""
    summary = []
    summary.append("=" * 60)
    summary.append("EXPERIMENT SUMMARY")
    summary.append("=" * 60)
    
    # Basic info
    exp_info = config.get("experiment_info", {})
    summary.append(f"Timestamp: {exp_info.get('timestamp', 'N/A')}")
    summary.append(f"Device: {exp_info.get('device', 'N/A')}")
    summary.append(f"Benchmark: {exp_info.get('benchmark', 'N/A')}")
    summary.append(f"Description: {exp_info.get('description', 'N/A')}")
    summary.append("")
    
    # Model architecture
    arch = config.get("model_architecture", {})
    summary.append("Model Architecture:")
    summary.append(f"  - P dimension: {arch.get('son_p_dim', 'N/A')}")
    summary.append(f"  - Phi hidden: {arch.get('son_phi_hidden', 'N/A')}")
    summary.append(f"  - Rho hidden: {arch.get('son_rho_hidden', 'N/A')}")
    summary.append(f"  - Trunk hidden: {arch.get('son_trunk_hidden', 'N/A')}")
    summary.append(f"  - Trunk layers: {arch.get('son_n_trunk_layers', 'N/A')}")
    summary.append(f"  - Aggregation: {arch.get('son_aggregation', 'N/A')}")
    
    # Parameter counts
    if 'total_parameters' in arch:
        from .model_utils import format_parameter_count
        total_params = arch['total_parameters']
        trainable_params = arch.get('trainable_parameters', total_params)
        summary.append(f"  - Total parameters: {total_params:,} ({format_parameter_count(total_params)})")
        summary.append(f"  - Trainable parameters: {trainable_params:,} ({format_parameter_count(trainable_params)})")
    
    summary.append("")
    
    # Training parameters
    train = config.get("training_parameters", {})
    summary.append("Training Parameters:")
    summary.append(f"  - Learning rate: {train.get('son_lr', 'N/A')}")
    summary.append(f"  - Epochs: {train.get('son_epochs', 'N/A')}")
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
        
        if "final_l2_relative_error" in results:
            # Single benchmark mode
            summary.append(f"  - Final L2 relative error: {results['final_l2_relative_error']:.6f}")
            summary.append(f"  - Benchmark task: {results.get('benchmark_task', 'N/A')}")
        else:
            # Legacy mode
            summary.append(f"  - Forward T error: {results.get('avg_rel_error_T_forward', 'N/A'):.6f}")
            summary.append(f"  - Inverse T_inv error: {results.get('avg_rel_error_T_inv_inverse', 'N/A'):.6f}")
        summary.append("")
    
    # Performance summary
    if "performance_summary" in config:
        perf = config["performance_summary"]
        summary.append("Performance Summary:")
        if "final_error" in perf:
            summary.append(f"  - Final error: {perf['final_error']:.6f}")
            summary.append(f"  - Error level: {perf.get('error_level', 'N/A')}")
        summary.append(f"  - Training completed: {perf.get('training_completed', 'N/A')}")
        summary.append("")
    
    summary.append("=" * 60)
    
    return "\n".join(summary)

def compare_experiments(config_paths):
    """
    Compare multiple experiment configurations and results.
    
    Args:
        config_paths (list): List of paths to configuration JSON files
    
    Returns:
        str: Formatted comparison string
    """
    configs = []
    for path in config_paths:
        try:
            configs.append(load_experiment_config(path))
        except Exception as e:
            print(f"Error loading config from {path}: {e}")
            continue
    
    if not configs:
        return "No valid configurations to compare."
    
    comparison = []
    comparison.append("=" * 80)
    comparison.append("EXPERIMENT COMPARISON")
    comparison.append("=" * 80)
    
    # Header
    comparison.append(f"{'Experiment':<15} {'Benchmark':<12} {'Final Error':<15} {'Epochs':<10} {'LR':<10} {'P-dim':<8}")
    comparison.append("-" * 80)
    
    # Data rows
    for i, config in enumerate(configs):
        exp_name = f"Exp_{i+1}"
        benchmark = config.get("experiment_info", {}).get("benchmark", "N/A")
        
        # Get final error
        eval_results = config.get("evaluation_results", {})
        if "final_l2_relative_error" in eval_results:
            final_error = f"{eval_results['final_l2_relative_error']:.2e}"
        elif "final_l2_relative_error_forward" in eval_results:
            final_error = f"{eval_results['final_l2_relative_error_forward']:.2e}"
        else:
            final_error = "N/A"
        
        epochs = config.get("training_parameters", {}).get("son_epochs", "N/A")
        lr = config.get("training_parameters", {}).get("son_lr", "N/A")
        p_dim = config.get("model_architecture", {}).get("son_p_dim", "N/A")
        
        comparison.append(f"{exp_name:<15} {benchmark:<12} {final_error:<15} {epochs:<10} {lr:<10} {p_dim:<8}")
    
    comparison.append("=" * 80)
    
    return "\n".join(comparison) 