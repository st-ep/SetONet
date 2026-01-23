#!/usr/bin/env python
"""
Configuration utilities for saving VIDON experiment configurations to JSON files.
"""

import json
import os
import torch
from datetime import datetime


def save_experiment_configuration(args, model, dataset, dataset_wrapper, device, log_dir, dataset_type="elastic_2d", test_results=None):
    """Save VIDON model and data configuration to JSON file."""
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get dataset information
    train_size = len(dataset['train']) if dataset else 0
    test_size = len(dataset['test']) if dataset else 0
    
    config = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "log_directory": log_dir
        },
        
        "model_architecture": {
            "model_type": "VIDON",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vidon_p_dim": args.vidon_p_dim,
            "vidon_n_heads": args.vidon_n_heads,
            "vidon_d_enc": args.vidon_d_enc,
            "vidon_head_output_size": args.vidon_head_output_size,
            "vidon_enc_hidden": args.vidon_enc_hidden,
            "vidon_enc_n_layers": args.vidon_enc_n_layers,
            "vidon_head_hidden": args.vidon_head_hidden,
            "vidon_head_n_layers": args.vidon_head_n_layers,
            "vidon_combine_hidden": args.vidon_combine_hidden,
            "vidon_combine_n_layers": args.vidon_combine_n_layers,
            "vidon_trunk_hidden": args.vidon_trunk_hidden,
            "vidon_n_trunk_layers": args.vidon_n_trunk_layers,
            "activation_fn": args.activation_fn,
            "input_size_src": 1 if dataset_type in ["darcy_1d", "derivative", "integral"] else 2,
            "output_size_src": 1,
            "input_size_tgt": 1 if dataset_type in ["darcy_1d", "derivative", "integral"] else 2,
            "output_size_tgt": 1
        },
        
        "training_parameters": {
            "vidon_lr": args.vidon_lr,
            "vidon_epochs": args.vidon_epochs,
            "batch_size": args.batch_size,
            "lr_schedule_steps": args.lr_schedule_steps,
            "lr_schedule_gammas": args.lr_schedule_gammas
        },
        
        "data_configuration": {
            "data_path": args.data_path,
            "dataset_type": dataset_type,
            "train_samples": train_size,
            "test_samples": test_size,
            "total_samples": train_size + test_size,
            "variable_sensors": getattr(args, 'variable_sensors', False),
            "train_sensor_dropoff": getattr(args, 'train_sensor_dropoff', 0.0),
            "eval_sensor_dropoff": getattr(args, 'eval_sensor_dropoff', 0.0),
            "replace_with_nearest": getattr(args, 'replace_with_nearest', False)
        },
        
        "dataset_structure": {
            "n_force_points": getattr(dataset_wrapper, 'n_force_points', None) if dataset_wrapper else None,
            "n_mesh_points": getattr(dataset_wrapper, 'n_mesh_points', None) if dataset_wrapper else None,
            "input_dim": getattr(dataset_wrapper, 'input_dim', None) if dataset_wrapper else None
        } if dataset_wrapper else {},
        
        "model_loading": {
            "load_model_path": args.load_model_path,
            "pre_trained_model_loaded": args.load_model_path is not None
        }
    }
    
    # Add evaluation results if provided
    if test_results is not None:
        config["evaluation_results"] = {
            "test_relative_l2_error": test_results.get("relative_l2_error"),
            "test_mse_loss": test_results.get("mse_loss"),
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_settings": {
                "eval_sensor_dropoff": getattr(args, 'eval_sensor_dropoff', 0.0),
                "replace_with_nearest": getattr(args, 'replace_with_nearest', False),
                "n_test_samples_evaluated": test_results.get("n_test_samples", "unknown")
            }
        }
    
    # Save configuration to JSON file
    config_path = os.path.join(log_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config_path

