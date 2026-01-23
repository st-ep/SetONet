#!/usr/bin/env python
"""
Configuration utilities for saving DeepONet experiment configurations to JSON files.
"""

import json
import os
import torch
from datetime import datetime


def save_experiment_configuration(args, model, dataset, dataset_wrapper, device, log_dir, dataset_type="elastic_2d", test_results=None):
    """Save DeepONet model and data configuration to JSON file."""
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
            "model_type": "DeepONet",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "don_p_dim": args.don_p_dim,
            "don_trunk_hidden": args.don_trunk_hidden,
            "don_n_trunk_layers": args.don_n_trunk_layers,
            "don_branch_hidden": args.don_branch_hidden,
            "don_n_branch_layers": args.don_n_branch_layers,
            "activation_fn": args.activation_fn,
            "input_size_src": 1 if dataset_type == "darcy_1d" else 2,
            "output_size_src": 1,
            "input_size_tgt": 1 if dataset_type == "darcy_1d" else 2,
            "output_size_tgt": 1,
            "use_deeponet_bias": True
        },
        
        "training_parameters": {
            "don_lr": args.don_lr,
            "don_epochs": args.don_epochs,
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
            "sensor_dropoff_method": "interpolation"  # DeepONet uses interpolation for missing sensors
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
                "sensor_dropoff_method": "interpolation",
                "n_test_samples_evaluated": test_results.get("n_test_samples", "unknown")
            }
        }
    
    # Save configuration to JSON file
    config_path = os.path.join(log_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    return config_path