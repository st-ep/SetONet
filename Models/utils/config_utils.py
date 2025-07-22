#!/usr/bin/env python
"""
Configuration utilities for saving experiment configurations to JSON files.
"""

import json
import os
import torch
from datetime import datetime


def save_experiment_configuration(args, model, dataset, dataset_wrapper, device, log_dir, dataset_type="elastic_2d", test_results=None):
    """Save model and data configuration to JSON file."""
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
            "model_type": "SetONet",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "son_p_dim": args.son_p_dim,
            "son_phi_hidden": args.son_phi_hidden,
            "son_rho_hidden": args.son_rho_hidden,
            "son_trunk_hidden": args.son_trunk_hidden,
            "son_n_trunk_layers": args.son_n_trunk_layers,
            "son_phi_output_size": args.son_phi_output_size,
            "son_aggregation": args.son_aggregation,
            "activation_fn": args.activation_fn,
            "pos_encoding_type": args.pos_encoding_type,
            "pos_encoding_dim": args.pos_encoding_dim,
            "pos_encoding_max_freq": args.pos_encoding_max_freq,
            "use_positional_encoding": (args.pos_encoding_type != 'skip'),
            "input_size_src": 1 if dataset_type == "darcy_1d" else 2,
            "output_size_src": 1,
            "input_size_tgt": 1 if dataset_type == "darcy_1d" else 2,
            "output_size_tgt": 1,
            "use_deeponet_bias": True,
            "attention_n_tokens": 1
        },
        
        "training_parameters": {
            "son_lr": args.son_lr,
            "son_epochs": args.son_epochs,
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

 