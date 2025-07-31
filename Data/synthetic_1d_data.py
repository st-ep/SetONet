#!/usr/bin/env python
"""synthetic_1d_data.py
----------------------------------
Dataset wrapper for synthetic 1D data generation (integral/derivative tasks).
"""
from __future__ import annotations

import torch
import numpy as np
import os
import torch.nn as nn
from Data.data_utils import generate_batch, apply_sensor_dropoff

class Synthetic1DDataGenerator:
    """Dataset wrapper for synthetic 1D data that's compatible with SetONet training loop."""
    
    def __init__(self, params, device, sensor_x_original, benchmark):
        """
        Initialize the synthetic data generator.
        
        Args:
            params: Dictionary with parameters including batch_size_train, scale, input_range, etc.
            device: PyTorch device
            sensor_x_original: Fixed sensor locations (None for variable sensors)
            benchmark: Either 'derivative' or 'integral'
        """
        print(f"📊 Setting up synthetic 1D data generator for {benchmark} task...")
        
        # Store basic info
        self.device = device
        self.params = params
        self.batch_size = params['batch_size_train']
        self.sensor_x_original = sensor_x_original
        self.benchmark = benchmark
        
        # Extract parameters
        self.scale = params['scale']
        self.input_range = params['input_range']
        self.n_trunk_points_train = params['n_trunk_points_train']
        self.sensor_size = params['sensor_size']
        self.variable_sensors = params.get('variable_sensors', False)
        self.replace_with_nearest = params.get('replace_with_nearest', False)
        
        # Dataset structure info (for compatibility with other parts of the code)
        self.n_force_points = self.sensor_size  # Number of sensor points
        self.n_mesh_points = self.n_trunk_points_train  # Number of query points
        self.input_dim = 1  # 1D coordinates
        
        if self.variable_sensors:
            print(f"✅ Synthetic 1D generator initialized with VARIABLE sensors ({self.sensor_size} per batch)")
        else:
            print(f"✅ Synthetic 1D generator initialized with FIXED sensors ({len(sensor_x_original)} total)")
    
    def sample(self, device=None):
        """Sample a batch using synthetic data generation (compatible with SetONet training loop)."""
        if device is None:
            device = self.device
            
        # Generate synthetic batch
        if self.variable_sensors:
            batch_data = generate_batch(
                batch_size=self.batch_size,
                n_trunk_points=self.n_trunk_points_train,
                sensor_x=None,
                scale=self.scale,
                input_range=self.input_range,
                device=device,
                constant_zero=True,
                variable_sensors=True,
                sensor_size=self.sensor_size
            )
            # Variable sensors return 6 values
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points, sensor_x_batch = batch_data
            sensor_x_to_use = sensor_x_batch
        else:
            batch_data = generate_batch(
                batch_size=self.batch_size,
                n_trunk_points=self.n_trunk_points_train,
                sensor_x=self.sensor_x_original,
                scale=self.scale,
                input_range=self.input_range,
                device=device,
                constant_zero=True
            )
            # Fixed sensors return 5 values
            f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = batch_data
            sensor_x_to_use = self.sensor_x_original
        
        # Use all sensor data during training (no sensor dropout during training for synthetic data)
        sensor_x_used = sensor_x_to_use
        f_at_sensors_used = f_at_sensors
        
        # Prepare data in SetONet format based on the benchmark
        if self.benchmark == 'derivative':
            # Derivative task: f(sensors) -> f'(trunk_points)
            xs = sensor_x_used.unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size, n_sensors, 1]
            us = f_at_sensors_used.unsqueeze(-1)  # [batch_size, n_sensors, 1]
            ys = batch_x_eval_points.unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size, n_trunk_points, 1]
            G_u_ys = f_prime_at_trunk.T.unsqueeze(-1)  # [batch_size, n_trunk_points, 1]
        elif self.benchmark == 'integral':
            # Integral task: f'(sensors) -> f(trunk_points)
            xs = sensor_x_used.unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size, n_sensors, 1]
            us = f_prime_at_sensors.unsqueeze(-1)  # [batch_size, n_sensors, 1]
            ys = batch_x_eval_points.unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size, n_trunk_points, 1]
            G_u_ys = f_at_trunk.T.unsqueeze(-1)  # [batch_size, n_trunk_points, 1]
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")
        
        return xs, us, ys, G_u_ys, None
    
    def evaluate_model(self, model, n_test_samples, batch_size, eval_sensor_dropoff=0.0, replace_with_nearest=False):
        """Evaluate model on synthetic test data."""
        from Models.utils.helper_utils import calculate_l2_relative_error
        
        model.eval()
        total_loss = 0.0
        total_rel_error = 0.0
        n_batches = n_test_samples // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                # Generate test batch
                if self.variable_sensors:
                    batch_data = generate_batch(
                        batch_size=batch_size,
                        n_trunk_points=self.n_trunk_points_train,
                        sensor_x=None,
                        scale=self.scale,
                        input_range=self.input_range,
                        device=self.device,
                        constant_zero=True,
                        variable_sensors=True,
                        sensor_size=self.sensor_size
                    )
                    f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points, sensor_x_batch = batch_data
                    sensor_x_to_use = sensor_x_batch
                else:
                    batch_data = generate_batch(
                        batch_size=batch_size,
                        n_trunk_points=self.n_trunk_points_train,
                        sensor_x=self.sensor_x_original,
                        scale=self.scale,
                        input_range=self.input_range,
                        device=self.device,
                        constant_zero=True
                    )
                    f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = batch_data
                    sensor_x_to_use = self.sensor_x_original
                
                # Apply sensor dropout if specified
                sensor_x_used = sensor_x_to_use
                f_at_sensors_used = f_at_sensors
                
                if eval_sensor_dropoff > 0.0:
                    if self.benchmark == 'derivative':
                        sensor_x_dropped, f_at_sensors_dropped = apply_sensor_dropoff(
                            sensor_x_to_use, f_at_sensors, eval_sensor_dropoff, replace_with_nearest
                        )
                        sensor_x_used = sensor_x_dropped
                        f_at_sensors_used = f_at_sensors_dropped
                    elif self.benchmark == 'integral':
                        sensor_x_dropped, f_prime_at_sensors_dropped = apply_sensor_dropoff(
                            sensor_x_to_use, f_prime_at_sensors, eval_sensor_dropoff, replace_with_nearest
                        )
                        sensor_x_used = sensor_x_dropped
                        f_prime_at_sensors = f_prime_at_sensors_dropped
                
                # Prepare inputs based on benchmark
                if self.benchmark == 'derivative':
                    xs = sensor_x_used.unsqueeze(0).expand(batch_size, -1, -1)
                    us = f_at_sensors_used.unsqueeze(-1)
                    ys = batch_x_eval_points.unsqueeze(0).expand(batch_size, -1, -1)
                    target = f_prime_at_trunk.T.unsqueeze(-1)
                else:  # integral
                    xs = sensor_x_used.unsqueeze(0).expand(batch_size, -1, -1)
                    us = f_prime_at_sensors.unsqueeze(-1)
                    ys = batch_x_eval_points.unsqueeze(0).expand(batch_size, -1, -1)
                    target = f_at_trunk.T.unsqueeze(-1)
                
                pred = model(xs, us, ys)
                
                mse_loss = torch.nn.MSELoss()(pred, target)
                total_loss += mse_loss.item()
                
                rel_error = calculate_l2_relative_error(pred.squeeze(-1), target.squeeze(-1))
                total_rel_error += rel_error.item()
        
        model.train()
        avg_loss = total_loss / n_batches
        avg_rel_error = total_rel_error / n_batches
        
        return avg_loss, avg_rel_error
    
    def get_tensorboard_callback(self, log_dir, args):
        """Create properly configured TensorBoard callback for synthetic data."""
        from Models.utils.tensorboard_callback import TensorBoardCallback
        
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        dummy_dataset = {'train': [], 'test': []}  # Required for callback interface
        
        callback = TensorBoardCallback(
            log_dir=tb_log_dir,
            dataset=dummy_dataset,
            dataset_wrapper=self,
            device=self.device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=args.eval_sensor_dropoff,
            replace_with_nearest=args.replace_with_nearest
        )
        
        print(f"TensorBoard logs will be saved to: {tb_log_dir}")
        return callback
    
    def prepare_test_results(self, avg_loss, avg_rel_error, n_test):
        """Prepare test results dictionary for config saving."""
        return {
            "relative_l2_error": avg_rel_error,
            "mse_loss": avg_loss,
            "n_test_samples": n_test
        }

def get_activation_function(activation_name):
    """Get activation function by name."""
    activation_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'swish': nn.SiLU  # SiLU is equivalent to Swish
    }
    return activation_map.get(activation_name.lower(), nn.ReLU)

def create_synthetic_setonet_model(args, device):
    """Create SetONet model for synthetic 1D benchmark."""
    from Models.SetONet import SetONet
    
    print(f"\n--- Initializing SetONet Model for {args.benchmark} ---")
    print(f"Using activation function: {args.activation_fn}")
    
    activation_fn = get_activation_function(args.activation_fn)
    
    model = SetONet(
        input_size_src=1,  # 1D coordinates (x)
        output_size_src=1,  # Scalar function values
        input_size_tgt=1,  # 1D coordinates (x)
        output_size_tgt=1,  # Scalar output values
        p=args.son_p_dim,
        phi_hidden_size=args.son_phi_hidden,
        rho_hidden_size=args.son_rho_hidden,
        trunk_hidden_size=args.son_trunk_hidden,
        n_trunk_layers=args.son_n_trunk_layers,
        activation_fn=activation_fn,
        use_deeponet_bias=True,
        phi_output_size=args.son_phi_output_size,
        initial_lr=args.son_lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
        pos_encoding_type=args.pos_encoding_type,
        pos_encoding_dim=args.pos_encoding_dim,
        pos_encoding_max_freq=args.pos_encoding_max_freq,
        aggregation_type=args.son_aggregation,
        use_positional_encoding=(args.pos_encoding_type != 'skip'),
        attention_n_tokens=1,
    ).to(device)
    
    return model

def load_synthetic_pretrained_model(setonet_model, args, device):
    """Load a pre-trained model if path is provided."""
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            setonet_model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print(f"Loaded pre-trained SetONet model from: {args.load_model_path}")
            return True
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")
            args.load_model_path = None
    
    return False