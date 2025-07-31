#!/usr/bin/env python
"""
TensorBoard callback for logging training and evaluation metrics during SetONet training.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from .helper_utils import calculate_l2_relative_error


class TensorBoardCallback:
    """Callback for logging metrics to TensorBoard during training."""
    
    def __init__(self, log_dir, dataset, dataset_wrapper, device, eval_frequency=1000, n_test_samples=100, 
                 eval_sensor_dropoff=0.0, replace_with_nearest=False):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: Directory for TensorBoard logs
            dataset: Dataset dictionary with 'train' and 'test' splits
            dataset_wrapper: Dataset wrapper instance for sampling (ElasticDataset, ChladniDataset, etc.)
            device: Device to run evaluation on
            eval_frequency: How often to evaluate on test set (in steps)
            n_test_samples: Number of test samples to evaluate
            eval_sensor_dropoff: Sensor dropout rate for evaluation
            replace_with_nearest: Whether to replace dropped sensors with nearest
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset
        self.dataset_wrapper = dataset_wrapper
        self.device = device
        self.eval_frequency = eval_frequency
        self.n_test_samples = n_test_samples
        self.eval_sensor_dropoff = eval_sensor_dropoff
        self.replace_with_nearest = replace_with_nearest
        
        print(f"TensorBoard logging to: {log_dir}")
        if eval_sensor_dropoff > 0.0:
            replacement_mode = "nearest replacement" if replace_with_nearest else "removal"
            print(f"Test evaluation with {eval_sensor_dropoff:.1%} sensor dropout ({replacement_mode})")
    
    def on_training_start(self, local_vars):
        """Called at the start of training."""
        print("Starting TensorBoard logging...")
    
    def on_step(self, local_vars):
        """Called after each training step."""
        step = local_vars['self'].total_steps
        loss = local_vars['loss']
        rel_l2_error = local_vars['rel_l2_error']
        current_lr = local_vars['current_lr']
        grad_norm = local_vars['norm']
        
        # Log training metrics
        self.writer.add_scalar('Training/MSE_Loss', loss.item(), step)
        self.writer.add_scalar('Training/Relative_L2_Error', rel_l2_error.item(), step)
        self.writer.add_scalar('Training/Learning_Rate', current_lr, step)
        self.writer.add_scalar('Training/Gradient_Norm', grad_norm, step)
        
        # Evaluate on test set periodically
        if step % self.eval_frequency == 0 and step > 0:
            model = local_vars['self']
            test_rel_l2 = self._evaluate_test_set(model)
            self.writer.add_scalar('Evaluation/Test_Relative_L2_Error', test_rel_l2, step)
    
    def on_training_end(self, local_vars):
        """Called at the end of training."""
        # Final evaluation
        model = local_vars['self']
        final_test_rel_l2 = self._evaluate_test_set(model)
        final_step = model.total_steps
        
        self.writer.add_scalar('Evaluation/Final_Test_Relative_L2_Error', final_test_rel_l2, final_step)
        print(f"Final Test Relative L2 Error: {final_test_rel_l2:.6f}")
        
        self.writer.close()
        print("TensorBoard logging completed.")
    
    def _evaluate_test_set(self, model):
        """Evaluate model on test set and return average relative L2 error."""
        model.eval()
        
        # Check if this is a Darcy-style dataset (has sensor/query indices for subsampling)
        if hasattr(self.dataset_wrapper, 'sensor_indices') and hasattr(self.dataset_wrapper, 'query_indices'):
            # Use dataset wrapper for evaluation (Darcy style with indexed subsampling)
            return self._evaluate_with_wrapper(model)
        elif hasattr(self.dataset_wrapper, 'benchmark') and hasattr(self.dataset_wrapper, 'sample'):
            # Use synthetic data generator for evaluation (Synthetic1DDataGenerator style)
            return self._evaluate_with_synthetic_generator(model)
        else:
            # Use direct dataset access for evaluation (Elastic/Chladni style with full point clouds)
            return self._evaluate_with_dataset(model)
    
    def _evaluate_with_wrapper(self, model):
        """Evaluate using test dataset directly for Darcy data."""
        # For Darcy data, evaluate on actual test data instead of training data
        test_data = self.dataset['test']
        n_test = min(self.n_test_samples, len(test_data))
        
        total_rel_error = 0.0
        
        with torch.no_grad():
            for i in range(n_test):
                sample = test_data[i]
                
                # Get sensor and query data (similar to main script evaluation)
                xs_data = torch.tensor(sample['u'], device=self.device)[self.dataset_wrapper.sensor_indices].unsqueeze(0).unsqueeze(-1)
                ys_data = self.dataset_wrapper.query_x.unsqueeze(0)
                target = torch.tensor(sample['s'], device=self.device)[self.dataset_wrapper.query_indices].unsqueeze(0).unsqueeze(-1)
                
                # Apply sensor dropout if specified
                if self.eval_sensor_dropoff > 0.0:
                    from Data.data_utils import apply_sensor_dropoff
                    
                    xs_dropped, us_dropped = apply_sensor_dropoff(
                        self.dataset_wrapper.sensor_x, 
                        xs_data.squeeze(0).squeeze(-1), 
                        self.eval_sensor_dropoff, 
                        self.replace_with_nearest
                    )
                    xs_data = us_dropped.unsqueeze(0).unsqueeze(-1)
                    sensor_x_used = xs_dropped.unsqueeze(0)
                else:
                    sensor_x_used = self.dataset_wrapper.sensor_x.unsqueeze(0)
                
                # Forward pass
                pred = model(sensor_x_used, xs_data, ys_data)
                
                # Calculate relative error
                rel_error = calculate_l2_relative_error(pred, target)
                total_rel_error += rel_error.item()
        
                model.train()
        return total_rel_error / n_test 
    
    def _evaluate_with_synthetic_generator(self, model):
        """Evaluate using synthetic data generator (for synthetic 1D benchmarks)."""
        from Data.data_utils import generate_batch, apply_sensor_dropoff
        
        # Use the same parameters as the dataset wrapper for consistency
        batch_size = 8  # Small batch size for evaluation
        n_eval_batches = max(1, self.n_test_samples // batch_size)
        
        total_rel_error = 0.0
        
        with torch.no_grad():
            for _ in range(n_eval_batches):
                # Generate test batch the same way as in the main evaluation
                if self.dataset_wrapper.variable_sensors:
                    batch_data = generate_batch(
                        batch_size=batch_size,
                        n_trunk_points=self.dataset_wrapper.n_trunk_points_train,
                        sensor_x=None,
                        scale=self.dataset_wrapper.scale,
                        input_range=self.dataset_wrapper.input_range,
                        device=self.device,
                        constant_zero=True,
                        variable_sensors=True,
                        sensor_size=self.dataset_wrapper.sensor_size
                    )
                    f_at_sensors, _, f_at_trunk, f_prime_at_trunk, batch_x_eval_points, sensor_x_batch = batch_data
                    sensor_x_to_use = sensor_x_batch
                else:
                    batch_data = generate_batch(
                        batch_size=batch_size,
                        n_trunk_points=self.dataset_wrapper.n_trunk_points_train,
                        sensor_x=self.dataset_wrapper.sensor_x_original,
                        scale=self.dataset_wrapper.scale,
                        input_range=self.dataset_wrapper.input_range,
                        device=self.device,
                        constant_zero=True
                    )
                    f_at_sensors, _, f_at_trunk, f_prime_at_trunk, batch_x_eval_points = batch_data
                    sensor_x_to_use = self.dataset_wrapper.sensor_x_original
                
                # Apply sensor dropout if specified
                sensor_x_used = sensor_x_to_use
                f_at_sensors_used = f_at_sensors
                
                if self.eval_sensor_dropoff > 0.0:
                    if self.dataset_wrapper.benchmark == 'derivative':
                        sensor_x_dropped, f_at_sensors_dropped = apply_sensor_dropoff(
                            sensor_x_to_use, f_at_sensors, self.eval_sensor_dropoff, self.replace_with_nearest
                        )
                        sensor_x_used = sensor_x_dropped
                        f_at_sensors_used = f_at_sensors_dropped
                    elif self.dataset_wrapper.benchmark == 'integral':
                        batch_x_eval_points_dropped, f_prime_at_trunk_dropped = apply_sensor_dropoff(
                            batch_x_eval_points, f_prime_at_trunk.T, self.eval_sensor_dropoff, self.replace_with_nearest
                        )
                        batch_x_eval_points = batch_x_eval_points_dropped
                        f_prime_at_trunk = f_prime_at_trunk_dropped.T
                
                # Prepare inputs based on benchmark (same logic as main script)
                if self.dataset_wrapper.benchmark == 'derivative':
                    xs = sensor_x_used.unsqueeze(0).expand(batch_size, -1, -1)
                    us = f_at_sensors_used.unsqueeze(-1)
                    ys = batch_x_eval_points.unsqueeze(0).expand(batch_size, -1, -1)
                    target = f_prime_at_trunk.T.unsqueeze(-1)
                elif self.dataset_wrapper.benchmark == 'integral':
                    xs = batch_x_eval_points.unsqueeze(0).expand(batch_size, -1, -1)
                    us = f_prime_at_trunk.T.unsqueeze(-1)
                    ys = sensor_x_used.unsqueeze(0).expand(batch_size, -1, -1)
                    target = f_at_sensors_used.unsqueeze(-1)
                else:
                    raise ValueError(f"Unknown benchmark: {self.dataset_wrapper.benchmark}")
                
                # Forward pass
                pred = model(xs, us, ys)
                
                # Calculate relative error for this batch
                rel_error = calculate_l2_relative_error(pred.squeeze(-1), target.squeeze(-1))
                total_rel_error += rel_error.item()
        
        model.train()
        return total_rel_error / n_eval_batches
    
    def _evaluate_with_dataset(self, model):
        """Evaluate using direct dataset access (elastic/chladni style)."""
        test_data = self.dataset['test']
        n_test = min(self.n_test_samples, len(test_data))
        
        total_rel_error = 0.0
        
        with torch.no_grad():
            for i in range(n_test):
                sample = test_data[i]
                
                # Load pre-normalized data
                xs_norm = torch.tensor(sample['X'], dtype=torch.float32, device=self.device)
                xs = xs_norm.unsqueeze(0)
                
                us_norm = torch.tensor(sample['u'], dtype=torch.float32, device=self.device).unsqueeze(0)
                us = us_norm.unsqueeze(-1)
                
                ys_norm = torch.tensor(sample['Y'], dtype=torch.float32, device=self.device)
                ys = ys_norm.unsqueeze(0)
                
                target_norm = torch.tensor(sample['s'], dtype=torch.float32, device=self.device).unsqueeze(0)
                target = target_norm.unsqueeze(-1)
                
                # Apply sensor dropout if specified
                xs_used = xs
                us_used = us
                if self.eval_sensor_dropoff > 0.0:
                    from Data.data_utils import apply_sensor_dropoff
                    
                    # Apply dropout to sensor data (remove batch dimension for dropout function)
                    xs_dropped, us_dropped = apply_sensor_dropoff(
                        xs.squeeze(0),  # Remove batch dimension: (n_sensors, 2)
                        us.squeeze(0).squeeze(-1),  # Remove batch and feature dimensions: (n_sensors,)
                        self.eval_sensor_dropoff,
                        self.replace_with_nearest
                    )
                    
                    # Add batch dimension back
                    xs_used = xs_dropped.unsqueeze(0)  # (1, n_remaining_sensors, 2)
                    us_used = us_dropped.unsqueeze(0).unsqueeze(-1)  # (1, n_remaining_sensors, 1)
                
                # Forward pass
                pred_norm = model(xs_used, us_used, ys)
                
                # Calculate relative error
                rel_error = calculate_l2_relative_error(pred_norm, target)
                total_rel_error += rel_error.item()
        
        model.train()
        return total_rel_error / n_test 