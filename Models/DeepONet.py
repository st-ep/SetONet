import torch
import torch.nn as nn
from tqdm import trange

from .utils.helper_utils import calculate_l2_relative_error

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    """
    Deep Operator Network.
    """
    def __init__(self, branch_net, trunk_net):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net

    def forward(self, branch_input, trunk_input):
        # branch_input shape: [batch_size, num_sensor_features] (e.g., [B, S])
        # trunk_input shape: [num_eval_points, num_trunk_features] (e.g., [T, 1] or [B, T, 1])
        
        b = self.branch_net(branch_input)  # Output shape: [batch_size, p]
        
        # Handle different trunk_input shapes
        if trunk_input.ndim == 2: # Shape [T, trunk_feature_dim]
            # This is the typical case if trunk_input is shared across the batch
            # or if batch_size is 1 and trunk_input is [T, trunk_feature_dim]
            t = self.trunk_net(trunk_input)    # Output shape: [T, p]
            # We need to perform a batched dot product or element-wise product sum.
            # If b is [B, p] and t is [T, p], we want an output [B, T].
            # This requires t to be [p, T] for matmul, or careful broadcasting.
            # For (b_i * t_j).sum(), we can do b @ t.T
            out = torch.matmul(b, t.T) # Output shape: [B, T]
        elif trunk_input.ndim == 3: # Shape [B, T, trunk_feature_dim]
            # This case handles per-batch-item trunk inputs.
            batch_s, num_points, trunk_feat_dim = trunk_input.shape
            # Reshape trunk_input to be processed by MLP: [B*T, trunk_feature_dim]
            t_reshaped = trunk_input.reshape(batch_s * num_points, trunk_feat_dim)
            t_processed = self.trunk_net(t_reshaped) # Output shape: [B*T, p]
            # Reshape t_processed back: [B, T, p]
            t = t_processed.reshape(batch_s, num_points, -1)
            # Element-wise product and sum over p dimension
            # b is [B, p], needs to be [B, 1, p] for broadcasting with t [B, T, p]
            out = torch.sum(b.unsqueeze(1) * t, dim=-1) # Output shape: [B, T]
        else:
            raise ValueError(f"Unsupported trunk_input ndim: {trunk_input.ndim}")
            
        return out 

class DeepONetWrapper(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim=1, p=32, trunk_hidden_size=256, n_trunk_layers=4, branch_hidden_size=256, n_branch_layers=4, activation_fn=nn.ReLU, initial_lr=5e-4, lr_schedule_steps=None, lr_schedule_gammas=None, use_deeponet_bias=True):
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.p = p
        
        # Determine trunk input dimension
        effective_trunk_input_dim = trunk_input_dim
        
        # Branch net: input is fixed size vector of sensor values
        branch_hidden_dims = [branch_hidden_size] * n_branch_layers
        self.branch_net = MLP(branch_input_dim, branch_hidden_dims, p, activation_fn)
        
        # Trunk net
        trunk_hidden_dims = [trunk_hidden_size] * n_trunk_layers
        self.trunk_net = MLP(effective_trunk_input_dim, trunk_hidden_dims, p, activation_fn)
        
        self.deeponet = DeepONet(self.branch_net, self.trunk_net)
        
        # Optimizer and training params
        self.opt = torch.optim.Adam(self.parameters(), lr=initial_lr)
        self.lr_schedule_steps = lr_schedule_steps or []
        self.lr_schedule_gammas = lr_schedule_gammas or []
        self.total_steps = 0
    
    def _update_lr(self):
        current_lr = self.opt.param_groups[0]['lr']
        if self.lr_schedule_steps:
            for i, step in enumerate(self.lr_schedule_steps):
                if self.total_steps == step:
                    gamma = self.lr_schedule_gammas[i]
                    current_lr *= gamma
                    for param_group in self.opt.param_groups:
                        param_group['lr'] = current_lr
                    break
        return current_lr
    
    def forward(self, xs, us, ys):
        # Ignore xs, as DeepONet doesn't use sensor locations explicitly
        branch_input = us.squeeze(-1)  # [B, S_current]
        
        # Pad or truncate branch_input to match expected dimension
        if branch_input.shape[1] != self.branch_input_dim:
            if branch_input.shape[1] < self.branch_input_dim:
                padding = self.branch_input_dim - branch_input.shape[1]
                branch_input = torch.nn.functional.pad(branch_input, (0, padding), mode='constant', value=0)
            else:
                branch_input = branch_input[:, :self.branch_input_dim]
        
        trunk_input = ys  # [B, Q, 1]
        
        out = self.deeponet(branch_input, trunk_input)
        return out.unsqueeze(-1)  # [B, Q, 1]
    
    def train_model(self, dataset, epochs, progress_bar=True, callback=None, log_dir=None):
        device = next(self.parameters()).device
        if callback is not None:
            callback.on_training_start(locals())
        
        # Initialize TensorBoard writer if log_dir is provided
        writer = None
        if log_dir:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = f"{log_dir}/tensorboard"
            writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard logging to: {tensorboard_dir}")
        
        bar = trange(epochs, disable=not progress_bar)
        for epoch in bar:
            current_lr = self._update_lr()
            
            xs, us, ys, G_u_ys, _ = dataset.sample(device=device)
            
            estimated_G_u_ys = self.forward(xs, us, ys)
            prediction_loss = torch.nn.MSELoss()(estimated_G_u_ys, G_u_ys)
            
            with torch.no_grad():
                pred_flat = estimated_G_u_ys.squeeze(-1)
                target_flat = G_u_ys.squeeze(-1)
                rel_l2_error = calculate_l2_relative_error(pred_flat, target_flat)
            
            loss = prediction_loss
            
            self.opt.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
            self.opt.step()
            
            self.total_steps += 1
            
            # Enhanced TensorBoard logging
            if writer:
                # Log training metrics more frequently for smooth curves
                if epoch % 10 == 0:  # Log every 10 epochs for smoother curves
                    writer.add_scalar('Training/MSE_Loss', loss.item(), epoch)
                    writer.add_scalar('Training/Relative_L2_Error', rel_l2_error.item(), epoch)
                    writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
                    writer.add_scalar('Training/Gradient_Norm', norm, epoch)
                    
                    # Log target and prediction statistics for monitoring
                    writer.add_scalar('Training/Target_Mean', target_flat.mean().item(), epoch)
                    writer.add_scalar('Training/Target_Std', target_flat.std().item(), epoch)
                    writer.add_scalar('Training/Prediction_Mean', pred_flat.mean().item(), epoch)
                    writer.add_scalar('Training/Prediction_Std', pred_flat.std().item(), epoch)
            
            bar.set_description(f"Step {self.total_steps} | Loss: {loss.item():.4e} | Rel L2: {rel_l2_error.item():.4f} | Grad Norm: {norm:.2f} | LR: {current_lr:.2e}")
            
            if callback is not None:
                callback.on_step(locals())
        
        # Close TensorBoard writer
        if writer:
            writer.close()
            print(f"\nTensorBoard logs saved. To view, run: tensorboard --logdir={tensorboard_dir}")
        
        if callback is not None:
            callback.on_training_end(locals()) 