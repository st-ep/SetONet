import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.Tanh):
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