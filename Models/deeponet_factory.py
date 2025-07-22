import torch
import torch.nn as nn
import os

from .deeponet_model import DeepONet, MLP

def get_activation_function(activation_name):
    if activation_name == "relu":
        return nn.ReLU
    elif activation_name == "tanh":
        return nn.Tanh
    elif activation_name == "gelu":
        return nn.GELU
    elif activation_name == "swish":
        return nn.SiLU
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")

def create_deeponet_model(args, device, sensor_size, trunk_points, benchmark):
    print(f"\n--- Initializing DeepONet Model for {args.benchmark} ---")
    print(f"Using activation function: {args.activation_fn}")
    
    activation_fn = get_activation_function(args.activation_fn)
    
    # Set branch input dim based on benchmark
    if benchmark == 'derivative':
        branch_input_dim = sensor_size
    elif benchmark == 'integral':
        branch_input_dim = trunk_points
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    # Trunk network: input dim=1
    trunk_input_dim = 1
    trunk_hidden_dims = [args.son_trunk_hidden] * (args.son_n_trunk_layers - 1)
    trunk_output_dim = args.son_p_dim
    trunk_net = MLP(trunk_input_dim, trunk_hidden_dims, trunk_output_dim, activation_fn)
    
    # Branch network: input branch_input_dim
    branch_hidden_dims = [args.don_branch_hidden] * (args.don_n_branch_layers - 1)
    branch_net = MLP(branch_input_dim, branch_hidden_dims, args.son_p_dim, activation_fn)
    
    deeponet_model = DeepONet(branch_net, trunk_net).to(device)
    
    return deeponet_model

def load_pretrained_model(deeponet_model, args, device):
    if args.load_model_path:
        if os.path.exists(args.load_model_path):
            deeponet_model.load_state_dict(torch.load(args.load_model_path, map_location=device))
            print(f"Loaded pre-trained DeepONet model from: {args.load_model_path}")
            return True
        else:
            print(f"Warning: Model path not found: {args.load_model_path}")
            args.load_model_path = None
    
    return False 