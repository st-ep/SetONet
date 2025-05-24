import torch
import torch.nn as nn
import os
from .SetONet import SetONet

def create_setonet_models(args, device):
    """
    Creates and initializes SetONet models (T and T_inv) based on arguments.
    
    Returns:
        tuple: (setonet_model_T, setonet_model_T_inv)
    """
    print("\n--- Initializing SetONet Models (Forward T and Inverse T_inv) ---")
    
    # Common arguments for both SetONet models
    setonet_common_args = dict(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=args.son_p_dim,
        phi_hidden_size=args.son_phi_hidden,
        rho_hidden_size=args.son_rho_hidden,
        trunk_hidden_size=args.son_trunk_hidden,
        n_trunk_layers=args.son_n_trunk_layers,
        activation_fn=nn.Tanh,
        use_deeponet_bias=True,
        phi_output_size=args.son_phi_output_size,
        pos_encoding_type=args.pos_encoding_type,
        aggregation_type=args.son_aggregation,
    )

    # Forward Model T: f(x_sensors) -> f'(y_trunk)
    setonet_model_T = SetONet(**setonet_common_args).to(device)

    # Inverse Model T_inv: f'(y_trunk) -> f(x_sensors)
    setonet_model_T_inv = SetONet(**setonet_common_args).to(device)
    
    return setonet_model_T, setonet_model_T_inv

def load_pretrained_models(setonet_model_T, setonet_model_T_inv, args, device):
    """
    Loads pre-trained models if paths are provided.
    
    Returns:
        bool: True if both models were loaded, False otherwise
    """
    models_loaded = False
    
    if args.load_model_T_path:
        if os.path.exists(args.load_model_T_path):
            setonet_model_T.load_state_dict(torch.load(args.load_model_T_path, map_location=device))
            print(f"Loaded pre-trained SetONet T model from: {args.load_model_T_path}")
            models_loaded = True
        else:
            print(f"Warning: Path for SetONet T model not found: {args.load_model_T_path}")
            args.load_model_T_path = None

    if args.load_model_T_inv_path:
        if os.path.exists(args.load_model_T_inv_path):
            setonet_model_T_inv.load_state_dict(torch.load(args.load_model_T_inv_path, map_location=device))
            print(f"Loaded pre-trained SetONet T_inv model from: {args.load_model_T_inv_path}")
            models_loaded = True
        else:
            print(f"Warning: Path for SetONet T_inv model not found: {args.load_model_T_inv_path}")
            args.load_model_T_inv_path = None
    
    return args.load_model_T_path and args.load_model_T_inv_path 