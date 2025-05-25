import torch

def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def format_parameter_count(param_count):
    """
    Format parameter count in a human-readable way.
    
    Args:
        param_count: Number of parameters
        
    Returns:
        str: Formatted string (e.g., "1.2M", "345K")
    """
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.1f}M"
    elif param_count >= 1_000:
        return f"{param_count / 1_000:.1f}K"
    else:
        return str(param_count)

def print_model_summary(model, model_name="Model"):
    """
    Print a summary of the model including parameter counts.
    
    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    param_info = count_parameters(model)
    
    print(f"\n--- {model_name} Summary ---")
    print(f"Total parameters: {param_info['total_parameters']:,} ({format_parameter_count(param_info['total_parameters'])})")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,} ({format_parameter_count(param_info['trainable_parameters'])})")
    if param_info['non_trainable_parameters'] > 0:
        print(f"Non-trainable parameters: {param_info['non_trainable_parameters']:,} ({format_parameter_count(param_info['non_trainable_parameters'])})")
    print("-" * 40) 