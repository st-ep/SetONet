import torch

def sample_trunk_points(n_points, input_range, device):
    """
    Generates linearly spaced points within the specified input_range on the given device.
    """
    return torch.linspace(input_range[0], input_range[1], n_points, device=device).view(-1, 1)

def sample_variable_sensor_points(sensor_size, input_range, device):
    """
    Generates random sensor points for variable sensor training.
    """
    sensor_x = torch.rand(sensor_size, device=device) * (input_range[1] - input_range[0]) + input_range[0]
    sensor_x = sensor_x.sort()[0]  # Sort for consistency
    return sensor_x.view(-1, 1)

def apply_sensor_dropoff(sensor_x, sensor_values, dropoff_rate, replace_with_nearest=False, random_seed=None):
    """
    Randomly drops sensors based on the specified drop-off rate.
    
    Args:
        sensor_x: Sensor locations [n_sensors, coord_dim] (1D, 2D, or higher dimensional)
        sensor_values: Sensor values [batch_size, n_sensors] or [n_sensors]
        dropoff_rate: Fraction of sensors to drop (0.0 to 1.0)
        replace_with_nearest: If True, replace dropped sensors with nearest remaining sensors
                             instead of removing them entirely (creates duplicate sensor pairs)
                             Uses Euclidean distance for multi-dimensional coordinates
    
    Returns:
        Tuple of (processed_sensor_x, processed_sensor_values)
    """
    if dropoff_rate == 0.0:
        return sensor_x, sensor_values
    
    n_sensors = sensor_x.shape[0]
    n_keep = max(1, int(n_sensors * (1.0 - dropoff_rate)))  # Keep at least 1 sensor
    
    if not replace_with_nearest:
        # Original behavior: just remove dropped sensors entirely
        keep_indices = torch.randperm(n_sensors, device=sensor_x.device)[:n_keep]
        keep_indices = keep_indices.sort()[0]  # Sort to maintain order
        
        dropped_sensor_x = sensor_x[keep_indices]
        
        if sensor_values.dim() == 1:
            dropped_sensor_values = sensor_values[keep_indices]
        else:
            dropped_sensor_values = sensor_values[:, keep_indices]
        
        return dropped_sensor_x, dropped_sensor_values
    
    else:
        # Leverage permutation invariance: replace dropped sensors with nearest remaining sensors
        # This creates duplicate (location, value) pairs to maintain the same input size
        
        # Get indices efficiently
        all_indices = torch.randperm(n_sensors, device=sensor_x.device)
        keep_indices = all_indices[:n_keep]
        drop_indices = all_indices[n_keep:]
        
        if len(drop_indices) == 0:
            return sensor_x, sensor_values
        
        # Clone original tensors to avoid modifying inputs
        processed_sensor_x = sensor_x.clone()
        processed_sensor_values = sensor_values.clone()
        
        # Vectorized nearest neighbor replacement for (location, value) pairs
        keep_positions = sensor_x[keep_indices]  # [n_keep, coord_dim]
        drop_positions = sensor_x[drop_indices]  # [n_drop, coord_dim]
        
        # Calculate distances: [n_drop, n_keep]
        if sensor_x.shape[1] == 1:
            # 1D coordinates: use absolute distance
            distances = torch.abs(drop_positions.unsqueeze(1) - keep_positions.unsqueeze(0)).squeeze(-1)
        else:
            # Multi-dimensional coordinates: use Euclidean distance
            # drop_positions: [n_drop, coord_dim] -> [n_drop, 1, coord_dim]
            # keep_positions: [n_keep, coord_dim] -> [1, n_keep, coord_dim]
            diff = drop_positions.unsqueeze(1) - keep_positions.unsqueeze(0)  # [n_drop, n_keep, coord_dim]
            distances = torch.norm(diff, dim=2)  # [n_drop, n_keep]
        
        # Find nearest keep indices for each drop index: [n_drop]
        nearest_keep_local_indices = torch.argmin(distances, dim=1)
        nearest_keep_global_indices = keep_indices[nearest_keep_local_indices]
        
        # Debug: Print replacement info for first few sensors
        if len(drop_indices) > 0 and torch.rand(1).item() < 0.0:  # Set to 0.0 to disable
            print(f"\nSensor replacement debug:")
            print(f"Dropped sensor indices: {drop_indices.tolist()}")
            print(f"Nearest remaining indices: {nearest_keep_global_indices.tolist()}")
            for i, (drop_idx, nearest_idx) in enumerate(zip(drop_indices, nearest_keep_global_indices)):
                if i < 3:  # Only show first 3 for brevity
                    orig_pos = sensor_x[drop_idx]
                    nearest_pos = sensor_x[nearest_idx]
                    if sensor_x.shape[1] == 1:
                        print(f"  Sensor {drop_idx.item()} at x={orig_pos.item():.3f} -> replaced with sensor {nearest_idx.item()} at x={nearest_pos.item():.3f}")
                    else:
                        print(f"  Sensor {drop_idx.item()} at {orig_pos.tolist()} -> replaced with sensor {nearest_idx.item()} at {nearest_pos.tolist()}")
        
        # Replace BOTH positions and values as pairs (this leverages permutation invariance)
        # Example: if sensor 2 is dropped and sensor 3 is nearest:
        # Original: [(x1,u1), (x2,u2), (x3,u3)] -> Result: [(x1,u1), (x3,u3), (x3,u3)]
        processed_sensor_x[drop_indices] = sensor_x[nearest_keep_global_indices]
        
        if sensor_values.dim() == 1:
            # For 1D sensor values: [n_sensors]
            processed_sensor_values[drop_indices] = sensor_values[nearest_keep_global_indices]
        else:
            # For batched sensor values: [batch_size, n_sensors]
            processed_sensor_values[:, drop_indices] = sensor_values[:, nearest_keep_global_indices]
        
        return processed_sensor_x, processed_sensor_values

def apply_sensor_dropoff_with_interpolation(sensor_x, sensor_values, dropoff_rate, random_seed=None):
    """
    Drops sensors randomly but interpolates missing values to maintain fixed input size.
    This is specifically designed for DeepONet which requires fixed-size branch input.
    
    Args:
        sensor_x: Sensor locations [n_sensors, coord_dim] (1D coordinates)
        sensor_values: Sensor values [batch_size, n_sensors] or [n_sensors]
        dropoff_rate: Fraction of sensors to drop (0.0 to 1.0)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (original_sensor_x, interpolated_sensor_values)
        - original_sensor_x: unchanged sensor locations (to maintain fixed size)
        - interpolated_sensor_values: values with dropped sensors replaced by interpolation
    """
    if dropoff_rate == 0.0:
        return sensor_x, sensor_values
    
    device = sensor_x.device
    n_sensors = sensor_x.shape[0]
    n_keep = max(1, int(n_sensors * (1.0 - dropoff_rate)))  # Keep at least 1 sensor
    
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Randomly select which sensors to keep
    all_indices = torch.randperm(n_sensors, device=device)
    keep_indices = all_indices[:n_keep].sort()[0]  # Sort to maintain order
    drop_indices = all_indices[n_keep:]
    
    if len(drop_indices) == 0:
        return sensor_x, sensor_values
    
    # Clone sensor values to avoid modifying input
    interpolated_values = sensor_values.clone()
    
    # Get coordinates for interpolation (assuming 1D sensors)
    sensor_coords = sensor_x.squeeze(-1)  # [n_sensors] - flatten to 1D coordinates
    keep_coords = sensor_coords[keep_indices]  # [n_keep]
    drop_coords = sensor_coords[drop_indices]  # [n_drop]
    
    # Helper function for linear interpolation (compatible with older PyTorch versions)
    def linear_interpolate(x_query, x_known, y_known):
        """
        Linear interpolation for a single query point.
        Args:
            x_query: single coordinate to interpolate at
            x_known: [n_known] coordinates of known points (must be sorted)
            y_known: [n_known] values at known coordinates
        Returns:
            interpolated value at x_query
        """
        # Handle edge cases
        if x_query <= x_known[0]:
            return y_known[0]
        if x_query >= x_known[-1]:
            return y_known[-1]
        
        # Find the two points to interpolate between
        idx = torch.searchsorted(x_known, x_query)
        if idx == 0:
            idx = 1
        
        x0, x1 = x_known[idx-1], x_known[idx]
        y0, y1 = y_known[idx-1], y_known[idx]
        
        # Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        weight = (x_query - x0) / (x1 - x0)
        return y0 + weight * (y1 - y0)
    
    if sensor_values.dim() == 1:
        # Handle 1D sensor values: [n_sensors]
        keep_values = sensor_values[keep_indices]  # [n_keep]
        
        # Interpolate missing values using linear interpolation
        for i, drop_coord in enumerate(drop_coords):
            interpolated_val = linear_interpolate(drop_coord, keep_coords, keep_values)
            interpolated_values[drop_indices[i]] = interpolated_val
            
    else:
        # Handle batched sensor values: [batch_size, n_sensors]
        keep_values = sensor_values[:, keep_indices]  # [batch_size, n_keep]
        
        # Interpolate for each batch item
        for batch_idx in range(sensor_values.shape[0]):
            batch_keep_values = keep_values[batch_idx]  # [n_keep]
            
            for i, drop_coord in enumerate(drop_coords):
                interpolated_val = linear_interpolate(drop_coord, keep_coords, batch_keep_values)
                interpolated_values[batch_idx, drop_indices[i]] = interpolated_val
    
    return sensor_x, interpolated_values

def apply_sensor_dropoff_with_2d_interpolation(sensor_coords, sensor_values, grid_shape, dropoff_rate, random_seed=None):
    """
    Drops sensors randomly but interpolates missing values using 2D bilinear interpolation to maintain fixed input size.
    This is specifically designed for DeepONet with 2D problems on regular grids (like Chladni plates).
    
    Args:
        sensor_coords: Sensor coordinates [n_sensors, 2] (2D coordinates on regular grid)
        sensor_values: Sensor values [batch_size, n_sensors] or [n_sensors]
        grid_shape: Tuple (nx, ny) representing the regular grid dimensions
        dropoff_rate: Fraction of sensors to drop (0.0 to 1.0)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        Tuple of (original_sensor_coords, interpolated_sensor_values)
        - original_sensor_coords: unchanged sensor coordinates (to maintain fixed size)
        - interpolated_sensor_values: values with dropped sensors replaced by bilinear interpolation
    """
    if dropoff_rate == 0.0:
        return sensor_coords, sensor_values
    
    device = sensor_coords.device
    n_sensors = sensor_coords.shape[0]
    n_keep = max(1, int(n_sensors * (1.0 - dropoff_rate)))  # Keep at least 1 sensor
    
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    # Randomly select which sensors to keep
    all_indices = torch.randperm(n_sensors, device=device)
    keep_indices = all_indices[:n_keep].sort()[0]  # Sort to maintain order
    drop_indices = all_indices[n_keep:]
    
    if len(drop_indices) == 0:
        return sensor_coords, sensor_values
    
    # Clone sensor values to avoid modifying input
    interpolated_values = sensor_values.clone()
    
    # Extract 2D coordinates
    nx, ny = grid_shape
    coords_x = sensor_coords[:, 0]  # [n_sensors]
    coords_y = sensor_coords[:, 1]  # [n_sensors]
    
    # Get coordinates of kept and dropped sensors
    keep_coords_x = coords_x[keep_indices]  # [n_keep]
    keep_coords_y = coords_y[keep_indices]  # [n_keep]
    drop_coords_x = coords_x[drop_indices]  # [n_drop]
    drop_coords_y = coords_y[drop_indices]  # [n_drop]
    
    # Helper function for true bilinear interpolation on regular grid
    def bilinear_interpolate(query_x, query_y, known_x, known_y, known_values, grid_shape):
        """
        True bilinear interpolation for a regular grid.
        Args:
            query_x, query_y: coordinates to interpolate at
            known_x, known_y: [n_known] coordinates of known points
            known_values: [n_known] values at known coordinates
            grid_shape: (nx, ny) dimensions of the regular grid
        Returns:
            interpolated value at (query_x, query_y)
        """
        nx, ny = grid_shape
        
        # Handle exact matches first
        exact_match_mask = (torch.abs(known_x - query_x) < 1e-10) & (torch.abs(known_y - query_y) < 1e-10)
        if torch.any(exact_match_mask):
            exact_idx = torch.where(exact_match_mask)[0][0]
            return known_values[exact_idx]
        
        # For regular grid, we need to find the 4 surrounding grid points
        # First, determine the grid spacing from known points
        x_coords_unique = torch.unique(known_x, sorted=True)
        y_coords_unique = torch.unique(known_y, sorted=True)
        
        if len(x_coords_unique) < 2 or len(y_coords_unique) < 2:
            # Fallback to nearest neighbor if we don't have enough points for bilinear
            distances_sq = (known_x - query_x)**2 + (known_y - query_y)**2
            nearest_idx = torch.argmin(distances_sq)
            return known_values[nearest_idx]
        
        # Find the surrounding grid cell
        # Handle boundary cases by clamping
        x_min_val, x_max_val = x_coords_unique[0], x_coords_unique[-1]
        y_min_val, y_max_val = y_coords_unique[0], y_coords_unique[-1]
        
        # Clamp query point to grid bounds
        query_x_clamped = torch.clamp(query_x, x_min_val, x_max_val)
        query_y_clamped = torch.clamp(query_y, y_min_val, y_max_val)
        
        # Find grid cell indices
        x_below_idx = torch.searchsorted(x_coords_unique, query_x_clamped, right=False)
        y_below_idx = torch.searchsorted(y_coords_unique, query_y_clamped, right=False)
        
        # Handle edge cases
        if x_below_idx == 0:
            x_below_idx = 1
        if y_below_idx == 0:
            y_below_idx = 1
        if x_below_idx >= len(x_coords_unique):
            x_below_idx = len(x_coords_unique) - 1
        if y_below_idx >= len(y_coords_unique):
            y_below_idx = len(y_coords_unique) - 1
        
        # Get the 4 corner coordinates
        x0, x1 = x_coords_unique[x_below_idx-1], x_coords_unique[x_below_idx]
        y0, y1 = y_coords_unique[y_below_idx-1], y_coords_unique[y_below_idx]
        
        # Find values at the 4 corners by searching in known points
        def find_value_at_coord(target_x, target_y):
            mask = (torch.abs(known_x - target_x) < 1e-10) & (torch.abs(known_y - target_y) < 1e-10)
            if torch.any(mask):
                return known_values[torch.where(mask)[0][0]]
            else:
                # If exact corner not found, use nearest neighbor
                distances = (known_x - target_x)**2 + (known_y - target_y)**2
                return known_values[torch.argmin(distances)]
        
        # Get values at 4 corners
        f00 = find_value_at_coord(x0, y0)  # bottom-left
        f10 = find_value_at_coord(x1, y0)  # bottom-right
        f01 = find_value_at_coord(x0, y1)  # top-left
        f11 = find_value_at_coord(x1, y1)  # top-right
        
        # Avoid division by zero
        if torch.abs(x1 - x0) < 1e-10:
            # Vertical line, interpolate in y direction only
            if torch.abs(y1 - y0) < 1e-10:
                return f00  # Single point
            t_y = (query_y_clamped - y0) / (y1 - y0)
            return f00 * (1 - t_y) + f01 * t_y
        elif torch.abs(y1 - y0) < 1e-10:
            # Horizontal line, interpolate in x direction only
            t_x = (query_x_clamped - x0) / (x1 - x0)
            return f00 * (1 - t_x) + f10 * t_x
        
        # Full bilinear interpolation
        # Normalize coordinates to [0,1] within the grid cell
        t_x = (query_x_clamped - x0) / (x1 - x0)
        t_y = (query_y_clamped - y0) / (y1 - y0)
        
        # Bilinear interpolation formula
        interpolated = (f00 * (1 - t_x) * (1 - t_y) +
                       f10 * t_x * (1 - t_y) +
                       f01 * (1 - t_x) * t_y +
                       f11 * t_x * t_y)
        
        return interpolated
    
    if sensor_values.dim() == 1:
        # Handle 1D sensor values: [n_sensors]
        keep_values = sensor_values[keep_indices]  # [n_keep]
        
        # Interpolate missing values using 2D bilinear interpolation
        for i, (drop_x, drop_y) in enumerate(zip(drop_coords_x, drop_coords_y)):
            interpolated_val = bilinear_interpolate(drop_x, drop_y, keep_coords_x, keep_coords_y, keep_values, grid_shape)
            interpolated_values[drop_indices[i]] = interpolated_val
            
    else:
        # Handle batched sensor values: [batch_size, n_sensors]
        keep_values = sensor_values[:, keep_indices]  # [batch_size, n_keep]
        
        # Interpolate for each batch item
        for batch_idx in range(sensor_values.shape[0]):
            batch_keep_values = keep_values[batch_idx]  # [n_keep]
            
            for i, (drop_x, drop_y) in enumerate(zip(drop_coords_x, drop_coords_y)):
                interpolated_val = bilinear_interpolate(drop_x, drop_y, keep_coords_x, keep_coords_y, batch_keep_values, grid_shape)
                interpolated_values[batch_idx, drop_indices[i]] = interpolated_val
    
    return sensor_coords, interpolated_values

def generate_batch(batch_size, n_trunk_points, sensor_x, scale, input_range, device, constant_zero=True, variable_sensors=False, sensor_size=None):
    """
    Generates a batch of polynomial functions and their derivatives.
    
    Args:
        batch_size: Number of samples in the batch
        n_trunk_points: Number of trunk evaluation points
        sensor_x: Fixed sensor locations (None if variable_sensors=True)
        scale: Scale for polynomial coefficients
        input_range: Input domain range
        device: PyTorch device
        constant_zero: Whether to set integration constant to zero
        variable_sensors: Whether to use different sensor locations per batch (not per sample)
        sensor_size: Number of sensor points (required if variable_sensors=True)
    
    Returns:
        f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval, [sensor_x_batch]
    """
    # Generate trunk points
    x_eval = sample_trunk_points(n_trunk_points, input_range, device)
    
    # Handle sensor locations
    if variable_sensors:
        if sensor_size is None:
            raise ValueError("sensor_size must be provided when variable_sensors=True")
        # Generate ONE set of sensor locations for the entire batch
        sensor_x_to_use = sample_variable_sensor_points(sensor_size, input_range, device)
    else:
        # Use fixed sensor locations for all samples
        if sensor_x is None:
            raise ValueError("sensor_x must be provided when variable_sensors=False")
        sensor_x_to_use = sensor_x
    
    # Generate polynomial coefficients
    a = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    b = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    c = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    if constant_zero:
        d = torch.zeros(batch_size, device=device)
    else:
        d = (torch.rand(batch_size, device=device) * 2 - 1) * scale
    
    # Vectorized evaluation at sensor points (much faster!)
    # sensor_x_to_use: [sensor_size, 1]
    # a, b, c, d: [batch_size]
    # Result: [batch_size, sensor_size]
    sensor_x_expanded = sensor_x_to_use.T  # [1, sensor_size]
    
    f_at_sensors = (a.unsqueeze(1) * sensor_x_expanded**3 + 
                   b.unsqueeze(1) * sensor_x_expanded**2 + 
                   c.unsqueeze(1) * sensor_x_expanded + 
                   d.unsqueeze(1))  # [batch_size, sensor_size]
    
    f_prime_at_sensors = (3 * a.unsqueeze(1) * sensor_x_expanded**2 + 
                         2 * b.unsqueeze(1) * sensor_x_expanded + 
                         c.unsqueeze(1))  # [batch_size, sensor_size]
    
    # Vectorized evaluation at trunk points (same as before)
    f_at_trunk = a.unsqueeze(1) * x_eval.T**3 + b.unsqueeze(1) * x_eval.T**2 + c.unsqueeze(1) * x_eval.T + d.unsqueeze(1)
    f_prime_at_trunk = 3 * a.unsqueeze(1) * x_eval.T**2 + 2 * b.unsqueeze(1) * x_eval.T + c.unsqueeze(1)
    
    f_at_trunk = f_at_trunk.T  # [n_trunk_points, batch_size]
    f_prime_at_trunk = f_prime_at_trunk.T  # [n_trunk_points, batch_size]
    
    if variable_sensors:
        return f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval, sensor_x_to_use
    else:
        return f_at_sensors, f_prime_at_sensors, f_at_trunk, f_prime_at_trunk, x_eval 