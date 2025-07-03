import torch
import numpy as np
import sys 
import os 
from datetime import datetime 
import argparse
from datasets import load_from_disk
from tqdm import trange

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import required modules
from Models.deeponet_model import DeepONet, MLP
import torch.nn as nn
from Plotting.plot_darcy_2d_utils import plot_multiple_darcy_results_deeponet

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train DeepONet for 2D Darcy flow problem.")
    
    parser.add_argument('--data_path', type=str, default="/home/titanv/Stepan/setprojects/SetONet/Data/darcy_2d_data/darcy64", help='Path to Darcy 2D dataset')
    parser.add_argument('--don_p_dim', type=int, default=64, help='Latent dimension p')
    parser.add_argument('--don_branch_hidden', type=int, default=256, help='Branch network hidden size')
    parser.add_argument('--don_trunk_hidden', type=int, default=256, help='Trunk network hidden size')
    parser.add_argument('--don_n_branch_layers', type=int, default=3, help='Branch network layers')
    parser.add_argument('--don_n_trunk_layers', type=int, default=4, help='Trunk network layers')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function')
    parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--don_epochs', type=int, default=9000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[1000, 3000, 5000], help="LR decay steps")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.5, 0.5, 0.5], help="LR decay factors")
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained model')
    
    return parser.parse_args()

def setup_logging(project_root):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(project_root, "logs", "DeepONet_darcy2d", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def get_activation_function(activation_name):
    activation_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'swish': nn.SiLU}
    return activation_map.get(activation_name.lower(), nn.ReLU)

class DarcyDatasetDeepONet:
    """Dataset wrapper for 2D Darcy flow data for DeepONet."""
    
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("📊 Loading Darcy 2D dataset for DeepONet...")
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        # Get dimensions from first sample
        sample_0 = train_data[0]
        k_field = np.array(sample_0['k'])  # permeability field (65x65)
        
        self.grid_size = k_field.shape[0]  # Should be 65 for 64x64 grid
        self.n_points = self.grid_size * self.grid_size  # Total number of points
        
        print(f"Grid size: {self.grid_size}x{self.grid_size}, Total points: {self.n_points}")
        
        # Create coordinate grid (same for all samples)
        x = np.linspace(0.0, 1.0, self.grid_size)
        y = np.linspace(0.0, 1.0, self.grid_size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
        self.coords = torch.tensor(coords, device=device, dtype=torch.float32)
        
        # Pre-allocate tensors on GPU for ALL data
        self.k_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        self.p_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        
        # Load data to GPU (flatten the 2D fields)
        for i in range(self.n_samples):
            sample = train_data[i]
            self.k_data[i] = torch.tensor(np.array(sample['k']).flatten(), device=device, dtype=torch.float32)
            self.p_data[i] = torch.tensor(np.array(sample['p']).flatten(), device=device, dtype=torch.float32)
        
        print(f"✅ Dataset loaded: {self.n_samples} samples, {self.n_points} points per sample")
        
    def sample(self, device=None):
        """Sample a batch for DeepONet training."""
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        
        # Branch input: permeability values (flattened k field)
        branch_input = self.k_data[indices]  # (batch_size, n_points)
        
        # Trunk input: coordinates (same for all samples in batch)
        trunk_input = self.coords.unsqueeze(0).expand(self.batch_size, -1, -1)  # (batch_size, n_points, 2)
        
        # Target: pressure values (flattened p field)
        target = self.p_data[indices]  # (batch_size, n_points)
        
        return branch_input, trunk_input, target

def create_model(args, device):
    activation_fn = get_activation_function(args.activation_fn)
    
    # Branch network: processes flattened permeability field
    branch_net = MLP(4225, [args.don_branch_hidden] * args.don_n_branch_layers, args.don_p_dim, activation_fn)
    
    # Trunk network: processes 2D coordinates
    trunk_net = MLP(2, [args.don_trunk_hidden] * args.don_n_trunk_layers, args.don_p_dim, activation_fn)
    
    return DeepONet(branch_net, trunk_net).to(device)

def train_model(model, dataset, args, device, log_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.don_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_schedule_steps, gamma=0.5)
    criterion = nn.MSELoss()
    
    model.train()
    epoch_losses = []
    
    bar = trange(args.don_epochs)
    for epoch in bar:
        current_lr = optimizer.param_groups[0]['lr']
        branch_input, trunk_input, target = dataset.sample(device=device)
        
        optimizer.zero_grad()
        pred = model(branch_input, trunk_input)
        loss = criterion(pred, target)
        
        with torch.no_grad():
            rel_l2_error = torch.norm(pred - target) / torch.norm(target)
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_losses.append(loss.item())
        bar.set_description(f"Step {epoch + 1} | Loss: {loss.item():.4e} | Rel L2: {rel_l2_error.item():.4f} | Grad Norm: {grad_norm:.2f} | LR: {current_lr:.2e}")
        
        if epoch % 1000 == 0 and epoch > 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': loss.item()}, 
                      os.path.join(log_dir, f"deeponet_checkpoint_epoch_{epoch}.pth"))
    
    return epoch_losses

def evaluate_model(model, dataset, darcy_dataset, device, n_test_samples=100):
    model.eval()
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))
    total_loss = total_rel_error = 0.0
    
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            
            # Prepare test data
            k_field = torch.tensor(np.array(sample['k']).flatten(), dtype=torch.float32, device=device)
            p_field = torch.tensor(np.array(sample['p']).flatten(), dtype=torch.float32, device=device)
            
            # Branch input: permeability field
            branch_input = k_field.unsqueeze(0)  # (1, n_points)
            
            # Trunk input: coordinates
            trunk_input = darcy_dataset.coords.unsqueeze(0)  # (1, n_points, 2)
            
            # Target: pressure field
            target = p_field.unsqueeze(0)  # (1, n_points)
            
            pred = model(branch_input, trunk_input)
            total_loss += torch.nn.MSELoss()(pred, target).item()
            total_rel_error += (torch.norm(pred - target) / torch.norm(target)).item()
    
    avg_loss, avg_rel_error = total_loss / n_test, total_rel_error / n_test
    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    
    model.train()
    return avg_loss, avg_rel_error

def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    log_dir = setup_logging(project_root)
    
    # Load dataset
    print(f"Loading dataset from: {args.data_path}")
    try:
        # Load train and test datasets separately
        from datasets import DatasetDict
        train_dataset = load_from_disk(os.path.join(args.data_path, "train"))
        test_dataset = load_from_disk(os.path.join(args.data_path, "test"))
        
        # Combine into DatasetDict
        dataset = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Run: python Data/darcy_2d_data/make_dataset.py")
        return
    
    darcy_dataset = DarcyDatasetDeepONet(dataset, batch_size=args.batch_size, device=device)
    model = create_model(args, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("Pre-trained model loaded")
    
    print(f"\nStarting training for {args.don_epochs} epochs...")
    train_model(model, darcy_dataset, args, device, log_dir)
    
    print("\nEvaluating model...")
    evaluate_model(model, dataset, darcy_dataset, device, n_test_samples=100)
    
    model_save_path = os.path.join(log_dir, "deeponet_darcy2d_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    print("Training completed!")

if __name__ == "__main__":
    main() 