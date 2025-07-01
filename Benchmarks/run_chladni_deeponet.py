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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train DeepONet for Chladni plate problem.")
    
    parser.add_argument('--data_path', type=str, default="/home/titanv/Stepan/setprojects/SetONet/Data/chladni_dataset", help='Path to dataset')
    parser.add_argument('--don_p_dim', type=int, default=64, help='Latent dimension p')
    parser.add_argument('--don_branch_hidden', type=int, default=256, help='Branch network hidden size')
    parser.add_argument('--don_trunk_hidden', type=int, default=256, help='Trunk network hidden size')
    parser.add_argument('--don_n_branch_layers', type=int, default=3, help='Branch network layers')
    parser.add_argument('--don_n_trunk_layers', type=int, default=4, help='Trunk network layers')
    parser.add_argument('--activation_fn', type=str, default="relu", choices=["relu", "tanh", "gelu", "swish"], help='Activation function')
    parser.add_argument('--don_lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--don_epochs', type=int, default=175000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument("--lr_schedule_steps", type=int, nargs='+', default=[25000, 75000, 125000, 175000], help="LR decay steps")
    parser.add_argument("--lr_schedule_gammas", type=float, nargs='+', default=[0.2, 0.5, 0.2, 0.5], help="LR decay factors")
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to pre-trained model')
    
    return parser.parse_args()

def setup_logging(project_root):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(project_root, "logs", "DeepONet_chladni", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir

def get_activation_function(activation_name):
    activation_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'swish': nn.SiLU}
    return activation_map.get(activation_name.lower(), nn.ReLU)

class ChladniDatasetDeepONet:
    def __init__(self, dataset, batch_size=64, device='cuda'):
        print("📊 Loading dataset...")
        self.batch_size = batch_size
        self.device = device
        train_data = dataset['train']
        self.n_samples = len(train_data)
        
        sample_0 = train_data[0]
        self.n_points = len(sample_0['X'])
        self.input_dim = len(sample_0['X'][0])
        
        # Pre-allocate and load all data to GPU
        self.X_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.u_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        self.Y_data = torch.zeros(self.n_samples, self.n_points, self.input_dim, device=device, dtype=torch.float32)
        self.s_data = torch.zeros(self.n_samples, self.n_points, device=device, dtype=torch.float32)
        
        for i in range(self.n_samples):
            sample = train_data[i]
            self.X_data[i] = torch.tensor(sample['X'], device=device, dtype=torch.float32)
            self.u_data[i] = torch.tensor(sample['u'], device=device, dtype=torch.float32)
            self.Y_data[i] = torch.tensor(sample['Y'], device=device, dtype=torch.float32)
            self.s_data[i] = torch.tensor(sample['s'], device=device, dtype=torch.float32)
        
        # Normalize data
        self.u_mean, self.u_std = self.u_data.mean(), self.u_data.std()
        self.s_mean, self.s_std = self.s_data.mean(), self.s_data.std()
        self.u_data_norm = (self.u_data - self.u_mean) / (self.u_std + 1e-8)
        self.s_data_norm = (self.s_data - self.s_mean) / (self.s_std + 1e-8)
        
        self.xy_mean = self.X_data.mean(dim=(0,1))
        self.xy_std = self.X_data.std(dim=(0,1)) + 1e-8
        self.X_data_norm = (self.X_data - self.xy_mean) / self.xy_std
        self.Y_data_norm = (self.Y_data - self.xy_mean) / self.xy_std
        print(f"✅ Dataset loaded: {self.n_samples} samples, {self.n_points} points")
        
    def sample(self, device=None):
        indices = torch.randint(0, self.n_samples, (self.batch_size,), device=self.device)
        return self.u_data_norm[indices], self.Y_data_norm[indices], self.s_data_norm[indices]
    
    def denormalize_displacement(self, s_norm):
        return s_norm * (self.s_std + 1e-8) + self.s_mean

def create_model(args, device):
    activation_fn = get_activation_function(args.activation_fn)
    
    branch_net = MLP(625, [args.don_branch_hidden] * args.don_n_branch_layers, args.don_p_dim, activation_fn)
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
        
        if epoch % 25000 == 0 and epoch > 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': loss.item()}, 
                      os.path.join(log_dir, f"deeponet_checkpoint_epoch_{epoch}.pth"))
    
    return epoch_losses

def evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100):
    model.eval()
    test_data = dataset['test']
    n_test = min(n_test_samples, len(test_data))
    total_loss = total_rel_error = 0.0
    
    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]
            us_orig = torch.tensor(sample['u'], dtype=torch.float32, device=device).unsqueeze(0)
            ys_raw = torch.tensor(sample['Y'], dtype=torch.float32, device=device)
            ys_norm = (ys_raw - chladni_dataset.xy_mean) / chladni_dataset.xy_std
            ys = ys_norm.unsqueeze(0)
            target_orig = torch.tensor(sample['s'], dtype=torch.float32, device=device).unsqueeze(0)
            
            us_norm = (us_orig - chladni_dataset.u_mean) / (chladni_dataset.u_std + 1e-8)
            target_norm = (target_orig - chladni_dataset.s_mean) / (chladni_dataset.s_std + 1e-8)
            
            pred_norm = model(us_norm, ys)
            total_loss += torch.nn.MSELoss()(pred_norm, target_norm).item()
            total_rel_error += (torch.norm(pred_norm - target_norm) / torch.norm(target_norm)).item()
    
    avg_loss, avg_rel_error = total_loss / n_test, total_rel_error / n_test
    print(f"Test Results - MSE Loss: {avg_loss:.6e}, Relative Error: {avg_rel_error:.6f}")
    
    model.train()
    return avg_loss, avg_rel_error



def main():
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = setup_logging(project_root)
    
    try:
        dataset = load_from_disk(args.data_path)
        print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    chladni_dataset = ChladniDatasetDeepONet(dataset, batch_size=args.batch_size, device=device)
    model = create_model(args, device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path, map_location=device)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        print("Pre-trained model loaded")
    
    train_model(model, chladni_dataset, args, device, log_dir)
    evaluate_model(model, dataset, chladni_dataset, device, n_test_samples=100)
    
    torch.save(model.state_dict(), os.path.join(log_dir, "deeponet_model.pth"))
    print("Training completed!")

if __name__ == "__main__":
    main() 