#!/usr/bin/env python
"""run_transport.py
----------------------------------
Train SetONet for optimal transport with decoupled query points.

This script trains SetONet to predict transport map displacement at query points
that are independent from the source sensor locations.

Dataset: Data/transport_q_data/transport_dataset
Mode: 'transport_map' (decoupled queries)
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Add the project root directory to sys.path
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
if project_root not in sys.path:
    sys.path.append(project_root)

from Data.transport_q_data.transport_dataset import load_transport_dataset
from Models.SetONet import SetONet
from Models.utils.config_utils import save_experiment_configuration
from Models.utils.tensorboard_callback import TensorBoardCallback
from Plotting.plot_transport_q_utils import plot_transport_q_overlay


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SetONet for optimal transport with decoupled queries (Strategy 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data parameters
    default_data_path = os.path.join(project_root, "Data", "transport_q_data", "transport_dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default=default_data_path,
        help="Path to transport-Q dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="transport_map",
        choices=["velocity_field", "transport_map", "transport_map_at_source", "density_transport"],
        help="Transport learning mode (transport_map uses decoupled queries)",
    )

    # Model architecture
    parser.add_argument("--son_p_dim", type=int, default=128, help="Latent dimension p for SetONet")
    parser.add_argument("--son_phi_hidden", type=int, default=256, help="Hidden size for phi network")
    parser.add_argument("--son_rho_hidden", type=int, default=256, help="Hidden size for rho network")
    parser.add_argument("--son_trunk_hidden", type=int, default=256, help="Hidden size for trunk network")
    parser.add_argument("--son_n_trunk_layers", type=int, default=4, help="Number of trunk layers")
    parser.add_argument("--son_phi_output_size", type=int, default=32, help="Phi output size before aggregation")
    parser.add_argument(
        "--son_aggregation",
        type=str,
        default="attention",
        choices=["mean", "attention", "sum"],
        help="Aggregation type for SetONet",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu", "swish"],
        help="Activation function",
    )
    parser.add_argument(
        "--son_branch_head_type",
        type=str,
        default="standard",
        choices=["standard", "petrov_attention", "galerkin_pou", "quadrature", "adaptive_quadrature"],
        help="Branch head type",
    )
    parser.add_argument("--son_pg_dk", type=int, default=None, help="PG attention key/query dim")
    parser.add_argument("--son_pg_dv", type=int, default=None, help="PG attention value dim")
    parser.add_argument("--son_pg_no_logw", action="store_true", help="Disable log(w) in PG attention")
    parser.add_argument("--son_galerkin_dk", type=int, default=None, help="Galerkin PoU key/query dim")
    parser.add_argument("--son_galerkin_dv", type=int, default=None, help="Galerkin PoU value dim")
    parser.add_argument("--son_quad_dk", type=int, default=None, help="Quadrature key/query dim")
    parser.add_argument("--son_quad_dv", type=int, default=None, help="Quadrature value dim")
    parser.add_argument(
        "--son_galerkin_normalize",
        type=str,
        default="total",
        choices=["none", "total", "token"],
        help="Galerkin PoU normalization",
    )
    parser.add_argument("--son_galerkin_learn_temperature", action="store_true", help="Learn Galerkin temperature")
    parser.add_argument("--son_adapt_quad_rank", type=int, default=4, help="Adaptive quadrature rank")
    parser.add_argument("--son_adapt_quad_hidden", type=int, default=64, help="Adaptive quadrature hidden dim")
    parser.add_argument("--son_adapt_quad_scale", type=float, default=0.1, help="Adaptive quadrature scale")

    # Training parameters
    parser.add_argument("--son_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--son_epochs", type=int, default=50000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--pos_encoding_type",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "skip"],
        help="Positional encoding type",
    )
    parser.add_argument("--pos_encoding_dim", type=int, default=64, help="Positional encoding dimension")
    parser.add_argument("--pos_encoding_max_freq", type=float, default=0.001, help="Max frequency for sinusoidal PE")
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        nargs="+",
        default=[15000, 30000, 125000, 175000, 1250000, 1500000],
        help="LR decay milestone steps",
    )
    parser.add_argument(
        "--lr_schedule_gammas",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5],
        help="LR decay factors",
    )

    # Model loading
    parser.add_argument("--load_model_path", type=str, default=None, help="Path to pre-trained model")

    # Random seed and device
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")

    # TensorBoard logging
    parser.add_argument("--enable_tensorboard", action="store_true", default=True, help="Enable TensorBoard logging")
    parser.add_argument("--tb_eval_frequency", type=int, default=1000, help="TensorBoard evaluation frequency")
    parser.add_argument("--tb_test_samples", type=int, default=100, help="Number of test samples for TB evaluation")

    # Logging directory
    parser.add_argument("--log_dir", type=str, default=None, help="Custom log directory")

    return parser.parse_args()


def setup_logging(project_root, custom_log_dir=None):
    """Setup logging directory."""
    if custom_log_dir:
        log_dir = custom_log_dir
    else:
        logs_base = os.path.join(project_root, "logs")
        model_folder = "SetONet_transport"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(logs_base, model_folder, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")
    return log_dir


def get_activation_function(activation_name):
    """Get activation function by name."""
    activation_map = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
    }
    return activation_map.get(activation_name.lower(), nn.ReLU)


def create_model(args, device):
    """Create SetONet model for transport map prediction."""
    activation_fn = get_activation_function(args.activation_fn)

    model = SetONet(
        input_size_src=2,  # 2D coordinates (x, y) of source points
        output_size_src=1,  # Uniform weights at source points
        input_size_tgt=2,  # 2D coordinates (x, y) of query points
        output_size_tgt=2,  # 2D displacement vectors (dx, dy)
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
        use_positional_encoding=(args.pos_encoding_type != "skip"),
        attention_n_tokens=1,
        branch_head_type=args.son_branch_head_type,
        pg_dk=args.son_pg_dk,
        pg_dv=args.son_pg_dv,
        pg_use_logw=(not args.son_pg_no_logw),
        galerkin_dk=args.son_galerkin_dk,
        galerkin_dv=args.son_galerkin_dv,
        quad_dk=args.son_quad_dk,
        quad_dv=args.son_quad_dv,
        galerkin_normalize=args.son_galerkin_normalize,
        galerkin_learn_temperature=args.son_galerkin_learn_temperature,
        adapt_quad_rank=args.son_adapt_quad_rank,
        adapt_quad_hidden=args.son_adapt_quad_hidden,
        adapt_quad_scale=args.son_adapt_quad_scale,
    ).to(device)

    return model


def evaluate_model(model, dataset, transport_dataset, device, n_test_samples=None):
    """
    Evaluate the model on test data using decoupled query points.

    Uses GLOBAL L2 relative error: ||all_pred - all_target|| / ||all_target||
    This accumulates predictions across all test samples for a single metric,
    which is more standard in papers and weights samples by magnitude.

    Args:
        model: Trained SetONet model
        dataset: HuggingFace dataset with 'test' split
        transport_dataset: TransportDataset wrapper
        device: Torch device
        n_test_samples: Number of test samples (None = use all)

    Returns:
        dict with evaluation metrics
    """
    model.eval()
    test_data = dataset["test"]
    n_test = len(test_data) if n_test_samples is None else min(n_test_samples, len(test_data))

    # Accumulate all predictions and targets for global L2
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(n_test):
            sample = test_data[i]

            # Load source points (sensors)
            source_points = torch.tensor(
                np.array(sample["source_points"]), device=device, dtype=torch.float32
            )

            # Load query points and ground truth (decoupled from sensors)
            query_points = torch.tensor(
                np.array(sample["query_points"]), device=device, dtype=torch.float32
            )
            query_vectors_gt = torch.tensor(
                np.array(sample["query_vectors"]), device=device, dtype=torch.float32
            )

            # Add batch dimension
            source_coords = source_points.unsqueeze(0)  # (1, n_sensors, 2)
            source_weights = torch.ones(
                1, source_points.shape[0], 1, device=device, dtype=torch.float32
            )
            query_coords = query_points.unsqueeze(0)  # (1, n_queries, 2)

            # Forward pass: predict displacement at query points
            pred_vectors = model(source_coords, source_weights, query_coords)  # (1, n_queries, 2)

            # Accumulate flattened predictions and targets
            all_preds.append(pred_vectors.reshape(-1))
            all_targets.append(query_vectors_gt.reshape(-1))

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)  # (n_test * n_queries * 2,)
    all_targets = torch.cat(all_targets, dim=0)  # (n_test * n_queries * 2,)

    # Compute GLOBAL metrics
    # Global MSE
    global_mse = torch.mean((all_preds - all_targets) ** 2).item()

    # Global L2 relative error: ||pred - target|| / ||target||
    error_norm = torch.norm(all_preds - all_targets).item()
    target_norm = torch.norm(all_targets).item()
    global_rel_l2 = error_norm / (target_norm + 1e-8)

    print(f"\nTest Results ({n_test} samples, {len(all_preds)} total points, global L2):")
    print(f"  Transport Map:  MSE = {global_mse:.6e},  Rel L2 = {global_rel_l2:.6f}")

    model.train()
    return {
        "mse_transport_map": global_mse,
        "rel_l2_transport_map": global_rel_l2,
        "relative_l2_error": global_rel_l2,  # Compatibility alias
        "mse_loss": global_mse,  # Compatibility alias
        "n_test_samples": n_test,
        "n_total_points": len(all_preds),
    }


def main():
    """Main training function."""
    args = parse_arguments()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup logging
    log_dir = setup_logging(project_root, args.log_dir)

    # Load dataset
    dataset, transport_dataset = load_transport_dataset(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        mode=args.mode,
    )

    if dataset is None or transport_dataset is None:
        print("Failed to load dataset. Exiting.")
        return

    # Verify we have decoupled queries
    if transport_dataset.has_queries:
        print(f"Decoupled queries enabled: {transport_dataset.n_query_points} query points per sample")
    else:
        print("WARNING: Dataset does not have decoupled queries. Using legacy ys=xs mode.")

    # Create model
    print("Creating SetONet model...")
    model = create_model(args, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load pre-trained model if specified
    if args.load_model_path:
        print(f"Loading pre-trained model from: {args.load_model_path}")
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))

    # Setup TensorBoard callback
    callback = None
    if args.enable_tensorboard:
        print("Setting up TensorBoard logging...")
        tb_log_dir = os.path.join(log_dir, "tensorboard")
        callback = TensorBoardCallback(
            log_dir=tb_log_dir,
            dataset=dataset,
            dataset_wrapper=transport_dataset,
            device=device,
            eval_frequency=args.tb_eval_frequency,
            n_test_samples=args.tb_test_samples,
            eval_sensor_dropoff=0.0,
            replace_with_nearest=False,
        )
        print(f"TensorBoard logs: {tb_log_dir}")
        print(f"View with: tensorboard --logdir {tb_log_dir}")

    # Train model
    print(f"\nStarting training for {args.son_epochs} epochs...")
    print(f"Training on {transport_dataset.n_source_points} sensors -> {transport_dataset.n_query_points} queries")

    model.train_model(
        dataset=transport_dataset,
        epochs=args.son_epochs,
        progress_bar=True,
        callback=callback,
    )

    # Evaluate model on ALL test samples (global L2)
    print("\nEvaluating model on test set...")
    test_results = evaluate_model(
        model, dataset, transport_dataset, device, n_test_samples=None
    )

    # Save experiment configuration
    save_experiment_configuration(
        args,
        model,
        dataset,
        transport_dataset,
        device,
        log_dir,
        dataset_type="transport_q",
        test_results=test_results,
    )

    # Generate plots
    print("\nGenerating plots...")
    for i in range(3):
        plot_save_path = os.path.join(log_dir, f"transport_sample_{i}.png")
        plot_transport_q_overlay(model, dataset, transport_dataset, device, sample_idx=i,
                                  save_path=plot_save_path, dataset_split="test")

    # Save model
    model_save_path = os.path.join(log_dir, "transport_setonet_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    print("\nTraining completed!")
    print(f"Final Rel L2 Error: {test_results['rel_l2_transport_map']:.6f}")


if __name__ == "__main__":
    main()
