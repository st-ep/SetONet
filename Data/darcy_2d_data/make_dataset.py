"""Generate separate train and test Hugging Face Datasets for 2‑D Darcy flow.

Run:
    python make_dataset.py

Creates:
- Data/darcy_2d_data/darcy64/train (1000 samples)
- Data/darcy_2d_data/darcy64/test (100 samples)
"""
from __future__ import annotations

import os
import config as cfg

from numpy.random import default_rng
from tqdm.auto import trange

from solve import solve_one

# Hugging Face Datasets
from datasets import Dataset, Features, Array2D


def generate_dataset(n_samples: int, dataset_dir: str, seed: int, split_name: str):
    """Generate a single dataset with specified parameters."""
    print(f"Generating {split_name}: {n_samples} samples...")
    
    rng = default_rng(seed)

    # Collect samples
    k_list, p_list = [], []
    for _ in trange(n_samples, desc=f"{split_name}"):
        k, p = solve_one(rng)
        k_list.append(k)
        p_list.append(p)

    data_dict = {"k": k_list, "p": p_list}

    features = Features(
        {
            "k": Array2D(
                shape=(cfg.RESOLUTION + 1, cfg.RESOLUTION + 1), dtype="float32"
            ),
            "p": Array2D(
                shape=(cfg.RESOLUTION + 1, cfg.RESOLUTION + 1), dtype="float32"
            ),
        }
    )

    ds = Dataset.from_dict(data_dict, features=features)
    ds.save_to_disk(dataset_dir, max_shard_size="500MB")

    print(f"✅ Saved {split_name}: {len(ds)} samples")
    return len(ds)


def main():
    """Generate both train and test datasets."""
    # Create base directory (we're already in Data/darcy_2d_data)
    base_dir = "darcy64"
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate datasets
    train_samples = generate_dataset(1000, os.path.join(base_dir, "train"), 42, "train")
    test_samples = generate_dataset(100, os.path.join(base_dir, "test"), 12345, "test")
    
    print(f"\n🎉 Complete: {train_samples + test_samples} total samples")


if __name__ == "__main__":
    main()
