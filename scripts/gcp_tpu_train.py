"""
TPU TRAINING SCRIPT (PyTorch XLA)
=================================

Optimized for Google Cloud TPUs.
Supports both single-device and multi-device training.

Usage:
    # Single device (for testing):
    python scripts/gcp_tpu_train.py --single
    
    # Multi-device (production):
    python scripts/gcp_tpu_train.py
"""

import os
import sys
import argparse

# =============================================================================
# CONFIG
# =============================================================================

BATCH_SIZE = 512  # Smaller batch for testing
EPOCHS = 5  # Fewer epochs for testing
LEARNING_RATE = 1e-4
SEQ_LEN = 60
DATA_DIR = "/tmp/data"

def get_dataset():
    """Create dataset - import heavy libs only when needed."""
    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset
    from pathlib import Path
    
    class ParquetDataset(Dataset):
        def __init__(self, file_paths):
            self.data = []
            for f in file_paths:
                try:
                    df = pd.read_parquet(f)
                    df['returns'] = df['close'].pct_change()
                    df['log_vol'] = np.log1p(df['volume'])
                    df = df.dropna()
                    self.data.append(df[['returns', 'log_vol']].values.astype(np.float32))
                except Exception as e:
                    print(f"Error loading {f}: {e}")
            
            if self.data:
                self.data = np.concatenate(self.data)
                print(f"Loaded {len(self.data):,} samples from {len(file_paths)} files")
            else:
                print("WARNING: No parquet files found, using dummy data")
                self.data = np.random.randn(10000, 2).astype(np.float32)
            
        def __len__(self):
            return len(self.data) - SEQ_LEN - 1

        def __getitem__(self, idx):
            x = self.data[idx : idx + SEQ_LEN]
            y = self.data[idx + SEQ_LEN, 0]
            return torch.tensor(x), torch.tensor(y)
    
    # Recursively find all parquet files
    files = []
    if os.path.exists(DATA_DIR):
        for f in Path(DATA_DIR).rglob('*.parquet'):
            files.append(str(f))
    
    print(f"Found {len(files)} parquet files")
    return ParquetDataset(files)

def create_model():
    """Create a simple model for testing."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(SEQ_LEN * 2, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

# =============================================================================
# SINGLE DEVICE TRAINING (for testing)
# =============================================================================

def train_single_device():
    """Train on a single TPU core - useful for debugging."""
    print("=" * 60)
    print("SINGLE DEVICE TPU TRAINING")
    print("=" * 60)
    
    # Import torch_xla here
    import torch
    import torch_xla.core.xla_model as xm
    from torch.utils.data import DataLoader
    
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Dataset
    dataset = get_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    print(f"DataLoader created: {len(loader)} batches")
    
    # Model
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            # Flatten and move to device
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output.squeeze(), target)
            loss.backward()
            
            xm.optimizer_step(optimizer)
            xm.mark_step()  # Important for TPU
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}: Loss {loss.item():.6f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} completed, Avg Loss: {avg_loss:.6f}")
    
    # Save model
    xm.save(model.state_dict(), "model_single_device.pth")
    print("Model saved to model_single_device.pth")
    print("=== Training Complete ===")

# =============================================================================
# MULTI-DEVICE TRAINING
# =============================================================================

def _mp_fn(index, flags):
    """Worker function for multi-device training."""
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from torch.utils.data import DataLoader, DistributedSampler
    
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    
    xm.master_print(f"=== Multi-Device TPU Training ===")
    xm.master_print(f"World size: {world_size}")
    
    dataset = get_dataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, drop_last=True)
    
    model = create_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    
    xm.master_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        para_loader = pl.ParallelLoader(loader, [device])
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(para_loader.per_device_loader(device)):
            data = data.view(data.size(0), -1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output.squeeze(), target)
            loss.backward()
            xm.optimizer_step(optimizer)
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                xm.master_print(f"  Batch {batch_idx}: Loss {loss.item():.6f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        xm.master_print(f'Epoch {epoch+1}/{EPOCHS} completed, Avg Loss: {avg_loss:.6f}')
        
        if rank == 0 and (epoch + 1) % 5 == 0:
            xm.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    
    xm.master_print("=== Training Complete ===")

def train_multi_device():
    """Train on multiple TPU cores."""
    print("=" * 60)
    print("MULTI-DEVICE TPU TRAINING")
    print("=" * 60)
    
    # Import xmp only when needed
    import torch_xla.distributed.xla_multiprocessing as xmp
    
    # Use spawn
    xmp.spawn(_mp_fn, args=({},), nprocs=None, start_method='spawn')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Use single device mode')
    args = parser.parse_args()
    
    print(f"Data directory: {DATA_DIR}")
    
    if args.single:
        train_single_device()
    else:
        train_multi_device()
