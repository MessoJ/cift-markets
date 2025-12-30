"""
TPU TRAINING SCRIPT (PyTorch XLA)
=================================

Optimized for Google Cloud TPUs (v3-8, v3-32, v4-8, etc.).
Uses PyTorch XLA to distribute training across TPU cores.

Usage:
    python scripts/gcp_tpu_train.py

IMPORTANT: Do NOT import torch_xla at module level when using multiprocessing.
           The TPU runtime must be initialized AFTER the process spawns.
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# CONFIG
# =============================================================================

BATCH_SIZE = 2048  # TPUs love large batches
EPOCHS = 50
LEARNING_RATE = 1e-4
SEQ_LEN = 60  # 1 hour of minute data
DATA_DIR = "/tmp/data"  # Where parquet files are stored

# =============================================================================
# DATASET
# =============================================================================

class ParquetDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for f in file_paths:
            df = pd.read_parquet(f)
            # Simple feature engineering
            df['returns'] = df['close'].pct_change()
            df['log_vol'] = np.log1p(df['volume'])
            df = df.dropna()
            self.data.append(df[['returns', 'log_vol']].values.astype(np.float32))
        
        if self.data:
            self.data = np.concatenate(self.data)
        else:
            # Fallback: create dummy data for testing
            print("WARNING: No parquet files found, using dummy data")
            self.data = np.random.randn(10000, 2).astype(np.float32)
        
    def __len__(self):
        return len(self.data) - SEQ_LEN - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + SEQ_LEN]
        y = self.data[idx + SEQ_LEN, 0]  # Predict next return
        return torch.tensor(x), torch.tensor(y)

# =============================================================================
# TRAINING LOOP (called inside each TPU worker)
# =============================================================================

def train_loop_fn(loader, model, optimizer, device, loss_fn, xm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.squeeze(), target)
        loss.backward()
        xm.optimizer_step(optimizer)
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            xm.master_print(f'  Batch {batch_idx}: Loss {loss.item():.6f}')
    
    return total_loss / max(num_batches, 1)

def _mp_fn(index, flags):
    """
    Worker function for each TPU core.
    
    CRITICAL: Import torch_xla here, not at module level, to avoid
    initializing TPU runtime before process spawn.
    """
    # Import torch_xla INSIDE the worker
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from torch.utils.data import DistributedSampler
    
    # Get TPU device for this worker
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    rank = xm.get_ordinal()
    
    xm.master_print(f"=== TPU Training Started ===")
    xm.master_print(f"World size: {world_size}, Rank: {rank}")
    xm.master_print(f"Device: {device}")
    
    # 1. Load Data
    if os.path.exists(DATA_DIR):
        files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.parquet')]
    else:
        files = []
    
    xm.master_print(f"Found {len(files)} parquet files")
    
    dataset = ParquetDataset(files)
    xm.master_print(f"Dataset size: {len(dataset)} samples")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=0,  # Use 0 for TPU to avoid issues
        drop_last=True
    )
    
    # 2. Model - Simple transformer for testing
    # Using a basic nn.Module instead of the complex OrderFlowTransformer for initial test
    model = torch.nn.Sequential(
        torch.nn.Linear(SEQ_LEN * 2, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    
    xm.master_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        para_loader = pl.ParallelLoader(loader, [device])
        
        # Flatten input for simple model
        class FlattenLoader:
            def __init__(self, loader):
                self.loader = loader
            def __iter__(self):
                for data, target in self.loader:
                    yield data.view(data.size(0), -1), target
        
        avg_loss = train_loop_fn(
            FlattenLoader(para_loader.per_device_loader(device)), 
            model, optimizer, device, loss_fn, xm
        )
        
        xm.master_print(f'Epoch {epoch+1}/{EPOCHS} completed, Avg Loss: {avg_loss:.6f}')
        
        # Save checkpoint (only on master)
        if rank == 0 and (epoch + 1) % 10 == 0:
            xm.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            xm.master_print(f"Checkpoint saved: model_epoch_{epoch+1}.pth")
    
    xm.master_print("=== Training Complete ===")

if __name__ == '__main__':
    # Import xmp here to avoid early TPU init
    import torch_xla.distributed.xla_multiprocessing as xmp
    
    print("Starting TPU training...")
    print(f"Looking for data in: {DATA_DIR}")
    
    # Use start_method='spawn' to avoid fork issues with TPU runtime
    # nprocs=None lets torch_xla auto-detect available TPU cores
    xmp.spawn(_mp_fn, args=({},), nprocs=None, start_method='spawn')
