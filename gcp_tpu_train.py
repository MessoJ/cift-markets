"""
TPU TRAINING SCRIPT (PyTorch XLA)
=================================

Optimized for Google Cloud TPUs (v3-8 or v4-8).
Uses PyTorch XLA to distribute training across TPU cores.

Usage:
    export XRT_TPU_CONFIG="localservice;0;localhost:51011"
    python scripts/gcp_tpu_train.py
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from cift.ml.transformer import TransformerModel  # Your existing model
import time
import os

# =============================================================================
# CONFIG
# =============================================================================

BATCH_SIZE = 2048  # TPUs love large batches
EPOCHS = 50
LEARNING_RATE = 1e-4
SEQ_LEN = 60  # 1 hour of minute data

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
        
        self.data = np.concatenate(self.data)
        
    def __len__(self):
        return len(self.data) - SEQ_LEN - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + SEQ_LEN]
        y = self.data[idx + SEQ_LEN, 0]  # Predict next return
        return torch.tensor(x), torch.tensor(y)

# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_loop_fn(loader, model, optimizer, device, loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.squeeze(), target)
        loss.backward()
        xm.optimizer_step(optimizer)
        
        if batch_idx % 10 == 0:
            xm.master_print(f'Batch {batch_idx}: Loss {loss.item():.4f}')

def _mp_fn(rank, flags):
    device = xm.xla_device()
    
    # 1. Load Data (In production, stream from GCS)
    # For now, assume data is downloaded to /tmp/data
    files = [f"/tmp/data/{f}" for f in os.listdir("/tmp/data") if f.endswith('.parquet')]
    dataset = ParquetDataset(files)
    sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
    
    # 2. Model
    model = TransformerModel(input_dim=2, d_model=128, nhead=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    
    # 3. Train
    for epoch in range(EPOCHS):
        para_loader = pl.ParallelLoader(loader, [device])
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, loss_fn)
        xm.master_print(f'Epoch {epoch} completed')
        
        # Save checkpoint
        if rank == 0:
            xm.save(model.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == '__main__':
    # Launch multiprocessing for TPU. Use `nprocs=None` to let torch_xla
    # auto-detect the number of available devices on the host (recommended),
    # or set `nprocs=1` for single-process debugging.
    xmp.spawn(_mp_fn, args=({},), nprocs=None, start_method='fork')
