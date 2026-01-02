"""
TPU Training Script v3 - Limited Data for Testing
==================================================
Uses subset of data to validate training loop first.
"""

import os
import sys
import math
import time

# =============================================================================
# CONFIG - SMALLER FOR TESTING
# =============================================================================

BATCH_SIZE = 256           # Smaller batch for faster testing
EPOCHS = 20                # Fewer epochs
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
SEQ_LEN = 60
WARMUP_EPOCHS = 2
VAL_SPLIT = 0.1

MAX_FILES = 24             # Only load 1 asset (24 files = 2 years)

D_MODEL = 64               # Smaller model
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

CHECKPOINT_DIR = "/tmp/checkpoints_v3"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_real_data():
    import pandas as pd
    import numpy as np
    import gcsfs
    
    print("=" * 60)
    print("LOADING LIMITED DATA FROM GCS")
    print("=" * 60)
    sys.stdout.flush()
    
    fs = gcsfs.GCSFileSystem()
    files = fs.glob("cift-data-united-option-388113/processed/BTCUSDT/**/*.parquet")
    print(f"Found {len(files)} BTCUSDT parquet files")
    sys.stdout.flush()
    
    # Load limited files
    all_data = []
    for i, f in enumerate(sorted(files)[:MAX_FILES]):
        try:
            df = pd.read_parquet('gs://' + f)
            df['returns'] = df['close'].pct_change()
            df['log_vol'] = np.log1p(df['volume'])
            df = df.dropna()
            features = df[['returns', 'log_vol']].values.astype(np.float32)
            all_data.append(features)
            print(f"  Loaded {i+1}/{min(len(files), MAX_FILES)}: {f.split('/')[-1]} ({len(features):,} rows)")
            sys.stdout.flush()
        except Exception as e:
            print(f"  Error: {e}")
            sys.stdout.flush()
    
    data = np.concatenate(all_data)
    print(f"\nTotal samples: {len(data):,}")
    
    # Normalize
    means = data.mean(axis=0)
    stds = data.std(axis=0) + 1e-8
    data = (data - means) / stds
    print(f"Normalized. Mean: {data.mean(axis=0)}, Std: {data.std(axis=0)}")
    sys.stdout.flush()
    
    return data

# =============================================================================
# DATASET & MODEL
# =============================================================================

def get_datasets(data, val_split=VAL_SPLIT):
    import torch
    from torch.utils.data import Dataset
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data) - SEQ_LEN - 1
        def __getitem__(self, idx):
            x = self.data[idx : idx + SEQ_LEN]
            next_return = self.data[idx + SEQ_LEN, 0]
            direction = 1.0 if next_return > 0 else 0.0
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(next_return, dtype=torch.float32),
                torch.tensor(direction, dtype=torch.float32)
            )
    
    split_idx = int(len(data) * (1 - val_split))
    return TimeSeriesDataset(data[:split_idx]), TimeSeriesDataset(data[split_idx:])


def build_model(input_dim=2):
    import torch
    import torch.nn as nn
    
    class SmallTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, D_MODEL)
            self.pos_enc = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
            encoder = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_MODEL*4, DROPOUT, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder, N_LAYERS)
            self.head = nn.Linear(D_MODEL, 1)
            
        def forward(self, x):
            x = self.input_proj(x) + self.pos_enc
            x = self.transformer(x)
            return self.head(x[:, -1, :]).squeeze(-1)
    
    model = SmallTransformer()
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    sys.stdout.flush()
    return model

# =============================================================================
# TRAINING
# =============================================================================

def train():
    import torch
    import torch_xla.core.xla_model as xm
    from torch.utils.data import DataLoader
    
    print("\n" + "=" * 60)
    print("TPU TRAINING v3 - LIMITED DATA")
    print("=" * 60)
    sys.stdout.flush()
    
    device = xm.xla_device()
    print(f"Device: {device}")
    sys.stdout.flush()
    
    # Load data
    data = load_real_data()
    train_ds, val_ds = get_datasets(data)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    print(f"Train: {len(train_ds):,} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_ds):,} samples, {len(val_loader)} batches")
    sys.stdout.flush()
    
    # Model
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    sys.stdout.flush()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, _, y_dir) in enumerate(train_loader):
            x, y_dir = x.to(device), y_dir.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y_dir)
            loss.backward()
            xm.optimizer_step(optimizer)
            
            train_loss += loss.item()
            train_correct += ((pred > 0).float() == y_dir).sum().item()
            train_total += y_dir.size(0)
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")
                sys.stdout.flush()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, _, y_dir in val_loader:
                x, y_dir = x.to(device), y_dir.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y_dir).item()
                val_correct += ((pred > 0).float() == y_dir).sum().item()
                val_total += y_dir.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{EPOCHS} | Time: {epoch_time:.1f}s")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        sys.stdout.flush()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print("  *** Best model saved ***")
            sys.stdout.flush()
        
        if epoch % 5 == 0:
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_epoch_{epoch}.pth")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    train()
