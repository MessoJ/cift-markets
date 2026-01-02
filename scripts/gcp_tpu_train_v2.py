"""
TPU PRODUCTION TRAINING SCRIPT v2 (PyTorch XLA)
================================================
Fixed data loading from GCS with proper error handling.
Training on real Binance 2023-2024 data.
"""

import os
import sys
import math
import argparse
import time

# =============================================================================
# CONFIG
# =============================================================================

BATCH_SIZE = 512           # Reduced for faster compilation
EPOCHS = 100               # More epochs for real data
LEARNING_RATE = 1e-4       # Adjusted LR for smaller batch
WEIGHT_DECAY = 1e-5        # Regularization
SEQ_LEN = 60               # 1 hour of minute data
WARMUP_EPOCHS = 5          # LR warmup
VAL_SPLIT = 0.1            # 10% validation

# Model architecture
D_MODEL = 128              # Transformer dimension
N_HEADS = 8                # Attention heads
N_LAYERS = 4               # Transformer layers
DROPOUT = 0.1              # Dropout rate

DATA_DIR = "gs://cift-data-united-option-388113/processed"
CHECKPOINT_DIR = "/tmp/checkpoints_v2"

# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# =============================================================================
# DATA LOADING - FIXED
# =============================================================================

def load_real_data():
    """Load real data from GCS with proper error handling."""
    import pandas as pd
    import numpy as np
    
    print("=" * 60)
    print("LOADING REAL DATA FROM GCS")
    print("=" * 60)
    
    # Import gcsfs
    try:
        import gcsfs
        print("[OK] gcsfs imported successfully")
    except ImportError as e:
        print(f"[FAIL] gcsfs import failed: {e}")
        print("Installing gcsfs...")
        os.system("pip install gcsfs")
        import gcsfs
    
    # Create filesystem
    try:
        fs = gcsfs.GCSFileSystem()
        print("[OK] GCSFileSystem created")
    except Exception as e:
        print(f"[FAIL] GCSFileSystem failed: {e}")
        raise
    
    # Find parquet files
    gcs_path = DATA_DIR.replace('gs://', '')
    print(f"Searching: {gcs_path}/**/*.parquet")
    
    try:
        files = fs.glob(f"{gcs_path}/**/*.parquet")
        print(f"[OK] Found {len(files)} parquet files")
    except Exception as e:
        print(f"[FAIL] Glob failed: {e}")
        raise
    
    if len(files) == 0:
        raise ValueError("No parquet files found!")
    
    # Show what we found
    print("\nFiles by asset:")
    asset_counts = {}
    for f in files:
        parts = f.split('/')
        if len(parts) >= 3:
            asset = parts[2]  # bucket/processed/ASSET/...
            asset_counts[asset] = asset_counts.get(asset, 0) + 1
    for asset, count in sorted(asset_counts.items()):
        print(f"  {asset}: {count} files")
    
    # Load all data
    print("\nLoading data...")
    all_data = []
    loaded_files = 0
    failed_files = 0
    total_rows = 0
    
    for i, f in enumerate(sorted(files)):
        try:
            df = pd.read_parquet('gs://' + f)
            
            # Calculate features
            df['returns'] = df['close'].pct_change()
            df['log_vol'] = np.log1p(df['volume'])
            df['volatility'] = df['returns'].rolling(20).std()
            df['momentum'] = df['close'].pct_change(20)
            
            # Drop NaN
            df = df.dropna()
            
            if len(df) > 0:
                features = df[['returns', 'log_vol', 'volatility', 'momentum']].values.astype(np.float32)
                all_data.append(features)
                loaded_files += 1
                total_rows += len(features)
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i+1}/{len(files)} files... ({total_rows:,} rows)")
                
        except Exception as e:
            print(f"  [FAIL] Error loading {f}: {e}")
            failed_files += 1
    
    print(f"\n[OK] Loaded {loaded_files} files, {failed_files} failed")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Concatenate
    data = np.concatenate(all_data)
    print(f"[OK] Total samples: {len(data):,}")
    print(f"  Features: {data.shape[1]}")
    print(f"  Memory: {data.nbytes / 1024 / 1024:.1f} MB")
    
    # Normalize features
    print("\nNormalizing features...")
    means = data.mean(axis=0)
    stds = data.std(axis=0) + 1e-8
    data = (data - means) / stds
    
    # Data stats
    print("\nNormalized data statistics:")
    for i, name in enumerate(['returns', 'log_vol', 'volatility', 'momentum']):
        print(f"  {name}: mean={data[:,i].mean():.4f}, std={data[:,i].std():.4f}")
    
    return data

# =============================================================================
# DATASET
# =============================================================================

def get_datasets(val_split=VAL_SPLIT):
    import torch
    from torch.utils.data import Dataset
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, data, n_features):
            self.data = data
            self.n_features = n_features
            
        def __len__(self):
            return len(self.data) - SEQ_LEN - 1

        def __getitem__(self, idx):
            x = self.data[idx : idx + SEQ_LEN]
            next_return = self.data[idx + SEQ_LEN, 0]  # Normalized return
            direction = 1.0 if next_return > 0 else 0.0
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(next_return, dtype=torch.float32),
                torch.tensor(direction, dtype=torch.float32)
            )
    
    # Load real data
    data = load_real_data()
    n_features = data.shape[1]
    
    # Train/val split (temporal)
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val: {len(val_data):,} samples")
    
    return TimeSeriesDataset(train_data, n_features), TimeSeriesDataset(val_data, n_features), n_features

# =============================================================================
# MODEL - Updated for 4 features
# =============================================================================

def build_model(n_features=4):
    import torch
    import torch.nn as nn
    
    class OrderFlowTransformer(nn.Module):
        def __init__(self, input_dim=4, d_model=D_MODEL, n_heads=N_HEADS, 
                     n_layers=N_LAYERS, dropout=DROPOUT):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            self.return_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
            self.direction_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            )
            
        def forward(self, x):
            x = self.input_proj(x) + self.pos_encoding
            x = self.transformer(x)
            x = x[:, -1, :]  # Last timestep
            return self.return_head(x).squeeze(-1), self.direction_head(x).squeeze(-1)
    
    model = OrderFlowTransformer(input_dim=n_features)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    return model

# =============================================================================
# TRAINING
# =============================================================================

def train_single_device():
    import torch
    import torch_xla.core.xla_model as xm
    from torch.utils.data import DataLoader
    
    print("\n" + "=" * 60)
    print("SINGLE DEVICE TPU TRAINING - REAL DATA v2")
    print("=" * 60)
    
    device = xm.xla_device()
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    
    # Data
    train_ds, val_ds, n_features = get_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = build_model(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LEARNING_RATE)
    
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        lr = scheduler.step(epoch - 1)
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (x, y_ret, y_dir) in enumerate(train_loader):
            x, y_ret, y_dir = x.to(device), y_ret.to(device), y_dir.to(device)
            
            optimizer.zero_grad()
            pred_ret, pred_dir = model(x)
            
            loss = mse_loss(pred_ret, y_ret) + bce_loss(pred_dir, y_dir)
            loss.backward()
            
            xm.optimizer_step(optimizer)
            
            train_loss += loss.item()
            preds = (pred_dir > 0).float()
            train_correct += (preds == y_dir).sum().item()
            train_total += y_dir.size(0)
            
            if batch_idx == 0:
                print(f"  Epoch {epoch}, Batch 0/{len(train_loader)}: Loss {loss.item():.6f}")
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y_ret, y_dir in val_loader:
                x, y_ret, y_dir = x.to(device), y_ret.to(device), y_dir.to(device)
                pred_ret, pred_dir = model(x)
                loss = mse_loss(pred_ret, y_ret) + bce_loss(pred_dir, y_dir)
                val_loss += loss.item()
                preds = (pred_dir > 0).float()
                val_correct += (preds == y_dir).sum().item()
                val_total += y_dir.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"Epoch {epoch}/{EPOCHS} | LR: {lr:.6f} | Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
        
        # Save best by loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_loss_model.pth")
            print(f"  *** New best loss model saved! ***")
        
        # Save best by accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_acc_model.pth")
            print(f"  *** New best accuracy model saved! ***")
        
        # Periodic save
        if epoch % 10 == 0:
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_epoch_{epoch}.pth")
            print(f"  Checkpoint saved: model_epoch_{epoch}.pth")
    
    # Save final
    xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/final_model.pth")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Models saved to: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    train_single_device()
