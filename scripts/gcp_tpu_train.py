"""
TPU PRODUCTION TRAINING SCRIPT (PyTorch XLA)
=============================================

Full OrderFlowTransformer training on Google Cloud TPU v3-32.
Optimized for maximum performance with proper validation.

Usage:
    # Single device (for testing):
    python scripts/gcp_tpu_train.py --single
    
    # All 8 local TPU cores (production):
    python scripts/gcp_tpu_train.py

    # Quick test mode:
    python scripts/gcp_tpu_train.py --single --test
"""

import os
import sys
import math
import argparse

# =============================================================================
# CONFIG
# =============================================================================

# Production settings
BATCH_SIZE = 2048          # TPUs love large batches
EPOCHS = 50                # Full training
LEARNING_RATE = 3e-4       # Higher LR for TPU
WEIGHT_DECAY = 1e-5        # Regularization
SEQ_LEN = 60               # 1 hour of minute data
WARMUP_EPOCHS = 3          # LR warmup
VAL_SPLIT = 0.1            # 10% validation

# Model architecture
D_MODEL = 128              # Transformer dimension
N_HEADS = 8                # Attention heads
N_LAYERS = 4               # Transformer layers
DROPOUT = 0.1              # Dropout rate

DATA_DIR = "gs://cift-data-united-option-388113/processed"
CHECKPOINT_DIR = "/tmp/checkpoints"

# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# =============================================================================
# DATASET
# =============================================================================

def get_datasets(val_split=VAL_SPLIT):
    """Create train and validation datasets."""
    import torch
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset
    from pathlib import Path
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data) - SEQ_LEN - 1

        def __getitem__(self, idx):
            x = self.data[idx : idx + SEQ_LEN]
            
            # Target: next return and direction
            next_return = self.data[idx + SEQ_LEN, 0]
            direction = 1.0 if next_return > 0 else 0.0
            
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(next_return, dtype=torch.float32),
                torch.tensor(direction, dtype=torch.float32)
            )
    
    # Recursively find all parquet files
    files = []
    if DATA_DIR.startswith('gs://'):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            # List all parquet files recursively
            # glob pattern: bucket/path/**/*.parquet
            # strip gs:// for gcsfs
            gcs_path = DATA_DIR.replace('gs://', '')
            files = ['gs://' + f for f in fs.glob(f"{gcs_path}/**/*.parquet")]
        except ImportError:
            print("gcsfs not installed, cannot read from GCS")
    elif os.path.exists(DATA_DIR):
        for f in Path(DATA_DIR).rglob('*.parquet'):
            files.append(str(f))
    
    print(f"Found {len(files)} parquet files")
    
    # Load all data
    all_data = []
    for f in sorted(files):
        try:
            df = pd.read_parquet(f)
            df['returns'] = df['close'].pct_change()
            df['log_vol'] = np.log1p(df['volume'])
            df = df.dropna()
            all_data.append(df[['returns', 'log_vol']].values.astype(np.float32))
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if all_data:
        data = np.concatenate(all_data)
        print(f"Loaded {len(data):,} total samples")
    else:
        print("WARNING: No data found, using dummy data")
        data = np.random.randn(100000, 2).astype(np.float32)
    
    # Train/val split (temporal - last 10% for validation)
    split_idx = int(len(data) * (1 - val_split))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data):,}")
    print(f"Val samples: {len(val_data):,}")
    
    return TimeSeriesDataset(train_data), TimeSeriesDataset(val_data)

# =============================================================================
# MODEL: OrderFlowTransformer
# =============================================================================

def create_transformer_model():
    """Create the full OrderFlowTransformer for TPU."""
    import torch
    import torch.nn as nn
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
            
        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)
    
    class OrderFlowTransformer(nn.Module):
        def __init__(
            self,
            input_dim=2,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            dropout=DROPOUT,
        ):
            super().__init__()
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Prediction heads
            self.return_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            
            self.direction_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
            
            self.confidence_head = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1),
                nn.Sigmoid(),
            )
            
        def forward(self, x):
            # x: [batch, seq_len, input_dim]
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            
            # Causal mask for autoregressive prediction
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            
            x = self.transformer(x, mask=mask)
            
            # Use last token for prediction
            last_hidden = x[:, -1, :]
            
            return {
                'return': self.return_head(last_hidden),
                'direction': self.direction_head(last_hidden),
                'confidence': self.confidence_head(last_hidden),
            }
    
    return OrderFlowTransformer()

# =============================================================================
# SINGLE DEVICE TRAINING
# =============================================================================

def train_single_device(test_mode=False):
    """Train on a single TPU core."""
    print("=" * 60)
    print("SINGLE DEVICE TPU TRAINING - PRODUCTION")
    print("=" * 60)
    
    import torch
    import torch_xla.core.xla_model as xm
    from torch.utils.data import DataLoader
    
    # Reduce epochs for test mode
    epochs = 2 if test_mode else EPOCHS
    batch_size = 512 if test_mode else BATCH_SIZE
    
    device = xm.xla_device()
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    
    # Create checkpoint dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Dataset
    train_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Model
    model = create_transformer_model().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, epochs, LEARNING_RATE)
    
    # Loss functions
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        lr = scheduler.step(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target_return, target_dir) in enumerate(train_loader):
            data = data.to(device)
            target_return = target_return.to(device)
            target_dir = target_dir.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Combined loss
            loss_return = mse_loss(outputs['return'].squeeze(), target_return)
            loss_dir = bce_loss(outputs['direction'].squeeze(), target_dir)
            loss = loss_return + 0.5 * loss_dir
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            xm.mark_step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Accuracy
            pred_dir = (outputs['direction'].squeeze() > 0.5).float()
            correct += (pred_dir == target_dir).sum().item()
            total += target_dir.size(0)
            
            if batch_idx % 500 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.6f}")
        
        avg_train_loss = train_loss / train_batches
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target_return, target_dir in val_loader:
                data = data.to(device)
                target_return = target_return.to(device)
                target_dir = target_dir.to(device)
                
                outputs = model(data)
                loss_return = mse_loss(outputs['return'].squeeze(), target_return)
                loss_dir = bce_loss(outputs['direction'].squeeze(), target_dir)
                loss = loss_return + 0.5 * loss_dir
                
                val_loss += loss.item()
                val_batches += 1
                
                pred_dir = (outputs['direction'].squeeze() > 0.5).float()
                val_correct += (pred_dir == target_dir).sum().item()
                val_total += target_dir.size(0)
                
                xm.mark_step()
        
        avg_val_loss = val_loss / max(val_batches, 1)
        val_acc = val_correct / max(val_total, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | LR: {lr:.6f}")
        print(f"  Train Loss: {avg_train_loss:.6f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.6f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            print(f"  *** New best model saved! ***")
        
        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_epoch_{epoch+1}.pth")
    
    # Save final model
    xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/final_model.pth")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final Val Acc: {val_acc:.4f}")
    print(f"Models saved to: {CHECKPOINT_DIR}/")
    
    # Quick Sharpe estimate (very rough)
    if val_acc > 0.52:
        estimated_sharpe = (val_acc - 0.5) * 10  # Rough heuristic
        print(f"\n*** Estimated Sharpe (rough): {estimated_sharpe:.2f} ***")
        print("Note: Actual Sharpe requires backtesting with transaction costs")

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
    
    xm.master_print("=" * 60)
    xm.master_print(f"MULTI-DEVICE TPU TRAINING - {world_size} cores")
    xm.master_print("=" * 60)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_dataset, val_dataset = get_datasets()
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=0, drop_last=True)
    
    xm.master_print(f"Train batches per core: {len(train_loader)}")
    
    model = create_transformer_model().to(device)
    xm.master_print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LEARNING_RATE)
    
    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        lr = scheduler.step(epoch)
        train_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        para_loader = pl.ParallelLoader(train_loader, [device])
        
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (data, target_return, target_dir) in enumerate(para_loader.per_device_loader(device)):
            optimizer.zero_grad()
            outputs = model(data)
            
            loss_return = mse_loss(outputs['return'].squeeze(), target_return)
            loss_dir = bce_loss(outputs['direction'].squeeze(), target_dir)
            loss = loss_return + 0.5 * loss_dir
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 500 == 0:
                xm.master_print(f"  Epoch {epoch+1}, Batch {batch_idx}: Loss {loss.item():.6f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        para_val_loader = pl.ParallelLoader(val_loader, [device])
        
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, target_return, target_dir in para_val_loader.per_device_loader(device):
                outputs = model(data)
                loss_return = mse_loss(outputs['return'].squeeze(), target_return)
                loss_dir = bce_loss(outputs['direction'].squeeze(), target_dir)
                loss = loss_return + 0.5 * loss_dir
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        xm.master_print(f"Epoch {epoch+1}/{EPOCHS} | LR: {lr:.6f} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
            xm.master_print("  *** New best model! ***")
        
        if rank == 0 and (epoch + 1) % 10 == 0:
            xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/model_epoch_{epoch+1}.pth")
    
    if rank == 0:
        xm.save(model.state_dict(), f"{CHECKPOINT_DIR}/final_model.pth")
    
    xm.master_print("\n=== Training Complete ===")
    xm.master_print(f"Best Val Loss: {best_val_loss:.6f}")

def train_multi_device():
    """Train on multiple TPU cores."""
    print("=" * 60)
    print("MULTI-DEVICE TPU TRAINING")
    print("=" * 60)
    
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(_mp_fn, args=({},), nprocs=None, start_method='spawn')

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single', action='store_true', help='Use single device mode')
    parser.add_argument('--test', action='store_true', help='Quick test mode (2 epochs)')
    args = parser.parse_args()
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    
    if args.single:
        train_single_device(test_mode=args.test)
    else:
        train_multi_device()
