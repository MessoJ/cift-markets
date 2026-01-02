#!/usr/bin/env python3
"""
V7: Production TPU Training with Transformer
- Uses full BTCUSDT dataset
- Proper Transformer architecture
- Based on working v6 XLA setup
"""
import os
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import gcsfs
import pyarrow.parquet as pq
import math

print('=== TPU Training V7 (Transformer) ===')

import torch_xla.core.xla_model as xm

# Config
BATCH_SIZE = 128
EPOCHS = 20
SEQ_LEN = 64
N_FEATURES = 4
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

device = xm.xla_device()
print(f'Device: {device}')

# Transformer Model
class OrderFlowTransformer(nn.Module):
    def __init__(self, n_features=4, d_model=64, n_heads=4, n_layers=2, 
                 seq_len=64, num_classes=3, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]  # Use last token
        return self.fc_out(x)

# Load all BTCUSDT data from GCS
print('Loading data from GCS...')
fs = gcsfs.GCSFileSystem(project='united-option-388113')
bucket = 'cift-data-united-option-388113'

# List all BTCUSDT parquet files
all_files = fs.glob(f'{bucket}/processed/BTCUSDT/**/*.parquet')
print(f'Found {len(all_files)} BTCUSDT files')

all_data = []
for i, fpath in enumerate(sorted(all_files)):
    try:
        with fs.open(fpath, 'rb') as f:
            df = pq.read_table(f).to_pandas()
            all_data.append(df)
            print(f'  Loaded {i+1}/{len(all_files)}: {fpath.split("/")[-1]} ({len(df)} rows)')
    except Exception as e:
        print(f'  Error loading {fpath}: {e}')

df = pd.concat(all_data, ignore_index=True) if all_data else None
if df is None or len(df) == 0:
    raise ValueError("No data loaded!")

print(f'\nTotal rows: {len(df)}')

# Extract features
df['returns'] = df['close'].pct_change().fillna(0)
df['log_vol'] = np.log1p(df['volume']).diff().fillna(0)
df['volatility'] = df['returns'].rolling(20).std().fillna(0)
df['momentum'] = df['returns'].rolling(10).mean().fillna(0)

# Create sequences
data = df[['returns', 'log_vol', 'volatility', 'momentum']].values.astype(np.float32)
print(f'Data shape: {data.shape}')

# Clip extreme values
data = np.clip(data, -10, 10)

# Create samples
print('Creating samples...')
samples = []
labels = []
for i in range(SEQ_LEN, len(data) - 1):
    seq = data[i-SEQ_LEN:i]
    future_ret = data[i, 0]
    # 3-class: down (-1), neutral (0), up (+1) based on threshold
    threshold = 0.0002  # 2 bps
    if future_ret > threshold:
        label = 2  # Up
    elif future_ret < -threshold:
        label = 0  # Down
    else:
        label = 1  # Neutral

    samples.append(seq)
    labels.append(label)

X = np.array(samples, dtype=np.float32)
y = np.array(labels, dtype=np.int64)
print(f'Samples: {len(X)}, Features: {X.shape[1]}x{X.shape[2]}')

# Normalize per feature
for f in range(X.shape[2]):
    feat = X[:, :, f]
    mean = feat.mean()
    std = feat.std() + 1e-8
    X[:, :, f] = (feat - mean) / std

# Split 90/10 train/val
n_train = int(len(X) * 0.9)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

print(f'Train: {len(X_train)}, Val: {len(X_val)}')
print(f'Class dist train: {np.bincount(y_train)}')
print(f'Class dist val: {np.bincount(y_val)}')

# DataLoader
train_ds = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train), 
    torch.from_numpy(y_train)
)
val_ds = torch.utils.data.TensorDataset(
    torch.from_numpy(X_val), 
    torch.from_numpy(y_val)
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, 
    drop_last=True, num_workers=0
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, 
    drop_last=True, num_workers=0
)

print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

# Model
model = OrderFlowTransformer(
    n_features=N_FEATURES,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    seq_len=SEQ_LEN,
    num_classes=3,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

num_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {num_params:,}')

# Warmup
print('\nWarming up XLA...')
warmup_x = torch.randn(BATCH_SIZE, SEQ_LEN, N_FEATURES, device=device)
warmup_y = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
xm.mark_step()
model.train()
out = model(warmup_x)
loss = criterion(out, warmup_y)
loss.backward()
xm.mark_step()
optimizer.zero_grad()
xm.mark_step()
print('Warmup complete!')

print('\nStarting training...')
best_val_acc = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    start = time.time()
    for batch_idx, (xb, yb) in enumerate(train_loader):
        xb = xb.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()
        
        epoch_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        
        if batch_idx % 100 == 0:
            print(f'  Epoch {epoch} Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}')
    
    elapsed = time.time() - start
    train_acc = 100 * correct / total
    avg_loss = epoch_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            xm.mark_step()
            val_loss += criterion(logits, yb).item()
            pred = logits.argmax(dim=1)
            val_correct += (pred == yb).sum().item()
            val_total += yb.size(0)
    
    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    
    print(f'Epoch {epoch}/{EPOCHS}: Train Loss={avg_loss:.4f}, Train Acc={train_acc:.1f}%, '
          f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.1f}% [{elapsed:.1f}s] LR={lr:.6f}')
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/tmp/transformer_v7_best.pt')
        print(f'  *** New best model saved! ***')

print('\n=== Training Complete! ===')
print(f'Best Val Accuracy: {best_val_acc:.1f}%')
if best_val_acc > 40:  # 3 classes, random = 33%
    print('SUCCESS: Model is learning (above random chance for 3 classes)!')
else:
    print('Model near random chance - may need more tuning')

# Save final model
torch.save(model.state_dict(), '/tmp/transformer_v7_final.pt')
print('Models saved to /tmp/')
