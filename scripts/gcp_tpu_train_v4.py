#!/usr/bin/env python3
"""
V4: Minimal TPU training with MLP (no Transformer complexity)
Goal: Prove training loop works, then upgrade model
"""
import os
os.environ['TPU_CHIPS_PER_PROCESS_BOUNDS'] = '1,1,1'
os.environ['TPU_PROCESS_BOUNDS'] = '1,1,1'

import torch
import torch.nn as nn
import numpy as np
import time
import gcsfs

import torch_xla.core.xla_model as xm

# Config
BATCH_SIZE = 256
EPOCHS = 10
SEQ_LEN = 32
N_FEATURES = 2
HIDDEN = 64

print('=== TPU Training V4 (MLP) ===')
device = xm.xla_device()
print(f'Device: {device}')

# Simple MLP Model (instead of Transformer)
class SimpleMLP(nn.Module):
    def __init__(self, seq_len=32, n_features=2, hidden=64):
        super().__init__()
        self.flatten_dim = seq_len * n_features
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)  # 3 classes: down/neutral/up
        )
    
    def forward(self, x):
        # x: (batch, seq, features) -> flatten
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x)

# Load real data from GCS
print('Loading data from GCS...')
fs = gcsfs.GCSFileSystem(project='united-option-388113')
bucket = 'cift-data-united-option-388113'

# Just load 1 file for quick test
parquet_path = f'{bucket}/processed/BTCUSDT/2024/01.parquet'
print(f'Reading {parquet_path}...')

import pyarrow.parquet as pq
with fs.open(parquet_path, 'rb') as f:
    df = pq.read_table(f).to_pandas()

print(f'Loaded {len(df)} rows')

# Extract features
df['returns'] = df['close'].pct_change().fillna(0)
df['log_vol'] = np.log1p(df['volume']).diff().fillna(0)

# Create sequences
data = df[['returns', 'log_vol']].values.astype(np.float32)
print(f'Data shape: {data.shape}')

# Create samples
samples = []
labels = []
for i in range(SEQ_LEN, len(data) - 1):
    seq = data[i-SEQ_LEN:i]
    future_ret = data[i, 0]  # next return
    label = 1 if future_ret > 0.0001 else (0 if future_ret < -0.0001 else 1)
    samples.append(seq)
    labels.append(label)

X = np.array(samples, dtype=np.float32)
y = np.array(labels, dtype=np.int64)
print(f'Samples: {len(X)}, Features: {X.shape[1]}x{X.shape[2]}')

# Normalize
mean = X.mean(axis=(0,1), keepdims=True)
std = X.std(axis=(0,1), keepdims=True) + 1e-8
X = (X - mean) / std

# Split
n_train = int(len(X) * 0.9)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

print(f'Train: {len(X_train)}, Val: {len(X_val)}')
print(f'Class dist: {np.bincount(y_train)}')

# DataLoader
train_ds = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train), 
    torch.from_numpy(y_train)
)
val_ds = torch.utils.data.TensorDataset(
    torch.from_numpy(X_val), 
    torch.from_numpy(y_val)
)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

# Model
model = SimpleMLP(SEQ_LEN, N_FEATURES, HIDDEN).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f'Model params: {sum(p.numel() for p in model.parameters())}')
print('\nStarting training...')

# Training loop
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
        
        epoch_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        
        if batch_idx % 50 == 0:
            print(f'  Epoch {epoch} Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}')
    
    elapsed = time.time() - start
    train_acc = 100 * correct / total
    avg_loss = epoch_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            val_correct += (pred == yb).sum().item()
            val_total += yb.size(0)
    
    val_acc = 100 * val_correct / val_total
    
    print(f'Epoch {epoch}/{EPOCHS}: Loss={avg_loss:.4f}, Train={train_acc:.1f}%, Val={val_acc:.1f}% [{elapsed:.1f}s]')

print('\n=== Training Complete! ===')
print(f'Final Val Accuracy: {val_acc:.1f}%')
if val_acc > 50:
    print('SUCCESS: Model is learning (above random chance)!')
else:
    print('Model at random chance - may need more data/epochs')
