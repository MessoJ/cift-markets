#!/usr/bin/env python3
"""
Model Evaluation Script
Tests the trained Transformer model on held-out data
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gcsfs
import pyarrow.parquet as pq
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Config - must match training
SEQ_LEN = 64
N_FEATURES = 4
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1

print('=== Model Evaluation ===')

# Model Architecture (same as training)
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
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc_out(x)

# Load model
print('Loading model...')

# Try local file first, then GCS
import os
local_path = '/tmp/transformer_v8_best.pt'
if os.path.exists(local_path):
    print(f'Loading from local: {local_path}')
    state_dict = torch.load(local_path, map_location='cpu')
else:
    print('Loading from GCS...')
    fs = gcsfs.GCSFileSystem(project='united-option-388113')
    bucket = 'cift-data-united-option-388113'
    with fs.open(f'{bucket}/models/transformer_v8_best.pt', 'rb') as f:
        state_dict = torch.load(f, map_location='cpu')

model = OrderFlowTransformer(
    n_features=N_FEATURES,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    seq_len=SEQ_LEN,
    num_classes=3,
    dropout=DROPOUT
)
model.load_state_dict(state_dict)
model.eval()
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

# Initialize GCS filesystem
fs = gcsfs.GCSFileSystem(project='united-option-388113')
bucket = 'cift-data-united-option-388113'

# Load test data (use ETHUSDT as out-of-sample test)
print('\nLoading ETHUSDT test data (out-of-sample)...')
all_files = fs.glob(f'{bucket}/processed/ETHUSDT/**/*.parquet')
print(f'Found {len(all_files)} ETHUSDT files')

all_data = []
for fpath in sorted(all_files)[:6]:  # Just use first 6 months
    with fs.open(fpath, 'rb') as f:
        df = pq.read_table(f).to_pandas()
        all_data.append(df)
        print(f'  Loaded {fpath.split("/")[-1]} ({len(df)} rows)')

df = pd.concat(all_data, ignore_index=True)
print(f'Total test rows: {len(df)}')

# Extract features (same as training)
df['returns'] = df['close'].pct_change().fillna(0)
df['log_vol'] = np.log1p(df['volume']).diff().fillna(0)
df['volatility'] = df['returns'].rolling(20).std().fillna(0)
df['momentum'] = df['returns'].rolling(10).mean().fillna(0)

data = df[['returns', 'log_vol', 'volatility', 'momentum']].values.astype(np.float32)
data = np.clip(data, -10, 10)

# Create samples
print('Creating test samples...')
samples = []
labels = []
actual_returns = []

threshold = 0.0002
for i in range(SEQ_LEN, len(data) - 1):
    seq = data[i-SEQ_LEN:i]
    future_ret = data[i, 0]
    
    if future_ret > threshold:
        label = 2  # Up
    elif future_ret < -threshold:
        label = 0  # Down
    else:
        label = 1  # Neutral
    
    samples.append(seq)
    labels.append(label)
    actual_returns.append(future_ret)

X = np.array(samples, dtype=np.float32)
y = np.array(labels, dtype=np.int64)
actual_returns = np.array(actual_returns)

# Normalize (using training statistics would be better, but approximate)
for f in range(X.shape[2]):
    feat = X[:, :, f]
    mean = feat.mean()
    std = feat.std() + 1e-8
    X[:, :, f] = (feat - mean) / std

print(f'Test samples: {len(X)}')
print(f'Class distribution: {np.bincount(y)}')

# Inference
print('\nRunning inference...')
X_tensor = torch.from_numpy(X)
batch_size = 256
all_preds = []
all_probs = []

with torch.no_grad():
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size]
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_probs.extend(probs.numpy())

predictions = np.array(all_preds)
probabilities = np.array(all_probs)

# Evaluation Metrics
print('\n' + '='*60)
print('CLASSIFICATION REPORT')
print('='*60)
print(classification_report(y, predictions, 
      target_names=['Down', 'Neutral', 'Up'], digits=4))

print('\n' + '='*60)
print('CONFUSION MATRIX')
print('='*60)
cm = confusion_matrix(y, predictions)
print('             Predicted')
print('             Down  Neutral   Up')
print(f'Actual Down  {cm[0,0]:6d} {cm[0,1]:7d} {cm[1,2]:6d}')
print(f'     Neutral {cm[1,0]:6d} {cm[1,1]:7d} {cm[1,2]:6d}')
print(f'     Up      {cm[2,0]:6d} {cm[2,1]:7d} {cm[2,2]:6d}')

# Trading Simulation
print('\n' + '='*60)
print('TRADING SIMULATION')
print('='*60)

# Simple strategy: Long on UP prediction, Short on DOWN, Flat on NEUTRAL
positions = np.zeros(len(predictions))
positions[predictions == 2] = 1   # Long
positions[predictions == 0] = -1  # Short

# Calculate returns
strategy_returns = positions * actual_returns

# Metrics
total_return = strategy_returns.sum()
sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252 * 1440)  # Annualized (assuming minute data)
win_rate = (strategy_returns > 0).sum() / (strategy_returns != 0).sum() if (strategy_returns != 0).sum() > 0 else 0
avg_win = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).sum() > 0 else 0
avg_loss = strategy_returns[strategy_returns < 0].mean() if (strategy_returns < 0).sum() > 0 else 0

print(f'Total Trades: {(positions != 0).sum():,}')
print(f'Long Trades:  {(positions == 1).sum():,}')
print(f'Short Trades: {(positions == -1).sum():,}')
print(f'')
print(f'Total Return: {total_return*100:.4f}%')
print(f'Sharpe Ratio: {sharpe:.2f}')
print(f'Win Rate:     {win_rate*100:.2f}%')
print(f'Avg Win:      {avg_win*10000:.2f} bps')
print(f'Avg Loss:     {avg_loss*10000:.2f} bps')

# Confidence-based filtering
print('\n' + '='*60)
print('HIGH CONFIDENCE TRADES (>60% probability)')
print('='*60)

high_conf_mask = probabilities.max(axis=1) > 0.6
high_conf_preds = predictions[high_conf_mask]
high_conf_returns = actual_returns[high_conf_mask]
high_conf_labels = y[high_conf_mask]

if len(high_conf_preds) > 0:
    hc_positions = np.zeros(len(high_conf_preds))
    hc_positions[high_conf_preds == 2] = 1
    hc_positions[high_conf_preds == 0] = -1
    
    hc_strategy_returns = hc_positions * high_conf_returns
    hc_total_return = hc_strategy_returns.sum()
    hc_win_rate = (hc_strategy_returns > 0).sum() / (hc_strategy_returns != 0).sum() if (hc_strategy_returns != 0).sum() > 0 else 0
    hc_accuracy = (high_conf_preds == high_conf_labels).mean()
    
    print(f'High Conf Trades: {(hc_positions != 0).sum():,} ({(hc_positions != 0).sum()/len(positions)*100:.1f}% of total)')
    print(f'High Conf Accuracy: {hc_accuracy*100:.2f}%')
    print(f'High Conf Return: {hc_total_return*100:.4f}%')
    print(f'High Conf Win Rate: {hc_win_rate*100:.2f}%')
else:
    print('No high confidence predictions found')

# Summary
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
accuracy = (predictions == y).mean()
print(f'Overall Accuracy: {accuracy*100:.2f}%')
print(f'Random Baseline:  33.33%')
print(f'Edge vs Random:   {(accuracy - 0.3333)*100:.2f}%')

if accuracy > 0.35 and sharpe > 0:
    print('\n✅ MODEL SHOWS POSITIVE EDGE - Ready for integration!')
    verdict = 'PASS'
elif accuracy > 0.34:
    print('\n⚠️ MODEL SHOWS SLIGHT EDGE - Consider more training')
    verdict = 'MARGINAL'
else:
    print('\n❌ MODEL AT RANDOM CHANCE - Needs improvement')
    verdict = 'FAIL'

print(f'\nVerdict: {verdict}')
