# GCP TRAINING GUIDE - Stat Arb System

## Complete Step-by-Step Guide

---

## PHASE 1: GCP SETUP (15 minutes)

### Step 1.1: Create GCP Project

```bash
# Install gcloud CLI if not installed
# Windows: https://cloud.google.com/sdk/docs/install

# Login and create project
gcloud auth login
gcloud projects create cift-stat-arb --name="CIFT Stat Arb"
gcloud config set project cift-stat-arb

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com
```

### Step 1.2: Create VM Instance

```bash
# Create GPU instance for training (or CPU for cheaper testing)
gcloud compute instances create stat-arb-trainer \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE

# OR cheaper CPU-only version for stat arb (doesn't need GPU):
gcloud compute instances create stat-arb-trainer \
    --zone=us-central1-a \
    --machine-type=e2-standard-8 \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --boot-disk-size=50GB
```

### Step 1.3: Create Storage Bucket

```bash
# Create bucket for data and models
gcloud storage buckets create gs://cift-stat-arb-data --location=us-central1
```

---

## PHASE 2: DATA ACQUISITION (30 minutes)

### Step 2.1: What Data You Need

| Data Type | Source | Format | Size |
|-----------|--------|--------|------|
| **Equity Daily Prices** | Yahoo Finance / Polygon | CSV/Parquet | ~50MB |
| **Crypto Funding Rates** | Binance API | JSON/CSV | ~10MB |
| **Crypto Spot Prices** | Binance API | CSV | ~100MB |

### Step 2.2: Data Download Script

Create this file locally: `scripts/download_data.py`

```python
"""
Data Download Script for GCP Training
Run this LOCALLY first, then upload to GCP
"""

import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import json

# Create data directory
os.makedirs('data/equity', exist_ok=True)
os.makedirs('data/crypto', exist_ok=True)
os.makedirs('data/funding', exist_ok=True)

# =============================================================================
# 1. EQUITY DATA
# =============================================================================

EQUITY_SYMBOLS = [
    # Energy
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY',
    # Banks
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'SCHW', 'AXP',
    # Retail
    'HD', 'LOW', 'WMT', 'TGT', 'COST', 'TJX', 'ROST', 'DG', 'BBY',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN',
    # Tech
    'MSFT', 'AAPL', 'GOOGL', 'META', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'IBM', 'AMD',
    # Staples
    'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'MDLZ'
]

print("Downloading equity data...")
end = datetime.now()
start = end - timedelta(days=5*365)  # 5 years

for symbol in EQUITY_SYMBOLS:
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        df.to_csv(f'data/equity/{symbol}.csv')
        print(f"  {symbol}: {len(df)} days")
    except Exception as e:
        print(f"  {symbol}: FAILED - {e}")

# Also save combined file
print("Creating combined equity file...")
all_data = {}
for symbol in EQUITY_SYMBOLS:
    try:
        df = pd.read_csv(f'data/equity/{symbol}.csv', index_col=0, parse_dates=True)
        all_data[symbol] = df['Close']
    except:
        pass

combined = pd.DataFrame(all_data)
combined.to_parquet('data/equity/all_equity_prices.parquet')
print(f"Combined: {len(combined)} days, {len(combined.columns)} stocks")

# =============================================================================
# 2. CRYPTO SPOT PRICES
# =============================================================================

CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
                  'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']

print("\nDownloading crypto spot prices...")

for symbol in CRYPTO_SYMBOLS:
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    
    for _ in range(10):  # ~1000 days
        params = {
            "symbol": symbol,
            "interval": "1d",
            "limit": 1000
        }
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            end_time = data[0][0] - 1
        except:
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df = df[['timestamp', 'close']].sort_values('timestamp')
        df.to_csv(f'data/crypto/{symbol}.csv', index=False)
        print(f"  {symbol}: {len(df)} days")

# =============================================================================
# 3. FUNDING RATES
# =============================================================================

print("\nDownloading funding rates...")

for symbol in CRYPTO_SYMBOLS:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    end_time = None
    
    for _ in range(10):
        params = {"symbol": symbol, "limit": 1000}
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if not data:
                break
            all_data.extend(data)
            end_time = data[0]["fundingTime"] - 1
        except:
            break
    
    if all_data:
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df[['timestamp', 'fundingRate', 'symbol']].sort_values('timestamp')
        df.to_csv(f'data/funding/{symbol}_funding.csv', index=False)
        print(f"  {symbol}: {len(df)} funding events")

print("\n✅ Data download complete!")
print(f"Total size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk('data') for f in fn) / 1e6:.1f} MB")
```

### Step 2.3: Run Data Download Locally

```powershell
# Run locally first
cd c:\Users\mesof\cift-markets
python scripts/download_data.py
```

---

## PHASE 3: UPLOAD TO GCP (10 minutes)

### Step 3.1: Upload Data to Cloud Storage

```powershell
# Upload data folder to GCP bucket
gcloud storage cp -r data/* gs://cift-stat-arb-data/data/
```

### Step 3.2: Upload Code to GCP

```powershell
# Upload ML code
gcloud storage cp -r cift/ml/* gs://cift-stat-arb-data/code/ml/
```

### Step 3.3: Create Training Script

Create `scripts/gcp_train.py`:

```python
"""
GCP Training Script for Stat Arb
This runs on the GCP VM
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from google.cloud import storage
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'bucket_name': 'cift-stat-arb-data',
    'data_path': 'data',
    'output_path': 'models',
    
    # Training parameters
    'lookback_days': 252 * 3,  # 3 years
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    
    # Stat arb parameters to optimize
    'entry_z_range': [1.5, 2.0, 2.5, 3.0],
    'exit_z_range': [0.0, 0.25, 0.5, 0.75],
    'lookback_range': [10, 15, 20, 30],
    'max_pairs_range': [5, 8, 10, 12, 15],
    
    # Cointegration parameters
    'pval_threshold': 0.05,
    'min_half_life': 5,
    'max_half_life': 60,
}

# =============================================================================
# DATA LOADING
# =============================================================================

def download_from_gcs(bucket_name, source_path, dest_path):
    """Download data from Google Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=source_path)
    for blob in blobs:
        local_path = os.path.join(dest_path, blob.name.replace(source_path + '/', ''))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded: {blob.name}")


def load_equity_data(data_dir):
    """Load equity price data"""
    parquet_path = os.path.join(data_dir, 'equity', 'all_equity_prices.parquet')
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    # Fallback to individual CSVs
    prices = {}
    equity_dir = os.path.join(data_dir, 'equity')
    for f in os.listdir(equity_dir):
        if f.endswith('.csv'):
            symbol = f.replace('.csv', '')
            df = pd.read_csv(os.path.join(equity_dir, f), index_col=0, parse_dates=True)
            prices[symbol] = df['Close']
    
    return pd.DataFrame(prices)


def load_funding_data(data_dir):
    """Load crypto funding rate data"""
    funding = {}
    funding_dir = os.path.join(data_dir, 'funding')
    
    for f in os.listdir(funding_dir):
        if f.endswith('.csv'):
            symbol = f.replace('_funding.csv', '')
            df = pd.read_csv(os.path.join(funding_dir, f), parse_dates=['timestamp'])
            df = df.set_index('timestamp')
            funding[symbol] = df['fundingRate']
    
    return pd.DataFrame(funding)

# =============================================================================
# STAT ARB MODELS (from cift/ml/stat_arb.py)
# =============================================================================

class KalmanState:
    """Kalman filter for dynamic hedge ratio"""
    def __init__(self, beta=1.0, Q=1e-5, R=1e-3):
        self.beta = beta
        self.P = 1.0
        self.Q = Q
        self.R = R
    
    def update(self, x, y):
        y_pred = self.beta * x
        error = y - y_pred
        
        self.P = self.P + self.Q
        K = self.P * x / (x * self.P * x + self.R)
        self.beta = self.beta + K * error
        self.P = (1 - K * x) * self.P
        
        return self.beta


def engle_granger_coint(y1, y2):
    """Engle-Granger cointegration test"""
    from scipy import stats
    
    # OLS regression
    X = np.column_stack([np.ones(len(y2)), y2])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    intercept, hedge = beta[0], beta[1]
    
    # Residuals
    spread = y1 - hedge * y2 - intercept
    
    # ADF test on residuals
    n = len(spread)
    diff = np.diff(spread)
    lag = spread[:-1]
    
    X_adf = np.column_stack([np.ones(len(lag)), lag])
    beta_adf = np.linalg.lstsq(X_adf, diff, rcond=None)[0]
    
    resid = diff - X_adf @ beta_adf
    se = np.sqrt(np.sum(resid**2) / (n - 3) / np.sum((lag - lag.mean())**2))
    adf_stat = beta_adf[1] / se
    
    # Approximate p-value (MacKinnon)
    if adf_stat < -3.96:
        pval = 0.01
    elif adf_stat < -3.41:
        pval = 0.05
    elif adf_stat < -3.12:
        pval = 0.10
    else:
        pval = 0.5
    
    return adf_stat, pval, hedge, intercept


def half_life_mean_reversion(spread):
    """Calculate half-life of mean reversion"""
    spread = np.array(spread)
    lag = spread[:-1]
    diff = np.diff(spread)
    
    X = np.column_stack([np.ones(len(lag)), lag])
    beta = np.linalg.lstsq(X, diff, rcond=None)[0]
    
    if beta[1] >= 0:
        return 999
    
    return -np.log(2) / beta[1]

# =============================================================================
# TRAINING LOGIC
# =============================================================================

def find_cointegrated_pairs(prices, config):
    """Find all cointegrated pairs"""
    symbols = list(prices.columns)
    pairs = []
    
    logger.info(f"Analyzing {len(symbols)} symbols for cointegration...")
    
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            try:
                p1 = prices[s1].dropna().values
                p2 = prices[s2].dropna().values
                
                # Align lengths
                min_len = min(len(p1), len(p2))
                p1 = p1[-min_len:]
                p2 = p2[-min_len:]
                
                if len(p1) < 100:
                    continue
                
                adf, pval, hedge, intercept = engle_granger_coint(p1, p2)
                
                if pval < config['pval_threshold']:
                    spread = p1 - hedge * p2 - intercept
                    hl = half_life_mean_reversion(spread)
                    
                    if config['min_half_life'] <= hl <= config['max_half_life']:
                        score = (1 - pval) * (1 - abs(hl - 20) / 40)
                        pairs.append({
                            's1': s1, 's2': s2,
                            'hedge': hedge, 'intercept': intercept,
                            'half_life': hl, 'pval': pval, 'score': score
                        })
            except:
                pass
    
    pairs.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    
    return pairs


def backtest_params(prices, pairs, entry_z, exit_z, lookback, max_pairs):
    """Backtest with specific parameters"""
    if not pairs:
        return {'sharpe': 0, 'annual': 0, 'maxdd': 0, 'trades': 0}
    
    trading_pairs = pairs[:max_pairs]
    n_days = len(prices)
    
    daily_returns = []
    positions = {}
    trades = 0
    
    for t in range(lookback, n_days):
        day_pnl = 0.0
        
        for pair in trading_pairs:
            s1, s2 = pair['s1'], pair['s2']
            
            try:
                p1 = prices[s1].iloc[:t+1].values
                p2 = prices[s2].iloc[:t+1].values
            except:
                continue
            
            # Kalman filter
            kalman = KalmanState(beta=pair['hedge'], Q=1e-5, R=1e-3)
            for i in range(min(100, len(p1))):
                idx = max(0, len(p1) - 100) + i
                hedge = kalman.update(p2[idx], p1[idx])
            
            spread = p1 - hedge * p2
            recent = spread[-lookback:]
            zscore = (spread[-1] - np.mean(recent)) / (np.std(recent) + 1e-10)
            
            pair_key = f"{s1}/{s2}"
            pos_size = 1.0 / max_pairs
            
            if pair_key in positions:
                pos = positions[pair_key]
                ret1 = (p1[-1] - p1[-2]) / p1[-2]
                ret2 = (p2[-1] - p2[-2]) / p2[-2]
                
                if pos['dir'] == 'long':
                    spread_ret = ret1 - pos['hedge'] * ret2
                else:
                    spread_ret = -ret1 + pos['hedge'] * ret2
                
                day_pnl += spread_ret * pos_size
                
                should_exit = False
                if pos['dir'] == 'long' and zscore >= -exit_z:
                    should_exit = True
                elif pos['dir'] == 'short' and zscore <= exit_z:
                    should_exit = True
                elif abs(zscore) > 4.0:
                    should_exit = True
                
                if should_exit:
                    day_pnl -= 0.001 * pos_size
                    del positions[pair_key]
                    trades += 1
            
            elif len(positions) < max_pairs:
                if zscore <= -entry_z:
                    positions[pair_key] = {'dir': 'long', 'hedge': hedge}
                    trades += 1
                elif zscore >= entry_z:
                    positions[pair_key] = {'dir': 'short', 'hedge': hedge}
                    trades += 1
        
        daily_returns.append(day_pnl)
    
    returns = np.array(daily_returns)
    
    if len(returns) > 30 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        annual = (1 + returns).prod() ** (252 / len(returns)) - 1
        cum = np.cumprod(1 + returns)
        maxdd = (cum / np.maximum.accumulate(cum) - 1).min()
    else:
        sharpe = annual = maxdd = 0
    
    return {
        'sharpe': sharpe,
        'annual': annual,
        'maxdd': maxdd,
        'trades': trades
    }


def grid_search(prices, pairs, config):
    """Grid search over all parameter combinations"""
    results = []
    total = (len(config['entry_z_range']) * len(config['exit_z_range']) * 
             len(config['lookback_range']) * len(config['max_pairs_range']))
    
    logger.info(f"Starting grid search over {total} combinations...")
    
    i = 0
    for entry_z in config['entry_z_range']:
        for exit_z in config['exit_z_range']:
            for lookback in config['lookback_range']:
                for max_pairs in config['max_pairs_range']:
                    i += 1
                    
                    result = backtest_params(prices, pairs, entry_z, exit_z, lookback, max_pairs)
                    result.update({
                        'entry_z': entry_z,
                        'exit_z': exit_z,
                        'lookback': lookback,
                        'max_pairs': max_pairs
                    })
                    results.append(result)
                    
                    if i % 20 == 0:
                        logger.info(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
    
    # Sort by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    return results


def upload_to_gcs(bucket_name, source_path, dest_path):
    """Upload results to GCS"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_path)
    blob.upload_from_filename(source_path)
    logger.info(f"Uploaded: {dest_path}")

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("STAT ARB TRAINING STARTED")
    logger.info("=" * 60)
    
    # Download data
    logger.info("Downloading data from GCS...")
    os.makedirs('local_data', exist_ok=True)
    download_from_gcs(CONFIG['bucket_name'], CONFIG['data_path'], 'local_data')
    
    # Load data
    logger.info("Loading equity data...")
    prices = load_equity_data('local_data')
    logger.info(f"Loaded: {len(prices.columns)} stocks, {len(prices)} days")
    
    # Find pairs
    pairs = find_cointegrated_pairs(prices, CONFIG)
    
    # Split data
    n = len(prices)
    train_end = int(n * CONFIG['train_split'])
    val_end = int(n * (CONFIG['train_split'] + CONFIG['val_split']))
    
    train_prices = prices.iloc[:train_end]
    val_prices = prices.iloc[train_end:val_end]
    test_prices = prices.iloc[val_end:]
    
    logger.info(f"Train: {len(train_prices)} days")
    logger.info(f"Val: {len(val_prices)} days")
    logger.info(f"Test: {len(test_prices)} days")
    
    # Grid search on training data
    logger.info("Running grid search on training data...")
    train_results = grid_search(train_prices, pairs, CONFIG)
    
    # Top 10 results
    logger.info("\nTop 10 parameter combinations (training):")
    for i, r in enumerate(train_results[:10]):
        logger.info(f"  {i+1}. Sharpe={r['sharpe']:.2f} | entry={r['entry_z']}, exit={r['exit_z']}, "
                   f"lookback={r['lookback']}, pairs={r['max_pairs']}")
    
    # Validate top 5 on validation set
    logger.info("\nValidating top 5 on validation data...")
    val_results = []
    for r in train_results[:5]:
        val_result = backtest_params(val_prices, pairs, r['entry_z'], r['exit_z'], 
                                     r['lookback'], r['max_pairs'])
        val_result.update({
            'entry_z': r['entry_z'],
            'exit_z': r['exit_z'],
            'lookback': r['lookback'],
            'max_pairs': r['max_pairs'],
            'train_sharpe': r['sharpe']
        })
        val_results.append(val_result)
        logger.info(f"  Train Sharpe={r['sharpe']:.2f} -> Val Sharpe={val_result['sharpe']:.2f}")
    
    # Select best (highest validation Sharpe)
    best = max(val_results, key=lambda x: x['sharpe'])
    logger.info(f"\nBest parameters: entry={best['entry_z']}, exit={best['exit_z']}, "
               f"lookback={best['lookback']}, pairs={best['max_pairs']}")
    
    # Final test
    logger.info("\nRunning final test...")
    test_result = backtest_params(test_prices, pairs, best['entry_z'], best['exit_z'],
                                  best['lookback'], best['max_pairs'])
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Train Sharpe: {best['train_sharpe']:.2f}")
    logger.info(f"Val Sharpe:   {best['sharpe']:.2f}")
    logger.info(f"Test Sharpe:  {test_result['sharpe']:.2f}")
    logger.info(f"Test Annual:  {test_result['annual']*100:.1f}%")
    logger.info(f"Test MaxDD:   {test_result['maxdd']*100:.1f}%")
    
    # Save results
    output = {
        'best_params': {
            'entry_z': best['entry_z'],
            'exit_z': best['exit_z'],
            'lookback': best['lookback'],
            'max_pairs': best['max_pairs']
        },
        'train_sharpe': best['train_sharpe'],
        'val_sharpe': best['sharpe'],
        'test_sharpe': test_result['sharpe'],
        'test_annual': test_result['annual'],
        'test_maxdd': test_result['maxdd'],
        'pairs': pairs[:20],  # Top 20 pairs
        'all_results': train_results[:50],  # Top 50 combinations
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Upload results
    upload_to_gcs(CONFIG['bucket_name'], 'training_results.json', 
                  f"models/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    upload_to_gcs(CONFIG['bucket_name'], 'training.log',
                  f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger.info("\n✅ Training complete! Results uploaded to GCS.")
    
    return output


if __name__ == "__main__":
    main()
```

---

## PHASE 4: RUN TRAINING ON GCP (20 minutes)

### Step 4.1: SSH into VM

```powershell
# Connect to VM
gcloud compute ssh stat-arb-trainer --zone=us-central1-a
```

### Step 4.2: Setup VM Environment

```bash
# On the VM:
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/stat_arb_env
source ~/stat_arb_env/bin/activate

# Install dependencies
pip install numpy pandas scipy google-cloud-storage pyarrow
```

### Step 4.3: Download Code and Run

```bash
# Download training script from GCS
gsutil cp gs://cift-stat-arb-data/code/gcp_train.py .

# Run training
python gcp_train.py

# Or run in background with logging
nohup python gcp_train.py > output.log 2>&1 &
```

---

## PHASE 5: MONITORING (Ongoing)

### Step 5.1: Monitor VM

```bash
# Check if running
ps aux | grep python

# Watch output
tail -f output.log

# Check GPU usage (if using GPU)
nvidia-smi

# Check memory/CPU
htop
```

### Step 5.2: Monitor from Local Machine

```powershell
# Check VM status
gcloud compute instances describe stat-arb-trainer --zone=us-central1-a

# Stream logs
gcloud compute ssh stat-arb-trainer --zone=us-central1-a --command="tail -f ~/output.log"

# Check GCS for results
gcloud storage ls gs://cift-stat-arb-data/models/
gcloud storage ls gs://cift-stat-arb-data/logs/
```

### Step 5.3: Set Up Cloud Monitoring (Optional)

```bash
# Install monitoring agent on VM
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install
```

---

## PHASE 6: WHAT TO LOOK FOR

### ✅ Good Signs

| Metric | Good Value | What It Means |
|--------|------------|---------------|
| **Train Sharpe** | 0.8 - 1.5 | Model is learning |
| **Val Sharpe** | Within 20% of train | Not overfitting |
| **Test Sharpe** | > 0.7 | Generalizes well |
| **Annual Return** | 8-15% | Realistic alpha |
| **Max Drawdown** | > -15% | Acceptable risk |
| **Trades** | 200-500/year | Enough liquidity |

### ⚠️ Warning Signs

| Issue | Sign | Action |
|-------|------|--------|
| **Overfitting** | Train >> Val Sharpe | Reduce parameters, add regularization |
| **Underfitting** | All Sharpes < 0.5 | Expand universe, reduce constraints |
| **Look-ahead bias** | Test >> Train | Check data alignment |
| **Too few trades** | < 50/year | Lower entry threshold |
| **Too many trades** | > 1000/year | Raise entry threshold |

### 📊 Key Metrics to Track

```
1. Sharpe Ratio (train/val/test) - Should be similar
2. Annual Return - Should be realistic (5-20%)
3. Max Drawdown - Should be < -20%
4. Number of Trades - Should be 200-500/year
5. Win Rate - Should be 45-55%
6. Average Trade Duration - Should match half-life
```

---

## PHASE 7: AFTER TRAINING - WHAT TO DO NEXT

### Step 7.1: Download Results

```powershell
# Download results locally
gcloud storage cp gs://cift-stat-arb-data/models/training_results_*.json .
gcloud storage cp gs://cift-stat-arb-data/logs/training_*.log .
```

### Step 7.2: Analyze Results

```python
# analyze_results.py
import json

with open('training_results_XXXXXXXX.json') as f:
    results = json.load(f)

print("Best Parameters:")
print(f"  Entry Z: {results['best_params']['entry_z']}")
print(f"  Exit Z: {results['best_params']['exit_z']}")
print(f"  Lookback: {results['best_params']['lookback']}")
print(f"  Max Pairs: {results['best_params']['max_pairs']}")

print(f"\nPerformance:")
print(f"  Train Sharpe: {results['train_sharpe']:.2f}")
print(f"  Val Sharpe: {results['val_sharpe']:.2f}")
print(f"  Test Sharpe: {results['test_sharpe']:.2f}")

print(f"\nTop Pairs:")
for p in results['pairs'][:10]:
    print(f"  {p['s1']}/{p['s2']}: HL={p['half_life']:.0f}, score={p['score']:.2f}")
```

### Step 7.3: Update Production Code

```powershell
# Update production_stat_arb.py with best parameters
# Replace the CONFIG section with optimized values
```

### Step 7.4: Deploy to Paper Trading

```python
# Create paper trading script with best params
# Run on small position sizes first
# Monitor for 2-4 weeks before real money
```

### Step 7.5: Cleanup GCP Resources

```powershell
# Stop VM to save costs
gcloud compute instances stop stat-arb-trainer --zone=us-central1-a

# Or delete if done
gcloud compute instances delete stat-arb-trainer --zone=us-central1-a

# Keep bucket for data (cheap storage)
```

---

## COST ESTIMATE

| Resource | Cost/Hour | Daily (8hr) | Monthly |
|----------|-----------|-------------|---------|
| e2-standard-8 (CPU) | $0.27 | $2.16 | ~$65 |
| n1-standard-8 + T4 GPU | $0.95 | $7.60 | ~$230 |
| Storage (50GB) | - | - | ~$1 |
| **Total (CPU training)** | - | **~$3** | - |

---

## QUICK REFERENCE - COMMAND CHEATSHEET

```powershell
# === LOCAL ===
# Download data
python scripts/download_data.py

# Upload to GCS
gcloud storage cp -r data/* gs://cift-stat-arb-data/data/
gcloud storage cp scripts/gcp_train.py gs://cift-stat-arb-data/code/

# === GCP ===
# Start VM
gcloud compute instances start stat-arb-trainer --zone=us-central1-a

# SSH
gcloud compute ssh stat-arb-trainer --zone=us-central1-a

# Stop VM
gcloud compute instances stop stat-arb-trainer --zone=us-central1-a

# === ON VM ===
source ~/stat_arb_env/bin/activate
python gcp_train.py

# === MONITORING ===
tail -f output.log
gcloud storage ls gs://cift-stat-arb-data/models/
```

---

## FILES SUMMARY

| File | Location | Purpose |
|------|----------|---------|
| `download_data.py` | `scripts/` | Download training data locally |
| `gcp_train.py` | `scripts/` → GCS | Main training script for GCP |
| `all_equity_prices.parquet` | `data/equity/` → GCS | Combined equity prices |
| `*_funding.csv` | `data/funding/` → GCS | Funding rate data |
| `training_results.json` | GCS `models/` | Output: best params + results |
| `training.log` | GCS `logs/` | Training logs |

---

## NEXT STEPS AFTER GCP TRAINING

1. **Validate** - Check test Sharpe is close to train/val
2. **Paper Trade** - Run for 2-4 weeks with no real money
3. **Small Live** - Start with 5-10% of intended capital
4. **Scale Up** - Gradually increase if performing well
5. **Monitor** - Track daily P&L, Sharpe, drawdowns
6. **Retrain** - Monthly with new data
