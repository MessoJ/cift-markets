"""
Data Download Script for GCP Training
Run this LOCALLY first, then upload to GCP

Usage:
    cd C:/Users/mesof/cift-markets
    python scripts/download_data.py
"""

import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import json
import time

# Create data directory
os.makedirs('data/equity', exist_ok=True)
os.makedirs('data/crypto', exist_ok=True)
os.makedirs('data/funding', exist_ok=True)

print("=" * 60)
print("DATA DOWNLOAD FOR GCP TRAINING")
print("=" * 60)

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

print(f"\n[1/3] Downloading {len(EQUITY_SYMBOLS)} equity symbols...")
end = datetime.now()
start = end - timedelta(days=5*365)  # 5 years

success_count = 0
for i, symbol in enumerate(EQUITY_SYMBOLS):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if len(df) > 100:
            df.to_csv(f'data/equity/{symbol}.csv')
            print(f"  [{i+1}/{len(EQUITY_SYMBOLS)}] {symbol}: {len(df)} days ✓")
            success_count += 1
        else:
            print(f"  [{i+1}/{len(EQUITY_SYMBOLS)}] {symbol}: Not enough data")
    except Exception as e:
        print(f"  [{i+1}/{len(EQUITY_SYMBOLS)}] {symbol}: FAILED - {e}")
    
    time.sleep(0.1)  # Rate limiting

print(f"\nEquity download complete: {success_count}/{len(EQUITY_SYMBOLS)} symbols")

# Create combined parquet file
print("Creating combined equity file...")
all_data = {}
for symbol in EQUITY_SYMBOLS:
    try:
        df = pd.read_csv(f'data/equity/{symbol}.csv', index_col=0, parse_dates=True)
        all_data[symbol] = df['Close']
    except:
        pass

if all_data:
    combined = pd.DataFrame(all_data).dropna()
    combined.to_parquet('data/equity/all_equity_prices.parquet')
    combined.to_csv('data/equity/all_equity_prices.csv')
    print(f"Combined file: {len(combined)} days, {len(combined.columns)} stocks")

# =============================================================================
# 2. CRYPTO SPOT PRICES
# =============================================================================

CRYPTO_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
                  'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']

print(f"\n[2/3] Downloading {len(CRYPTO_SYMBOLS)} crypto spot prices...")

for symbol in CRYPTO_SYMBOLS:
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_time = None
    
    for _ in range(5):  # ~500 days
        params = {
            "symbol": symbol,
            "interval": "1d",
            "limit": 1000
        }
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not data or isinstance(data, dict):
                break
            all_data.extend(data)
            end_time = data[0][0] - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close'] = df['close'].astype(float)
        df = df[['timestamp', 'close']].sort_values('timestamp').drop_duplicates()
        df.to_csv(f'data/crypto/{symbol}.csv', index=False)
        print(f"  {symbol}: {len(df)} days ✓")

# =============================================================================
# 3. FUNDING RATES
# =============================================================================

print(f"\n[3/3] Downloading funding rates...")

for symbol in CRYPTO_SYMBOLS:
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_data = []
    end_time = None
    
    for _ in range(5):
        params = {"symbol": symbol, "limit": 1000}
        if end_time:
            params["endTime"] = end_time
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not data or isinstance(data, dict):
                break
            all_data.extend(data)
            end_time = data[0]["fundingTime"] - 1
            time.sleep(0.1)
        except Exception as e:
            print(f"  {symbol}: Error - {e}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df = df[['timestamp', 'fundingRate', 'symbol']].sort_values('timestamp').drop_duplicates()
        df.to_csv(f'data/funding/{symbol}_funding.csv', index=False)
        print(f"  {symbol}: {len(df)} funding events ✓")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

# Calculate total size
total_size = 0
for dp, dn, fn in os.walk('data'):
    for f in fn:
        total_size += os.path.getsize(os.path.join(dp, f))

print(f"\nData directory: data/")
print(f"  equity/: {len(os.listdir('data/equity'))} files")
print(f"  crypto/: {len(os.listdir('data/crypto'))} files")
print(f"  funding/: {len(os.listdir('data/funding'))} files")
print(f"\nTotal size: {total_size / 1e6:.1f} MB")

print("\n✅ Next step: Upload to GCP")
print("   gcloud storage cp -r data/* gs://cift-stat-arb-data/data/")
