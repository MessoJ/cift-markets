"""
INTRADAY DATA DOWNLOADER
========================

Downloads 1-minute and 1-hour data for testing "Institutional Mode".
Uses yfinance (free) which allows:
- 7 days of 1-minute data
- 730 days of 1-hour data

For production backtesting (years of 1-min data), we need Alpaca or Databento.
"""

import yfinance as yf
import pandas as pd
import os
import time

# Create directories
os.makedirs('data/intraday/1h', exist_ok=True)
os.makedirs('data/intraday/1m', exist_ok=True)

# Top liquid stocks for intraday stat arb
SYMBOLS = [
    'XOM', 'CVX', 'COP', # Energy
    'JPM', 'BAC', 'WFC', # Banks
    'MSFT', 'AAPL', 'GOOGL', # Tech
    'GLD', 'SLV', # Commodities
    'SPY', 'QQQ', 'IWM' # ETFs (Great for pairs)
]

def download_intraday():
    print("="*60)
    print("DOWNLOADING INTRADAY DATA (FREE TIER)")
    print("="*60)
    
    # 1. Download 1-Hour Data (Last 2 years)
    print("\n[1/2] Downloading 1-Hour Data (2 Years)...")
    for symbol in SYMBOLS:
        print(f"  Fetching {symbol}...", end='\r')
        try:
            # yfinance limit for 1h is 730 days
            df = yf.download(symbol, period="2y", interval="1h", progress=False)
            if not df.empty:
                df.to_parquet(f'data/intraday/1h/{symbol}.parquet')
        except Exception as e:
            print(f"  Failed {symbol}: {e}")
            
    print(f"\n  ✅ 1-Hour data saved to data/intraday/1h/")

    # 2. Download 1-Minute Data (Last 7 days)
    print("\n[2/2] Downloading 1-Minute Data (Last 7 Days)...")
    print("  Note: This is for immediate testing. For full backtest, we need paid data.")
    
    for symbol in SYMBOLS:
        print(f"  Fetching {symbol}...", end='\r')
        try:
            # yfinance limit for 1m is 7 days
            df = yf.download(symbol, period="7d", interval="1m", progress=False)
            if not df.empty:
                df.to_parquet(f'data/intraday/1m/{symbol}.parquet')
        except Exception as e:
            print(f"  Failed {symbol}: {e}")
            
    print(f"\n  ✅ 1-Minute data saved to data/intraday/1m/")
    print("\nDONE. You can now run the Institutional Engine on this data.")

if __name__ == "__main__":
    download_intraday()
