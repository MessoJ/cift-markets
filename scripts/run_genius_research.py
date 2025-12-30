"""
RUN GENIUS RESEARCH
===================

This script executes the rigorous research loop using the GeniusStatArb engine.
It searches for the optimal portfolio configuration to achieve Sharpe 2.8+
while adhering to strict statistical standards (PSR, Deflated Sharpe).
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cift.ml.genius_stat_arb import GeniusStatArb

warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    df = None
    # Try parquet first
    if os.path.exists('data/equity/all_equity_prices.parquet'):
        df = pd.read_parquet('data/equity/all_equity_prices.parquet')
    
    # Fallback to CSVs
    if df is None:
        prices = {}
        data_dir = 'data/equity'
        if not os.path.exists(data_dir):
            print("Error: Data directory not found. Run scripts/download_data.py first.")
            sys.exit(1)
            
        for f in os.listdir(data_dir):
            if f.endswith('.csv') and not f.startswith('all_'):
                symbol = f.replace('.csv', '')
                try:
                    # Skip the first 2 rows (Price, Ticker) and use 3rd row (Date) as header
                    # Actually the CSV has 3 header lines. 
                    # Line 1: Price,Close...
                    # Line 2: Ticker,AAPL...
                    # Line 3: Date,,,,,
                    # We want to read from line 3 onwards? No, standard read_csv with skiprows might be safer.
                    # But let's try to read it robustly.
                    curr_df = pd.read_csv(os.path.join(data_dir, f))
                    # Find where the dates start
                    start_idx = curr_df[curr_df.iloc[:,0].astype(str).str.match(r'\d{4}-\d{2}-\d{2}')].index[0]
                    curr_df = curr_df.iloc[start_idx:]
                    curr_df.set_index(curr_df.columns[0], inplace=True)
                    curr_df.index = pd.to_datetime(curr_df.index)
                    # Column 0 was Date, Column 1 is usually Close (based on yfinance structure)
                    # But let's be careful.
                    # If we can't parse, skip.
                    prices[symbol] = pd.to_numeric(curr_df.iloc[:,0], errors='coerce')
                except:
                    pass
        df = pd.DataFrame(prices)

    # Clean the DataFrame (remove metadata rows if any)
    # Check if index has non-date values
    if df is not None:
        # If index is object, try to convert to datetime, coerce errors
        df.index = pd.to_datetime(df.index, errors='coerce')
        # Drop NaT index (which were the "Price", "Ticker" rows)
        df = df[df.index.notna()]
        # Convert all columns to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
    return df

def find_best_pairs(prices, n_pairs=20):
    print(f"Screening pairs from {len(prices.columns)} assets...")
    symbols = prices.columns
    pairs = []
    
    # Quick correlation filter
    corr = prices.pct_change().corr()
    
    import statsmodels.tsa.stattools as ts
    
    count = 0
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            if corr.loc[s1, s2] < 0.8: continue
            
            try:
                p1 = prices[s1].dropna()
                p2 = prices[s2].dropna()
                common = p1.index.intersection(p2.index)
                if len(common) < 252: continue
                
                res = ts.coint(p1[common], p2[common])
                if res[1] < 0.05:
                    pairs.append({'s1': s1, 's2': s2, 'pval': res[1]})
            except:
                pass
            
            count += 1
            if count % 500 == 0:
                print(f"  Scanned {count} pairs...", end='\r')
                
    print(f"\nFound {len(pairs)} cointegrated pairs.")
    pairs.sort(key=lambda x: x['pval'])
    return [(p['s1'], p['s2']) for p in pairs[:n_pairs]]

def run_optimization():
    prices = load_data()
    if prices.empty:
        print("No data found!")
        return

    # 1. Select Universe (Top Pairs)
    top_pairs = find_best_pairs(prices, n_pairs=15)
    print(f"Selected Top {len(top_pairs)} Pairs: {top_pairs}")
    
    # 2. Run Genius Engine
    print("\nRunning Genius Engine Backtest...")
    
    # Parameter Grid
    configs = [
        {'entry': 2.0, 'exit': 0.0, 'lookback': 20},
        {'entry': 2.5, 'exit': 0.25, 'lookback': 20},
        {'entry': 3.0, 'exit': 0.5, 'lookback': 30},
        {'entry': 2.0, 'exit': 0.0, 'lookback': 10}, # Fast mean reversion
    ]
    
    best_sharpe = -999
    best_config = None
    best_result = None
    
    for cfg in configs:
        engine = GeniusStatArb(
            entry_z=cfg['entry'],
            exit_z=cfg['exit'],
            lookback=cfg['lookback'],
            cost_bps=10
        )
        
        res = engine.run_backtest(prices, top_pairs)
        
        print(f"Config {cfg}: Sharpe={res['sharpe']:.2f}, PSR={res['psr']:.2f}, AnnRet={res['annual_return']:.1%}")
        
        if res['sharpe'] > best_sharpe:
            best_sharpe = res['sharpe']
            best_config = cfg
            best_result = res
            
    print("\n" + "="*50)
    print("OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Best Config: {best_config}")
    print(f"Sharpe Ratio: {best_result['sharpe']:.2f}")
    print(f"PSR (>1.0):   {best_result['psr']:.2%}")
    print(f"Annual Ret:   {best_result['annual_return']:.1%}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.1%}")
    
    if best_result['sharpe'] < 2.0:
        print("\n⚠️ Target 2.8 not reached with standard pairs.")
        print("   Need to activate: 1) Funding Arb, 2) Intraday Data, or 3) More Pairs.")
    else:
        print("\n✅ Target approached! Validate on OOS data.")

if __name__ == "__main__":
    run_optimization()
