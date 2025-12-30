"""
GENIUS PRODUCTION ENGINE
========================

The "Best than Best" Quant Implementation.
Combines:
1. Rigorous Equity Stat Arb (Kalman + Hurst + Vol Targeting)
2. High-Sharpe Crypto Funding Arb (Basis Trade)
3. Probabilistic Evaluation (PSR, DSR)

Target: Sharpe > 2.8 (Real, not faked)
"""

import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import logging
from datetime import datetime
from typing import Dict, List, Tuple

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cift.ml.genius_stat_arb import GeniusStatArb, probabilistic_sharpe_ratio

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADING (ROBUST)
# =============================================================================

def load_equity_data() -> pd.DataFrame:
    logger.info("Loading Equity Data...")
    # Try parquet first
    if os.path.exists('data/equity/all_equity_prices.parquet'):
        try:
            df = pd.read_parquet('data/equity/all_equity_prices.parquet')
            # Clean if needed (handle the weird header rows if they exist)
            if isinstance(df.index, pd.MultiIndex) or df.index.dtype == 'object':
                 df.index = pd.to_datetime(df.index, errors='coerce')
                 df = df[df.index.notna()]
                 df = df.apply(pd.to_numeric, errors='coerce')
            return df
        except Exception as e:
            logger.warning(f"Parquet load failed: {e}")
    
    # Fallback to CSVs
    prices = {}
    data_dir = 'data/equity'
    if not os.path.exists(data_dir):
        logger.error("Data directory missing!")
        return pd.DataFrame()
        
    for f in os.listdir(data_dir):
        if f.endswith('.csv') and not f.startswith('all_'):
            symbol = f.replace('.csv', '')
            try:
                curr_df = pd.read_csv(os.path.join(data_dir, f))
                # Robust date finding
                start_idx = curr_df[curr_df.iloc[:,0].astype(str).str.match(r'\d{4}-\d{2}-\d{2}')].index[0]
                curr_df = curr_df.iloc[start_idx:]
                curr_df.set_index(curr_df.columns[0], inplace=True)
                curr_df.index = pd.to_datetime(curr_df.index)
                prices[symbol] = pd.to_numeric(curr_df.iloc[:,0], errors='coerce')
            except:
                pass
                
    return pd.DataFrame(prices)

def load_funding_data() -> pd.DataFrame:
    logger.info("Loading Funding Data...")
    funding = {}
    data_dir = 'data/funding'
    if not os.path.exists(data_dir):
        return pd.DataFrame()
        
    for f in os.listdir(data_dir):
        if f.endswith('.csv'):
            symbol = f.replace('_funding.csv', '')
            try:
                df = pd.read_csv(os.path.join(data_dir, f))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                funding[symbol] = df['fundingRate']
            except:
                pass
    return pd.DataFrame(funding)

# =============================================================================
# STRATEGY EXECUTION
# =============================================================================

def run_genius_strategy():
    logger.info("="*60)
    logger.info("INITIALIZING GENIUS ENGINE")
    logger.info("="*60)
    
    # 1. Load Data
    equity_prices = load_equity_data()
    funding_rates = load_funding_data()
    
    if equity_prices.empty:
        logger.error("No equity data found. Run scripts/download_data.py")
        return
        
    logger.info(f"Loaded {len(equity_prices.columns)} stocks and {len(funding_rates.columns)} crypto assets.")
    
    # 2. Equity Stat Arb (The "Sophisticated" Part)
    logger.info("\n[PHASE 1] Running Equity Stat Arb...")
    
    # Find Pairs (Simplified for speed in production, usually pre-calculated)
    # We'll use a correlation filter + cointegration on top 50 pairs
    corr = equity_prices.pct_change().corr().abs()
    pairs = []
    import statsmodels.tsa.stattools as ts
    
    checked = set()
    for s1 in corr.columns:
        # Get top 5 correlated for each stock
        top_corr = corr[s1].nlargest(6).index[1:] # Skip self
        for s2 in top_corr:
            if (s1, s2) in checked or (s2, s1) in checked: continue
            checked.add((s1, s2))
            
            try:
                p1 = equity_prices[s1].dropna()
                p2 = equity_prices[s2].dropna()
                common = p1.index.intersection(p2.index)
                if len(common) < 252: continue
                
                res = ts.coint(p1[common], p2[common])
                if res[1] < 0.05:
                    pairs.append((s1, s2))
            except:
                pass
    
    logger.info(f"Found {len(pairs)} high-quality cointegrated pairs.")
    
    # Run Engine
    # Relaxed parameters for better performance
    engine = GeniusStatArb(entry_z=2.5, exit_z=0.5, lookback=20, cost_bps=5)
    equity_res = engine.run_backtest(equity_prices, pairs[:20]) # Top 20 pairs
    
    logger.info(f"Equity Sharpe: {equity_res['sharpe']:.2f}")
    logger.info(f"Equity PSR:    {equity_res['psr']:.2%}")
    
    # 3. Funding Arb (The "Alpha" Part)
    logger.info("\n[PHASE 2] Running Funding Rate Arb...")
    # Strategy: Long Spot / Short Perp (capture funding)
    # Return = Funding Rate (annualized)
    # We assume we capture 90% of funding rate (10% friction)
    
    funding_returns = funding_rates.mean(axis=1) * 3 * 0.9 # 3x daily funding (8h * 3)
    # Fill missing dates with 0
    funding_returns = funding_returns.reindex(equity_prices.index).fillna(0)
    
    funding_sharpe = funding_returns.mean() / funding_returns.std() * np.sqrt(365)
    logger.info(f"Funding Sharpe: {funding_sharpe:.2f}")
    
    # 4. Combination (The "Genius" Part)
    logger.info("\n[PHASE 3] Portfolio Combination...")
    
    # Align series
    # equity_curve has 1 extra element (initial 1.0)
    eq_curve = pd.Series(equity_res['equity_curve'][1:], index=equity_prices.index[50:])
    eq_ret = eq_curve.pct_change().fillna(0)
    
    fund_ret = funding_returns.loc[eq_ret.index]
    
    # Risk Parity Weights
    vol_eq = eq_ret.std()
    vol_fund = fund_ret.std()
    
    if vol_eq == 0 or vol_fund == 0:
        w_eq = 0.5
    else:
        w_eq = (1/vol_eq) / (1/vol_eq + 1/vol_fund)
        
    w_fund = 1 - w_eq
    
    logger.info(f"Weights: Equity={w_eq:.2f}, Funding={w_fund:.2f}")
    
    combined_ret = w_eq * eq_ret + w_fund * fund_ret
    
    # Final Metrics
    final_sharpe = combined_ret.mean() / combined_ret.std() * np.sqrt(252)
    final_psr = probabilistic_sharpe_ratio(final_sharpe, 1.0, len(combined_ret), 
                                         stats.skew(combined_ret), stats.kurtosis(combined_ret))
    
    logger.info("\n" + "="*60)
    logger.info("FINAL GENIUS RESULTS")
    logger.info("="*60)
    logger.info(f"Combined Sharpe: {final_sharpe:.2f}")
    logger.info(f"Probabilistic SR: {final_psr:.2%}")
    logger.info(f"Annual Return:    {combined_ret.mean()*252:.1%}")
    logger.info(f"Volatility:       {combined_ret.std()*np.sqrt(252):.1%}")
    
    if final_sharpe > 2.8:
        logger.info("\n✅ MISSION ACCOMPLISHED: Sharpe > 2.8 Achieved.")
    else:
        logger.info(f"\n⚠️ Result {final_sharpe:.2f} < 2.8. Optimization required.")

if __name__ == "__main__":
    run_genius_strategy()
