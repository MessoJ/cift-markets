"""
GCP Training Script for Stat Arb
================================

This runs on the GCP VM to train and optimize parameters.

Setup on VM:
    source ~/stat_arb_env/bin/activate
    gsutil cp gs://cift-stat-arb-data/code/gcp_train.py .
    python gcp_train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

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

def download_from_gcs(bucket_name: str, source_path: str, dest_path: str):
    """Download data from Google Cloud Storage"""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        blobs = list(bucket.list_blobs(prefix=source_path))
        logger.info(f"Found {len(blobs)} files in gs://{bucket_name}/{source_path}")
        
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            local_path = os.path.join(dest_path, blob.name.replace(source_path + '/', ''))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info(f"  Downloaded: {blob.name}")
    except ImportError:
        logger.warning("google-cloud-storage not installed, using local data")
    except Exception as e:
        logger.error(f"GCS download failed: {e}")


def load_equity_data(data_dir: str) -> pd.DataFrame:
    """Load equity price data"""
    parquet_path = os.path.join(data_dir, 'equity', 'all_equity_prices.parquet')
    csv_path = os.path.join(data_dir, 'equity', 'all_equity_prices.csv')
    
    if os.path.exists(parquet_path):
        logger.info(f"Loading from parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)
    
    if os.path.exists(csv_path):
        logger.info(f"Loading from csv: {csv_path}")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Fallback to individual CSVs
    logger.info("Loading individual CSV files...")
    prices = {}
    equity_dir = os.path.join(data_dir, 'equity')
    
    if not os.path.exists(equity_dir):
        logger.error(f"Directory not found: {equity_dir}")
        return pd.DataFrame()
    
    for f in os.listdir(equity_dir):
        if f.endswith('.csv') and not f.startswith('all_'):
            symbol = f.replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(equity_dir, f), index_col=0, parse_dates=True)
                prices[symbol] = df['Close']
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    
    return pd.DataFrame(prices)


def load_funding_data(data_dir: str) -> pd.DataFrame:
    """Load crypto funding rate data"""
    funding = {}
    funding_dir = os.path.join(data_dir, 'funding')
    
    if not os.path.exists(funding_dir):
        logger.warning(f"Funding directory not found: {funding_dir}")
        return pd.DataFrame()
    
    for f in os.listdir(funding_dir):
        if f.endswith('.csv'):
            symbol = f.replace('_funding.csv', '')
            try:
                df = pd.read_csv(os.path.join(funding_dir, f), parse_dates=['timestamp'])
                df = df.set_index('timestamp')
                funding[symbol] = df['fundingRate']
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
    
    return pd.DataFrame(funding)

# =============================================================================
# STAT ARB MODELS
# =============================================================================

class KalmanState:
    """Kalman filter for dynamic hedge ratio estimation"""
    def __init__(self, beta: float = 1.0, Q: float = 1e-5, R: float = 1e-3):
        self.beta = beta
        self.P = 1.0
        self.Q = Q
        self.R = R
    
    def update(self, x: float, y: float) -> float:
        y_pred = self.beta * x
        error = y - y_pred
        
        self.P = self.P + self.Q
        K = self.P * x / (x * self.P * x + self.R)
        self.beta = self.beta + K * error
        self.P = (1 - K * x) * self.P
        
        return self.beta


def engle_granger_coint(y1: np.ndarray, y2: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Engle-Granger cointegration test
    Returns: (adf_stat, pvalue, hedge_ratio, intercept)
    """
    # OLS regression: y1 = intercept + hedge * y2 + error
    X = np.column_stack([np.ones(len(y2)), y2])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]
    intercept, hedge = beta[0], beta[1]
    
    # Residuals (spread)
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
    
    # Approximate p-value (MacKinnon critical values for cointegration)
    if adf_stat < -3.96:
        pval = 0.01
    elif adf_stat < -3.41:
        pval = 0.05
    elif adf_stat < -3.12:
        pval = 0.10
    else:
        pval = 0.5
    
    return adf_stat, pval, hedge, intercept


def half_life_mean_reversion(spread: np.ndarray) -> float:
    """Calculate half-life of mean reversion using OLS"""
    spread = np.array(spread)
    lag = spread[:-1]
    diff = np.diff(spread)
    
    X = np.column_stack([np.ones(len(lag)), lag])
    beta = np.linalg.lstsq(X, diff, rcond=None)[0]
    
    if beta[1] >= 0:
        return 999  # Not mean-reverting
    
    return -np.log(2) / beta[1]


def calculate_hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """Calculate Hurst exponent (< 0.5 = mean-reverting)"""
    if len(series) < max_lag * 2:
        return 0.5
    
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        tau.append(np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))))
    
    tau = np.array(tau)
    valid = tau > 0
    if not valid.any():
        return 0.5
    
    try:
        reg = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
        return reg[0]
    except:
        return 0.5

# =============================================================================
# TRAINING LOGIC
# =============================================================================

def find_cointegrated_pairs(prices: pd.DataFrame, config: Dict) -> List[Dict]:
    """Find all cointegrated pairs in the universe"""
    symbols = list(prices.columns)
    pairs = []
    
    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    logger.info(f"Analyzing {total_pairs} potential pairs from {len(symbols)} symbols...")
    
    analyzed = 0
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            analyzed += 1
            
            try:
                p1 = prices[s1].dropna().values
                p2 = prices[s2].dropna().values
                
                # Align lengths
                min_len = min(len(p1), len(p2))
                p1 = p1[-min_len:]
                p2 = p2[-min_len:]
                
                if len(p1) < 100:
                    continue
                
                # Cointegration test
                adf, pval, hedge, intercept = engle_granger_coint(p1, p2)
                
                if pval >= config['pval_threshold']:
                    continue
                
                # Half-life
                spread = p1 - hedge * p2 - intercept
                hl = half_life_mean_reversion(spread)
                
                if not (config['min_half_life'] <= hl <= config['max_half_life']):
                    continue
                
                # Hurst exponent
                hurst = calculate_hurst(spread)
                
                # Score: prioritize low p-value, good half-life, mean-reverting
                score = (1 - pval) * (1 - abs(hl - 20) / 40) * (1 - hurst)
                
                pairs.append({
                    's1': s1, 's2': s2,
                    'hedge': float(hedge), 
                    'intercept': float(intercept),
                    'half_life': float(hl), 
                    'pval': float(pval),
                    'hurst': float(hurst),
                    'score': float(score)
                })
                
            except Exception as e:
                continue
            
            if analyzed % 500 == 0:
                logger.info(f"  Progress: {analyzed}/{total_pairs} pairs analyzed, {len(pairs)} cointegrated")
    
    pairs.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    
    return pairs


def backtest_params(
    prices: pd.DataFrame, 
    pairs: List[Dict], 
    entry_z: float, 
    exit_z: float, 
    lookback: int, 
    max_pairs: int
) -> Dict:
    """Backtest strategy with specific parameters"""
    
    if not pairs:
        return {'sharpe': 0, 'annual': 0, 'maxdd': 0, 'trades': 0, 'win_rate': 0}
    
    trading_pairs = pairs[:max_pairs]
    n_days = len(prices)
    
    daily_returns = []
    positions = {}
    trades = 0
    wins = 0
    
    for t in range(lookback, n_days):
        day_pnl = 0.0
        
        for pair in trading_pairs:
            s1, s2 = pair['s1'], pair['s2']
            
            try:
                p1 = prices[s1].iloc[:t+1].values
                p2 = prices[s2].iloc[:t+1].values
            except:
                continue
            
            if len(p1) < lookback or len(p2) < lookback:
                continue
            
            # Kalman filter for dynamic hedge
            kalman = KalmanState(beta=pair['hedge'], Q=1e-5, R=1e-3)
            for i in range(min(100, len(p1))):
                idx = max(0, len(p1) - 100) + i
                hedge = kalman.update(p2[idx], p1[idx])
            
            # Calculate z-score
            spread = p1 - hedge * p2
            recent = spread[-lookback:]
            mu = np.mean(recent)
            sigma = np.std(recent) + 1e-10
            zscore = (spread[-1] - mu) / sigma
            
            pair_key = f"{s1}/{s2}"
            pos_size = 1.0 / max_pairs
            
            # Position management
            if pair_key in positions:
                pos = positions[pair_key]
                
                # Daily P&L
                ret1 = (p1[-1] - p1[-2]) / p1[-2] if len(p1) > 1 else 0
                ret2 = (p2[-1] - p2[-2]) / p2[-2] if len(p2) > 1 else 0
                
                if pos['dir'] == 'long':
                    spread_ret = ret1 - pos['hedge'] * ret2
                else:
                    spread_ret = -ret1 + pos['hedge'] * ret2
                
                day_pnl += spread_ret * pos_size
                
                # Exit logic
                should_exit = False
                if pos['dir'] == 'long' and zscore >= -exit_z:
                    should_exit = True
                elif pos['dir'] == 'short' and zscore <= exit_z:
                    should_exit = True
                elif abs(zscore) > 4.0:  # Stop loss
                    should_exit = True
                
                if should_exit:
                    day_pnl -= 0.001 * pos_size  # Transaction cost
                    
                    # Track win/loss
                    if pos['dir'] == 'long':
                        trade_pnl = zscore - pos['entry_z']
                    else:
                        trade_pnl = pos['entry_z'] - zscore
                    
                    if trade_pnl > 0:
                        wins += 1
                    
                    del positions[pair_key]
                    trades += 1
            
            elif len(positions) < max_pairs:
                # Entry logic
                if zscore <= -entry_z:
                    positions[pair_key] = {
                        'dir': 'long', 
                        'hedge': hedge,
                        'entry_z': zscore
                    }
                    trades += 1
                elif zscore >= entry_z:
                    positions[pair_key] = {
                        'dir': 'short', 
                        'hedge': hedge,
                        'entry_z': zscore
                    }
                    trades += 1
        
        daily_returns.append(day_pnl)
    
    returns = np.array(daily_returns)
    
    # Calculate metrics
    if len(returns) > 30 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        total_ret = (1 + returns).prod() - 1
        annual = float((1 + total_ret) ** (252 / len(returns)) - 1)
        cum = np.cumprod(1 + returns)
        maxdd = float((cum / np.maximum.accumulate(cum) - 1).min())
        win_rate = float(wins / trades) if trades > 0 else 0
    else:
        sharpe = annual = maxdd = win_rate = 0.0
    
    return {
        'sharpe': sharpe,
        'annual': annual,
        'maxdd': maxdd,
        'trades': trades,
        'win_rate': win_rate
    }


def grid_search(prices: pd.DataFrame, pairs: List[Dict], config: Dict) -> List[Dict]:
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
                        best_so_far = max(results, key=lambda x: x['sharpe'])
                        logger.info(f"Progress: {i}/{total} ({i/total*100:.0f}%) | "
                                   f"Best Sharpe so far: {best_so_far['sharpe']:.2f}")
    
    # Sort by Sharpe ratio
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    
    return results


def upload_to_gcs(bucket_name: str, source_path: str, dest_path: str):
    """Upload results to Google Cloud Storage"""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(dest_path)
        blob.upload_from_filename(source_path)
        logger.info(f"Uploaded: gs://{bucket_name}/{dest_path}")
    except ImportError:
        logger.warning("google-cloud-storage not installed, skipping upload")
    except Exception as e:
        logger.error(f"Upload failed: {e}")

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("STAT ARB TRAINING STARTED")
    logger.info(f"Start time: {start_time}")
    logger.info("=" * 70)
    
    # Create local data directory
    os.makedirs('local_data', exist_ok=True)
    
    # Try to download from GCS, or use local data
    logger.info("\nStep 1: Loading data...")
    
    if os.path.exists('data'):
        logger.info("Using local data/ directory")
        data_dir = 'data'
    else:
        logger.info("Downloading data from GCS...")
        download_from_gcs(CONFIG['bucket_name'], CONFIG['data_path'], 'local_data')
        data_dir = 'local_data'
    
    # Load equity data
    prices = load_equity_data(data_dir)
    
    if prices.empty:
        logger.error("No price data loaded!")
        return None
    
    logger.info(f"Loaded: {len(prices.columns)} stocks, {len(prices)} days")
    logger.info(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    
    # Find cointegrated pairs
    logger.info("\nStep 2: Finding cointegrated pairs...")
    pairs = find_cointegrated_pairs(prices, CONFIG)
    
    if not pairs:
        logger.error("No cointegrated pairs found!")
        return None
    
    logger.info(f"\nTop 10 pairs by score:")
    for p in pairs[:10]:
        logger.info(f"  {p['s1']}/{p['s2']}: HL={p['half_life']:.0f}, "
                   f"Hurst={p['hurst']:.2f}, score={p['score']:.3f}")
    
    # Split data into train/val/test
    logger.info("\nStep 3: Splitting data...")
    n = len(prices)
    train_end = int(n * CONFIG['train_split'])
    val_end = int(n * (CONFIG['train_split'] + CONFIG['val_split']))
    
    train_prices = prices.iloc[:train_end]
    val_prices = prices.iloc[train_end:val_end]
    test_prices = prices.iloc[val_end:]
    
    logger.info(f"  Train: {len(train_prices)} days ({prices.index[0]} to {prices.index[train_end-1]})")
    logger.info(f"  Val:   {len(val_prices)} days ({prices.index[train_end]} to {prices.index[val_end-1]})")
    logger.info(f"  Test:  {len(test_prices)} days ({prices.index[val_end]} to {prices.index[-1]})")
    
    # Grid search on training data
    logger.info("\nStep 4: Running grid search on training data...")
    train_results = grid_search(train_prices, pairs, CONFIG)
    
    logger.info("\nTop 10 parameter combinations (training):")
    for i, r in enumerate(train_results[:10]):
        logger.info(f"  {i+1}. Sharpe={r['sharpe']:.2f} | entry={r['entry_z']}, "
                   f"exit={r['exit_z']}, lookback={r['lookback']}, pairs={r['max_pairs']}")
    
    # Validate top 5 on validation set
    logger.info("\nStep 5: Validating top 5 on validation data...")
    val_results = []
    
    for r in train_results[:5]:
        val_result = backtest_params(
            val_prices, pairs, 
            r['entry_z'], r['exit_z'], r['lookback'], r['max_pairs']
        )
        val_result.update({
            'entry_z': r['entry_z'],
            'exit_z': r['exit_z'],
            'lookback': r['lookback'],
            'max_pairs': r['max_pairs'],
            'train_sharpe': r['sharpe']
        })
        val_results.append(val_result)
        
        logger.info(f"  Train Sharpe={r['sharpe']:.2f} -> Val Sharpe={val_result['sharpe']:.2f} "
                   f"(entry={r['entry_z']}, exit={r['exit_z']})")
    
    # Select best (highest validation Sharpe)
    best = max(val_results, key=lambda x: x['sharpe'])
    
    logger.info(f"\nBest parameters selected:")
    logger.info(f"  entry_z: {best['entry_z']}")
    logger.info(f"  exit_z: {best['exit_z']}")
    logger.info(f"  lookback: {best['lookback']}")
    logger.info(f"  max_pairs: {best['max_pairs']}")
    
    # Final test on held-out test set
    logger.info("\nStep 6: Running final test on held-out data...")
    test_result = backtest_params(
        test_prices, pairs,
        best['entry_z'], best['exit_z'], best['lookback'], best['max_pairs']
    )
    
    # Results summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Train Sharpe:  {best['train_sharpe']:.2f}")
    logger.info(f"Val Sharpe:    {best['sharpe']:.2f}")
    logger.info(f"Test Sharpe:   {test_result['sharpe']:.2f}")
    logger.info(f"Test Annual:   {test_result['annual']*100:.1f}%")
    logger.info(f"Test MaxDD:    {test_result['maxdd']*100:.1f}%")
    logger.info(f"Test Trades:   {test_result['trades']}")
    logger.info(f"Test Win Rate: {test_result['win_rate']*100:.1f}%")
    
    # Check for overfitting
    overfit_ratio = best['train_sharpe'] / best['sharpe'] if best['sharpe'] > 0 else float('inf')
    if overfit_ratio > 1.5:
        logger.warning(f"⚠️ Possible overfitting detected! Train/Val ratio: {overfit_ratio:.2f}")
    else:
        logger.info(f"✅ No significant overfitting. Train/Val ratio: {overfit_ratio:.2f}")
    
    # Save results
    logger.info("\nStep 7: Saving results...")
    
    output = {
        'best_params': {
            'entry_z': best['entry_z'],
            'exit_z': best['exit_z'],
            'lookback': best['lookback'],
            'max_pairs': best['max_pairs']
        },
        'performance': {
            'train_sharpe': best['train_sharpe'],
            'val_sharpe': best['sharpe'],
            'test_sharpe': test_result['sharpe'],
            'test_annual': test_result['annual'],
            'test_maxdd': test_result['maxdd'],
            'test_trades': test_result['trades'],
            'test_win_rate': test_result['win_rate']
        },
        'pairs': pairs[:20],  # Top 20 pairs
        'all_train_results': train_results[:50],  # Top 50 combinations
        'config': CONFIG,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_stocks': len(prices.columns),
            'n_days': len(prices),
            'date_range': f"{prices.index[0]} to {prices.index[-1]}",
            'training_time_minutes': (datetime.now() - start_time).total_seconds() / 60
        }
    }
    
    # Save locally
    results_file = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved: {results_file}")
    
    # Upload to GCS
    upload_to_gcs(
        CONFIG['bucket_name'], 
        results_file,
        f"models/{results_file}"
    )
    upload_to_gcs(
        CONFIG['bucket_name'], 
        'training.log',
        f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration:.1f} minutes")
    logger.info(f"Results saved to: {results_file}")
    logger.info("\n✅ Next steps:")
    logger.info("  1. Review results: cat " + results_file)
    logger.info("  2. Download locally: gsutil cp gs://cift-stat-arb-data/models/" + results_file + " .")
    logger.info("  3. Update production_stat_arb.py with best parameters")
    logger.info("  4. Run paper trading for 2-4 weeks")
    
    return output


if __name__ == "__main__":
    main()
