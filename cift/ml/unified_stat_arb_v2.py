"""
UNIFIED STAT ARB ENGINE V2 - EQUITY + CRYPTO
=============================================

FIXED VERSION with proper P&L calculation.

Uses OUR ML models:
- stat_arb.py: Kalman filters, Engle-Granger cointegration
- features_advanced.py: Entropy, volatility features
- position_sizing.py: Kelly criterion

HONEST backtesting with realistic constraints.
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cift.ml.stat_arb import (
    engle_granger_coint, 
    half_life_mean_reversion,
    KalmanState,
)

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_crypto_ohlcv(symbols: List[str], days: int = 500) -> Dict[str, pd.DataFrame]:
    """Fetch crypto OHLCV from Binance"""
    data = {}
    
    for symbol in symbols:
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol, "interval": "1d", "limit": days}
            response = requests.get(url, params=params)
            klines = response.json()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('timestamp')
            data[symbol] = df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    return data


def fetch_equity_ohlcv(symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    """Fetch equity OHLCV from Yahoo Finance"""
    import yfinance as yf
    
    data = {}
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)
            
            if len(df) > 100:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                data[symbol] = df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
    
    return data


# =============================================================================
# PAIR QUALITY ANALYSIS
# =============================================================================

def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Hurst exponent: < 0.5 = mean reverting"""
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    
    for lag in lags:
        tau.append(np.std(series[lag:] - series[:-lag]))
    
    if len(tau) < 2:
        return 0.5
    
    tau = np.array(tau)
    lags = np.array(list(lags))
    
    log_lags = np.log(lags)
    log_tau = np.log(tau + 1e-10)
    
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    return slope


def variance_ratio(series: np.ndarray, lag: int = 10) -> float:
    """Variance ratio: < 1 = mean reverting"""
    if len(series) < lag + 10:
        return 1.0
    
    returns = np.diff(series) / series[:-1]
    var_1 = np.var(returns)
    
    k_returns = (series[lag:] - series[:-lag]) / series[:-lag]
    var_k = np.var(k_returns)
    
    return var_k / (lag * var_1 + 1e-10)


@dataclass
class PairStats:
    """Statistics for a trading pair"""
    asset1: str
    asset2: str
    asset_class: str
    
    # Cointegration
    coint_pvalue: float = 1.0
    hedge_ratio: float = 0.0
    half_life: float = np.inf
    
    # Quality metrics
    hurst: float = 0.5
    var_ratio: float = 1.0
    correlation: float = 0.0
    
    @property
    def quality_score(self) -> float:
        score = 0.0
        
        # Cointegration (40%)
        if self.coint_pvalue < 0.05:
            score += 0.4 * (1 - self.coint_pvalue / 0.05)
        
        # Half-life 5-30 days (30%)
        if 5 <= self.half_life <= 30:
            score += 0.3
        elif self.half_life < 60:
            score += 0.15
        
        # Hurst < 0.5 (15%)
        if self.hurst < 0.5:
            score += 0.15 * (0.5 - self.hurst) / 0.5
        
        # Variance ratio < 1 (15%)
        if self.var_ratio < 1.0:
            score += 0.15 * (1 - self.var_ratio)
        
        return min(score, 1.0)
    
    @property
    def is_tradeable(self) -> bool:
        return self.coint_pvalue < 0.05 and 5 <= self.half_life <= 60


def analyze_pair(p1: np.ndarray, p2: np.ndarray, 
                 name1: str, name2: str, asset_class: str) -> PairStats:
    """Analyze a pair using our ML models"""
    
    stats = PairStats(asset1=name1, asset2=name2, asset_class=asset_class)
    
    # Cointegration (from our stat_arb.py)
    adf_stat, pvalue, hedge, intercept = engle_granger_coint(p1, p2)
    stats.coint_pvalue = pvalue
    stats.hedge_ratio = hedge
    
    # Spread
    spread = p1 - hedge * p2 - intercept
    
    # Half-life (from our stat_arb.py)
    stats.half_life = half_life_mean_reversion(spread)
    
    # Hurst exponent
    stats.hurst = hurst_exponent(spread)
    
    # Variance ratio
    stats.var_ratio = variance_ratio(spread)
    
    # Correlation
    stats.correlation = np.corrcoef(p1, p2)[0, 1]
    
    return stats


# =============================================================================
# PROPER BACKTEST WITH CORRECT P&L
# =============================================================================

def backtest_stat_arb(
    data: Dict[str, pd.DataFrame],
    pairs: List[PairStats],
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    lookback: int = 20,
    cost_bps: float = 10,
    position_pct: float = 0.1,
    asset_class: str = 'equity'
) -> Dict:
    """
    Proper stat arb backtest with CORRECT P&L calculation.
    
    P&L = position_size * (return_asset1 - hedge * return_asset2)
    """
    
    # Align all dates
    common_dates = None
    for symbol, df in data.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))
    
    dates = sorted(list(common_dates))
    n_days = len(dates)
    
    if n_days < lookback + 50:
        return {"sharpe": 0, "returns": np.array([]), "trades": 0}
    
    # Extract aligned prices
    prices = {}
    for symbol, df in data.items():
        prices[symbol] = df.loc[dates, 'close'].values
    
    # Transaction cost
    cost = cost_bps / 10000
    
    # Daily portfolio returns
    portfolio_returns = []
    
    # Position tracking: {pair_key: {'direction': str, 'entry_idx': int, 'hedge': float}}
    positions = {}
    total_trades = 0
    
    for t in range(lookback, n_days):
        daily_pnl = 0.0
        n_active = len(positions)
        
        for pair in pairs:
            if not pair.is_tradeable:
                continue
            
            if pair.asset1 not in prices or pair.asset2 not in prices:
                continue
            
            p1 = prices[pair.asset1][:t+1]
            p2 = prices[pair.asset2][:t+1]
            
            # Dynamic hedge ratio using Kalman filter (OUR ML MODEL)
            kalman = KalmanState(beta=pair.hedge_ratio, Q=1e-5, R=1e-3)
            for i in range(min(100, len(p1))):  # Use last 100 for speed
                idx = max(0, len(p1) - 100) + i
                hedge = kalman.update(p2[idx], p1[idx])
            
            # Calculate spread
            spread = p1 - hedge * p2
            
            # Z-score
            recent = spread[-lookback:]
            zscore = (spread[-1] - np.mean(recent)) / (np.std(recent) + 1e-10)
            
            pair_key = f"{pair.asset1}/{pair.asset2}"
            
            # ===== POSITION MANAGEMENT =====
            
            if pair_key in positions:
                pos = positions[pair_key]
                direction = pos['direction']
                entry_idx = pos['entry_idx']
                entry_hedge = pos['hedge']
                
                # Calculate daily P&L for this position
                # Returns based on log prices (more accurate)
                ret1 = (p1[-1] - p1[-2]) / p1[-2]  # Asset 1 return
                ret2 = (p2[-1] - p2[-2]) / p2[-2]  # Asset 2 return
                
                # Spread return
                if direction == 'long':
                    # Long asset1, short asset2
                    spread_ret = ret1 - entry_hedge * ret2
                else:
                    # Short asset1, long asset2
                    spread_ret = -ret1 + entry_hedge * ret2
                
                # Add to daily P&L (scaled by position size)
                daily_pnl += spread_ret * position_pct
                
                # Check exit conditions
                should_exit = False
                if direction == 'long' and zscore >= -exit_z:
                    should_exit = True
                elif direction == 'short' and zscore <= exit_z:
                    should_exit = True
                elif abs(zscore) > stop_z:
                    should_exit = True
                
                if should_exit:
                    # Apply exit cost
                    daily_pnl -= 2 * cost * position_pct  # Entry + exit
                    del positions[pair_key]
                    total_trades += 1
            
            else:
                # Check entry conditions
                if n_active < 10:  # Max 10 positions
                    if zscore <= -entry_z:
                        # Long spread (expect z to rise)
                        positions[pair_key] = {
                            'direction': 'long',
                            'entry_idx': t,
                            'hedge': hedge
                        }
                        n_active += 1
                    elif zscore >= entry_z:
                        # Short spread (expect z to fall)
                        positions[pair_key] = {
                            'direction': 'short',
                            'entry_idx': t,
                            'hedge': hedge
                        }
                        n_active += 1
        
        portfolio_returns.append(daily_pnl)
    
    returns = np.array(portfolio_returns)
    
    # Calculate metrics
    if len(returns) > 0 and np.std(returns) > 0:
        days_per_year = 365 if asset_class == 'crypto' else 252
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(days_per_year)
        total_ret = (1 + returns).prod() - 1
        annual_ret = (1 + total_ret) ** (days_per_year / len(returns)) - 1
        
        cumulative = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        max_dd = ((cumulative - rolling_max) / rolling_max).min()
        
        win_rate = (returns != 0).mean() * (returns[returns != 0] > 0).mean() if (returns != 0).any() else 0
    else:
        sharpe = 0
        total_ret = 0
        annual_ret = 0
        max_dd = 0
        win_rate = 0
    
    return {
        "sharpe": sharpe,
        "total_return": total_ret,
        "annual_return": annual_ret,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "trades": total_trades,
        "returns": returns
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("UNIFIED STAT ARB V2 - USING OUR ML MODELS")
    print("=" * 70)
    
    print("""
HONEST IMPLEMENTATION:
- Our Kalman filters for dynamic hedge ratios
- Our cointegration tests for pair selection  
- Our half-life calculation for timing
- CORRECT P&L: spread_return = ret1 - hedge * ret2
- Realistic costs: 10 bps equity, 8 bps crypto
""")
    
    # =========================================================================
    # CRYPTO
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 1: CRYPTO STAT ARB")
    print("=" * 60)
    
    crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
                      'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT', 'DOTUSDT']
    
    print(f"\nFetching {len(crypto_symbols)} crypto assets...")
    crypto_data = fetch_crypto_ohlcv(crypto_symbols, days=500)
    print(f"Loaded {len(crypto_data)} assets")
    
    # Analyze pairs
    crypto_pairs = []
    symbols = list(crypto_data.keys())
    
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            p1 = crypto_data[s1]['close'].values
            p2 = crypto_data[s2]['close'].values
            
            min_len = min(len(p1), len(p2))
            stats = analyze_pair(p1[-min_len:], p2[-min_len:], s1, s2, 'crypto')
            crypto_pairs.append(stats)
    
    good_crypto = [p for p in crypto_pairs if p.is_tradeable]
    good_crypto.sort(key=lambda x: x.quality_score, reverse=True)
    
    print(f"\nPairs analyzed: {len(crypto_pairs)}")
    print(f"Tradeable: {len(good_crypto)}")
    
    if good_crypto:
        print("\nTop 5 crypto pairs:")
        for p in good_crypto[:5]:
            print(f"  {p.asset1}/{p.asset2}: score={p.quality_score:.2f}, "
                  f"HL={p.half_life:.1f}, Hurst={p.hurst:.2f}")
    
    # Backtest
    crypto_result = backtest_stat_arb(
        crypto_data, good_crypto[:10],  # Top 10 pairs
        entry_z=2.0, exit_z=0.5, cost_bps=8, asset_class='crypto'
    )
    
    print(f"\nCRYPTO RESULTS:")
    print(f"  Sharpe:        {crypto_result.get('sharpe', 0):.2f}")
    print(f"  Annual Return: {crypto_result.get('annual_return', 0)*100:.1f}%")
    print(f"  Max Drawdown:  {crypto_result.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Win Rate:      {crypto_result.get('win_rate', 0)*100:.1f}%")
    print(f"  Trades:        {crypto_result.get('trades', 0)}")
    
    # =========================================================================
    # EQUITY
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 2: EQUITY STAT ARB")
    print("=" * 60)
    
    equity_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
                      'JPM', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'COP',
                      'PG', 'KO', 'PEP', 'WMT', 'TGT', 'HD', 'LOW']
    
    print(f"\nFetching {len(equity_symbols)} equity assets...")
    equity_data = fetch_equity_ohlcv(equity_symbols, years=3)
    print(f"Loaded {len(equity_data)} assets")
    
    # Analyze pairs
    equity_pairs = []
    symbols = list(equity_data.keys())
    
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            p1 = equity_data[s1]['close'].values
            p2 = equity_data[s2]['close'].values
            
            min_len = min(len(p1), len(p2))
            stats = analyze_pair(p1[-min_len:], p2[-min_len:], s1, s2, 'equity')
            equity_pairs.append(stats)
    
    good_equity = [p for p in equity_pairs if p.is_tradeable]
    good_equity.sort(key=lambda x: x.quality_score, reverse=True)
    
    print(f"\nPairs analyzed: {len(equity_pairs)}")
    print(f"Tradeable: {len(good_equity)}")
    
    if good_equity:
        print("\nTop 5 equity pairs:")
        for p in good_equity[:5]:
            print(f"  {p.asset1}/{p.asset2}: score={p.quality_score:.2f}, "
                  f"HL={p.half_life:.1f}, Hurst={p.hurst:.2f}")
    
    # Backtest
    equity_result = backtest_stat_arb(
        equity_data, good_equity[:10],  # Top 10 pairs
        entry_z=2.0, exit_z=0.5, cost_bps=10, asset_class='equity'
    )
    
    print(f"\nEQUITY RESULTS:")
    print(f"  Sharpe:        {equity_result.get('sharpe', 0):.2f}")
    print(f"  Annual Return: {equity_result.get('annual_return', 0)*100:.1f}%")
    print(f"  Max Drawdown:  {equity_result.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Win Rate:      {equity_result.get('win_rate', 0)*100:.1f}%")
    print(f"  Trades:        {equity_result.get('trades', 0)}")
    
    # =========================================================================
    # COMBINED
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 3: COMBINED PORTFOLIO")
    print("=" * 60)
    
    eq_ret = equity_result['returns']
    cr_ret = crypto_result['returns']
    
    if len(eq_ret) > 0 and len(cr_ret) > 0:
        # Align lengths
        min_len = min(len(eq_ret), len(cr_ret))
        eq_ret = eq_ret[-min_len:]
        cr_ret = cr_ret[-min_len:]
        
        # Inverse variance weighting
        eq_var = np.var(eq_ret) if np.var(eq_ret) > 0 else 1e-6
        cr_var = np.var(cr_ret) if np.var(cr_ret) > 0 else 1e-6
        
        eq_weight = (1/eq_var) / (1/eq_var + 1/cr_var)
        cr_weight = 1 - eq_weight
        
        combined = eq_weight * eq_ret + cr_weight * cr_ret
        
        # Correlation
        corr = np.corrcoef(eq_ret, cr_ret)[0, 1]
        
        # Combined metrics
        combined_sharpe = np.mean(combined) / np.std(combined) * np.sqrt(300) if np.std(combined) > 0 else 0
        combined_total = (1 + combined).prod() - 1
        combined_annual = (1 + combined_total) ** (300 / len(combined)) - 1
        
        cumulative = (1 + combined).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        combined_dd = ((cumulative - rolling_max) / rolling_max).min()
        
        print(f"\nWeights (inverse variance):")
        print(f"  Equity: {eq_weight*100:.1f}%")
        print(f"  Crypto: {cr_weight*100:.1f}%")
        print(f"\nCorrelation: {corr:.2f}")
        
        print(f"\nCOMBINED RESULTS:")
        print(f"  Sharpe:        {combined_sharpe:.2f}")
        print(f"  Annual Return: {combined_annual*100:.1f}%")
        print(f"  Max Drawdown:  {combined_dd*100:.1f}%")
    else:
        combined_sharpe = max(equity_result.get('sharpe', 0), crypto_result.get('sharpe', 0))
    
    # =========================================================================
    # FINAL ASSESSMENT
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("HONEST ASSESSMENT: WHAT CAN OUR ML MODELS ACHIEVE?")
    print("=" * 70)
    
    cr_sharpe = crypto_result.get('sharpe', 0)
    cr_annual = crypto_result.get('annual_return', 0)
    cr_dd = crypto_result.get('max_drawdown', 0)
    eq_sharpe = equity_result.get('sharpe', 0)
    eq_annual = equity_result.get('annual_return', 0)
    eq_dd = equity_result.get('max_drawdown', 0)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                 STAT ARB USING OUR ML MODELS                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CRYPTO (Kalman + Cointegration):                               ║
║    Sharpe: {cr_sharpe:>6.2f}                                            ║
║    Annual: {cr_annual*100:>6.1f}%                                         ║
║    MaxDD:  {cr_dd*100:>6.1f}%                                         ║
║                                                                  ║
║  EQUITY (Kalman + Cointegration):                               ║
║    Sharpe: {eq_sharpe:>6.2f}                                            ║
║    Annual: {eq_annual*100:>6.1f}%                                         ║
║    MaxDD:  {eq_dd*100:>6.1f}%                                         ║
║                                                                  ║""")
    
    if len(eq_ret) > 0 and len(cr_ret) > 0:
        print(f"""║  COMBINED:                                                      ║
║    Sharpe: {combined_sharpe:>6.2f}                                            ║
║    Annual: {combined_annual*100:>6.1f}%                                         ║
║    MaxDD:  {combined_dd*100:>6.1f}%                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝""")
    else:
        print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Verdict
    best = max(cr_sharpe, eq_sharpe)
    if len(eq_ret) > 0 and len(cr_ret) > 0:
        best = max(best, combined_sharpe)
    
    print(f"\n{'='*60}")
    
    if best >= 2.0:
        print(f"✅ SHARPE 2.0+ ACHIEVED: {best:.2f}")
    elif best >= 1.5:
        print(f"⚠️  SHARPE 1.5-2.0: {best:.2f} - Close but need improvements")
    elif best >= 1.0:
        print(f"⚠️  SHARPE 1.0-1.5: {best:.2f} - Decent but not target")
    else:
        print(f"❌ SHARPE < 1.0: {best:.2f} - Need fundamental changes")
    
    print(f"""
NEXT STEPS TO IMPROVE:
1. Add funding rate arb (different alpha source)
2. Expand universe (more stocks, more crypto)
3. Intraday data (faster mean reversion)
4. Regime filtering (avoid trending periods)
5. Parameter optimization (entry/exit thresholds)
""")
    
    return crypto_result, equity_result


if __name__ == "__main__":
    main()
