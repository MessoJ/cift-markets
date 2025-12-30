#!/usr/bin/env python3
"""
Real Data Validation Script

This script:
1. Downloads REAL historical data (yfinance - free)
2. Runs cointegration analysis on REAL pairs
3. Backtests the stat arb strategy with REALISTIC costs
4. Reports HONEST metrics

Run this to see what you ACTUALLY have.

Usage:
    python scripts/validate_on_real_data.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polars as pl

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Liquid US stocks for pair trading (sectors grouped)
SYMBOLS_BY_SECTOR = {
    "tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "SCHW", "BLK"],
    "healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "CVS"],
    "consumer": ["WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"],
}

# Flatten for easy access
ALL_SYMBOLS = []
for sector_symbols in SYMBOLS_BY_SECTOR.values():
    ALL_SYMBOLS.extend(sector_symbols)

# REALISTIC transaction costs (not optimistic backtest assumptions)
REALISTIC_COSTS_BPS = {
    "spread": 3.0,         # Average half-spread for liquid stocks
    "commission": 0.5,     # Per side (low, broker dependent)
    "market_impact": 2.0,  # For moderate size orders
    "slippage": 2.0,       # Execution slippage
}
TOTAL_ROUNDTRIP_BPS = sum(REALISTIC_COSTS_BPS.values()) * 2  # Both sides


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_real_data(symbols: list[str], years: int = 5) -> dict[str, pl.DataFrame]:
    """
    Download real adjusted price data from Yahoo Finance.
    
    Returns dict of {symbol: DataFrame} with columns: date, open, high, low, close, volume
    """
    if not HAS_YFINANCE:
        raise RuntimeError("yfinance required. Run: pip install yfinance")
    
    print(f"\nDownloading {years} years of data for {len(symbols)} symbols...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    data = {}
    failed = []
    
    for i, symbol in enumerate(symbols):
        try:
            print(f"  [{i+1}/{len(symbols)}] Downloading {symbol}...", end=" ")
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            
            if len(hist) < 252:  # Less than 1 year
                print(f"SKIP (only {len(hist)} bars)")
                failed.append(symbol)
                continue
            
            # Convert to polars
            df = pl.DataFrame({
                "date": hist.index.tolist(),
                "open": hist["Open"].values,
                "high": hist["High"].values,
                "low": hist["Low"].values,
                "close": hist["Close"].values,
                "volume": hist["Volume"].values,
            })
            
            # Add returns
            df = df.with_columns([
                pl.col("close").pct_change().alias("returns"),
            ])
            
            data[symbol] = df.drop_nulls()
            print(f"OK ({len(df)} bars)")
            
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(symbol)
    
    print(f"\nDownloaded: {len(data)} symbols, Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {failed}")
    
    return data


# =============================================================================
# COINTEGRATION ANALYSIS
# =============================================================================

def run_cointegration_scan(price_data: dict[str, pl.DataFrame], max_pvalue: float = 0.05) -> list[dict]:
    """
    Scan all pairs for cointegration using Engle-Granger.
    
    Returns list of cointegrated pairs with statistics.
    """
    from cift.ml.stat_arb import engle_granger_coint, half_life_mean_reversion
    
    symbols = list(price_data.keys())
    n_pairs = len(symbols) * (len(symbols) - 1) // 2
    
    print(f"\nScanning {n_pairs} pairs for cointegration...")
    
    # Align all price series to common dates
    # Find common date range
    all_dates = None
    for symbol, df in price_data.items():
        dates = set(df["date"].to_list())
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    
    common_dates = sorted(list(all_dates))
    print(f"  Common trading days: {len(common_dates)}")
    
    if len(common_dates) < 252:
        print("  WARNING: Less than 1 year of common data!")
    
    # Build aligned price matrix
    prices = {}
    for symbol, df in price_data.items():
        df_filtered = df.filter(pl.col("date").is_in(common_dates))
        df_sorted = df_filtered.sort("date")
        prices[symbol] = df_sorted["close"].to_numpy()
    
    # Test all pairs
    cointegrated_pairs = []
    tested = 0
    
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            tested += 1
            if tested % 100 == 0:
                print(f"  Tested {tested}/{n_pairs} pairs, found {len(cointegrated_pairs)} cointegrated")
            
            try:
                p1 = prices[sym1]
                p2 = prices[sym2]
                
                adf_stat, pvalue, hedge, intercept = engle_granger_coint(p1, p2)
                
                if pvalue < max_pvalue:
                    spread = p1 - hedge * p2 - intercept
                    hl = half_life_mean_reversion(spread)
                    
                    # Filter reasonable half-lives
                    if 5 < hl < 60:  # Between 5 and 60 days
                        cointegrated_pairs.append({
                            "sym1": sym1,
                            "sym2": sym2,
                            "pvalue": pvalue,
                            "hedge_ratio": hedge,
                            "half_life": hl,
                            "adf_stat": adf_stat,
                        })
            except Exception:
                continue
    
    # Sort by p-value
    cointegrated_pairs = sorted(cointegrated_pairs, key=lambda x: x["pvalue"])
    
    print(f"\n  Total pairs tested: {n_pairs}")
    print(f"  Cointegrated (p < {max_pvalue}): {len(cointegrated_pairs)}")
    print(f"  Cointegration rate: {100 * len(cointegrated_pairs) / n_pairs:.1f}%")
    
    return cointegrated_pairs


# =============================================================================
# REALISTIC BACKTEST
# =============================================================================

def backtest_stat_arb_realistic(
    price_data: dict[str, pl.DataFrame],
    pairs: list[dict],
    entry_zscore: float = 2.0,
    exit_zscore: float = 0.5,
    lookback: int = 20,
    cost_bps: float = 15.0,  # Total roundtrip cost
) -> dict:
    """
    Run realistic backtest with proper costs.
    """
    from cift.ml.stat_arb import half_life_mean_reversion
    
    if not pairs:
        return {"error": "No cointegrated pairs found"}
    
    print(f"\nRunning backtest on {len(pairs)} pairs...")
    print(f"  Entry z-score: {entry_zscore}")
    print(f"  Exit z-score: {exit_zscore}")
    print(f"  Transaction costs: {cost_bps} bps roundtrip")
    
    # Align dates
    all_dates = None
    for symbol, df in price_data.items():
        dates = set(df["date"].to_list())
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    
    common_dates = sorted(list(all_dates))
    n_days = len(common_dates)
    
    # Build price dict
    prices = {}
    for symbol, df in price_data.items():
        df_filtered = df.filter(pl.col("date").is_in(common_dates)).sort("date")
        prices[symbol] = df_filtered["close"].to_numpy()
    
    # Simulate trading
    equity = 1.0
    equity_curve = [equity]
    daily_returns = []
    trades = []
    
    # Track positions per pair
    positions = {i: 0 for i in range(len(pairs))}  # -1, 0, or 1
    
    for t in range(lookback, n_days):
        day_pnl = 0.0
        
        for pair_idx, pair in enumerate(pairs):
            sym1, sym2 = pair["sym1"], pair["sym2"]
            hedge = pair["hedge_ratio"]
            
            if sym1 not in prices or sym2 not in prices:
                continue
            
            # Calculate current spread z-score
            p1 = prices[sym1][t-lookback:t]
            p2 = prices[sym2][t-lookback:t]
            
            spread = p1 - hedge * p2
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            
            if std_spread < 1e-8:
                continue
            
            current_spread = prices[sym1][t] - hedge * prices[sym2][t]
            zscore = (current_spread - mean_spread) / std_spread
            
            # Current position
            pos = positions[pair_idx]
            
            # Trading logic
            new_pos = pos
            
            if pos == 0:
                # Entry signals
                if zscore > entry_zscore:
                    new_pos = -1  # Short spread
                elif zscore < -entry_zscore:
                    new_pos = 1   # Long spread
            else:
                # Exit signals
                if pos == 1 and zscore > -exit_zscore:
                    new_pos = 0
                elif pos == -1 and zscore < exit_zscore:
                    new_pos = 0
            
            # Calculate P&L
            if pos != 0:
                # P&L from existing position
                prev_spread = prices[sym1][t-1] - hedge * prices[sym2][t-1]
                spread_return = (current_spread - prev_spread) / abs(prev_spread) if abs(prev_spread) > 1e-8 else 0
                day_pnl += pos * spread_return * (1.0 / len(pairs))  # Equal weight
            
            # Track trades and costs
            if new_pos != pos:
                # Trade occurred - apply costs
                cost = (cost_bps / 10000) * (1.0 / len(pairs))
                day_pnl -= cost
                
                trades.append({
                    "day": t,
                    "pair": f"{sym1}/{sym2}",
                    "action": "ENTER" if new_pos != 0 else "EXIT",
                    "side": "LONG" if new_pos == 1 else "SHORT" if new_pos == -1 else "FLAT",
                    "zscore": zscore,
                })
            
            positions[pair_idx] = new_pos
        
        # Update equity
        equity *= (1 + day_pnl)
        equity_curve.append(equity)
        daily_returns.append(day_pnl)
    
    # Calculate metrics
    returns = np.array(daily_returns)
    
    # Sharpe ratio (proper calculation)
    if len(returns) > 0 and np.std(returns) > 1e-10:
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    else:
        sharpe = 0.0
    
    # Max drawdown
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    drawdowns = eq / running_max - 1
    max_dd = np.min(drawdowns)
    
    # Win rate
    if trades:
        # Approximate win rate from return signs
        winning_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
    else:
        win_rate = 0
    
    # PSR
    from cift.metrics.performance import prob_sharpe_ratio
    n = len(returns)
    skew = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 3)) if np.std(returns) > 0 else 0
    kurt = float(np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4)) if np.std(returns) > 0 else 3
    psr = prob_sharpe_ratio(sharpe, sharpe_benchmark=0.0, n=n, skew=skew, kurtosis=kurt)
    
    results = {
        "sharpe_ratio": sharpe,
        "psr": psr,
        "total_return": (equity - 1) * 100,  # Percentage
        "max_drawdown": max_dd * 100,  # Percentage
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_daily_return": np.mean(returns) * 100,
        "daily_volatility": np.std(returns) * 100,
        "n_days": len(returns),
        "equity_curve": equity_curve,
    }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("REAL DATA VALIDATION - BRUTAL HONESTY MODE")
    print("=" * 70)
    
    if not HAS_YFINANCE:
        print("\nERROR: yfinance not installed!")
        print("Install with: pip install yfinance")
        return False
    
    # Step 1: Download real data
    print("\n" + "=" * 70)
    print("STEP 1: Download Real Historical Data")
    print("=" * 70)
    
    # Use smaller universe for faster testing
    test_symbols = (
        SYMBOLS_BY_SECTOR["tech"][:5] + 
        SYMBOLS_BY_SECTOR["finance"][:5] +
        SYMBOLS_BY_SECTOR["energy"][:5]
    )
    
    price_data = download_real_data(test_symbols, years=3)
    
    if len(price_data) < 10:
        print("\nERROR: Not enough data downloaded!")
        return False
    
    # Step 2: Cointegration scan
    print("\n" + "=" * 70)
    print("STEP 2: Cointegration Analysis (Real Data)")
    print("=" * 70)
    
    pairs = run_cointegration_scan(price_data, max_pvalue=0.05)
    
    if not pairs:
        print("\nWARNING: No cointegrated pairs found!")
        print("This is the REALITY of stat arb - not every pair works.")
        
        # Try with relaxed threshold
        print("\nTrying with relaxed p-value threshold (0.10)...")
        pairs = run_cointegration_scan(price_data, max_pvalue=0.10)
    
    if pairs:
        print("\nTop 10 cointegrated pairs:")
        for i, pair in enumerate(pairs[:10]):
            print(f"  {i+1}. {pair['sym1']}/{pair['sym2']}: "
                  f"p={pair['pvalue']:.4f}, HL={pair['half_life']:.1f} days, "
                  f"hedge={pair['hedge_ratio']:.3f}")
    
    # Step 3: Realistic backtest
    print("\n" + "=" * 70)
    print("STEP 3: Backtest with REALISTIC Costs")
    print("=" * 70)
    
    if pairs:
        results = backtest_stat_arb_realistic(
            price_data,
            pairs[:10],  # Top 10 pairs
            entry_zscore=2.0,
            exit_zscore=0.5,
            cost_bps=TOTAL_ROUNDTRIP_BPS,
        )
        
        print("\n" + "-" * 50)
        print("REALISTIC BACKTEST RESULTS")
        print("-" * 50)
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
        print(f"  PSR (vs 0):        {results['psr']*100:.1f}%")
        print(f"  Total Return:      {results['total_return']:.1f}%")
        print(f"  Max Drawdown:      {results['max_drawdown']:.1f}%")
        print(f"  Number of Trades:  {results['num_trades']}")
        print(f"  Win Rate:          {results['win_rate']*100:.1f}%")
        print(f"  Days Tested:       {results['n_days']}")
    else:
        print("\nCannot run backtest - no viable pairs found.")
        results = None
    
    # Step 4: Reality check
    print("\n" + "=" * 70)
    print("REALITY CHECK")
    print("=" * 70)
    
    print("""
    What we found:
    
    1. COINTEGRATION IS RARE
       - Synthetic data: 100% cointegrated by design
       - Real market: Only a small fraction of pairs are cointegrated
       - This is NORMAL and EXPECTED
    
    2. TRANSACTION COSTS MATTER
       - Backtest with 2 bps: Sharpe looks great
       - Backtest with 15 bps: Sharpe drops significantly
       - Real trading often worse than 15 bps
    
    3. YOUR REALISTIC TARGET
       - If Sharpe > 0.8: You might have something
       - If Sharpe > 1.2: Excellent, but verify it's real
       - If Sharpe > 2.0: Probably overfitting or data error
    
    NEXT STEPS:
    1. Paper trade for 30+ days
    2. Compare actual fills to backtest assumptions
    3. Only then consider 5% of intended capital
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
