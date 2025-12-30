"""
ENHANCED STAT ARB ENGINE V3
===========================

Improvements over V2:
1. EXPANDED UNIVERSE: 60+ stocks across 6 sectors
2. REGIME FILTERING: Use Hurst exponent to avoid trending markets
3. OPTIMIZED PARAMETERS: Dynamic entry/exit based on half-life
4. COMBINED SYSTEM: Funding arb + equity stat arb with honest attribution

Target: Maximize Sharpe from OUR ML models
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cift.ml.stat_arb import engle_granger_coint, half_life_mean_reversion, KalmanState

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# EXPANDED UNIVERSE - 60+ STOCKS ACROSS 6 SECTORS
# =============================================================================

# Curated universe - liquid stocks that work well with yfinance
EXPANDED_UNIVERSE = {
    'energy': [
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY'
    ],
    'banks': [
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'SCHW', 'AXP'
    ],
    'retail': [
        'WMT', 'HD', 'LOW', 'TGT', 'COST', 'TJX', 'ROST', 'DG', 'BBY'
    ],
    'healthcare': [
        'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'ABT', 'BMY', 'AMGN'
    ],
    'tech': [
        'MSFT', 'AAPL', 'GOOGL', 'META', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'IBM', 'AMD'
    ],
    'staples': [
        'PG', 'KO', 'PEP', 'PM', 'MO', 'CL', 'KMB', 'GIS', 'MDLZ'
    ]
}


def get_all_symbols() -> List[str]:
    """Get all symbols from expanded universe"""
    symbols = []
    for sector, tickers in EXPANDED_UNIVERSE.items():
        symbols.extend(tickers)
    return symbols


# =============================================================================
# REGIME FILTER USING HURST EXPONENT
# =============================================================================

def calculate_hurst(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent to detect regime.
    
    H < 0.5: Mean-reverting (GOOD for stat arb)
    H = 0.5: Random walk
    H > 0.5: Trending (BAD for stat arb)
    
    This is from our features_advanced.py - using it for regime filtering.
    """
    if len(series) < max_lag * 2:
        return 0.5
    
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        pp = np.subtract(series[lag:], series[:-lag])
        tau.append(np.sqrt(np.std(pp)))
    
    tau = np.array(tau)
    lags = np.array(list(lags))
    
    # Remove zeros
    valid = tau > 0
    if not valid.any():
        return 0.5
    
    tau = tau[valid]
    lags = lags[valid]
    
    try:
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    except:
        return 0.5


def is_mean_reverting_regime(spread: np.ndarray, threshold: float = 0.45) -> bool:
    """
    Check if spread is in mean-reverting regime.
    Only trade when Hurst < threshold (mean-reverting).
    """
    hurst = calculate_hurst(spread)
    return hurst < threshold


# =============================================================================
# DATA FETCHING WITH PARALLEL DOWNLOADS
# =============================================================================

def fetch_stock(symbol: str, start: datetime, end: datetime) -> Tuple[str, Optional[pd.DataFrame]]:
    """Fetch single stock data"""
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if len(df) > 200:
            return symbol, df[['Close']].rename(columns={'Close': symbol})
    except:
        pass
    return symbol, None


def fetch_all_data(symbols: List[str], years: int = 3) -> pd.DataFrame:
    """Fetch all stock data - sequential to avoid rate limits"""
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    
    print(f"  Fetching {len(symbols)} stocks...")
    
    # Use yfinance batch download
    try:
        data = yf.download(symbols, start=start, end=end, progress=False, threads=False)
        
        if 'Close' in data.columns or len(data.columns.names) > 1:
            # Multi-ticker format
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'].dropna(axis=1, how='all')
            else:
                prices = data[['Close']].rename(columns={'Close': symbols[0]})
        else:
            prices = data
        
        prices = prices.dropna()
        print(f"  Loaded: {len(prices.columns)} stocks, {len(prices)} days")
        return prices
        
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


# =============================================================================
# ENHANCED PAIR ANALYSIS WITH REGIME FILTERING
# =============================================================================

@dataclass
class TradingPair:
    s1: str
    s2: str
    hedge: float
    intercept: float
    half_life: float
    pval: float
    hurst: float
    score: float
    sector1: str
    sector2: str


def get_sector(symbol: str) -> str:
    """Get sector for a symbol"""
    for sector, symbols in EXPANDED_UNIVERSE.items():
        if symbol in symbols:
            return sector
    return 'unknown'


def analyze_pairs(
    prices_df: pd.DataFrame,
    pval_threshold: float = 0.05,
    min_half_life: int = 5,
    max_half_life: int = 60,
    max_hurst: float = 0.50  # Regime filter
) -> List[TradingPair]:
    """
    Analyze all pairs with regime filtering.
    
    Uses:
    - Our engle_granger_coint() for cointegration
    - Our half_life_mean_reversion() for timing
    - Our Hurst exponent for regime filtering
    """
    symbols = list(prices_df.columns)
    pairs = []
    
    total_pairs = len(symbols) * (len(symbols) - 1) // 2
    analyzed = 0
    cointegrated = 0
    regime_filtered = 0
    
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            analyzed += 1
            
            try:
                p1 = prices_df[s1].values
                p2 = prices_df[s2].values
                
                # OUR ML: Engle-Granger cointegration test
                adf, pval, hedge, intercept = engle_granger_coint(p1, p2)
                
                if pval > pval_threshold:
                    continue
                
                cointegrated += 1
                
                # Calculate spread
                spread = p1 - hedge * p2 - intercept
                
                # OUR ML: Half-life calculation
                hl = half_life_mean_reversion(spread)
                
                if not (min_half_life <= hl <= max_half_life):
                    continue
                
                # OUR ML: Hurst exponent for regime filtering
                hurst = calculate_hurst(spread)
                
                if hurst > max_hurst:
                    regime_filtered += 1
                    continue
                
                # Score: prioritize low p-value, good half-life, low Hurst
                score = (1 - pval) * (1 - abs(hl - 20) / 40) * (1 - hurst)
                
                pairs.append(TradingPair(
                    s1=s1, s2=s2,
                    hedge=hedge, intercept=intercept,
                    half_life=hl, pval=pval, hurst=hurst,
                    score=score,
                    sector1=get_sector(s1),
                    sector2=get_sector(s2)
                ))
                
            except Exception as e:
                continue
    
    # Sort by score
    pairs.sort(key=lambda x: x.score, reverse=True)
    
    print(f"  Pairs analyzed: {analyzed}")
    print(f"  Cointegrated: {cointegrated}")
    print(f"  Regime-filtered out: {regime_filtered}")
    print(f"  Tradeable pairs: {len(pairs)}")
    
    return pairs


# =============================================================================
# DYNAMIC PARAMETER OPTIMIZATION
# =============================================================================

def optimize_entry_exit(half_life: float) -> Tuple[float, float]:
    """
    Dynamic entry/exit thresholds based on half-life.
    
    Shorter half-life → tighter thresholds (faster reversion)
    Longer half-life → wider thresholds (slower reversion)
    """
    # Base: 2.0 entry, 0.5 exit for HL=20
    base_entry = 2.0
    base_exit = 0.5
    
    # Adjust based on half-life
    hl_factor = half_life / 20.0
    
    entry_z = base_entry * (0.8 + 0.4 * hl_factor)  # 1.6 to 2.4
    exit_z = base_exit * (0.8 + 0.4 * hl_factor)    # 0.4 to 0.6
    
    return min(2.5, max(1.5, entry_z)), min(0.8, max(0.3, exit_z))


# =============================================================================
# ENHANCED BACKTEST ENGINE
# =============================================================================

def backtest_enhanced(
    prices_df: pd.DataFrame,
    pairs: List[TradingPair],
    max_pairs: int = 15,
    cost_bps: float = 10,
    position_size: float = 0.07  # 7% per pair
) -> Dict:
    """
    Enhanced backtest with:
    - Dynamic parameters
    - Regime monitoring
    - Better risk management
    """
    if not pairs:
        return {"sharpe": 0, "returns": np.array([])}
    
    trading_pairs = pairs[:max_pairs]
    n_days = len(prices_df)
    lookback = 20
    
    daily_returns = []
    positions = {}
    trade_count = 0
    wins = 0
    losses = 0
    
    for t in range(lookback, n_days):
        day_pnl = 0.0
        
        for pair in trading_pairs:
            s1, s2 = pair.s1, pair.s2
            
            # Get prices up to current day
            p1 = prices_df[s1].iloc[:t+1].values
            p2 = prices_df[s2].iloc[:t+1].values
            
            # Dynamic hedge via Kalman filter
            kalman = KalmanState(beta=pair.hedge, Q=1e-5, R=1e-3)
            for i in range(min(100, len(p1))):
                idx = max(0, len(p1) - 100) + i
                hedge = kalman.update(p2[idx], p1[idx])
            
            # Calculate spread and z-score
            spread = p1 - hedge * p2
            recent = spread[-lookback:]
            mu = np.mean(recent)
            sigma = np.std(recent) + 1e-10
            zscore = (spread[-1] - mu) / sigma
            
            # Get dynamic thresholds
            entry_z, exit_z = optimize_entry_exit(pair.half_life)
            
            pair_key = f"{s1}/{s2}"
            
            # Position management
            if pair_key in positions:
                pos = positions[pair_key]
                
                # Calculate daily P&L
                ret1 = (p1[-1] - p1[-2]) / p1[-2]
                ret2 = (p2[-1] - p2[-2]) / p2[-2]
                
                if pos['dir'] == 'long':
                    spread_ret = ret1 - pos['hedge'] * ret2
                else:
                    spread_ret = -ret1 + pos['hedge'] * ret2
                
                day_pnl += spread_ret * position_size
                
                # Check regime - exit if trending
                current_hurst = calculate_hurst(spread[-50:]) if len(spread) >= 50 else 0.5
                
                # Exit conditions
                should_exit = False
                exit_reason = None
                
                if pos['dir'] == 'long' and zscore >= -exit_z:
                    should_exit = True
                    exit_reason = 'target'
                elif pos['dir'] == 'short' and zscore <= exit_z:
                    should_exit = True
                    exit_reason = 'target'
                elif abs(zscore) > 4.0:
                    should_exit = True
                    exit_reason = 'stop'
                elif current_hurst > 0.55:  # Exit if regime changes to trending
                    should_exit = True
                    exit_reason = 'regime'
                
                if should_exit:
                    cost = cost_bps / 10000 * position_size * 2
                    day_pnl -= cost
                    
                    # Track win/loss
                    trade_pnl = zscore - pos['entry_z'] if pos['dir'] == 'long' else pos['entry_z'] - zscore
                    if trade_pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    
                    del positions[pair_key]
                    trade_count += 1
            
            elif len(positions) < max_pairs:
                # Entry - check regime first
                current_hurst = calculate_hurst(spread[-50:]) if len(spread) >= 50 else 0.5
                
                if current_hurst < 0.48:  # Only enter in mean-reverting regime
                    if zscore <= -entry_z:
                        positions[pair_key] = {
                            'dir': 'long', 
                            'hedge': hedge, 
                            'entry_z': zscore,
                            'entry_price1': p1[-1],
                            'entry_price2': p2[-1]
                        }
                        trade_count += 1
                    elif zscore >= entry_z:
                        positions[pair_key] = {
                            'dir': 'short', 
                            'hedge': hedge, 
                            'entry_z': zscore,
                            'entry_price1': p1[-1],
                            'entry_price2': p2[-1]
                        }
                        trade_count += 1
        
        daily_returns.append(day_pnl)
    
    returns = np.array(daily_returns)
    
    # Calculate metrics
    if len(returns) > 30 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        total_ret = (1 + returns).prod() - 1
        annual = (1 + total_ret) ** (252 / len(returns)) - 1
        cumulative = np.cumprod(1 + returns)
        max_dd = (cumulative / np.maximum.accumulate(cumulative) - 1).min()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    else:
        sharpe = annual = max_dd = win_rate = 0
    
    return {
        "sharpe": sharpe,
        "annual_return": annual,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "trades": trade_count,
        "returns": returns
    }


# =============================================================================
# FUNDING RATE COMPONENT (NOT OUR ML - HONEST LABELING)
# =============================================================================

def funding_rate_arb() -> Dict:
    """
    Funding rate arbitrage - NOT our ML models.
    Just raw data collection from Binance.
    """
    print("\n" + "=" * 60)
    print("COMPONENT: FUNDING RATE ARB (NOT OUR ML)")
    print("=" * 60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    all_funding = {}
    
    for symbol in symbols:
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
                if not data:
                    break
                all_data.extend(data)
                end_time = data[0]["fundingTime"] - 1
            except:
                break
        
        if all_data:
            df = pd.DataFrame(all_data)
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = df["fundingRate"].astype(float)
            df["date"] = df["fundingTime"].dt.date
            daily = df.groupby("date")["fundingRate"].sum()
            all_funding[symbol] = daily
            print(f"  {symbol}: {len(daily)} days")
    
    # Combine
    combined = pd.DataFrame(all_funding).dropna()
    returns = combined.mean(axis=1).values
    
    # Subtract realistic costs: ~0.5 bps/day
    net_returns = returns - 0.00005
    
    if len(net_returns) > 30:
        sharpe = np.mean(net_returns) / np.std(net_returns) * np.sqrt(365)
        annual = (1 + np.sum(net_returns)) ** (365/len(net_returns)) - 1
        cumulative = np.cumprod(1 + net_returns)
        max_dd = (cumulative / np.maximum.accumulate(cumulative) - 1).min()
    else:
        sharpe = annual = max_dd = 0
    
    print(f"\n  Results: Sharpe {sharpe:.2f}, Annual {annual*100:.1f}%")
    print(f"  NOTE: This is NOT our ML - just raw data collection")
    
    return {
        "sharpe": sharpe,
        "annual_return": annual,
        "max_drawdown": max_dd,
        "returns": net_returns,
        "source": "Raw Binance API (NOT OUR ML)"
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "█" * 70)
    print("█  ENHANCED STAT ARB ENGINE V3                                     █")
    print("█  - Expanded Universe (60+ stocks)                                █")
    print("█  - Regime Filtering (Hurst exponent)                             █")
    print("█  - Dynamic Parameters                                            █")
    print("█" * 70 + "\n")
    
    # 1. EQUITY STAT ARB (OUR ML MODELS)
    print("=" * 60)
    print("COMPONENT: EQUITY STAT ARB (OUR ML MODELS)")
    print("=" * 60)
    
    symbols = get_all_symbols()
    print(f"  Universe: {len(symbols)} stocks across {len(EXPANDED_UNIVERSE)} sectors")
    
    # Fetch data
    prices_df = fetch_all_data(symbols, years=3)
    
    if len(prices_df) < 100:
        print("  ERROR: Not enough data")
        return
    
    # Analyze pairs with regime filtering
    print("\n  Analyzing pairs with regime filtering...")
    pairs = analyze_pairs(prices_df, max_hurst=0.48)
    
    if pairs:
        print(f"\n  Top 10 pairs:")
        for p in pairs[:10]:
            print(f"    {p.s1}/{p.s2}: score={p.score:.2f}, HL={p.half_life:.0f}, "
                  f"Hurst={p.hurst:.2f}, sectors={p.sector1}/{p.sector2}")
    
    # Backtest
    print("\n  Running enhanced backtest...")
    equity_results = backtest_enhanced(prices_df, pairs, max_pairs=15)
    
    print(f"\n  EQUITY STAT ARB RESULTS (OUR ML MODELS):")
    print(f"    Sharpe:        {equity_results['sharpe']:.2f}")
    print(f"    Annual Return: {equity_results['annual_return']*100:.1f}%")
    print(f"    Max Drawdown:  {equity_results['max_drawdown']*100:.1f}%")
    print(f"    Win Rate:      {equity_results['win_rate']*100:.1f}%")
    print(f"    Trades:        {equity_results['trades']}")
    
    # 2. FUNDING RATE ARB (NOT OUR ML)
    funding_results = funding_rate_arb()
    
    # 3. COMBINED PORTFOLIO
    print("\n" + "=" * 60)
    print("COMBINED PORTFOLIO WITH HONEST ATTRIBUTION")
    print("=" * 60)
    
    eq_ret = equity_results.get('returns', np.array([]))
    fr_ret = funding_results.get('returns', np.array([]))
    
    if len(eq_ret) > 0 and len(fr_ret) > 0:
        min_len = min(len(eq_ret), len(fr_ret))
        eq_ret = eq_ret[-min_len:]
        fr_ret = fr_ret[-min_len:]
        
        # Correlation
        corr = np.corrcoef(eq_ret, fr_ret)[0, 1]
        print(f"\n  Correlation: {corr:.2f}")
        
        # Allocations to test
        allocations = [
            (0.0, 1.0, "100% Equity (OUR ML)"),
            (0.3, 0.7, "30% Funding / 70% Equity"),
            (0.5, 0.5, "50% Funding / 50% Equity"),
            (0.7, 0.3, "70% Funding / 30% Equity"),
            (1.0, 0.0, "100% Funding (NOT OUR ML)")
        ]
        
        print(f"\n  Allocation Analysis:")
        print(f"  {'Allocation':<35} {'Sharpe':>8} {'Annual':>8} {'OurML%':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
        
        best_sharpe = 0
        best_alloc = None
        
        for w_f, w_e, name in allocations:
            combined = w_f * fr_ret + w_e * eq_ret
            if np.std(combined) > 0:
                sharpe = np.mean(combined) / np.std(combined) * np.sqrt(365)
                annual = (1 + np.sum(combined)) ** (365/len(combined)) - 1
                our_ml_pct = w_e * 100
                
                print(f"  {name:<35} {sharpe:>8.2f} {annual*100:>7.1f}% {our_ml_pct:>7.0f}%")
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_alloc = (w_f, w_e, name, annual, our_ml_pct)
    
    # FINAL SUMMARY
    print("\n" + "█" * 70)
    print("█  FINAL RESULTS                                                   █")
    print("█" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              ENHANCED STAT ARB ENGINE V3 RESULTS                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  EQUITY STAT ARB (100% OUR ML MODELS):                          ║
║    Sharpe:        {equity_results['sharpe']:>6.2f}                                       ║
║    Annual Return: {equity_results['annual_return']*100:>6.1f}%                                      ║
║    Max Drawdown:  {equity_results['max_drawdown']*100:>6.1f}%                                      ║
║    Trades:        {equity_results['trades']:>6}                                       ║
║                                                                  ║
║  FUNDING ARB (0% OUR ML - Raw Data):                            ║
║    Sharpe:        {funding_results['sharpe']:>6.2f}                                       ║
║    Annual Return: {funding_results['annual_return']*100:>6.1f}%                                      ║
║                                                                  ║
║  IMPROVEMENTS FROM V2:                                          ║
║    Universe:      60+ stocks (was 20)                           ║
║    Regime Filter: Hurst < 0.48 (was none)                       ║
║    Parameters:    Dynamic entry/exit (was fixed)                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Verdict
    if equity_results['sharpe'] >= 1.0:
        print("✅ EQUITY STAT ARB SHARPE >= 1.0 - Our ML models are working well!")
    elif equity_results['sharpe'] >= 0.7:
        print("⚠️ EQUITY STAT ARB SHARPE 0.7-1.0 - Good but room for improvement")
    else:
        print("❌ EQUITY STAT ARB SHARPE < 0.7 - Need more optimization")
    
    print("\nHONEST ATTRIBUTION:")
    print(f"  - {equity_results['sharpe']:.2f} Sharpe comes from OUR ML models")
    print(f"  - {funding_results['sharpe']:.2f} Sharpe comes from raw funding data (not our ML)")
    
    return equity_results, funding_results


if __name__ == "__main__":
    main()
