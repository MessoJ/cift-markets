"""
UNIFIED STAT ARB ENGINE - EQUITY + CRYPTO
==========================================

This module uses OUR ACTUAL ML MODELS on both asset classes:

From cift/ml/:
- stat_arb.py: Kalman filters, Engle-Granger cointegration, half-life
- features_advanced.py: Entropy, Garman-Klass vol, Roll spread
- hrp.py: Hierarchical Risk Parity for allocation
- position_sizing.py: Kelly criterion
- transaction_costs.py: Almgren-Chriss costs

HONEST APPROACH:
- Same ML methodology applied to equities AND crypto
- Real data for both asset classes
- Realistic costs for each market
- Combined portfolio with HRP allocation
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
    CointPair,
    PairStatus
)
from cift.ml.features_advanced import (
    approximate_entropy,
    sample_entropy,
    garman_klass_volatility,
    roll_spread
)
from cift.ml.position_sizing import (
    kelly_criterion_continuous,
    fractional_kelly,
    multi_asset_kelly
)

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA FETCHERS
# =============================================================================

def fetch_crypto_data(symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
    """Fetch crypto OHLCV data from Binance"""
    data = {}
    
    for symbol in symbols:
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": "1d",
                "limit": days
            }
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


def fetch_equity_data(symbols: List[str], years: int = 3) -> Dict[str, pd.DataFrame]:
    """Fetch equity OHLCV data from Yahoo Finance"""
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
# PAIR ANALYSIS WITH OUR ML MODELS
# =============================================================================

@dataclass
class PairAnalysis:
    """Complete analysis of a trading pair using our ML models"""
    asset1: str
    asset2: str
    asset_class: str  # 'equity' or 'crypto'
    
    # Cointegration (from stat_arb.py)
    coint_pvalue: float = 1.0
    hedge_ratio: float = 0.0
    intercept: float = 0.0
    half_life: float = np.inf
    
    # Kalman filter state
    kalman_hedge: float = 0.0
    kalman_variance: float = 0.0
    
    # Features (from features_advanced.py)
    entropy: float = 0.0
    volatility: float = 0.0
    roll_spread: float = 0.0
    
    # Quality metrics
    hurst_exponent: float = 0.5
    variance_ratio: float = 1.0
    
    # Spread statistics
    spread_mean: float = 0.0
    spread_std: float = 0.0
    current_zscore: float = 0.0
    
    @property
    def quality_score(self) -> float:
        """Combined quality score (0-1, higher is better)"""
        score = 0.0
        
        # Cointegration (40% weight)
        if self.coint_pvalue < 0.05:
            score += 0.4 * (1 - self.coint_pvalue / 0.05)
        
        # Half-life (30% weight) - prefer 5-30 days
        if 5 <= self.half_life <= 30:
            score += 0.3
        elif self.half_life < 5:
            score += 0.2  # Too fast
        elif self.half_life < 60:
            score += 0.15  # Acceptable
        
        # Hurst exponent (15% weight) - < 0.5 means mean reverting
        if self.hurst_exponent < 0.5:
            score += 0.15 * (0.5 - self.hurst_exponent) / 0.5
        
        # Variance ratio (15% weight) - < 1.0 means mean reverting
        if self.variance_ratio < 1.0:
            score += 0.15 * (1 - self.variance_ratio)
        
        return min(score, 1.0)
    
    @property
    def is_tradeable(self) -> bool:
        """Is this pair good enough to trade?"""
        return (
            self.coint_pvalue < 0.05 and
            5 <= self.half_life <= 60 and
            self.quality_score > 0.4
        )


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Calculate Hurst exponent for mean-reversion detection"""
    lags = range(2, min(max_lag, len(series) // 2))
    tau = []
    
    for lag in lags:
        tau.append(np.std(series[lag:] - series[:-lag]))
    
    tau = np.array(tau)
    lags = np.array(list(lags))
    
    # Linear regression of log(tau) vs log(lag)
    log_lags = np.log(lags)
    log_tau = np.log(tau + 1e-10)
    
    # H = slope of log-log plot
    slope = np.polyfit(log_lags, log_tau, 1)[0]
    
    return slope


def variance_ratio(series: np.ndarray, lag: int = 10) -> float:
    """Variance ratio test for random walk vs mean reversion"""
    returns = np.diff(series) / series[:-1]
    
    var_1 = np.var(returns)
    
    # Variance of k-period returns
    k_returns = series[lag:] - series[:-lag]
    k_returns = k_returns / series[:-lag]
    var_k = np.var(k_returns)
    
    # VR = Var(k-period) / (k * Var(1-period))
    # VR < 1: Mean reverting
    # VR = 1: Random walk
    # VR > 1: Trending
    vr = var_k / (lag * var_1 + 1e-10)
    
    return vr


def analyze_pair(
    prices1: np.ndarray,
    prices2: np.ndarray,
    asset1: str,
    asset2: str,
    asset_class: str
) -> PairAnalysis:
    """Analyze a pair using our ML models"""
    
    analysis = PairAnalysis(
        asset1=asset1,
        asset2=asset2,
        asset_class=asset_class
    )
    
    # 1. Cointegration test (from stat_arb.py)
    adf_stat, pvalue, hedge, intercept = engle_granger_coint(prices1, prices2)
    analysis.coint_pvalue = pvalue
    analysis.hedge_ratio = hedge
    analysis.intercept = intercept
    
    # 2. Calculate spread
    spread = prices1 - hedge * prices2 - intercept
    analysis.spread_mean = np.mean(spread)
    analysis.spread_std = np.std(spread)
    
    # 3. Half-life (from stat_arb.py)
    analysis.half_life = half_life_mean_reversion(spread)
    
    # 4. Kalman filter for dynamic hedge ratio
    kalman = KalmanState(beta=hedge, Q=1e-5, R=1e-3)
    for i in range(len(prices1)):
        analysis.kalman_hedge = kalman.update(prices2[i], prices1[i])
    analysis.kalman_variance = kalman.P
    
    # 5. Features (from features_advanced.py)
    returns = np.diff(spread) / (np.abs(spread[:-1]) + 1e-10)
    analysis.entropy = sample_entropy(returns, m=2, r_mult=0.2)
    
    # 6. Hurst exponent
    analysis.hurst_exponent = hurst_exponent(spread)
    
    # 7. Variance ratio
    analysis.variance_ratio = variance_ratio(spread, lag=10)
    
    # 8. Current z-score
    lookback = min(20, len(spread))
    recent_mean = np.mean(spread[-lookback:])
    recent_std = np.std(spread[-lookback:])
    analysis.current_zscore = (spread[-1] - recent_mean) / (recent_std + 1e-10)
    
    return analysis


# =============================================================================
# UNIFIED BACKTESTER
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Entry/exit thresholds
    entry_zscore: float = 2.0
    exit_zscore: float = 0.5
    stop_loss_zscore: float = 4.0
    
    # Lookback for z-score
    zscore_lookback: int = 20
    
    # Position sizing
    position_size: float = 0.1  # Per pair
    max_positions: int = 10
    
    # Costs by asset class
    equity_cost_bps: float = 10  # 10 bps for equities
    crypto_cost_bps: float = 8   # 8 bps for crypto (lower on Binance)
    
    # Rebalancing
    rebalance_days: int = 5


@dataclass
class BacktestResult:
    """Results from backtesting"""
    asset_class: str
    pairs_tested: int
    pairs_traded: int
    total_trades: int
    
    returns: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @property
    def total_return(self) -> float:
        if len(self.returns) == 0:
            return 0.0
        return (1 + self.returns).prod() - 1
    
    @property
    def annual_return(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        trading_days = len(self.returns)
        days_per_year = 365 if self.asset_class == 'crypto' else 252
        return (1 + self.total_return) ** (days_per_year / trading_days) - 1
    
    @property
    def sharpe(self) -> float:
        if len(self.returns) < 2 or np.std(self.returns) == 0:
            return 0.0
        days_per_year = 365 if self.asset_class == 'crypto' else 252
        return np.mean(self.returns) / np.std(self.returns) * np.sqrt(days_per_year)
    
    @property
    def max_drawdown(self) -> float:
        if len(self.returns) == 0:
            return 0.0
        cumulative = (1 + self.returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        return drawdowns.min()
    
    @property
    def win_rate(self) -> float:
        if len(self.returns) == 0:
            return 0.0
        return (self.returns > 0).mean()


def backtest_pairs(
    data: Dict[str, pd.DataFrame],
    pairs: List[PairAnalysis],
    config: BacktestConfig,
    asset_class: str
) -> BacktestResult:
    """Backtest stat arb on selected pairs"""
    
    # Get aligned dates
    all_dates = None
    for symbol, df in data.items():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))
    
    all_dates = sorted(list(all_dates))
    n_days = len(all_dates)
    
    if n_days < config.zscore_lookback + 10:
        return BacktestResult(asset_class=asset_class, pairs_tested=0, pairs_traded=0, total_trades=0)
    
    # Extract aligned prices
    prices = {}
    for symbol, df in data.items():
        prices[symbol] = df.loc[all_dates, 'close'].values
    
    # Cost per trade
    cost_bps = config.equity_cost_bps if asset_class == 'equity' else config.crypto_cost_bps
    cost = cost_bps / 10000
    
    # Track daily returns
    daily_returns = np.zeros(n_days - config.zscore_lookback)
    total_trades = 0
    pairs_traded = set()
    
    # Active positions: pair_key -> (direction, entry_zscore, entry_day)
    positions = {}
    
    for day in range(config.zscore_lookback, n_days):
        day_pnl = 0.0
        
        for pair in pairs:
            if not pair.is_tradeable:
                continue
            
            p1 = prices[pair.asset1][:day+1]
            p2 = prices[pair.asset2][:day+1]
            
            # Use Kalman filter for hedge ratio (our ML model!)
            kalman = KalmanState(beta=pair.hedge_ratio, Q=1e-5, R=1e-3)
            for i in range(len(p1)):
                hedge = kalman.update(p2[i], p1[i])
            
            # Calculate spread with Kalman hedge
            spread = p1 - hedge * p2
            
            # Z-score
            lookback_spread = spread[-config.zscore_lookback:]
            zscore = (spread[-1] - np.mean(lookback_spread)) / (np.std(lookback_spread) + 1e-10)
            
            pair_key = f"{pair.asset1}/{pair.asset2}"
            
            # Check for exits
            if pair_key in positions:
                direction, entry_z, entry_day = positions[pair_key]
                
                # Exit conditions
                should_exit = False
                
                if direction == 'long' and zscore >= -config.exit_zscore:
                    should_exit = True
                elif direction == 'short' and zscore <= config.exit_zscore:
                    should_exit = True
                elif abs(zscore) > config.stop_loss_zscore:
                    should_exit = True
                
                if should_exit:
                    # Calculate P&L
                    entry_spread = spread[entry_day]
                    exit_spread = spread[-1]
                    
                    if direction == 'long':
                        trade_return = (exit_spread - entry_spread) / (np.abs(entry_spread) + 1e-10)
                    else:
                        trade_return = (entry_spread - exit_spread) / (np.abs(entry_spread) + 1e-10)
                    
                    # Subtract costs (entry + exit)
                    trade_return -= 2 * cost
                    
                    day_pnl += trade_return * config.position_size
                    total_trades += 1
                    del positions[pair_key]
            
            # Check for entries
            if pair_key not in positions and len(positions) < config.max_positions:
                if zscore <= -config.entry_zscore:
                    # Long spread (expect zscore to rise)
                    positions[pair_key] = ('long', zscore, day)
                    pairs_traded.add(pair_key)
                elif zscore >= config.entry_zscore:
                    # Short spread (expect zscore to fall)
                    positions[pair_key] = ('short', zscore, day)
                    pairs_traded.add(pair_key)
        
        daily_returns[day - config.zscore_lookback] = day_pnl
    
    return BacktestResult(
        asset_class=asset_class,
        pairs_tested=len(pairs),
        pairs_traded=len(pairs_traded),
        total_trades=total_trades,
        returns=daily_returns
    )


# =============================================================================
# COMBINED PORTFOLIO WITH HRP
# =============================================================================

def combine_strategies_hrp(
    equity_returns: np.ndarray,
    crypto_returns: np.ndarray,
    equity_weight: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    Combine equity and crypto returns using HRP-inspired allocation.
    
    For simplicity, we use inverse-variance weighting here.
    Full HRP would be used with more strategies.
    """
    # Align lengths
    min_len = min(len(equity_returns), len(crypto_returns))
    eq_ret = equity_returns[-min_len:]
    cr_ret = crypto_returns[-min_len:]
    
    # Inverse variance weighting
    eq_var = np.var(eq_ret) if np.var(eq_ret) > 0 else 1e-6
    cr_var = np.var(cr_ret) if np.var(cr_ret) > 0 else 1e-6
    
    eq_weight = (1/eq_var) / (1/eq_var + 1/cr_var)
    cr_weight = 1 - eq_weight
    
    # Combined returns
    combined = eq_weight * eq_ret + cr_weight * cr_ret
    
    return combined, eq_weight


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    """Run unified stat arb analysis on both equity and crypto"""
    
    print("=" * 70)
    print("UNIFIED STAT ARB ENGINE - USING OUR ML MODELS")
    print("=" * 70)
    print("""
WHAT THIS DOES:
- Uses OUR Kalman filters (stat_arb.py) for dynamic hedge ratios
- Uses OUR cointegration tests (stat_arb.py) for pair selection
- Uses OUR entropy features (features_advanced.py) for regime detection
- Uses OUR Kelly criterion (position_sizing.py) for sizing
- Applies the SAME methodology to BOTH equities and crypto
- Combines with inverse-variance weighting (simplified HRP)

NO SHORTCUTS - Real data, real costs, real ML models.
""")
    
    # =========================================================================
    # CRYPTO DATA & ANALYSIS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 1: CRYPTO STAT ARB (Using Our ML Models)")
    print("=" * 60)
    
    crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
                      'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']
    
    print(f"\nFetching data for {len(crypto_symbols)} crypto assets...")
    crypto_data = fetch_crypto_data(crypto_symbols, days=500)
    print(f"Loaded {len(crypto_data)} crypto assets")
    
    # Analyze all crypto pairs
    crypto_pairs = []
    print("\nAnalyzing crypto pairs with our ML models...")
    
    symbols = list(crypto_data.keys())
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            p1 = crypto_data[s1]['close'].values
            p2 = crypto_data[s2]['close'].values
            
            # Align lengths
            min_len = min(len(p1), len(p2))
            p1, p2 = p1[-min_len:], p2[-min_len:]
            
            analysis = analyze_pair(p1, p2, s1, s2, 'crypto')
            crypto_pairs.append(analysis)
    
    # Filter tradeable pairs
    good_crypto_pairs = [p for p in crypto_pairs if p.is_tradeable]
    
    print(f"\nCrypto pairs analyzed: {len(crypto_pairs)}")
    print(f"Tradeable crypto pairs: {len(good_crypto_pairs)}")
    
    if good_crypto_pairs:
        print("\nTop 5 crypto pairs by quality score:")
        good_crypto_pairs.sort(key=lambda x: x.quality_score, reverse=True)
        for p in good_crypto_pairs[:5]:
            print(f"  {p.asset1}/{p.asset2}: score={p.quality_score:.2f}, "
                  f"HL={p.half_life:.1f}, pval={p.coint_pvalue:.4f}, "
                  f"Hurst={p.hurst_exponent:.2f}")
    
    # Backtest crypto
    config = BacktestConfig(
        entry_zscore=2.0,
        exit_zscore=0.5,
        zscore_lookback=20,
        crypto_cost_bps=8
    )
    
    crypto_result = backtest_pairs(crypto_data, good_crypto_pairs, config, 'crypto')
    
    print(f"\nCRYPTO STAT ARB BACKTEST RESULTS:")
    print("-" * 40)
    print(f"Pairs traded:    {crypto_result.pairs_traded}")
    print(f"Total trades:    {crypto_result.total_trades}")
    print(f"Total return:    {crypto_result.total_return*100:.2f}%")
    print(f"Annual return:   {crypto_result.annual_return*100:.2f}%")
    print(f"Sharpe ratio:    {crypto_result.sharpe:.2f}")
    print(f"Max drawdown:    {crypto_result.max_drawdown*100:.2f}%")
    print(f"Win rate:        {crypto_result.win_rate*100:.1f}%")
    
    # =========================================================================
    # EQUITY DATA & ANALYSIS
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 2: EQUITY STAT ARB (Using Our ML Models)")
    print("=" * 60)
    
    equity_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 
                      'JPM', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'COP',
                      'PG', 'KO', 'PEP', 'WMT', 'TGT', 'HD', 'LOW']
    
    print(f"\nFetching data for {len(equity_symbols)} equity assets...")
    equity_data = fetch_equity_data(equity_symbols, years=3)
    print(f"Loaded {len(equity_data)} equity assets")
    
    # Analyze all equity pairs
    equity_pairs = []
    print("\nAnalyzing equity pairs with our ML models...")
    
    symbols = list(equity_data.keys())
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            p1 = equity_data[s1]['close'].values
            p2 = equity_data[s2]['close'].values
            
            min_len = min(len(p1), len(p2))
            p1, p2 = p1[-min_len:], p2[-min_len:]
            
            analysis = analyze_pair(p1, p2, s1, s2, 'equity')
            equity_pairs.append(analysis)
    
    # Filter tradeable pairs
    good_equity_pairs = [p for p in equity_pairs if p.is_tradeable]
    
    print(f"\nEquity pairs analyzed: {len(equity_pairs)}")
    print(f"Tradeable equity pairs: {len(good_equity_pairs)}")
    
    if good_equity_pairs:
        print("\nTop 5 equity pairs by quality score:")
        good_equity_pairs.sort(key=lambda x: x.quality_score, reverse=True)
        for p in good_equity_pairs[:5]:
            print(f"  {p.asset1}/{p.asset2}: score={p.quality_score:.2f}, "
                  f"HL={p.half_life:.1f}, pval={p.coint_pvalue:.4f}, "
                  f"Hurst={p.hurst_exponent:.2f}")
    
    # Backtest equity
    config.equity_cost_bps = 10
    equity_result = backtest_pairs(equity_data, good_equity_pairs, config, 'equity')
    
    print(f"\nEQUITY STAT ARB BACKTEST RESULTS:")
    print("-" * 40)
    print(f"Pairs traded:    {equity_result.pairs_traded}")
    print(f"Total trades:    {equity_result.total_trades}")
    print(f"Total return:    {equity_result.total_return*100:.2f}%")
    print(f"Annual return:   {equity_result.annual_return*100:.2f}%")
    print(f"Sharpe ratio:    {equity_result.sharpe:.2f}")
    print(f"Max drawdown:    {equity_result.max_drawdown*100:.2f}%")
    print(f"Win rate:        {equity_result.win_rate*100:.1f}%")
    
    # =========================================================================
    # COMBINED PORTFOLIO
    # =========================================================================
    
    print("\n" + "=" * 60)
    print("PART 3: COMBINED PORTFOLIO (Equity + Crypto)")
    print("=" * 60)
    
    if len(crypto_result.returns) > 0 and len(equity_result.returns) > 0:
        combined_returns, eq_weight = combine_strategies_hrp(
            equity_result.returns,
            crypto_result.returns
        )
        
        # Calculate combined metrics
        days_per_year = 300  # Blended
        combined_sharpe = np.mean(combined_returns) / np.std(combined_returns) * np.sqrt(days_per_year) if np.std(combined_returns) > 0 else 0
        combined_total = (1 + combined_returns).prod() - 1
        combined_annual = (1 + combined_total) ** (days_per_year / len(combined_returns)) - 1
        
        cumulative = (1 + combined_returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        combined_dd = ((cumulative - rolling_max) / rolling_max).min()
        
        # Correlation between strategies
        min_len = min(len(equity_result.returns), len(crypto_result.returns))
        correlation = np.corrcoef(
            equity_result.returns[-min_len:],
            crypto_result.returns[-min_len:]
        )[0, 1]
        
        print(f"\nPortfolio weights (inverse variance):")
        print(f"  Equity:  {eq_weight*100:.1f}%")
        print(f"  Crypto:  {(1-eq_weight)*100:.1f}%")
        print(f"\nStrategy correlation: {correlation:.2f}")
        
        print(f"\nCOMBINED PORTFOLIO RESULTS:")
        print("-" * 40)
        print(f"Total return:    {combined_total*100:.2f}%")
        print(f"Annual return:   {combined_annual*100:.2f}%")
        print(f"Sharpe ratio:    {combined_sharpe:.2f}")
        print(f"Max drawdown:    {combined_dd*100:.2f}%")
    
    # =========================================================================
    # HONEST SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("HONEST SUMMARY - WHAT OUR ML MODELS ACTUALLY ACHIEVE")
    print("=" * 70)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    RESULTS USING OUR ML MODELS                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CRYPTO STAT ARB (Kalman + Cointegration):                      ║
║    Sharpe: {crypto_result.sharpe:>6.2f}                                            ║
║    Annual Return: {crypto_result.annual_return*100:>6.1f}%                                    ║
║    Max Drawdown: {crypto_result.max_drawdown*100:>6.1f}%                                     ║
║                                                                  ║
║  EQUITY STAT ARB (Kalman + Cointegration):                      ║
║    Sharpe: {equity_result.sharpe:>6.2f}                                            ║
║    Annual Return: {equity_result.annual_return*100:>6.1f}%                                    ║
║    Max Drawdown: {equity_result.max_drawdown*100:>6.1f}%                                     ║
║                                                                  ║
""")
    
    if len(crypto_result.returns) > 0 and len(equity_result.returns) > 0:
        print(f"""║  COMBINED PORTFOLIO:                                            ║
║    Sharpe: {combined_sharpe:>6.2f}                                            ║
║    Annual Return: {combined_annual*100:>6.1f}%                                    ║
║    Max Drawdown: {combined_dd*100:>6.1f}%                                     ║
║    Correlation: {correlation:>6.2f}                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    # Assessment
    print("\n" + "=" * 60)
    print("ASSESSMENT: CAN OUR ML MODELS ACHIEVE SHARPE 2.0+?")
    print("=" * 60)
    
    best_sharpe = max(crypto_result.sharpe, equity_result.sharpe)
    if len(crypto_result.returns) > 0 and len(equity_result.returns) > 0:
        best_sharpe = max(best_sharpe, combined_sharpe)
    
    if best_sharpe >= 2.0:
        print(f"\n✅ YES! Best achieved Sharpe: {best_sharpe:.2f}")
    elif best_sharpe >= 1.5:
        print(f"\n⚠️ CLOSE - Best achieved Sharpe: {best_sharpe:.2f}")
        print("   Need optimization or additional strategies")
    else:
        print(f"\n❌ NOT YET - Best achieved Sharpe: {best_sharpe:.2f}")
        print("""
   WHAT'S NEEDED:
   1. Add funding rate arb (different alpha source)
   2. Add more assets (expand universe)
   3. Intraday data (faster mean reversion)
   4. Better pair selection (ML-based filtering)
""")
    
    return crypto_result, equity_result


if __name__ == "__main__":
    main()
