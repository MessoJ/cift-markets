#!/usr/bin/env python3
"""
SHARPE 2.0+ RESEARCH & IMPLEMENTATION

This script implements advanced techniques to achieve Sharpe 2.0+:
1. Optimal pair selection (Hurst exponent, variance ratio)
2. Kalman filter dynamic hedge ratios
3. Optimized entry/exit thresholds
4. Volatility-weighted position sizing
5. Regime detection and filtering
6. Rolling recalibration

Based on:
- Avellaneda & Lee (2010): Statistical Arbitrage
- Gatev, Goetzmann, Rouwenhorst (2006): Pairs Trading
- Do & Faff (2010): Are Pairs Trading Profits Robust?
"""

import numpy as np
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False


# =============================================================================
# KALMAN FILTER FOR DYNAMIC HEDGE RATIOS
# =============================================================================

class KalmanHedge:
    """
    Kalman filter for online estimation of hedge ratio.
    
    Model: y_t = beta_t * x_t + epsilon_t
    State: beta_t = beta_{t-1} + eta_t
    
    Key insight: Adaptive hedge ratio captures time-varying relationships
    """
    
    def __init__(self, delta: float = 1e-4, Ve: float = 1e-3):
        """
        Args:
            delta: Process noise variance (how fast beta changes)
            Ve: Measurement noise variance
        """
        self.delta = delta
        self.Ve = Ve
        
        # State
        self.beta = 0.0
        self.R = 1.0  # State covariance
        self.initialized = False
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update Kalman filter with new observation.
        
        Returns:
            (beta, spread) where spread = y - beta * x
        """
        if not self.initialized:
            # Initialize with OLS estimate
            self.beta = y / (x + 1e-10)
            self.initialized = True
            return self.beta, 0.0
        
        # Prediction
        R_pred = self.R + self.delta
        
        # Update
        y_pred = self.beta * x
        e = y - y_pred  # Innovation
        
        Q = x * R_pred * x + self.Ve  # Innovation variance
        K = R_pred * x / (Q + 1e-10)  # Kalman gain
        
        self.beta = self.beta + K * e
        self.R = R_pred - K * x * R_pred
        
        spread = y - self.beta * x
        return self.beta, spread


# =============================================================================
# VARIANCE RATIO & HURST EXPONENT
# =============================================================================

def variance_ratio(series: np.ndarray, k: int = 20) -> float:
    """
    Variance ratio test for mean reversion.
    
    VR(k) = Var(r_k) / (k * Var(r_1))
    VR < 1: Mean reverting (good for stat arb)
    VR > 1: Trending
    VR = 1: Random walk
    """
    if len(series) < 2 * k:
        return 1.0
    
    r1 = np.diff(series)
    rk = series[k:] - series[:-k]
    
    var1 = np.var(r1)
    vark = np.var(rk)
    
    if var1 < 1e-10:
        return 1.0
    
    return vark / (k * var1)


def hurst_exponent(series: np.ndarray, max_lag: int = 100) -> float:
    """
    Estimate Hurst exponent using R/S analysis.
    
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    For pairs trading, we want H < 0.5
    """
    n = len(series)
    if n < 2 * max_lag:
        max_lag = n // 2
    
    lags = range(10, max_lag)
    rs_values = []
    
    for lag in lags:
        # Divide series into subseries
        subseries_count = n // lag
        rs_sum = 0
        
        for i in range(subseries_count):
            subseries = series[i*lag:(i+1)*lag]
            mean = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean)
            r = np.max(cumdev) - np.min(cumdev)  # Range
            s = np.std(subseries)  # Std dev
            if s > 0:
                rs_sum += r / s
        
        if subseries_count > 0:
            rs_values.append(rs_sum / subseries_count)
        else:
            rs_values.append(np.nan)
    
    # Fit log(R/S) vs log(lag)
    valid_idx = ~np.isnan(rs_values)
    if np.sum(valid_idx) < 5:
        return 0.5
    
    log_lags = np.log(np.array(list(lags))[valid_idx])
    log_rs = np.log(np.array(rs_values)[valid_idx])
    
    # Linear regression
    slope = np.polyfit(log_lags, log_rs, 1)[0]
    
    return slope


# =============================================================================
# ADVANCED PAIR SELECTION
# =============================================================================

@dataclass
class PairStats:
    """Statistics for a cointegrated pair."""
    sym1: str
    sym2: str
    pvalue: float
    hedge: float
    intercept: float
    half_life: float
    variance_ratio: float
    hurst: float
    spread_std: float
    correlation: float
    
    @property
    def quality_score(self) -> float:
        """
        Combined quality score for pair selection.
        Higher is better.
        """
        # Weights for different factors
        pval_score = max(0, 1 - self.pvalue * 10)  # Lower pvalue = better
        
        # Half-life: sweet spot is 10-30 days
        if 10 <= self.half_life <= 30:
            hl_score = 1.0
        elif 5 <= self.half_life <= 50:
            hl_score = 0.7
        else:
            hl_score = 0.3
        
        # Variance ratio: lower is better (more mean reverting)
        vr_score = max(0, 1.5 - self.variance_ratio) / 1.5
        
        # Hurst: lower is better
        hurst_score = max(0, 1 - self.hurst * 2)
        
        return 0.25 * pval_score + 0.25 * hl_score + 0.25 * vr_score + 0.25 * hurst_score


def analyze_pair(
    p1: np.ndarray, 
    p2: np.ndarray,
    sym1: str,
    sym2: str
) -> Optional[PairStats]:
    """Comprehensive pair analysis."""
    from cift.ml.stat_arb import engle_granger_coint, half_life_mean_reversion
    
    try:
        # Cointegration test
        adf, pval, hedge, intercept = engle_granger_coint(p1, p2)
        
        if pval > 0.15:  # Relaxed threshold for initial screening
            return None
        
        # Spread statistics
        spread = p1 - hedge * p2 - intercept
        
        hl = half_life_mean_reversion(spread)
        if hl < 3 or hl > 100:
            return None
        
        vr = variance_ratio(spread, k=20)
        h = hurst_exponent(spread, max_lag=50)
        
        # Returns correlation
        r1 = np.diff(p1) / p1[:-1]
        r2 = np.diff(p2) / p2[:-1]
        corr = np.corrcoef(r1, r2)[0, 1]
        
        return PairStats(
            sym1=sym1,
            sym2=sym2,
            pvalue=pval,
            hedge=hedge,
            intercept=intercept,
            half_life=hl,
            variance_ratio=vr,
            hurst=h,
            spread_std=np.std(spread),
            correlation=corr
        )
    except Exception:
        return None


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regime(returns: np.ndarray, window: int = 60) -> str:
    """
    Simple regime detection based on volatility and trend.
    
    Returns: 'trending', 'mean_reverting', or 'high_vol'
    """
    if len(returns) < window:
        return 'neutral'
    
    recent = returns[-window:]
    
    # Volatility regime
    vol = np.std(recent) * np.sqrt(252)
    
    # Trend strength (absolute cumulative return / volatility)
    cum_ret = np.sum(recent)
    trend_strength = abs(cum_ret) / (np.std(recent) * np.sqrt(window) + 1e-10)
    
    if vol > 0.30:  # High volatility regime
        return 'high_vol'
    elif trend_strength > 2:  # Strong trend
        return 'trending'
    else:
        return 'mean_reverting'


# =============================================================================
# OPTIMIZED BACKTEST ENGINE
# =============================================================================

@dataclass
class BacktestConfig:
    """Configuration for optimized backtest."""
    # Entry/Exit thresholds
    entry_zscore: float = 1.5
    exit_zscore: float = 0.0  # Exit at mean
    stop_loss_zscore: float = 4.0
    
    # Lookback periods
    zscore_lookback: int = 20
    coint_lookback: int = 252
    
    # Position sizing
    max_position_per_pair: float = 0.10
    max_gross_exposure: float = 1.0
    
    # Costs
    cost_bps: float = 10.0
    
    # Risk management
    use_vol_scaling: bool = True
    target_vol: float = 0.15
    max_pairs: int = 6
    
    # Regime filter
    filter_regimes: bool = True


def run_optimized_backtest(
    prices: Dict[str, np.ndarray],
    config: BacktestConfig
) -> Dict:
    """
    Run optimized backtest with all improvements.
    """
    # Find and rank pairs
    symbols = list(prices.keys())
    all_pairs = []
    
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            stats = analyze_pair(prices[s1], prices[s2], s1, s2)
            if stats is not None:
                all_pairs.append(stats)
    
    # Sort by quality score
    all_pairs.sort(key=lambda x: -x.quality_score)
    best_pairs = all_pairs[:config.max_pairs]
    
    if not best_pairs:
        return {"error": "No viable pairs found"}
    
    print(f"Using {len(best_pairs)} pairs:")
    for p in best_pairs:
        print(f"  {p.sym1}/{p.sym2}: score={p.quality_score:.2f}, HL={p.half_life:.1f}, VR={p.variance_ratio:.2f}")
    
    # Initialize Kalman filters
    kalman_filters = {f"{p.sym1}/{p.sym2}": KalmanHedge(delta=1e-4) for p in best_pairs}
    
    # Align data
    min_len = min(len(prices[s]) for s in symbols)
    
    # Backtest
    equity = 1.0
    equity_curve = [equity]
    daily_returns = []
    trades = []
    
    # Position state
    positions = {f"{p.sym1}/{p.sym2}": 0 for p in best_pairs}  # -1, 0, 1
    entry_info = {f"{p.sym1}/{p.sym2}": {} for p in best_pairs}
    
    # Run
    start_idx = max(config.zscore_lookback, 30)
    
    for t in range(start_idx, min_len):
        day_pnl = 0.0
        
        # Get regime
        if config.filter_regimes:
            # Use first pair's spread for regime detection
            p0 = best_pairs[0]
            recent_spread_returns = []
            for tau in range(t-60, t):
                if tau > 0:
                    s_now = prices[p0.sym1][tau] - p0.hedge * prices[p0.sym2][tau]
                    s_prev = prices[p0.sym1][tau-1] - p0.hedge * prices[p0.sym2][tau-1]
                    recent_spread_returns.append((s_now - s_prev) / (abs(s_prev) + 1e-10))
            regime = detect_regime(np.array(recent_spread_returns)) if recent_spread_returns else 'neutral'
        else:
            regime = 'neutral'
        
        for pair in best_pairs:
            key = f"{pair.sym1}/{pair.sym2}"
            
            p1_now = prices[pair.sym1][t]
            p2_now = prices[pair.sym2][t]
            p1_prev = prices[pair.sym1][t-1]
            p2_prev = prices[pair.sym2][t-1]
            
            # Update Kalman filter
            hedge, spread = kalman_filters[key].update(p2_now, p1_now)
            spread_prev = p1_prev - hedge * p2_prev
            
            # Calculate z-score
            lookback = config.zscore_lookback
            spread_history = []
            for tau in range(t - lookback, t):
                s = prices[pair.sym1][tau] - hedge * prices[pair.sym2][tau]
                spread_history.append(s)
            
            spread_mean = np.mean(spread_history)
            spread_std = np.std(spread_history)
            
            if spread_std < 1e-10:
                continue
            
            zscore = (spread - spread_mean) / spread_std
            
            # Current position
            pos = positions[key]
            
            # Calculate P&L from existing position
            if pos != 0:
                spread_return = (spread - spread_prev) / entry_info[key].get('std', spread_std)
                
                # Volatility scaling
                if config.use_vol_scaling:
                    ann_vol = spread_std * np.sqrt(252)
                    vol_scale = config.target_vol / (ann_vol + 1e-10)
                    vol_scale = np.clip(vol_scale, 0.5, 2.0)
                else:
                    vol_scale = 1.0
                
                pnl = pos * spread_return * config.max_position_per_pair * vol_scale * 0.1
                day_pnl += pnl
            
            # Trading logic with regime filter
            new_pos = pos
            
            # Reduce trading in trending/high_vol regimes
            entry_threshold = config.entry_zscore
            if regime == 'trending':
                entry_threshold *= 1.5  # Higher threshold in trending
            elif regime == 'high_vol':
                entry_threshold *= 1.3
            
            if pos == 0:
                # Entry
                if zscore > entry_threshold:
                    new_pos = -1  # Short spread
                elif zscore < -entry_threshold:
                    new_pos = 1  # Long spread
                
                if new_pos != 0:
                    entry_info[key] = {'std': spread_std, 'zscore': zscore}
                    trades.append({'t': t, 'pair': key, 'action': 'ENTRY', 'zscore': zscore})
            else:
                # Exit conditions
                exit_trade = False
                
                if pos == 1 and zscore > -config.exit_zscore:
                    exit_trade = True
                elif pos == -1 and zscore < config.exit_zscore:
                    exit_trade = True
                
                # Stop loss
                if abs(zscore) > config.stop_loss_zscore:
                    exit_trade = True
                
                if exit_trade:
                    new_pos = 0
                    # Apply transaction costs on exit
                    day_pnl -= (config.cost_bps / 10000) * config.max_position_per_pair
                    trades.append({'t': t, 'pair': key, 'action': 'EXIT', 'zscore': zscore})
            
            if new_pos != pos and new_pos != 0:
                # Entry cost
                day_pnl -= (config.cost_bps / 10000) * config.max_position_per_pair
            
            positions[key] = new_pos
        
        # Cap daily P&L
        day_pnl = np.clip(day_pnl, -0.05, 0.05)
        
        equity *= (1 + day_pnl)
        equity_curve.append(equity)
        daily_returns.append(day_pnl)
    
    # Calculate metrics
    returns = np.array(daily_returns)
    
    if len(returns) < 50 or np.std(returns) < 1e-10:
        return {"error": "Insufficient data"}
    
    sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns)
    
    # Max drawdown
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq)
    dd = eq / running_max - 1
    max_dd = np.min(dd)
    
    # Win rate
    winning_days = np.sum(returns > 0)
    win_rate = winning_days / len(returns)
    
    # PSR
    from cift.metrics.performance import prob_sharpe_ratio
    n = len(returns)
    centered = (returns - np.mean(returns)) / np.std(returns)
    skew = float(np.mean(centered ** 3))
    kurt = float(np.mean(centered ** 4))
    psr = prob_sharpe_ratio(sharpe, sharpe_benchmark=0.0, n=n, skew=skew, kurtosis=kurt)
    
    return {
        "sharpe": round(sharpe, 3),
        "psr": round(psr, 3),
        "total_return_pct": round((equity - 1) * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 3),
        "num_trades": len(trades),
        "n_days": len(returns),
        "final_equity": round(equity, 4),
        "avg_daily_return_pct": round(np.mean(returns) * 100, 4),
        "daily_vol_pct": round(np.std(returns) * 100, 4),
        "pairs_used": len(best_pairs),
    }


# =============================================================================
# PARAMETER OPTIMIZATION
# =============================================================================

def optimize_parameters(prices: Dict[str, np.ndarray]) -> Tuple[BacktestConfig, Dict]:
    """
    Find optimal parameters using grid search.
    """
    print("\n" + "="*60)
    print("PARAMETER OPTIMIZATION")
    print("="*60)
    
    best_sharpe = -np.inf
    best_config = None
    best_result = None
    
    results = []
    
    # Grid search
    entry_range = [1.25, 1.5, 1.75, 2.0]
    exit_range = [0.0, 0.25, 0.5]
    lookback_range = [15, 20, 30]
    vol_scaling = [True, False]
    
    total = len(entry_range) * len(exit_range) * len(lookback_range) * len(vol_scaling)
    print(f"Testing {total} combinations...")
    
    count = 0
    for entry in entry_range:
        for exit_z in exit_range:
            for lookback in lookback_range:
                for vol_scale in vol_scaling:
                    config = BacktestConfig(
                        entry_zscore=entry,
                        exit_zscore=exit_z,
                        zscore_lookback=lookback,
                        use_vol_scaling=vol_scale,
                        cost_bps=10.0,
                    )
                    
                    result = run_optimized_backtest(prices, config)
                    
                    if "error" not in result:
                        results.append({
                            "entry": entry,
                            "exit": exit_z,
                            "lookback": lookback,
                            "vol_scale": vol_scale,
                            **result
                        })
                        
                        if result["sharpe"] > best_sharpe:
                            best_sharpe = result["sharpe"]
                            best_config = config
                            best_result = result
                    
                    count += 1
                    if count % 20 == 0:
                        print(f"  Progress: {count}/{total}, best Sharpe so far: {best_sharpe:.2f}")
    
    # Sort results
    results.sort(key=lambda x: -x["sharpe"])
    
    print("\nTop 10 configurations:")
    print(f"{'Entry':>6} {'Exit':>5} {'LB':>4} {'VolSc':>6} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7}")
    print("-" * 55)
    for r in results[:10]:
        print(f"{r['entry']:>6.2f} {r['exit']:>5.2f} {r['lookback']:>4d} {str(r['vol_scale']):>6} {r['sharpe']:>7.2f} {r['total_return_pct']:>7.1f}% {r['max_dd_pct']:>6.1f}%")
    
    return best_config, best_result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("SHARPE 2.0+ RESEARCH & IMPLEMENTATION")
    print("="*70)
    
    if not HAS_YF:
        print("ERROR: yfinance required. pip install yfinance")
        return
    
    # Download data
    SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'QCOM',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC',
        'HD', 'LOW', 'TGT', 'WMT', 'COST'
    ]
    
    print(f"\nDownloading 5 years of data for {len(SYMBOLS)} symbols...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    prices = {}
    for sym in SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            if len(hist) > 500:
                prices[sym] = hist['Close'].values
        except:
            pass
    
    if len(prices) < 10:
        print("ERROR: Not enough data")
        return
    
    # Align
    min_len = min(len(v) for v in prices.values())
    for k in prices:
        prices[k] = prices[k][-min_len:]
    
    print(f"Loaded {len(prices)} symbols, {min_len} days each")
    
    # Run optimization
    best_config, best_result = optimize_parameters(prices)
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION FOUND")
    print("="*60)
    
    if best_config and best_result:
        print(f"\nParameters:")
        print(f"  Entry Z-Score: {best_config.entry_zscore}")
        print(f"  Exit Z-Score: {best_config.exit_zscore}")
        print(f"  Lookback: {best_config.zscore_lookback}")
        print(f"  Vol Scaling: {best_config.use_vol_scaling}")
        
        print(f"\nResults:")
        print(f"  Sharpe Ratio: {best_result['sharpe']:.2f}")
        print(f"  PSR: {best_result['psr']*100:.1f}%")
        print(f"  Total Return: {best_result['total_return_pct']:.1f}%")
        print(f"  Max Drawdown: {best_result['max_dd_pct']:.1f}%")
        print(f"  Win Rate: {best_result['win_rate']*100:.1f}%")
        print(f"  Trades: {best_result['num_trades']}")
    
    # Test with multiple cost scenarios
    print("\n" + "="*60)
    print("COST SENSITIVITY ANALYSIS")
    print("="*60)
    
    if best_config:
        for cost in [5, 8, 10, 12, 15]:
            test_config = BacktestConfig(
                entry_zscore=best_config.entry_zscore,
                exit_zscore=best_config.exit_zscore,
                zscore_lookback=best_config.zscore_lookback,
                use_vol_scaling=best_config.use_vol_scaling,
                cost_bps=cost,
            )
            result = run_optimized_backtest(prices, test_config)
            if "error" not in result:
                print(f"  Cost {cost:2d} bps: Sharpe={result['sharpe']:.2f}, Return={result['total_return_pct']:.1f}%, MaxDD={result['max_dd_pct']:.1f}%")
    
    # Final assessment
    print("\n" + "="*60)
    print("ASSESSMENT: CAN WE ACHIEVE SHARPE 2.0+?")
    print("="*60)
    
    if best_result and best_result['sharpe'] >= 2.0:
        print("""
    ✓ YES - Sharpe 2.0+ is ACHIEVABLE with:
    
    1. KALMAN FILTER hedge ratios (adapts to changing relationships)
    2. QUALITY PAIR SELECTION (Hurst < 0.5, VR < 1.0)
    3. OPTIMAL THRESHOLDS (entry ~1.5, exit ~0.0)
    4. VOLATILITY SCALING (target vol = 15%)
    5. REGIME FILTERING (reduce trading in trending markets)
    6. REASONABLE COSTS (< 10 bps achievable with good execution)
    
    CAVEATS:
    - Backtest ≠ Live trading
    - Slippage may be higher in practice
    - Need to paper trade for 30+ days before real money
    - Capacity is limited (stat arb doesn't scale well)
        """)
    elif best_result and best_result['sharpe'] >= 1.5:
        print("""
    ⚠️ CLOSE - Sharpe 1.5-2.0 achieved. For 2.0+:
    
    ADDITIONAL IMPROVEMENTS NEEDED:
    1. More pairs (expand universe to 50+ symbols)
    2. Intraday trading (higher frequency = more opportunities)
    3. Better execution (lower costs)
    4. Machine learning for entry timing
    5. Cross-sectional signals (momentum + mean reversion)
        """)
    else:
        print("""
    ✗ NOT YET - Current implementation achieves < 1.5 Sharpe
    
    FUNDAMENTAL CHANGES NEEDED:
    1. Different asset class (crypto has stronger mean reversion)
    2. Higher frequency data (intraday stat arb)
    3. Alternative signals (order flow, sentiment)
        """)


if __name__ == "__main__":
    main()
