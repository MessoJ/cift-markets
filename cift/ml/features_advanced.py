"""
Advanced Feature Engineering for High-Sharpe Strategies

This module implements state-of-the-art features from academic research:
- Entropy features (regime detection)
- VPIN (Volume-Synchronized Probability of Informed Trading)
- Garman-Klass volatility
- Roll spread estimator
- Kyle's Lambda (market impact)
- Corwin-Schultz spread estimator

References:
- De Prado, "Advances in Financial Machine Learning" (2018)
- Easley, Lopez de Prado, O'Hara, "Flow Toxicity and Liquidity" (2012)
- Roll, "A Simple Implicit Measure of the Bid-Ask Spread" (1984)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import polars as pl


# =============================================================================
# ENTROPY FEATURES - Regime Detection
# =============================================================================

def approximate_entropy(series: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """
    Approximate Entropy (ApEn) - Measures regularity/predictability.
    
    Low entropy = Structured/trending market (trade momentum)
    High entropy = Random/noisy market (trade mean-reversion or avoid)
    
    Args:
        series: Price or return series
        m: Embedding dimension (pattern length)
        r_mult: Tolerance as multiple of std dev
        
    Returns:
        ApEn value (0 = perfectly regular, higher = more random)
        
    Reference:
        Pincus (1991), "Approximate entropy as a measure of system complexity"
    """
    n = len(series)
    if n < m + 2:
        return np.nan
    
    series = np.asarray(series, dtype=np.float64)
    r = r_mult * np.std(series)
    
    if r == 0:
        return np.nan
    
    def _phi(m_val: int) -> float:
        """Count similar patterns."""
        patterns = np.array([series[i:i + m_val] for i in range(n - m_val + 1)])
        counts = np.zeros(len(patterns))
        
        for i, pattern in enumerate(patterns):
            # Chebyshev distance
            distances = np.max(np.abs(patterns - pattern), axis=1)
            counts[i] = np.sum(distances <= r)
        
        # Avoid log(0)
        counts = np.maximum(counts, 1)
        return np.sum(np.log(counts / len(patterns))) / len(patterns)
    
    return _phi(m) - _phi(m + 1)


def sample_entropy(series: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) - More robust version of ApEn.
    
    Removes self-matches, reducing bias for short series.
    
    Args:
        series: Price or return series
        m: Embedding dimension
        r_mult: Tolerance as multiple of std dev
        
    Returns:
        SampEn value
        
    Reference:
        Richman & Moorman (2000), "Physiological time-series analysis"
    """
    n = len(series)
    if n < m + 2:
        return np.nan
    
    series = np.asarray(series, dtype=np.float64)
    r = r_mult * np.std(series)
    
    if r == 0:
        return np.nan
    
    def _count_matches(m_val: int) -> int:
        patterns = np.array([series[i:i + m_val] for i in range(n - m_val)])
        count = 0
        
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                    count += 1
        
        return count
    
    a = _count_matches(m + 1)
    b = _count_matches(m)
    
    if b == 0 or a == 0:
        return np.nan
    
    return -np.log(a / b)


def rolling_entropy(
    df: pl.DataFrame,
    close_col: str = "close",
    window: int = 50,
    entropy_type: str = "sample"
) -> pl.Series:
    """
    Rolling entropy calculation for regime detection.
    
    Args:
        df: DataFrame with price data
        close_col: Name of close price column
        window: Rolling window size
        entropy_type: "sample" or "approximate"
        
    Returns:
        Series of entropy values
    """
    prices = df[close_col].to_numpy()
    returns = np.diff(prices) / prices[:-1]
    
    n = len(returns)
    entropy_values = np.full(n + 1, np.nan)  # +1 to match original length
    
    entropy_fn = sample_entropy if entropy_type == "sample" else approximate_entropy
    
    for i in range(window, n + 1):
        window_data = returns[i - window:i]
        entropy_values[i] = entropy_fn(window_data)
    
    return pl.Series("entropy", entropy_values)


# =============================================================================
# VPIN - Volume-Synchronized Probability of Informed Trading
# =============================================================================

def classify_volume_bulk(
    prices: np.ndarray,
    volumes: np.ndarray,
    method: str = "tick"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify volume as buy or sell using bulk classification.
    
    Methods:
    - "tick": Sign of price change
    - "lee_ready": Compare to mid-price (simplified)
    
    Args:
        prices: Price series
        volumes: Volume series
        method: Classification method
        
    Returns:
        Tuple of (buy_volume, sell_volume) arrays
    """
    n = len(prices)
    buy_vol = np.zeros(n)
    sell_vol = np.zeros(n)
    
    if method == "tick":
        # Tick rule: classify based on price direction
        for i in range(1, n):
            if prices[i] > prices[i - 1]:
                buy_vol[i] = volumes[i]
            elif prices[i] < prices[i - 1]:
                sell_vol[i] = volumes[i]
            else:
                # No change: split evenly
                buy_vol[i] = volumes[i] / 2
                sell_vol[i] = volumes[i] / 2
    else:
        # Default to tick rule
        return classify_volume_bulk(prices, volumes, method="tick")
    
    return buy_vol, sell_vol


def calculate_vpin(
    df: pl.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
    bucket_size: int = 50,
    n_buckets: int = 50
) -> pl.Series:
    """
    Calculate VPIN - Volume-Synchronized Probability of Informed Trading.
    
    VPIN measures order flow toxicity. High VPIN precedes market crashes
    and high volatility periods.
    
    Args:
        df: DataFrame with price and volume data
        close_col: Close price column
        volume_col: Volume column
        bucket_size: Volume per bucket
        n_buckets: Number of buckets for VPIN calculation
        
    Returns:
        VPIN series (0 to 1, higher = more toxic/informed trading)
        
    Reference:
        Easley, Lopez de Prado, O'Hara (2012)
        "Flow Toxicity and Liquidity in a High-Frequency World"
    """
    prices = df[close_col].to_numpy()
    volumes = df[volume_col].to_numpy()
    
    n = len(prices)
    
    # Classify volume
    buy_vol, sell_vol = classify_volume_bulk(prices, volumes)
    
    # Create volume buckets
    cumulative_vol = np.cumsum(volumes)
    total_vol = cumulative_vol[-1]
    
    if total_vol < bucket_size * n_buckets:
        # Not enough volume for meaningful VPIN
        return pl.Series("vpin", np.full(n, np.nan))
    
    # Simplified: use rolling window approach for bar-based data
    vpin_values = np.full(n, np.nan)
    
    window = n_buckets
    for i in range(window, n):
        window_buy = np.sum(buy_vol[i - window:i])
        window_sell = np.sum(sell_vol[i - window:i])
        total = window_buy + window_sell
        
        if total > 0:
            # VPIN = |Buy - Sell| / Total
            vpin_values[i] = abs(window_buy - window_sell) / total
    
    return pl.Series("vpin", vpin_values)


# =============================================================================
# VOLATILITY ESTIMATORS
# =============================================================================

def garman_klass_volatility(
    df: pl.DataFrame,
    window: int = 30,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close"
) -> pl.Series:
    """
    Garman-Klass Volatility Estimator.
    
    More efficient than close-to-close volatility by using OHLC data.
    Approximately 5x more efficient than simple close-to-close.
    
    Args:
        df: DataFrame with OHLC data
        window: Rolling window
        open_col, high_col, low_col, close_col: Column names
        
    Returns:
        Garman-Klass volatility series
        
    Reference:
        Garman & Klass (1980), "On the Estimation of Security Price Volatilities"
    """
    log_hl = (pl.col(high_col) / pl.col(low_col)).log()
    log_co = (pl.col(close_col) / pl.col(open_col)).log()
    
    # GK formula: 0.5 * (H-L)^2 - (2*ln(2) - 1) * (C-O)^2
    gk_term = 0.5 * log_hl.pow(2) - (2 * math.log(2) - 1) * log_co.pow(2)
    
    return gk_term.rolling_mean(window_size=window).sqrt().alias("vol_gk")


def parkinson_volatility(
    df: pl.DataFrame,
    window: int = 30,
    high_col: str = "high",
    low_col: str = "low"
) -> pl.Series:
    """
    Parkinson Volatility Estimator (High-Low range based).
    
    More efficient than close-to-close when drift is zero.
    
    Args:
        df: DataFrame with high/low data
        window: Rolling window
        
    Returns:
        Parkinson volatility series
    """
    log_hl_sq = (pl.col(high_col) / pl.col(low_col)).log().pow(2)
    
    # Parkinson formula: sqrt(1 / (4 * N * ln(2)) * sum(ln(H/L)^2))
    return (log_hl_sq.rolling_mean(window_size=window) / (4 * math.log(2))).sqrt().alias("vol_parkinson")


# =============================================================================
# SPREAD ESTIMATORS
# =============================================================================

def roll_spread(
    df: pl.DataFrame,
    window: int = 30,
    close_col: str = "close"
) -> pl.Series:
    """
    Roll Spread Estimator - Infers bid-ask spread from price autocovariance.
    
    Based on the insight that bid-ask bounce creates negative autocorrelation.
    
    Args:
        df: DataFrame with close prices
        window: Rolling window
        
    Returns:
        Estimated effective spread (in price units)
        
    Reference:
        Roll (1984), "A Simple Implicit Measure of the Effective Bid-Ask Spread"
    """
    # Roll spread = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
    # When cov is positive, spread is undefined (set to 0)
    
    delta_p = pl.col(close_col).diff()
    delta_p_lag = delta_p.shift(1)
    
    # Rolling covariance approximation
    # cov(X, Y) = E[XY] - E[X]E[Y]
    # For simplicity, use rolling mean of product
    
    # This is approximate - proper implementation would use rolling covariance
    cov_approx = (delta_p * delta_p_lag).rolling_mean(window_size=window)
    
    # spread = 2 * sqrt(max(0, -cov))
    return (2 * (-cov_approx).clip(lower_bound=0).sqrt()).alias("roll_spread")


def corwin_schultz_spread(
    df: pl.DataFrame,
    window: int = 1,  # Usually daily, so window=1 for each bar
    high_col: str = "high",
    low_col: str = "low"
) -> pl.Series:
    """
    Corwin-Schultz High-Low Spread Estimator.
    
    Uses the ratio of high-low ranges across adjacent periods
    to estimate the bid-ask spread.
    
    Args:
        df: DataFrame with high/low data
        window: Aggregation window (1 for bar-by-bar)
        
    Returns:
        Estimated spread as fraction of price
        
    Reference:
        Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask Spreads"
    """
    # Beta = E[sum of squared log(H/L) over 2 consecutive bars]
    log_hl = (pl.col(high_col) / pl.col(low_col)).log()
    log_hl_sq = log_hl.pow(2)
    
    # Gamma = log(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))^2
    h_max = pl.max_horizontal(pl.col(high_col), pl.col(high_col).shift(1))
    l_min = pl.min_horizontal(pl.col(low_col), pl.col(low_col).shift(1))
    gamma = (h_max / l_min).log().pow(2)
    
    # Beta = sum of individual log_hl_sq for 2 periods
    beta = log_hl_sq + log_hl_sq.shift(1)
    
    # Alpha calculation
    sqrt_2 = math.sqrt(2)
    k = (sqrt_2 - 1) / (3 - 2 * sqrt_2)
    
    alpha = (gamma.sqrt() - beta.sqrt()) / k - (gamma / k).sqrt()
    
    # Spread = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    exp_alpha = alpha.exp()
    spread = 2 * (exp_alpha - 1) / (1 + exp_alpha)
    
    # Clip negative values (estimation error)
    return spread.clip(lower_bound=0).alias("cs_spread")


# =============================================================================
# MARKET IMPACT / LIQUIDITY
# =============================================================================

def kyle_lambda(
    df: pl.DataFrame,
    window: int = 30,
    close_col: str = "close",
    volume_col: str = "volume"
) -> pl.Series:
    """
    Kyle's Lambda - Measures price impact per unit of volume.
    
    Higher lambda = lower liquidity = higher market impact.
    
    Estimated via regression: delta_P = lambda * sign(V) * sqrt(|V|) + epsilon
    
    Simplified version using rolling correlation.
    
    Args:
        df: DataFrame with price and volume data
        window: Rolling window
        
    Returns:
        Kyle's Lambda series
        
    Reference:
        Kyle (1985), "Continuous Auctions and Insider Trading"
    """
    # Simplified: lambda ≈ |return| / sqrt(volume)
    ret = pl.col(close_col).pct_change().abs()
    sqrt_vol = pl.col(volume_col).sqrt()
    
    # Avoid div by zero
    lambda_est = ret / (sqrt_vol + 1)
    
    return lambda_est.rolling_mean(window_size=window).alias("kyle_lambda")


def amihud_illiquidity_ratio(
    df: pl.DataFrame,
    window: int = 30,
    close_col: str = "close",
    volume_col: str = "volume"
) -> pl.Series:
    """
    Amihud Illiquidity Ratio - Price impact measure.
    
    ILLIQ = |Return| / Dollar Volume
    
    Higher = less liquid, more price impact.
    
    Args:
        df: DataFrame
        window: Rolling window
        
    Returns:
        Amihud illiquidity series
        
    Reference:
        Amihud (2002), "Illiquidity and stock returns"
    """
    ret = pl.col(close_col).pct_change().abs()
    dollar_vol = pl.col(close_col) * pl.col(volume_col)
    
    illiq = ret / (dollar_vol + 1e-9)
    
    return illiq.rolling_mean(window_size=window).alias("amihud")


# =============================================================================
# COMBINED FEATURE EXTRACTION
# =============================================================================

def get_advanced_features(
    df: pl.DataFrame,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    volume_col: str = "volume",
    window: int = 30
) -> pl.DataFrame:
    """
    Extract all advanced features at once.
    
    Returns DataFrame with:
    - Garman-Klass volatility
    - Parkinson volatility  
    - Roll spread
    - Corwin-Schultz spread
    - Kyle's Lambda
    - Amihud illiquidity
    
    Note: Entropy and VPIN are computationally expensive,
    call separately if needed.
    """
    df = df.with_columns([
        garman_klass_volatility(df, window, open_col, high_col, low_col, close_col),
        parkinson_volatility(df, window, high_col, low_col),
        roll_spread(df, window, close_col),
        corwin_schultz_spread(df, window=1, high_col=high_col, low_col=low_col),
        kyle_lambda(df, window, close_col, volume_col),
        amihud_illiquidity_ratio(df, window, close_col, volume_col),
    ])
    
    return df


# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

def classify_market_regime(
    entropy: float,
    volatility: float,
    vol_percentile: float = 0.5,
    entropy_percentile: float = 0.5,
    vol_threshold_low: float = 0.3,
    vol_threshold_high: float = 0.7,
    entropy_threshold_low: float = 0.3,
    entropy_threshold_high: float = 0.7
) -> str:
    """
    Classify market regime based on entropy and volatility.
    
    Returns one of:
    - "trending_low_vol": Low entropy + Low volatility → Strong momentum signals
    - "trending_high_vol": Low entropy + High volatility → Careful momentum  
    - "ranging_low_vol": High entropy + Low volatility → Mean reversion
    - "ranging_high_vol": High entropy + High volatility → Avoid/reduce exposure
    
    Args:
        entropy: Current entropy value
        volatility: Current volatility value
        vol_percentile: Volatility as percentile (0-1)
        entropy_percentile: Entropy as percentile (0-1)
        vol_threshold_low/high: Percentile thresholds
        entropy_threshold_low/high: Percentile thresholds
    """
    low_entropy = entropy_percentile < entropy_threshold_low
    high_entropy = entropy_percentile > entropy_threshold_high
    low_vol = vol_percentile < vol_threshold_low
    high_vol = vol_percentile > vol_threshold_high
    
    if low_entropy and low_vol:
        return "trending_low_vol"
    elif low_entropy and high_vol:
        return "trending_high_vol"
    elif high_entropy and low_vol:
        return "ranging_low_vol"
    elif high_entropy and high_vol:
        return "ranging_high_vol"
    else:
        return "neutral"
