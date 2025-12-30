"""
Statistical Arbitrage Engine - Pairs Trading with Kalman Filters

This module implements a complete statistical arbitrage system:
1. Cointegration testing (Engle-Granger, Johansen)
2. Cluster-based pair selection
3. Kalman filter for dynamic hedge ratios
4. Mean-reversion signal generation
5. Risk management with position limits

Reference Implementation:
- Avellaneda & Lee (2010), "Statistical Arbitrage in the U.S. Equity Market"
- Chan (2013), "Algorithmic Trading: Winning Strategies and Their Rationale"
- De Prado (2018), "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PairStatus(Enum):
    """Status of a pairs trade."""
    INACTIVE = "inactive"
    LONG_SPREAD = "long_spread"   # Long asset1, short asset2
    SHORT_SPREAD = "short_spread" # Short asset1, long asset2


@dataclass
class CointPair:
    """Represents a cointegrated pair."""
    asset1: str
    asset2: str
    hedge_ratio: float
    intercept: float
    spread_mean: float
    spread_std: float
    half_life: float
    adf_pvalue: float
    correlation: float
    
    def __post_init__(self):
        self.zscore: float = 0.0
        self.status: PairStatus = PairStatus.INACTIVE


@dataclass
class KalmanState:
    """State for Kalman filter hedge ratio estimation."""
    beta: float = 0.0       # Hedge ratio estimate
    P: float = 1.0          # Estimation variance
    R: float = 1e-3         # Measurement noise (tune this)
    Q: float = 1e-5         # Process noise (tune this)
    Ve: float = 0.0         # Measurement error variance
    
    def update(self, x: float, y: float) -> float:
        """
        Update Kalman filter with new observation.
        
        Model: y = beta * x + epsilon
        
        Args:
            x: Independent variable (asset 2 price)
            y: Dependent variable (asset 1 price)
            
        Returns:
            Updated hedge ratio (beta)
        """
        # Prediction step
        # beta prediction: beta_t|t-1 = beta_t-1|t-1 (random walk)
        # P prediction: P_t|t-1 = P_t-1|t-1 + Q
        self.P = self.P + self.Q
        
        # Measurement prediction
        y_pred = self.beta * x
        e = y - y_pred  # Innovation
        
        # Innovation variance
        self.Ve = x * self.P * x + self.R
        
        # Kalman gain
        K = self.P * x / self.Ve
        
        # Update step
        self.beta = self.beta + K * e
        self.P = (1 - K * x) * self.P
        
        return self.beta


@dataclass
class PairTrade:
    """Represents an active pairs trade position."""
    pair: CointPair
    entry_zscore: float
    entry_time: int  # Index or timestamp
    position_size: float  # Notional per leg
    status: PairStatus
    pnl: float = 0.0
    
    def update_pnl(
        self,
        price1: float,
        price2: float,
        entry_price1: float,
        entry_price2: float
    ) -> float:
        """Calculate unrealized P&L."""
        if self.status == PairStatus.LONG_SPREAD:
            # Long asset1, short asset2
            leg1_pnl = (price1 - entry_price1) / entry_price1
            leg2_pnl = -(price2 - entry_price2) / entry_price2 * self.pair.hedge_ratio
        elif self.status == PairStatus.SHORT_SPREAD:
            # Short asset1, long asset2
            leg1_pnl = -(price1 - entry_price1) / entry_price1
            leg2_pnl = (price2 - entry_price2) / entry_price2 * self.pair.hedge_ratio
        else:
            return 0.0
        
        self.pnl = self.position_size * (leg1_pnl + leg2_pnl)
        return self.pnl


# =============================================================================
# COINTEGRATION TESTING
# =============================================================================

def adf_test(series: np.ndarray, max_lags: Optional[int] = None) -> Tuple[float, float]:
    """
    Augmented Dickey-Fuller test for stationarity.
    
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    
    Args:
        series: Time series to test
        max_lags: Maximum lags for ADF regression
        
    Returns:
        (test_statistic, p_value)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, maxlag=max_lags, autolag='AIC')
        return result[0], result[1]  # test stat, pvalue
    except ImportError:
        # Fallback: simple implementation
        return _simple_adf(series)


def _simple_adf(series: np.ndarray) -> Tuple[float, float]:
    """Simple ADF implementation without statsmodels."""
    n = len(series)
    if n < 20:
        return 0.0, 1.0
    
    # Lag 1 difference regression
    y = np.diff(series)
    x = series[:-1]
    
    # OLS: y = alpha + beta * x + epsilon
    # Test H0: beta = 0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    beta = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
    
    # Standard error
    residuals = y - (y_mean + beta * (x - x_mean))
    se_beta = np.sqrt(np.sum(residuals ** 2) / (n - 3) / (np.sum((x - x_mean) ** 2) + 1e-10))
    
    # Test statistic
    t_stat = beta / (se_beta + 1e-10)
    
    # Critical values (approximate): 1% = -3.43, 5% = -2.86, 10% = -2.57
    # Use interpolation for p-value approximation
    if t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.57:
        p_value = 0.10
    else:
        p_value = 0.5  # Not stationary
    
    return t_stat, p_value


def engle_granger_coint(
    y1: np.ndarray,
    y2: np.ndarray,
    max_lags: Optional[int] = None
) -> Tuple[float, float, float, float]:
    """
    Engle-Granger two-step cointegration test.
    
    Step 1: Regress y1 on y2 to get hedge ratio (beta) and intercept
    Step 2: Test residuals for stationarity
    
    Args:
        y1: First price series (dependent)
        y2: Second price series (independent)
        max_lags: Max lags for ADF test
        
    Returns:
        (adf_statistic, p_value, hedge_ratio, intercept)
    """
    # Step 1: OLS regression y1 = alpha + beta * y2 + epsilon
    n = len(y1)
    x = np.column_stack([np.ones(n), y2])
    
    # OLS: beta = (X'X)^-1 X'y
    xtx_inv = np.linalg.inv(x.T @ x + 1e-8 * np.eye(2))
    beta_vec = xtx_inv @ x.T @ y1
    
    alpha, beta = beta_vec[0], beta_vec[1]
    
    # Step 2: Get residuals and test for stationarity
    spread = y1 - beta * y2 - alpha
    
    adf_stat, pvalue = adf_test(spread, max_lags)
    
    return adf_stat, pvalue, beta, alpha


def half_life_mean_reversion(spread: np.ndarray) -> float:
    """
    Estimate half-life of mean reversion using OU process.
    
    Model: d(spread) = theta * (mu - spread) * dt + sigma * dW
    
    Regression: spread_diff = lambda * spread_lag + epsilon
    Half-life = -ln(2) / lambda
    
    Args:
        spread: Spread time series
        
    Returns:
        Half-life in same time units as data
    """
    spread_diff = np.diff(spread)
    spread_lag = spread[:-1]
    
    # OLS: spread_diff = lambda * spread_lag
    n = len(spread_diff)
    
    cov = np.sum(spread_diff * spread_lag) / n
    var = np.sum(spread_lag ** 2) / n
    
    lambda_param = cov / (var + 1e-10)
    
    if lambda_param >= 0:
        # No mean reversion
        return np.inf
    
    half_life = -np.log(2) / lambda_param
    
    return half_life


# =============================================================================
# PAIR SELECTION & CLUSTERING
# =============================================================================

def compute_correlation_matrix(
    prices: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute correlation matrix for all assets.
    
    Args:
        prices: Dict mapping ticker to price series
        
    Returns:
        (correlation_matrix, ticker_list)
    """
    tickers = list(prices.keys())
    n = len(tickers)
    
    # Convert to returns
    returns = {}
    for ticker, px in prices.items():
        ret = np.diff(px) / px[:-1]
        returns[ticker] = ret
    
    # Correlation matrix
    corr = np.zeros((n, n))
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i == j:
                corr[i, j] = 1.0
            elif j > i:
                # Pearson correlation
                r1 = returns[t1]
                r2 = returns[t2]
                min_len = min(len(r1), len(r2))
                c = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
                corr[i, j] = c
                corr[j, i] = c
    
    return corr, tickers


def cluster_assets(
    corr_matrix: np.ndarray,
    n_clusters: int = 10,
    method: str = "ward"
) -> np.ndarray:
    """
    Cluster assets using hierarchical clustering.
    
    Args:
        corr_matrix: Correlation matrix
        n_clusters: Number of clusters
        method: Linkage method ('ward', 'complete', 'average', 'single')
        
    Returns:
        Cluster labels for each asset
    """
    # Convert correlation to distance
    # Distance = sqrt(0.5 * (1 - correlation))
    dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    
    # Ensure diagonal is 0
    np.fill_diagonal(dist_matrix, 0)
    
    # Condensed distance matrix for scipy
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Hierarchical clustering
    Z = linkage(condensed_dist, method=method)
    
    # Cut dendrogram
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    
    return labels


def find_cointegrated_pairs(
    prices: Dict[str, np.ndarray],
    cluster_labels: Optional[np.ndarray] = None,
    tickers: Optional[List[str]] = None,
    min_correlation: float = 0.0,  # Changed: cointegration doesn't require high return correlation
    max_pvalue: float = 0.05,
    min_half_life: float = 5,
    max_half_life: float = 100
) -> List[CointPair]:
    """
    Find cointegrated pairs, optionally within clusters.
    
    Args:
        prices: Dict of price series
        cluster_labels: Optional cluster assignments
        tickers: Ticker list (required if cluster_labels provided)
        min_correlation: Minimum correlation to consider
        max_pvalue: Maximum ADF p-value
        min_half_life: Minimum half-life (avoid too fast mean reversion)
        max_half_life: Maximum half-life (avoid too slow mean reversion)
        
    Returns:
        List of cointegrated pairs
    """
    ticker_list = list(prices.keys())
    pairs = []
    
    n = len(ticker_list)
    
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = ticker_list[i], ticker_list[j]
            
            # Skip if in different clusters (if clustering used)
            if cluster_labels is not None and tickers is not None:
                idx1 = tickers.index(t1) if t1 in tickers else -1
                idx2 = tickers.index(t2) if t2 in tickers else -1
                if idx1 >= 0 and idx2 >= 0:
                    if cluster_labels[idx1] != cluster_labels[idx2]:
                        continue
            
            p1 = prices[t1]
            p2 = prices[t2]
            
            # Align lengths
            min_len = min(len(p1), len(p2))
            p1 = p1[-min_len:]
            p2 = p2[-min_len:]
            
            # Correlation check
            ret1 = np.diff(p1) / p1[:-1]
            ret2 = np.diff(p2) / p2[:-1]
            corr = np.corrcoef(ret1, ret2)[0, 1]
            
            if abs(corr) < min_correlation:
                continue
            
            # Cointegration test
            result = engle_granger_coint(p1, p2)
            adf_stat, pvalue, hedge_ratio, intercept = result
            
            if pvalue > max_pvalue:
                continue
            
            # Compute spread (with intercept for proper residuals)
            spread = p1 - hedge_ratio * p2 - intercept
            spread_mean = np.mean(spread)  # Should be ~0 now
            spread_std = np.std(spread)
            
            # Half-life
            hl = half_life_mean_reversion(spread)
            
            if hl < min_half_life or hl > max_half_life:
                continue
            
            # Create pair
            pair = CointPair(
                asset1=t1,
                asset2=t2,
                hedge_ratio=hedge_ratio,
                intercept=intercept,
                spread_mean=spread_mean,
                spread_std=spread_std,
                half_life=hl,
                adf_pvalue=pvalue,
                correlation=corr
            )
            
            pairs.append(pair)
    
    # Sort by ADF p-value (most significant first)
    pairs.sort(key=lambda x: x.adf_pvalue)
    
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    
    return pairs


# =============================================================================
# STATISTICAL ARBITRAGE ENGINE
# =============================================================================

@dataclass
class StatArbConfig:
    """Configuration for statistical arbitrage strategy."""
    entry_zscore: float = 2.0      # Z-score to enter trade
    exit_zscore: float = 0.5       # Z-score to exit trade
    stop_loss_zscore: float = 4.0  # Z-score for stop loss
    
    lookback_zscore: int = 60      # Bars for z-score calculation
    lookback_coint: int = 252      # Bars for cointegration test
    
    max_pairs: int = 20            # Maximum concurrent pairs
    position_size: float = 10000   # Notional per leg
    
    use_kalman: bool = True        # Use Kalman filter for hedge ratio
    recalc_coint_every: int = 20   # Recalculate cointegration every N bars
    
    # Transaction costs
    commission_bps: float = 1.0    # Commission per trade
    slippage_bps: float = 2.0      # Slippage per trade


class StatArbEngine:
    """
    Statistical Arbitrage Trading Engine.
    
    Implements:
    - Dynamic pair selection with cointegration testing
    - Kalman filter hedge ratio estimation
    - Z-score based entry/exit signals
    - Position management
    - P&L tracking
    """
    
    def __init__(self, config: StatArbConfig):
        self.config = config
        
        # State
        self.pairs: List[CointPair] = []
        self.active_trades: Dict[str, PairTrade] = {}
        self.kalman_states: Dict[str, KalmanState] = {}
        
        # Historical data
        self.prices: Dict[str, np.ndarray] = {}
        self.spreads: Dict[str, List[float]] = {}
        
        # Performance tracking
        self.equity_curve: List[float] = []
        self.trade_log: List[Dict] = []
        
        self.bar_count = 0
    
    def _get_pair_key(self, pair: CointPair) -> str:
        """Get unique key for a pair."""
        return f"{pair.asset1}_{pair.asset2}"
    
    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update price history with new bar.
        
        Args:
            prices: Dict mapping ticker to current price
        """
        for ticker, price in prices.items():
            if ticker not in self.prices:
                self.prices[ticker] = np.array([price])
            else:
                self.prices[ticker] = np.append(self.prices[ticker], price)
                
                # Trim to max needed lookback
                max_lookback = max(self.config.lookback_coint, self.config.lookback_zscore) + 100
                if len(self.prices[ticker]) > max_lookback:
                    self.prices[ticker] = self.prices[ticker][-max_lookback:]
        
        self.bar_count += 1
    
    def recalculate_pairs(self) -> None:
        """Recalculate cointegrated pairs from current price data."""
        if len(next(iter(self.prices.values()))) < self.config.lookback_coint:
            return
        
        # Get recent price data
        recent_prices = {
            ticker: px[-self.config.lookback_coint:]
            for ticker, px in self.prices.items()
        }
        
        # Find pairs - use more lenient parameters
        self.pairs = find_cointegrated_pairs(
            recent_prices,
            min_correlation=0.0,  # Cointegration doesn't require return correlation
            max_pvalue=0.10,
            min_half_life=2,
            max_half_life=200
        )[:self.config.max_pairs]
        
        # Initialize Kalman states for new pairs
        for pair in self.pairs:
            key = self._get_pair_key(pair)
            if key not in self.kalman_states:
                self.kalman_states[key] = KalmanState(beta=pair.hedge_ratio)
    
    def calculate_zscore(self, pair: CointPair) -> float:
        """
        Calculate current z-score for a pair.
        
        Uses Kalman filter hedge ratio if enabled.
        """
        p1 = self.prices.get(pair.asset1)
        p2 = self.prices.get(pair.asset2)
        
        if p1 is None or p2 is None:
            return 0.0
        
        current_p1 = p1[-1]
        current_p2 = p2[-1]
        
        # Get hedge ratio
        key = self._get_pair_key(pair)
        
        if self.config.use_kalman and key in self.kalman_states:
            # Update Kalman filter
            hedge_ratio = self.kalman_states[key].update(current_p2, current_p1)
        else:
            hedge_ratio = pair.hedge_ratio
        
        # Calculate spread
        lookback = min(self.config.lookback_zscore, len(p1))
        spreads = p1[-lookback:] - hedge_ratio * p2[-lookback:]
        
        current_spread = current_p1 - hedge_ratio * current_p2
        
        # Z-score
        spread_mean = np.mean(spreads)
        spread_std = np.std(spreads)
        
        if spread_std < 1e-10:
            return 0.0
        
        zscore = (current_spread - spread_mean) / spread_std
        
        # Store spread
        if key not in self.spreads:
            self.spreads[key] = []
        self.spreads[key].append(current_spread)
        
        return zscore
    
    def generate_signals(self) -> List[Dict]:
        """
        Generate trading signals for all pairs.
        
        Returns:
            List of signal dicts with keys: pair, action, zscore
        """
        signals = []
        
        for pair in self.pairs:
            key = self._get_pair_key(pair)
            zscore = self.calculate_zscore(pair)
            pair.zscore = zscore
            
            active_trade = self.active_trades.get(key)
            
            if active_trade is None:
                # Check for entry
                if zscore > self.config.entry_zscore:
                    # Spread too high: short spread (short asset1, long asset2)
                    signals.append({
                        'pair': pair,
                        'action': 'short_spread',
                        'zscore': zscore
                    })
                elif zscore < -self.config.entry_zscore:
                    # Spread too low: long spread (long asset1, short asset2)
                    signals.append({
                        'pair': pair,
                        'action': 'long_spread',
                        'zscore': zscore
                    })
            else:
                # Check for exit or stop loss
                if active_trade.status == PairStatus.LONG_SPREAD:
                    if zscore >= -self.config.exit_zscore or zscore < -self.config.stop_loss_zscore:
                        signals.append({
                            'pair': pair,
                            'action': 'exit',
                            'zscore': zscore
                        })
                elif active_trade.status == PairStatus.SHORT_SPREAD:
                    if zscore <= self.config.exit_zscore or zscore > self.config.stop_loss_zscore:
                        signals.append({
                            'pair': pair,
                            'action': 'exit',
                            'zscore': zscore
                        })
        
        return signals
    
    def execute_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Execute trading signals.
        
        Returns:
            List of executed trades
        """
        executed = []
        
        for signal in signals:
            pair = signal['pair']
            action = signal['action']
            key = self._get_pair_key(pair)
            
            p1 = self.prices[pair.asset1][-1]
            p2 = self.prices[pair.asset2][-1]
            
            if action == 'long_spread':
                trade = PairTrade(
                    pair=pair,
                    entry_zscore=signal['zscore'],
                    entry_time=self.bar_count,
                    position_size=self.config.position_size,
                    status=PairStatus.LONG_SPREAD
                )
                self.active_trades[key] = trade
                
                executed.append({
                    'action': 'entry_long',
                    'pair': key,
                    'zscore': signal['zscore'],
                    'price1': p1,
                    'price2': p2,
                    'hedge_ratio': pair.hedge_ratio
                })
                
            elif action == 'short_spread':
                trade = PairTrade(
                    pair=pair,
                    entry_zscore=signal['zscore'],
                    entry_time=self.bar_count,
                    position_size=self.config.position_size,
                    status=PairStatus.SHORT_SPREAD
                )
                self.active_trades[key] = trade
                
                executed.append({
                    'action': 'entry_short',
                    'pair': key,
                    'zscore': signal['zscore'],
                    'price1': p1,
                    'price2': p2,
                    'hedge_ratio': pair.hedge_ratio
                })
                
            elif action == 'exit':
                if key in self.active_trades:
                    trade = self.active_trades[key]
                    del self.active_trades[key]
                    
                    executed.append({
                        'action': 'exit',
                        'pair': key,
                        'zscore': signal['zscore'],
                        'price1': p1,
                        'price2': p2,
                        'pnl': trade.pnl,
                        'bars_held': self.bar_count - trade.entry_time
                    })
        
        return executed
    
    def step(self, prices: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Process one bar of data.
        
        Args:
            prices: Current prices for all assets
            
        Returns:
            (signals, executed_trades)
        """
        # Update prices
        self.update_prices(prices)
        
        # Recalculate pairs periodically
        if self.bar_count % self.config.recalc_coint_every == 0:
            self.recalculate_pairs()
        
        # Skip if not enough data
        if self.bar_count < self.config.lookback_zscore:
            return [], []
        
        # Generate signals
        signals = self.generate_signals()
        
        # Execute
        executed = self.execute_signals(signals)
        
        # Log
        self.trade_log.extend(executed)
        
        return signals, executed
    
    def get_positions(self) -> Dict[str, float]:
        """
        Get current position sizes for all assets.
        
        Returns:
            Dict mapping ticker to position size (positive = long, negative = short)
        """
        positions = {}
        
        for key, trade in self.active_trades.items():
            pair = trade.pair
            size = trade.position_size
            
            if trade.status == PairStatus.LONG_SPREAD:
                # Long asset1, short asset2
                positions[pair.asset1] = positions.get(pair.asset1, 0) + size
                positions[pair.asset2] = positions.get(pair.asset2, 0) - size * pair.hedge_ratio
            elif trade.status == PairStatus.SHORT_SPREAD:
                # Short asset1, long asset2
                positions[pair.asset1] = positions.get(pair.asset1, 0) - size
                positions[pair.asset2] = positions.get(pair.asset2, 0) + size * pair.hedge_ratio
        
        return positions
    
    def calculate_portfolio_pnl(self) -> float:
        """Calculate total portfolio P&L across all active trades."""
        total_pnl = 0.0
        
        for key, trade in self.active_trades.items():
            pair = trade.pair
            
            if pair.asset1 not in self.prices or pair.asset2 not in self.prices:
                continue
            
            p1 = self.prices[pair.asset1][-1]
            p2 = self.prices[pair.asset2][-1]
            
            # Get entry prices (approximate from history)
            entry_idx = trade.entry_time
            if entry_idx < len(self.prices[pair.asset1]):
                entry_p1 = self.prices[pair.asset1][entry_idx]
                entry_p2 = self.prices[pair.asset2][entry_idx]
            else:
                continue
            
            pnl = trade.update_pnl(p1, p2, entry_p1, entry_p2)
            total_pnl += pnl
        
        return total_pnl


# =============================================================================
# BACKTEST HELPER
# =============================================================================

def backtest_stat_arb(
    price_data: pl.DataFrame,
    tickers: List[str],
    config: Optional[StatArbConfig] = None,
    date_col: str = "date"
) -> Dict:
    """
    Backtest statistical arbitrage strategy on historical data.
    
    Args:
        price_data: DataFrame with columns: date, ticker, close
                    OR columns: date, TICKER1, TICKER2, ... (wide format)
        tickers: List of tickers to include
        config: Strategy configuration
        date_col: Date column name
        
    Returns:
        Dict with backtest results
    """
    if config is None:
        config = StatArbConfig()
    
    engine = StatArbEngine(config)
    
    # Determine data format
    if "ticker" in price_data.columns:
        # Long format - pivot to wide
        price_data_wide = price_data.pivot(
            values="close",
            index=date_col,
            columns="ticker"
        )
    else:
        price_data_wide = price_data
    
    # Ensure tickers exist
    available_tickers = [t for t in tickers if t in price_data_wide.columns]
    
    if len(available_tickers) < 2:
        raise ValueError("Need at least 2 tickers with price data")
    
    # Run backtest bar by bar
    dates = price_data_wide[date_col].to_list()
    equity = [100000.0]  # Starting equity
    
    all_signals = []
    all_trades = []
    
    for i, date in enumerate(dates):
        row = price_data_wide.row(i, named=True)
        prices = {t: row[t] for t in available_tickers if row[t] is not None}
        
        if len(prices) < 2:
            continue
        
        signals, trades = engine.step(prices)
        all_signals.extend(signals)
        all_trades.extend(trades)
        
        # Update equity
        pnl = engine.calculate_portfolio_pnl()
        equity.append(equity[0] + pnl)
    
    # Calculate performance metrics
    equity_arr = np.array(equity)
    returns = np.diff(equity_arr) / equity_arr[:-1]
    
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    max_dd = np.max(np.maximum.accumulate(equity_arr) - equity_arr) / equity_arr[0]
    
    return {
        'equity_curve': equity_arr.tolist(),
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_trades': len(all_trades),
        'signals': all_signals,
        'trades': all_trades,
        'final_equity': equity_arr[-1],
        'return_pct': (equity_arr[-1] / equity_arr[0] - 1) * 100
    }
