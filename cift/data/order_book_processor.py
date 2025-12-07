"""
CIFT Markets - Order Book Processor

Transforms raw L2/L3 order book data into ML-ready features.

Features Computed:
- Order book imbalance (multiple levels)
- Spread analytics (absolute, relative, volatility)
- Volume pressure indicators
- Price level clustering
- Arrival rate estimation (for Hawkes)
- Microstructure metrics

Performance:
- Numba JIT compilation for hot paths
- Rolling window calculations with minimal memory
- Sub-millisecond feature extraction
"""

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

# Try to import numba for JIT compilation
try:
    from numba import jit, float64, int64
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, falling back to pure Python")
    # Create no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class OrderBookSnapshot:
    """
    Point-in-time snapshot of order book state with computed features.
    """
    symbol: str
    timestamp: int  # Nanoseconds
    
    # Best bid/ask
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    
    # Computed features
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    
    # Imbalance metrics
    imbalance_l1: float = 0.0      # Top of book imbalance
    imbalance_l5: float = 0.0      # Top 5 levels
    imbalance_l10: float = 0.0     # Top 10 levels
    weighted_imbalance: float = 0.0 # Distance-weighted
    
    # Volume metrics
    bid_depth_5: int = 0           # Total bid volume top 5
    ask_depth_5: int = 0           # Total ask volume top 5
    bid_depth_10: int = 0
    ask_depth_10: int = 0
    
    # Pressure metrics
    bid_pressure: float = 0.0      # Rate of bid additions
    ask_pressure: float = 0.0      # Rate of ask additions
    net_pressure: float = 0.0      # bid_pressure - ask_pressure
    
    # Microstructure
    trade_flow: float = 0.0        # Signed trade volume
    vpin: float = 0.0              # Volume-synchronized probability of informed trading
    kyle_lambda: float = 0.0       # Kyle's lambda (price impact)
    
    # Arrival rates (for Hawkes)
    bid_arrival_rate: float = 0.0  # Orders per second
    ask_arrival_rate: float = 0.0
    trade_arrival_rate: float = 0.0
    cancel_rate: float = 0.0
    
    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1e9, tz=timezone.utc)


@dataclass
class TradeFlow:
    """Aggregated trade flow data."""
    symbol: str
    timestamp: int
    buy_volume: int = 0
    sell_volume: int = 0
    buy_count: int = 0
    sell_count: int = 0
    vwap_buy: float = 0.0
    vwap_sell: float = 0.0
    
    @property
    def net_volume(self) -> int:
        return self.buy_volume - self.sell_volume
    
    @property
    def imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total


# ============================================================================
# JIT-COMPILED FUNCTIONS
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def compute_imbalance_numba(
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
        levels: int
    ) -> float:
        """Compute order book imbalance for given levels."""
        bid_sum = 0.0
        ask_sum = 0.0
        
        for i in range(min(levels, len(bid_sizes))):
            bid_sum += bid_sizes[i]
        
        for i in range(min(levels, len(ask_sizes))):
            ask_sum += ask_sizes[i]
        
        total = bid_sum + ask_sum
        if total == 0:
            return 0.0
        
        return (bid_sum - ask_sum) / total
    
    @jit(nopython=True, cache=True)
    def compute_weighted_imbalance_numba(
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
        mid_price: float,
        levels: int
    ) -> float:
        """
        Compute distance-weighted imbalance.
        
        Orders closer to mid price are weighted higher.
        """
        bid_weighted = 0.0
        ask_weighted = 0.0
        
        for i in range(min(levels, len(bid_prices))):
            if bid_prices[i] > 0:
                distance = mid_price - bid_prices[i]
                if distance > 0:
                    weight = 1.0 / (1.0 + distance)
                    bid_weighted += bid_sizes[i] * weight
        
        for i in range(min(levels, len(ask_prices))):
            if ask_prices[i] > 0:
                distance = ask_prices[i] - mid_price
                if distance > 0:
                    weight = 1.0 / (1.0 + distance)
                    ask_weighted += ask_sizes[i] * weight
        
        total = bid_weighted + ask_weighted
        if total == 0:
            return 0.0
        
        return (bid_weighted - ask_weighted) / total
    
    @jit(nopython=True, cache=True)
    def compute_arrival_rate_numba(
        timestamps: np.ndarray,
        current_time: int,
        window_ns: int
    ) -> float:
        """Compute event arrival rate in events per second."""
        count = 0
        cutoff = current_time - window_ns
        
        for i in range(len(timestamps)):
            if timestamps[i] >= cutoff:
                count += 1
        
        window_seconds = window_ns / 1e9
        return count / window_seconds if window_seconds > 0 else 0.0
    
    @jit(nopython=True, cache=True)
    def compute_vpin_numba(
        buy_volumes: np.ndarray,
        sell_volumes: np.ndarray,
        bucket_volume: float
    ) -> float:
        """
        Compute Volume-Synchronized Probability of Informed Trading (VPIN).
        
        VPIN = sum(|V_buy - V_sell|) / (n * bucket_volume)
        """
        n = len(buy_volumes)
        if n == 0:
            return 0.0
        
        total_imbalance = 0.0
        for i in range(n):
            total_imbalance += abs(buy_volumes[i] - sell_volumes[i])
        
        denominator = n * bucket_volume
        if denominator == 0:
            return 0.0
        
        return total_imbalance / denominator

else:
    # Pure Python fallbacks
    def compute_imbalance_numba(bid_sizes, ask_sizes, levels):
        bid_sum = sum(bid_sizes[:levels])
        ask_sum = sum(ask_sizes[:levels])
        total = bid_sum + ask_sum
        return (bid_sum - ask_sum) / total if total > 0 else 0.0
    
    def compute_weighted_imbalance_numba(bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price, levels):
        bid_weighted = sum(
            size / (1 + mid_price - price)
            for price, size in zip(bid_prices[:levels], bid_sizes[:levels])
            if price > 0 and mid_price > price
        )
        ask_weighted = sum(
            size / (1 + price - mid_price)
            for price, size in zip(ask_prices[:levels], ask_sizes[:levels])
            if price > 0 and price > mid_price
        )
        total = bid_weighted + ask_weighted
        return (bid_weighted - ask_weighted) / total if total > 0 else 0.0
    
    def compute_arrival_rate_numba(timestamps, current_time, window_ns):
        cutoff = current_time - window_ns
        count = sum(1 for t in timestamps if t >= cutoff)
        window_seconds = window_ns / 1e9
        return count / window_seconds if window_seconds > 0 else 0.0
    
    def compute_vpin_numba(buy_volumes, sell_volumes, bucket_volume):
        n = len(buy_volumes)
        if n == 0:
            return 0.0
        total_imbalance = sum(abs(b - s) for b, s in zip(buy_volumes, sell_volumes))
        denominator = n * bucket_volume
        return total_imbalance / denominator if denominator > 0 else 0.0


# ============================================================================
# ORDER BOOK PROCESSOR
# ============================================================================

class OrderBookProcessor:
    """
    Transforms raw order book data into ML features.
    
    Maintains rolling windows for:
    - Event timestamps (for arrival rate)
    - Trade flow (for signed volume)
    - Price changes (for volatility)
    - Order book snapshots (for time-series features)
    """
    
    def __init__(
        self,
        symbol: str,
        window_seconds: float = 1.0,
        max_levels: int = 20,
        vpin_buckets: int = 50,
        vpin_bucket_volume: float = 10000,
    ):
        """
        Initialize processor.
        
        Args:
            symbol: Symbol to process
            window_seconds: Rolling window for rate calculations
            max_levels: Max order book levels to track
            vpin_buckets: Number of buckets for VPIN calculation
            vpin_bucket_volume: Volume per VPIN bucket
        """
        self.symbol = symbol
        self.window_ns = int(window_seconds * 1e9)
        self.max_levels = max_levels
        self.vpin_buckets = vpin_buckets
        self.vpin_bucket_volume = vpin_bucket_volume
        
        # Current order book state
        self.bid_prices = np.zeros(max_levels, dtype=np.float64)
        self.bid_sizes = np.zeros(max_levels, dtype=np.int64)
        self.ask_prices = np.zeros(max_levels, dtype=np.float64)
        self.ask_sizes = np.zeros(max_levels, dtype=np.int64)
        
        # Rolling windows
        self._bid_timestamps: Deque[int] = deque(maxlen=10000)
        self._ask_timestamps: Deque[int] = deque(maxlen=10000)
        self._trade_timestamps: Deque[int] = deque(maxlen=10000)
        self._cancel_timestamps: Deque[int] = deque(maxlen=10000)
        
        # Trade flow tracking
        self._trade_signs: Deque[int] = deque(maxlen=1000)  # +1 buy, -1 sell
        self._trade_sizes: Deque[int] = deque(maxlen=1000)
        self._trade_prices: Deque[float] = deque(maxlen=1000)
        
        # VPIN buckets
        self._vpin_buy_volumes: Deque[float] = deque(maxlen=vpin_buckets)
        self._vpin_sell_volumes: Deque[float] = deque(maxlen=vpin_buckets)
        self._current_bucket_buy: float = 0.0
        self._current_bucket_sell: float = 0.0
        self._current_bucket_total: float = 0.0
        
        # Kyle's lambda calculation
        self._price_changes: Deque[float] = deque(maxlen=100)
        self._signed_volumes: Deque[float] = deque(maxlen=100)
        
        # Last snapshot
        self._last_snapshot: Optional[OrderBookSnapshot] = None
        self._last_mid_price: float = 0.0
        
        logger.info(f"OrderBookProcessor initialized for {symbol}")
    
    def update_book(
        self,
        bid_prices: List[float],
        bid_sizes: List[int],
        ask_prices: List[float],
        ask_sizes: List[int],
        timestamp: int,
    ) -> OrderBookSnapshot:
        """
        Update order book state and compute features.
        
        Args:
            bid_prices: Bid prices (best first)
            bid_sizes: Bid sizes
            ask_prices: Ask prices (best first)
            ask_sizes: Ask sizes
            timestamp: Update timestamp (nanoseconds)
            
        Returns:
            OrderBookSnapshot with computed features
        """
        # Update internal arrays
        n_bids = min(len(bid_prices), self.max_levels)
        n_asks = min(len(ask_prices), self.max_levels)
        
        self.bid_prices[:n_bids] = bid_prices[:n_bids]
        self.bid_sizes[:n_bids] = bid_sizes[:n_bids]
        self.ask_prices[:n_asks] = ask_prices[:n_asks]
        self.ask_sizes[:n_asks] = ask_sizes[:n_asks]
        
        # Zero out unused levels
        if n_bids < self.max_levels:
            self.bid_prices[n_bids:] = 0
            self.bid_sizes[n_bids:] = 0
        if n_asks < self.max_levels:
            self.ask_prices[n_asks:] = 0
            self.ask_sizes[n_asks:] = 0
        
        # Compute snapshot
        return self._compute_snapshot(timestamp)
    
    def record_order(self, is_bid: bool, timestamp: int):
        """Record order arrival for rate calculation."""
        if is_bid:
            self._bid_timestamps.append(timestamp)
        else:
            self._ask_timestamps.append(timestamp)
    
    def record_cancel(self, timestamp: int):
        """Record order cancellation."""
        self._cancel_timestamps.append(timestamp)
    
    def record_trade(
        self,
        price: float,
        size: int,
        is_buy: bool,
        timestamp: int,
    ):
        """
        Record trade for flow analysis.
        
        Args:
            price: Trade price
            size: Trade size
            is_buy: True if buyer-initiated
            timestamp: Trade timestamp
        """
        self._trade_timestamps.append(timestamp)
        
        sign = 1 if is_buy else -1
        self._trade_signs.append(sign)
        self._trade_sizes.append(size)
        self._trade_prices.append(price)
        
        # Update VPIN buckets
        if is_buy:
            self._current_bucket_buy += size
        else:
            self._current_bucket_sell += size
        self._current_bucket_total += size
        
        # Check if bucket is full
        if self._current_bucket_total >= self.vpin_bucket_volume:
            self._vpin_buy_volumes.append(self._current_bucket_buy)
            self._vpin_sell_volumes.append(self._current_bucket_sell)
            self._current_bucket_buy = 0.0
            self._current_bucket_sell = 0.0
            self._current_bucket_total = 0.0
        
        # Update Kyle's lambda data
        if self._last_mid_price > 0:
            mid = (self.bid_prices[0] + self.ask_prices[0]) / 2 if self.bid_prices[0] > 0 else price
            price_change = mid - self._last_mid_price
            signed_volume = sign * size
            self._price_changes.append(price_change)
            self._signed_volumes.append(signed_volume)
    
    def _compute_snapshot(self, timestamp: int) -> OrderBookSnapshot:
        """Compute full feature snapshot."""
        # Basic prices
        best_bid = self.bid_prices[0] if self.bid_prices[0] > 0 else 0.0
        best_ask = self.ask_prices[0] if self.ask_prices[0] > 0 else 0.0
        best_bid_size = int(self.bid_sizes[0])
        best_ask_size = int(self.ask_sizes[0])
        
        mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0.0
        spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0.0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0.0
        
        # Imbalances at different levels
        imbalance_l1 = compute_imbalance_numba(self.bid_sizes, self.ask_sizes, 1)
        imbalance_l5 = compute_imbalance_numba(self.bid_sizes, self.ask_sizes, 5)
        imbalance_l10 = compute_imbalance_numba(self.bid_sizes, self.ask_sizes, 10)
        
        weighted_imbalance = compute_weighted_imbalance_numba(
            self.bid_prices, self.bid_sizes,
            self.ask_prices, self.ask_sizes,
            mid_price, 10
        )
        
        # Depth metrics
        bid_depth_5 = int(np.sum(self.bid_sizes[:5]))
        ask_depth_5 = int(np.sum(self.ask_sizes[:5]))
        bid_depth_10 = int(np.sum(self.bid_sizes[:10]))
        ask_depth_10 = int(np.sum(self.ask_sizes[:10]))
        
        # Arrival rates
        bid_arr_ts = np.array(list(self._bid_timestamps), dtype=np.int64)
        ask_arr_ts = np.array(list(self._ask_timestamps), dtype=np.int64)
        trade_arr_ts = np.array(list(self._trade_timestamps), dtype=np.int64)
        cancel_arr_ts = np.array(list(self._cancel_timestamps), dtype=np.int64)
        
        bid_arrival_rate = compute_arrival_rate_numba(bid_arr_ts, timestamp, self.window_ns)
        ask_arrival_rate = compute_arrival_rate_numba(ask_arr_ts, timestamp, self.window_ns)
        trade_arrival_rate = compute_arrival_rate_numba(trade_arr_ts, timestamp, self.window_ns)
        cancel_rate = compute_arrival_rate_numba(cancel_arr_ts, timestamp, self.window_ns)
        
        # Net pressure
        bid_pressure = bid_arrival_rate
        ask_pressure = ask_arrival_rate
        net_pressure = bid_pressure - ask_pressure
        
        # Trade flow
        if self._trade_signs:
            recent_signs = list(self._trade_signs)[-100:]
            recent_sizes = list(self._trade_sizes)[-100:]
            trade_flow = sum(s * sz for s, sz in zip(recent_signs, recent_sizes))
        else:
            trade_flow = 0.0
        
        # VPIN
        if len(self._vpin_buy_volumes) > 0:
            buy_vols = np.array(self._vpin_buy_volumes, dtype=np.float64)
            sell_vols = np.array(self._vpin_sell_volumes, dtype=np.float64)
            vpin = compute_vpin_numba(buy_vols, sell_vols, self.vpin_bucket_volume)
        else:
            vpin = 0.0
        
        # Kyle's lambda (price impact coefficient)
        if len(self._price_changes) >= 10:
            price_changes = np.array(self._price_changes, dtype=np.float64)
            signed_volumes = np.array(self._signed_volumes, dtype=np.float64)
            
            # Simple regression: price_change = lambda * signed_volume
            var_v = np.var(signed_volumes)
            if var_v > 0:
                cov = np.cov(price_changes, signed_volumes)[0, 1]
                kyle_lambda = cov / var_v
            else:
                kyle_lambda = 0.0
        else:
            kyle_lambda = 0.0
        
        # Update state
        self._last_mid_price = mid_price
        
        snapshot = OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=timestamp,
            bid_price=best_bid,
            bid_size=best_bid_size,
            ask_price=best_ask,
            ask_size=best_ask_size,
            mid_price=mid_price,
            spread=spread,
            spread_bps=spread_bps,
            imbalance_l1=imbalance_l1,
            imbalance_l5=imbalance_l5,
            imbalance_l10=imbalance_l10,
            weighted_imbalance=weighted_imbalance,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            bid_pressure=bid_pressure,
            ask_pressure=ask_pressure,
            net_pressure=net_pressure,
            trade_flow=trade_flow,
            vpin=vpin,
            kyle_lambda=kyle_lambda,
            bid_arrival_rate=bid_arrival_rate,
            ask_arrival_rate=ask_arrival_rate,
            trade_arrival_rate=trade_arrival_rate,
            cancel_rate=cancel_rate,
        )
        
        self._last_snapshot = snapshot
        return snapshot
    
    def get_feature_vector(self) -> np.ndarray:
        """
        Get current features as numpy array for ML model input.
        
        Returns:
            Feature vector of shape (num_features,)
        """
        if self._last_snapshot is None:
            return np.zeros(20)
        
        snap = self._last_snapshot
        
        return np.array([
            snap.mid_price,
            snap.spread_bps,
            snap.imbalance_l1,
            snap.imbalance_l5,
            snap.imbalance_l10,
            snap.weighted_imbalance,
            snap.bid_depth_5,
            snap.ask_depth_5,
            snap.bid_depth_10,
            snap.ask_depth_10,
            snap.bid_pressure,
            snap.ask_pressure,
            snap.net_pressure,
            snap.trade_flow,
            snap.vpin,
            snap.kyle_lambda,
            snap.bid_arrival_rate,
            snap.ask_arrival_rate,
            snap.trade_arrival_rate,
            snap.cancel_rate,
        ], dtype=np.float32)
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            "mid_price",
            "spread_bps",
            "imbalance_l1",
            "imbalance_l5",
            "imbalance_l10",
            "weighted_imbalance",
            "bid_depth_5",
            "ask_depth_5",
            "bid_depth_10",
            "ask_depth_10",
            "bid_pressure",
            "ask_pressure",
            "net_pressure",
            "trade_flow",
            "vpin",
            "kyle_lambda",
            "bid_arrival_rate",
            "ask_arrival_rate",
            "trade_arrival_rate",
            "cancel_rate",
        ]
    
    @property
    def last_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Get last computed snapshot."""
        return self._last_snapshot


__all__ = [
    "OrderBookProcessor",
    "OrderBookSnapshot",
    "TradeFlow",
]
