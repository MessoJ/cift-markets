"""
CIFT Rust Bindings - High-Performance Modules

This module provides easy access to the Rust-accelerated components:

1. FastFeatureExtractor - ML feature extraction (100x faster than Python)
2. FastIndicators - Technical indicators (RSI, MACD, Bollinger, etc.)
3. FastJsonParser - simd-json WebSocket message parsing (10x faster)
4. FastOrderBook - Order book management (<10μs per operation)
5. FastMarketData - Market data calculations (VWAP, OFI, microprice)
6. FastRiskEngine - Risk validation and position limits

Usage:
    from cift.rust_bindings import (
        FastFeatureExtractor,
        FastIndicators,
        FastJsonParser,
        is_rust_available,
    )
    
    if is_rust_available():
        extractor = FastFeatureExtractor()
        features = extractor.process_tick(100.0, 1000, 99.99, 100.01, 500, 500)
        
        indicators = FastIndicators()
        rsi = FastIndicators.rsi(prices, period=14)
"""

import logging
from typing import Optional, List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# Try to import Rust module
_rust_available = False
_rust_import_error: Optional[str] = None

try:
    import cift_core
    _rust_available = True
    logger.info("✅ Rust cift_core module loaded successfully")
except ImportError as e:
    _rust_import_error = str(e)
    logger.warning(f"⚠️ Rust cift_core not available: {e}")
    logger.info("💡 To build: cd rust_core && maturin develop --release")


def is_rust_available() -> bool:
    """Check if Rust acceleration is available."""
    return _rust_available


def get_rust_import_error() -> Optional[str]:
    """Get the import error if Rust module failed to load."""
    return _rust_import_error


# ============================================================================
# FAST FEATURE EXTRACTOR
# ============================================================================

if _rust_available:
    FastFeatureExtractor = cift_core.FastFeatureExtractor
else:
    # Fallback Python implementation
    class FastFeatureExtractor:
        """
        Fallback Python implementation of FastFeatureExtractor.
        
        For production performance, build the Rust module:
            cd rust_core && maturin develop --release
        """
        
        def __init__(self):
            logger.warning("Using Python fallback for FastFeatureExtractor (slower)")
            self._last_price = 0.0
            self._returns = []
            
        def process_tick(
            self,
            price: float,
            volume: float,
            bid: float,
            ask: float,
            bid_size: float,
            ask_size: float,
        ) -> Dict[str, float]:
            """Process a tick and return feature dict."""
            ret = 0.0
            if self._last_price > 0:
                ret = (price / self._last_price - 1) if self._last_price > 0 else 0
            
            self._last_price = price
            self._returns.append(ret)
            if len(self._returns) > 100:
                self._returns.pop(0)
            
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid if mid > 0 else 0
            total_size = bid_size + ask_size
            ofi = (bid_size - ask_size) / total_size if total_size > 0 else 0
            imbalance = ofi
            
            microprice = (bid * ask_size + ask * bid_size) / total_size if total_size > 0 else mid
            
            return {
                "return_1": ret,
                "return_5": sum(self._returns[-5:]) if len(self._returns) >= 5 else 0,
                "return_20": sum(self._returns[-20:]) if len(self._returns) >= 20 else 0,
                "return_60": sum(self._returns[-60:]) if len(self._returns) >= 60 else 0,
                "volatility_20": self._std(self._returns[-20:]),
                "volatility_60": self._std(self._returns[-60:]),
                "volatility_ratio": 1.0,
                "price_deviation": 0.0,
                "volume": volume,
                "volume_zscore": 0.0,
                "volume_ma_ratio": 1.0,
                "spread": spread,
                "spread_zscore": 0.0,
                "spread_ma_ratio": 1.0,
                "ofi": ofi,
                "ofi_mean": 0.0,
                "ofi_cumulative": 0.0,
                "imbalance": imbalance,
                "log_pressure": 0.0,
                "microprice": microprice,
                "microprice_deviation": (microprice - mid) / mid * 10000 if mid > 0 else 0,
                "rsi": 50.0,
                "momentum_divergence": 0.0,
                "price": price,
                "mid": mid,
                "bid": bid,
                "ask": ask,
                "trade_intensity": 1.0,
            }
        
        def process_tick_array(self, *args) -> List[float]:
            """Process tick and return flat array."""
            d = self.process_tick(*args)
            return list(d.values())
        
        def batch_extract(self, prices, volumes, bids, asks, bid_sizes, ask_sizes) -> List[List[float]]:
            """Batch extract features."""
            results = []
            for i in range(len(prices)):
                arr = self.process_tick_array(
                    prices[i],
                    volumes[i] if i < len(volumes) else 0,
                    bids[i] if i < len(bids) else prices[i],
                    asks[i] if i < len(asks) else prices[i],
                    bid_sizes[i] if i < len(bid_sizes) else 100,
                    ask_sizes[i] if i < len(ask_sizes) else 100,
                )
                results.append(arr)
            return results
        
        @staticmethod
        def feature_names() -> List[str]:
            return [
                "return_1", "return_5", "return_20", "return_60",
                "volatility_20", "volatility_60", "volatility_ratio", "price_deviation",
                "volume", "volume_zscore", "volume_ma_ratio",
                "spread", "spread_zscore", "spread_ma_ratio",
                "ofi", "ofi_mean", "ofi_cumulative", "imbalance", "log_pressure",
                "microprice", "microprice_deviation",
                "rsi", "momentum_divergence",
                "price", "mid", "bid", "ask", "trade_intensity",
            ]
        
        def reset(self):
            self._last_price = 0.0
            self._returns = []
        
        @staticmethod
        def batch_returns(prices: List[float]) -> List[float]:
            if len(prices) < 2:
                return []
            import math
            return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices)) if prices[i-1] > 0]
        
        @staticmethod
        def rolling_std(values: List[float], window: int) -> List[float]:
            result = []
            for i in range(window - 1, len(values)):
                w = values[i - window + 1:i + 1]
                mean = sum(w) / len(w)
                var = sum((x - mean) ** 2 for x in w) / len(w)
                result.append(var ** 0.5)
            return result
        
        @staticmethod
        def _std(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            mean = sum(values) / len(values)
            var = sum((x - mean) ** 2 for x in values) / len(values)
            return var ** 0.5


# ============================================================================
# FAST INDICATORS
# ============================================================================

if _rust_available:
    FastIndicators = cift_core.FastIndicators
else:
    class FastIndicators:
        """
        Fallback Python implementation of FastIndicators.
        
        For production performance, build the Rust module:
            cd rust_core && maturin develop --release
        """
        
        def __init__(self):
            logger.warning("Using Python fallback for FastIndicators (slower)")
        
        @staticmethod
        def rsi(prices: List[float], period: int = 14) -> List[float]:
            """Calculate RSI."""
            if len(prices) <= period:
                return [50.0] * len(prices)
            
            gains = []
            losses = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                gains.append(max(change, 0))
                losses.append(max(-change, 0))
            
            rsi_values = [50.0] * period
            
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100
            rsi_values.append(100 - (100 / (1 + rs)))
            
            alpha = 1 / period
            for i in range(period, len(gains)):
                avg_gain = (1 - alpha) * avg_gain + alpha * gains[i]
                avg_loss = (1 - alpha) * avg_loss + alpha * losses[i]
                rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100
                rsi_values.append(100 - (100 / (1 + rs)))
            
            return rsi_values
        
        @staticmethod
        def macd(
            prices: List[float],
            fast_period: int = 12,
            slow_period: int = 26,
            signal_period: int = 9,
        ) -> Tuple[List[float], List[float], List[float]]:
            """Calculate MACD."""
            def ema(vals, period):
                if not vals:
                    return []
                result = []
                alpha = 2 / (period + 1)
                ema_val = sum(vals[:period]) / min(period, len(vals))
                for i, v in enumerate(vals):
                    if i < period:
                        result.append(sum(vals[:i+1]) / (i + 1))
                    else:
                        ema_val = alpha * v + (1 - alpha) * ema_val
                        result.append(ema_val)
                return result
            
            ema_fast = ema(prices, fast_period)
            ema_slow = ema(prices, slow_period)
            
            macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
            signal_line = ema(macd_line, signal_period)
            histogram = [m - s for m, s in zip(macd_line, signal_line)]
            
            return macd_line, signal_line, histogram
        
        @staticmethod
        def bollinger_bands(
            prices: List[float],
            period: int = 20,
            num_std: float = 2.0,
        ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
            """Calculate Bollinger Bands."""
            n = len(prices)
            upper, middle, lower, bandwidth, percent_b = [], [], [], [], []
            
            for i in range(n):
                start = max(0, i - period + 1)
                window = prices[start:i + 1]
                mean = sum(window) / len(window)
                var = sum((x - mean) ** 2 for x in window) / len(window)
                std = var ** 0.5
                
                u = mean + num_std * std
                l = mean - num_std * std
                
                upper.append(u)
                middle.append(mean)
                lower.append(l)
                bandwidth.append((u - l) / mean if mean > 1e-10 else 0)
                percent_b.append((prices[i] - l) / (u - l) if (u - l) > 1e-10 else 0.5)
            
            return upper, middle, lower, bandwidth, percent_b
        
        @staticmethod
        def atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
            """Calculate ATR."""
            n = min(len(high), len(low), len(close))
            if n < 2:
                return [0.0] * n
            
            tr = [high[0] - low[0]]
            for i in range(1, n):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr.append(max(hl, hc, lc))
            
            atr_values = []
            alpha = 1 / period
            atr_val = sum(tr[:period]) / min(period, n)
            
            for i in range(n):
                if i < period:
                    atr_values.append(sum(tr[:i+1]) / (i + 1))
                else:
                    atr_val = (1 - alpha) * atr_val + alpha * tr[i]
                    atr_values.append(atr_val)
            
            return atr_values
        
        @staticmethod
        def stochastic(
            high: List[float],
            low: List[float],
            close: List[float],
            k_period: int = 14,
            k_slow: int = 3,
            d_period: int = 3,
        ) -> Tuple[List[float], List[float]]:
            """Calculate Stochastic Oscillator."""
            n = min(len(high), len(low), len(close))
            
            raw_k = []
            for i in range(n):
                start = max(0, i - k_period + 1)
                highest = max(high[start:i+1])
                lowest = min(low[start:i+1])
                k = 100 * (close[i] - lowest) / (highest - lowest) if (highest - lowest) > 1e-10 else 50
                raw_k.append(k)
            
            def sma(vals, period):
                result = []
                for i in range(len(vals)):
                    start = max(0, i - period + 1)
                    result.append(sum(vals[start:i+1]) / len(vals[start:i+1]))
                return result
            
            k_values = sma(raw_k, k_slow)
            d_values = sma(k_values, d_period)
            
            return k_values, d_values
        
        @staticmethod
        def vwap(high: List[float], low: List[float], close: List[float], volume: List[float]) -> List[float]:
            """Calculate VWAP."""
            n = min(len(high), len(low), len(close), len(volume))
            cum_pv = 0.0
            cum_vol = 0.0
            vwap_values = []
            
            for i in range(n):
                typical = (high[i] + low[i] + close[i]) / 3
                cum_pv += typical * volume[i]
                cum_vol += volume[i]
                vwap_values.append(cum_pv / cum_vol if cum_vol > 1e-10 else close[i])
            
            return vwap_values
        
        @staticmethod
        def ema(prices: List[float], period: int) -> List[float]:
            """Calculate EMA."""
            if not prices or period == 0:
                return [0.0] * len(prices)
            
            result = []
            alpha = 2 / (period + 1)
            ema_val = sum(prices[:period]) / min(period, len(prices))
            
            for i, v in enumerate(prices):
                if i < period:
                    result.append(sum(prices[:i+1]) / (i + 1))
                else:
                    ema_val = alpha * v + (1 - alpha) * ema_val
                    result.append(ema_val)
            
            return result
        
        @staticmethod
        def sma(prices: List[float], period: int) -> List[float]:
            """Calculate SMA."""
            if not prices or period == 0:
                return [0.0] * len(prices)
            
            result = []
            for i in range(len(prices)):
                start = max(0, i - period + 1)
                result.append(sum(prices[start:i+1]) / len(prices[start:i+1]))
            
            return result
        
        @staticmethod
        def all_indicators(
            high: List[float],
            low: List[float],
            close: List[float],
            volume: List[float],
        ) -> Dict[str, List[float]]:
            """Calculate all indicators at once."""
            result = {}
            
            result["rsi_14"] = FastIndicators.rsi(close, 14)
            
            macd_line, signal_line, histogram = FastIndicators.macd(close, 12, 26, 9)
            result["macd_line"] = macd_line
            result["macd_signal"] = signal_line
            result["macd_histogram"] = histogram
            
            upper, middle, lower, bandwidth, percent_b = FastIndicators.bollinger_bands(close, 20, 2.0)
            result["bb_upper"] = upper
            result["bb_middle"] = middle
            result["bb_lower"] = lower
            result["bb_bandwidth"] = bandwidth
            result["bb_percent_b"] = percent_b
            
            result["atr_14"] = FastIndicators.atr(high, low, close, 14)
            
            k, d = FastIndicators.stochastic(high, low, close, 14, 3, 3)
            result["stoch_k"] = k
            result["stoch_d"] = d
            
            result["vwap"] = FastIndicators.vwap(high, low, close, volume)
            
            result["ema_9"] = FastIndicators.ema(close, 9)
            result["ema_20"] = FastIndicators.ema(close, 20)
            result["ema_50"] = FastIndicators.ema(close, 50)
            
            result["sma_20"] = FastIndicators.sma(close, 20)
            result["sma_50"] = FastIndicators.sma(close, 50)
            result["sma_200"] = FastIndicators.sma(close, 200)
            
            return result


# ============================================================================
# FAST JSON PARSER
# ============================================================================

if _rust_available:
    FastJsonParser = cift_core.PyFastJsonParser
else:
    import json
    
    class FastJsonParser:
        """
        Fallback Python implementation of FastJsonParser.
        
        For production performance, build the Rust module:
            cd rust_core && maturin develop --release
        """
        
        def __init__(self):
            logger.warning("Using Python fallback for FastJsonParser (slower)")
        
        def parse_finnhub(self, json_str: str) -> List[Dict[str, Any]]:
            """Parse Finnhub message."""
            try:
                msg = json.loads(json_str)
                if msg.get("type") != "trade":
                    return []
                
                trades = []
                for t in msg.get("data", []):
                    trades.append({
                        "symbol": t.get("s", ""),
                        "price": t.get("p", 0.0),
                        "size": t.get("v", 0.0),
                        "timestamp_ms": t.get("t", 0),
                        "exchange": "FINNHUB",
                    })
                return trades
            except Exception as e:
                logger.error(f"Parse error: {e}")
                return []
        
        def parse_finnhub_arrays(self, json_str: str) -> Tuple[List[str], List[float], List[float], List[int]]:
            """Parse Finnhub and return arrays."""
            trades = self.parse_finnhub(json_str)
            symbols = [t["symbol"] for t in trades]
            prices = [t["price"] for t in trades]
            sizes = [t["size"] for t in trades]
            timestamps = [t["timestamp_ms"] for t in trades]
            return symbols, prices, sizes, timestamps
        
        def parse_alpaca_trade(self, json_str: str) -> Dict[str, Any]:
            """Parse Alpaca trade message."""
            try:
                t = json.loads(json_str)
                from datetime import datetime
                ts = datetime.fromisoformat(t.get("t", "").replace("Z", "+00:00"))
                return {
                    "symbol": t.get("S", ""),
                    "price": t.get("p", 0.0),
                    "size": t.get("s", 0),
                    "timestamp_ms": int(ts.timestamp() * 1000),
                    "exchange": t.get("x", ""),
                }
            except Exception as e:
                logger.error(f"Parse error: {e}")
                return {}


# ============================================================================
# FAST ORDER BOOK
# ============================================================================

if _rust_available:
    FastOrderBook = cift_core.FastOrderBook
else:
    class FastOrderBook:
        """Fallback - use cift/data/order_book_processor.py instead."""
        def __init__(self, symbol: str):
            logger.warning("Using Python fallback for FastOrderBook")
            self.symbol = symbol


# ============================================================================
# FAST MARKET DATA
# ============================================================================

if _rust_available:
    FastMarketData = cift_core.FastMarketData
else:
    class FastMarketData:
        """Fallback Python implementation."""
        def __init__(self):
            logger.warning("Using Python fallback for FastMarketData")
        
        def calculate_vwap(self, ticks: List[Tuple[float, float]]) -> float:
            if not ticks:
                return 0.0
            total_pv = sum(p * v for p, v in ticks)
            total_v = sum(v for _, v in ticks)
            return total_pv / total_v if total_v > 0 else 0.0
        
        def calculate_ofi(self, bid_volumes: List[float], ask_volumes: List[float]) -> float:
            total_bid = sum(bid_volumes)
            total_ask = sum(ask_volumes)
            total = total_bid + total_ask
            return (total_bid - total_ask) / total if total > 0 else 0.0
        
        def calculate_microprice(self, bid: float, ask: float, bid_vol: float, ask_vol: float) -> float:
            total = bid_vol + ask_vol
            return (bid * ask_vol + ask * bid_vol) / total if total > 0 else (bid + ask) / 2


# ============================================================================
# FAST RISK ENGINE
# ============================================================================

if _rust_available:
    FastRiskEngine = cift_core.FastRiskEngine
else:
    class FastRiskEngine:
        """Fallback Python implementation."""
        def __init__(self, max_pos: float, max_notional: float, max_leverage: float):
            self.max_pos = max_pos
            self.max_notional = max_notional
            self.max_leverage = max_leverage
        
        def check_order(self, symbol, side, qty, price, position, account_value):
            notional = qty * price
            if notional > self.max_notional:
                return False, "Exceeds max notional"
            if abs(position + qty) > self.max_pos:
                return False, "Exceeds max position"
            return True, "OK"


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_module_info() -> Dict[str, Any]:
    """Get information about the Rust module."""
    return {
        "rust_available": _rust_available,
        "import_error": _rust_import_error,
        "classes": [
            "FastFeatureExtractor",
            "FastIndicators",
            "FastJsonParser",
            "FastOrderBook",
            "FastMarketData",
            "FastRiskEngine",
        ],
        "build_instructions": "cd rust_core && maturin develop --release",
    }


def benchmark_rust_vs_python(n_samples: int = 10000) -> Dict[str, float]:
    """Benchmark Rust vs Python implementations."""
    import time
    import random
    
    prices = [100 + random.random() * 10 for _ in range(n_samples)]
    
    results = {}
    
    # RSI benchmark
    start = time.perf_counter()
    FastIndicators.rsi(prices, 14)
    results["rsi_time_ms"] = (time.perf_counter() - start) * 1000
    
    # Feature extraction benchmark
    extractor = FastFeatureExtractor()
    start = time.perf_counter()
    for i in range(min(1000, n_samples)):
        extractor.process_tick(prices[i], 1000, prices[i] - 0.01, prices[i] + 0.01, 500, 500)
    results["feature_extraction_1000_ticks_ms"] = (time.perf_counter() - start) * 1000
    
    results["using_rust"] = _rust_available
    
    return results


__all__ = [
    "FastFeatureExtractor",
    "FastIndicators", 
    "FastJsonParser",
    "FastOrderBook",
    "FastMarketData",
    "FastRiskEngine",
    "is_rust_available",
    "get_rust_import_error",
    "get_module_info",
    "benchmark_rust_vs_python",
]
