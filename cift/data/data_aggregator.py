"""
CIFT Markets - Market Data Aggregator

Unified interface for multiple market data sources.

Features:
- Multi-source data fusion (Polygon, Databento)
- Order book consolidation
- Feature extraction orchestration
- Real-time streaming
- Historical data replay

This is the main entry point for market data in the inference pipeline.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from loguru import logger

from cift.data.databento_connector import (
    DatabentoConnector,
    L3Order,
    L3OrderBook,
)
from cift.data.order_book_processor import OrderBookProcessor, OrderBookSnapshot
from cift.data.polygon_l2_connector import (
    AggregateBar,
    L2Quote,
    PolygonL2Connector,
    Trade,
)

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class MarketTick:
    """Unified market tick representation."""

    symbol: str
    timestamp: int  # Nanoseconds
    source: str  # "polygon", "databento"

    # Price data
    bid_price: float
    ask_price: float
    mid_price: float
    last_price: float

    # Size data
    bid_size: int
    ask_size: int
    last_size: int

    # Computed
    spread: float
    imbalance: float


@dataclass
class AggregatedFeatures:
    """
    Aggregated features from all sources for a symbol.

    This is the input to the ML models.
    """

    symbol: str
    timestamp: int

    # Order book features (from processor)
    book_features: np.ndarray  # Shape: (20,)

    # Time-series features (multiple timeframes)
    price_returns_1s: float
    price_returns_5s: float
    price_returns_30s: float
    price_returns_1m: float

    volatility_1m: float
    volatility_5m: float

    # Volume features
    volume_1m: int
    buy_volume_1m: int
    sell_volume_1m: int
    volume_imbalance_1m: float

    # Aggregate bar features
    bar_close: float
    bar_vwap: float
    bar_range: float
    bar_body: float
    bar_is_bullish: bool

    def to_numpy(self) -> np.ndarray:
        """Convert all features to numpy array."""
        return np.concatenate(
            [
                self.book_features,
                np.array(
                    [
                        self.price_returns_1s,
                        self.price_returns_5s,
                        self.price_returns_30s,
                        self.price_returns_1m,
                        self.volatility_1m,
                        self.volatility_5m,
                        self.volume_1m,
                        self.buy_volume_1m,
                        self.sell_volume_1m,
                        self.volume_imbalance_1m,
                        self.bar_close,
                        self.bar_vwap,
                        self.bar_range,
                        self.bar_body,
                        1.0 if self.bar_is_bullish else 0.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )


# ============================================================================
# MARKET DATA AGGREGATOR
# ============================================================================


class MarketDataAggregator:
    """
    Unified market data aggregation and feature extraction.

    Combines multiple data sources and provides:
    - Real-time feature extraction
    - Historical data replay
    - Multi-symbol coordination
    - Callback-based event dispatch
    """

    def __init__(
        self,
        use_polygon: bool = True,
        use_databento: bool = False,
        polygon_delayed: bool = True,  # Use delayed data for free tier
    ):
        """
        Initialize aggregator.

        Args:
            use_polygon: Enable Polygon.io connector
            use_databento: Enable Databento connector
            polygon_delayed: Use delayed Polygon data
        """
        self.use_polygon = use_polygon
        self.use_databento = use_databento

        # Data connectors
        self._polygon: PolygonL2Connector | None = None
        self._databento: DatabentoConnector | None = None

        if use_polygon:
            self._polygon = PolygonL2Connector(use_delayed=polygon_delayed)

        if use_databento:
            self._databento = DatabentoConnector()

        # Order book processors per symbol
        self._processors: dict[str, OrderBookProcessor] = {}

        # Price history for returns calculation
        self._price_history: dict[str, list] = defaultdict(list)  # (timestamp, price)
        self._max_history = 1000

        # Volume tracking
        self._volume_history: dict[str, list] = defaultdict(list)  # (timestamp, volume, is_buy)

        # Latest bars
        self._latest_bars: dict[str, AggregateBar] = {}

        # Callbacks
        self._tick_callbacks: list[Callable[[MarketTick], None]] = []
        self._feature_callbacks: list[Callable[[AggregatedFeatures], None]] = []
        self._snapshot_callbacks: list[Callable[[OrderBookSnapshot], None]] = []

        # State
        self._running = False
        self._subscribed_symbols: set[str] = set()

        logger.info(
            f"MarketDataAggregator initialized (polygon={use_polygon}, databento={use_databento})"
        )

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    async def connect(self) -> bool:
        """Connect to all enabled data sources."""
        success = True

        if self._polygon:
            if await self._polygon.connect():
                # Set up callbacks
                self._polygon.on_quote(self._handle_polygon_quote)
                self._polygon.on_trade(self._handle_polygon_trade)
                self._polygon.on_aggregate(self._handle_polygon_aggregate)
                logger.info("Polygon connected")
            else:
                logger.error("Polygon connection failed")
                success = False

        if self._databento:
            if await self._databento.connect():
                self._databento.on_order(self._handle_databento_order)
                self._databento.on_trade(self._handle_databento_trade)
                self._databento.on_book_update(self._handle_databento_book)
                logger.info("Databento connected")
            else:
                logger.error("Databento connection failed")
                success = False

        self._running = success
        return success

    async def disconnect(self):
        """Disconnect from all data sources."""
        self._running = False

        if self._polygon:
            await self._polygon.disconnect()

        if self._databento:
            await self._databento.disconnect()

        logger.info("MarketDataAggregator disconnected")

    async def subscribe(self, symbols: list[str]):
        """
        Subscribe to market data for symbols.

        Args:
            symbols: List of symbols to subscribe to
        """
        for symbol in symbols:
            symbol = symbol.upper()

            # Create processor if needed
            if symbol not in self._processors:
                self._processors[symbol] = OrderBookProcessor(symbol)

            self._subscribed_symbols.add(symbol)

        # Subscribe on connectors
        if self._polygon and self._polygon.is_connected:
            await self._polygon.subscribe_quotes(symbols)
            await self._polygon.subscribe_trades(symbols)
            await self._polygon.subscribe_aggregates(symbols)

        if self._databento and self._databento.is_connected:
            await self._databento.subscribe_l3(symbols)
            await self._databento.subscribe_trades(symbols)

        logger.info(f"Subscribed to: {symbols}")

    async def unsubscribe(self, symbols: list[str]):
        """Unsubscribe from symbols."""
        for symbol in symbols:
            symbol = symbol.upper()
            self._subscribed_symbols.discard(symbol)

        if self._polygon and self._polygon.is_connected:
            await self._polygon.unsubscribe_quotes(symbols)
            await self._polygon.unsubscribe_trades(symbols)

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on_tick(self, callback: Callable[[MarketTick], None]):
        """Register tick callback."""
        self._tick_callbacks.append(callback)

    def on_features(self, callback: Callable[[AggregatedFeatures], None]):
        """Register feature update callback."""
        self._feature_callbacks.append(callback)

    def on_snapshot(self, callback: Callable[[OrderBookSnapshot], None]):
        """Register order book snapshot callback."""
        self._snapshot_callbacks.append(callback)

    # ========================================================================
    # DATA HANDLERS
    # ========================================================================

    def _handle_polygon_quote(self, quote: L2Quote):
        """Handle Polygon quote update."""
        symbol = quote.symbol

        if symbol not in self._processors:
            self._processors[symbol] = OrderBookProcessor(symbol)

        processor = self._processors[symbol]

        # Update book (single level from NBBO)
        snapshot = processor.update_book(
            bid_prices=[quote.bid_price],
            bid_sizes=[quote.bid_size],
            ask_prices=[quote.ask_price],
            ask_sizes=[quote.ask_size],
            timestamp=quote.timestamp,
        )

        # Record price
        self._record_price(symbol, quote.timestamp, quote.mid_price)

        # Create tick
        tick = MarketTick(
            symbol=symbol,
            timestamp=quote.timestamp,
            source="polygon",
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            mid_price=quote.mid_price,
            last_price=quote.mid_price,
            bid_size=quote.bid_size,
            ask_size=quote.ask_size,
            last_size=0,
            spread=quote.spread,
            imbalance=quote.imbalance,
        )

        # Dispatch callbacks
        for callback in self._tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

        for callback in self._snapshot_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Snapshot callback error: {e}")

        # Generate aggregated features
        self._emit_features(symbol, snapshot)

    def _handle_polygon_trade(self, trade: Trade):
        """Handle Polygon trade update."""
        symbol = trade.symbol

        if symbol not in self._processors:
            return

        processor = self._processors[symbol]

        # Estimate buy/sell from tick rule (simplified)
        # In practice, compare to prevailing quote
        is_buy = True  # Placeholder - would use quote comparison

        processor.record_trade(
            price=trade.price,
            size=trade.size,
            is_buy=is_buy,
            timestamp=trade.timestamp,
        )

        # Record volume
        self._record_volume(symbol, trade.timestamp, trade.size, is_buy)

    def _handle_polygon_aggregate(self, bar: AggregateBar):
        """Handle Polygon aggregate bar."""
        self._latest_bars[bar.symbol] = bar

    def _handle_databento_order(self, order: L3Order):
        """Handle Databento L3 order update."""
        symbol = order.symbol

        if symbol not in self._processors:
            self._processors[symbol] = OrderBookProcessor(symbol)

        processor = self._processors[symbol]
        processor.record_order(order.is_bid, order.timestamp)

    def _handle_databento_trade(self, trade):
        """Handle Databento trade."""
        symbol = trade.symbol

        if symbol not in self._processors:
            return

        processor = self._processors[symbol]
        processor.record_trade(
            price=trade.price,
            size=trade.size,
            is_buy=trade.is_buy,
            timestamp=trade.timestamp,
        )

    def _handle_databento_book(self, book: L3OrderBook):
        """Handle Databento book update."""
        symbol = book.symbol

        if symbol not in self._processors:
            self._processors[symbol] = OrderBookProcessor(symbol)

        processor = self._processors[symbol]

        # Convert L3 book levels to arrays
        bids, asks = book.get_levels(20)

        bid_prices = [level.price for level in bids]
        bid_sizes = [level.size for level in bids]
        ask_prices = [level.price for level in asks]
        ask_sizes = [level.size for level in asks]

        snapshot = processor.update_book(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            timestamp=book.last_update,
        )

        # Dispatch snapshot callbacks
        for callback in self._snapshot_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Snapshot callback error: {e}")

        # Generate features
        self._emit_features(symbol, snapshot)

    # ========================================================================
    # FEATURE GENERATION
    # ========================================================================

    def _record_price(self, symbol: str, timestamp: int, price: float):
        """Record price for returns calculation."""
        history = self._price_history[symbol]
        history.append((timestamp, price))

        # Trim history
        if len(history) > self._max_history:
            history.pop(0)

    def _record_volume(self, symbol: str, timestamp: int, volume: int, is_buy: bool):
        """Record volume for aggregation."""
        history = self._volume_history[symbol]
        history.append((timestamp, volume, is_buy))

        if len(history) > self._max_history:
            history.pop(0)

    def _compute_returns(self, symbol: str, window_ns: int, current_ts: int) -> float:
        """Compute price returns over window."""
        history = self._price_history[symbol]
        if len(history) < 2:
            return 0.0

        current_price = history[-1][1]

        # Find price at window start
        cutoff = current_ts - window_ns
        for ts, price in reversed(history[:-1]):
            if ts <= cutoff:
                if price > 0:
                    return (current_price - price) / price
                break

        return 0.0

    def _compute_volatility(self, symbol: str, window_ns: int, current_ts: int) -> float:
        """Compute realized volatility over window."""
        history = self._price_history[symbol]
        if len(history) < 10:
            return 0.0

        cutoff = current_ts - window_ns
        prices = [p for ts, p in history if ts >= cutoff]

        if len(prices) < 2:
            return 0.0

        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0:
                returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

        if returns:
            return float(np.std(returns))
        return 0.0

    def _compute_volume_stats(self, symbol: str, window_ns: int, current_ts: int) -> tuple:
        """Compute volume statistics over window."""
        history = self._volume_history[symbol]

        cutoff = current_ts - window_ns
        total_volume = 0
        buy_volume = 0
        sell_volume = 0

        for ts, vol, is_buy in history:
            if ts >= cutoff:
                total_volume += vol
                if is_buy:
                    buy_volume += vol
                else:
                    sell_volume += vol

        imbalance = 0.0
        if total_volume > 0:
            imbalance = (buy_volume - sell_volume) / total_volume

        return total_volume, buy_volume, sell_volume, imbalance

    def _emit_features(self, symbol: str, snapshot: OrderBookSnapshot):
        """Generate and emit aggregated features."""
        if not self._feature_callbacks:
            return

        processor = self._processors.get(symbol)
        if not processor:
            return

        current_ts = snapshot.timestamp

        # Compute returns at different timeframes
        returns_1s = self._compute_returns(symbol, int(1e9), current_ts)
        returns_5s = self._compute_returns(symbol, int(5e9), current_ts)
        returns_30s = self._compute_returns(symbol, int(30e9), current_ts)
        returns_1m = self._compute_returns(symbol, int(60e9), current_ts)

        # Volatility
        vol_1m = self._compute_volatility(symbol, int(60e9), current_ts)
        vol_5m = self._compute_volatility(symbol, int(300e9), current_ts)

        # Volume stats
        vol_total, vol_buy, vol_sell, vol_imb = self._compute_volume_stats(
            symbol, int(60e9), current_ts
        )

        # Bar data
        bar = self._latest_bars.get(symbol)
        bar_close = bar.close if bar else snapshot.mid_price
        bar_vwap = bar.vwap if bar else snapshot.mid_price
        bar_range = bar.range if bar else 0.0
        bar_body = bar.body if bar else 0.0
        bar_is_bullish = bar.is_bullish if bar else True

        # Create aggregated features
        features = AggregatedFeatures(
            symbol=symbol,
            timestamp=current_ts,
            book_features=processor.get_feature_vector(),
            price_returns_1s=returns_1s,
            price_returns_5s=returns_5s,
            price_returns_30s=returns_30s,
            price_returns_1m=returns_1m,
            volatility_1m=vol_1m,
            volatility_5m=vol_5m,
            volume_1m=vol_total,
            buy_volume_1m=vol_buy,
            sell_volume_1m=vol_sell,
            volume_imbalance_1m=vol_imb,
            bar_close=bar_close,
            bar_vwap=bar_vwap,
            bar_range=bar_range,
            bar_body=bar_body,
            bar_is_bullish=bar_is_bullish,
        )

        # Dispatch
        for callback in self._feature_callbacks:
            try:
                callback(features)
            except Exception as e:
                logger.error(f"Feature callback error: {e}")

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def get_processor(self, symbol: str) -> OrderBookProcessor | None:
        """Get order book processor for symbol."""
        return self._processors.get(symbol.upper())

    def get_latest_snapshot(self, symbol: str) -> OrderBookSnapshot | None:
        """Get latest order book snapshot for symbol."""
        processor = self._processors.get(symbol.upper())
        return processor.last_snapshot if processor else None

    def get_latest_bar(self, symbol: str) -> AggregateBar | None:
        """Get latest aggregate bar for symbol."""
        return self._latest_bars.get(symbol.upper())

    @property
    def is_running(self) -> bool:
        """Check if aggregator is running."""
        return self._running

    @property
    def subscribed_symbols(self) -> set[str]:
        """Get set of subscribed symbols."""
        return self._subscribed_symbols.copy()


__all__ = [
    "MarketDataAggregator",
    "MarketTick",
    "AggregatedFeatures",
]
