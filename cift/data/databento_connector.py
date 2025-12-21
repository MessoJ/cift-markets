"""
CIFT Markets - Databento L3 Order Book Connector

Institutional-grade Level 3 order book data from Databento.

Features:
- Full L3 order book (individual order tracking)
- Market-by-Order (MBO) data
- Market-by-Price (MBP) aggregated levels
- Nanosecond precision timestamps
- Historical + live data

API Documentation: https://docs.databento.com/

Performance Targets:
- <1ms message processing
- Full order book reconstruction
- Tick-by-tick order lifecycle tracking
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, IntEnum
from typing import Any

from loguru import logger

from cift.core.config import settings

# ============================================================================
# DATABENTO ENUMS
# ============================================================================

class Schema(str, Enum):
    """Databento data schemas."""
    MBO = "mbo"           # Market-by-Order (L3)
    MBP_1 = "mbp-1"       # Top of book
    MBP_10 = "mbp-10"     # Top 10 levels
    TRADES = "trades"     # Trade ticks
    OHLCV_1S = "ohlcv-1s" # 1-second bars
    OHLCV_1M = "ohlcv-1m" # 1-minute bars


class Action(IntEnum):
    """Order action types for MBO data."""
    ADD = 65      # 'A' - New order
    CANCEL = 67   # 'C' - Order cancelled
    MODIFY = 77   # 'M' - Order modified
    TRADE = 84    # 'T' - Trade execution
    FILL = 70     # 'F' - Order filled
    CLEAR = 82    # 'R' - Clear book


class Side(IntEnum):
    """Order side."""
    ASK = 65  # 'A'
    BID = 66  # 'B'
    NONE = 78 # 'N'


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class L3Order:
    """Individual order in the order book (L3 data)."""
    order_id: int
    symbol: str
    side: Side
    price: float
    size: int
    action: Action
    timestamp: int  # Nanoseconds since epoch
    sequence: int
    flags: int = 0
    channel_id: int = 0

    @property
    def datetime_utc(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1e9, tz=UTC)

    @property
    def is_bid(self) -> bool:
        """Check if bid side."""
        return self.side == Side.BID

    @property
    def is_ask(self) -> bool:
        """Check if ask side."""
        return self.side == Side.ASK


@dataclass
class BookLevel:
    """Single price level in aggregated book."""
    price: float
    size: int
    count: int  # Number of orders at this level

    def __lt__(self, other):
        return self.price < other.price


@dataclass
class L3OrderBook:
    """
    Full Level 3 order book with individual order tracking.

    Maintains:
    - Individual orders by order_id
    - Aggregated price levels
    - Order lifecycle tracking
    """
    symbol: str
    bids: dict[int, L3Order] = field(default_factory=dict)  # order_id -> Order
    asks: dict[int, L3Order] = field(default_factory=dict)
    bid_levels: dict[float, BookLevel] = field(default_factory=dict)  # price -> level
    ask_levels: dict[float, BookLevel] = field(default_factory=dict)
    last_update: int = 0
    sequence: int = 0

    def apply_order(self, order: L3Order):
        """Apply order action to the book."""
        self.last_update = order.timestamp
        self.sequence = max(self.sequence, order.sequence)

        if order.action == Action.ADD:
            self._add_order(order)
        elif order.action == Action.CANCEL:
            self._cancel_order(order)
        elif order.action == Action.MODIFY:
            self._modify_order(order)
        elif order.action in (Action.TRADE, Action.FILL):
            self._fill_order(order)
        elif order.action == Action.CLEAR:
            self._clear_book()

    def _add_order(self, order: L3Order):
        """Add new order to book."""
        orders = self.bids if order.is_bid else self.asks
        levels = self.bid_levels if order.is_bid else self.ask_levels

        orders[order.order_id] = order

        # Update aggregated level
        if order.price in levels:
            level = levels[order.price]
            level.size += order.size
            level.count += 1
        else:
            levels[order.price] = BookLevel(order.price, order.size, 1)

    def _cancel_order(self, order: L3Order):
        """Cancel order from book."""
        orders = self.bids if order.is_bid else self.asks
        levels = self.bid_levels if order.is_bid else self.ask_levels

        if order.order_id in orders:
            existing = orders.pop(order.order_id)
            if existing.price in levels:
                level = levels[existing.price]
                level.size -= existing.size
                level.count -= 1
                if level.size <= 0:
                    del levels[existing.price]

    def _modify_order(self, order: L3Order):
        """Modify existing order."""
        # Cancel old, add new
        self._cancel_order(order)
        self._add_order(order)

    def _fill_order(self, order: L3Order):
        """Process trade/fill - reduce order size."""
        orders = self.bids if order.is_bid else self.asks
        levels = self.bid_levels if order.is_bid else self.ask_levels

        if order.order_id in orders:
            existing = orders[order.order_id]
            existing.size -= order.size

            if existing.price in levels:
                level = levels[existing.price]
                level.size -= order.size
                if level.size <= 0:
                    del levels[existing.price]

            if existing.size <= 0:
                del orders[order.order_id]

    def _clear_book(self):
        """Clear entire order book."""
        self.bids.clear()
        self.asks.clear()
        self.bid_levels.clear()
        self.ask_levels.clear()

    def get_top_of_book(self) -> tuple[BookLevel | None, BookLevel | None]:
        """Get best bid and ask."""
        best_bid = max(self.bid_levels.values(), key=lambda x: x.price) if self.bid_levels else None
        best_ask = min(self.ask_levels.values(), key=lambda x: x.price) if self.ask_levels else None
        return best_bid, best_ask

    def get_levels(self, depth: int = 10) -> tuple[list[BookLevel], list[BookLevel]]:
        """Get top N levels on each side."""
        bids = sorted(self.bid_levels.values(), key=lambda x: -x.price)[:depth]
        asks = sorted(self.ask_levels.values(), key=lambda x: x.price)[:depth]
        return bids, asks

    @property
    def spread(self) -> float | None:
        """Calculate bid-ask spread."""
        best_bid, best_ask = self.get_top_of_book()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None

    @property
    def mid_price(self) -> float | None:
        """Calculate mid price."""
        best_bid, best_ask = self.get_top_of_book()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None

    @property
    def imbalance(self) -> float:
        """
        Calculate order book imbalance.

        Returns value from -1 (all asks) to 1 (all bids).
        """
        best_bid, best_ask = self.get_top_of_book()
        if not best_bid or not best_ask:
            return 0.0

        total = best_bid.size + best_ask.size
        if total == 0:
            return 0.0

        return (best_bid.size - best_ask.size) / total

    @property
    def depth_imbalance(self, levels: int = 5) -> float:
        """Calculate imbalance across multiple levels."""
        bids, asks = self.get_levels(levels)

        bid_volume = sum(level.size for level in bids)
        ask_volume = sum(level.size for level in asks)

        total = bid_volume + ask_volume
        if total == 0:
            return 0.0

        return (bid_volume - ask_volume) / total


@dataclass
class TradeRecord:
    """Trade execution record."""
    symbol: str
    price: float
    size: int
    side: Side
    timestamp: int
    sequence: int
    trade_id: int = 0

    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp / 1e9, tz=UTC)

    @property
    def is_buy(self) -> bool:
        """Trade was buyer-initiated."""
        return self.side == Side.BID


# ============================================================================
# DATABENTO CONNECTOR
# ============================================================================

class DatabentoConnector:
    """
    Institutional-grade Databento L3 market data connector.

    Features:
    - Full L3 order book reconstruction
    - Real-time and historical data
    - Multiple venue support
    - Nanosecond precision

    Note: Requires databento Python SDK
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Databento connector.

        Args:
            api_key: Databento API key
        """
        self.api_key = api_key or getattr(settings, 'databento_api_key', '')

        # Order books by symbol
        self._books: dict[str, L3OrderBook] = {}

        # Callbacks
        self._order_callbacks: list[Callable[[L3Order], None]] = []
        self._trade_callbacks: list[Callable[[TradeRecord], None]] = []
        self._book_callbacks: list[Callable[[L3OrderBook], None]] = []

        # Connection state
        self._client = None
        self._live_client = None
        self._connected = False

        # Stats
        self._stats = {
            "orders_processed": 0,
            "trades_processed": 0,
            "book_updates": 0,
        }

        logger.info("DatabentoConnector initialized")

    async def connect(self, dataset: str = "XNAS.ITCH") -> bool:
        """
        Connect to Databento.

        Args:
            dataset: Dataset to connect to (e.g., XNAS.ITCH for NASDAQ)

        Returns:
            True if connected
        """
        try:
            # Import databento SDK
            try:
                import databento as db
            except ImportError:
                logger.error("databento package not installed. Run: pip install databento")
                return False

            if not self.api_key:
                logger.error("Databento API key not configured")
                return False

            # Create historical client for data access
            self._client = db.Historical(self.api_key)

            # Create live client for real-time data
            self._live_client = db.Live(self.api_key)

            self._connected = True
            logger.info(f"Databento connected to {dataset}")
            return True

        except Exception as e:
            logger.error(f"Databento connection failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Databento."""
        self._connected = False
        if self._live_client:
            try:
                await self._live_client.stop()
            except Exception:
                pass
        logger.info("Databento disconnected")

    async def subscribe_l3(
        self,
        symbols: list[str],
        dataset: str = "XNAS.ITCH",
    ):
        """
        Subscribe to L3 order book data.

        Args:
            symbols: List of symbols
            dataset: Dataset (exchange)
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        try:

            # Initialize books for symbols
            for symbol in symbols:
                if symbol not in self._books:
                    self._books[symbol] = L3OrderBook(symbol)

            # Start live subscription
            await self._live_client.subscribe(
                dataset=dataset,
                schema=Schema.MBO.value,
                symbols=symbols,
            )

            # Start receiving messages
            asyncio.create_task(self._receive_l3_messages())

            logger.info(f"Subscribed to L3 data for: {symbols}")

        except Exception as e:
            logger.error(f"L3 subscription failed: {e}")
            raise

    async def subscribe_trades(
        self,
        symbols: list[str],
        dataset: str = "XNAS.ITCH",
    ):
        """Subscribe to trade data."""
        if not self._connected:
            raise RuntimeError("Not connected")

        try:

            await self._live_client.subscribe(
                dataset=dataset,
                schema=Schema.TRADES.value,
                symbols=symbols,
            )

            asyncio.create_task(self._receive_trade_messages())
            logger.info(f"Subscribed to trades for: {symbols}")

        except Exception as e:
            logger.error(f"Trade subscription failed: {e}")
            raise

    async def _receive_l3_messages(self):
        """Process L3 order messages."""
        try:
            async for record in self._live_client:
                order = self._parse_mbo_record(record)
                if order:
                    self._stats["orders_processed"] += 1

                    # Update order book
                    if order.symbol in self._books:
                        self._books[order.symbol].apply_order(order)
                        self._stats["book_updates"] += 1

                        # Notify book callbacks
                        for callback in self._book_callbacks:
                            try:
                                callback(self._books[order.symbol])
                            except Exception as e:
                                logger.error(f"Book callback error: {e}")

                    # Notify order callbacks
                    for callback in self._order_callbacks:
                        try:
                            callback(order)
                        except Exception as e:
                            logger.error(f"Order callback error: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"L3 message processing error: {e}")

    async def _receive_trade_messages(self):
        """Process trade messages."""
        try:
            async for record in self._live_client:
                trade = self._parse_trade_record(record)
                if trade:
                    self._stats["trades_processed"] += 1

                    for callback in self._trade_callbacks:
                        try:
                            callback(trade)
                        except Exception as e:
                            logger.error(f"Trade callback error: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Trade message processing error: {e}")

    def _parse_mbo_record(self, record: Any) -> L3Order | None:
        """Parse Databento MBO record to L3Order."""
        try:
            return L3Order(
                order_id=record.order_id,
                symbol=record.symbol,
                side=Side(record.side),
                price=record.price / 1e9,  # Databento uses fixed-point
                size=record.size,
                action=Action(record.action),
                timestamp=record.ts_recv,
                sequence=record.sequence,
                flags=record.flags,
                channel_id=record.channel_id,
            )
        except Exception as e:
            logger.warning(f"Failed to parse MBO record: {e}")
            return None

    def _parse_trade_record(self, record: Any) -> TradeRecord | None:
        """Parse Databento trade record."""
        try:
            return TradeRecord(
                symbol=record.symbol,
                price=record.price / 1e9,
                size=record.size,
                side=Side(record.side),
                timestamp=record.ts_recv,
                sequence=record.sequence,
            )
        except Exception as e:
            logger.warning(f"Failed to parse trade record: {e}")
            return None

    # ========================================================================
    # HISTORICAL DATA
    # ========================================================================

    async def get_historical_l3(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        dataset: str = "XNAS.ITCH",
    ) -> list[L3Order]:
        """
        Fetch historical L3 data.

        Args:
            symbols: List of symbols
            start: Start time
            end: End time
            dataset: Dataset

        Returns:
            List of L3 orders
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        try:

            data = self._client.timeseries.get_range(
                dataset=dataset,
                schema=Schema.MBO.value,
                symbols=symbols,
                start=start,
                end=end,
            )

            orders = []
            for record in data:
                order = self._parse_mbo_record(record)
                if order:
                    orders.append(order)

            logger.info(f"Fetched {len(orders)} historical L3 orders")
            return orders

        except Exception as e:
            logger.error(f"Historical L3 fetch failed: {e}")
            raise

    async def get_historical_trades(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        dataset: str = "XNAS.ITCH",
    ) -> list[TradeRecord]:
        """Fetch historical trades."""
        if not self._connected:
            raise RuntimeError("Not connected")

        try:

            data = self._client.timeseries.get_range(
                dataset=dataset,
                schema=Schema.TRADES.value,
                symbols=symbols,
                start=start,
                end=end,
            )

            trades = []
            for record in data:
                trade = self._parse_trade_record(record)
                if trade:
                    trades.append(trade)

            logger.info(f"Fetched {len(trades)} historical trades")
            return trades

        except Exception as e:
            logger.error(f"Historical trade fetch failed: {e}")
            raise

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on_order(self, callback: Callable[[L3Order], None]):
        """Register callback for order updates."""
        self._order_callbacks.append(callback)

    def on_trade(self, callback: Callable[[TradeRecord], None]):
        """Register callback for trade updates."""
        self._trade_callbacks.append(callback)

    def on_book_update(self, callback: Callable[[L3OrderBook], None]):
        """Register callback for order book updates."""
        self._book_callbacks.append(callback)

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def get_book(self, symbol: str) -> L3OrderBook | None:
        """Get order book for symbol."""
        return self._books.get(symbol)

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    @property
    def stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()


__all__ = [
    "DatabentoConnector",
    "L3Order",
    "L3OrderBook",
    "BookLevel",
    "TradeRecord",
    "Schema",
    "Action",
    "Side",
]
