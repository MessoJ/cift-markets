"""
CIFT Markets - Polygon.io L2 Order Book Connector

Real-time Level 2 market data via WebSocket.

Features:
- L2 order book quotes (NBBO + depth)
- Trade tick stream
- Aggregated bars
- Market status
- Connection management with auto-reconnect

API Documentation: https://polygon.io/docs/websockets

Performance Targets:
- <5ms message processing latency
- 99.9% uptime with auto-reconnect
- Support for 500+ concurrent symbol subscriptions
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import aiohttp
from loguru import logger

from cift.core.config import settings

# ============================================================================
# DATA MODELS
# ============================================================================


class MessageType(str, Enum):
    """Polygon WebSocket message types."""

    TRADE = "T"  # Trade tick
    QUOTE = "Q"  # NBBO quote
    AGGREGATE_MIN = "AM"  # Minute aggregate
    AGGREGATE_SEC = "A"  # Second aggregate
    STATUS = "status"  # Connection status
    ERROR = "error"  # Error message


@dataclass
class L2Quote:
    """Level 2 quote data (NBBO)."""

    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    bid_exchange: int
    ask_exchange: int
    timestamp: int  # Unix nanoseconds
    conditions: list[int] = field(default_factory=list)

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask_price - self.bid_price

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.mid_price == 0:
            return 0
        return (self.spread / self.mid_price) * 10000

    @property
    def imbalance(self) -> float:
        """Calculate order imbalance (-1 to 1)."""
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0
        return (self.bid_size - self.ask_size) / total

    @property
    def datetime_utc(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1e9, tz=UTC)


@dataclass
class Trade:
    """Trade tick data."""

    symbol: str
    price: float
    size: int
    exchange: int
    timestamp: int  # Unix nanoseconds
    conditions: list[int] = field(default_factory=list)
    trade_id: str = ""
    tape: int = 0

    @property
    def is_buy(self) -> bool:
        """Estimate trade direction (simple heuristic)."""
        # Condition 0 = regular trade, could be buy or sell
        # In practice, we'd use quote comparison
        return True  # Placeholder

    @property
    def value(self) -> float:
        """Calculate trade value."""
        return self.price * self.size

    @property
    def datetime_utc(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1e9, tz=UTC)


@dataclass
class AggregateBar:
    """Aggregate bar (OHLCV)."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    timestamp: int  # Start of bar (Unix ms)
    transactions: int = 0

    @property
    def datetime_utc(self) -> datetime:
        """Convert timestamp to datetime."""
        return datetime.fromtimestamp(self.timestamp / 1000, tz=UTC)

    @property
    def range(self) -> float:
        """Calculate bar range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate candle body size."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """Check if bullish candle."""
        return self.close >= self.open


# ============================================================================
# L2 CONNECTOR
# ============================================================================


class PolygonL2Connector:
    """
    Real-time Polygon.io L2 market data connector.

    Provides:
    - NBBO quote stream
    - Trade tick stream
    - Aggregate bar stream
    - Order book reconstruction from L2 data

    Performance:
    - Async/await for non-blocking I/O
    - Message batching for efficiency
    - Auto-reconnect with exponential backoff
    """

    # Polygon WebSocket endpoints
    STOCKS_WS_URL = "wss://socket.polygon.io/stocks"
    DELAYED_WS_URL = "wss://delayed.polygon.io/stocks"

    def __init__(
        self,
        api_key: str | None = None,
        use_delayed: bool = False,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize L2 connector.

        Args:
            api_key: Polygon API key
            use_delayed: Use delayed data (for free tier)
            max_reconnect_attempts: Max reconnection attempts
            reconnect_delay: Initial reconnect delay (seconds)
        """
        self.api_key = api_key or settings.polygon_api_key
        if not self.api_key:
            raise ValueError("Polygon API key not configured")

        self.ws_url = self.DELAYED_WS_URL if use_delayed else self.STOCKS_WS_URL
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay

        # Connection state
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._connected = False
        self._authenticated = False
        self._reconnect_count = 0

        # Subscriptions
        self._subscribed_quotes: set[str] = set()
        self._subscribed_trades: set[str] = set()
        self._subscribed_aggs: set[str] = set()

        # Callbacks
        self._quote_callbacks: list[Callable[[L2Quote], None]] = []
        self._trade_callbacks: list[Callable[[Trade], None]] = []
        self._aggregate_callbacks: list[Callable[[AggregateBar], None]] = []
        self._error_callbacks: list[Callable[[str], None]] = []

        # Message queue for processing
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

        # Stats
        self._stats = {
            "messages_received": 0,
            "quotes_processed": 0,
            "trades_processed": 0,
            "aggregates_processed": 0,
            "errors": 0,
            "reconnects": 0,
            "last_message_time": None,
        }

        # Background tasks
        self._receive_task: asyncio.Task | None = None
        self._process_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

        logger.info(f"PolygonL2Connector initialized (delayed={use_delayed})")

    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================

    async def connect(self) -> bool:
        """
        Establish WebSocket connection to Polygon.

        Returns:
            True if connected successfully
        """
        if self._connected:
            logger.warning("Already connected")
            return True

        try:
            # Create session if needed
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            logger.info(f"Connecting to Polygon WebSocket: {self.ws_url}")

            self._ws = await self._session.ws_connect(
                self.ws_url,
                heartbeat=30,
                receive_timeout=60,
            )

            self._connected = True
            logger.info("WebSocket connected, authenticating...")

            # Authenticate
            auth_msg = {"action": "auth", "params": self.api_key}
            await self._ws.send_json(auth_msg)

            # Wait for auth response
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if isinstance(data, list):
                        for item in data:
                            if item.get("ev") == "status":
                                status = item.get("status")
                                if status == "auth_success":
                                    self._authenticated = True
                                    logger.info("Polygon authentication successful")
                                    break
                                elif status == "auth_failed":
                                    logger.error(f"Authentication failed: {item.get('message')}")
                                    await self.disconnect()
                                    return False
                    if self._authenticated:
                        break

            if not self._authenticated:
                logger.error("Authentication timeout")
                await self.disconnect()
                return False

            # Start background tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._process_task = asyncio.create_task(self._process_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Resubscribe to existing subscriptions
            await self._resubscribe()

            self._reconnect_count = 0
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            self._authenticated = False
            return False

    async def disconnect(self):
        """Close WebSocket connection."""
        self._connected = False
        self._authenticated = False

        # Cancel background tasks
        for task in [self._receive_task, self._process_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket
        if self._ws and not self._ws.closed:
            await self._ws.close()

        # Close session
        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Polygon WebSocket disconnected")

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        self._reconnect_count += 1
        self._stats["reconnects"] += 1

        if self._reconnect_count > self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False

        delay = self.reconnect_delay * (2 ** (self._reconnect_count - 1))
        delay = min(delay, 60)  # Cap at 60 seconds

        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)

        return await self.connect()

    async def _resubscribe(self):
        """Resubscribe to all channels after reconnect."""
        if self._subscribed_quotes:
            symbols = list(self._subscribed_quotes)
            self._subscribed_quotes.clear()
            await self.subscribe_quotes(symbols)

        if self._subscribed_trades:
            symbols = list(self._subscribed_trades)
            self._subscribed_trades.clear()
            await self.subscribe_trades(symbols)

        if self._subscribed_aggs:
            symbols = list(self._subscribed_aggs)
            self._subscribed_aggs.clear()
            await self.subscribe_aggregates(symbols)

    # ========================================================================
    # SUBSCRIPTIONS
    # ========================================================================

    async def subscribe_quotes(self, symbols: list[str]):
        """
        Subscribe to L2 quote stream for symbols.

        Args:
            symbols: List of stock symbols
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        # Format symbols for subscription
        params = ",".join([f"Q.{s.upper()}" for s in symbols])

        msg = {"action": "subscribe", "params": params}
        await self._ws.send_json(msg)

        self._subscribed_quotes.update(s.upper() for s in symbols)
        logger.info(f"Subscribed to quotes: {symbols}")

    async def subscribe_trades(self, symbols: list[str]):
        """
        Subscribe to trade tick stream for symbols.

        Args:
            symbols: List of stock symbols
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        params = ",".join([f"T.{s.upper()}" for s in symbols])

        msg = {"action": "subscribe", "params": params}
        await self._ws.send_json(msg)

        self._subscribed_trades.update(s.upper() for s in symbols)
        logger.info(f"Subscribed to trades: {symbols}")

    async def subscribe_aggregates(self, symbols: list[str], per_second: bool = False):
        """
        Subscribe to aggregate bar stream.

        Args:
            symbols: List of stock symbols
            per_second: Use per-second bars (default: per-minute)
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        prefix = "A" if per_second else "AM"
        params = ",".join([f"{prefix}.{s.upper()}" for s in symbols])

        msg = {"action": "subscribe", "params": params}
        await self._ws.send_json(msg)

        self._subscribed_aggs.update(s.upper() for s in symbols)
        logger.info(f"Subscribed to aggregates: {symbols}")

    async def unsubscribe_quotes(self, symbols: list[str]):
        """Unsubscribe from quote stream."""
        if not self._connected:
            return

        params = ",".join([f"Q.{s.upper()}" for s in symbols])
        msg = {"action": "unsubscribe", "params": params}
        await self._ws.send_json(msg)

        for s in symbols:
            self._subscribed_quotes.discard(s.upper())

    async def unsubscribe_trades(self, symbols: list[str]):
        """Unsubscribe from trade stream."""
        if not self._connected:
            return

        params = ",".join([f"T.{s.upper()}" for s in symbols])
        msg = {"action": "unsubscribe", "params": params}
        await self._ws.send_json(msg)

        for s in symbols:
            self._subscribed_trades.discard(s.upper())

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on_quote(self, callback: Callable[[L2Quote], None]):
        """Register callback for quote updates."""
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callable[[Trade], None]):
        """Register callback for trade updates."""
        self._trade_callbacks.append(callback)

    def on_aggregate(self, callback: Callable[[AggregateBar], None]):
        """Register callback for aggregate bar updates."""
        self._aggregate_callbacks.append(callback)

    def on_error(self, callback: Callable[[str], None]):
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    # ========================================================================
    # MESSAGE PROCESSING
    # ========================================================================

    async def _receive_loop(self):
        """Background task to receive WebSocket messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._message_queue.put(msg.data)
                    self._stats["messages_received"] += 1
                    self._stats["last_message_time"] = time.time()

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed by server")
                    break

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

        # Attempt reconnect if still supposed to be connected
        if self._connected:
            self._connected = False
            await self._reconnect()

    async def _process_loop(self):
        """Background task to process messages from queue."""
        while True:
            try:
                msg_data = await self._message_queue.get()
                await self._process_message(msg_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                self._stats["errors"] += 1

    async def _heartbeat_loop(self):
        """Background task to send heartbeats."""
        while self._connected:
            try:
                await asyncio.sleep(30)
                if self._ws and not self._ws.closed:
                    # Polygon doesn't require explicit heartbeats,
                    # but we can use this for connection health monitoring
                    last_msg = self._stats["last_message_time"]
                    if last_msg and (time.time() - last_msg) > 60:
                        logger.warning("No messages received in 60s, connection may be stale")
            except asyncio.CancelledError:
                break

    async def _process_message(self, msg_data: str):
        """Process a single WebSocket message."""
        try:
            data = json.loads(msg_data)

            if not isinstance(data, list):
                data = [data]

            for item in data:
                ev_type = item.get("ev")

                if ev_type == "Q":
                    # Quote message
                    quote = self._parse_quote(item)
                    if quote:
                        self._stats["quotes_processed"] += 1
                        for callback in self._quote_callbacks:
                            try:
                                callback(quote)
                            except Exception as e:
                                logger.error(f"Quote callback error: {e}")

                elif ev_type == "T":
                    # Trade message
                    trade = self._parse_trade(item)
                    if trade:
                        self._stats["trades_processed"] += 1
                        for callback in self._trade_callbacks:
                            try:
                                callback(trade)
                            except Exception as e:
                                logger.error(f"Trade callback error: {e}")

                elif ev_type in ("AM", "A"):
                    # Aggregate bar
                    agg = self._parse_aggregate(item)
                    if agg:
                        self._stats["aggregates_processed"] += 1
                        for callback in self._aggregate_callbacks:
                            try:
                                callback(agg)
                            except Exception as e:
                                logger.error(f"Aggregate callback error: {e}")

                elif ev_type == "status":
                    status = item.get("status")
                    message = item.get("message", "")
                    logger.info(f"Status: {status} - {message}")

                elif ev_type == "error":
                    error_msg = item.get("message", "Unknown error")
                    logger.error(f"Polygon error: {error_msg}")
                    self._stats["errors"] += 1
                    for callback in self._error_callbacks:
                        callback(error_msg)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            self._stats["errors"] += 1

    def _parse_quote(self, data: dict[str, Any]) -> L2Quote | None:
        """Parse quote message to L2Quote object."""
        try:
            return L2Quote(
                symbol=data.get("sym", data.get("S", "")),
                bid_price=float(data.get("bp", data.get("p", 0))),
                bid_size=int(data.get("bs", data.get("s", 0))),
                ask_price=float(data.get("ap", data.get("P", 0))),
                ask_size=int(data.get("as", data.get("S", 0))),
                bid_exchange=int(data.get("bx", 0)),
                ask_exchange=int(data.get("ax", 0)),
                timestamp=int(data.get("t", 0)),
                conditions=data.get("c", []),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse quote: {e}")
            return None

    def _parse_trade(self, data: dict[str, Any]) -> Trade | None:
        """Parse trade message to Trade object."""
        try:
            return Trade(
                symbol=data.get("sym", data.get("S", "")),
                price=float(data.get("p", 0)),
                size=int(data.get("s", 0)),
                exchange=int(data.get("x", 0)),
                timestamp=int(data.get("t", 0)),
                conditions=data.get("c", []),
                trade_id=str(data.get("i", "")),
                tape=int(data.get("z", 0)),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse trade: {e}")
            return None

    def _parse_aggregate(self, data: dict[str, Any]) -> AggregateBar | None:
        """Parse aggregate message to AggregateBar object."""
        try:
            return AggregateBar(
                symbol=data.get("sym", data.get("S", "")),
                open=float(data.get("o", 0)),
                high=float(data.get("h", 0)),
                low=float(data.get("l", 0)),
                close=float(data.get("c", 0)),
                volume=int(data.get("v", 0)),
                vwap=float(data.get("vw", 0)),
                timestamp=int(data.get("s", data.get("t", 0))),
                transactions=int(data.get("n", 0)),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse aggregate: {e}")
            return None

    # ========================================================================
    # PROPERTIES & UTILITIES
    # ========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._connected and self._authenticated

    @property
    def stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        return self._stats.copy()

    @property
    def subscribed_symbols(self) -> dict[str, set[str]]:
        """Get all subscribed symbols by type."""
        return {
            "quotes": self._subscribed_quotes.copy(),
            "trades": self._subscribed_trades.copy(),
            "aggregates": self._subscribed_aggs.copy(),
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


async def create_polygon_connector(
    symbols: list[str],
    subscribe_quotes: bool = True,
    subscribe_trades: bool = True,
    subscribe_aggs: bool = True,
) -> PolygonL2Connector:
    """
    Create and connect a Polygon L2 connector with subscriptions.

    Args:
        symbols: Symbols to subscribe to
        subscribe_quotes: Subscribe to L2 quotes
        subscribe_trades: Subscribe to trades
        subscribe_aggs: Subscribe to aggregate bars

    Returns:
        Connected PolygonL2Connector instance
    """
    connector = PolygonL2Connector()

    if await connector.connect():
        if subscribe_quotes:
            await connector.subscribe_quotes(symbols)
        if subscribe_trades:
            await connector.subscribe_trades(symbols)
        if subscribe_aggs:
            await connector.subscribe_aggregates(symbols)

    return connector


__all__ = [
    "PolygonL2Connector",
    "L2Quote",
    "Trade",
    "AggregateBar",
    "MessageType",
    "create_polygon_connector",
]
