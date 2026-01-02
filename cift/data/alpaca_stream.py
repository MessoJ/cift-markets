"""
CIFT Markets - Alpaca Real-time Market Data Streaming

WebSocket streaming for real-time stock data via Alpaca API.

Features:
- Real-time quotes (NBBO - best bid/offer)
- Trade ticks
- Minute/daily bars
- LULD (Limit Up/Limit Down) bands
- Trading status events
- Order imbalances
- Auto-reconnect with exponential backoff

API Documentation: https://docs.alpaca.markets/docs/streaming-market-data

Note: Alpaca provides L1 data (NBBO). For full L2 order book depth, use Polygon.
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
# CONSTANTS
# ============================================================================


class AlpacaFeed(str, Enum):
    """Alpaca market data feeds."""
    SIP = "sip"  # Securities Information Processor (all exchanges)
    IEX = "iex"  # IEX only (free tier)
    DELAYED_SIP = "delayed_sip"  # 15-minute delayed SIP
    TEST = "test"  # Test stream (always available)


class MessageType(str, Enum):
    """Alpaca WebSocket message types."""
    SUCCESS = "success"
    ERROR = "error"
    SUBSCRIPTION = "subscription"
    TRADE = "t"
    QUOTE = "q"
    BAR = "b"
    DAILY_BAR = "d"
    UPDATED_BAR = "u"
    LULD = "l"  # Limit Up/Limit Down
    STATUS = "s"  # Trading status
    CORRECTION = "c"  # Trade correction
    CANCEL_ERROR = "x"  # Trade cancel/error
    IMBALANCE = "i"  # Order imbalance


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class AlpacaQuote:
    """
    Real-time quote data (NBBO - National Best Bid and Offer).
    
    This is L1 data - the best bid and ask across all exchanges.
    """
    symbol: str
    bid_price: float
    bid_size: int  # In round lots (100 shares)
    ask_price: float
    ask_size: int  # In round lots (100 shares)
    bid_exchange: str
    ask_exchange: str
    timestamp: datetime
    conditions: list[str] = field(default_factory=list)
    tape: str = ""
    
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
    def bid_notional(self) -> float:
        """Bid size in dollars (size * 100 * price)."""
        return self.bid_size * 100 * self.bid_price
    
    @property
    def ask_notional(self) -> float:
        """Ask size in dollars (size * 100 * price)."""
        return self.ask_size * 100 * self.ask_price
    
    @property
    def imbalance(self) -> float:
        """
        Calculate order imbalance from BBO (-1 to 1).
        
        Positive = more bid size (bullish pressure)
        Negative = more ask size (bearish pressure)
        """
        total = self.bid_size + self.ask_size
        if total == 0:
            return 0
        return (self.bid_size - self.ask_size) / total


@dataclass
class AlpacaTrade:
    """Real-time trade tick."""
    symbol: str
    price: float
    size: int
    exchange: str
    trade_id: int
    timestamp: datetime
    conditions: list[str] = field(default_factory=list)
    tape: str = ""
    
    @property
    def notional(self) -> float:
        """Trade value in dollars."""
        return self.price * self.size
    
    @property
    def is_odd_lot(self) -> bool:
        """Check if trade is an odd lot (< 100 shares)."""
        return self.size < 100


@dataclass
class AlpacaBar:
    """OHLCV bar (minute, daily, or updated)."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    trade_count: int
    timestamp: datetime
    bar_type: str = "minute"  # minute, daily, updated


@dataclass
class AlpacaLULD:
    """Limit Up/Limit Down price bands."""
    symbol: str
    limit_up: float
    limit_down: float
    indicator: str
    timestamp: datetime
    tape: str = ""


@dataclass 
class AlpacaTradingStatus:
    """Trading halt/resume status."""
    symbol: str
    status_code: str
    status_message: str
    reason_code: str
    reason_message: str
    timestamp: datetime
    tape: str = ""


@dataclass
class AlpacaImbalance:
    """Order imbalance during trading halts."""
    symbol: str
    price: float
    timestamp: datetime
    tape: str = ""


# ============================================================================
# ALPACA STREAMING CLIENT
# ============================================================================


class AlpacaStreamClient:
    """
    High-performance WebSocket streaming client for Alpaca market data.
    
    Supports:
    - Real-time quotes (NBBO)
    - Trade ticks
    - Minute/daily/updated bars
    - LULD bands
    - Trading status
    - Order imbalances
    - Auto-reconnect with exponential backoff
    
    Usage:
        client = AlpacaStreamClient()
        client.on_quote(my_quote_handler)
        client.on_trade(my_trade_handler)
        
        await client.connect()
        await client.subscribe_quotes(["AAPL", "MSFT"])
        await client.subscribe_trades(["SPY"])
        
        # Run until done
        await client.run()
    """
    
    # Stream URLs
    STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"
    STREAM_URL_SANDBOX = "wss://stream.data.sandbox.alpaca.markets/v2/{feed}"
    
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: AlpacaFeed = AlpacaFeed.IEX,
        use_sandbox: bool = False,
        max_reconnect_attempts: int = 10,
        reconnect_delay: float = 1.0,
    ):
        """
        Initialize Alpaca streaming client.
        
        Args:
            api_key: Alpaca API key (defaults to config)
            secret_key: Alpaca secret key (defaults to config)
            feed: Data feed (SIP requires paid subscription)
            use_sandbox: Use sandbox environment
            max_reconnect_attempts: Max reconnection attempts
            reconnect_delay: Initial delay between reconnects (exponential backoff)
        """
        self.api_key = api_key or settings.alpaca_api_key
        self.secret_key = secret_key or settings.alpaca_secret_key
        self.feed = feed
        self.use_sandbox = use_sandbox
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        # WebSocket
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._authenticated = False
        self._reconnect_attempts = 0
        
        # Subscriptions
        self._subscribed_trades: set[str] = set()
        self._subscribed_quotes: set[str] = set()
        self._subscribed_bars: set[str] = set()
        self._subscribed_daily_bars: set[str] = set()
        self._subscribed_statuses: set[str] = set()
        self._subscribed_lulds: set[str] = set()
        self._subscribed_imbalances: set[str] = set()
        
        # Callbacks
        self._quote_callbacks: list[Callable[[AlpacaQuote], None]] = []
        self._trade_callbacks: list[Callable[[AlpacaTrade], None]] = []
        self._bar_callbacks: list[Callable[[AlpacaBar], None]] = []
        self._luld_callbacks: list[Callable[[AlpacaLULD], None]] = []
        self._status_callbacks: list[Callable[[AlpacaTradingStatus], None]] = []
        self._imbalance_callbacks: list[Callable[[AlpacaImbalance], None]] = []
        self._error_callbacks: list[Callable[[dict], None]] = []
        
        # Metrics
        self._message_count = 0
        self._quote_count = 0
        self._trade_count = 0
        self._last_message_time: float = 0
        self._connect_time: float | None = None
        
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and not self._ws.closed
    
    @property
    def is_authenticated(self) -> bool:
        """Check if authenticated with Alpaca."""
        return self._authenticated
    
    @property
    def is_configured(self) -> bool:
        """Check if API keys are configured."""
        return bool(self.api_key and self.secret_key)
    
    def _get_stream_url(self) -> str:
        """Get WebSocket stream URL."""
        base = self.STREAM_URL_SANDBOX if self.use_sandbox else self.STREAM_URL
        return base.format(feed=self.feed.value)
    
    # ========================================================================
    # CONNECTION
    # ========================================================================
    
    async def connect(self) -> bool:
        """
        Connect to Alpaca WebSocket stream.
        
        Returns:
            True if connected and authenticated successfully
        """
        if not self.is_configured:
            logger.error("Alpaca API keys not configured")
            return False
        
        url = self._get_stream_url()
        logger.info(f"Connecting to Alpaca stream: {url}")
        
        try:
            # Create session
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=30, connect=10)
                self._session = aiohttp.ClientSession(timeout=timeout)
            
            # Connect WebSocket
            self._ws = await self._session.ws_connect(
                url,
                heartbeat=30,
                compress=15,  # RFC-7692 compression
            )
            
            self._connect_time = time.time()
            
            # Wait for welcome message
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if isinstance(data, list) and len(data) > 0:
                    if data[0].get("T") == "success" and data[0].get("msg") == "connected":
                        logger.info("Connected to Alpaca stream")
                    else:
                        logger.error(f"Unexpected welcome message: {data}")
                        return False
            
            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key,
            }
            await self._ws.send_json(auth_msg)
            
            # Wait for auth response
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if isinstance(data, list) and len(data) > 0:
                    if data[0].get("T") == "success" and data[0].get("msg") == "authenticated":
                        logger.info("Authenticated with Alpaca")
                        self._authenticated = True
                        self._reconnect_attempts = 0
                        return True
                    elif data[0].get("T") == "error":
                        logger.error(f"Auth failed: {data[0]}")
                        return False
            
            logger.error("Failed to authenticate")
            return False
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self._running = False
        self._authenticated = False
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
            logger.info("Disconnected from Alpaca stream")
        
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _reconnect(self):
        """Reconnect with exponential backoff."""
        while self._running and self._reconnect_attempts < self.max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self.reconnect_delay * (2 ** (self._reconnect_attempts - 1))
            delay = min(delay, 60)  # Cap at 60 seconds
            
            logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_attempts})")
            await asyncio.sleep(delay)
            
            if await self.connect():
                # Restore subscriptions
                await self._restore_subscriptions()
                return True
        
        logger.error("Max reconnection attempts reached")
        return False
    
    async def _restore_subscriptions(self):
        """Restore subscriptions after reconnect."""
        if self._subscribed_quotes:
            await self.subscribe_quotes(list(self._subscribed_quotes))
        if self._subscribed_trades:
            await self.subscribe_trades(list(self._subscribed_trades))
        if self._subscribed_bars:
            await self.subscribe_bars(list(self._subscribed_bars))
        if self._subscribed_daily_bars:
            await self.subscribe_daily_bars(list(self._subscribed_daily_bars))
        if self._subscribed_statuses:
            await self.subscribe_statuses(list(self._subscribed_statuses))
        if self._subscribed_lulds:
            await self.subscribe_lulds(list(self._subscribed_lulds))
    
    # ========================================================================
    # SUBSCRIPTIONS
    # ========================================================================
    
    async def _subscribe(self, channel: str, symbols: list[str]) -> bool:
        """Send subscription message."""
        if not self.is_connected:
            logger.error("Not connected")
            return False
        
        msg = {"action": "subscribe", channel: symbols}
        await self._ws.send_json(msg)
        
        # Wait for subscription confirmation
        # (handled in message loop)
        return True
    
    async def _unsubscribe(self, channel: str, symbols: list[str]) -> bool:
        """Send unsubscription message."""
        if not self.is_connected:
            return False
        
        msg = {"action": "unsubscribe", channel: symbols}
        await self._ws.send_json(msg)
        return True
    
    async def subscribe_quotes(self, symbols: list[str]) -> bool:
        """Subscribe to real-time quotes (NBBO)."""
        self._subscribed_quotes.update(symbols)
        return await self._subscribe("quotes", symbols)
    
    async def subscribe_trades(self, symbols: list[str]) -> bool:
        """Subscribe to trade ticks."""
        self._subscribed_trades.update(symbols)
        return await self._subscribe("trades", symbols)
    
    async def subscribe_bars(self, symbols: list[str]) -> bool:
        """Subscribe to minute bars."""
        self._subscribed_bars.update(symbols)
        return await self._subscribe("bars", symbols)
    
    async def subscribe_daily_bars(self, symbols: list[str]) -> bool:
        """Subscribe to daily bars."""
        self._subscribed_daily_bars.update(symbols)
        return await self._subscribe("dailyBars", symbols)
    
    async def subscribe_statuses(self, symbols: list[str]) -> bool:
        """Subscribe to trading status events."""
        self._subscribed_statuses.update(symbols)
        return await self._subscribe("statuses", symbols)
    
    async def subscribe_lulds(self, symbols: list[str]) -> bool:
        """Subscribe to LULD bands."""
        self._subscribed_lulds.update(symbols)
        return await self._subscribe("lulds", symbols)
    
    async def subscribe_imbalances(self, symbols: list[str]) -> bool:
        """Subscribe to order imbalances."""
        self._subscribed_imbalances.update(symbols)
        return await self._subscribe("imbalances", symbols)
    
    async def unsubscribe_quotes(self, symbols: list[str]) -> bool:
        """Unsubscribe from quotes."""
        self._subscribed_quotes -= set(symbols)
        return await self._unsubscribe("quotes", symbols)
    
    async def unsubscribe_trades(self, symbols: list[str]) -> bool:
        """Unsubscribe from trades."""
        self._subscribed_trades -= set(symbols)
        return await self._unsubscribe("trades", symbols)
    
    # ========================================================================
    # CALLBACKS
    # ========================================================================
    
    def on_quote(self, callback: Callable[[AlpacaQuote], None]):
        """Register quote callback."""
        self._quote_callbacks.append(callback)
        return self
    
    def on_trade(self, callback: Callable[[AlpacaTrade], None]):
        """Register trade callback."""
        self._trade_callbacks.append(callback)
        return self
    
    def on_bar(self, callback: Callable[[AlpacaBar], None]):
        """Register bar callback."""
        self._bar_callbacks.append(callback)
        return self
    
    def on_luld(self, callback: Callable[[AlpacaLULD], None]):
        """Register LULD callback."""
        self._luld_callbacks.append(callback)
        return self
    
    def on_status(self, callback: Callable[[AlpacaTradingStatus], None]):
        """Register trading status callback."""
        self._status_callbacks.append(callback)
        return self
    
    def on_imbalance(self, callback: Callable[[AlpacaImbalance], None]):
        """Register imbalance callback."""
        self._imbalance_callbacks.append(callback)
        return self
    
    def on_error(self, callback: Callable[[dict], None]):
        """Register error callback."""
        self._error_callbacks.append(callback)
        return self
    
    # ========================================================================
    # MESSAGE PROCESSING
    # ========================================================================
    
    async def run(self):
        """
        Run the message processing loop.
        
        This should be called after connect() and subscribe_*.
        Runs until disconnect() is called.
        """
        self._running = True
        
        while self._running:
            if not self.is_connected:
                if not await self._reconnect():
                    break
                continue
            
            try:
                msg = await self._ws.receive()
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._last_message_time = time.time()
                    await self._process_message(msg.data)
                    
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed")
                    if self._running:
                        await self._reconnect()
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg}")
                    if self._running:
                        await self._reconnect()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                if self._running:
                    await asyncio.sleep(1)
    
    async def _process_message(self, raw: str):
        """Process incoming WebSocket message."""
        try:
            messages = json.loads(raw)
            
            # Messages come in arrays
            if not isinstance(messages, list):
                messages = [messages]
            
            for msg in messages:
                self._message_count += 1
                msg_type = msg.get("T")
                
                if msg_type == MessageType.QUOTE.value:
                    await self._handle_quote(msg)
                elif msg_type == MessageType.TRADE.value:
                    await self._handle_trade(msg)
                elif msg_type in (MessageType.BAR.value, MessageType.DAILY_BAR.value, MessageType.UPDATED_BAR.value):
                    await self._handle_bar(msg)
                elif msg_type == MessageType.LULD.value:
                    await self._handle_luld(msg)
                elif msg_type == MessageType.STATUS.value:
                    await self._handle_status(msg)
                elif msg_type == MessageType.IMBALANCE.value:
                    await self._handle_imbalance(msg)
                elif msg_type == MessageType.SUBSCRIPTION.value:
                    self._handle_subscription(msg)
                elif msg_type == MessageType.ERROR.value:
                    self._handle_error(msg)
                elif msg_type == MessageType.SUCCESS.value:
                    pass  # Already handled in connect/auth
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    async def _handle_quote(self, data: dict):
        """Handle quote message."""
        try:
            quote = AlpacaQuote(
                symbol=data["S"],
                bid_price=data["bp"],
                bid_size=data["bs"],
                ask_price=data["ap"],
                ask_size=data["as"],
                bid_exchange=data["bx"],
                ask_exchange=data["ax"],
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                conditions=data.get("c", []),
                tape=data.get("z", ""),
            )
            
            self._quote_count += 1
            
            for callback in self._quote_callbacks:
                try:
                    result = callback(quote)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Quote callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse quote: {e}")
    
    async def _handle_trade(self, data: dict):
        """Handle trade message."""
        try:
            trade = AlpacaTrade(
                symbol=data["S"],
                price=data["p"],
                size=data["s"],
                exchange=data["x"],
                trade_id=data["i"],
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                conditions=data.get("c", []),
                tape=data.get("z", ""),
            )
            
            self._trade_count += 1
            
            for callback in self._trade_callbacks:
                try:
                    result = callback(trade)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Trade callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse trade: {e}")
    
    async def _handle_bar(self, data: dict):
        """Handle bar message."""
        try:
            bar_type = {
                "b": "minute",
                "d": "daily",
                "u": "updated",
            }.get(data["T"], "minute")
            
            bar = AlpacaBar(
                symbol=data["S"],
                open=data["o"],
                high=data["h"],
                low=data["l"],
                close=data["c"],
                volume=data["v"],
                vwap=data.get("vw", 0),
                trade_count=data.get("n", 0),
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                bar_type=bar_type,
            )
            
            for callback in self._bar_callbacks:
                try:
                    result = callback(bar)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Bar callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse bar: {e}")
    
    async def _handle_luld(self, data: dict):
        """Handle LULD message."""
        try:
            luld = AlpacaLULD(
                symbol=data["S"],
                limit_up=data["u"],
                limit_down=data["d"],
                indicator=data["i"],
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                tape=data.get("z", ""),
            )
            
            for callback in self._luld_callbacks:
                try:
                    result = callback(luld)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"LULD callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse LULD: {e}")
    
    async def _handle_status(self, data: dict):
        """Handle trading status message."""
        try:
            status = AlpacaTradingStatus(
                symbol=data["S"],
                status_code=data["sc"],
                status_message=data["sm"],
                reason_code=data.get("rc", ""),
                reason_message=data.get("rm", ""),
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                tape=data.get("z", ""),
            )
            
            for callback in self._status_callbacks:
                try:
                    result = callback(status)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Status callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse status: {e}")
    
    async def _handle_imbalance(self, data: dict):
        """Handle order imbalance message."""
        try:
            imbalance = AlpacaImbalance(
                symbol=data["S"],
                price=data["p"],
                timestamp=datetime.fromisoformat(data["t"].replace("Z", "+00:00")),
                tape=data.get("z", ""),
            )
            
            for callback in self._imbalance_callbacks:
                try:
                    result = callback(imbalance)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Imbalance callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to parse imbalance: {e}")
    
    def _handle_subscription(self, data: dict):
        """Handle subscription confirmation."""
        logger.debug(f"Subscription updated: trades={data.get('trades', [])}, "
                    f"quotes={data.get('quotes', [])}, bars={data.get('bars', [])}")
    
    def _handle_error(self, data: dict):
        """Handle error message."""
        code = data.get("code", 0)
        msg = data.get("msg", "Unknown error")
        
        logger.error(f"Alpaca error [{code}]: {msg}")
        
        for callback in self._error_callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error callback error: {e}")
    
    # ========================================================================
    # METRICS
    # ========================================================================
    
    def get_stats(self) -> dict:
        """Get streaming statistics."""
        uptime = 0
        if self._connect_time:
            uptime = time.time() - self._connect_time
        
        return {
            "connected": self.is_connected,
            "authenticated": self.is_authenticated,
            "feed": self.feed.value,
            "uptime_seconds": uptime,
            "message_count": self._message_count,
            "quote_count": self._quote_count,
            "trade_count": self._trade_count,
            "subscribed_quotes": list(self._subscribed_quotes),
            "subscribed_trades": list(self._subscribed_trades),
            "subscribed_bars": list(self._subscribed_bars),
            "reconnect_attempts": self._reconnect_attempts,
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def create_and_connect_client(
    symbols: list[str],
    subscribe_quotes: bool = True,
    subscribe_trades: bool = True,
    subscribe_bars: bool = False,
    feed: AlpacaFeed = AlpacaFeed.IEX,
) -> AlpacaStreamClient:
    """
    Create and connect an Alpaca streaming client.
    
    Args:
        symbols: Symbols to subscribe to
        subscribe_quotes: Subscribe to quotes
        subscribe_trades: Subscribe to trades
        subscribe_bars: Subscribe to minute bars
        feed: Data feed to use
    
    Returns:
        Connected AlpacaStreamClient
    """
    client = AlpacaStreamClient(feed=feed)
    
    if not await client.connect():
        raise ConnectionError("Failed to connect to Alpaca stream")
    
    if subscribe_quotes:
        await client.subscribe_quotes(symbols)
    if subscribe_trades:
        await client.subscribe_trades(symbols)
    if subscribe_bars:
        await client.subscribe_bars(symbols)
    
    return client


# ============================================================================
# GLOBAL CLIENT INSTANCE
# ============================================================================


_global_client: AlpacaStreamClient | None = None


async def get_alpaca_stream() -> AlpacaStreamClient:
    """Get or create global Alpaca stream client."""
    global _global_client
    
    if _global_client is None or not _global_client.is_connected:
        _global_client = AlpacaStreamClient()
        await _global_client.connect()
    
    return _global_client


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "AlpacaStreamClient",
    "AlpacaQuote",
    "AlpacaTrade", 
    "AlpacaBar",
    "AlpacaLULD",
    "AlpacaTradingStatus",
    "AlpacaImbalance",
    "AlpacaFeed",
    "create_and_connect_client",
    "get_alpaca_stream",
]
