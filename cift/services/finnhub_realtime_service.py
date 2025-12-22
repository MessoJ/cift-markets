"""
CIFT Markets - Finnhub Real-Time WebSocket Service

FREE real-time stock data via WebSocket.
Unlike Polygon (which requires paid tier for real-time), Finnhub provides
FREE WebSocket streaming for US stocks.

Limitations (Free Tier):
- 60 API calls/minute
- Real-time WebSocket for US stocks (FREE!)
- Up to 50 symbols per WebSocket connection

Get your free API key at: https://finnhub.io/

This service is COMPLEMENTARY to Polygon:
- Polygon: Best for news, historical data
- Finnhub: Best for real-time quotes (FREE WebSocket)
"""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime

import aiohttp
from loguru import logger

from cift.core.config import settings
from cift.core.database import get_postgres_pool


class FinnhubRealtimeService:
    """
    Finnhub real-time WebSocket service.

    Features:
    - FREE real-time quotes via WebSocket
    - Up to 50 symbols per connection
    - Automatic reconnection
    - Price update callbacks for UI
    """

    # WebSocket endpoint
    WS_URL = "wss://ws.finnhub.io"

    # REST API endpoint
    REST_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None):
        """Initialize Finnhub service."""
        self.api_key = api_key or getattr(settings, "finnhub_api_key", "")

        if not self.api_key:
            logger.warning("Finnhub API key not configured - real-time streaming unavailable")
            logger.info("Get a FREE API key at: https://finnhub.io/")
            self._available = False
        else:
            self._available = True
            logger.info("Finnhub service initialized with API key")

        self.session: aiohttp.ClientSession | None = None
        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self._subscribed_symbols: set[str] = set()
        self._running = False
        self._callbacks: list[Callable] = []
        self._last_prices: dict[str, dict] = {}

    @property
    def is_available(self) -> bool:
        """Check if service is available."""
        return self._available

    async def initialize(self):
        """Initialize HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close connections."""
        await self.disconnect_websocket()
        if self.session:
            await self.session.close()
            self.session = None

    # ========================================================================
    # REST API
    # ========================================================================

    async def get_quote(self, symbol: str) -> dict | None:
        """Get current quote for a symbol (REST API)."""
        if not self._available:
            return None

        await self.initialize()

        url = f"{self.REST_URL}/quote"
        params = {"symbol": symbol, "token": self.api_key}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": symbol,
                        "price": data.get("c", 0),  # Current price
                        "open": data.get("o", 0),
                        "high": data.get("h", 0),
                        "low": data.get("l", 0),
                        "prev_close": data.get("pc", 0),
                        "change": data.get("d", 0),
                        "change_percent": data.get("dp", 0),
                        "timestamp": data.get("t", 0),
                    }
                else:
                    logger.warning(f"Finnhub quote error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Finnhub quote failed: {e}")
            return None

    async def get_company_news(
        self, symbol: str, from_date: str = None, to_date: str = None
    ) -> list[dict]:
        """Get company news (complements Polygon news)."""
        if not self._available:
            return []

        await self.initialize()

        if from_date is None:
            from_date = datetime.utcnow().strftime("%Y-%m-%d")
        if to_date is None:
            to_date = datetime.utcnow().strftime("%Y-%m-%d")

        url = f"{self.REST_URL}/company-news"
        params = {"symbol": symbol, "from": from_date, "to": to_date, "token": self.api_key}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception as e:
            logger.error(f"Finnhub news failed: {e}")
            return []

    # ========================================================================
    # WEBSOCKET STREAMING
    # ========================================================================

    def add_callback(self, callback: Callable):
        """Add a callback for price updates."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def connect_websocket(self) -> bool:
        """Connect to Finnhub WebSocket."""
        if not self._available:
            logger.warning("Cannot connect WebSocket - no Finnhub API key")
            return False

        await self.initialize()

        try:
            ws_url = f"{self.WS_URL}?token={self.api_key}"
            self.ws = await self.session.ws_connect(ws_url)
            logger.success("Connected to Finnhub WebSocket")
            return True
        except Exception as e:
            logger.error(f"Finnhub WebSocket connection failed: {e}")
            return False

    async def disconnect_websocket(self):
        """Disconnect WebSocket."""
        if self.ws:
            await self.ws.close()
            self.ws = None
            self._subscribed_symbols.clear()
            logger.info("Disconnected from Finnhub WebSocket")

    async def subscribe(self, symbols: list[str]):
        """Subscribe to symbols for real-time updates."""
        if not self.ws:
            connected = await self.connect_websocket()
            if not connected:
                return

        for symbol in symbols:
            if symbol not in self._subscribed_symbols:
                await self.ws.send_json({"type": "subscribe", "symbol": symbol})
                self._subscribed_symbols.add(symbol)
                logger.debug(f"Subscribed to {symbol}")

        logger.info(f"Subscribed to {len(self._subscribed_symbols)} symbols")

    async def unsubscribe(self, symbols: list[str]):
        """Unsubscribe from symbols."""
        if not self.ws:
            return

        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                await self.ws.send_json({"type": "unsubscribe", "symbol": symbol})
                self._subscribed_symbols.discard(symbol)
                logger.debug(f"Unsubscribed from {symbol}")

    async def start_streaming(self, symbols: list[str] = None):
        """Start streaming real-time data."""
        if not self._available:
            logger.warning("Finnhub not available - no API key")
            return

        self._running = True

        # Default symbols to track
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY", "QQQ", "JPM"]

        # Connect and subscribe
        await self.subscribe(symbols)

        # Start message loop
        asyncio.create_task(self._message_loop())
        logger.success(f"Started real-time streaming for {len(symbols)} symbols")

    async def stop_streaming(self):
        """Stop streaming."""
        self._running = False
        await self.disconnect_websocket()
        logger.info("Stopped real-time streaming")

    async def _message_loop(self):
        """Main WebSocket message loop."""
        reconnect_delay = 1
        max_reconnect_delay = 60

        while self._running:
            try:
                if not self.ws or self.ws.closed:
                    logger.info("Reconnecting to Finnhub WebSocket...")
                    connected = await self.connect_websocket()
                    if not connected:
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue

                    # Resubscribe to symbols
                    symbols = list(self._subscribed_symbols)
                    self._subscribed_symbols.clear()
                    await self.subscribe(symbols)
                    reconnect_delay = 1

                # Wait for message
                msg = await asyncio.wait_for(self.ws.receive(), timeout=30)

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("Finnhub WebSocket closed")
                    self.ws = None

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Finnhub WebSocket error: {msg.data}")
                    self.ws = None

            except TimeoutError:
                # Send ping to keep connection alive
                if self.ws and not self.ws.closed:
                    await self.ws.ping()

            except Exception as e:
                logger.error(f"Finnhub message loop error: {e}")
                self.ws = None
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _handle_message(self, data: str):
        """Handle incoming WebSocket message."""
        try:
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "trade":
                # Real-time trade data
                trades = message.get("data", [])

                for trade in trades:
                    symbol = trade.get("s")
                    price = trade.get("p")
                    volume = trade.get("v")
                    timestamp = trade.get("t")

                    if symbol and price:
                        # Update last price cache
                        self._last_prices[symbol] = {
                            "symbol": symbol,
                            "price": price,
                            "volume": volume,
                            "timestamp": timestamp,
                            "updated_at": datetime.utcnow(),
                        }

                        # Call registered callbacks
                        for callback in self._callbacks:
                            try:
                                await callback(symbol, price, volume, timestamp)
                            except Exception as e:
                                logger.warning(f"Callback error: {e}")

                        # Update database cache (throttled)
                        await self._update_cache(symbol, price, volume)

            elif msg_type == "ping":
                # Respond to ping
                if self.ws:
                    await self.ws.send_json({"type": "pong"})

            elif msg_type == "error":
                logger.error(f"Finnhub error: {message.get('msg', 'Unknown')}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid Finnhub message: {data[:100]}")
        except Exception as e:
            logger.error(f"Message handling error: {e}")

    async def _update_cache(self, symbol: str, price: float, volume: int = None):
        """Update market_data_cache with real-time price (throttled)."""
        try:
            # Only update if price changed significantly or every 10 seconds
            last = self._last_prices.get(symbol, {})
            last_update = last.get("last_db_update")

            if last_update:
                seconds_since = (datetime.utcnow() - last_update).total_seconds()
                if seconds_since < 10:  # Throttle to every 10 seconds
                    return

            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE market_data_cache
                    SET price = $1, updated_at = NOW()
                    WHERE symbol = $2
                """,
                    price,
                    symbol,
                )

            self._last_prices[symbol]["last_db_update"] = datetime.utcnow()

        except Exception as e:
            logger.warning(f"Cache update failed for {symbol}: {e}")

    def get_last_price(self, symbol: str) -> dict | None:
        """Get last known price for a symbol."""
        return self._last_prices.get(symbol)

    def get_all_prices(self) -> dict[str, dict]:
        """Get all cached prices."""
        return self._last_prices.copy()


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_finnhub_service: FinnhubRealtimeService | None = None


def get_finnhub_service() -> FinnhubRealtimeService:
    """Get global Finnhub service instance."""
    global _finnhub_service
    if _finnhub_service is None:
        _finnhub_service = FinnhubRealtimeService()
    return _finnhub_service


# ============================================================================
# CLI TEST
# ============================================================================


async def main():
    """Test Finnhub service."""
    import sys

    service = FinnhubRealtimeService()

    if not service.is_available:
        print("❌ Finnhub API key not configured")
        print("Get your FREE key at: https://finnhub.io/")
        return

    await service.initialize()

    try:
        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "quote":
                symbol = sys.argv[2] if len(sys.argv) > 2 else "AAPL"
                quote = await service.get_quote(symbol)
                if quote:
                    print(f"{symbol}: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")
                else:
                    print(f"Failed to get quote for {symbol}")

            elif command == "stream":
                symbols = sys.argv[2:] if len(sys.argv) > 2 else ["AAPL", "MSFT"]

                async def print_trade(symbol, price, volume, timestamp):
                    print(f"📈 {symbol}: ${price:.2f} (vol: {volume})")

                service.add_callback(print_trade)
                await service.start_streaming(symbols)

                # Stream for 60 seconds
                await asyncio.sleep(60)
                await service.stop_streaming()

            else:
                print("Unknown command. Available: quote, stream")
        else:
            print("Finnhub Real-Time Service")
            print("Usage: python -m cift.services.finnhub_realtime_service <command>")
            print("Commands: quote [symbol], stream [symbols...]")

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
