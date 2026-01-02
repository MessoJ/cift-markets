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

Real-time trade data is:
1. Stored in QuestDB for Time & Sales persistence
2. Cached in Dragonfly for fast retrieval (last 200 trades per symbol)
3. Broadcast to WebSocket clients for live updates
4. Aggregated into 1-minute bars for real-time chart updates
"""

import asyncio
import json
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime

import aiohttp
from loguru import logger

from cift.core.config import settings
from cift.core.database import get_postgres_pool, questdb_manager, redis_manager


# Bar aggregation state for 1-minute bars
_bar_aggregators: dict[str, dict] = {}  # {symbol: {open, high, low, close, volume, start_time}}
_bar_lock = asyncio.Lock()


async def _aggregate_trade_to_bar(symbol: str, price: float, volume: int, timestamp_ms: int):
    """
    Aggregate a trade into 1-minute bars and broadcast updates.
    
    Called for each incoming trade from Finnhub WebSocket.
    Broadcasts bar updates to /ws/bars subscribers.
    """
    global _bar_aggregators
    
    from cift.api.routes.market_data import publish_bar_update
    
    async with _bar_lock:
        now = datetime.utcnow()
        # Round down to current minute
        current_minute = now.replace(second=0, microsecond=0)
        
        if symbol not in _bar_aggregators:
            # Start new bar
            _bar_aggregators[symbol] = {
                "o": price,
                "h": price,
                "l": price,
                "c": price,
                "v": volume,
                "t": current_minute.isoformat(),
                "start_time": current_minute
            }
        else:
            bar = _bar_aggregators[symbol]
            bar_start = bar.get("start_time")
            
            if bar_start < current_minute:
                # New minute - publish completed bar and start new one
                try:
                    await publish_bar_update(symbol, "1m", bar)
                except Exception as e:
                    logger.debug(f"Bar publish error: {e}")
                
                # Start new bar
                _bar_aggregators[symbol] = {
                    "o": price,
                    "h": price,
                    "l": price,
                    "c": price,
                    "v": volume,
                    "t": current_minute.isoformat(),
                    "start_time": current_minute
                }
            else:
                # Update current bar
                bar["h"] = max(bar["h"], price)
                bar["l"] = min(bar["l"], price)
                bar["c"] = price
                bar["v"] += volume


class FinnhubRealtimeService:
    """
    Finnhub real-time WebSocket service.

    Features:
    - FREE real-time quotes via WebSocket
    - Up to 50 symbols per connection
    - Automatic reconnection
    - Price update callbacks for UI
    - Real-time trade persistence to QuestDB
    - In-memory trade cache for Time & Sales
    """

    # WebSocket endpoint
    WS_URL = "wss://ws.finnhub.io"

    # REST API endpoint
    REST_URL = "https://finnhub.io/api/v1"
    
    # Trade cache settings
    MAX_TRADES_PER_SYMBOL = 200  # Keep last 200 trades per symbol in memory
    CACHE_TTL_SECONDS = 3600    # Redis cache TTL (1 hour)
    
    # QuestDB batch settings
    BATCH_SIZE = 100            # Batch inserts for performance
    BATCH_INTERVAL = 1.0        # Flush every 1 second

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
        
        # Real-time trade storage
        self._trade_cache: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.MAX_TRADES_PER_SYMBOL))
        self._trade_batch: list[dict] = []  # Batch for QuestDB inserts
        self._batch_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

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

        # Check cache first (Aggressive caching for Free Tier)
        cache_key = f"finnhub:quote:{symbol}"
        cached = await redis_manager.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

        await self.initialize()

        url = f"{self.REST_URL}/quote"
        params = {"symbol": symbol, "token": self.api_key}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    current_price = data.get("c", 0)
                    result = {
                        "symbol": symbol,
                        "price": current_price,  # Current price
                        "open": data.get("o", 0),
                        "high": data.get("h", 0),
                        "low": data.get("l", 0),
                        "close": current_price,  # Same as current price for real-time
                        "volume": 0,  # Finnhub doesn't provide volume in quote endpoint
                        "prev_close": data.get("pc", 0),
                        "change": data.get("d", 0),
                        "change_percent": data.get("dp", 0),
                        "timestamp": data.get("t", 0),
                    }
                    # Cache for 60 seconds to respect rate limits
                    await redis_manager.set(cache_key, json.dumps(result), expire=60)
                    return result
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

    async def get_company_profile(self, symbol: str) -> dict | None:
        """Get company profile (Fundamental Data)."""
        if not self._available:
            return None
        await self.initialize()
        url = f"{self.REST_URL}/stock/profile2"
        params = {"symbol": symbol, "token": self.api_key}
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Finnhub profile failed: {e}")
            return None

    async def get_financials(self, symbol: str) -> dict | None:
        """Get basic financials."""
        if not self._available:
            return None
        await self.initialize()
        url = f"{self.REST_URL}/stock/metric"
        params = {"symbol": symbol, "metric": "all", "token": self.api_key}
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Finnhub financials failed: {e}")
            return None

    async def get_candles(self, symbol: str, resolution: str, from_ts: int, to_ts: int) -> dict | None:
        """Get stock candles (Historical Data)."""
        if not self._available:
            return None
            
        # Check cache first (Aggressive caching for Free Tier)
        # Cache key includes resolution and time range (rounded to hour)
        cache_key = f"finnhub:candles:{symbol}:{resolution}:{from_ts}:{to_ts}"
        cached = await redis_manager.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
                
        await self.initialize()
        url = f"{self.REST_URL}/stock/candle"
        params = {
            "symbol": symbol,
            "resolution": resolution, # 1, 5, 15, 30, 60, D, W, M
            "from": from_ts,
            "to": to_ts,
            "token": self.api_key
        }
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Cache for 1 hour
                    await redis_manager.set(cache_key, json.dumps(data), expire=3600)
                    return data
                return None
        except Exception as e:
            logger.error(f"Finnhub candles failed: {e}")
            return None

    async def get_financials_reported(self, symbol: str, freq: str = "quarterly") -> dict | None:
        """Get reported financial statements (Income, Balance Sheet, Cash Flow)."""
        if not self._available:
            return None
        await self.initialize()
        url = f"{self.REST_URL}/stock/financials-reported"
        params = {"symbol": symbol, "freq": freq, "token": self.api_key}
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Finnhub reported financials failed: {e}")
            return None

    async def get_earnings_estimates(self, symbol: str) -> dict | None:
        """Get earnings estimates."""
        if not self._available:
            return None
        await self.initialize()
        url = f"{self.REST_URL}/stock/eps-estimate"
        params = {"symbol": symbol, "token": self.api_key}
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Finnhub estimates failed: {e}")
            return None

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
        
        # Start periodic flush task for QuestDB batching
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
        logger.success(f"Started real-time streaming for {len(symbols)} symbols (trades -> QuestDB + Redis)")

    async def stop_streaming(self):
        """Stop streaming."""
        self._running = False
        
        # Cancel flush task
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush of remaining trades
        await self._flush_batch()
        
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
                    volume = trade.get("v", 0)
                    timestamp = trade.get("t")  # Unix milliseconds
                    conditions = trade.get("c", [])  # Trade conditions

                    if symbol and price:
                        # Create trade record
                        trade_record = {
                            "symbol": symbol,
                            "price": float(price),
                            "size": int(volume) if volume else 1,
                            "timestamp": timestamp,
                            "time": datetime.utcfromtimestamp(timestamp / 1000).isoformat() if timestamp else datetime.utcnow().isoformat(),
                            "side": self._infer_trade_side(symbol, price),  # Infer buy/sell
                            "exchange": "FINNHUB",
                            "conditions": conditions,
                            "trade_id": str(uuid.uuid4())[:8],  # Short unique ID
                        }
                        
                        # 1. Update in-memory cache (instant access)
                        self._trade_cache[symbol].appendleft(trade_record)
                        
                        # 2. Add to batch for QuestDB persistence
                        await self._add_to_batch(trade_record)
                        
                        # 3. Update last price cache
                        self._last_prices[symbol] = {
                            "symbol": symbol,
                            "price": price,
                            "volume": volume,
                            "timestamp": timestamp,
                            "updated_at": datetime.utcnow(),
                        }

                        # 4. Call registered callbacks (for real-time UI updates)
                        for callback in self._callbacks:
                            try:
                                await callback(symbol, price, volume, timestamp)
                            except Exception as e:
                                logger.warning(f"Callback error: {e}")

                        # 5. Update market_data_cache (throttled)
                        await self._update_cache(symbol, price, volume)
                        
                        # 6. Cache in Redis/Dragonfly for fast Time & Sales retrieval
                        await self._cache_trade_to_redis(symbol, trade_record)
                        
                        # 7. Aggregate into 1-minute bars for WebSocket chart streaming
                        asyncio.create_task(_aggregate_trade_to_bar(
                            symbol, price, int(volume) if volume else 1, timestamp
                        ))

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
    
    def _infer_trade_side(self, symbol: str, price: float) -> str:
        """Infer trade side based on price movement (tick rule)."""
        last = self._last_prices.get(symbol)
        if last:
            last_price = last.get("price", 0)
            if price > last_price:
                return "buy"
            elif price < last_price:
                return "sell"
        return "unknown"
    
    async def _add_to_batch(self, trade: dict):
        """Add trade to batch for QuestDB insert."""
        async with self._batch_lock:
            self._trade_batch.append(trade)
            if len(self._trade_batch) >= self.BATCH_SIZE:
                await self._flush_batch()
    
    async def _flush_batch(self):
        """Flush trade batch to QuestDB."""
        if not self._trade_batch:
            return
            
        trades_to_insert = self._trade_batch.copy()
        self._trade_batch.clear()
        
        try:
            await questdb_manager.initialize()
            
            # Build batch INSERT
            values_parts = []
            for t in trades_to_insert:
                # Convert timestamp to datetime string for QuestDB
                ts = datetime.utcfromtimestamp(t["timestamp"] / 1000) if t.get("timestamp") else datetime.utcnow()
                ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                
                values_parts.append(
                    f"('{ts_str}', '{t['symbol']}', {t['price']}, {t['size']}, "
                    f"'{t['side']}', '{t['trade_id']}', '{t['exchange']}', '{t['side']}')"
                )
            
            if values_parts:
                query = f"""
                    INSERT INTO trade_executions 
                    (timestamp, symbol, price, size, side, trade_id, exchange, aggressive_side)
                    VALUES {', '.join(values_parts)}
                """
                await questdb_manager.execute(query)
                logger.debug(f"Flushed {len(trades_to_insert)} trades to QuestDB")
                
        except Exception as e:
            logger.error(f"Failed to flush trades to QuestDB: {e}")
            # Re-add failed trades to batch for retry
            async with self._batch_lock:
                self._trade_batch.extend(trades_to_insert[:50])  # Limit retry size
    
    async def _periodic_flush(self):
        """Periodically flush trade batch."""
        while self._running:
            await asyncio.sleep(self.BATCH_INTERVAL)
            async with self._batch_lock:
                if self._trade_batch:
                    await self._flush_batch()
    
    async def _cache_trade_to_redis(self, symbol: str, trade: dict):
        """Cache trade in Redis/Dragonfly for fast Time & Sales retrieval."""
        try:
            await redis_manager.initialize()
            
            # Use Redis list with capped size
            key = f"timesales:{symbol}"
            trade_json = json.dumps(trade)
            
            # LPUSH + LTRIM to keep last N trades
            await redis_manager.client.lpush(key, trade_json)
            await redis_manager.client.ltrim(key, 0, self.MAX_TRADES_PER_SYMBOL - 1)
            await redis_manager.client.expire(key, self.CACHE_TTL_SECONDS)
            
        except Exception as e:
            # Non-critical - log and continue
            logger.debug(f"Redis cache error for {symbol}: {e}")
    
    def get_recent_trades(self, symbol: str, limit: int = 50) -> list[dict]:
        """Get recent trades from in-memory cache."""
        trades = list(self._trade_cache.get(symbol, []))
        return trades[:limit]
    
    async def get_trades_from_cache(self, symbol: str, limit: int = 50) -> list[dict]:
        """Get trades from Redis cache."""
        try:
            await redis_manager.initialize()
            key = f"timesales:{symbol}"
            
            trades_json = await redis_manager.client.lrange(key, 0, limit - 1)
            return [json.loads(t) for t in trades_json]
            
        except Exception as e:
            logger.debug(f"Redis read error for {symbol}: {e}")
            # Fallback to in-memory cache
            return self.get_recent_trades(symbol, limit)

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
