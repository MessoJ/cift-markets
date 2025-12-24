"""
CIFT Markets - Polygon.io Real-Time Market Data Service

Fetches real-time market data and news from Polygon.io (now Massive.com).
Automatically updates market_data_cache and provides WebSocket streaming.

API Documentation: https://polygon.io/docs/
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import aiohttp
from loguru import logger

from cift.core.config import settings
from cift.core.database import get_postgres_pool
from cift.services.finnhub_realtime_service import FinnhubRealtimeService
from cift.services.alltick_service import AlltickService


class PolygonRealtimeService:
    """
    Real-time market data service using Polygon.io API.

    Features:
    - Real-time quotes and snapshots
    - Historical OHLCV data
    - Market news with sentiment
    - Automatic cache updates
    - WebSocket streaming support
    """

    # API endpoints
    BASE_URL = "https://api.polygon.io"

    # Popular symbols to track
    DEFAULT_SYMBOLS = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "BRK.B",
        "JPM",
        "V",
        "JNJ",
        "WMT",
        "PG",
        "XOM",
        "UNH",
        "HD",
        "MA",
        "COST",
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "VTI",
        "VOO",
    ]

    # Index symbols
    INDEX_SYMBOLS = {
        "SPX": "I:SPX",  # S&P 500
        "NDX": "I:NDX",  # NASDAQ 100
        "DJI": "I:DJI",  # Dow Jones
        "VIX": "I:VIX",  # VIX
    }

    def __init__(self, api_key: str | None = None):
        """Initialize Polygon service."""
        self.api_key = api_key or settings.polygon_api_key

        if not self.api_key:
            logger.warning("Polygon API key not configured - using mock data")
            self._mock_mode = True
        else:
            self._mock_mode = False
            logger.info("Polygon.io service initialized with API key")

        self.session: aiohttp.ClientSession | None = None
        self._initialized = False
        self._rate_limit_remaining = 5
        self._rate_limit_reset = datetime.utcnow()
        
        # Fallback Services
        self.finnhub = FinnhubRealtimeService()
        self.alltick = AlltickService()

    async def initialize(self):
        """Initialize HTTP session."""
        if self._initialized:
            return

        connector = aiohttp.TCPConnector(limit=50, limit_per_host=20, ttl_dns_cache=300)

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
        # Initialize fallbacks
        await self.finnhub.initialize()
        await self.alltick.initialize()

        self._initialized = True
        logger.info("Polygon HTTP session initialized")

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self._initialized = False
            logger.info("Polygon session closed")
            
        await self.finnhub.close()
        await self.alltick.close()

    async def _request(self, endpoint: str, params: dict | None = None) -> dict[str, Any] | None:
        """Make HTTP request to Polygon API."""
        if self._mock_mode:
            return None

        if not self._initialized:
            await self.initialize()

        url = f"{self.BASE_URL}{endpoint}"

        # Add API key to params
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        try:
            async with self.session.get(url, params=params) as response:
                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._request(endpoint, params)

                # Handle Forbidden (403) - Fallback to Mock Mode
                if response.status == 403:
                    logger.error("Polygon API 403 Forbidden - Invalid Key or Expired Subscription. Switching to Mock Mode.")
                    self._mock_mode = True
                    return None

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            logger.error(f"Polygon API error: {e.status} - {e.message}")
            return None

        except Exception as e:
            logger.error(f"Polygon request failed: {e}")
            return None

    # ========================================================================
    # REAL-TIME QUOTES
    # ========================================================================

    async def get_snapshot(self, symbol: str) -> dict[str, Any] | None:
        """Get real-time snapshot for a symbol."""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        return await self._request(endpoint)

    async def get_all_snapshots(self) -> dict[str, Any] | None:
        """Get snapshots for all tickers (requires paid plan)."""
        endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
        return await self._request(endpoint)

    def _generate_mock_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Generate realistic mock quotes when API is unavailable."""
        import random
        quotes = {}
        
        # Base prices for common symbols to make them look realistic
        base_prices = {
            "AAPL": 175.0, "MSFT": 380.0, "GOOGL": 140.0, "AMZN": 145.0,
            "TSLA": 240.0, "META": 350.0, "NVDA": 480.0, "AMD": 110.0,
            "SPY": 470.0, "QQQ": 400.0, "IWM": 195.0, "DIA": 370.0,
            "BTC": 45000.0, "ETH": 2400.0
        }
        
        for symbol in symbols:
            # Get base price or generate random one based on hash of symbol
            if symbol in base_prices:
                base = base_prices[symbol]
            else:
                # Deterministic random base price from symbol string
                seed = sum(ord(c) for c in symbol)
                base = 50.0 + (seed % 450)
            
            # Add some random fluctuation
            volatility = 0.02  # 2% volatility
            change_pct = random.uniform(-volatility, volatility) * 100
            current_price = base * (1 + change_pct / 100)
            
            open_price = base
            high_price = max(open_price, current_price) * (1 + random.uniform(0, 0.005))
            low_price = min(open_price, current_price) * (1 - random.uniform(0, 0.005))
            
            quotes[symbol] = {
                "symbol": symbol,
                "price": round(current_price, 2),
                "bid": round(current_price * 0.9998, 2),
                "ask": round(current_price * 1.0002, 2),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(current_price, 2),
                "volume": int(random.uniform(100000, 5000000)),
                "prev_close": round(base, 2),
                "change": round(current_price - base, 2),
                "change_percent": round(change_pct, 2),
            }
            
        return quotes

    async def get_quotes_batch(self, symbols: list[str]) -> dict[str, dict]:
        """Get quotes for multiple symbols using real-time snapshots with fallback."""
        # Check for mock mode first
        if self._mock_mode:
            return self._generate_mock_quotes(symbols)

        quotes = {}

        for symbol in symbols:
            try:
                # 1. Try Real-time Snapshot (Best for paid plans)
                data = await self.get_snapshot(symbol)
                
                if data and data.get("status") == "OK":
                    ticker = data.get("ticker", {})
                    last_trade = ticker.get("lastTrade", {})
                    last_quote = ticker.get("lastQuote", {})
                    day = ticker.get("day", {})
                    prev_day = ticker.get("prevDay", {})
                    
                    current_price = last_trade.get("p") or day.get("c") or prev_day.get("c") or 0
                    open_price = day.get("o") or prev_day.get("c") or 0
                    
                    # Calculate change
                    if open_price and current_price:
                        change = current_price - open_price
                        change_pct = (change / open_price * 100)
                    else:
                        change = ticker.get("todaysChange", 0)
                        change_pct = ticker.get("todaysChangePerc", 0)

                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": float(current_price),
                        "bid": float(last_quote.get("p", 0)) if last_quote.get("p") else None,
                        "ask": float(last_quote.get("P", 0)) if last_quote.get("P") else None,
                        "open": float(open_price),
                        "high": float(day.get("h") or 0),
                        "low": float(day.get("l") or 0),
                        "close": float(current_price),
                        "volume": int(day.get("v") or 0),
                        "prev_close": float(prev_day.get("c", 0)),
                        "change": float(change),
                        "change_percent": float(change_pct),
                    }
                    logger.debug(f"Fetched snapshot for {symbol}: ${current_price}")
                    continue # Success, move to next symbol

                # 2. Fallback: Finnhub (Free Real-time)
                logger.warning(f"Polygon snapshot failed for {symbol}, trying Finnhub")
                finnhub_quote = await self.finnhub.get_quote(symbol)
                if finnhub_quote:
                    # Ensure bid/ask keys exist
                    if "bid" not in finnhub_quote:
                        finnhub_quote["bid"] = None
                    if "ask" not in finnhub_quote:
                        finnhub_quote["ask"] = None
                        
                    quotes[symbol] = finnhub_quote
                    logger.debug(f"Fetched Finnhub quote for {symbol}: ${finnhub_quote['price']}")
                    continue

                # 3. Fallback: Alltick (Global)
                logger.warning(f"Finnhub failed for {symbol}, trying Alltick")
                alltick_quote = await self.alltick.get_quote(symbol)
                if alltick_quote:
                    # Ensure bid/ask keys exist
                    if "bid" not in alltick_quote:
                        alltick_quote["bid"] = None
                    if "ask" not in alltick_quote:
                        alltick_quote["ask"] = None
                        
                    quotes[symbol] = alltick_quote
                    continue

                # 4. Fallback: Previous Close (Free tier compatible)
                logger.warning(f"Alltick failed for {symbol}, falling back to previous close")
                data = await self.get_previous_close(symbol)
                
                if data and data.get("status") == "OK" and data.get("results"):
                    bar = data["results"][0]
                    current_price = bar.get("c", 0)
                    open_price = bar.get("o", 0)
                    
                    if open_price:
                        change = current_price - open_price
                        change_pct = (change / open_price * 100)
                    else:
                        change = 0
                        change_pct = 0

                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": float(current_price),
                        "bid": None,
                        "ask": None,
                        "open": float(open_price),
                        "high": float(bar.get("h", 0)),
                        "low": float(bar.get("l", 0)),
                        "close": float(current_price),
                        "volume": int(bar.get("v", 0)),
                        "prev_close": float(open_price), # Approx
                        "change": float(change),
                        "change_percent": float(change_pct),
                    }
                    logger.debug(f"Fetched prev close for {symbol}: ${current_price}")

            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")

        return quotes

    async def get_previous_close(self, symbol: str) -> dict[str, Any] | None:
        """Get previous day's OHLC."""
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        return await self._request(endpoint)

    # ========================================================================
    # HISTORICAL DATA
    # ========================================================================

    def _generate_mock_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        from_date: datetime,
        to_date: datetime,
    ) -> list[dict]:
        """Generate mock historical data."""
        import random
        results = []
        
        # Determine time delta based on timespan
        if timespan == "minute":
            delta = timedelta(minutes=multiplier)
        elif timespan == "hour":
            delta = timedelta(hours=multiplier)
        elif timespan == "day":
            delta = timedelta(days=multiplier)
        else:
            delta = timedelta(days=1) # Default
            
        current_time = from_date
        
        # Base price
        seed = sum(ord(c) for c in symbol)
        price = 50.0 + (seed % 450)
        
        while current_time <= to_date:
            # Random walk
            change = random.uniform(-0.005, 0.005)
            price = price * (1 + change)
            
            high = price * (1 + random.uniform(0, 0.002))
            low = price * (1 - random.uniform(0, 0.002))
            
            results.append({
                "t": int(current_time.timestamp() * 1000),
                "o": round(price, 2),
                "h": round(high, 2),
                "l": round(low, 2),
                "c": round(price, 2),
                "v": int(random.uniform(1000, 50000)),
                "n": int(random.uniform(10, 100)),
                "vw": round(price, 2)
            })
            
            current_time += delta
            
        return results

    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int = 1,
        timespan: str = "minute",
        from_date: datetime = None,
        to_date: datetime = None,
        limit: int = 5000,
    ) -> list[dict]:
        """Get OHLCV bars."""
        if from_date is None:
            from_date = datetime.utcnow() - timedelta(days=7)
        if to_date is None:
            to_date = datetime.utcnow()

        # Check mock mode
        if self._mock_mode:
            return self._generate_mock_aggregates(symbol, multiplier, timespan, from_date, to_date)

        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_str}/{to_str}"

        params = {"limit": limit, "adjusted": "true", "sort": "asc"}

        data = await self._request(endpoint, params)

        if data and data.get("results"):
            return data["results"]
        return []

    # ========================================================================
    # NEWS
    # ========================================================================

    async def get_news(
        self, ticker: str | None = None, limit: int = 50, order: str = "desc"
    ) -> list[dict]:
        """
        Get market news from Polygon.

        Args:
            ticker: Optional symbol to filter news
            limit: Max articles to return
            order: Sort order (asc/desc by published date)

        Returns:
            List of news articles
        """
        endpoint = "/v2/reference/news"

        params = {"limit": limit, "order": order, "sort": "published_utc"}

        if ticker:
            params["ticker"] = ticker

        data = await self._request(endpoint, params)

        if data and data.get("results"):
            return data["results"]
        return []

    async def fetch_and_store_news(self, symbols: list[str] | None = None, limit: int = 50) -> int:
        """
        Fetch news from Polygon and store in database.

        Returns:
            Number of articles stored
        """
        articles = []

        # Get general market news
        general_news = await self.get_news(limit=limit)
        articles.extend(general_news)

        # Get symbol-specific news
        if symbols:
            for symbol in symbols[:10]:  # Limit to 10 symbols
                symbol_news = await self.get_news(ticker=symbol, limit=10)
                articles.extend(symbol_news)
                await asyncio.sleep(0.5)  # Rate limiting

        if not articles:
            logger.warning("No news fetched from Polygon")
            return 0

        # Store in database
        pool = await get_postgres_pool()
        stored = 0

        async with pool.acquire() as conn:
            for article in articles:
                try:
                    # Extract data - generate proper UUID from article URL or id
                    polygon_id = article.get("id", "")
                    article_url = article.get("article_url", "")

                    # Generate UUID from the hash of URL (deterministic)
                    import uuid

                    article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, article_url or polygon_id))

                    title = article.get("title", "")
                    summary = article.get("description", "")
                    url = article_url
                    source = article.get("publisher", {}).get("name", "Unknown")
                    author = article.get("author", "")

                    # Parse published time - make timezone-naive for database
                    published_str = article.get("published_utc", "")
                    if published_str:
                        published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                        # Convert to naive datetime for database
                        published_at = published_at.replace(tzinfo=None)
                    else:
                        published_at = datetime.utcnow()

                    # Get symbols mentioned
                    tickers = article.get("tickers", [])
                    import json
                    tickers_json = json.dumps(tickers)

                    # Get image
                    image_url = article.get("image_url", "")

                    # Determine sentiment from keywords (basic)
                    sentiment = self._analyze_sentiment(title + " " + summary)

                    # Categories from insights
                    insights = article.get("insights", [])
                    categories = [i.get("sentiment", "neutral") for i in insights]
                    if not categories:
                        categories = ["general"]

                    await conn.execute(
                        """
                        INSERT INTO news_articles (
                            id, title, summary, content, url, source, author,
                            published_at, symbols, category, sentiment,
                            image_url, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            symbols = EXCLUDED.symbols,
                            sentiment = EXCLUDED.sentiment
                    """,
                        article_id,
                        title[:500],
                        summary[:1000] if summary else "",
                        summary,
                        url,
                        source[:100],
                        author[:200] if author else None,
                        published_at,
                        tickers_json,  # json string for jsonb
                        categories[0] if categories else "general",
                        sentiment,
                        image_url,
                        datetime.utcnow(),
                    )
                    stored += 1

                except Exception as e:
                    logger.warning(f"Failed to store article: {e}")

        logger.success(f"Stored {stored} news articles from Polygon")
        return stored

    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis from keywords."""
        text_lower = text.lower()

        positive_words = [
            "surge",
            "rally",
            "gain",
            "rise",
            "up",
            "bullish",
            "record",
            "beat",
            "exceed",
            "growth",
            "positive",
            "strong",
        ]
        negative_words = [
            "fall",
            "drop",
            "decline",
            "down",
            "bearish",
            "miss",
            "below",
            "weak",
            "negative",
            "loss",
            "crash",
            "fear",
        ]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    # ========================================================================
    # CACHE UPDATE
    # ========================================================================

    async def update_market_cache(self, symbols: list[str] | None = None) -> int:
        """
        Update market_data_cache with live quotes from Polygon.

        Returns:
            Number of symbols updated
        """
        if symbols is None:
            # Include indices and crypto in default update
            symbols = self.DEFAULT_SYMBOLS + list(self.INDEX_SYMBOLS.keys()) + ["BTC-USD", "ETH-USD", "GC=F", "CL=F"]

        logger.info(f"Updating market cache for {len(symbols)} symbols...")

        quotes = await self.get_quotes_batch(symbols)

        if not quotes:
            logger.warning("No quotes received from Polygon")
            return 0

        pool = await get_postgres_pool()
        updated = 0

        async with pool.acquire() as conn:
            for symbol, quote in quotes.items():
                try:
                    await conn.execute(
                        """
                        INSERT INTO market_data_cache (
                            symbol, price, bid, ask, open, high, low, close, volume,
                            prev_close, change, change_pct, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                        ON CONFLICT (symbol) DO UPDATE SET
                            price = EXCLUDED.price,
                            bid = EXCLUDED.bid,
                            ask = EXCLUDED.ask,
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume,
                            prev_close = EXCLUDED.prev_close,
                            change = EXCLUDED.change,
                            change_pct = EXCLUDED.change_pct,
                            updated_at = NOW()
                    """,
                        symbol,
                        quote["price"],
                        quote.get("bid"),
                        quote.get("ask"),
                        quote["open"],
                        quote["high"],
                        quote["low"],
                        quote["close"],
                        int(quote["volume"]),
                        quote["prev_close"],
                        quote["change"],
                        quote["change_percent"],
                    )
                    updated += 1

                except Exception as e:
                    logger.error(f"Failed to update cache for {symbol}: {e}")

        logger.success(f"Updated market cache for {updated} symbols")
        return updated

    async def update_ohlcv_bars(
        self, symbols: list[str] | None = None, days: int = 5, timespan: str = "minute"
    ) -> int:
        """
        Fetch and store OHLCV bars to PostgreSQL.

        Returns:
            Total bars stored
        """
        if symbols is None:
            symbols = self.DEFAULT_SYMBOLS[:8]  # Limit for rate limiting

        logger.info(f"Fetching {days} days of {timespan} bars for {len(symbols)} symbols...")

        pool = await get_postgres_pool()
        total_bars = 0

        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=days)

        for symbol in symbols:
            try:
                bars = await self.get_aggregates(
                    symbol=symbol,
                    multiplier=1,
                    timespan=timespan,
                    from_date=from_date,
                    to_date=to_date,
                    limit=5000,
                )

                if not bars:
                    logger.warning(f"No bars for {symbol}")
                    continue

                async with pool.acquire() as conn:
                    for bar in bars:
                        try:
                            # Convert timestamp from ms to datetime
                            timestamp = datetime.utcfromtimestamp(bar["t"] / 1000)

                            await conn.execute(
                                """
                                INSERT INTO ohlcv_bars (
                                    symbol, timestamp, timeframe, open, high, low, close, volume
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                                ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                                    open = EXCLUDED.open,
                                    high = EXCLUDED.high,
                                    low = EXCLUDED.low,
                                    close = EXCLUDED.close,
                                    volume = EXCLUDED.volume
                            """,
                                symbol,
                                timestamp,
                                "1min",
                                bar["o"],
                                bar["h"],
                                bar["l"],
                                bar["c"],
                                int(bar["v"]),
                            )
                            total_bars += 1

                        except Exception:
                            # Skip duplicates
                            pass

                logger.info(f"Stored {len(bars)} bars for {symbol}")

                # Rate limiting (free tier: 5 req/min)
                await asyncio.sleep(12)

            except Exception as e:
                logger.error(f"Error fetching bars for {symbol}: {e}")

        logger.success(f"Total bars stored: {total_bars}")
        return total_bars

    # ========================================================================
    # MARKET STATUS
    # ========================================================================

    async def get_market_status(self) -> dict[str, Any]:
        """Get current market status (open/closed)."""
        endpoint = "/v1/marketstatus/now"
        data = await self._request(endpoint)

        if data:
            return {
                "market": data.get("market", "unknown"),
                "serverTime": data.get("serverTime"),
                "exchanges": data.get("exchanges", {}),
                "currencies": data.get("currencies", {}),
            }

        return {"market": "unknown"}


# ============================================================================
# BACKGROUND WORKER
# ============================================================================


class PolygonBackgroundWorker:
    """
    Background worker that continuously updates market data.
    """

    def __init__(self, update_interval: int = 60):
        """
        Initialize background worker.

        Args:
            update_interval: Seconds between updates (default 60 for free tier)
        """
        self.service = PolygonRealtimeService()
        self.update_interval = update_interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start background updates."""
        if self._running:
            return

        self._running = True
        await self.service.initialize()
        self._task = asyncio.create_task(self._update_loop())
        logger.info(f"Polygon background worker started (interval: {self.update_interval}s)")

    async def stop(self):
        """Stop background updates."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.service.close()
        logger.info("Polygon background worker stopped")

    async def _update_loop(self):
        """Main update loop."""
        news_counter = 0

        while self._running:
            try:
                # Update market quotes every interval
                await self.service.update_market_cache()

                # Update news less frequently (every 5 intervals)
                news_counter += 1
                if news_counter >= 5:
                    await self.service.fetch_and_store_news(
                        symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"], limit=20
                    )
                    news_counter = 0

                await asyncio.sleep(self.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")
                await asyncio.sleep(self.update_interval)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

polygon_realtime_service = PolygonRealtimeService()
polygon_worker = PolygonBackgroundWorker()


# ============================================================================
# CLI COMMANDS
# ============================================================================


async def main():
    """CLI for testing Polygon service."""
    import sys

    service = PolygonRealtimeService()
    await service.initialize()

    try:
        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "quotes":
                symbols = sys.argv[2:] if len(sys.argv) > 2 else ["AAPL", "MSFT", "GOOGL"]
                quotes = await service.get_quotes_batch(symbols)
                for symbol, quote in quotes.items():
                    print(f"{symbol}: ${quote['price']:.2f} ({quote['change_percent']:+.2f}%)")

            elif command == "news":
                ticker = sys.argv[2] if len(sys.argv) > 2 else None
                news = await service.get_news(ticker=ticker, limit=5)
                for article in news:
                    print(f"- {article.get('title', 'No title')}")
                    print(f"  Source: {article.get('publisher', {}).get('name', 'Unknown')}")
                    print()

            elif command == "update-cache":
                count = await service.update_market_cache()
                print(f"Updated {count} symbols in cache")

            elif command == "update-news":
                count = await service.fetch_and_store_news(limit=50)
                print(f"Stored {count} news articles")

            elif command == "fetch-bars":
                symbols = sys.argv[2:] if len(sys.argv) > 2 else ["AAPL"]
                count = await service.update_ohlcv_bars(symbols=symbols, days=5)
                print(f"Stored {count} OHLCV bars")

            elif command == "status":
                status = await service.get_market_status()
                print(f"Market status: {status}")

            else:
                print(
                    "Unknown command. Available: quotes, news, update-cache, update-news, fetch-bars, status"
                )
        else:
            print("Polygon.io Real-Time Service")
            print("Usage: python -m cift.services.polygon_realtime_service <command>")
            print("Commands: quotes, news, update-cache, update-news, fetch-bars, status")

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
