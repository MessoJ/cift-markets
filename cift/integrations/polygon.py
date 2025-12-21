"""
CIFT Markets - Polygon.io Integration

High-quality market data from Polygon.io.

Features:
- Real-time quotes and trades
- Historical aggregates (bars)
- Company fundamentals
- Market status
- Reference data

API Documentation: https://polygon.io/docs/
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import aiohttp
from loguru import logger

from cift.core.config import settings

# ============================================================================
# CONSTANTS
# ============================================================================

class PolygonTimespan(str, Enum):
    """Polygon timespan values."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


# ============================================================================
# POLYGON CLIENT
# ============================================================================

class PolygonClient:
    """
    Async Polygon.io API client.

    High-performance async implementation for market data.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Polygon client.

        Args:
            api_key: Polygon API key (defaults to config)
        """
        self.api_key = api_key or settings.polygon_api_key

        if not self.api_key:
            raise ValueError("Polygon API key not configured")

        self.base_url = "https://api.polygon.io"
        self.session: aiohttp.ClientSession | None = None
        self._initialized = False

    async def initialize(self):
        """Initialize HTTP session with connection pooling."""
        if self._initialized:
            return

        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300
        )

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )

        self._initialized = True
        logger.info("Polygon client initialized")

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self._initialized = False
            logger.info("Polygon client closed")

    async def _request(
        self,
        endpoint: str,
        params: dict | None = None
    ) -> dict[str, Any]:
        """
        Make HTTP request to Polygon API.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response
        """
        if not self._initialized:
            await self.initialize()

        url = f"{self.base_url}{endpoint}"

        # Add API key to params
        if params is None:
            params = {}
        params["apiKey"] = self.api_key

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientResponseError as e:
            logger.error(f"Polygon API error: {e.status} - {e.message}")
            raise

        except Exception as e:
            logger.error(f"Polygon request failed: {e}")
            raise

    # ========================================================================
    # REAL-TIME DATA
    # ========================================================================

    async def get_last_trade(self, symbol: str) -> dict[str, Any]:
        """
        Get last trade for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Last trade data
        """
        endpoint = f"/v2/last/trade/{symbol}"
        return await self._request(endpoint)

    async def get_last_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get last quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Last quote data (bid/ask)
        """
        endpoint = f"/v2/last/nbbo/{symbol}"
        return await self._request(endpoint)

    async def get_snapshot(self, symbol: str) -> dict[str, Any]:
        """
        Get snapshot of symbol (day's OHLC, prev day, etc.).

        Args:
            symbol: Stock symbol

        Returns:
            Snapshot data
        """
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        return await self._request(endpoint)

    async def get_snapshots_all(self) -> dict[str, Any]:
        """
        Get snapshots for all tickers.

        Returns:
            Snapshots for all symbols
        """
        endpoint = "/v2/snapshot/locale/us/markets/stocks/tickers"
        return await self._request(endpoint)

    # ========================================================================
    # HISTORICAL DATA
    # ========================================================================

    async def get_aggregates(
        self,
        symbol: str,
        multiplier: int,
        timespan: PolygonTimespan,
        from_date: datetime,
        to_date: datetime,
        limit: int = 5000
    ) -> dict[str, Any]:
        """
        Get aggregate bars (OHLCV) for a symbol.

        Args:
            symbol: Stock symbol
            multiplier: Size of timespan multiplier (e.g., 1 for 1 minute)
            timespan: Timespan unit
            from_date: Start date
            to_date: End date
            limit: Max results

        Returns:
            Aggregate bar data

        Example:
            # Get 5-minute bars
            await client.get_aggregates("AAPL", 5, PolygonTimespan.MINUTE, ...)
        """
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan.value}/{from_str}/{to_str}"

        params = {"limit": limit, "adjusted": "true", "sort": "asc"}

        return await self._request(endpoint, params)

    async def get_daily_open_close(
        self,
        symbol: str,
        date: datetime
    ) -> dict[str, Any]:
        """
        Get OHLC for a specific day.

        Args:
            symbol: Stock symbol
            date: Date

        Returns:
            Daily OHLC data
        """
        date_str = date.strftime("%Y-%m-%d")
        endpoint = f"/v1/open-close/{symbol}/{date_str}"
        return await self._request(endpoint)

    async def get_previous_close(self, symbol: str) -> dict[str, Any]:
        """
        Get previous day's OHLC.

        Args:
            symbol: Stock symbol

        Returns:
            Previous day OHLC
        """
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        return await self._request(endpoint)

    # ========================================================================
    # REFERENCE DATA
    # ========================================================================

    async def get_ticker_details(self, symbol: str) -> dict[str, Any]:
        """
        Get ticker details (company info).

        Args:
            symbol: Stock symbol

        Returns:
            Company details
        """
        endpoint = f"/v3/reference/tickers/{symbol}"
        return await self._request(endpoint)

    async def get_tickers(
        self,
        market: str = "stocks",
        limit: int = 100
    ) -> dict[str, Any]:
        """
        Get list of tickers.

        Args:
            market: Market type
            limit: Max results

        Returns:
            List of tickers
        """
        endpoint = "/v3/reference/tickers"
        params = {"market": market, "limit": limit, "active": "true"}
        return await self._request(endpoint, params)

    async def get_market_status(self) -> dict[str, Any]:
        """
        Get market status (open/closed).

        Returns:
            Market status
        """
        endpoint = "/v1/marketstatus/now"
        return await self._request(endpoint)

    async def get_market_holidays(self) -> list[dict[str, Any]]:
        """
        Get market holidays.

        Returns:
            List of holidays
        """
        endpoint = "/v1/marketstatus/upcoming"
        response = await self._request(endpoint)
        return response.get("data", [])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def get_latest_price_polygon(symbol: str) -> float | None:
    """
    Get latest price from Polygon (for price lookup).

    Args:
        symbol: Stock symbol

    Returns:
        Latest price or None
    """
    client = PolygonClient()
    await client.initialize()

    try:
        snapshot = await client.get_snapshot(symbol)
        ticker = snapshot.get("ticker", {})
        price = ticker.get("day", {}).get("c")  # Last close price

        return float(price) if price else None

    except Exception as e:
        logger.error(f"Failed to get price from Polygon: {e}")
        return None

    finally:
        await client.close()


async def ingest_polygon_data(
    symbols: list[str],
    days: int = 30,
    timespan: PolygonTimespan = PolygonTimespan.MINUTE
):
    """
    Ingest historical data from Polygon to QuestDB.

    Args:
        symbols: List of symbols
        days: Number of days of history
        timespan: Bar timespan
    """
    client = PolygonClient()
    await client.initialize()

    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Ingesting {days} days of Polygon data for {len(symbols)} symbols...")

        for symbol in symbols:
            try:
                response = await client.get_aggregates(
                    symbol=symbol,
                    multiplier=1,
                    timespan=timespan,
                    from_date=start_date,
                    to_date=end_date
                )

                results = response.get("results", [])

                if not results:
                    logger.warning(f"No Polygon data for {symbol}")
                    continue

                logger.info(f"Got {len(results)} bars for {symbol} from Polygon")

                # TODO: Insert to QuestDB

                # Rate limiting (Polygon free tier: 5 req/min)
                import asyncio
                await asyncio.sleep(12)  # 5 requests per minute = 12 seconds between requests

            except Exception as e:
                logger.error(f"Error ingesting Polygon data for {symbol}: {e}")
                continue

    finally:
        await client.close()


# ============================================================================
# GLOBAL CLIENT INSTANCE
# ============================================================================

polygon_client = PolygonClient()


# Export public API
__all__ = [
    "PolygonClient",
    "PolygonTimespan",
    "polygon_client",
    "get_latest_price_polygon",
    "ingest_polygon_data",
]
