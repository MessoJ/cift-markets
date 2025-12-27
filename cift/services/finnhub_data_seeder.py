"""
CIFT Markets - Finnhub Historical Data Seeder

Seeds real market data from Finnhub into PostgreSQL and QuestDB.
This service fetches REAL historical OHLCV data and company information.

Features:
- Historical candlestick data (1min to monthly)
- Company profiles (market cap, sector, industry)
- Quote data with bid/ask
- Earnings calendar
- Pattern recognition

API Limits (Free Tier):
- 60 API calls/minute
- Use async with rate limiting
"""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp
from loguru import logger

from cift.core.config import settings
from cift.core.database import db_manager


class FinnhubDataSeeder:
    """
    Seeds real market data from Finnhub API.

    Usage:
        seeder = FinnhubDataSeeder()
        await seeder.seed_all()
    """

    REST_URL = "https://finnhub.io/api/v1"

    # Rate limiting: 60 requests per minute
    MAX_REQUESTS_PER_MINUTE = 55  # Leave buffer
    REQUEST_INTERVAL = 60.0 / MAX_REQUESTS_PER_MINUTE  # ~1.09 seconds

    # Major symbols to seed
    DEFAULT_SYMBOLS = [
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        # Blue chips
        "JPM", "V", "JNJ", "UNH", "PG", "HD", "DIS",
        # Popular trading stocks
        "AMD", "NFLX", "PYPL", "SQ", "COIN", "PLTR", "RIVN",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        # Indices (use ETF proxies)
    ]

    # Timeframe mapping to Finnhub resolution
    TIMEFRAME_MAP = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "1d": "D",
        "1w": "W",
        "1M": "M",
    }

    def __init__(self, api_key: str | None = None):
        """Initialize seeder with API key."""
        self.api_key = api_key or getattr(settings, 'finnhub_api_key', '')

        if not self.api_key:
            raise ValueError(
                "Finnhub API key required. Get FREE key at: https://finnhub.io/"
            )

        self.session: aiohttp.ClientSession | None = None
        self._last_request_time = 0.0
        self._request_count = 0

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time

        if elapsed < self.REQUEST_INTERVAL:
            await asyncio.sleep(self.REQUEST_INTERVAL - elapsed)

        self._last_request_time = time.time()
        self._request_count += 1

        if self._request_count % 10 == 0:
            logger.debug(f"Finnhub requests made: {self._request_count}")

    async def _get(self, endpoint: str, params: dict = None) -> Any | None:
        """Make rate-limited GET request to Finnhub API."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

        await self._rate_limit()

        url = f"{self.REST_URL}/{endpoint}"
        params = params or {}
        params["token"] = self.api_key

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Rate limited by Finnhub, waiting 60s...")
                    await asyncio.sleep(60)
                    return await self._get(endpoint, params)
                else:
                    logger.warning(f"Finnhub API error {response.status}: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Finnhub request failed: {e}")
            return None

    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    # ========================================================================
    # COMPANY DATA
    # ========================================================================

    async def get_company_profile(self, symbol: str) -> dict | None:
        """
        Get company profile from Finnhub.

        Returns:
            {
                "country": "US",
                "currency": "USD",
                "exchange": "NASDAQ NMS - GLOBAL MARKET",
                "finnhubIndustry": "Technology",
                "ipo": "1980-12-12",
                "logo": "https://...",
                "marketCapitalization": 2800000,  # in millions
                "name": "Apple Inc",
                "phone": "14089961010",
                "shareOutstanding": 16319.44,
                "ticker": "AAPL",
                "weburl": "https://www.apple.com/"
            }
        """
        return await self._get("stock/profile2", {"symbol": symbol})

    async def get_quote(self, symbol: str) -> dict | None:
        """
        Get real-time quote from Finnhub.

        Returns:
            {
                "c": 261.74,    # Current price
                "d": 0.81,      # Change
                "dp": 0.3103,   # Percent change
                "h": 263.31,    # High price of the day
                "l": 260.68,    # Low price of the day
                "o": 261.07,    # Open price of the day
                "pc": 260.93,   # Previous close price
                "t": 1640185200 # Timestamp
            }
        """
        return await self._get("quote", {"symbol": symbol})

    async def get_earnings_calendar(self, symbol: str) -> list | None:
        """Get upcoming earnings dates for a symbol."""
        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

        data = await self._get("calendar/earnings", {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
        })

        return data.get("earningsCalendar", []) if data else []

    async def get_pattern_recognition(self, symbol: str, resolution: str = "D") -> dict | None:
        """
        Get technical pattern recognition from Finnhub.

        Patterns detected: Head and Shoulders, Double Top/Bottom, Triangle, etc.
        """
        return await self._get("scan/pattern", {
            "symbol": symbol,
            "resolution": resolution,
        })

    async def get_support_resistance(self, symbol: str, resolution: str = "D") -> dict | None:
        """Get support/resistance levels from Finnhub."""
        return await self._get("scan/support-resistance", {
            "symbol": symbol,
            "resolution": resolution,
        })

    # ========================================================================
    # HISTORICAL CANDLE DATA
    # ========================================================================

    async def get_candles(
        self,
        symbol: str,
        resolution: str = "D",
        from_timestamp: int = None,
        to_timestamp: int = None,
        count: int = 500,
    ) -> list[dict] | None:
        """
        Get historical OHLCV candles from Finnhub.

        Args:
            symbol: Stock symbol
            resolution: 1, 5, 15, 30, 60, D, W, M
            from_timestamp: Start timestamp (defaults to count bars ago)
            to_timestamp: End timestamp (defaults to now)
            count: Number of bars if no from_timestamp

        Returns:
            List of candle dicts: [{timestamp, open, high, low, close, volume}, ...]
        """
        if to_timestamp is None:
            to_timestamp = int(datetime.now(UTC).timestamp())

        if from_timestamp is None:
            # Calculate from_timestamp based on resolution and count
            seconds_per_bar = {
                "1": 60, "5": 300, "15": 900, "30": 1800,
                "60": 3600, "D": 86400, "W": 604800, "M": 2592000
            }
            bar_seconds = seconds_per_bar.get(resolution, 86400)
            from_timestamp = to_timestamp - (count * bar_seconds)

        data = await self._get("stock/candle", {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_timestamp,
            "to": to_timestamp,
        })

        if not data or data.get("s") == "no_data":
            logger.warning(f"No candle data for {symbol}")
            return []

        if data.get("s") != "ok":
            logger.warning(f"Candle fetch failed for {symbol}: {data}")
            return []

        # Convert to list of dicts
        candles = []
        timestamps = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])

        for i in range(len(timestamps)):
            candles.append({
                "timestamp": datetime.fromtimestamp(timestamps[i], tz=UTC),
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "volume": int(volumes[i]) if i < len(volumes) else 0,
            })

        return candles

    # ========================================================================
    # DATABASE SEEDING
    # ========================================================================

    async def seed_company_profile(self, symbol: str) -> bool:
        """Seed company profile to database."""
        from datetime import datetime

        profile = await self.get_company_profile(symbol)
        if not profile:
            return False

        # Parse IPO date string to date object
        ipo_date = None
        ipo_str = profile.get("ipo")
        if ipo_str:
            try:
                ipo_date = datetime.strptime(ipo_str, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                pass

        try:
            async with db_manager.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO company_profiles (
                        symbol, name, exchange, industry, sector,
                        market_cap, shares_outstanding, ipo_date,
                        logo_url, website, currency, country,
                        updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                        name = EXCLUDED.name,
                        exchange = EXCLUDED.exchange,
                        industry = EXCLUDED.industry,
                        market_cap = EXCLUDED.market_cap,
                        shares_outstanding = EXCLUDED.shares_outstanding,
                        logo_url = EXCLUDED.logo_url,
                        website = EXCLUDED.website,
                        updated_at = NOW()
                """,
                    symbol,
                    profile.get("name"),
                    profile.get("exchange"),
                    profile.get("finnhubIndustry"),
                    profile.get("finnhubIndustry"),  # Sector same as industry for now
                    profile.get("marketCapitalization"),  # In millions
                    profile.get("shareOutstanding"),
                    ipo_date,  # Converted to date object
                    profile.get("logo"),
                    profile.get("weburl"),
                    profile.get("currency", "USD"),
                    profile.get("country", "US"),
                )

            logger.info(f"✅ Seeded company profile: {symbol} ({profile.get('name')})")
            return True

        except Exception as e:
            logger.error(f"Failed to seed profile for {symbol}: {e}")
            return False

    async def seed_quote(self, symbol: str) -> bool:
        """Seed latest quote to market_data_cache."""
        quote = await self.get_quote(symbol)
        if not quote or quote.get("c") is None:
            return False

        try:
            async with db_manager.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_data_cache (
                        symbol, price, open, high, low, close, prev_close,
                        change, change_pct, volume, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                        price = EXCLUDED.price,
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        prev_close = EXCLUDED.prev_close,
                        change = EXCLUDED.change,
                        change_pct = EXCLUDED.change_pct,
                        volume = EXCLUDED.volume,
                        updated_at = NOW()
                """,
                    symbol,
                    quote.get("c"),  # Current price
                    quote.get("o"),  # Open
                    quote.get("h"),  # High
                    quote.get("l"),  # Low
                    quote.get("c"),  # Close (current for intraday)
                    quote.get("pc"),  # Previous close
                    quote.get("d"),  # Change
                    quote.get("dp"),  # Change percent
                    0,  # Volume not in quote endpoint
                )

            logger.info(f"✅ Seeded quote: {symbol} @ ${quote.get('c'):.2f} ({quote.get('dp'):+.2f}%)")
            return True

        except Exception as e:
            logger.error(f"Failed to seed quote for {symbol}: {e}")
            return False

    async def seed_candles(
        self,
        symbol: str,
        timeframe: str = "1d",
        count: int = 365,
    ) -> int:
        """
        Seed historical candles to ohlcv_bars table.

        Returns:
            Number of candles seeded
        """
        resolution = self.TIMEFRAME_MAP.get(timeframe, "D")
        candles = await self.get_candles(symbol, resolution, count=count)

        if not candles:
            return 0

        try:
            async with db_manager.pool.acquire() as conn:
                # Use batch insert for efficiency
                for candle in candles:
                    await conn.execute("""
                        INSERT INTO ohlcv_bars (
                            timestamp, symbol, timeframe,
                            open, high, low, close, volume
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """,
                        candle["timestamp"],
                        symbol,
                        timeframe,
                        candle["open"],
                        candle["high"],
                        candle["low"],
                        candle["close"],
                        candle["volume"],
                    )

            logger.info(f"✅ Seeded {len(candles)} candles for {symbol} ({timeframe})")
            return len(candles)

        except Exception as e:
            logger.error(f"Failed to seed candles for {symbol}: {e}")
            return 0

    async def seed_earnings(self, symbol: str) -> bool:
        """Seed earnings calendar to database."""
        from datetime import datetime

        earnings = await self.get_earnings_calendar(symbol)
        if not earnings:
            return False

        try:
            async with db_manager.pool.acquire() as conn:
                seeded_count = 0
                for earning in earnings:
                    # Parse date string to date object
                    date_str = earning.get("date")
                    earnings_date = None
                    if date_str:
                        try:
                            earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        except (ValueError, TypeError):
                            continue
                    else:
                        continue

                    # Determine earnings time (before/after market)
                    hour = earning.get("hour")
                    earnings_time = "bmo"  # Before Market Open default
                    if hour:
                        if "amc" in str(hour).lower() or "after" in str(hour).lower():
                            earnings_time = "amc"  # After Market Close
                        elif "dmh" in str(hour).lower() or "during" in str(hour).lower():
                            earnings_time = "dmh"  # During Market Hours

                    # Check if already exists
                    existing = await conn.fetchval("""
                        SELECT id FROM earnings_calendar
                        WHERE symbol = $1 AND earnings_date = $2
                    """, symbol, earnings_date)

                    if existing:
                        # Update existing record
                        await conn.execute("""
                            UPDATE earnings_calendar SET
                                earnings_time = $1,
                                eps_estimate = $2,
                                eps_actual = $3,
                                revenue_estimate = $4,
                                revenue_actual = $5
                            WHERE id = $6
                        """,
                            earnings_time,
                            earning.get("epsEstimate"),
                            earning.get("epsActual"),
                            earning.get("revenueEstimate"),
                            earning.get("revenueActual"),
                            existing,
                        )
                    else:
                        # Insert new record
                        await conn.execute("""
                            INSERT INTO earnings_calendar (
                                symbol, earnings_date, earnings_time,
                                eps_estimate, eps_actual,
                                revenue_estimate, revenue_actual
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                            symbol,
                            earnings_date,
                            earnings_time,
                            earning.get("epsEstimate"),
                            earning.get("epsActual"),
                            earning.get("revenueEstimate"),
                            earning.get("revenueActual"),
                        )
                    seeded_count += 1

            logger.info(f"✅ Seeded {seeded_count} earnings events for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to seed earnings for {symbol}: {e}")
            return False

    # ========================================================================
    # FULL SEEDING
    # ========================================================================

    async def seed_symbol(self, symbol: str, include_candles: bool = True) -> dict:
        """
        Seed all data for a single symbol.

        Returns:
            Summary of what was seeded
        """
        result = {
            "symbol": symbol,
            "profile": False,
            "quote": False,
            "candles_1d": 0,
            "candles_1h": 0,
            "earnings": False,
        }

        # Company profile
        result["profile"] = await self.seed_company_profile(symbol)

        # Latest quote
        result["quote"] = await self.seed_quote(symbol)

        if include_candles:
            # Daily candles (1 year)
            result["candles_1d"] = await self.seed_candles(symbol, "1d", count=365)

            # Hourly candles (30 days) - only for active symbols
            if symbol in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ"]:
                result["candles_1h"] = await self.seed_candles(symbol, "1h", count=720)

        # Earnings calendar
        result["earnings"] = await self.seed_earnings(symbol)

        return result

    async def seed_all(self, symbols: list[str] = None, include_candles: bool = True) -> dict:
        """
        Seed all data for multiple symbols.

        Args:
            symbols: List of symbols (defaults to DEFAULT_SYMBOLS)
            include_candles: Whether to include historical candles

        Returns:
            Summary of seeding results
        """
        symbols = symbols or self.DEFAULT_SYMBOLS

        logger.info(f"🌱 Starting Finnhub data seeding for {len(symbols)} symbols...")
        start_time = time.time()

        results = {
            "total_symbols": len(symbols),
            "successful": 0,
            "failed": 0,
            "total_candles": 0,
            "details": [],
        }

        for symbol in symbols:
            try:
                result = await self.seed_symbol(symbol, include_candles)
                results["details"].append(result)

                if result["profile"] or result["quote"]:
                    results["successful"] += 1
                else:
                    results["failed"] += 1

                results["total_candles"] += result["candles_1d"] + result["candles_1h"]

            except Exception as e:
                logger.error(f"Failed to seed {symbol}: {e}")
                results["failed"] += 1
                results["details"].append({
                    "symbol": symbol,
                    "error": str(e),
                })

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = round(elapsed, 2)

        logger.success(
            f"🌱 Seeding complete: {results['successful']}/{results['total_symbols']} symbols, "
            f"{results['total_candles']} candles in {elapsed:.1f}s"
        )

        await self.close()
        return results


# ============================================================================
# CLI
# ============================================================================

async def main():
    """CLI for running seeder."""
    import sys

    # Initialize database connection for CLI usage
    from cift.core.database import close_all_connections, initialize_all_connections

    try:
        await initialize_all_connections()
        print("✅ Database connections initialized")
    except Exception as e:
        print(f"⚠️ Database init failed: {e}")
        print("Continuing without database - will only fetch data, not save")

    seeder = FinnhubDataSeeder()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "all":
            # Seed all default symbols
            results = await seeder.seed_all()
            print(f"\nSeeding complete: {results['successful']}/{results['total_symbols']} symbols")

        elif command == "symbol":
            # Seed single symbol
            symbol = sys.argv[2].upper() if len(sys.argv) > 2 else "AAPL"
            result = await seeder.seed_symbol(symbol)
            print(f"\nSeeded {symbol}: {result}")

        elif command == "quote":
            # Just get quote for symbol
            symbol = sys.argv[2].upper() if len(sys.argv) > 2 else "AAPL"
            quote = await seeder.get_quote(symbol)
            if quote:
                print(f"{symbol}: ${quote['c']:.2f} ({quote['dp']:+.2f}%)")
            else:
                print(f"No quote for {symbol}")

        elif command == "profile":
            # Get company profile
            symbol = sys.argv[2].upper() if len(sys.argv) > 2 else "AAPL"
            profile = await seeder.get_company_profile(symbol)
            if profile:
                print(f"{symbol}: {profile.get('name')}")
                print(f"  Market Cap: ${profile.get('marketCapitalization', 0):,.0f}M")
                print(f"  Industry: {profile.get('finnhubIndustry')}")
            else:
                print(f"No profile for {symbol}")

        else:
            print("Unknown command")
            print("Usage: python -m cift.services.finnhub_data_seeder <command>")
            print("Commands: all, symbol [SYM], quote [SYM], profile [SYM]")
    else:
        print("Finnhub Data Seeder")
        print("Usage: python -m cift.services.finnhub_data_seeder <command>")
        print("Commands: all, symbol [SYM], quote [SYM], profile [SYM]")

    await seeder.close()

    # Close database connections
    try:
        await close_all_connections()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
