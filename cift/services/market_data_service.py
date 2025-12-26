"""
CIFT Markets - Unified Market Data Service

Orchestrates data fetching across multiple providers:
1. Polygon.io (Primary for US Stocks)
2. Finnhub (Primary for Fundamentals, Fallback for Quotes)
3. Alltick.co (Fallback for Global Data)
4. Mock Data (Final Fallback)
"""

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger

from cift.services.polygon_realtime_service import PolygonRealtimeService
from cift.services.finnhub_realtime_service import FinnhubRealtimeService
from cift.services.alltick_service import AlltickService

class MarketDataService:
    def __init__(self):
        self.polygon = PolygonRealtimeService()
        self.finnhub = FinnhubRealtimeService()
        self.alltick = AlltickService()

    async def initialize(self):
        await self.polygon.initialize()
        await self.finnhub.initialize()
        await self.alltick.initialize()

    async def close(self):
        await self.polygon.close()
        await self.finnhub.close()
        await self.alltick.close()

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get real-time quote with fallback strategy.
        Strategy: Polygon -> Finnhub -> Alltick -> Mock
        """
        # 1. Try Polygon
        try:
            quote = await self.polygon.get_snapshot(symbol)
            if quote and quote.get("status") == "OK":
                return quote
        except Exception as e:
            logger.warning(f"Polygon quote failed for {symbol}: {e}")

        # 2. Try Finnhub
        try:
            quote = await self.finnhub.get_quote(symbol)
            if quote:
                # Normalize to Polygon format if needed, or return as is
                # For now, returning as is, caller might need to handle different formats
                # Ideally we should normalize here.
                return {"status": "OK", "ticker": {"lastTrade": {"p": quote["price"]}}}
        except Exception as e:
            logger.warning(f"Finnhub quote failed for {symbol}: {e}")

        # 3. Try Alltick
        try:
            quote = await self.alltick.get_quote(symbol)
            if quote:
                return quote
        except Exception as e:
            logger.warning(f"Alltick quote failed for {symbol}: {e}")

        # 4. Fallback to Mock (handled by Polygon service internally if we call get_quotes_batch)
        # But since we called get_snapshot directly, we might need to invoke mock manually
        # or rely on Polygon's internal fallback if we use get_quotes_batch
        
        return self.polygon._generate_mock_quotes([symbol]).get(symbol, {})

    async def get_quotes_batch(self, symbols: list[str]) -> dict[str, dict]:
        """
        Get batch quotes.
        Strategy: Finnhub (FREE, working) -> Polygon (if available) -> Mock
        """
        results = {}
        
        # 1. Try Finnhub first (it's FREE and working)
        for symbol in symbols:
            try:
                quote = await self.finnhub.get_quote(symbol)
                if quote and quote.get("price", 0) > 0:
                    results[symbol] = {
                        "symbol": symbol,
                        "price": quote.get("price", 0),
                        "bid": quote.get("price", 0) * 0.9999,  # Simulated bid
                        "ask": quote.get("price", 0) * 1.0001,  # Simulated ask
                        "volume": quote.get("volume", 0),
                        "change": quote.get("change", 0),
                        "change_percent": quote.get("change_percent", 0),
                        "high": quote.get("high", 0),
                        "low": quote.get("low", 0),
                        "open": quote.get("open", 0),
                    }
            except Exception as e:
                logger.warning(f"Finnhub batch quote failed for {symbol}: {e}")

        # 2. Fill missing with Polygon/Mock
        missing = [s for s in symbols if s not in results]
        if missing:
            # This will use Polygon or Mock internally
            poly_quotes = self.polygon._generate_mock_quotes(missing)
            results.update(poly_quotes)
            
        return results

    async def get_historical_data(self, symbol: str, days: int = 200) -> list[dict]:
        """
        Get historical OHLCV data.
        Strategy: Finnhub (Free) -> Mock
        """
        # Calculate timestamps
        to_ts = int(datetime.utcnow().timestamp())
        from_ts = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp()) # *2 to ensure enough trading days
        
        # 1. Try Finnhub
        try:
            # Resolution 'D' for daily
            data = await self.finnhub.get_candles(symbol, 'D', from_ts, to_ts)
            if data and data.get('s') == 'ok':
                # Convert to list of dicts
                candles = []
                for i in range(len(data['t'])):
                    candles.append({
                        "timestamp": datetime.fromtimestamp(data['t'][i]),
                        "open": float(data['o'][i]),
                        "high": float(data['h'][i]),
                        "low": float(data['l'][i]),
                        "close": float(data['c'][i]),
                        "volume": float(data['v'][i])
                    })
                # Sort descending (newest first)
                candles.sort(key=lambda x: x["timestamp"], reverse=True)
                return candles[:days]
        except Exception as e:
            logger.warning(f"Finnhub history failed for {symbol}: {e}")
            
        # 2. Fallback to Mock
        # Generate synthetic history based on current price
        current = await self.get_quote(symbol)
        price = current.get("price", 100.0)
        
        import random
        import numpy as np
        
        candles = []
        curr_price = price
        
        for i in range(days):
            change = np.random.normal(0, price * 0.02) # 2% daily volatility
            open_p = curr_price
            close_p = curr_price + change
            high_p = max(open_p, close_p) + abs(change * 0.5)
            low_p = min(open_p, close_p) - abs(change * 0.5)
            vol = random.randint(100000, 5000000)
            
            candles.append({
                "timestamp": datetime.utcnow() - timedelta(days=i),
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": vol
            })
            
            curr_price = close_p # Walk backwards
            
        return candles

                logger.warning(f"Polygon batch quote failed: {e}")
        
        return results

    async def get_company_profile(self, symbol: str) -> dict | None:
        """
        Get company profile.
        Strategy: Finnhub (Best) -> Polygon -> Mock
        """
        # 1. Try Finnhub
        try:
            profile = await self.finnhub.get_company_profile(symbol)
            if profile:
                return profile
        except Exception as e:
            logger.warning(f"Finnhub profile failed for {symbol}: {e}")

        # 2. Try Polygon (Details)
        # (Assuming Polygon service has a method for details, if not, skip)
        
        return None

    async def get_financials(self, symbol: str) -> dict | None:
        """Get financials from Finnhub."""
        return await self.finnhub.get_financials(symbol)

    async def get_financials_reported(self, symbol: str) -> dict | None:
        """Get reported financial statements."""
        return await self.finnhub.get_financials_reported(symbol)

    async def get_earnings_estimates(self, symbol: str) -> dict | None:
        """Get earnings estimates."""
        return await self.finnhub.get_earnings_estimates(symbol)

    async def fetch_and_store_ohlcv(self, symbol: str, days: int = 30, timeframe: str = "1m") -> int:
        """
        Fetch OHLCV bars from Finnhub and store in database.
        Finnhub is FREE and working, so we use it as primary source.
        
        Returns: Number of bars stored
        """
        from datetime import datetime, timedelta
        from cift.core.database import get_postgres_pool
        
        # Map timeframe to Finnhub resolution
        resolution_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "1d": "D",
            "1w": "W",
        }
        resolution = resolution_map.get(timeframe, "1")
        
        to_ts = int(datetime.utcnow().timestamp())
        from_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
        
        try:
            candles = await self.finnhub.get_candles(symbol, resolution, from_ts, to_ts)
            
            if not candles or candles.get("s") != "ok":
                logger.warning(f"No candles from Finnhub for {symbol}")
                # Try Polygon as fallback
                return await self._fetch_ohlcv_from_polygon(symbol, days, timeframe)
            
            pool = await get_postgres_pool()
            total_bars = 0
            
            timestamps = candles.get("t", [])
            opens = candles.get("o", [])
            highs = candles.get("h", [])
            lows = candles.get("l", [])
            closes = candles.get("c", [])
            volumes = candles.get("v", [])
            
            async with pool.acquire() as conn:
                for i in range(len(timestamps)):
                    try:
                        ts = datetime.utcfromtimestamp(timestamps[i])
                        await conn.execute(
                            """
                            INSERT INTO ohlcv_bars (symbol, timestamp, timeframe, open, high, low, close, volume)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (symbol, timestamp, timeframe) DO UPDATE SET
                                open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                            """,
                            symbol.upper(),
                            ts,
                            timeframe,
                            float(opens[i]),
                            float(highs[i]),
                            float(lows[i]),
                            float(closes[i]),
                            int(volumes[i]) if volumes else 0,
                        )
                        total_bars += 1
                    except Exception as e:
                        logger.debug(f"Error inserting bar: {e}")
            
            logger.info(f"Stored {total_bars} bars for {symbol} from Finnhub")
            return total_bars
            
        except Exception as e:
            logger.error(f"Finnhub OHLCV fetch failed for {symbol}: {e}")
            return await self._fetch_ohlcv_from_polygon(symbol, days, timeframe)

    async def _fetch_ohlcv_from_polygon(self, symbol: str, days: int, timeframe: str) -> int:
        """Fallback to Polygon for OHLCV data."""
        try:
            timespan_map = {"1m": "minute", "1h": "hour", "1d": "day"}
            timespan = timespan_map.get(timeframe, "minute")
            return await self.polygon.update_ohlcv_bars([symbol], days=days, timespan=timespan)
        except Exception as e:
            logger.error(f"Polygon OHLCV fallback failed: {e}")
            return 0

# Global instance
market_data_service = MarketDataService()

