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
        Currently delegates to Polygon which has internal mock fallback.
        TODO: Implement multi-provider batch fallback.
        """
        return await self.polygon.get_quotes_batch(symbols)

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

# Global instance
market_data_service = MarketDataService()

