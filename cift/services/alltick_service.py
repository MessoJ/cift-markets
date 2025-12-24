"""
CIFT Markets - Alltick.co Market Data Service

Global market data provider.
Used as a fallback for Polygon and Finnhub.
"""

import aiohttp
from loguru import logger
from cift.core.config import settings

class AlltickService:
    """
    Alltick.co market data service.
    """
    
    # Base URL (Verify this in documentation)
    BASE_URL = "https://quote.tradeswitcher.com/quote-b-api" 

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or getattr(settings, "alltick_api_key", "")
        if not self.api_key:
            logger.warning("Alltick API key not configured")
            self._available = False
        else:
            self._available = True
            logger.info("Alltick service initialized")
            
        self.session: aiohttp.ClientSession | None = None

    @property
    def is_available(self) -> bool:
        return self._available

    async def initialize(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_quote(self, symbol: str) -> dict | None:
        """Get real-time quote."""
        if not self._available:
            return None
        
        await self.initialize()
        # Endpoint structure is hypothetical - needs verification
        url = f"{self.BASE_URL}/depth-tick" 
        params = {"token": self.api_key, "symbol": symbol}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    # Map response to standard format
                    return data
                return None
        except Exception as e:
            logger.error(f"Alltick quote failed: {e}")
            return None

    async def get_kline(self, symbol: str, period: str) -> list[dict]:
        """Get historical k-line data."""
        if not self._available:
            return []
            
        await self.initialize()
        url = f"{self.BASE_URL}/kline"
        params = {"token": self.api_key, "symbol": symbol, "period": period}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return []
        except Exception as e:
            logger.error(f"Alltick kline failed: {e}")
            return []
