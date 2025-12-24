"""
Financial Modeling Prep (FMP) Economic Calendar Service

Fetches REAL economic calendar events from FMP's free API.
Replaces mock/sample data with actual upcoming economic events.

Required Environment Variables:
- FMP_API_KEY: Your Financial Modeling Prep API key

Get your FREE API key at: https://financialmodelingprep.com/developer/docs/
Free tier: 250 API calls/day
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import aiohttp

from cift.core.logging import logger


class FMPEconomicCalendarService:
    """
    Real economic calendar data from Financial Modeling Prep.
    
    Features:
    - Free API (250 calls/day)
    - Real upcoming economic events (FOMC, CPI, NFP, etc.)
    - Global coverage (US, EU, UK, Japan, China, etc.)
    - Impact ratings (high, medium, low)
    - Forecast vs Previous vs Actual values
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    # Country code to full name mapping
    COUNTRY_MAP = {
        "US": "United States",
        "EU": "European Union",
        "GB": "United Kingdom",
        "JP": "Japan",
        "CN": "China",
        "DE": "Germany",
        "FR": "France",
        "CA": "Canada",
        "AU": "Australia",
        "NZ": "New Zealand",
        "CH": "Switzerland",
        "IN": "India",
        "BR": "Brazil",
        "MX": "Mexico",
        "KR": "South Korea",
    }
    
    def __init__(self):
        """Initialize the FMP service."""
        self.api_key = os.getenv('FMP_API_KEY', '')
        self._available = bool(self.api_key)
        
        if not self._available:
            logger.warning("FMP_API_KEY not configured - economic calendar will use fallback data")
            logger.info("Get a FREE API key at: https://financialmodelingprep.com/")
    
    @property
    def is_available(self) -> bool:
        """Check if service is configured."""
        return self._available
    
    async def fetch_economic_calendar(
        self,
        days_ahead: int = 30,
        days_behind: int = 7,
    ) -> list[dict[str, Any]]:
        """
        Fetch economic calendar events from FMP API.
        
        Args:
            days_ahead: Number of days in the future to fetch
            days_behind: Number of days in the past to fetch
            
        Returns:
            List of economic events
        """
        if not self._available:
            logger.warning("FMP API key not available, returning empty list")
            return []
        
        today = datetime.now()
        from_date = (today - timedelta(days=days_behind)).strftime("%Y-%m-%d")
        to_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        url = f"{self.BASE_URL}/economic_calendar"
        params = {
            "from": from_date,
            "to": to_date,
            "apikey": self.api_key,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Fetched {len(data)} economic events from FMP")
                        return data
                    elif response.status == 401:
                        logger.error("FMP API key is invalid")
                        return []
                    elif response.status == 429:
                        logger.warning("FMP API rate limit exceeded")
                        return []
                    else:
                        logger.error(f"FMP API error: {response.status}")
                        return []
                        
        except asyncio.TimeoutError:
            logger.error("FMP API request timed out")
            return []
        except Exception as e:
            logger.error(f"FMP API error: {e}")
            return []
    
    def transform_event(self, event: dict) -> dict[str, Any]:
        """
        Transform FMP event to match database schema.
        
        Args:
            event: Raw event from FMP API
            
        Returns:
            Transformed event matching economic_events table schema
        """
        # Parse the date
        date_str = event.get("date", "")
        try:
            if "T" in date_str:
                event_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                # Default time for events without specific time
                event_date = event_date.replace(hour=8, minute=30)
        except (ValueError, TypeError):
            event_date = datetime.now()
        
        # Map country code to full name
        country_code = event.get("country", "")
        country = self.COUNTRY_MAP.get(country_code, country_code)
        
        # Normalize impact level
        impact = event.get("impact", "medium").lower()
        if impact not in ("high", "medium", "low"):
            impact = "medium"
        
        return {
            "id": str(uuid.uuid4()),
            "title": event.get("event", "Unknown Event"),
            "country": country,
            "event_date": event_date,
            "impact": impact,
            "forecast": str(event.get("estimate", "")) if event.get("estimate") else None,
            "previous": str(event.get("previous", "")) if event.get("previous") else None,
            "actual": str(event.get("actual", "")) if event.get("actual") else None,
            "currency": event.get("currency", "USD"),
        }
    
    async def get_filtered_events(
        self,
        days_ahead: int = 30,
        high_impact_only: bool = False,
        major_countries_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get filtered and transformed economic events.
        
        Args:
            days_ahead: Days to look ahead
            high_impact_only: Only return high-impact events
            major_countries_only: Only return events from major economies
            
        Returns:
            List of transformed events ready for database insertion
        """
        raw_events = await self.fetch_economic_calendar(days_ahead=days_ahead)
        
        if not raw_events:
            return []
        
        # Transform all events
        events = [self.transform_event(e) for e in raw_events]
        
        # Filter by major countries
        if major_countries_only:
            major_countries = [
                "United States", "European Union", "United Kingdom",
                "Japan", "China", "Germany", "Canada", "Australia"
            ]
            events = [e for e in events if e["country"] in major_countries]
        
        # Filter by impact
        if high_impact_only:
            events = [e for e in events if e["impact"] == "high"]
        
        # Sort by date
        events.sort(key=lambda e: e["event_date"])
        
        logger.info(f"Filtered to {len(events)} economic events")
        return events


async def populate_economic_calendar_from_api():
    """
    Populate the economic_events table with real data from FMP.
    
    This function should be called:
    - On application startup
    - Via scheduled job (daily)
    - When user requests calendar refresh
    """
    from cift.core.database import get_postgres_pool
    
    service = FMPEconomicCalendarService()
    
    if not service.is_available:
        logger.warning("FMP service not available, skipping economic calendar update")
        return False
    
    logger.info("Fetching real economic calendar data from FMP...")
    
    events = await service.get_filtered_events(
        days_ahead=60,
        high_impact_only=False,
        major_countries_only=True,
    )
    
    if not events:
        logger.warning("No events received from FMP API")
        return False
    
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Clear future events (keep historical for reference)
            await conn.execute("""
                DELETE FROM economic_events
                WHERE event_date >= NOW() - INTERVAL '1 day'
            """)
            
            # Insert new events
            inserted = 0
            for event in events:
                try:
                    await conn.execute("""
                        INSERT INTO economic_events (
                            id, title, country, event_date, impact,
                            forecast, previous, actual, currency
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (id) DO NOTHING
                    """,
                        uuid.UUID(event["id"]),
                        event["title"],
                        event["country"],
                        event["event_date"],
                        event["impact"],
                        event["forecast"],
                        event["previous"],
                        event["actual"],
                        event["currency"],
                    )
                    inserted += 1
                except Exception as e:
                    logger.warning(f"Failed to insert event: {e}")
            
            logger.success(f"Inserted {inserted} economic events from FMP")
            return True
            
    except Exception as e:
        logger.error(f"Failed to populate economic calendar: {e}")
        return False


# Global instance
fmp_calendar_service = FMPEconomicCalendarService()
