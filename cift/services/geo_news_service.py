"""
CIFT Markets - Geographic News Intelligence Service

Advanced geo-located financial news aggregation for interactive globe visualization.
Provides country-specific market data, economic indicators, and news correlation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel

from cift.core.database import get_postgres_pool


class GeoLocation(BaseModel):
    country: str
    country_code: str  # ISO 3166-1 alpha-2
    latitude: float
    longitude: float
    region: str | None = None
    city: str | None = None


class GeoNewsEvent(BaseModel):
    id: str
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    location: GeoLocation
    impact_level: int = 1  # 1-5 scale
    categories: list[str] = []
    symbols_affected: list[str] = []
    economic_indicators: list[str] = []
    sentiment_score: float = 0.0  # -1 to 1


class CountryMarketData(BaseModel):
    country_code: str
    country_name: str
    major_index: str | None = None
    index_value: float | None = None
    index_change_pct: float | None = None
    currency: str | None = None
    currency_rate: float | None = None
    gdp_growth: float | None = None
    inflation_rate: float | None = None
    unemployment_rate: float | None = None
    interest_rate: float | None = None
    market_cap_usd: float | None = None


class GeoNewsService:
    """Advanced geographic news intelligence and visualization service."""

    def __init__(self):
        # Geographic mappings for major markets
        self.country_mappings = {
            "US": {
                "name": "United States",
                "lat": 39.8283,
                "lng": -98.5795,
                "major_index": "S&P 500",
                "currency": "USD",
                "timezone": "America/New_York"
            },
            "GB": {
                "name": "United Kingdom",
                "lat": 55.3781,
                "lng": -3.4360,
                "major_index": "FTSE 100",
                "currency": "GBP",
                "timezone": "Europe/London"
            },
            "DE": {
                "name": "Germany",
                "lat": 51.1657,
                "lng": 10.4515,
                "major_index": "DAX",
                "currency": "EUR",
                "timezone": "Europe/Berlin"
            },
            "JP": {
                "name": "Japan",
                "lat": 36.2048,
                "lng": 138.2529,
                "major_index": "Nikkei 225",
                "currency": "JPY",
                "timezone": "Asia/Tokyo"
            },
            "CN": {
                "name": "China",
                "lat": 35.8617,
                "lng": 104.1954,
                "major_index": "Shanghai Composite",
                "currency": "CNY",
                "timezone": "Asia/Shanghai"
            },
            "CA": {
                "name": "Canada",
                "lat": 56.1304,
                "lng": -106.3468,
                "major_index": "TSX Composite",
                "currency": "CAD",
                "timezone": "America/Toronto"
            },
            "AU": {
                "name": "Australia",
                "lat": -25.2744,
                "lng": 133.7751,
                "major_index": "ASX 200",
                "currency": "AUD",
                "timezone": "Australia/Sydney"
            },
            "BR": {
                "name": "Brazil",
                "lat": -14.2350,
                "lng": -51.9253,
                "major_index": "Bovespa",
                "currency": "BRL",
                "timezone": "America/Sao_Paulo"
            },
            "IN": {
                "name": "India",
                "lat": 20.5937,
                "lng": 78.9629,
                "major_index": "BSE Sensex",
                "currency": "INR",
                "timezone": "Asia/Kolkata"
            },
            "KR": {
                "name": "South Korea",
                "lat": 35.9078,
                "lng": 127.7669,
                "major_index": "KOSPI",
                "currency": "KRW",
                "timezone": "Asia/Seoul"
            },
            "FR": {
                "name": "France",
                "lat": 46.2276,
                "lng": 2.2137,
                "major_index": "CAC 40",
                "currency": "EUR",
                "timezone": "Europe/Paris"
            },
            "IT": {
                "name": "Italy",
                "lat": 41.8719,
                "lng": 12.5674,
                "major_index": "FTSE MIB",
                "currency": "EUR",
                "timezone": "Europe/Rome"
            }
        }

        # Economic indicator mappings
        self.economic_indicators = [
            "GDP Growth", "Inflation Rate", "Unemployment Rate",
            "Interest Rate", "Trade Balance", "Consumer Confidence",
            "Manufacturing PMI", "Services PMI", "Retail Sales"
        ]

    async def get_global_news_heatmap(
        self,
        hours_back: int = 24,
        impact_threshold: int = 2
    ) -> dict[str, Any]:
        """Get global news heatmap data for interactive globe visualization."""

        logger.info(f"Generating global news heatmap for last {hours_back} hours")

        # Generate geo-located news events
        geo_events = await self._generate_geo_news_events(hours_back, impact_threshold)

        # Get market data for major countries
        country_data = await self._get_country_market_data()

        # Calculate news density and sentiment by country
        country_metrics = self._calculate_country_metrics(geo_events)

        # Combine data for visualization
        globe_data = []
        for country_code, mapping in self.country_mappings.items():
            country_info = {
                "country_code": country_code,
                "country_name": mapping["name"],
                "latitude": mapping["lat"],
                "longitude": mapping["lng"],
                "major_index": mapping.get("major_index"),
                "currency": mapping.get("currency"),

                # Market data
                "market_data": country_data.get(country_code, {}),

                # News metrics
                "news_count": country_metrics.get(country_code, {}).get("count", 0),
                "avg_sentiment": country_metrics.get(country_code, {}).get("sentiment", 0),
                "max_impact": country_metrics.get(country_code, {}).get("max_impact", 1),
                "recent_events": country_metrics.get(country_code, {}).get("events", [])
            }

            globe_data.append(country_info)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "hours_back": hours_back,
            "total_events": len(geo_events),
            "countries": globe_data,
            "global_sentiment": self._calculate_global_sentiment(geo_events),
            "top_stories": await self._get_top_global_stories(geo_events, 10)
        }

    async def get_country_detail(
        self,
        country_code: str,
        days_back: int = 7
    ) -> dict[str, Any]:
        """Get detailed information for a specific country."""

        if country_code not in self.country_mappings:
            raise ValueError(f"Country {country_code} not supported")

        mapping = self.country_mappings[country_code]

        # Get country-specific news
        news_events = await self._get_country_news(country_code, days_back)

        # Get market data
        market_data = await self._get_single_country_market_data(country_code)

        # Get economic indicators
        economic_data = await self._get_economic_indicators(country_code)

        return {
            "country_code": country_code,
            "country_name": mapping["name"],
            "location": {
                "latitude": mapping["lat"],
                "longitude": mapping["lng"]
            },
            "market_data": market_data,
            "economic_indicators": economic_data,
            "news_events": news_events,
            "news_summary": {
                "total_events": len(news_events),
                "avg_sentiment": sum(e.get("sentiment_score", 0) for e in news_events) / len(news_events) if news_events else 0,
                "categories": self._extract_categories(news_events),
                "trending_symbols": self._extract_trending_symbols(news_events)
            }
        }

    async def _generate_geo_news_events(
        self,
        hours_back: int,
        impact_threshold: int
    ) -> list[GeoNewsEvent]:
        """Generate realistic geo-located news events for visualization."""

        # In production, this would integrate with real news APIs
        # For now, generate realistic mock data

        events = []
        current_time = datetime.utcnow()

        # Generate events for each major country
        for country_code, mapping in self.country_mappings.items():
            # Number of events based on market size and activity
            event_count = self._get_country_event_count(country_code)

            for i in range(event_count):
                # Generate realistic event
                event_time = current_time - timedelta(
                    hours=hours_back * (i + 1) / event_count
                )

                event = self._generate_country_event(
                    country_code, mapping, event_time, impact_threshold
                )

                if event and event.impact_level >= impact_threshold:
                    events.append(event)

        return events

    def _get_country_event_count(self, country_code: str) -> int:
        """Get expected number of events for country based on market activity."""

        # Major markets get more events
        major_markets = {"US": 8, "GB": 4, "DE": 4, "JP": 4, "CN": 5}
        return major_markets.get(country_code, 2)

    def _generate_country_event(
        self,
        country_code: str,
        mapping: dict,
        event_time: datetime,
        min_impact: int
    ) -> GeoNewsEvent | None:
        """Generate a realistic news event for a country."""

        import random

        # Event templates by type
        event_templates = {
            "monetary_policy": {
                "title": "{country} Central Bank {action} Interest Rates by {amount}",
                "summary": "Central bank announces {action} in interest rates to {reason}.",
                "impact": 4,
                "categories": ["monetary_policy", "interest_rates"],
                "indicators": ["Interest Rate", "Currency"]
            },
            "gdp_data": {
                "title": "{country} GDP Grows {percent}% in Q{quarter}",
                "summary": "Economic growth {trend} expectations amid {factors}.",
                "impact": 3,
                "categories": ["economic_data", "gdp"],
                "indicators": ["GDP Growth"]
            },
            "trade_deal": {
                "title": "{country} Signs Major Trade Agreement with {partner}",
                "summary": "New trade partnership expected to boost {sectors} sectors.",
                "impact": 3,
                "categories": ["trade", "international"],
                "indicators": ["Trade Balance"]
            },
            "regulatory": {
                "title": "{country} Regulators Announce New {sector} Rules",
                "summary": "New regulations aim to {objective} in the {sector} sector.",
                "impact": 2,
                "categories": ["regulation", "policy"],
                "indicators": ["Regulatory Environment"]
            },
            "corporate": {
                "title": "Major {country} Company Reports {result} Earnings",
                "summary": "{company} performance {direction} on {factors}.",
                "impact": 2,
                "categories": ["earnings", "corporate"],
                "indicators": ["Corporate Performance"]
            }
        }

        # Randomly select event type
        event_type = random.choice(list(event_templates.keys()))
        template = event_templates[event_type]

        # Generate specific values
        values = self._generate_event_values(country_code, event_type)

        # Check impact threshold
        impact = template["impact"] + random.randint(-1, 1)
        if impact < min_impact:
            return None

        # Create event
        location = GeoLocation(
            country=mapping["name"],
            country_code=country_code,
            latitude=mapping["lat"] + random.uniform(-2, 2),  # Add some variation
            longitude=mapping["lng"] + random.uniform(-2, 2)
        )

        event = GeoNewsEvent(
            id=str(uuid4()),
            title=template["title"].format(country=mapping["name"], **values),
            summary=template["summary"].format(**values),
            url=f"https://example-news.com/{country_code.lower()}-{event_type}-{int(event_time.timestamp())}",
            source=f"{mapping['name']} Financial Times",
            published_at=event_time,
            location=location,
            impact_level=impact,
            categories=template["categories"],
            symbols_affected=self._get_affected_symbols(country_code, event_type),
            economic_indicators=template["indicators"],
            sentiment_score=random.uniform(-0.5, 0.8)  # Slightly positive bias
        )

        return event

    def _generate_event_values(self, country_code: str, event_type: str) -> dict:
        """Generate specific values for event templates."""

        import random

        if event_type == "monetary_policy":
            return {
                "action": random.choice(["Raises", "Cuts", "Maintains"]),
                "amount": f"{random.choice([0.25, 0.5, 0.75])}%",
                "reason": random.choice([
                    "combat inflation", "stimulate growth", "maintain stability"
                ])
            }
        elif event_type == "gdp_data":
            return {
                "percent": round(random.uniform(0.5, 4.2), 1),
                "quarter": random.choice(["1", "2", "3", "4"]),
                "trend": random.choice(["exceeds", "meets", "falls short of"]),
                "factors": random.choice([
                    "strong consumer spending", "export growth", "government investment"
                ])
            }
        elif event_type == "trade_deal":
            partners = ["United States", "European Union", "China", "Japan", "ASEAN"]
            return {
                "partner": random.choice(partners),
                "sectors": random.choice(["technology", "manufacturing", "agriculture"])
            }
        elif event_type == "regulatory":
            return {
                "sector": random.choice(["fintech", "banking", "technology", "energy"]),
                "objective": random.choice([
                    "enhance transparency", "improve security", "promote competition"
                ])
            }
        elif event_type == "corporate":
            return {
                "company": f"Leading {self.country_mappings[country_code]['name']} Corporation",
                "result": random.choice(["Strong", "Mixed", "Disappointing"]),
                "direction": random.choice(["improves", "declines", "stabilizes"]),
                "factors": random.choice([
                    "market expansion", "operational efficiency", "global demand"
                ])
            }

        return {}

    def _get_affected_symbols(self, country_code: str, event_type: str) -> list[str]:
        """Get symbols that might be affected by the event."""

        # Mock affected symbols by country
        country_symbols = {
            "US": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "GB": ["BP", "HSBA", "VODAFONE", "LLOY"],
            "DE": ["SAP", "ASML", "SIE", "BAYER"],
            "JP": ["SONY", "TOYOTA", "SoftBank"],
            "CN": ["BABA", "TENCENT", "JD"],
        }

        import random
        symbols = country_symbols.get(country_code, [])
        return random.sample(symbols, min(2, len(symbols))) if symbols else []

    async def _get_country_market_data(self) -> dict[str, dict]:
        """Get market data for all countries."""

        # Mock market data - in production, integrate with real market APIs
        import random

        market_data = {}
        for country_code, mapping in self.country_mappings.items():
            base_value = random.uniform(3000, 15000)
            change_pct = random.uniform(-3.5, 2.8)

            market_data[country_code] = {
                "index_name": mapping.get("major_index"),
                "index_value": round(base_value, 2),
                "index_change_pct": round(change_pct, 2),
                "currency": mapping.get("currency"),
                "currency_rate": round(random.uniform(0.5, 150), 4),
                "market_status": random.choice(["open", "closed", "pre_market"]),
                "volume": random.randint(100000000, 2000000000)
            }

        return market_data

    async def _get_single_country_market_data(self, country_code: str) -> dict:
        """Get detailed market data for a single country."""

        all_data = await self._get_country_market_data()
        return all_data.get(country_code, {})

    async def _get_economic_indicators(self, country_code: str) -> dict:
        """Get economic indicators for a country."""

        import random

        return {
            "gdp_growth": round(random.uniform(-2.0, 5.0), 2),
            "inflation_rate": round(random.uniform(0.5, 8.0), 2),
            "unemployment_rate": round(random.uniform(2.0, 12.0), 2),
            "interest_rate": round(random.uniform(0.0, 6.0), 2),
            "consumer_confidence": round(random.uniform(80, 120), 1),
            "manufacturing_pmi": round(random.uniform(45, 65), 1),
            "last_updated": datetime.utcnow().isoformat()
        }

    async def _get_country_news(
        self,
        country_code: str,
        days_back: int
    ) -> list[dict]:
        """Get news events for a specific country."""

        events = await self._generate_geo_news_events(days_back * 24, 1)

        # Filter events for this country
        country_events = [
            {
                "id": event.id,
                "title": event.title,
                "summary": event.summary,
                "url": event.url,
                "source": event.source,
                "published_at": event.published_at.isoformat(),
                "impact_level": event.impact_level,
                "categories": event.categories,
                "symbols_affected": event.symbols_affected,
                "sentiment_score": event.sentiment_score
            }
            for event in events
            if event.location.country_code == country_code
        ]

        return country_events

    def _calculate_country_metrics(self, events: list[GeoNewsEvent]) -> dict:
        """Calculate aggregate metrics by country."""

        country_metrics = {}

        for event in events:
            country = event.location.country_code

            if country not in country_metrics:
                country_metrics[country] = {
                    "count": 0,
                    "sentiment_sum": 0,
                    "max_impact": 0,
                    "events": []
                }

            metrics = country_metrics[country]
            metrics["count"] += 1
            metrics["sentiment_sum"] += event.sentiment_score
            metrics["max_impact"] = max(metrics["max_impact"], event.impact_level)

            # Store recent high-impact events
            if event.impact_level >= 3:
                metrics["events"].append({
                    "title": event.title,
                    "impact": event.impact_level,
                    "sentiment": event.sentiment_score,
                    "published_at": event.published_at.isoformat()
                })

        # Calculate averages
        for country, metrics in country_metrics.items():
            if metrics["count"] > 0:
                metrics["sentiment"] = metrics["sentiment_sum"] / metrics["count"]
            else:
                metrics["sentiment"] = 0

            # Keep only top 5 events
            metrics["events"] = sorted(
                metrics["events"],
                key=lambda x: x["impact"],
                reverse=True
            )[:5]

        return country_metrics

    def _calculate_global_sentiment(self, events: list[GeoNewsEvent]) -> float:
        """Calculate overall global sentiment."""

        if not events:
            return 0.0

        # Weight by impact level
        weighted_sum = sum(event.sentiment_score * event.impact_level for event in events)
        total_weight = sum(event.impact_level for event in events)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def _get_top_global_stories(
        self,
        events: list[GeoNewsEvent],
        limit: int
    ) -> list[dict]:
        """Get top global stories by impact."""

        # Sort by impact level and recency
        sorted_events = sorted(
            events,
            key=lambda x: (x.impact_level, x.published_at),
            reverse=True
        )

        return [
            {
                "id": event.id,
                "title": event.title,
                "summary": event.summary,
                "country": event.location.country,
                "country_code": event.location.country_code,
                "impact_level": event.impact_level,
                "sentiment_score": event.sentiment_score,
                "published_at": event.published_at.isoformat(),
                "categories": event.categories
            }
            for event in sorted_events[:limit]
        ]

    def _extract_categories(self, events: list[dict]) -> list[dict]:
        """Extract trending news categories."""

        from collections import Counter

        all_categories = []
        for event in events:
            all_categories.extend(event.get("categories", []))

        category_counts = Counter(all_categories)

        return [
            {"category": cat, "count": count}
            for cat, count in category_counts.most_common(10)
        ]

    def _extract_trending_symbols(self, events: list[dict]) -> list[dict]:
        """Extract trending symbols from news."""

        from collections import Counter

        all_symbols = []
        for event in events:
            all_symbols.extend(event.get("symbols_affected", []))

        symbol_counts = Counter(all_symbols)

        return [
            {"symbol": symbol, "mention_count": count}
            for symbol, count in symbol_counts.most_common(10)
        ]

    async def store_geo_events(self, events: list[GeoNewsEvent]):
        """Store geo-located events in database."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            for event in events:
                try:
                    await conn.execute("""
                        INSERT INTO geo_news_events (
                            id, title, summary, url, source, published_at,
                            country, country_code, latitude, longitude,
                            impact_level, categories, symbols_affected,
                            economic_indicators, sentiment_score
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                        ON CONFLICT (url) DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            impact_level = EXCLUDED.impact_level,
                            sentiment_score = EXCLUDED.sentiment_score
                    """,
                        event.id, event.title, event.summary, event.url, event.source,
                        event.published_at, event.location.country, event.location.country_code,
                        event.location.latitude, event.location.longitude,
                        event.impact_level, json.dumps(event.categories),
                        json.dumps(event.symbols_affected), json.dumps(event.economic_indicators),
                        event.sentiment_score
                    )

                except Exception as e:
                    logger.warning(f"Failed to store geo event {event.id}: {e}")


# Global geo news service
_geo_news_service = None

def get_geo_news_service() -> GeoNewsService:
    """Get the global geo news service instance."""
    global _geo_news_service
    if _geo_news_service is None:
        _geo_news_service = GeoNewsService()
    return _geo_news_service


# Background task for geo news data generation
async def generate_geo_news_data():
    """Background task to generate geo news data for globe visualization."""

    logger.info("🌍 Generating geo news data for globe visualization...")

    try:
        geo_service = get_geo_news_service()

        # Generate global heatmap data
        heatmap_data = await geo_service.get_global_news_heatmap(hours_back=48)

        logger.success(f"Generated globe data: {heatmap_data['total_events']} events across {len(heatmap_data['countries'])} countries")

        return heatmap_data

    except Exception as e:
        logger.error(f"Geo news data generation failed: {e}")


if __name__ == "__main__":
    # Test the geo news service
    async def test_geo_news():
        service = GeoNewsService()

        # Test heatmap generation
        heatmap = await service.get_global_news_heatmap(hours_back=24)
        print(f"Generated heatmap with {heatmap['total_events']} events")

        # Test country detail
        us_detail = await service.get_country_detail("US", days_back=3)
        print(f"US has {us_detail['news_summary']['total_events']} recent events")

    asyncio.run(test_geo_news())
