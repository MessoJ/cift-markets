"""
News Geographic Enrichment Service

Maps news sources and symbols to countries and adds geographic coordinates.
NO MOCK DATA - Uses real source-to-country mappings.

Usage:
    python scripts/enrich_news_geo.py
"""

import asyncio
import os

import asyncpg
from loguru import logger

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "cift_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "cift_pass"),
    "database": os.getenv("POSTGRES_DB", "cift_markets"),
}

# Source-to-Country mapping (based on known news sources)
SOURCE_COUNTRY_MAP = {
    # US Sources
    "Yahoo": ("United States", "US", 37.0902, -95.7129, "Americas"),
    "Bloomberg": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "Reuters": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "CNBC": ("United States", "US", 40.7579, -73.9855, "Americas"),
    "MarketWatch": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "The Wall Street Journal": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "Barron's": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "Investor's Business Daily": ("United States", "US", 34.0522, -118.2437, "Americas"),
    "TheStreet": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "Seeking Alpha": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "Motley Fool": ("United States", "US", 38.8951, -77.0364, "Americas"),
    "Benzinga": ("United States", "US", 42.3314, -83.0458, "Americas"),
    "Barchart.com": ("United States", "US", 41.8781, -87.6298, "Americas"),
    "Thefly.com": ("United States", "US", 40.7128, -74.0060, "Americas"),
    "Yahoo Entertainment": ("United States", "US", 37.4163, -122.0256, "Americas"),
    "Biztoc.com": ("United States", "US", 37.7749, -122.4194, "Americas"),
    "Moneyweb.co.za": ("South Africa", "ZA", -25.7479, 28.2293, "Africa"),

    # UK Sources
    "BBC": ("United Kingdom", "GB", 51.5074, -0.1278, "Europe"),
    "Financial Times": ("United Kingdom", "GB", 51.5074, -0.1278, "Europe"),
    "The Guardian": ("United Kingdom", "GB", 51.5074, -0.1278, "Europe"),
    "The Telegraph": ("United Kingdom", "GB", 51.5074, -0.1278, "Europe"),
    "City A.M.": ("United Kingdom", "GB", 51.5074, -0.1278, "Europe"),

    # European Sources
    "Dpa-international.com": ("Germany", "DE", 52.5200, 13.4050, "Europe"),
    "Handelsblatt": ("Germany", "DE", 51.2277, 6.7735, "Europe"),
    "Les Echos": ("France", "FR", 48.8566, 2.3522, "Europe"),
    "Il Sole 24 Ore": ("Italy", "IT", 45.4642, 9.1900, "Europe"),

    # Asian Sources
    "Nikkei": ("Japan", "JP", 35.6762, 139.6503, "Asia"),
    "The Times of India": ("India", "IN", 19.0760, 72.8777, "Asia"),
    "Economic Times": ("India", "IN", 28.6139, 77.2090, "Asia"),
    "South China Morning Post": ("Hong Kong", "HK", 22.3193, 114.1694, "Asia"),
    "The Straits Times": ("Singapore", "SG", 1.3521, 103.8198, "Asia"),

    # Canadian Sources
    "Globe and Mail": ("Canada", "CA", 43.6532, -79.3832, "Americas"),
    "Financial Post": ("Canada", "CA", 43.6532, -79.3832, "Americas"),

    # Australian Sources
    "Australian Financial Review": ("Australia", "AU", -33.8688, 151.2093, "Oceania"),
    "Sydney Morning Herald": ("Australia", "AU", -33.8688, 151.2093, "Oceania"),
}

# Symbol-to-Country mapping (for symbol-specific news)
SYMBOL_COUNTRY_MAP = {
    "AAPL": ("United States", "US", 37.3346, -122.0090, "Americas"),  # Cupertino
    "GOOGL": ("United States", "US", 37.4220, -122.0841, "Americas"),  # Mountain View
    "MSFT": ("United States", "US", 47.6440, -122.1290, "Americas"),  # Redmond
    "AMZN": ("United States", "US", 47.6062, -122.3321, "Americas"),  # Seattle
    "TSLA": ("United States", "US", 30.2672, -97.7431, "Americas"),  # Austin
    "META": ("United States", "US", 37.4847, -122.1477, "Americas"),  # Menlo Park
    "NVDA": ("United States", "US", 37.3708, -121.9643, "Americas"),  # Santa Clara
    "AMD": ("United States", "US", 37.3860, -121.9636, "Americas"),  # Santa Clara
    "BABA": ("China", "CN", 30.2741, 120.1551, "Asia"),  # Hangzhou
    "TSM": ("Taiwan", "TW", 25.0330, 121.5654, "Asia"),  # Taipei
    "SAP": ("Germany", "DE", 49.2935, 8.6411, "Europe"),  # Walldorf
    "ASML": ("Netherlands", "NL", 51.4416, 5.4697, "Europe"),  # Veldhoven
}


async def enrich_news_with_geo(pool: asyncpg.Pool):
    """Enrich existing news articles with geographic data"""

    logger.info("Starting geographic enrichment of news articles...")

    # Get all articles without geo data
    async with pool.acquire() as conn:
        articles = await conn.fetch(
            """
            SELECT id, source, symbols
            FROM news_articles
            WHERE country_code IS NULL
            """
        )

        logger.info(f"Found {len(articles)} articles to enrich")

        enriched_count = 0

        for article in articles:
            article_id = article['id']
            source = article['source']
            symbols = article['symbols'] or []

            # Try to match by source first
            geo_data = None
            for source_key, data in SOURCE_COUNTRY_MAP.items():
                if source and source_key.lower() in source.lower():
                    geo_data = data
                    break

            # If no source match, try symbol-based geo
            if not geo_data and symbols:
                for symbol in symbols:
                    if symbol in SYMBOL_COUNTRY_MAP:
                        geo_data = SYMBOL_COUNTRY_MAP[symbol]
                        break

            # Default to US if no match (since most financial news is US-centric)
            if not geo_data:
                geo_data = ("United States", "US", 40.7128, -74.0060, "Americas")

            country, country_code, lat, lng, region = geo_data

            # Update article
            try:
                await conn.execute(
                    """
                    UPDATE news_articles
                    SET country = $1, country_code = $2, latitude = $3, longitude = $4, region = $5
                    WHERE id = $6
                    """,
                    country, country_code, lat, lng, region, article_id
                )
                enriched_count += 1

                if enriched_count % 50 == 0:
                    logger.info(f"Enriched {enriched_count}/{len(articles)} articles...")

            except Exception as e:
                logger.error(f"Error enriching article {article_id}: {e}")
                continue

        logger.info(f"✅ Successfully enriched {enriched_count} articles with geographic data")


async def main():
    """Main entry point"""

    # Create database connection pool
    pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=5)

    try:
        await enrich_news_with_geo(pool)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
