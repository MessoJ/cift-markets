"""
Generate News Geotags and Connections
Auto-tags news articles with geographic locations and creates connections between markets.
"""

import asyncio
import os
import re

import asyncpg
from loguru import logger

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "cift_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "cift_password"),
    "database": os.getenv("POSTGRES_DB", "cift_markets"),
}

# Exchange keywords for detection
EXCHANGE_KEYWORDS = {
    "NYSE": ["NYSE", "New York Stock Exchange", "Wall Street", "New York"],
    "NASDAQ": ["NASDAQ", "Nasdaq", "tech stocks"],
    "LSE": ["LSE", "London Stock Exchange", "FTSE", "London"],
    "TSE": ["TSE", "Tokyo Stock Exchange", "Nikkei", "Tokyo"],
    "SSE": ["SSE", "Shanghai Stock Exchange", "Shanghai", "China A-shares"],
    "SZSE": ["SZSE", "Shenzhen Stock Exchange", "Shenzhen"],
    "HKEX": ["HKEX", "Hong Kong Stock Exchange", "Hang Seng", "Hong Kong"],
    "BSE": ["BSE", "Bombay Stock Exchange", "Sensex", "Mumbai"],
    "NSE": ["NSE", "National Stock Exchange of India"],
    "ENX": ["Euronext", "Paris", "CAC 40"],
    "DB": ["Deutsche Börse", "Frankfurt", "DAX"],
    "TSX": ["TSX", "Toronto Stock Exchange", "Toronto"],
    "ASX": ["ASX", "Australian Securities Exchange", "Sydney"],
    "KRX": ["KRX", "Korea Exchange", "KOSPI", "Seoul"],
    "B3": ["B3", "BM&F Bovespa", "São Paulo", "Ibovespa"],
    "TADAWUL": ["Tadawul", "Saudi Stock Exchange", "Riyadh"],
    "NSE_KE": ["Nairobi Securities Exchange", "Nairobi", "Kenya"],
    "JSE": ["JSE", "Johannesburg Stock Exchange", "Johannesburg"],
}

# Country detection keywords
COUNTRY_KEYWORDS = {
    "US": ["United States", "USA", "U.S.", "America", "American"],
    "GB": ["United Kingdom", "UK", "Britain", "British"],
    "CN": ["China", "Chinese", "Beijing"],
    "JP": ["Japan", "Japanese", "Tokyo"],
    "DE": ["Germany", "German", "Frankfurt"],
    "FR": ["France", "French", "Paris"],
    "CA": ["Canada", "Canadian", "Toronto"],
    "AU": ["Australia", "Australian", "Sydney"],
    "IN": ["India", "Indian", "Mumbai"],
    "HK": ["Hong Kong"],
    "KR": ["South Korea", "Korean", "Seoul"],
    "BR": ["Brazil", "Brazilian"],
    "SA": ["Saudi Arabia", "Saudi"],
    "KE": ["Kenya", "Nairobi"],
    "ZA": ["South Africa", "Johannesburg"],
}


async def get_exchanges(conn: asyncpg.Connection) -> dict[str, dict]:
    """Fetch all stock exchanges."""
    query = "SELECT id, code, name, country_code, lat, lng FROM stock_exchanges"
    rows = await conn.fetch(query)

    exchanges = {}
    for row in rows:
        exchanges[row['code']] = {
            'id': row['id'],
            'code': row['code'],
            'name': row['name'],
            'country_code': row['country_code'],
            'lat': row['lat'],
            'lng': row['lng'],
        }

    return exchanges


def detect_exchanges_in_text(text: str, exchanges: dict) -> list[tuple[str, float]]:
    """
    Detect mentions of stock exchanges in text.
    Returns list of (exchange_code, relevance_score) tuples.
    """
    text_upper = text.upper()
    detected = []

    for code, keywords in EXCHANGE_KEYWORDS.items():
        if code not in exchanges:
            continue

        # Count keyword mentions
        mentions = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.upper()) + r'\b'
            mentions += len(re.findall(pattern, text_upper))

        if mentions > 0:
            # Calculate relevance score (0-1)
            # More mentions = higher score
            relevance = min(mentions * 0.2, 1.0)
            detected.append((code, relevance))

    # Sort by relevance
    detected.sort(key=lambda x: x[1], reverse=True)

    return detected


def detect_country_in_text(text: str) -> list[str]:
    """Detect country mentions in text."""
    text_upper = text.upper()
    detected = []

    for country_code, keywords in COUNTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.upper() in text_upper:
                detected.append(country_code)
                break

    return list(set(detected))


def detect_connections(article: dict, detected_exchanges: list[tuple[str, float]]) -> list[dict]:
    """
    Detect connections between markets based on article content.
    Returns list of connection dictionaries.
    """
    connections = []

    if len(detected_exchanges) < 2:
        return connections

    # Get article text
    text = f"{article['title']} {article['summary']}".lower()

    # Determine connection type based on keywords
    connection_type = "correlation"  # default

    if any(word in text for word in ["trade", "trading", "deal", "merger", "acquisition", "partnership"]):
        connection_type = "trade"
    elif any(word in text for word in ["impact", "affect", "influence", "spillover", "effect"]):
        connection_type = "impact"
    elif any(word in text for word in ["similar", "follow", "track", "mirror", "correlation"]):
        connection_type = "correlation"

    # Create connections between all detected exchanges
    for i in range(len(detected_exchanges) - 1):
        source_code, source_relevance = detected_exchanges[i]
        for j in range(i + 1, len(detected_exchanges)):
            target_code, target_relevance = detected_exchanges[j]

            # Connection strength based on relevance scores
            strength = (source_relevance + target_relevance) / 2

            connections.append({
                'source_code': source_code,
                'target_code': target_code,
                'connection_type': connection_type,
                'strength': strength,
            })

    return connections


async def process_article(
    conn: asyncpg.Connection,
    article: dict,
    exchanges: dict,
    stats: dict,
):
    """Process a single article to create geotags and connections."""

    article_id = article['id']
    text = f"{article['title']} {article['summary']} {article['content'] or ''}"

    # Detect exchanges
    detected_exchanges = detect_exchanges_in_text(text, exchanges)

    if not detected_exchanges:
        # Try to detect by country
        countries = detect_country_in_text(text)
        for country_code in countries:
            # Find exchanges in this country
            country_exchanges = [
                (code, 0.3) for code, ex in exchanges.items()
                if ex['country_code'] == country_code
            ]
            detected_exchanges.extend(country_exchanges)

    if not detected_exchanges:
        stats['no_location'] += 1
        return

    # Create geotags
    for exchange_code, relevance in detected_exchanges[:5]:  # Limit to top 5
        exchange = exchanges[exchange_code]

        try:
            await conn.execute("""
                INSERT INTO news_geotags (article_id, exchange_id, country_code, lat, lng, relevance_score)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT DO NOTHING
            """, article_id, exchange['id'], exchange['country_code'],
                exchange['lat'], exchange['lng'], relevance)

            stats['geotags_created'] += 1
        except Exception as e:
            logger.error(f"Error creating geotag: {e}")

    # Detect and create connections
    connections = detect_connections(article, detected_exchanges)

    for conn_data in connections:
        source_id = exchanges[conn_data['source_code']]['id']
        target_id = exchanges[conn_data['target_code']]['id']

        try:
            await conn.execute("""
                INSERT INTO news_connections
                (source_exchange_id, target_exchange_id, article_id, connection_type, strength)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (source_exchange_id, target_exchange_id, article_id) DO NOTHING
            """, source_id, target_id, article_id,
                conn_data['connection_type'], conn_data['strength'])

            stats['connections_created'] += 1
        except Exception as e:
            logger.error(f"Error creating connection: {e}")


async def main():
    """Main function to generate geotags and connections."""

    logger.info("=" * 60)
    logger.info("  NEWS GEOTAG & CONNECTION GENERATOR")
    logger.info("=" * 60)
    logger.info("")

    # Connect to database
    conn = await asyncpg.connect(**DB_CONFIG)

    try:
        # Get all exchanges
        logger.info("📍 Loading stock exchanges...")
        exchanges = await get_exchanges(conn)
        logger.info(f"✅ Loaded {len(exchanges)} exchanges")

        # Get articles without geotags (or all recent articles)
        logger.info("📰 Fetching articles...")
        query = """
            SELECT a.id, a.title, a.summary, a.content, a.published_at
            FROM news_articles a
            WHERE a.published_at >= NOW() - INTERVAL '30 days'
            ORDER BY a.published_at DESC
            LIMIT 1000
        """

        articles = await conn.fetch(query)
        logger.info(f"✅ Found {len(articles)} articles to process")
        logger.info("")

        # Process articles
        stats = {
            'processed': 0,
            'geotags_created': 0,
            'connections_created': 0,
            'no_location': 0,
        }

        logger.info("🔄 Processing articles...")
        for i, article in enumerate(articles, 1):
            await process_article(conn, article, exchanges, stats)
            stats['processed'] += 1

            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(articles)} articles processed")

        logger.info("")
        logger.info("=" * 60)
        logger.info("  RESULTS")
        logger.info("=" * 60)
        logger.info(f"✅ Articles processed: {stats['processed']}")
        logger.info(f"✅ Geotags created: {stats['geotags_created']}")
        logger.info(f"✅ Connections created: {stats['connections_created']}")
        logger.info(f"⚠️  Articles without location: {stats['no_location']}")

        # Show top exchanges by article count
        logger.info("")
        logger.info("📊 Top Exchanges by Article Count:")
        top_exchanges = await conn.fetch("""
            SELECT e.code, e.name, COUNT(DISTINCT gt.article_id) as count
            FROM stock_exchanges e
            JOIN news_geotags gt ON gt.exchange_id = e.id
            GROUP BY e.code, e.name
            ORDER BY count DESC
            LIMIT 10
        """)

        for i, row in enumerate(top_exchanges, 1):
            logger.info(f"  {i}. {row['code']:8s} {row['name']:40s} {row['count']} articles")

        # Show connection types
        logger.info("")
        logger.info("🔗 Connection Types:")
        connection_types = await conn.fetch("""
            SELECT connection_type, COUNT(*) as count
            FROM news_connections
            GROUP BY connection_type
            ORDER BY count DESC
        """)

        for row in connection_types:
            logger.info(f"  {row['connection_type']:15s} {row['count']} connections")

        logger.info("")
        logger.info("✅ DONE!")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
