"""
Globe API Routes
Provides data for interactive 3D globe visualization:
- Stock exchange markers with news counts
- News connection arcs between markets
- Country-level news aggregation
- Advanced search and filtering
"""

from datetime import datetime, timedelta
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.services.geo_news_service import get_geo_news_service

router = APIRouter(prefix="/globe", tags=["globe"])


async def get_db():
    """Database dependency."""
    pool = await get_postgres_pool()
    async with pool.acquire() as conn:
        yield conn


@router.get("/heatmap")
async def get_global_heatmap(
    hours_back: int = Query(24, ge=1, le=168),
    impact_threshold: int = Query(2, ge=1, le=5),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get global news heatmap data for interactive globe visualization.

    Query params:
    - hours_back: Hours to look back (1-168)
    - impact_threshold: Minimum impact level (1-5)
    """

    try:
        geo_service = get_geo_news_service()
        heatmap_data = await geo_service.get_global_news_heatmap(
            hours_back=hours_back,
            impact_threshold=impact_threshold
        )

        return heatmap_data

    except Exception as e:
        logger.error(f"Failed to get global heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/country/{country_code}")
async def get_country_detail(
    country_code: str,
    days_back: int = Query(7, ge=1, le=30),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get detailed information for a specific country.

    Path params:
    - country_code: ISO 3166-1 alpha-2 country code

    Query params:
    - days_back: Days to look back (1-30)
    """

    try:
        geo_service = get_geo_news_service()
        country_data = await geo_service.get_country_detail(
            country_code=country_code.upper(),
            days_back=days_back
        )

        return country_data

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get country detail: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/exchanges")
async def get_globe_exchanges_legacy(
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    min_articles: int = Query(0, ge=0),
    db: asyncpg.Connection = Depends(get_db),
):
    """
    Get all stock exchanges with news counts and sentiment (legacy endpoint).

    Query params:
    - timeframe: 1h, 24h, 7d, 30d
    - min_articles: Minimum article count to include
    """

    # Calculate time window
    time_windows = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    start_time = datetime.utcnow() - time_windows[timeframe]

    # Query exchanges with article counts
    query = """
        SELECT
            e.id,
            e.code,
            e.name,
            e.country,
            e.country_code,
            e.lat,
            e.lng,
            e.timezone,
            e.market_cap_usd,
            e.website,
            e.icon_url,
            COUNT(DISTINCT gt.article_id) as news_count,
            COALESCE(
                AVG(CASE
                    WHEN a.sentiment = 'positive' THEN 0.7
                    WHEN a.sentiment = 'negative' THEN -0.7
                    ELSE 0
                END),
                0
            ) as avg_sentiment,
            ARRAY_AGG(DISTINCT a.categories->>0) FILTER (WHERE a.categories IS NOT NULL) as categories,
            json_agg(
                json_build_object(
                    'id', a.id,
                    'title', a.title,
                    'summary', a.summary,
                    'published_at', a.published_at,
                    'sentiment', a.sentiment,
                    'category', a.categories->>0
                ) ORDER BY a.published_at DESC
            ) FILTER (WHERE a.id IS NOT NULL) as latest_articles
        FROM stock_exchanges e
        LEFT JOIN news_geotags gt ON gt.exchange_id = e.id
        LEFT JOIN news_articles a ON a.id = gt.article_id
            AND a.published_at >= $1
        WHERE e.is_active = true
        GROUP BY e.id, e.code, e.name, e.country, e.country_code,
                 e.lat, e.lng, e.timezone, e.market_cap_usd, e.website, e.icon_url
        HAVING COUNT(DISTINCT gt.article_id) >= $2
        ORDER BY COUNT(DISTINCT gt.article_id) DESC
    """

    try:
        rows = await db.fetch(query, start_time, min_articles)

        exchanges = []
        total_news = 0

        for row in rows:
            # Get flag emoji for country
            flag = get_flag_emoji(row['country_code'])

            # Limit latest articles to top 5
            latest = row['latest_articles'][:5] if row['latest_articles'] else []

            exchange_data = {
                "id": str(row['id']),
                "code": row['code'],
                "name": row['name'],
                "country": row['country'],
                "country_code": row['country_code'],
                "flag": flag,
                "lat": float(row['lat']),
                "lng": float(row['lng']),
                "timezone": row['timezone'],
                "market_cap_usd": row['market_cap_usd'],
                "website": row['website'],
                "icon_url": row['icon_url'],
                "news_count": row['news_count'],
                "sentiment_score": float(row['avg_sentiment']),
                "categories": row['categories'] or [],
                "latest_articles": latest,
            }

            exchanges.append(exchange_data)
            total_news += row['news_count']

        return {
            "exchanges": exchanges,
            "total_count": len(exchanges),
            "total_news_count": total_news,
            "timeframe": timeframe,
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching globe exchanges: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch exchange data") from e


@router.get("/arcs")
async def get_news_arcs(
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    min_strength: float = Query(0.3, ge=0.0, le=1.0),
    connection_type: str | None = Query(None, regex="^(trade|impact|correlation|all)$"),
    db: asyncpg.Connection = Depends(get_db),
):
    """
    Get news connections for arc visualization.

    Query params:
    - timeframe: 1h, 24h, 7d, 30d
    - min_strength: Minimum connection strength (0-1)
    - connection_type: trade, impact, correlation, or all
    """

    time_windows = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    start_time = datetime.utcnow() - time_windows[timeframe]

    # Build query with optional type filter
    type_filter = ""
    params = [start_time, min_strength]

    if connection_type and connection_type != "all":
        type_filter = "AND nc.connection_type = $3"
        params.append(connection_type)

    query = f"""
        SELECT
            nc.id,
            nc.connection_type,
            nc.strength,
            COUNT(DISTINCT nc.article_id) as article_count,
            se_source.code as source_code,
            se_source.name as source_name,
            se_source.lat as source_lat,
            se_source.lng as source_lng,
            se_target.code as target_code,
            se_target.name as target_name,
            se_target.lat as target_lat,
            se_target.lng as target_lng,
            json_agg(
                json_build_object(
                    'id', a.id,
                    'title', a.title,
                    'category', a.categories->>0,
                    'sentiment', a.sentiment
                ) ORDER BY a.published_at DESC
            ) FILTER (WHERE a.id IS NOT NULL) as articles
        FROM news_connections nc
        JOIN stock_exchanges se_source ON se_source.id = nc.source_exchange_id
        JOIN stock_exchanges se_target ON se_target.id = nc.target_exchange_id
        LEFT JOIN news_articles a ON a.id = nc.article_id
            AND a.published_at >= $1
        WHERE nc.strength >= $2
        {type_filter}
        GROUP BY nc.id, nc.connection_type, nc.strength,
                 se_source.code, se_source.name, se_source.lat, se_source.lng,
                 se_target.code, se_target.name, se_target.lat, se_target.lng
        HAVING COUNT(DISTINCT nc.article_id) > 0
        ORDER BY nc.strength DESC, COUNT(DISTINCT nc.article_id) DESC
        LIMIT 100
    """

    try:
        rows = await db.fetch(query, *params)

        arcs = []
        for row in rows:
            # Determine arc color based on connection type
            color_map = {
                "trade": ["#00ff88", "#0088ff"],      # Green to Blue
                "impact": ["#ff8800", "#ff0088"],     # Orange to Pink
                "correlation": ["#8800ff", "#00ffff"], # Purple to Cyan
            }

            color = color_map.get(row['connection_type'], ["#0088ff", "#00ff88"])

            arc_data = {
                "id": str(row['id']),
                "source": {
                    "code": row['source_code'],
                    "name": row['source_name'],
                    "lat": float(row['source_lat']),
                    "lng": float(row['source_lng']),
                },
                "target": {
                    "code": row['target_code'],
                    "name": row['target_name'],
                    "lat": float(row['target_lat']),
                    "lng": float(row['target_lng']),
                },
                "article_count": row['article_count'],
                "connection_type": row['connection_type'],
                "strength": float(row['strength']),
                "color": color,
                "articles": row['articles'][:3] if row['articles'] else [],
            }

            arcs.append(arc_data)

        return {
            "arcs": arcs,
            "total_count": len(arcs),
            "timeframe": timeframe,
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching news arcs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch arc data") from e


@router.get("/boundaries")
async def get_political_boundaries(
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    db: asyncpg.Connection = Depends(get_db),
):
    """
    Get country-level news aggregation for political boundary visualization.

    Query params:
    - timeframe: 1h, 24h, 7d, 30d
    """

    time_windows = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    start_time = datetime.utcnow() - time_windows[timeframe]

    query = """
        SELECT
            gt.country_code,
            e.country as country_name,
            COUNT(DISTINCT gt.article_id) as article_count,
            COALESCE(
                AVG(CASE
                    WHEN a.sentiment = 'positive' THEN 0.7
                    WHEN a.sentiment = 'negative' THEN -0.7
                    ELSE 0
                END),
                0
            ) as avg_sentiment,
            ARRAY_AGG(DISTINCT a.categories->>0) FILTER (WHERE a.categories IS NOT NULL) as categories,
            ARRAY_AGG(DISTINCT e.code) FILTER (WHERE e.code IS NOT NULL) as exchanges
        FROM news_geotags gt
        LEFT JOIN news_articles a ON a.id = gt.article_id
            AND a.published_at >= $1
        LEFT JOIN stock_exchanges e ON e.country_code = gt.country_code
        WHERE gt.country_code IS NOT NULL
        GROUP BY gt.country_code, e.country
        HAVING COUNT(DISTINCT gt.article_id) > 0
        ORDER BY COUNT(DISTINCT gt.article_id) DESC
    """

    try:
        rows = await db.fetch(query, start_time)

        countries = []
        for row in rows:
            flag = get_flag_emoji(row['country_code'])

            country_data = {
                "country_code": row['country_code'],
                "name": row['country_name'],
                "flag": flag,
                "article_count": row['article_count'],
                "sentiment_score": float(row['avg_sentiment']),
                "top_categories": row['categories'] or [],
                "exchanges": row['exchanges'] or [],
            }

            countries.append(country_data)

        return {
            "countries": countries,
            "total_count": len(countries),
            "timeframe": timeframe,
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching boundary data: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch boundary data") from e


@router.get("/search")
async def search_globe_data(
    q: str = Query("", min_length=0),
    exchanges: list[str] = Query([]),
    countries: list[str] = Query([]),
    categories: list[str] = Query([]),
    sentiment: str | None = Query(None, regex="^(positive|neutral|negative)$"),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    db: asyncpg.Connection = Depends(get_db),
):
    """
    Search and filter globe data.

    Query params:
    - q: Search query (exchange name, country, etc.)
    - exchanges: Filter by exchange codes (e.g., NYSE, LSE)
    - countries: Filter by country codes (e.g., US, GB)
    - categories: Filter by news categories
    - sentiment: positive, neutral, or negative
    - date_from/date_to: Date range filter
    """

    # Set default date range if not provided
    if not date_from:
        date_from = datetime.utcnow() - timedelta(days=7)
    if not date_to:
        date_to = datetime.utcnow()

    # Build dynamic WHERE clauses
    where_clauses = ["a.published_at BETWEEN $1 AND $2"]
    params = [date_from, date_to]
    param_count = 2

    # Text search
    if q:
        param_count += 1
        where_clauses.append(f"""(
            e.name ILIKE ${param_count} OR
            e.code ILIKE ${param_count} OR
            e.country ILIKE ${param_count}
        )""")
        params.append(f"%{q}%")

    # Exchange filter
    if exchanges:
        param_count += 1
        where_clauses.append(f"e.code = ANY(${param_count})")
        params.append(exchanges)

    # Country filter
    if countries:
        param_count += 1
        where_clauses.append(f"e.country_code = ANY(${param_count})")
        params.append(countries)

    # Category filter
    if categories:
        param_count += 1
        # Check if any of the requested categories exist in the article's categories array
        where_clauses.append(f"a.categories ?| ${param_count}")
        params.append(categories)

    # Sentiment filter
    if sentiment:
        sentiment_ranges = {
            "positive": "a.sentiment_score > 0.2",
            "neutral": "a.sentiment_score BETWEEN -0.2 AND 0.2",
            "negative": "a.sentiment_score < -0.2",
        }
        where_clauses.append(sentiment_ranges[sentiment])

    where_clause = " AND ".join(where_clauses)

    query = f"""
        SELECT
            e.id,
            e.code,
            e.name,
            e.country,
            e.country_code,
            e.lat,
            e.lng,
            COUNT(DISTINCT a.id) as article_count,
            COALESCE(AVG(a.sentiment_score), 0) as avg_sentiment
        FROM stock_exchanges e
        LEFT JOIN news_geotags gt ON gt.exchange_id = e.id
        LEFT JOIN news_articles a ON a.id = gt.article_id
        WHERE {where_clause}
        GROUP BY e.id, e.code, e.name, e.country, e.country_code, e.lat, e.lng
        HAVING COUNT(DISTINCT a.id) > 0
        ORDER BY COUNT(DISTINCT a.id) DESC
    """

    try:
        rows = await db.fetch(query, *params)

        results = []
        total_articles = 0

        for row in rows:
            flag = get_flag_emoji(row['country_code'])

            result = {
                "id": str(row['id']),
                "code": row['code'],
                "name": row['name'],
                "country": row['country'],
                "country_code": row['country_code'],
                "flag": flag,
                "lat": float(row['lat']),
                "lng": float(row['lng']),
                "article_count": row['article_count'],
                "sentiment_score": float(row['avg_sentiment']),
            }

            results.append(result)
            total_articles += row['article_count']

        return {
            "results": results,
            "total_count": len(results),
            "total_articles": total_articles,
            "filters_applied": {
                "query": q,
                "exchanges": exchanges,
                "countries": countries,
                "categories": categories,
                "sentiment": sentiment,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
            },
            "execution_time_ms": 0,  # TODO: Add timing
        }

    except Exception as e:
        logger.error(f"Error searching globe data: {e}")
        raise HTTPException(status_code=500, detail="Failed to search globe data") from e


@router.get("/ships")
async def get_tracked_ships(
    ship_type: str | None = Query(None),
    min_importance: int = Query(0, ge=0, le=100),
    status: str | None = Query(None),
    db: asyncpg.Connection = Depends(get_db),
):
    """
    Get tracked ships with real-time positions.

    Query params:
    - ship_type: Filter by type (oil_tanker, lng_carrier, container, bulk_carrier, chemical_tanker)
    - min_importance: Minimum importance score (0-100)
    - status: Filter by status (underway, at_anchor, moored)
    """

    query = """
        SELECT
            id,
            mmsi,
            imo,
            ship_name,
            ship_type,
            flag_country,
            flag_country_code,
            deadweight_tonnage,
            current_lat,
            current_lng,
            current_speed,
            current_course,
            current_status,
            destination,
            eta,
            cargo_type,
            cargo_value_usd,
            importance_score,
            news_count,
            avg_sentiment,
            last_updated
        FROM ships_current_status
        WHERE current_lat IS NOT NULL
            AND current_lng IS NOT NULL
    """

    params = []
    param_count = 0

    if ship_type:
        param_count += 1
        query += f" AND ship_type = ${param_count}"
        params.append(ship_type)

    if min_importance > 0:
        param_count += 1
        query += f" AND importance_score >= ${param_count}"
        params.append(min_importance)

    if status:
        param_count += 1
        query += f" AND current_status = ${param_count}"
        params.append(status)

    query += " ORDER BY importance_score DESC, cargo_value_usd DESC"

    try:
        # Check if table exists first
        table_check = await db.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'ships_current_status'
            )
        """)

        if not table_check:
            logger.warning("ships_current_status table does not exist, returning empty ships list")
            return {
                "ships": [],
                "total_count": 0,
                "filters": {
                    "ship_type": ship_type,
                    "min_importance": min_importance,
                    "status": status,
                },
                "last_updated": datetime.utcnow().isoformat(),
                "note": "Ships tracking not yet configured"
            }

        rows = await db.fetch(query, *params)

        ships = []
        for row in rows:
            ship_data = {
                "id": str(row['id']),
                "mmsi": row['mmsi'],
                "imo": row['imo'],
                "ship_name": row['ship_name'],
                "ship_type": row['ship_type'],
                "flag_country": row['flag_country'],
                "flag_country_code": row['flag_country_code'],
                "deadweight_tonnage": row['deadweight_tonnage'],
                "current_lat": float(row['current_lat']),
                "current_lng": float(row['current_lng']),
                "current_speed": float(row['current_speed']) if row['current_speed'] else 0.0,
                "current_course": float(row['current_course']) if row['current_course'] else 0.0,
                "current_status": row['current_status'],
                "destination": row['destination'],
                "eta": row['eta'].isoformat() if row['eta'] else None,
                "cargo_type": row['cargo_type'],
                "cargo_value_usd": row['cargo_value_usd'],
                "importance_score": row['importance_score'],
                "news_count": row['news_count'],
                "avg_sentiment": float(row['avg_sentiment']) if row['avg_sentiment'] is not None else 0.0,
                "last_updated": row['last_updated'].isoformat() if row['last_updated'] else None,
            }

            ships.append(ship_data)

        return {
            "ships": ships,
            "total_count": len(ships),
            "filters": {
                "ship_type": ship_type,
                "min_importance": min_importance,
                "status": status,
            },
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching ships: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ship data") from e


@router.get("/countries/{country_code}")
async def get_country_details(
    country_code: str,
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    db: asyncpg.Connection = Depends(get_db),
):
    """
    Get comprehensive country details including:
    - Economic indicators (GDP, inflation, unemployment)
    - News analysis (sentiment, article count)
    - Top market-relevant news
    - Exchange and asset counts

    Path params:
    - country_code: ISO 2-letter country code (e.g., 'US', 'NG', 'CN')

    Query params:
    - timeframe: News analysis window (1h, 24h, 7d, 30d)
    """

    try:
        country_code = country_code.upper()

        # Calculate time window
        time_windows = {
            "1h": timedelta(hours=1),
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }
        datetime.utcnow() - time_windows.get(timeframe, timedelta(hours=24))

        # Get news analysis for the country (simplified - no exchange join for now)
        # Return placeholder values since we don't have the exact schema
        news_stats = {
            'news_count': 0,
            'avg_sentiment': None,
            'positive_count': 0,
            'neutral_count': 0,
            'negative_count': 0
        }
        top_news = None
        recent_news = []

        # Get exchange count
        exchanges_query = """
            SELECT COUNT(*) as count
            FROM stock_exchanges
            WHERE country_code = $1
        """
        exchanges_count = await db.fetchval(exchanges_query, country_code)

        # Get asset count
        assets_query = """
            SELECT COUNT(*) as count
            FROM asset_locations
            WHERE country_code = $1
        """
        assets_count = await db.fetchval(assets_query, country_code)

        # Get country name from exchanges table
        country_info_query = """
            SELECT country, country_code
            FROM stock_exchanges
            WHERE country_code = $1
            LIMIT 1
        """
        country_info = await db.fetchrow(country_info_query, country_code)

        # If not found in exchanges, try assets
        if not country_info:
            country_info_query = """
                SELECT country, country_code
                FROM asset_locations
                WHERE country_code = $1
                LIMIT 1
            """
            country_info = await db.fetchrow(country_info_query, country_code)

        if not country_info:
            raise HTTPException(status_code=404, detail=f"Country {country_code} not found")

        # Format response
        response = {
            "code": country_code,
            "name": country_info['country'],
            "flag": get_flag_emoji(country_code),

            # Economic indicators (placeholder - would come from economic_indicators table)
            "gdp": None,
            "gdp_growth": None,
            "inflation": None,
            "unemployment": None,

            # News analysis
            "sentiment": float(news_stats['avg_sentiment']) if news_stats['avg_sentiment'] else None,
            "news_count": news_stats['news_count'] or 0,
            "news_breakdown": {
                "positive": news_stats['positive_count'] or 0,
                "neutral": news_stats['neutral_count'] or 0,
                "negative": news_stats['negative_count'] or 0,
            },

            # Market presence
            "exchanges_count": exchanges_count or 0,
            "assets_count": assets_count or 0,

            # Top news
            "top_news": {
                "title": top_news['title'],
                "source": top_news['source'],
                "published_at": top_news['published_at'].isoformat(),
                "sentiment": float(top_news['sentiment_score']) if top_news['sentiment_score'] is not None else 0.0,
            } if top_news else None,

            # Recent news
            "recent_news": [
                {
                    "title": row['title'],
                    "source": row['source'],
                    "sentiment": float(row['sentiment_score']) if row['sentiment_score'] is not None else 0.0,
                }
                for row in recent_news
            ] if recent_news else [],
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching country details for {country_code}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch country details") from e


def get_flag_emoji(country_code: str) -> str:
    """Convert ISO country code to flag emoji."""
    if not country_code or len(country_code) != 2:
        return "🌍"

    # Convert to regional indicator symbols
    return "".join(chr(127397 + ord(char)) for char in country_code.upper())
