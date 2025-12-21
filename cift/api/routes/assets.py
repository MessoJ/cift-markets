"""
Asset Locations API Routes
Provides endpoints for major market-moving locations (central banks, commodities, tech HQs, etc.)
"""

from datetime import datetime, timedelta
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/assets", tags=["assets"])


def get_flag_emoji(country_code: str) -> str:
    """Convert country code to flag emoji."""
    if not country_code or len(country_code) != 2:
        return "🌍"
    return "".join(chr(127397 + ord(c)) for c in country_code.upper())


@router.get("/")
async def get_asset_locations(
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    asset_type: str | None = Query(
        None,
        regex="^(central_bank|commodity_market|government|tech_hq|energy|all)$"
    ),
    status: str | None = Query(None, regex="^(operational|unknown|issue|all)$"),
    min_importance: int = Query(0, ge=0, le=100),
    db: asyncpg.Connection = Depends(get_postgres_pool),
):
    """
    Get all asset locations with current status and news mentions.

    Query params:
    - timeframe: 1h, 24h, 7d, 30d - time window for news analysis
    - asset_type: Filter by type (or 'all' for no filter)
    - status: operational, unknown, issue (or 'all' for no filter)
    - min_importance: Minimum importance score (0-100)
    """

    time_windows = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    start_time = datetime.utcnow() - time_windows[timeframe]

    # Build dynamic query based on filters
    type_filter = "" if not asset_type or asset_type == "all" else "AND al.asset_type = $4"
    status_filter = "" if not status or status == "all" else "AND asl.status = $5"

    query = f"""
        WITH latest_status AS (
            SELECT DISTINCT ON (asset_id)
                asset_id,
                status,
                sentiment_score,
                news_count,
                last_news_at,
                checked_at
            FROM asset_status_log
            ORDER BY asset_id, checked_at DESC
        ),
        asset_news AS (
            SELECT
                anm.asset_id,
                COUNT(DISTINCT anm.article_id) as article_count,
                COALESCE(
                    AVG(CASE
                        WHEN a.sentiment = 'positive' THEN 0.7
                        WHEN a.sentiment = 'negative' THEN -0.7
                        ELSE 0
                    END),
                    0
                ) as avg_sentiment,
                MAX(a.published_at) as last_article_at,
                ARRAY_AGG(DISTINCT a.category) FILTER (WHERE a.category IS NOT NULL) as categories,
                json_agg(
                    json_build_object(
                        'id', a.id,
                        'title', a.title,
                        'summary', a.summary,
                        'published_at', a.published_at,
                        'sentiment', a.sentiment,
                        'category', a.category
                    ) ORDER BY a.published_at DESC
                ) FILTER (WHERE a.id IS NOT NULL) as latest_articles
            FROM asset_news_mentions anm
            JOIN news_articles a ON a.id = anm.article_id
            WHERE a.published_at >= $1
            GROUP BY anm.asset_id
        )
        SELECT
            al.id,
            al.code,
            al.name,
            al.asset_type,
            al.country,
            al.country_code,
            al.city,
            al.lat,
            al.lng,
            al.timezone,
            al.description,
            al.importance_score,
            al.website,
            al.icon_url,
            COALESCE(ls.status, 'unknown') as current_status,
            COALESCE(an.avg_sentiment, 0) as sentiment_score,
            COALESCE(an.article_count, 0) as news_count,
            COALESCE(an.last_article_at, NULL) as last_news_at,
            COALESCE(an.categories, ARRAY[]::text[]) as categories,
            COALESCE(an.latest_articles, '[]'::json) as latest_articles
        FROM asset_locations al
        LEFT JOIN latest_status ls ON al.id = ls.asset_id
        LEFT JOIN asset_news an ON al.id = an.asset_id
        WHERE al.is_active = true
            AND al.importance_score >= $2
            {type_filter}
            {status_filter}
        ORDER BY al.importance_score DESC, al.name
    """

    try:
        # Build params list dynamically
        params = [start_time, min_importance]
        if asset_type and asset_type != "all":
            params.append(asset_type)
        if status and status != "all":
            params.append(status)

        rows = await db.fetch(query, *params)

        assets = []
        status_summary = {"operational": 0, "unknown": 0, "issue": 0}

        for row in rows:
            flag = get_flag_emoji(row['country_code'])

            # Limit latest articles to top 5
            latest = row['latest_articles'][:5] if row['latest_articles'] else []

            # Determine current status if not set
            current_status = row['current_status']
            if current_status == 'unknown' and row['news_count'] > 0:
                # Auto-determine status from sentiment
                if row['sentiment_score'] > 0.3:
                    current_status = 'operational'
                elif row['sentiment_score'] < -0.3:
                    current_status = 'issue'

            status_summary[current_status] = status_summary.get(current_status, 0) + 1

            asset_data = {
                "id": str(row['id']),
                "code": row['code'],
                "name": row['name'],
                "asset_type": row['asset_type'],
                "country": row['country'],
                "country_code": row['country_code'],
                "city": row['city'],
                "flag": flag,
                "lat": float(row['lat']),
                "lng": float(row['lng']),
                "timezone": row['timezone'],
                "description": row['description'],
                "importance_score": row['importance_score'],
                "website": row['website'],
                "icon_url": row['icon_url'],
                "current_status": current_status,
                "sentiment_score": float(row['sentiment_score']),
                "news_count": row['news_count'],
                "last_news_at": row['last_news_at'].isoformat() if row['last_news_at'] else None,
                "categories": row['categories'] or [],
                "latest_articles": latest,
            }

            assets.append(asset_data)

        return {
            "assets": assets,
            "total_count": len(assets),
            "status_summary": status_summary,
            "timeframe": timeframe,
            "last_updated": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching asset locations: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch asset data")


@router.get("/{asset_id}")
async def get_asset_detail(
    asset_id: UUID,
    timeframe: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    db: asyncpg.Connection = Depends(get_postgres_pool),
):
    """Get detailed information for a specific asset location."""

    time_windows = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
    }
    start_time = datetime.utcnow() - time_windows[timeframe]

    query = """
        WITH latest_status AS (
            SELECT DISTINCT ON (asset_id)
                status,
                sentiment_score,
                news_count,
                last_news_at,
                checked_at,
                status_reason
            FROM asset_status_log
            WHERE asset_id = $1
            ORDER BY asset_id, checked_at DESC
            LIMIT 1
        ),
        asset_articles AS (
            SELECT
                a.id,
                a.title,
                a.summary,
                a.published_at,
                a.sentiment,
                a.category,
                a.url,
                anm.relevance_score
            FROM asset_news_mentions anm
            JOIN news_articles a ON a.id = anm.article_id
            WHERE anm.asset_id = $1 AND a.published_at >= $2
            ORDER BY a.published_at DESC
            LIMIT 20
        )
        SELECT
            al.id,
            al.code,
            al.name,
            al.asset_type,
            al.country,
            al.country_code,
            al.city,
            al.lat,
            al.lng,
            al.timezone,
            al.description,
            al.importance_score,
            al.website,
            al.icon_url,
            COALESCE(ls.status, 'unknown') as current_status,
            ls.sentiment_score,
            ls.news_count,
            ls.last_news_at,
            ls.checked_at as status_checked_at,
            ls.status_reason,
            (SELECT json_agg(row_to_json(asset_articles.*)) FROM asset_articles) as articles
        FROM asset_locations al
        LEFT JOIN latest_status ls ON TRUE
        WHERE al.id = $1 AND al.is_active = true
    """

    try:
        row = await db.fetchrow(query, asset_id, start_time)

        if not row:
            raise HTTPException(status_code=404, detail="Asset not found")

        flag = get_flag_emoji(row['country_code'])

        return {
            "id": str(row['id']),
            "code": row['code'],
            "name": row['name'],
            "asset_type": row['asset_type'],
            "country": row['country'],
            "country_code": row['country_code'],
            "city": row['city'],
            "flag": flag,
            "lat": float(row['lat']),
            "lng": float(row['lng']),
            "timezone": row['timezone'],
            "description": row['description'],
            "importance_score": row['importance_score'],
            "website": row['website'],
            "icon_url": row['icon_url'],
            "current_status": row['current_status'],
            "sentiment_score": float(row['sentiment_score']) if row['sentiment_score'] else 0.0,
            "news_count": row['news_count'] or 0,
            "last_news_at": row['last_news_at'].isoformat() if row['last_news_at'] else None,
            "status_checked_at": row['status_checked_at'].isoformat() if row['status_checked_at'] else None,
            "status_reason": row['status_reason'],
            "articles": row['articles'] or [],
            "timeframe": timeframe,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching asset detail: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch asset detail")


@router.get("/types/summary")
async def get_asset_types_summary(
    db: asyncpg.Connection = Depends(get_postgres_pool),
):
    """Get summary statistics for each asset type."""

    query = """
        SELECT
            asset_type,
            COUNT(*) as total_count,
            ROUND(AVG(importance_score), 1) as avg_importance,
            MIN(importance_score) as min_importance,
            MAX(importance_score) as max_importance
        FROM asset_locations
        WHERE is_active = true
        GROUP BY asset_type
        ORDER BY COUNT(*) DESC
    """

    try:
        rows = await db.fetch(query)

        return {
            "types": [
                {
                    "asset_type": row['asset_type'],
                    "total_count": row['total_count'],
                    "avg_importance": float(row['avg_importance']),
                    "min_importance": row['min_importance'],
                    "max_importance": row['max_importance'],
                }
                for row in rows
            ]
        }

    except Exception as e:
        logger.error(f"Error fetching asset types summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch types summary")
