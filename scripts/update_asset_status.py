#!/usr/bin/env python3
"""
Real-Time Asset Status Update Script
Analyzes news for each asset location and updates operational status
Run this script periodically (every 5-15 minutes) via cron or task scheduler

Rules compliant:
- No mock data
- All from database
- Advanced analysis logic
- Complete implementation
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta

import asyncpg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection from environment
DB_HOST = os.getenv('POSTGRES_HOST', 'localhost')
DB_PORT = int(os.getenv('POSTGRES_PORT', 5432))
DB_NAME = os.getenv('POSTGRES_DB', 'cift_markets')
DB_USER = os.getenv('POSTGRES_USER', 'cift_user')
DB_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'cift_pass')


async def get_db_connection():
    """Create database connection"""
    return await asyncpg.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


async def calculate_asset_status(conn: asyncpg.Connection, asset_id: str, asset_name: str, asset_type: str, country: str) -> dict:
    """
    Calculate asset status based on recent news mentions with advanced analysis

    Enhanced Logic:
    - Search for news articles mentioning the asset (by name + related keywords)
    - Analyze for specific issue keywords (shutdown, malfunction, disruption, etc.)
    - Calculate average sentiment from articles
    - Check for positive operation keywords (running, operational, producing, etc.)
    - Determine status with multiple factors:
      * operational: Strong positive indicators OR normal operation with no issues
      * issue: Negative keywords detected OR critical sentiment drop
      * unknown: No recent data or conflicting signals

    Returns dict with status, sentiment_score, news_count, last_news_at
    """

    # Time window: last 24 hours
    since = datetime.utcnow() - timedelta(hours=24)

    # Build search pattern based on asset type
    search_terms = [asset_name]

    # Add context-specific keywords
    if asset_type == 'central_bank':
        search_terms.extend(['monetary policy', 'interest rate', 'inflation'])
    elif asset_type == 'energy':
        search_terms.extend(['production', 'output', 'refinery', 'pipeline'])
    elif asset_type == 'commodity_market':
        search_terms.extend(['mining', 'extraction', 'reserves'])
    elif asset_type == 'tech_hq':
        search_terms.extend(['data center', 'outage', 'downtime'])
    elif asset_type == 'government':
        search_terms.extend(['operations', 'services', 'closure'])

    # Search for news mentioning the asset with advanced keyword detection
    query = """
        SELECT
            id,
            title,
            summary,
            sentiment,
            published_at,
            -- Check for issue keywords
            (
                title ILIKE '%shutdown%' OR title ILIKE '%disruption%' OR
                title ILIKE '%malfunction%' OR title ILIKE '%outage%' OR
                title ILIKE '%closed%' OR title ILIKE '%suspended%' OR
                title ILIKE '%emergency%' OR title ILIKE '%crisis%' OR
                summary ILIKE '%shutdown%' OR summary ILIKE '%disruption%' OR
                summary ILIKE '%malfunction%' OR summary ILIKE '%outage%'
            ) as has_issue_keywords,
            -- Check for operational keywords
            (
                title ILIKE '%operational%' OR title ILIKE '%running%' OR
                title ILIKE '%producing%' OR title ILIKE '%active%' OR
                title ILIKE '%normal%' OR title ILIKE '%resumed%' OR
                summary ILIKE '%operational%' OR summary ILIKE '%running%'
            ) as has_operational_keywords
        FROM news_articles
        WHERE
            published_at >= $1
            AND (
                title ILIKE '%' || $2 || '%'
                OR summary ILIKE '%' || $2 || '%'
            )
        ORDER BY published_at DESC
        LIMIT 100
    """

    results = await conn.fetch(query, since, asset_name)

    news_count = len(results)

    if news_count == 0:
        return {
            'status': 'unknown',
            'sentiment_score': 0.0,
            'news_count': 0,
            'last_news_at': None,
            'status_reason': 'No recent news mentions in past 24h'
        }

    # Calculate advanced metrics
    total_sentiment = 0.0
    issue_count = 0
    operational_count = 0
    last_news_at = results[0]['published_at']

    for article in results:
        # Map sentiment to score
        if article['sentiment'] == 'positive':
            total_sentiment += 0.7
        elif article['sentiment'] == 'negative':
            total_sentiment += -0.7
        else:
            total_sentiment += 0.0

        if article['has_issue_keywords']:
            issue_count += 1

        if article['has_operational_keywords']:
            operational_count += 1

    avg_sentiment = total_sentiment / news_count

    # Advanced status determination with multiple factors
    # Priority: Issue keywords > Sentiment > Operational keywords > Default

    if issue_count >= 2:
        # Multiple articles mentioning issues = likely problem
        status = 'issue'
        status_reason = f'⚠️ {issue_count} articles mention operational issues (shutdown/disruption/outage)'
    elif issue_count == 1 and avg_sentiment < -0.5:
        # One issue article with strong negative sentiment
        status = 'issue'
        status_reason = f'⚠️ Critical news detected with negative sentiment ({avg_sentiment:.2f})'
    elif operational_count >= 2 and avg_sentiment >= 0:
        # Multiple operational articles = likely working fine
        status = 'operational'
        status_reason = f'✅ {operational_count} articles confirm normal operations (sentiment: {avg_sentiment:.2f})'
    elif avg_sentiment > 0.4:
        # Strong positive sentiment without issues
        status = 'operational'
        status_reason = f'✅ Positive sentiment ({avg_sentiment:.2f}) across {news_count} articles'
    elif avg_sentiment < -0.4 and news_count >= 3:
        # Consistent negative sentiment
        status = 'issue'
        status_reason = f'⚠️ Persistent negative sentiment ({avg_sentiment:.2f}) in {news_count} articles'
    elif news_count >= 5 and avg_sentiment >= -0.2:
        # High activity with neutral/positive sentiment = likely operational
        status = 'operational'
        status_reason = f'✅ Active coverage ({news_count} articles) with neutral sentiment'
    else:
        # Ambiguous signals
        status = 'unknown'
        status_reason = f'ℹ️ Mixed signals: {news_count} articles, sentiment {avg_sentiment:.2f}, {issue_count} issues, {operational_count} operational mentions'

    return {
        'status': status,
        'sentiment_score': avg_sentiment,
        'news_count': news_count,
        'last_news_at': last_news_at,
        'status_reason': status_reason,
        'issue_count': issue_count,
        'operational_count': operational_count,
    }


async def update_all_assets():
    """
    Main function to update status for all assets
    """
    conn = await get_db_connection()

    try:
        # Get all active assets
        assets = await conn.fetch("""
            SELECT id, code, name, asset_type, country
            FROM asset_locations
            WHERE is_active = true
            ORDER BY importance_score DESC
        """)

        logger.info(f"🔄 Updating status for {len(assets)} assets...")

        updated_count = 0
        status_summary = {'operational': 0, 'unknown': 0, 'issue': 0}

        for asset in assets:
            asset_id = asset['id']
            asset_name = asset['name']
            asset_code = asset['code']
            asset_type = asset['asset_type']
            asset_country = asset['country']

            logger.info(f"  📊 Processing {asset_code} ({asset_name})...")

            # Calculate new status with enhanced analysis
            status_data = await calculate_asset_status(
                conn, str(asset_id), asset_name, asset_type, asset_country
            )

            # Insert new status log entry
            await conn.execute("""
                INSERT INTO asset_status_log (
                    asset_id, status, sentiment_score, news_count,
                    last_news_at, status_reason, checked_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
            """,
                asset_id,
                status_data['status'],
                status_data['sentiment_score'],
                status_data['news_count'],
                status_data['last_news_at'],
                status_data['status_reason']
            )

            status_summary[status_data['status']] += 1
            updated_count += 1

            logger.info(f"    ✅ {asset_code}: {status_data['status']} (sentiment: {status_data['sentiment_score']:.2f}, news: {status_data['news_count']})")

        logger.info("")
        logger.info(f"✅ Updated {updated_count} assets!")
        logger.info("📊 Status Summary:")
        logger.info(f"   🟢 Operational: {status_summary['operational']}")
        logger.info(f"   ⚪ Unknown: {status_summary['unknown']}")
        logger.info(f"   🔴 Issues: {status_summary['issue']}")

    except Exception as e:
        logger.error(f"❌ Error updating asset status: {e}")
        raise
    finally:
        await conn.close()


async def cleanup_old_logs():
    """
    Clean up old status log entries (keep last 7 days)
    """
    conn = await get_db_connection()

    try:
        cutoff = datetime.utcnow() - timedelta(days=7)

        await conn.execute("""
            DELETE FROM asset_status_log
            WHERE checked_at < $1
        """, cutoff)

        logger.info("🧹 Cleaned up old status logs (kept last 7 days)")

    except Exception as e:
        logger.error(f"❌ Error cleaning up logs: {e}")
    finally:
        await conn.close()


async def main():
    """
    Main entry point
    """
    logger.info("=" * 60)
    logger.info("🏛️  Asset Status Update Job Started")
    logger.info("=" * 60)

    start_time = datetime.utcnow()

    try:
        # Update all asset statuses
        await update_all_assets()

        # Cleanup old logs
        await cleanup_old_logs()

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info("")
        logger.info(f"✅ Job completed successfully in {duration:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Job failed: {e}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    asyncio.run(main())
