"""
NEWS & MARKET MOVERS API ROUTES
Handles market news, top gainers/losers, and economic calendar.
All data is fetched from database - NO MOCK DATA.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool, get_questdb_pool
from cift.core.logging import logger
from cift.services.news_service import get_news_service

router = APIRouter(prefix="/news", tags=["news"])


# ============================================================================
# MODELS
# ============================================================================

class NewsArticle(BaseModel):
    """News article model"""
    id: str
    title: str
    summary: str
    source: str
    url: str
    published_at: datetime
    category: str
    sentiment: Optional[str] = None  # 'positive', 'negative', 'neutral'
    symbols: List[str] = []
    image_url: Optional[str] = None


class MarketMover(BaseModel):
    """Market mover model"""
    symbol: str
    name: str
    price: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    market_cap: Optional[Decimal] = None


class EconomicEvent(BaseModel):
    """Economic calendar event model"""
    id: str
    title: str
    country: str
    date: datetime
    impact: str  # 'high', 'medium', 'low'
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    currency: str


# ============================================================================
# ENDPOINTS - NEWS
# ============================================================================

@router.get("/articles")
async def get_news(
    category: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get market news articles from database"""
    
    pool = await get_postgres_pool()
    
    # Build query based on filters
    query = """
        SELECT 
            id::text,
            title,
            summary,
            source,
            url,
            published_at,
            category,
            sentiment,
            symbols,
            image_url
        FROM news_articles
        WHERE published_at >= $1
    """
    params = [datetime.utcnow() - timedelta(days=7)]  # Last 7 days
    param_count = 2
    
    if category and category != 'all':
        query += f" AND category = ${param_count}"
        params.append(category)
        param_count += 1
    
    if symbol:
        query += f" AND ${param_count} = ANY(symbols)"
        params.append(symbol.upper())
        param_count += 1
    
    query += f" ORDER BY published_at DESC LIMIT ${param_count} OFFSET ${param_count + 1}"
    params.extend([limit, offset])
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        articles = []
        for row in rows:
            # Filter out known CORS-blocked domains
            image_url = row['image_url']
            if image_url and any(domain in image_url for domain in ['cryptoslate.com', 'medium.com']):
                image_url = None
            
            articles.append({
                "id": row['id'],
                "title": row['title'],
                "summary": row['summary'],
                "source": row['source'],
                "url": row['url'],
                "published_at": row['published_at'],
                "category": row['category'] or "general",
                "sentiment": row['sentiment'],
                "symbols": row['symbols'] or [],
                "image_url": image_url,
            })
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM news_articles WHERE published_at >= $1"
        count_params = [datetime.utcnow() - timedelta(days=7)]
        
        if category and category != 'all':
            count_query += f" AND category = ${len(count_params) + 1}"
            count_params.append(category)
        
        if symbol:
            count_query += f" AND ${len(count_params) + 1} = ANY(symbols)"
            count_params.append(symbol.upper())
        
        total = await conn.fetchval(count_query, *count_params)
        
        return {
            "articles": articles,
            "total": total,
            "limit": limit,
            "offset": offset,
        }


@router.get("/articles/{article_id}")
async def get_article(
    article_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get single news article from database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                summary,
                content,
                source,
                url,
                published_at,
                category,
                sentiment,
                symbols,
                image_url,
                author
            FROM news_articles
            WHERE id = $1::uuid
            """,
            article_id,
        )
        
        if not row:
            raise HTTPException(status_code=404, detail="Article not found")
        
        return {
            "id": row['id'],
            "title": row['title'],
            "summary": row['summary'],
            "content": row['content'],
            "source": row['source'],
            "url": row['url'],
            "published_at": row['published_at'],
            "category": row['category'],
            "sentiment": row['sentiment'],
            "symbols": row['symbols'] or [],
            "image_url": row['image_url'],
            "author": row['author'],
        }


# ============================================================================
# ENDPOINTS - MARKET MOVERS
# ============================================================================

@router.get("/movers/{mover_type}")
async def get_market_movers(
    mover_type: str,
    limit: int = 20,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get market movers (gainers, losers, most active) from database"""
    if mover_type not in ['gainers', 'losers', 'active']:
        raise HTTPException(status_code=400, detail="Invalid mover type")
    
    pool = await get_questdb_pool()
    
    # First, get the most recent data timestamp from QuestDB
    async with pool.acquire() as conn:
        latest_row = await conn.fetchrow("SELECT max(timestamp) as latest FROM ticks")
        if not latest_row or not latest_row['latest']:
            logger.warning(f"No tick data available in QuestDB")
            return []
        
        latest_timestamp = latest_row['latest']
        # Use the day of the latest available data
        data_start = latest_timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Calculate movers from available tick data (QuestDB compatible)
    if mover_type == 'gainers':
        order_clause = "ORDER BY change_percent DESC"
    elif mover_type == 'losers':
        order_clause = "ORDER BY change_percent ASC"
    else:  # active
        order_clause = "ORDER BY total_volume DESC"
    
    # QuestDB doesn't support HAVING, use subquery
    query = f"""
        SELECT * FROM (
            SELECT 
                symbol,
                first(price) as open_price,
                last(price) as current_price,
                sum(volume) as total_volume,
                count(*) as tick_count,
                ((last(price) - first(price)) / first(price) * 100) as change_percent
            FROM ticks
            WHERE timestamp >= $1
            GROUP BY symbol
        )
        WHERE tick_count > 10
        {order_clause}
        LIMIT $2
    """
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, data_start, limit)
        
        movers = []
        pg_pool = await get_postgres_pool()
        
        for row in rows:
            symbol = row['symbol']
            current_price = float(row['current_price'])
            open_price = float(row['open_price'])
            change = current_price - open_price
            change_percent = float(row['change_percent'])
            
            # Get company name from postgres
            async with pg_pool.acquire() as pg_conn:
                name_row = await pg_conn.fetchrow(
                    "SELECT name, market_cap FROM symbols WHERE symbol = $1",
                    symbol,
                )
                name = name_row['name'] if name_row else symbol
                market_cap = float(name_row['market_cap']) if (name_row and name_row['market_cap']) else None
            
            movers.append(
                MarketMover(
                    symbol=symbol,
                    name=name,
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=int(row['total_volume']),
                    market_cap=market_cap,
                )
            )
        
        # If no data, log warning and return empty
        if not movers:
            logger.warning(f"No tick data found for {mover_type}. Run 'python scripts/populate_today.py' to generate data.")
        
        return movers


@router.get("/market-summary")
async def get_market_summary(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get market summary (indices) from database"""
    pool = await get_questdb_pool()
    
    indices = ['SPY', 'QQQ', 'DIA', 'IWM']  # S&P 500, NASDAQ, Dow, Russell 2000
    
    # First, get the most recent data timestamp
    async with pool.acquire() as conn:
        latest_row = await conn.fetchrow("SELECT max(timestamp) as latest FROM ticks")
        if not latest_row or not latest_row['latest']:
            return []
        
        data_start = latest_row['latest'].replace(hour=0, minute=0, second=0, microsecond=0)
    
    summary = []
    
    async with pool.acquire() as conn:
        for index_symbol in indices:
            # Query from ticks table since market_quotes may not exist
            row = await conn.fetchrow(
                """
                SELECT 
                    symbol,
                    first(price) as open_price,
                    last(price) as price,
                    sum(volume) as volume,
                    ((last(price) - first(price)) / first(price) * 100) as change_percent
                FROM ticks
                WHERE symbol = $1 AND timestamp >= $2
                GROUP BY symbol
                """,
                index_symbol,
                data_start,
            )
            
            if row:
                open_price = float(row['open_price'])
                current_price = float(row['price'])
                change = current_price - open_price
                change_percent = float(row['change_percent']) if row['change_percent'] else 0
                
                # Get index name
                pg_pool = await get_postgres_pool()
                async with pg_pool.acquire() as pg_conn:
                    name_row = await pg_conn.fetchrow(
                        "SELECT name FROM symbols WHERE symbol = $1",
                        index_symbol,
                    )
                    name = name_row['name'] if name_row else index_symbol
                
                summary.append({
                    "symbol": row['symbol'],
                    "name": name,
                    "price": current_price,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": int(row['volume']) if row['volume'] else 0,
                })
    
    return summary


# ============================================================================
# ENDPOINTS - ECONOMIC CALENDAR
# ============================================================================

@router.get("/economic-calendar")
async def get_economic_calendar(
    days_ahead: int = 7,
    impact: Optional[str] = None,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get economic calendar events from database"""
    pool = await get_postgres_pool()
    
    # Show events from 2 days ago to days_ahead in the future
    start_date = datetime.utcnow() - timedelta(days=2)
    end_date = datetime.utcnow() + timedelta(days=days_ahead)
    
    query = """
        SELECT 
            id::text,
            title,
            country,
            event_date,
            impact,
            forecast,
            previous,
            actual,
            currency
        FROM economic_events
        WHERE event_date BETWEEN $1 AND $2
    """
    params = [start_date, end_date]
    
    if impact:
        query += " AND impact = $3"
        params.append(impact)
    
    query += " ORDER BY event_date ASC"
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        return [
            EconomicEvent(
                id=row['id'],
                title=row['title'],
                country=row['country'],
                date=row['event_date'],
                impact=row['impact'],
                forecast=row['forecast'],
                previous=row['previous'],
                actual=row['actual'],
                currency=row['currency'],
            )
            for row in rows
        ]


@router.get("/earnings-calendar")
async def get_earnings_calendar(
    days_ahead: int = 14,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get earnings calendar from database"""
    pool = await get_postgres_pool()
    
    start_date = datetime.utcnow().date()
    end_date = start_date + timedelta(days=days_ahead)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT 
                symbol,
                company_name,
                earnings_date,
                earnings_time,
                eps_estimate,
                eps_actual,
                revenue_estimate,
                revenue_actual
            FROM earnings_calendar
            WHERE earnings_date BETWEEN $1 AND $2
            ORDER BY earnings_date ASC, symbol ASC
            """,
            start_date,
            end_date,
        )
        
        return [
            {
                "symbol": row['symbol'],
                "company_name": row['company_name'],
                "earnings_date": row['earnings_date'].isoformat(),
                "earnings_time": row['earnings_time'],
                "eps_estimate": float(row['eps_estimate']) if row['eps_estimate'] else None,
                "eps_actual": float(row['eps_actual']) if row['eps_actual'] else None,
                "revenue_estimate": float(row['revenue_estimate']) if row['revenue_estimate'] else None,
                "revenue_actual": float(row['revenue_actual']) if row['revenue_actual'] else None,
            }
            for row in rows
        ]


# ============================================================================
# ENDPOINTS - SENTIMENT
# ============================================================================

@router.get("/sentiment/{symbol}")
async def get_symbol_sentiment(
    symbol: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get news sentiment analysis for a symbol from database"""
    pool = await get_postgres_pool()
    
    # Get sentiment from recent news
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT sentiment, COUNT(*) as count
            FROM news_articles
            WHERE $1 = ANY(symbols)
            AND published_at >= $2
            AND sentiment IS NOT NULL
            GROUP BY sentiment
            """,
            symbol.upper(),
            datetime.utcnow() - timedelta(days=7),
        )
        
        sentiment_counts = {row['sentiment']: row['count'] for row in rows}
        total = sum(sentiment_counts.values())
        
        if total == 0:
            return {
                "symbol": symbol.upper(),
                "sentiment": "neutral",
                "positive_percent": 0,
                "negative_percent": 0,
                "neutral_percent": 0,
                "total_articles": 0,
            }
        
        return {
            "symbol": symbol.upper(),
            "sentiment": max(sentiment_counts, key=sentiment_counts.get),
            "positive_percent": round((sentiment_counts.get('positive', 0) / total) * 100, 2),
            "negative_percent": round((sentiment_counts.get('negative', 0) / total) * 100, 2),
            "neutral_percent": round((sentiment_counts.get('neutral', 0) / total) * 100, 2),
            "total_articles": total,
        }


# ============================================================================
# ENDPOINTS - GLOBE DATA
# ============================================================================

@router.get("/globe-data")
async def get_globe_news_data(
    hours: int = 24,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get news data aggregated by country for globe visualization.
    Returns country-level statistics for 3D globe plotting.
    """
    pool = await get_postgres_pool()
    
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    
    async with pool.acquire() as conn:
        # Aggregate news by country
        country_data = await conn.fetch(
            """
            SELECT 
                country,
                country_code,
                region,
                latitude,
                longitude,
                COUNT(*) as article_count,
                COUNT(DISTINCT CASE WHEN sentiment = 'positive' THEN id END) as positive_count,
                COUNT(DISTINCT CASE WHEN sentiment = 'negative' THEN id END) as negative_count,
                COUNT(DISTINCT CASE WHEN sentiment = 'neutral' THEN id END) as neutral_count,
                MAX(published_at) as latest_article_time
            FROM news_articles
            WHERE published_at >= $1
                AND country_code IS NOT NULL
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
            GROUP BY country, country_code, region, latitude, longitude
            HAVING COUNT(*) > 0
            ORDER BY article_count DESC
            """,
            time_threshold
        )
        
        # Get top headlines per country (top 3)
        countries_list = []
        for row in country_data:
            country_code = row['country_code']
            
            # Get top 3 headlines for this country
            headlines = await conn.fetch(
                """
                SELECT id::text, title, source, sentiment, published_at
                FROM news_articles
                WHERE country_code = $1
                    AND published_at >= $2
                ORDER BY published_at DESC
                LIMIT 3
                """,
                country_code,
                time_threshold
            )
            
            # Calculate sentiment score (-1 to 1)
            total_articles = row['article_count']
            if total_articles > 0:
                sentiment_score = (
                    (row['positive_count'] - row['negative_count']) / total_articles
                )
            else:
                sentiment_score = 0
            
            countries_list.append({
                "code": country_code,
                "name": row['country'],
                "region": row['region'],
                "lat": float(row['latitude']),
                "lng": float(row['longitude']),
                "article_count": row['article_count'],
                "sentiment_score": round(sentiment_score, 3),
                "sentiment_breakdown": {
                    "positive": row['positive_count'],
                    "negative": row['negative_count'],
                    "neutral": row['neutral_count']
                },
                "latest_time": row['latest_article_time'].isoformat() if row['latest_article_time'] else None,
                "top_headlines": [
                    {
                        "id": h['id'],
                        "title": h['title'],
                        "source": h['source'],
                        "sentiment": h['sentiment'],
                        "published_at": h['published_at'].isoformat() if h['published_at'] else None
                    }
                    for h in headlines
                ]
            })
        
        # Get breaking news (last hour)
        breaking_news = await conn.fetch(
            """
            SELECT 
                id::text,
                title,
                source,
                country_code,
                sentiment,
                published_at
            FROM news_articles
            WHERE published_at >= $1
            ORDER BY published_at DESC
            LIMIT 10
            """,
            datetime.utcnow() - timedelta(hours=1)
        )
        
        return {
            "countries": countries_list,
            "total_countries": len(countries_list),
            "total_articles": sum(c['article_count'] for c in countries_list),
            "time_range_hours": hours,
            "breaking_news": [
                {
                    "id": article['id'],
                    "title": article['title'],
                    "source": article['source'],
                    "country_code": article['country_code'],
                    "sentiment": article['sentiment'],
                    "published_at": article['published_at'].isoformat() if article['published_at'] else None
                }
                for article in breaking_news
            ]
        }
