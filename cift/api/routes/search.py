"""
CIFT Markets - Global Search API Routes

Unified search across all platform entities:
- Stock symbols & market data
- Orders (pending, filled, cancelled)
- Positions (open positions)
- Watchlists
- News articles
- Assets & locations

NO MOCK DATA - All results from PostgreSQL database.
"""

from datetime import datetime
from uuid import UUID

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/search", tags=["Search"])


# ============================================================================
# MODELS
# ============================================================================

class SearchResult(BaseModel):
    """Single search result item"""
    id: str
    type: str  # symbol, order, position, watchlist, news, asset
    title: str
    subtitle: str | None = None
    description: str | None = None
    link: str
    icon: str | None = None
    metadata: dict | None = None
    relevance_score: float = 0.0


class SearchResponse(BaseModel):
    """Search response with categorized results"""
    query: str
    total_results: int
    results: list[SearchResult]
    categories: dict  # Count by type
    took_ms: float


# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@router.get("", response_model=SearchResponse)
async def global_search(
    q: str = Query(..., min_length=1, max_length=100, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    types: str | None = Query(None, description="Comma-separated types to search: symbol,order,position,watchlist,news,asset"),
    user_id: UUID = Depends(get_current_user_id),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Global search across all platform entities.

    Searches:
    - Stock symbols (market data)
    - User orders
    - User positions
    - User watchlists
    - News articles
    - Asset locations

    Results are ranked by relevance and limited to requested count.
    """
    start_time = datetime.now()

    # Parse search types filter
    search_types = types.split(',') if types else ['symbol', 'order', 'position', 'watchlist', 'news', 'asset']
    search_types = [t.strip() for t in search_types]

    query_upper = q.upper()
    query_lower = q.lower()
    query_pattern = f"%{query_lower}%"

    results = []
    categories = {}

    try:
        async with pool.acquire() as conn:
            # ================================================================
            # 1. SEARCH SYMBOLS (Market Data)
            # ================================================================
            if 'symbol' in search_types:
                try:
                    # Search in market_data_cache and join with symbols for metadata
                    symbol_rows = await conn.fetch(
                        """
                        SELECT 
                            m.symbol,
                            m.price,
                            m.change,
                            m.change_pct,
                            m.volume,
                            s.name,
                            s.asset_type,
                            s.exchange
                        FROM market_data_cache m
                        LEFT JOIN symbols s ON m.symbol = s.symbol
                        WHERE m.symbol ILIKE $1 OR s.name ILIKE $1
                        ORDER BY 
                            CASE WHEN m.symbol = $2 THEN 0 ELSE 1 END,
                            m.volume DESC NULLS LAST
                        LIMIT $3
                        """,
                        query_pattern,
                        query_upper,
                        limit
                    )

                    for row in symbol_rows:
                        change_indicator = "🟢" if row['change'] and row['change'] > 0 else "🔴" if row['change'] and row['change'] < 0 else "⚪"
                        name = row['name'] or row['symbol']
                        results.append(SearchResult(
                            id=row['symbol'],
                            type='symbol',
                            title=row['symbol'],
                            subtitle=name,
                            description=f"${row['price']:.2f} • {change_indicator} {row['change_pct']:.2f}%" if row['price'] else "Price N/A",
                            link=f"/trading?symbol={row['symbol']}",
                            icon="📈",
                            metadata={
                                'price': float(row['price']) if row['price'] else None,
                                'change': float(row['change']) if row['change'] else None,
                                'change_pct': float(row['change_pct']) if row['change_pct'] else None,
                                'volume': row['volume'],
                                'name': name,
                                'type': row['asset_type'] or 'stock',
                                'exchange': row['exchange']
                            },
                            relevance_score=100.0 if row['symbol'] == query_upper else 80.0
                        ))

                    categories['symbol'] = len(symbol_rows)
                except Exception as e:
                    logger.warning(f"Symbol search failed: {e}")
                    categories['symbol'] = 0

            # ================================================================
            # 2. SEARCH ORDERS
            # ================================================================
            if 'order' in search_types:
                try:
                    order_rows = await conn.fetch(
                        """
                        SELECT
                            id, symbol, side, order_type, quantity,
                            filled_quantity, limit_price, status,
                            created_at
                        FROM orders
                        WHERE user_id = $1
                        AND (
                            symbol ILIKE $2
                            OR id::text ILIKE $2
                            OR status ILIKE $2
                        )
                        ORDER BY created_at DESC
                        LIMIT $3
                        """,
                        user_id,
                        query_pattern,
                        limit
                    )

                    for row in order_rows:
                        status_emoji = {"pending": "⏳", "filled": "✅", "cancelled": "❌", "rejected": "🚫"}.get(row['status'], "📋")
                        results.append(SearchResult(
                            id=str(row['id']),
                            type='order',
                            title=f"{row['symbol']} {row['side'].upper()}",
                            subtitle=f"{row['quantity']} @ ${row['limit_price']:.2f}" if row['limit_price'] else f"{row['quantity']} shares",
                            description=f"{status_emoji} {row['status'].capitalize()} • {row['order_type']}",
                            link=f"/orders?id={row['id']}",
                            icon="📝",
                            metadata={
                                'symbol': row['symbol'],
                                'side': row['side'],
                                'quantity': float(row['quantity']),
                                'status': row['status']
                            },
                            relevance_score=70.0
                        ))

                    categories['order'] = len(order_rows)
                except Exception as e:
                    logger.warning(f"Order search failed: {e}")
                    categories['order'] = 0

            # ================================================================
            # 3. SEARCH POSITIONS
            # ================================================================
            if 'position' in search_types:
                try:
                    position_rows = await conn.fetch(
                        """
                        SELECT
                            id, symbol, quantity, avg_cost,
                            current_price, unrealized_pnl, unrealized_pnl_pct,
                            market_value
                        FROM positions
                        WHERE user_id = $1
                        AND quantity != 0
                        AND symbol ILIKE $2
                        ORDER BY ABS(market_value) DESC
                        LIMIT $3
                        """,
                        user_id,
                        query_pattern,
                        limit
                    )

                    for row in position_rows:
                        pnl_emoji = "🟢" if row['unrealized_pnl'] and row['unrealized_pnl'] > 0 else "🔴" if row['unrealized_pnl'] and row['unrealized_pnl'] < 0 else "⚪"
                        results.append(SearchResult(
                            id=str(row['id']),
                            type='position',
                            title=row['symbol'],
                            subtitle=f"{row['quantity']} shares @ ${row['current_price']:.2f}" if row['current_price'] else f"{row['quantity']} shares",
                            description=f"{pnl_emoji} ${row['unrealized_pnl']:.2f} ({row['unrealized_pnl_pct']:.2f}%)" if row['unrealized_pnl'] and row['unrealized_pnl_pct'] else None,
                            link=f"/portfolio?symbol={row['symbol']}",
                            icon="💼",
                            metadata={
                                'symbol': row['symbol'],
                                'quantity': float(row['quantity']),
                                'avg_cost': float(row['avg_cost']) if row['avg_cost'] else None,
                                'unrealized_pnl': float(row['unrealized_pnl']) if row['unrealized_pnl'] else None
                            },
                            relevance_score=90.0
                        ))

                    categories['position'] = len(position_rows)
                except Exception as e:
                    logger.warning(f"Position search failed: {e}")
                    categories['position'] = 0

            # ================================================================
            # 4. SEARCH WATCHLISTS
            # ================================================================
            if 'watchlist' in search_types:
                try:
                    watchlist_rows = await conn.fetch(
                        """
                        SELECT
                            id, name, description, symbols, is_default,
                            created_at
                        FROM watchlists
                        WHERE user_id = $1
                        AND (
                            name ILIKE $2
                            OR description ILIKE $2
                            OR $3 = ANY(symbols)
                        )
                        ORDER BY is_default DESC, created_at DESC
                        LIMIT $4
                        """,
                        user_id,
                        query_pattern,
                        query_upper,
                        limit
                    )

                    for row in watchlist_rows:
                        results.append(SearchResult(
                            id=str(row['id']),
                            type='watchlist',
                            title=row['name'],
                            subtitle=f"{len(row['symbols'])} symbols" if row['symbols'] else "Empty list",
                            description=row['description'] or "No description",
                            link=f"/watchlists?id={row['id']}",
                            icon="⭐" if row['is_default'] else "📋",
                            metadata={
                                'symbols': row['symbols'],
                                'is_default': row['is_default']
                            },
                            relevance_score=60.0
                        ))

                    categories['watchlist'] = len(watchlist_rows)
                except Exception as e:
                    logger.warning(f"Watchlist search failed: {e}")
                    categories['watchlist'] = 0

            # ================================================================
            # 5. SEARCH NEWS ARTICLES
            # ================================================================
            if 'news' in search_types:
                try:
                    news_rows = await conn.fetch(
                        """
                        SELECT
                            id, title, summary, categories, sentiment,
                            published_at, url
                        FROM news_articles
                        WHERE
                            title ILIKE $1
                            OR summary ILIKE $1
                            OR categories::text ILIKE $1
                        ORDER BY published_at DESC
                        LIMIT $2
                        """,
                        query_pattern,
                        limit
                    )

                    for row in news_rows:
                        sentiment_emoji = {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(row['sentiment'], "📰")
                        # Extract first category from JSONB array
                        cats = json.loads(row["categories"]) if isinstance(row["categories"], str) else row["categories"]
                        category = cats[0] if cats and len(cats) > 0 else "News"
                        
                        results.append(SearchResult(
                            id=str(row['id']),
                            type='news',
                            title=row['title'],
                            subtitle=category,
                            description=row['summary'][:100] + "..." if row['summary'] and len(row['summary']) > 100 else row['summary'],
                            link=f"/news/{row['id']}",
                            icon=sentiment_emoji,
                            metadata={
                                'category': category,
                                'sentiment': row['sentiment'],
                                'published_at': row['published_at'].isoformat() if row['published_at'] else None
                            },
                            relevance_score=50.0
                        ))

                    categories['news'] = len(news_rows)
                except Exception as e:
                    logger.warning(f"News search failed: {e}")
                    categories['news'] = 0

            # ================================================================
            # 6. SEARCH ASSET LOCATIONS
            # ================================================================
            if 'asset' in search_types:
                try:
                    asset_rows = await conn.fetch(
                        """
                        SELECT
                            id, code, name, asset_type, country,
                            city, importance_score
                        FROM asset_locations
                        WHERE is_active = true
                        AND (
                            code ILIKE $1
                            OR name ILIKE $1
                            OR city ILIKE $1
                            OR country ILIKE $1
                        )
                        ORDER BY importance_score DESC
                        LIMIT $2
                        """,
                        query_pattern,
                        limit
                    )

                    for row in asset_rows:
                        type_emoji = {
                            'central_bank': '🏦',
                            'commodity_market': '📦',
                            'government': '🏛️',
                            'tech_hq': '💻',
                            'energy': '⚡'
                        }.get(row['asset_type'], '🌍')

                        results.append(SearchResult(
                            id=str(row['id']),
                            type='asset',
                            title=f"{row['name']} ({row['code']})",
                            subtitle=f"{row['city']}, {row['country']}",
                            description=f"{row['asset_type'].replace('_', ' ').title()}",
                            link=f"/globe?asset={row['id']}",
                            icon=type_emoji,
                            metadata={
                                'code': row['code'],
                                'asset_type': row['asset_type'],
                                'importance_score': row['importance_score']
                            },
                            relevance_score=float(row['importance_score']) / 100 * 50
                        ))

                    categories['asset'] = len(asset_rows)
                except Exception as e:
                    logger.warning(f"Asset search failed: {e}")
                    categories['asset'] = 0

    except Exception as e:
        logger.error(f"Global search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        ) from e

    # Sort results by relevance score
    results.sort(key=lambda x: x.relevance_score, reverse=True)

    # Limit total results
    results = results[:limit]

    # Calculate elapsed time
    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000

    logger.info(f"Search '{q}' returned {len(results)} results in {elapsed_ms:.2f}ms")

    return SearchResponse(
        query=q,
        total_results=len(results),
        results=results,
        categories=categories,
        took_ms=round(elapsed_ms, 2)
    )


@router.get("/suggestions", response_model=list[str])
async def get_search_suggestions(
    q: str = Query(..., min_length=1, max_length=50),
    limit: int = Query(10, ge=1, le=20),
    pool: asyncpg.Pool = Depends(get_postgres_pool),
):
    """
    Get search suggestions/autocomplete.

    Returns popular symbols and terms matching the query.
    Super fast for typeahead/autocomplete.
    """
    query_pattern = f"{q.upper()}%"

    try:
        async with pool.acquire() as conn:
            # Get symbol suggestions
            rows = await conn.fetch(
                """
                SELECT DISTINCT symbol
                FROM market_data
                WHERE symbol LIKE $1
                ORDER BY symbol
                LIMIT $2
                """,
                query_pattern,
                limit
            )

            suggestions = [row['symbol'] for row in rows]

            return suggestions

    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return []
