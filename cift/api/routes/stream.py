"""
CIFT Markets - Real-Time Streaming API Routes

Server-Sent Events (SSE) endpoints for real-time price streaming.
"""

import asyncio
from typing import AsyncGenerator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/stream", tags=["streaming"])


# ============================================================================
# SERVER-SENT EVENTS FOR REAL-TIME PRICES
# ============================================================================

async def price_stream_generator(
    symbols: list[str],
    interval: float = 1.0
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream of real-time price updates.
    
    Args:
        symbols: List of symbols to stream
        interval: Update interval in seconds
    """
    try:
        # Import the Finnhub service
        from cift.api.routes.admin import get_finnhub_service
        service = await get_finnhub_service()
        
        last_prices = {}
        
        while True:
            updates = []
            
            for symbol in symbols:
                price = service.last_prices.get(symbol.upper())
                
                # Only send if price changed
                if price is not None and price != last_prices.get(symbol):
                    last_prices[symbol] = price
                    updates.append({
                        "symbol": symbol,
                        "price": price,
                        "timestamp": asyncio.get_event_loop().time()
                    })
            
            if updates:
                # SSE format: data: {json}\n\n
                import json
                for update in updates:
                    yield f"data: {json.dumps(update)}\n\n"
            else:
                # Send heartbeat to keep connection alive
                yield f": heartbeat\n\n"
            
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        logger.info("Price stream cancelled")
        raise
    except Exception as e:
        logger.error(f"Price stream error: {e}")
        yield f"event: error\ndata: {str(e)}\n\n"


@router.get("/prices")
async def stream_prices(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    interval: float = Query(1.0, ge=0.1, le=10.0, description="Update interval in seconds"),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Stream real-time prices via Server-Sent Events (SSE).
    
    Connect from frontend:
    ```javascript
    const eventSource = new EventSource('/api/v1/stream/prices?symbols=AAPL,MSFT&interval=1');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`${data.symbol}: $${data.price}`);
    };
    ```
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="At least one symbol required")
    
    if len(symbol_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed")
    
    return StreamingResponse(
        price_stream_generator(symbol_list, interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# ============================================================================
# DATABASE POLLING STREAM (Fallback when WebSocket not available)
# ============================================================================

async def db_price_stream_generator(
    symbols: list[str],
    interval: float = 2.0
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream from database cache (polling fallback).
    """
    import json
    
    last_prices = {}
    pool = await get_postgres_pool()
    
    try:
        while True:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT symbol, price, change_pct, updated_at
                    FROM market_data_cache
                    WHERE symbol = ANY($1)
                    """,
                    symbols
                )
                
                updates = []
                for row in rows:
                    symbol = row['symbol']
                    price = float(row['price'])
                    
                    if price != last_prices.get(symbol):
                        last_prices[symbol] = price
                        updates.append({
                            "symbol": symbol,
                            "price": price,
                            "change_pct": float(row['change_pct']) if row['change_pct'] else 0,
                            "timestamp": row['updated_at'].isoformat()
                        })
                
                if updates:
                    for update in updates:
                        yield f"data: {json.dumps(update)}\n\n"
                else:
                    yield f": heartbeat\n\n"
            
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        logger.info("DB price stream cancelled")
        raise
    except Exception as e:
        logger.error(f"DB price stream error: {e}")
        yield f"event: error\ndata: {str(e)}\n\n"


@router.get("/prices-cached")
async def stream_cached_prices(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
    interval: float = Query(2.0, ge=1.0, le=30.0, description="Update interval in seconds"),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Stream prices from database cache via Server-Sent Events.
    
    Use this as fallback when WebSocket streaming is not available.
    Updates are slower but reliable.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    if not symbol_list:
        raise HTTPException(status_code=400, detail="At least one symbol required")
    
    if len(symbol_list) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 symbols allowed")
    
    return StreamingResponse(
        db_price_stream_generator(symbol_list, interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================================
# NEWS STREAM
# ============================================================================

async def news_stream_generator(
    symbols: list[str] | None = None,
    interval: float = 30.0
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream of news updates.
    """
    import json
    from datetime import datetime, timedelta
    
    pool = await get_postgres_pool()
    last_check = datetime.utcnow() - timedelta(minutes=5)
    
    try:
        while True:
            async with pool.acquire() as conn:
                if symbols:
                    rows = await conn.fetch(
                        """
                        SELECT id, title, source_name, source_url, category,
                               sentiment_score, published_at
                        FROM news_articles
                        WHERE published_at > $1
                        AND (symbols && $2 OR $2 IS NULL)
                        ORDER BY published_at DESC
                        LIMIT 10
                        """,
                        last_check,
                        symbols
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT id, title, source_name, source_url, category,
                               sentiment_score, published_at
                        FROM news_articles
                        WHERE published_at > $1
                        ORDER BY published_at DESC
                        LIMIT 10
                        """,
                        last_check
                    )
                
                if rows:
                    last_check = datetime.utcnow()
                    for row in rows:
                        article = {
                            "id": str(row['id']),
                            "title": row['title'],
                            "source": row['source_name'],
                            "url": row['source_url'],
                            "category": row['category'],
                            "sentiment": float(row['sentiment_score']) if row['sentiment_score'] else None,
                            "published_at": row['published_at'].isoformat()
                        }
                        yield f"event: news\ndata: {json.dumps(article)}\n\n"
                else:
                    yield f": heartbeat\n\n"
            
            await asyncio.sleep(interval)
            
    except asyncio.CancelledError:
        logger.info("News stream cancelled")
        raise
    except Exception as e:
        logger.error(f"News stream error: {e}")
        yield f"event: error\ndata: {str(e)}\n\n"


@router.get("/news")
async def stream_news(
    symbols: str = Query(None, description="Optional comma-separated list of symbols"),
    interval: float = Query(30.0, ge=10.0, le=300.0, description="Update interval in seconds"),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Stream news updates via Server-Sent Events.
    
    Connect from frontend:
    ```javascript
    const eventSource = new EventSource('/api/v1/stream/news?symbols=AAPL,MSFT');
    eventSource.addEventListener('news', (event) => {
        const article = JSON.parse(event.data);
        console.log(`News: ${article.title}`);
    });
    ```
    """
    symbol_list = None
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    return StreamingResponse(
        news_stream_generator(symbol_list, interval),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
