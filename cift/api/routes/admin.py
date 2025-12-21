"""
CIFT Markets - Admin API Routes

Administrative endpoints for data management and system operations.
Requires admin privileges.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# MODELS
# ============================================================================

class DataUpdateRequest(BaseModel):
    """Request to update market data."""
    symbols: list[str] | None = None
    update_quotes: bool = True
    update_news: bool = True
    update_bars: bool = False
    days: int = 5


class DataUpdateResponse(BaseModel):
    """Response from data update."""
    status: str
    quotes_updated: int = 0
    news_stored: int = 0
    bars_stored: int = 0
    message: str


# ============================================================================
# HELPER: Check admin status
# ============================================================================

async def require_admin(user_id: UUID = Depends(get_current_user_id)) -> UUID:
    """Verify user is an admin."""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT is_superuser FROM users WHERE id = $1",
            user_id
        )

        if not row or not row['is_superuser']:
            raise HTTPException(status_code=403, detail="Admin access required")

    return user_id


# ============================================================================
# ENDPOINTS - MARKET DATA
# ============================================================================

@router.post("/update-market-data")
async def update_market_data(
    request: DataUpdateRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(require_admin),
) -> DataUpdateResponse:
    """
    Trigger market data update from Polygon.io.

    Fetches real-time quotes, news, and optionally historical bars.
    """
    try:
        from cift.services.polygon_realtime_service import PolygonRealtimeService

        service = PolygonRealtimeService()
        await service.initialize()

        results = {
            "quotes_updated": 0,
            "news_stored": 0,
            "bars_stored": 0
        }

        try:
            if request.update_quotes:
                results["quotes_updated"] = await service.update_market_cache(
                    symbols=request.symbols
                )

            if request.update_news:
                results["news_stored"] = await service.fetch_and_store_news(
                    symbols=request.symbols,
                    limit=50
                )

            if request.update_bars:
                # Run in background for large operations
                async def fetch_bars():
                    try:
                        svc = PolygonRealtimeService()
                        await svc.initialize()
                        await svc.update_ohlcv_bars(
                            symbols=request.symbols,
                            days=request.days
                        )
                        await svc.close()
                    except Exception as e:
                        logger.error(f"Background bar fetch failed: {e}")

                background_tasks.add_task(fetch_bars)
                results["bars_stored"] = -1  # Indicates running in background

        finally:
            await service.close()

        return DataUpdateResponse(
            status="success",
            quotes_updated=results["quotes_updated"],
            news_stored=results["news_stored"],
            bars_stored=results["bars_stored"],
            message=f"Market data updated. Bars: {'running in background' if results['bars_stored'] == -1 else results['bars_stored']}"
        )

    except Exception as e:
        logger.error(f"Market data update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-data-status")
async def get_market_data_status(
    user_id: UUID = Depends(require_admin),
):
    """Get status of market data sources."""
    try:
        from cift.services.polygon_realtime_service import PolygonRealtimeService

        service = PolygonRealtimeService()
        await service.initialize()

        try:
            market_status = await service.get_market_status()

            # Get cache stats
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                cache_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM market_data_cache"
                )
                latest_update = await conn.fetchval(
                    "SELECT MAX(updated_at) FROM market_data_cache"
                )
                news_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM news_articles WHERE published_at > NOW() - INTERVAL '24 hours'"
                )
                bars_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM ohlcv_bars"
                )

            return {
                "polygon_configured": bool(service.api_key),
                "market_status": market_status,
                "cache": {
                    "symbols_cached": cache_count,
                    "last_updated": latest_update.isoformat() if latest_update else None
                },
                "news": {
                    "articles_24h": news_count
                },
                "historical": {
                    "total_bars": bars_count
                }
            }

        finally:
            await service.close()

    except Exception as e:
        logger.error(f"Failed to get market data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start-realtime-worker")
async def start_realtime_worker(
    user_id: UUID = Depends(require_admin),
):
    """Start the background worker for real-time data updates."""
    try:
        from cift.services.polygon_realtime_service import polygon_worker

        await polygon_worker.start()

        return {
            "status": "started",
            "message": "Real-time market data worker started",
            "update_interval": polygon_worker.update_interval
        }

    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-realtime-worker")
async def stop_realtime_worker(
    user_id: UUID = Depends(require_admin),
):
    """Stop the background worker."""
    try:
        from cift.services.polygon_realtime_service import polygon_worker

        await polygon_worker.stop()

        return {
            "status": "stopped",
            "message": "Real-time market data worker stopped"
        }

    except Exception as e:
        logger.error(f"Failed to stop worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS - FINNHUB WEBSOCKET (Real-Time Streaming)
# ============================================================================

# Global WebSocket service instance
_finnhub_service = None


async def get_finnhub_service():
    """Get or create Finnhub WebSocket service instance."""
    global _finnhub_service
    if _finnhub_service is None:
        from cift.services.finnhub_realtime_service import FinnhubRealtimeService
        _finnhub_service = FinnhubRealtimeService()
    return _finnhub_service


@router.post("/websocket/connect")
async def connect_websocket(
    user_id: UUID = Depends(require_admin),
):
    """
    Connect to Finnhub WebSocket for real-time streaming.

    Requires FINNHUB_API_KEY in environment.
    Get free API key at: https://finnhub.io
    """
    try:
        service = await get_finnhub_service()
        await service.connect()

        return {
            "status": "connected",
            "message": "WebSocket connected to Finnhub",
            "subscribed_symbols": list(service.subscribed_symbols)
        }

    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/websocket/disconnect")
async def disconnect_websocket(
    user_id: UUID = Depends(require_admin),
):
    """Disconnect from Finnhub WebSocket."""
    try:
        service = await get_finnhub_service()
        await service.disconnect()

        return {
            "status": "disconnected",
            "message": "WebSocket disconnected"
        }

    except Exception as e:
        logger.error(f"WebSocket disconnection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SubscribeRequest(BaseModel):
    """WebSocket subscription request."""
    symbols: list[str]


@router.post("/websocket/subscribe")
async def subscribe_symbols(
    request: SubscribeRequest,
    user_id: UUID = Depends(require_admin),
):
    """Subscribe to real-time updates for specific symbols."""
    try:
        service = await get_finnhub_service()

        if not service.connected:
            raise HTTPException(
                status_code=400,
                detail="WebSocket not connected. Call /admin/websocket/connect first"
            )

        for symbol in request.symbols:
            await service.subscribe(symbol.upper())

        return {
            "status": "subscribed",
            "symbols": request.symbols,
            "total_subscribed": len(service.subscribed_symbols)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subscribe failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/websocket/unsubscribe")
async def unsubscribe_symbols(
    request: SubscribeRequest,
    user_id: UUID = Depends(require_admin),
):
    """Unsubscribe from real-time updates for specific symbols."""
    try:
        service = await get_finnhub_service()

        for symbol in request.symbols:
            await service.unsubscribe(symbol.upper())

        return {
            "status": "unsubscribed",
            "symbols": request.symbols,
            "remaining_subscribed": len(service.subscribed_symbols)
        }

    except Exception as e:
        logger.error(f"Unsubscribe failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/websocket/status")
async def get_websocket_status(
    user_id: UUID = Depends(require_admin),
):
    """Get WebSocket connection status and statistics."""
    try:
        service = await get_finnhub_service()

        return {
            "connected": service.connected,
            "subscribed_symbols": list(service.subscribed_symbols),
            "symbol_count": len(service.subscribed_symbols),
            "stats": service.stats,
            "latest_prices": {
                s: service.last_prices.get(s)
                for s in list(service.subscribed_symbols)[:10]  # First 10
            }
        }

    except Exception as e:
        logger.error(f"Failed to get WebSocket status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/websocket/price/{symbol}")
async def get_realtime_price(
    symbol: str,
    user_id: UUID = Depends(require_admin),
):
    """Get the latest real-time price for a symbol."""
    try:
        service = await get_finnhub_service()

        symbol = symbol.upper()
        price = service.last_prices.get(symbol)

        if price is None:
            # Check if subscribed
            if symbol not in service.subscribed_symbols:
                raise HTTPException(
                    status_code=404,
                    detail=f"Symbol {symbol} not subscribed. Subscribe first."
                )
            return {
                "symbol": symbol,
                "price": None,
                "message": "Price not yet received, waiting for next trade"
            }

        return {
            "symbol": symbol,
            "price": price,
            "source": "finnhub_websocket",
            "realtime": True
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get realtime price: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ENDPOINTS - SYSTEM
# ============================================================================

@router.get("/system-stats")
async def get_system_stats(
    user_id: UUID = Depends(require_admin),
):
    """Get system statistics."""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # User stats
        user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
        active_users = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE last_login > NOW() - INTERVAL '7 days'"
        )

        # Trading stats
        order_count = await conn.fetchval("SELECT COUNT(*) FROM orders")
        position_count = await conn.fetchval("SELECT COUNT(*) FROM positions")

        # Data stats
        symbol_count = await conn.fetchval("SELECT COUNT(*) FROM symbols")
        news_count = await conn.fetchval("SELECT COUNT(*) FROM news_articles")

        return {
            "users": {
                "total": user_count,
                "active_7d": active_users
            },
            "trading": {
                "total_orders": order_count,
                "open_positions": position_count
            },
            "data": {
                "symbols_tracked": symbol_count,
                "news_articles": news_count
            },
            "timestamp": datetime.utcnow().isoformat()
        }
