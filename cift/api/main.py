"""
CIFT Markets - Main FastAPI Application

Production-grade API with monitoring, tracing, and comprehensive middleware.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.sessions import SessionMiddleware

from cift.core.config import settings
from cift.core.database import (
    check_all_connections,
    initialize_all_connections,
)
from cift.core.limiter import limiter
from cift.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting CIFT Markets API...")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug mode: {settings.app_debug}")
    logger.info(f"Frontend URL: {settings.frontend_url}")
    logger.info(f"API Base URL: {settings.api_base_url}")

    # Initialize all database connections (with timeout protection)
    try:
        # Use asyncio.wait_for to timeout database init after 5 seconds
        await asyncio.wait_for(initialize_all_connections(), timeout=5.0)
        logger.info("✅ All database connections initialized")
    except TimeoutError:
        logger.warning("⚠️ Database initialization timed out (5s). API will start in degraded mode.")
        logger.warning("Some endpoints may not work without database connectivity.")
    except Exception as e:
        logger.error(f"⚠️ Failed to initialize connections: {e}")
        logger.warning("API starting in degraded mode - database endpoints will fail")

    # Initialize execution engine (skip if databases failed)
    try:
        from cift.core.execution_engine import execution_engine
        await asyncio.wait_for(execution_engine.start(), timeout=3.0)
        logger.info("✅ Execution engine started")
    except Exception as e:
        logger.warning(f"⚠️ Execution engine failed to start: {e}")

    # Start market data source - use REAL Polygon data if API key is available
    market_data_task = None
    use_real_data = bool(settings.polygon_api_key)

    try:
        from cift.api.routes.market_data import publish_price_update

        # --- FINNHUB WEBSOCKET (Real-time Streaming) ---
        if settings.finnhub_api_key:
            try:
                from cift.services.finnhub_realtime_service import FinnhubRealtimeService
                finnhub_service = FinnhubRealtimeService()
                
                async def handle_finnhub_update(symbol, price, volume, timestamp):
                    """Handle real-time update from Finnhub."""
                    # Broadcast to WebSocket clients
                    await publish_price_update(
                        symbol=symbol,
                        price=price,
                        bid=price * 0.9999, # Simulated bid/ask
                        ask=price * 1.0001
                    )
                    
                    # Update Database Cache (Price/Volume only)
                    # We don't update OHLC here as Finnhub trade updates don't have it
                    try:
                        from cift.core.database import get_postgres_pool
                        pool = await get_postgres_pool()
                        if pool:
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    INSERT INTO market_data_cache (symbol, price, volume, updated_at)
                                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                                    ON CONFLICT (symbol) DO UPDATE SET
                                        price = EXCLUDED.price,
                                        volume = EXCLUDED.volume,
                                        updated_at = CURRENT_TIMESTAMP
                                """, symbol, float(price), int(volume) if volume else 0)
                    except Exception as e:
                        # Don't log every DB error to avoid spam, but maybe log occasionally
                        pass

                finnhub_service.add_callback(handle_finnhub_update)
                # Start streaming with default symbols
                await finnhub_service.start_streaming()
                logger.info("✅ Finnhub WebSocket started (Real-time streaming)")
            except Exception as e:
                logger.error(f"Failed to start Finnhub WebSocket: {e}")

        if use_real_data:
            # USE REAL POLYGON DATA
            from cift.services.polygon_realtime_service import PolygonRealtimeService

            polygon_service = PolygonRealtimeService()
            await polygon_service.initialize()

            async def fetch_and_broadcast_real_data():
                """Fetch real market data from Polygon and broadcast."""
                # Use all default symbols
                symbols = polygon_service.DEFAULT_SYMBOLS
                logger.info(f"Starting real-time data fetch for {len(symbols)} symbols")
                
                while True:
                    try:
                        # Fetch quotes for all symbols
                        # The service handles rate limiting internally
                        quotes = await polygon_service.get_quotes_batch(symbols)
                        
                        if not quotes:
                            logger.warning("No quotes fetched from Polygon")
                            await asyncio.sleep(60)
                            continue

                        for symbol, quote in quotes.items():
                            await publish_price_update(
                                symbol=symbol,
                                price=quote["price"],
                                bid=quote["price"] * 0.9999,
                                ask=quote["price"] * 1.0001,
                            )
                            # Also update the database cache
                            from cift.core.database import get_postgres_pool
                            pool = await get_postgres_pool()
                            async with pool.acquire() as conn:
                                await conn.execute("""
                                    INSERT INTO market_data_cache (symbol, price, bid, ask, volume, change, change_pct, high, low, open)
                                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                                    ON CONFLICT (symbol) DO UPDATE SET
                                        price = EXCLUDED.price,
                                        volume = EXCLUDED.volume,
                                        change = EXCLUDED.change,
                                        change_pct = EXCLUDED.change_pct,
                                        high = EXCLUDED.high,
                                        low = EXCLUDED.low,
                                        open = EXCLUDED.open,
                                        updated_at = CURRENT_TIMESTAMP
                                """, symbol, quote["price"], quote["price"]*0.9999, quote["price"]*1.0001,
                                    quote.get("volume", 0), quote.get("change", 0), quote.get("change_percent", 0),
                                    quote.get("high", quote["price"]), quote.get("low", quote["price"]), quote.get("open", quote["price"]))
                        
                        # Wait before next batch
                        # If we are on free tier, the _request method already slept a lot
                        # If we are on paid tier, we can update more frequently
                        await asyncio.sleep(15) 
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Real data fetch error: {e}")
                        await asyncio.sleep(30)

            market_data_task = asyncio.create_task(fetch_and_broadcast_real_data())
            logger.info("✅ REAL Polygon market data started (live prices from polygon.io)")
            
            # Update 52-week high/low data on startup (runs in background)
            async def update_52week_on_startup():
                """Update 52-week data shortly after startup."""
                await asyncio.sleep(30)  # Wait 30s before starting
                logger.info("📊 Starting 52-week high/low data update...")
                try:
                    updated = await polygon_service.update_52week_highs_lows()
                    logger.success(f"✅ Updated 52-week data for {updated} symbols")
                except Exception as e:
                    logger.error(f"Failed to update 52-week data: {e}")
            
            asyncio.create_task(update_52week_on_startup())
        else:
            # FALLBACK TO SIMULATOR
            from cift.core.market_simulator import simulator

            async def broadcast_tick(tick_data):
                """Broadcast simulated tick to WebSocket subscribers."""
                await publish_price_update(
                    symbol=tick_data["symbol"],
                    price=tick_data["price"],
                    bid=tick_data.get("bid"),
                    ask=tick_data.get("ask"),
                )

            market_data_task = asyncio.create_task(simulator.generate_updates(broadcast_tick))
            logger.warning("⚠️ Market data SIMULATOR started (no Polygon API key - using fake data)")
    except Exception as e:
        logger.warning(f"⚠️ Market simulator failed to start: {e}")

    # Start background tasks (KYC processing, portfolio snapshots, etc.)
    try:
        from cift.core.scheduler import setup_background_tasks
        await setup_background_tasks()
    except Exception as e:
        logger.warning(f"⚠️ Background tasks setup failed: {e}")

    # TODO: Initialize Kafka consumers for market data (production)
    # TODO: Load ML models into memory

    logger.success("✅ CIFT Markets API started successfully")

    yield

    # Shutdown market data source
    if market_data_task:
        try:
            market_data_task.cancel()
            try:
                await market_data_task
            except asyncio.CancelledError:
                pass
            if not use_real_data:
                from cift.core.market_simulator import simulator
                simulator.stop()
            logger.info("Market data source stopped")
        except Exception as e:
            logger.warning(f"⚠️ Failed to stop market data: {e}")

    # Shutdown background tasks
    try:
        from cift.core.scheduler import stop_scheduler
        stop_scheduler()
    except Exception as e:
        logger.warning(f"⚠️ Failed to stop background tasks: {e}")
app = FastAPI(
    title="CIFT Markets API",
    description="Computational Intelligence for Financial Trading - Production API",
    version="0.1.0",
    docs_url="/docs" if settings.app_debug else None,
    redoc_url="/redoc" if settings.app_debug else None,
    lifespan=lifespan,
)

# Initialize Rate Limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS middleware - MUST be first to handle preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:3002",
        settings.frontend_url,  # Add configured frontend URL
        "http://20.250.40.67:3000", # Explicitly add remote IP just in case
        "http://20.250.40.67",      # Add remote IP without port
        "https://20.250.40.67",     # Add https just in case
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight for 1 hour
)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Session middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    session_cookie="cift_session",
    max_age=3600,
)

# Trusted host middleware (production)
if settings.app_env == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["ciftmarkets.com", "*.ciftmarkets.com"],
    )

# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "CIFT Markets API",
        "version": "0.1.0",
        "description": "Computational Intelligence for Financial Trading",
        "status": "operational",
        "environment": settings.app_env,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "version": "0.1.0",
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - verify all dependencies are available by querying actual services."""
    # Check all database connections with real queries
    connection_status = await check_all_connections()

    # Determine if system is ready (all critical services must be healthy)
    critical_services = ["postgres", "questdb", "redis"]
    is_ready = all(
        connection_status.get(service) == "healthy"
        for service in critical_services
    )

    # TODO: Add Kafka health check when implemented
    # TODO: Add ML model status check when implemented

    return {
        "ready": is_ready,
        "timestamp": logger._core.clock.now().isoformat() if hasattr(logger, '_core') else None,
        "checks": {
            "postgres": connection_status.get("postgres", "unknown"),
            "questdb": connection_status.get("questdb", "unknown"),
            "redis": connection_status.get("redis", "unknown"),
            "kafka": "not_implemented",  # Will implement in Phase 1
            "models": "not_loaded",  # Will implement in Phase 3
        },
    }


# ============================================================================
# METRICS
# ============================================================================

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Mount static files for uploads
upload_dir = "uploads"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")


# ============================================================================
# ROUTERS
# ============================================================================

# Import route modules
from cift.api.routes import (  # noqa: E402
    admin,
    alerts,
    analytics,
    assets,
    auth,
    drilldowns,
    funding,
    globe,
    market_data,
    news,
    notifications,
    onboarding,
    screener,
    search,
    statements,
    support,
    trading,
    transactions,
    verify,
    watchlists,
    webhooks,
)

try:
    from cift.api.routes import settings as settings_routes
    SETTINGS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Settings routes not available: {e}")
    SETTINGS_AVAILABLE = False

# Include routers with /api/v1 prefix
app.include_router(auth.router, prefix="/api/v1")
app.include_router(market_data.router, prefix="/api/v1")
app.include_router(trading.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(drilldowns.router, prefix="/api/v1")
app.include_router(watchlists.router, prefix="/api/v1")
app.include_router(transactions.router, prefix="/api/v1")

# Chart-related routes
from cift.api.routes import chart_drawings, chart_templates, price_alerts  # noqa: E402

app.include_router(chart_drawings.router, prefix="/api/v1")
app.include_router(chart_templates.router, prefix="/api/v1")
app.include_router(price_alerts.router, prefix="/api/v1")

# Company data routes (fundamentals, earnings, patterns)
try:
    from cift.api.routes import company_data
    app.include_router(company_data.router, prefix="/api/v1")
    logger.info("Company data routes loaded")
except ImportError as e:
    logger.warning(f"Company data routes not available: {e}")

# Public verification endpoint (no auth required)
app.include_router(verify.router, prefix="/api/v1")

# Webhook endpoints (no auth required - verified via signature)
app.include_router(webhooks.router, prefix="/api/v1")

# Critical feature routers (Phase 5-7 completion)
app.include_router(funding.router, prefix="/api/v1")
app.include_router(onboarding.router, prefix="/api/v1")
app.include_router(support.router, prefix="/api/v1")
app.include_router(news.router, prefix="/api/v1")
app.include_router(globe.router, prefix="/api/v1")
app.include_router(assets.router, prefix="/api/v1/globe")
app.include_router(screener.router, prefix="/api/v1")
app.include_router(statements.router, prefix="/api/v1")
app.include_router(alerts.router, prefix="/api/v1")
app.include_router(notifications.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")

# Stock Analysis Engine (AI-powered recommendations)
try:
    from cift.api.routes import analysis
    app.include_router(analysis.router, prefix="/api/v1")
    logger.info("Stock analysis routes loaded")
except Exception as e:
    logger.warning(f"Stock analysis routes not available: {e}")

# Real-time streaming routes (SSE)
try:
    from cift.api.routes import stream
    app.include_router(stream.router, prefix="/api/v1")
    logger.info("Real-time streaming routes loaded")
except ImportError as e:
    logger.warning(f"Streaming routes not available: {e}")

if SETTINGS_AVAILABLE:
    app.include_router(settings_routes.router, prefix="/api/v1")

# ML Inference routes (Phase 8 - ML Implementation)
try:
    from cift.api.routes import inference
    app.include_router(inference.router)  # Already has /api/v1 prefix
    logger.info("ML Inference routes loaded")
except ImportError as e:
    logger.warning(f"ML Inference routes not available: {e}")

# TODO: Add remaining routers (Future phases):
# - Backtests (/api/v1/backtests) - Backtesting engine
# - Strategies (/api/v1/strategies) - Strategy management


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "cift.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.hot_reload,
        log_level=settings.log_level.lower(),
    )
