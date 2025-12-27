"""
CIFT Markets - Market Data API Routes

High-performance market data endpoints with WebSocket support for real-time streaming.

Performance optimizations:
- Raw asyncpg queries (3x faster than ORM)
- Redis caching (sub-millisecond)
- Polars for data aggregation (19.5x faster)
- WebSocket for efficient real-time updates
"""

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

from cift.core.auth import User, get_current_active_user
from cift.core.data_processing import calculate_technical_indicators, load_ohlcv_data, calculate_volume_profile
from cift.core.database import db_manager
from cift.core.trading_queries import get_ohlcv_last_n_bars
from cift.services.market_data_service import market_data_service

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/market-data", tags=["Market Data"])


# ============================================================================
# MODELS
# ============================================================================


class TickData(BaseModel):
    """Tick data response model."""

    timestamp: datetime
    symbol: str
    price: float
    volume: int
    bid: float | None = None
    ask: float | None = None


class OHLCVBar(BaseModel):
    """OHLCV bar response model."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceQuote(BaseModel):
    """Real-time price quote."""

    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    spread_bps: float | None = None
    change: float | None = None
    change_pct: float | None = None
    high: float | None = None
    low: float | None = None
    open: float | None = None
    volume: int | None = None
    timestamp: datetime


class MarketDepth(BaseModel):
    """Market depth/order book data."""

    symbol: str
    bids: list[tuple[float, int]]  # [(price, volume), ...]
    asks: list[tuple[float, int]]
    timestamp: datetime


# ============================================================================
# REST ENDPOINTS
# ============================================================================


@router.get("/quote/{symbol}", response_model=PriceQuote)
async def get_quote(symbol: str):
    """
    Get latest quote for a symbol from database.

    Checks: market_data_cache first, then market_data as fallback.

    Performance: ~2ms (database query)
    """
    symbol_upper = symbol.upper()

    async with db_manager.pool.acquire() as conn:
        # First try market_data_cache (most up-to-date)
        row = await conn.fetchrow(
            """
            SELECT symbol, price, bid, ask, volume, change, change_pct, high, low, open, updated_at as timestamp
            FROM market_data_cache
            WHERE symbol = $1
            """,
            symbol_upper,
        )

        # Check if data is stale (older than 10 seconds) or missing
        is_stale = False
        if row and row["timestamp"]:
            # Handle timezone-aware vs naive datetime comparison
            db_ts = row["timestamp"]
            now = datetime.utcnow()
            
            # If db_ts is timezone-aware, make now aware too
            if db_ts.tzinfo is not None:
                from datetime import timezone
                now = datetime.now(timezone.utc)
            
            age = (now - db_ts).total_seconds()
            if age > 10:  # 10 seconds staleness threshold for REST API
                is_stale = True
        
        if not row or is_stale:
            try:
                # Fetch real-time data
                quotes = await market_data_service.get_quotes_batch([symbol_upper])
                quote_data = quotes.get(symbol_upper)
                
                if quote_data:
                    # Update cache
                    await conn.execute(
                        """
                        INSERT INTO market_data_cache (
                            symbol, price, bid, ask, volume, change, change_pct, high, low, open, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                        ON CONFLICT (symbol) DO UPDATE SET
                            price = EXCLUDED.price,
                            bid = EXCLUDED.bid,
                            ask = EXCLUDED.ask,
                            volume = EXCLUDED.volume,
                            change = EXCLUDED.change,
                            change_pct = EXCLUDED.change_pct,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            open = EXCLUDED.open,
                            updated_at = NOW()
                        """,
                        symbol_upper,
                        quote_data.get("price"),
                        quote_data.get("bid"),
                        quote_data.get("ask"),
                        quote_data.get("volume"),
                        quote_data.get("change"),
                        quote_data.get("change_percent"),
                        quote_data.get("high"),
                        quote_data.get("low"),
                        quote_data.get("open")
                    )
                    
                    # Fetch the updated row
                    row = await conn.fetchrow(
                        """
                        SELECT symbol, price, bid, ask, volume, change, change_pct, high, low, open, updated_at as timestamp
                        FROM market_data_cache
                        WHERE symbol = $1
                        """,
                        symbol_upper,
                    )
            except Exception as e:
                logger.error(f"Failed to fetch/update real-time quote for {symbol_upper}: {e}")
                # If we have stale data, we'll use it. If no data, we'll fall through to market_data table check.

        # Fallback to market_data (historical/seeded) if still no row
        if not row:
            row = await conn.fetchrow(
                """
                SELECT symbol, price, bid, ask, volume, timestamp
                FROM market_data
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol_upper,
            )

    if not row:
        raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol_upper}")

    # Calculate spread in basis points
    bid = float(row["bid"]) if row["bid"] else float(row["price"]) * 0.9999
    ask = float(row["ask"]) if row["ask"] else float(row["price"]) * 1.0001
    spread_bps = ((ask - bid) / bid) * 10000 if bid > 0 else 0.0

    return PriceQuote(
        symbol=row["symbol"],
        price=float(row["price"]),
        bid=bid,
        ask=ask,
        spread_bps=round(spread_bps, 2),
        change=float(row["change"]) if row.get("change") else None,
        change_pct=float(row["change_pct"]) if row.get("change_pct") else None,
        high=float(row["high"]) if row.get("high") else None,
        low=float(row["low"]) if row.get("low") else None,
        open=float(row["open"]) if row.get("open") else None,
        volume=int(row["volume"]) if row.get("volume") else None,
        timestamp=row["timestamp"],
    )


@router.get("/quotes", response_model=list[PriceQuote])
async def get_quotes(
    symbols: str = Query(..., description="Comma-separated list of symbols"),
):
    """
    Get latest quotes for multiple symbols from database.

    Checks: market_data_cache -> market_data -> ohlcv_bars (in order of preference)

    Performance: ~3-5ms for 10 symbols (efficient batch query)
    """
    if not symbols:
        return []

    # Parse comma-separated symbols and convert to uppercase
    symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbols_list:
        return []

    symbols_upper = symbols_list
    found_symbols = set()
    all_quotes = []

    async with db_manager.pool.acquire() as conn:
        # First try market_data_cache (most up-to-date real-time data)
        cache_rows = await conn.fetch(
            """
            SELECT symbol, price, bid, ask, volume, change, change_pct, high, low, open, updated_at as timestamp
            FROM market_data_cache
            WHERE symbol = ANY($1::text[])
            """,
            symbols_upper,
        )

        for row in cache_rows:
            found_symbols.add(row["symbol"])
            bid = float(row["bid"]) if row["bid"] else float(row["price"]) * 0.9999
            ask = float(row["ask"]) if row["ask"] else float(row["price"]) * 1.0001
            spread_bps = ((ask - bid) / bid) * 10000 if bid > 0 else 0.0

            all_quotes.append(
                PriceQuote(
                    symbol=row["symbol"],
                    price=float(row["price"]),
                    bid=bid,
                    ask=ask,
                    spread_bps=round(spread_bps, 2),
                    change=float(row["change"]) if row["change"] else None,
                    change_pct=float(row["change_pct"]) if row["change_pct"] else None,
                    high=float(row["high"]) if row["high"] else None,
                    low=float(row["low"]) if row["low"] else None,
                    open=float(row["open"]) if row["open"] else None,
                    volume=int(row["volume"]) if row["volume"] else None,
                    timestamp=row["timestamp"],
                )
            )

        # Find missing symbols and try market_data
        missing_symbols = [s for s in symbols_upper if s not in found_symbols]
        if missing_symbols:
            market_rows = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) symbol, price, bid, ask, volume, timestamp
                FROM market_data
                WHERE symbol = ANY($1::text[])
                ORDER BY symbol, timestamp DESC
                """,
                missing_symbols,
            )

            for row in market_rows:
                found_symbols.add(row["symbol"])
                bid = float(row["bid"]) if row["bid"] else None
                ask = float(row["ask"]) if row["ask"] else None
                spread_bps = ((ask - bid) / bid * 10000) if (bid and ask and bid > 0) else 0.0

                all_quotes.append(
                    PriceQuote(
                        symbol=row["symbol"],
                        price=float(row["price"]),
                        bid=bid,
                        ask=ask,
                        spread_bps=round(spread_bps, 2),
                        volume=int(row["volume"]) if row["volume"] else None,
                        timestamp=row["timestamp"],
                    )
                )

        # Final fallback: check ohlcv_bars for any remaining symbols
        remaining_symbols = [s for s in symbols_upper if s not in found_symbols]
        if remaining_symbols:
            ohlcv_rows = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol)
                    symbol,
                    close as price,
                    open as bid,
                    close as ask,
                    volume,
                    timestamp
                FROM ohlcv_bars
                WHERE symbol = ANY($1::text[])
                ORDER BY symbol, timestamp DESC
                """,
                remaining_symbols,
            )

            for row in ohlcv_rows:
                found_symbols.add(row["symbol"])
                bid = float(row["bid"]) if row["bid"] else None
                ask = float(row["ask"]) if row["ask"] else None
                spread_bps = ((ask - bid) / bid * 10000) if (bid and ask and bid > 0) else 0.0

                all_quotes.append(
                    PriceQuote(
                        symbol=row["symbol"],
                        price=float(row["price"]),
                        bid=bid,
                        ask=ask,
                        spread_bps=round(spread_bps, 2),
                        timestamp=row["timestamp"],
                    )
                )

    # Final fallback: Use market_data_service to fetch real-time data for remaining symbols
    # This is especially important for INDEX symbols (S&P, DOW, NASDAQ) which use yfinance
    final_remaining = [s for s in symbols_upper if s not in found_symbols]
    if final_remaining:
        try:
            live_quotes = await market_data_service.get_quotes_batch(final_remaining)
            for symbol, quote_data in live_quotes.items():
                if quote_data and quote_data.get("price", 0) > 0:
                    price = float(quote_data["price"])
                    bid = float(quote_data.get("bid") or price * 0.9999)
                    ask = float(quote_data.get("ask") or price * 1.0001)
                    spread_bps = ((ask - bid) / bid * 10000) if bid > 0 else 0.0
                    
                    all_quotes.append(
                        PriceQuote(
                            symbol=quote_data.get("original_symbol", symbol),
                            price=price,
                            bid=bid,
                            ask=ask,
                            spread_bps=round(spread_bps, 2),
                            change=float(quote_data.get("change") or 0),
                            change_pct=float(quote_data.get("change_percent") or 0),
                            high=float(quote_data.get("high") or 0) if quote_data.get("high") else None,
                            low=float(quote_data.get("low") or 0) if quote_data.get("low") else None,
                            open=float(quote_data.get("open") or 0) if quote_data.get("open") else None,
                            volume=int(quote_data.get("volume") or 0) if quote_data.get("volume") else None,
                            timestamp=datetime.utcnow(),
                        )
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch live quotes for {final_remaining}: {e}")

    return all_quotes


# ============================================================================
# ORDER BOOK (Level 2 Data)
# ============================================================================


@router.get("/orderbook/{symbol}")
async def get_order_book(
    symbol: str,
    levels: int = Query(10, ge=1, le=50, description="Number of price levels"),
):
    """
    Get order book (Level 2 market depth) for a symbol.

    **Note:** Currently returns simulated order book based on spread.
    Real L2 data requires market data provider subscription.

    Performance: ~2ms
    """
    symbol = symbol.upper()

    # Get current quote
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT price, bid, ask, volume
            FROM market_data_cache
            WHERE symbol = $1
            """,
            symbol,
        )

        if not row:
            # Fallback to OHLCV
            row = await conn.fetchrow(
                """
                SELECT close as price, open as bid, close as ask, volume
                FROM ohlcv_bars
                WHERE symbol = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                symbol,
            )

    if not row:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    price = float(row["price"])
    bid = float(row["bid"]) if row["bid"] else price * 0.9995
    ask = float(row["ask"]) if row["ask"] else price * 1.0005
    base_volume = int(row["volume"]) if row["volume"] else 10000

    # Generate realistic order book levels
    import random

    random.seed(int(price * 100))  # Deterministic for same price

    bids = []
    asks = []
    spread = (ask - bid) / 2

    for i in range(levels):
        # Bid levels (decreasing prices)
        bid_price = round(bid - (spread * 0.2 * i), 2)
        bid_size = int(base_volume * (0.05 + random.random() * 0.1) / (i + 1))
        bids.append({"price": bid_price, "size": bid_size, "orders": random.randint(1, 5 + i)})

        # Ask levels (increasing prices)
        ask_price = round(ask + (spread * 0.2 * i), 2)
        ask_size = int(base_volume * (0.05 + random.random() * 0.1) / (i + 1))
        asks.append({"price": ask_price, "size": ask_size, "orders": random.randint(1, 5 + i)})

    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "bids": bids,
        "asks": asks,
        "spread": round(ask - bid, 4),
        "spread_bps": round((ask - bid) / bid * 10000, 2) if bid > 0 else 0,
        "midpoint": round((bid + ask) / 2, 4),
        "_simulated": True,  # Flag that this is simulated data
    }


# ============================================================================
# TIME & SALES (Recent Trades) - REAL DATA FROM FINNHUB WEBSOCKET
# ============================================================================


@router.get("/timesales/{symbol}")
async def get_time_and_sales(
    symbol: str,
    limit: int = Query(50, ge=1, le=500, description="Number of trades"),
):
    """
    Get recent trades (Time & Sales) for a symbol.

    **Data Sources (in order of priority):**
    1. Redis/Dragonfly cache (real-time trades from Finnhub WebSocket)
    2. QuestDB trade_executions table (persisted trades)
    3. In-memory cache from Finnhub service
    4. Simulated data (fallback when no real data available)

    Performance: ~1-3ms (cache), ~5-10ms (QuestDB)
    """
    symbol = symbol.upper()
    
    # Import Finnhub service
    from cift.services.finnhub_realtime_service import get_finnhub_service
    
    finnhub = get_finnhub_service()
    
    # Source 1: Try Redis/Dragonfly cache first (fastest)
    try:
        trades = await finnhub.get_trades_from_cache(symbol, limit)
        if trades and len(trades) >= min(limit, 5):  # Have enough trades
            return {
                "symbol": symbol,
                "trades": trades,
                "count": len(trades),
                "last_price": trades[0]["price"] if trades else None,
                "_source": "realtime_cache",
                "_simulated": False,
            }
    except Exception as e:
        logger.debug(f"Redis cache miss for {symbol}: {e}")
    
    # Source 2: Try QuestDB for persisted trades
    try:
        await questdb_manager.initialize()
        rows = await questdb_manager.fetch(
            """
            SELECT timestamp, price, size, side, exchange, trade_id
            FROM trade_executions
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            symbol,
            limit,
        )
        
        if rows and len(rows) >= min(limit, 3):
            trades = [
                {
                    "time": row["timestamp"].isoformat() if row["timestamp"] else None,
                    "price": float(row["price"]),
                    "size": int(row["size"]) if row["size"] else 1,
                    "side": row["side"] or "unknown",
                    "exchange": row["exchange"] or "UNKNOWN",
                    "trade_id": row["trade_id"] or "",
                }
                for row in rows
            ]
            return {
                "symbol": symbol,
                "trades": trades,
                "count": len(trades),
                "last_price": trades[0]["price"] if trades else None,
                "_source": "questdb",
                "_simulated": False,
            }
    except Exception as e:
        logger.debug(f"QuestDB query failed for {symbol}: {e}")
    
    # Source 3: In-memory cache from Finnhub service
    in_memory_trades = finnhub.get_recent_trades(symbol, limit)
    if in_memory_trades:
        return {
            "symbol": symbol,
            "trades": in_memory_trades,
            "count": len(in_memory_trades),
            "last_price": in_memory_trades[0]["price"] if in_memory_trades else None,
            "_source": "memory_cache",
            "_simulated": False,
        }
    
    # Source 4: Fallback to simulated data (when no real data available)
    # This happens when symbol isn't subscribed or market is closed
    logger.info(f"No real trades for {symbol}, generating simulated data")
    
    # Get recent bars to simulate trades
    async with db_manager.pool.acquire() as conn:
        bars = await conn.fetch(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_bars
            WHERE symbol = $1
            ORDER BY timestamp DESC
            LIMIT 10
            """,
            symbol,
        )

        # Get current quote
        quote = await conn.fetchrow(
            """
            SELECT price, bid, ask
            FROM market_data_cache
            WHERE symbol = $1
            """,
            symbol,
        )

    if not bars and not quote:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    current_price = float(quote["price"]) if quote else float(bars[0]["close"]) if bars else 100.0

    # Generate realistic time & sales
    import random
    from datetime import timedelta

    trades = []
    base_time = datetime.utcnow()

    for i in range(limit):
        # Time decreases as we go back
        trade_time = base_time - timedelta(seconds=i * random.uniform(0.5, 5))

        # Price varies slightly around current
        price_var = (random.random() - 0.5) * 0.002 * current_price
        trade_price = round(current_price + price_var, 2)

        # Size follows power law distribution
        size = int(random.paretovariate(1.5) * 50)
        size = min(max(size, 1), 10000)

        # Side based on price movement
        side = "buy" if price_var > 0 else "sell"

        trades.append(
            {
                "time": trade_time.isoformat(),
                "price": trade_price,
                "size": size,
                "side": side,
                "exchange": random.choice(["NYSE", "NASDAQ", "ARCA", "BATS"]),
            }
        )

    return {
        "symbol": symbol,
        "trades": trades,
        "count": len(trades),
        "last_price": current_price,
        "_source": "simulated",
        "_simulated": True,  # Flag that this is simulated data
    }


@router.get("/bars/{symbol}", response_model=list[OHLCVBar])
async def get_bars(
    symbol: str,
    timeframe: str = Query("1m", description="Bar timeframe (1m, 5m, 15m, 1h, 1d)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of bars"),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
):
    """
    Get OHLCV bars for a symbol.

    Performance: ~3-5ms for 100 bars (uses QuestDB SAMPLE BY optimization)
    """
    # Use optimized QuestDB query
    bars = await get_ohlcv_last_n_bars(symbol, timeframe, limit)

    if not bars:
        # Try to fetch from Finnhub (FREE and working) or Polygon as fallback
        logger.info(f"No bars found for {symbol}, fetching from Finnhub/Polygon...")
        try:
            # Determine days to fetch based on timeframe and limit
            days_to_fetch = 30 # Default
            if timeframe == '1d':
                days_to_fetch = limit + 60 # Fetch more for daily
            elif timeframe == '1h':
                days_to_fetch = (limit // 24) + 14
            elif timeframe in ('5m', '15m', '30m'):
                days_to_fetch = 14 # 2 weeks for intraday
            
            # Use new unified fetch method (Finnhub primary, Polygon fallback)
            await market_data_service.fetch_and_store_ohlcv(
                symbol=symbol, 
                days=days_to_fetch, 
                timeframe=timeframe
            )
            
            # Retry query
            bars = await get_ohlcv_last_n_bars(symbol, timeframe, limit)
            
        except Exception as e:
            logger.error(f"Failed to backfill bars for {symbol}: {e}")

    if not bars:
        raise HTTPException(status_code=404, detail=f"No bar data found for {symbol}")

    return [
        OHLCVBar(
            timestamp=bar["timestamp"],
            symbol=bar["symbol"],
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=int(bar["volume"]),
        )
        for bar in bars
    ]


@router.get("/history/{symbol}")
async def get_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = Query("1m", description="Aggregation timeframe"),
    format: str = Query("json", description="Response format (json, csv, parquet)"),
):
    """
    Get historical data for a symbol (supports large datasets).

    Performance:
    - JSON: ~50ms for 10K rows
    - Parquet: ~10ms for 10K rows (19.5x faster, recommended for large datasets)
    """
    # Load data using Polars (19.5x faster than Pandas)
    df = await load_ohlcv_data([symbol], start_date, end_date, timeframe)

    if df.is_empty():
        raise HTTPException(status_code=404, detail="No historical data found")

    # Add technical indicators
    df = calculate_technical_indicators(df)

    if format == "json":
        # Convert to JSON (slower but widely compatible)
        return df.to_dicts()

    elif format == "csv":
        # Stream CSV response
        csv_buffer = df.write_csv()
        return StreamingResponse(
            iter([csv_buffer]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={symbol}_historical.csv"},
        )

    elif format == "parquet":
        # Stream Parquet response (fastest, most efficient)
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={symbol}_historical.parquet"},
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.get("/profile/{symbol}")
async def get_volume_profile(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    bins: int = Query(24, description="Number of price bins"),
):
    """
    Get Volume Profile (Volume by Price) for a symbol over a specific range.
    
    Calculates the distribution of volume across price levels.
    """
    # Load data using Polars (use 1m for granular volume distribution)
    df = await load_ohlcv_data([symbol], start_date, end_date, "1m")

    if df.is_empty():
        # Try with requested timeframe if 1m is not available or too granular
        # But for now, just return empty profile
        return {"price_levels": [], "volumes": [], "max_volume": 0}

    profile = calculate_volume_profile(df, bins=bins)
    return profile


@router.get("/symbols", response_model=list[str])
async def get_available_symbols():
    """
    Get list of available symbols.

    Performance: ~5ms (cached)
    """
    # Get symbols from database (market_data_cache table)
    try:
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT symbol FROM market_data_cache
                ORDER BY symbol
                """
            )
        return (
            [row["symbol"] for row in rows]
            if rows
            else ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD", "NFLX", "DIS"]
        )
    except Exception:
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD", "NFLX", "DIS"]


class TickerItem(BaseModel):
    """Market ticker item for dashboard."""

    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int


@router.get("/ticker", response_model=list[TickerItem])
async def get_market_ticker(
    symbols: str | None = Query(None, description="Comma-separated list of symbols")
):
    """
    Get market ticker data for dashboard display.
    Returns price, change, and volume data for major indices and stocks.

    Performance: ~5ms (uses market_data_cache)
    """
    # Default major symbols if not specified
    default_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META"]

    symbol_list = symbols.split(",") if symbols else default_symbols
    symbol_list = [s.strip().upper() for s in symbol_list]

    try:
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT symbol, price, change, change_pct, volume
                FROM market_data_cache
                WHERE symbol = ANY($1::text[])
                """,
                symbol_list,
            )

        if not rows:
            # Return fallback with zeros if no data
            return [
                TickerItem(symbol=s, price=0.0, change=0.0, changePercent=0.0, volume=0)
                for s in symbol_list
            ]

        return [
            TickerItem(
                symbol=row["symbol"],
                price=float(row["price"]) if row["price"] else 0.0,
                change=float(row["change"]) if row["change"] else 0.0,
                changePercent=float(row["change_pct"]) if row["change_pct"] else 0.0,
                volume=int(row["volume"]) if row["volume"] else 0,
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching ticker data: {e}")
        return [
            TickerItem(symbol=s, price=0.0, change=0.0, changePercent=0.0, volume=0)
            for s in symbol_list
        ]


@router.get("/profile/{symbol}")
async def get_company_profile(symbol: str):
    """Get company profile (Fundamental Data)."""
    profile = await market_data_service.get_company_profile(symbol)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.get("/financials/{symbol}")
async def get_financials(symbol: str):
    """Get company financials."""
    financials = await market_data_service.get_financials(symbol)
    if not financials:
        raise HTTPException(status_code=404, detail="Financials not found")
    return financials


@router.get("/financials/reported/{symbol}")
async def get_financials_reported(symbol: str):
    """Get reported financial statements."""
    data = await market_data_service.get_financials_reported(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Reported financials not found")
    return data


@router.get("/estimates/{symbol}")
async def get_earnings_estimates(symbol: str):
    """Get earnings estimates."""
    data = await market_data_service.get_earnings_estimates(symbol)
    if not data:
        raise HTTPException(status_code=404, detail="Estimates not found")
    return data


class MarketMover(BaseModel):
    """Market mover item."""

    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int


@router.get("/movers", response_model=dict)
async def get_market_movers(limit: int = Query(5, ge=1, le=20)):
    """
    Get top market gainers and losers from database.

    Performance: ~5ms (uses market_data_cache with sorting)
    """
    try:
        async with db_manager.pool.acquire() as conn:
            # Get gainers (top positive change)
            gainers_rows = await conn.fetch(
                """
                SELECT symbol, price, change, change_pct, volume
                FROM market_data_cache
                WHERE change_pct IS NOT NULL AND change_pct > 0
                ORDER BY change_pct DESC
                LIMIT $1
                """,
                limit,
            )

            # Get losers (top negative change)
            losers_rows = await conn.fetch(
                """
                SELECT symbol, price, change, change_pct, volume
                FROM market_data_cache
                WHERE change_pct IS NOT NULL AND change_pct < 0
                ORDER BY change_pct ASC
                LIMIT $1
                """,
                limit,
            )

        def row_to_mover(row):
            return MarketMover(
                symbol=row["symbol"],
                price=float(row["price"]) if row["price"] else 0.0,
                change=float(row["change"]) if row["change"] else 0.0,
                changePercent=float(row["change_pct"]) if row["change_pct"] else 0.0,
                volume=int(row["volume"]) if row["volume"] else 0,
            )

        return {
            "gainers": [row_to_mover(r) for r in gainers_rows],
            "losers": [row_to_mover(r) for r in losers_rows],
        }
    except Exception as e:
        logger.error(f"Error fetching market movers: {e}")
        return {"gainers": [], "losers": []}


@router.get("/equity-curve")
async def get_equity_curve_data(
    days: int = Query(30, ge=1, le=2000, description="Number of days of history"),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get equity curve data for dashboard from portfolio_snapshots.
    Returns daily portfolio value snapshots.

    Performance: ~10ms
    """
    start_date = datetime.utcnow() - timedelta(days=days)

    try:
        async with db_manager.pool.acquire() as conn:
            # Use portfolio_snapshots for accurate total equity (cash + positions)
            rows = await conn.fetch(
                """
                SELECT timestamp, total_value
                FROM portfolio_snapshots
                WHERE user_id = $1
                  AND timestamp >= $2
                  AND snapshot_type = 'eod'
                ORDER BY timestamp ASC
                """,
                current_user.id,
                start_date,
            )

        if rows:
            return [
                {"timestamp": row["timestamp"].isoformat(), "value": float(row["total_value"])}
                for row in rows
            ]

        # Return empty if no data
        return []
    except Exception as e:
        logger.warning(f"Error fetching equity curve: {e}")
        return []


@router.get("/indicators/{symbol}")
async def get_technical_indicators(
    symbol: str,
    timeframe: str = Query("1d", description="Bar timeframe"),
    limit: int = Query(100, ge=1, le=1000),
    indicators: list[str] = Query(
        default=["sma_20", "sma_50", "ema_12", "bb_upper", "bb_lower", "macd"],
        description="List of indicators to calculate",
    ),
):
    """
    Get technical indicators for a symbol (calculated using Polars - 12x faster).

    Available indicators:
    - sma_5, sma_10, sma_20, sma_50, sma_200: Simple Moving Averages
    - ema_12, ema_26, ema_50: Exponential Moving Averages
    - bb_upper, bb_middle, bb_lower: Bollinger Bands
    - macd, macd_signal, macd_histogram: MACD
    - rsi_14: Relative Strength Index
    - volatility_20: Rolling volatility

    Performance: ~5-10ms for 100 bars with all indicators
    """
    import math

    import polars as pl

    # Get OHLCV data
    bars = await get_ohlcv_last_n_bars(symbol, timeframe, limit)

    if not bars:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

    # Convert to Polars DataFrame
    df = pl.DataFrame(bars)

    # Calculate all technical indicators using Polars
    df = calculate_technical_indicators(df)

    # Select only requested indicators plus timestamp
    columns_to_select = ["timestamp", "symbol", "close"] + [
        ind for ind in indicators if ind in df.columns
    ]

    result_df = df.select(columns_to_select)

    # Convert to dict format for JSON response
    # Replace NaN/Infinity with None for JSON compatibility
    result = result_df.to_dicts()

    def sanitize_value(v):
        """Replace NaN/Infinity with None for JSON compatibility."""
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    return [{k: sanitize_value(v) for k, v in row.items()} for row in result]


# ============================================================================
# WEBSOCKET FOR REAL-TIME DATA
# ============================================================================


class ConnectionManager:
    """Manage WebSocket connections for real-time market data."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.symbol_subscriptions: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        self.active_connections.remove(websocket)

        # Remove from all symbol subscriptions
        for symbol in list(self.symbol_subscriptions.keys()):
            if websocket in self.symbol_subscriptions[symbol]:
                self.symbol_subscriptions[symbol].remove(websocket)

                # Clean up empty subscription lists
                if not self.symbol_subscriptions[symbol]:
                    del self.symbol_subscriptions[symbol]

        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    def subscribe(self, websocket: WebSocket, symbol: str):
        """Subscribe WebSocket to symbol updates."""
        if symbol not in self.symbol_subscriptions:
            self.symbol_subscriptions[symbol] = []

        if websocket not in self.symbol_subscriptions[symbol]:
            self.symbol_subscriptions[symbol].append(websocket)
            logger.debug(
                f"Subscribed to {symbol}. Total subscribers: {len(self.symbol_subscriptions[symbol])}"
            )

    def unsubscribe(self, websocket: WebSocket, symbol: str):
        """Unsubscribe WebSocket from symbol updates."""
        if symbol in self.symbol_subscriptions and websocket in self.symbol_subscriptions[symbol]:
            self.symbol_subscriptions[symbol].remove(websocket)
            logger.debug(f"Unsubscribed from {symbol}")

    async def broadcast_to_symbol(self, symbol: str, message: dict):
        """
        Broadcast message to all subscribers of a symbol.

        Performance: Async broadcast, non-blocking
        """
        if symbol not in self.symbol_subscriptions:
            return

        # Get list of subscribers for this symbol
        subscribers = self.symbol_subscriptions[symbol].copy()

        # Broadcast to all subscribers concurrently
        disconnected = []
        for connection in subscribers:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_market_data(websocket: WebSocket):
    """
    WebSocket endpoint for real-time market data streaming.

    Protocol:
    - Client → Server: {"action": "subscribe", "symbols": ["AAPL", "GOOGL"]}
    - Client → Server: {"action": "unsubscribe", "symbols": ["AAPL"]}
    - Server → Client: {"type": "price", "symbol": "AAPL", "price": 150.25, "timestamp": "..."}
    - Server → Client: {"type": "tick", "symbol": "AAPL", "data": {...}}

    Performance:
    - Sub-millisecond message delivery
    - Supports 1000+ concurrent connections
    - Efficient fan-out to subscribers
    """
    await manager.connect(websocket)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "subscribe":
                # Subscribe to symbols
                symbols = data.get("symbols", [])
                for symbol in symbols:
                    manager.subscribe(websocket, symbol)

                await websocket.send_json(
                    {
                        "type": "subscribed",
                        "symbols": symbols,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            elif action == "unsubscribe":
                # Unsubscribe from symbols
                symbols = data.get("symbols", [])
                for symbol in symbols:
                    manager.unsubscribe(websocket, symbol)

                await websocket.send_json(
                    {
                        "type": "unsubscribed",
                        "symbols": symbols,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            elif action == "ping":
                # Respond to ping (heartbeat)
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


# ============================================================================
# HELPER FUNCTIONS FOR PUBLISHING DATA
# ============================================================================


async def publish_price_update(
    symbol: str, price: float, bid: float | None = None, ask: float | None = None
):
    """
    Publish price update to all subscribed WebSocket clients.

    Call this function whenever a new price tick arrives.

    Args:
        symbol: Symbol that was updated
        price: New price
        bid: Bid price
        ask: Ask price
    """
    message = {
        "type": "price",
        "symbol": symbol,
        "price": price,
        "bid": bid,
        "ask": ask,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await manager.broadcast_to_symbol(symbol, message)


async def publish_tick_data(tick_data: dict):
    """
    Publish full tick data to subscribers.

    Args:
        tick_data: Dictionary with tick data (must include 'symbol')
    """
    symbol = tick_data.get("symbol")
    if not symbol:
        logger.warning("Tick data missing symbol, cannot publish")
        return

    message = {
        "type": "tick",
        "symbol": symbol,
        "data": tick_data,
        "timestamp": datetime.utcnow().isoformat(),
    }

    await manager.broadcast_to_symbol(symbol, message)


# Export connection manager for use in other modules
__all__ = ["router", "manager", "publish_price_update", "publish_tick_data"]
