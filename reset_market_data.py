import asyncio
import os
import sys
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cift.core.database import db_manager
from cift.services.market_data_service import market_data_service

# Default symbols to pre-fetch
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", 
    "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD", 
    "MA", "COST", "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO"
]

async def reset_market_data():
    """
    Truncate market data tables and fetch fresh data from Finnhub.
    """
    logger.info("Starting market data reset...")
    
    # 1. Initialize database
    await db_manager.initialize()
    
    # 2. Truncate tables
    logger.info("Truncating database tables...")
    async with db_manager.pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE ohlcv_bars RESTART IDENTITY CASCADE;")
        await conn.execute("TRUNCATE TABLE market_data RESTART IDENTITY CASCADE;")
        await conn.execute("TRUNCATE TABLE market_data_cache RESTART IDENTITY CASCADE;")
        
    logger.success("Tables truncated.")

    # 3. Initialize Service
    logger.info("Initializing Market Data Service...")
    await market_data_service.initialize()
    
    # 4. Fetch fresh quotes for all symbols (uses Finnhub - FREE)
    logger.info(f"Fetching fresh quotes for {len(DEFAULT_SYMBOLS)} symbols...")
    quotes = await market_data_service.get_quotes_batch(DEFAULT_SYMBOLS)
    logger.info(f"Fetched {len(quotes)} quotes")
    
    # 5. Store quotes in cache
    async with db_manager.pool.acquire() as conn:
        for symbol, quote in quotes.items():
            try:
                await conn.execute(
                    """
                    INSERT INTO market_data_cache (symbol, price, bid, ask, volume, change, change_pct, high, low, open, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
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
                    symbol,
                    quote.get("price"),
                    quote.get("bid"),
                    quote.get("ask"),
                    quote.get("volume"),
                    quote.get("change"),
                    quote.get("change_pct"),
                    quote.get("high"),
                    quote.get("low"),
                    quote.get("open"),
                )
            except Exception as e:
                logger.error(f"Error caching quote for {symbol}: {e}")
    
    logger.success(f"Cached {len(quotes)} quotes in database")
    
    # 6. Fetch historical bars for a few key symbols (to speed up initial loads)
    key_symbols = ["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ"]
    logger.info(f"Fetching historical bars for {len(key_symbols)} key symbols...")
    
    for symbol in key_symbols:
        try:
            bars = await market_data_service.fetch_and_store_ohlcv(symbol, days=30, timeframe="1d")
            logger.info(f"  {symbol}: {bars} daily bars")
        except Exception as e:
            logger.error(f"  {symbol}: Failed - {e}")
    
    logger.success("Market data reset complete!")
    await market_data_service.close()

if __name__ == "__main__":
    asyncio.run(reset_market_data())
