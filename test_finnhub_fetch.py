import asyncio
import os
from dotenv import load_dotenv
from cift.services.market_data_service import market_data_service
from cift.core.database import db_manager

# Load environment variables
load_dotenv(".env.production")

async def test_fetch():
    symbol = "AAPL"
    print(f"Testing fetch for {symbol}...")
    
    # Initialize database pool
    await db_manager.initialize()
    
    try:
        # Initialize service
        await market_data_service.initialize()
        
        # Check API key
        print(f"Finnhub API Key present: {bool(market_data_service.finnhub.api_key)}")
        if market_data_service.finnhub.api_key:
            print(f"API Key starts with: {market_data_service.finnhub.api_key[:4]}...")
        
        # Try to fetch and store
        count = await market_data_service.fetch_and_store_ohlcv(symbol, days=5, timeframe="1d")
        print(f"Fetched and stored {count} bars.")
        
        if count > 0:
            # Verify data in DB
            async with db_manager.pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT count(*) FROM ohlcv_bars WHERE symbol = $1 AND timeframe = '1d'",
                    symbol
                )
                print(f"Rows in DB: {row['count']}")
                
                # Show a sample
                sample = await conn.fetchrow(
                    "SELECT * FROM ohlcv_bars WHERE symbol = $1 AND timeframe = '1d' ORDER BY timestamp DESC LIMIT 1",
                    symbol
                )
                print(f"Sample bar: {sample}")
        else:
            print("No bars fetched. Check logs.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await market_data_service.close()
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(test_fetch())
