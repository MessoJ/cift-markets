"""
Seed market_quotes data in QuestDB for testing the screener
"""
import asyncio
import asyncpg
from datetime import datetime

# Sample market data for all symbols in our database
MARKET_DATA = [
    # Technology
    ('AAPL', 182.45, 2.31, 1.28, 52340000),
    ('MSFT', 378.91, 4.23, 1.13, 19234000),
    ('GOOGL', 139.85, -1.23, -0.87, 28450000),
    ('NVDA', 495.22, 8.45, 1.74, 45670000),
    ('META', 342.78, 5.67, 1.68, 12340000),
    
    # Healthcare
    ('JNJ', 159.87, 0.45, 0.28, 6780000),
    ('UNH', 523.45, 3.21, 0.62, 2340000),
    ('PFE', 28.92, -0.34, -1.16, 34560000),
    
    # Financial
    ('JPM', 154.67, 1.23, 0.80, 8970000),
    ('BAC', 34.12, 0.23, 0.68, 45670000),
    ('V', 254.32, 2.10, 0.83, 5670000),
    
    # Consumer
    ('AMZN', 151.23, 2.34, 1.57, 34560000),
    ('TSLA', 242.84, -3.45, -1.40, 98760000),
    ('WMT', 162.45, 0.78, 0.48, 7890000),
    ('HD', 341.23, 1.56, 0.46, 3450000),
    
    # Energy
    ('XOM', 108.45, 1.23, 1.15, 15670000),
    ('CVX', 148.92, 1.67, 1.13, 8970000),
    
    # Industrial
    ('BA', 215.67, -2.34, -1.07, 4560000),
    ('CAT', 289.45, 3.12, 1.09, 2340000),
    
    # Materials
    ('LIN', 405.78, 2.45, 0.61, 1230000),
    
    # ETFs
    ('SPY', 456.78, 2.34, 0.51, 87600000),
    ('QQQ', 389.45, 3.21, 0.83, 45670000),
    ('IWM', 198.23, 0.98, 0.50, 23450000),
]


async def seed_market_data():
    """Insert market data into QuestDB"""
    try:
        # Connect to QuestDB via PostgreSQL wire protocol
        conn = await asyncpg.connect(
            host='localhost',
            port=8812,
            user='admin',
            password='quest',
            database='qdb'
        )
        
        print("✅ Connected to QuestDB")
        
        # Create table if not exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_quotes (
                symbol SYMBOL,
                price DOUBLE,
                change DOUBLE,
                change_percent DOUBLE,
                volume LONG,
                timestamp TIMESTAMP
            ) timestamp(timestamp) PARTITION BY DAY;
        """)
        print("✅ Table created/verified")
        
        # Insert data
        timestamp = datetime.now()
        for symbol, price, change, change_pct, volume in MARKET_DATA:
            await conn.execute(
                """
                INSERT INTO market_quotes (symbol, price, change, change_percent, volume, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                symbol, price, change, change_pct, volume, timestamp
            )
            print(f"✅ Inserted {symbol}: ${price}")
        
        print(f"\n🎉 Successfully seeded {len(MARKET_DATA)} stocks!")
        
        await conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure QuestDB is running:")
        print("   docker-compose up questdb")


if __name__ == "__main__":
    asyncio.run(seed_market_data())
