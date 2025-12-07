"""
Initialize QuestDB with market data schema and sample data.
"""

import asyncpg
import asyncio
from datetime import datetime, timedelta


async def init_questdb():
    """Initialize QuestDB schema and sample data."""
    
    # Connect to QuestDB via PostgreSQL wire protocol
    conn = await asyncpg.connect(
        host='localhost',
        port=8812,
        user='admin',
        password='quest',
        database='qdb'
    )
    
    try:
        print("✅ Connected to QuestDB")
        
        # Create ticks table
        print("Creating ticks table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                timestamp TIMESTAMP,
                symbol SYMBOL CAPACITY 500 CACHE,
                price DOUBLE,
                volume LONG,
                bid DOUBLE,
                ask DOUBLE,
                exchange SYMBOL CAPACITY 50 CACHE,
                conditions SYMBOL CAPACITY 100 CACHE
            ) TIMESTAMP(timestamp) PARTITION BY DAY WAL;
        """)
        print("✅ ticks table created")
        
        # Insert sample data for AAPL (30 days)
        print("Inserting sample AAPL data...")
        for i in range(30, -1, -1):
            timestamp = datetime.utcnow() - timedelta(days=i)
            price = 150.0 + (i * 0.67)  # Trending price
            volume = 1000 + (i * 100)
            
            await conn.execute("""
                INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, timestamp, 'AAPL', price, volume, price - 0.02, price + 0.02, 'NASDAQ', 'REGULAR')
        
        print(f"✅ Inserted 31 AAPL data points")
        
        # Insert sample data for MSFT
        print("Inserting sample MSFT data...")
        for i in range(30, -1, -1):
            timestamp = datetime.utcnow() - timedelta(days=i)
            price = 300.0 + (i * 1.67)
            volume = 800 + (i * 50)
            
            await conn.execute("""
                INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, timestamp, 'MSFT', price, volume, price - 0.05, price + 0.05, 'NASDAQ', 'REGULAR')
        
        print(f"✅ Inserted 31 MSFT data points")
        
        # Insert current data for other symbols
        symbols = ['GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'SPY', 'QQQ']
        prices = [2800.0, 3100.0, 250.0, 320.0, 450.0, 180.0, 450.0, 380.0]
        
        print(f"Inserting data for {len(symbols)} additional symbols...")
        now = datetime.utcnow()
        for symbol, price in zip(symbols, prices):
            volume = 500 + (hash(symbol) % 2000)
            await conn.execute("""
                INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, now, symbol, price, volume, price - 0.10, price + 0.10, 'NASDAQ', 'REGULAR')
        
        print(f"✅ Inserted data for {len(symbols)} symbols")
        
        # Verify data
        row = await conn.fetchrow("SELECT count(*) as cnt FROM ticks")
        print(f"\n✅ Total rows in ticks table: {row['cnt']}")
        
        # Show sample
        rows = await conn.fetch("SELECT symbol, count(*) as cnt FROM ticks GROUP BY symbol ORDER BY symbol")
        print("\n📊 Data by symbol:")
        for row in rows:
            print(f"  {row['symbol']}: {row['cnt']} ticks")
        
        print("\n✅ QuestDB initialization complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(init_questdb())
