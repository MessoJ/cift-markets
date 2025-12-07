"""
Quick population script - 30 days of dense 1-minute data
Optimized for speed: generates ~300K ticks in under 2 minutes
"""

import asyncpg
import asyncio
import random
from datetime import datetime, timedelta
import sys


async def populate_quick():
    """Generate 30 days of 1-minute bar data (market hours only)"""
    
    conn = await asyncpg.connect(
        host='localhost',
        port=8812,
        user='admin',
        password='quest',
        database='qdb'
    )
    
    try:
        print("✅ Connected to QuestDB")
        
        # Clear existing data
        print("\n🗑️  Clearing existing data...")
        await conn.execute("TRUNCATE TABLE ticks")
        print("✅ Table cleared")
        
        symbols = {
            'AAPL': 170.0,
            'MSFT': 350.0,
            'GOOGL': 140.0,
            'AMZN': 155.0,
            'TSLA': 245.0,
            'META': 485.0,
            'NVDA': 485.0,
            'AMD': 140.0,
        }
        
        days = 30
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        total_inserted = 0
        
        for symbol, base_price in symbols.items():
            print(f"\n📊 Generating {symbol} ({base_price})...")
            
            current_time = start_date.replace(hour=13, minute=30, second=0, microsecond=0)
            price = base_price
            records = []
            symbol_count = 0
            
            while current_time <= end_date:
                # Skip weekends
                if current_time.weekday() >= 5:
                    current_time += timedelta(days=1)
                    continue
                
                # Market hours only (9:30 AM - 4:00 PM EST = 13:30 - 20:00 UTC)
                hour = current_time.hour
                if hour < 13 or hour >= 20:
                    current_time = current_time.replace(hour=13, minute=30) + timedelta(days=1)
                    continue
                
                # Generate realistic price movement
                change = random.gauss(0, 0.001) * price
                price = max(price + change, base_price * 0.5)  # Don't go below 50% of base
                
                volume = random.randint(500, 5000)
                
                records.append((
                    current_time,
                    symbol,
                    price,
                    volume,
                    price - 0.02,  # bid
                    price + 0.02,  # ask
                    'NASDAQ',
                    'REGULAR'
                ))
                
                current_time += timedelta(minutes=1)
                symbol_count += 1
                
                # Batch insert every 1000 records
                if len(records) >= 1000:
                    await conn.executemany("""
                        INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, records)
                    total_inserted += len(records)
                    print(f"  Progress: {symbol_count:,} ticks | Total: {total_inserted:,}", end='\r')
                    records = []
            
            # Insert remaining
            if records:
                await conn.executemany("""
                    INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, records)
                total_inserted += len(records)
            
            print(f"  ✅ {symbol}: {symbol_count:,} ticks")
        
        # Verify
        row = await conn.fetchrow("SELECT count(*) as cnt FROM ticks")
        print(f"\n✅ Total ticks inserted: {row['cnt']:,}")
        
        # Show summary
        rows = await conn.fetch("""
            SELECT symbol, count(*) as cnt,
                   min(timestamp) as first_ts,
                   max(timestamp) as last_ts
            FROM ticks
            GROUP BY symbol
            ORDER BY symbol
        """)
        
        print("\n📊 Data summary:")
        for row in rows:
            print(f"  {row['symbol']}: {row['cnt']:,} ticks | {row['first_ts'].strftime('%Y-%m-%d')} to {row['last_ts'].strftime('%Y-%m-%d')}")
        
        print("\n✅ Quick population complete!")
        print("\n💡 Chart data available:")
        print("   - 1m timeframe: ~390 bars per day (11,700 for 30 days)")
        print("   - 1h timeframe: ~6.5 bars per day (195 for 30 days)")
        print("   - 1d timeframe: 30 bars (1 month)")
        print("\n🚀 Ready to test charts!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(populate_quick())
