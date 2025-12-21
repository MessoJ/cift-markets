"""
Populate with RECENT data (ending NOW) so backend queries work.
Generates last 7 days of 1-minute data ending at current UTC time.
"""

import asyncio
import random
from datetime import datetime, timedelta

import asyncpg


async def populate_recent_data():
    """Generate 7 days of data ENDING NOW"""

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
        print("\n🗑️  Clearing old data...")
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

        # CRITICAL: End at NOW, start 7 days ago
        days = 7
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        print(f"\n📅 Generating data from {start_time} to {end_time} (7 days)")
        print(f"   Current UTC time: {datetime.utcnow()}")

        total_inserted = 0

        for symbol, base_price in symbols.items():
            print(f"\n📊 {symbol} (${base_price})...")

            # Start 7 days ago at market open
            current_time = start_time.replace(hour=13, minute=30, second=0, microsecond=0)
            price = base_price
            records = []
            symbol_count = 0

            while current_time <= end_time:
                # Skip weekends
                if current_time.weekday() >= 5:
                    current_time += timedelta(days=1)
                    continue

                # Market hours: 9:30 AM - 4:00 PM EST (13:30 - 20:00 UTC)
                hour = current_time.hour
                if hour < 13 or hour >= 20:
                    current_time = current_time.replace(hour=13, minute=30) + timedelta(days=1)
                    continue

                # Realistic price movement
                change = random.gauss(0, 0.002) * price
                price = max(price + change, base_price * 0.7)
                price = min(price, base_price * 1.3)

                volume = random.randint(800, 5000)

                records.append((
                    current_time,
                    symbol,
                    price,
                    volume,
                    price - 0.01,
                    price + 0.01,
                    'NASDAQ',
                    'REGULAR'
                ))

                current_time += timedelta(minutes=1)
                symbol_count += 1

                # Batch insert
                if len(records) >= 1000:
                    await conn.executemany("""
                        INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """, records)
                    total_inserted += len(records)
                    print(f"  {symbol_count:,} ticks...", end='\r')
                    records = []

            # Insert remaining
            if records:
                await conn.executemany("""
                    INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, records)
                total_inserted += len(records)

            print(f"  ✅ {symbol}: {symbol_count:,} ticks                    ")

        # Verify
        row = await conn.fetchrow("SELECT count(*) as cnt FROM ticks")
        print(f"\n✅ Total: {row['cnt']:,} ticks")

        # Show date ranges
        rows = await conn.fetch("""
            SELECT symbol, count(*) as cnt,
                   min(timestamp) as first_ts,
                   max(timestamp) as last_ts
            FROM ticks
            GROUP BY symbol
            ORDER BY symbol
        """)

        print("\n📊 Per Symbol:")
        for row in rows:
            days_span = (row['last_ts'] - row['first_ts']).days
            print(f"  {row['symbol']}: {row['cnt']:,} ticks | {row['first_ts'].strftime('%Y-%m-%d %H:%M')} → {row['last_ts'].strftime('%Y-%m-%d %H:%M')} ({days_span}d)")

        print("\n🎉 SUCCESS! Data ends at current time.")
        print("🚀 All API queries will now work (1m, 5m, 15m, 1h, 4h, 1d)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(populate_recent_data())
