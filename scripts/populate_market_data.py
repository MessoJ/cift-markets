"""
Populate QuestDB with realistic intraday market data for chart testing.
Generates multiple timeframes of data with realistic price movements.
"""

import asyncpg
import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Tuple


def generate_realistic_prices(
    start_price: float,
    num_points: int,
    volatility: float = 0.02,
    trend: float = 0.0001
) -> List[float]:
    """
    Generate realistic price movements using geometric Brownian motion.
    
    Args:
        start_price: Starting price
        num_points: Number of price points to generate
        volatility: Price volatility (standard deviation)
        trend: Drift/trend component
    """
    prices = [start_price]
    
    for i in range(num_points - 1):
        # Geometric Brownian motion
        random_shock = random.gauss(0, volatility)
        new_price = prices[-1] * (1 + trend + random_shock)
        prices.append(new_price)
    
    return prices


def generate_ohlc_from_price(
    price: float,
    volatility: float = 0.005
) -> Tuple[float, float, float, float]:
    """
    Generate realistic OHLC from a base price.
    
    Returns: (open, high, low, close)
    """
    # Generate 4 prices around the base price
    prices_in_bar = [price * (1 + random.gauss(0, volatility)) for _ in range(10)]
    
    open_price = prices_in_bar[0]
    close_price = prices_in_bar[-1]
    high_price = max(prices_in_bar)
    low_price = min(prices_in_bar)
    
    # Ensure OHLC validity
    high_price = max(high_price, open_price, close_price)
    low_price = min(low_price, open_price, close_price)
    
    return open_price, high_price, low_price, close_price


async def populate_intraday_data(conn, symbol: str, base_price: float, days: int = 180):
    """Populate with 1-minute intraday data."""
    print(f"\n📊 Generating {days} days of 1-minute data for {symbol}...")
    
    # Generate for market hours only (9:30 AM - 4:00 PM EST)
    minutes_per_day = 390  # 6.5 hours * 60 minutes
    total_minutes = days * minutes_per_day
    
    # Generate base price series
    base_prices = generate_realistic_prices(
        base_price,
        total_minutes,
        volatility=0.001,  # Lower volatility for 1-minute bars
        trend=0.00002
    )
    
    start_date = datetime.utcnow() - timedelta(days=days)
    current_time = start_date.replace(hour=13, minute=30, second=0, microsecond=0)  # 9:30 AM EST
    
    records = []
    for i, price in enumerate(base_prices):
        # Skip weekends
        if current_time.weekday() >= 5:
            current_time += timedelta(minutes=1)
            continue
        
        # Skip non-market hours
        hour = current_time.hour
        if hour < 13 or hour >= 20:  # Before 9:30 AM or after 4:00 PM EST
            current_time = current_time.replace(hour=13, minute=30) + timedelta(days=1)
            continue
        
        open_p, high, low, close = generate_ohlc_from_price(price, volatility=0.002)
        volume = random.randint(500, 5000)
        
        records.append((
            current_time,
            symbol,
            open_p,
            volume,
            open_p - 0.02,
            open_p + 0.02,
            'NASDAQ',
            'REGULAR'
        ))
        
        current_time += timedelta(minutes=1)
        
        # Batch insert every 1000 records
        if len(records) >= 1000:
            await conn.executemany("""
                INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, records)
            print(f"  Inserted {len(records)} records...")
            records = []
    
    # Insert remaining records
    if records:
        await conn.executemany("""
            INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask, exchange, conditions)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, records)
        print(f"  Inserted {len(records)} records...")
    
    print(f"✅ Completed {symbol}")


async def populate_all():
    """Populate QuestDB with comprehensive market data."""
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
        
        # Popular symbols with their base prices
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
        
        # Generate 180 days (6 months) of 1-minute data for each symbol
        # This gives us proper data density for all timeframes:
        # - 1m: ~390 bars per day (market hours)
        # - 5m: ~78 bars per day
        # - 1h: ~6.5 bars per day
        # - 1d: ~180 bars (6 months)
        for symbol, base_price in symbols.items():
            await populate_intraday_data(conn, symbol, base_price, days=180)
        
        # Verify
        row = await conn.fetchrow("SELECT count(*) as cnt FROM ticks")
        print(f"\n✅ Total ticks inserted: {row['cnt']:,}")
        
        # Show distribution
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
            print(f"  {row['symbol']}: {row['cnt']:,} ticks | {row['first_ts']} to {row['last_ts']}")
        
        print("\n✅ Market data population complete!")
        print("\n💡 You can now test charts with:")
        print("   - All timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
        print("   - 180 days (6 months) of historical data")
        print("   - 8 popular symbols")
        print("   - ~560,000 total ticks generated")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(populate_all())
