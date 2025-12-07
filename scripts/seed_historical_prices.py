#!/usr/bin/env python3
"""
CIFT Markets - Historical Price Data Seeder

This script generates realistic historical OHLCV data for testing and development.
For production, you would use a data provider like Polygon.io, Alpha Vantage, or Yahoo Finance.

Data Providers (require API keys):
- Polygon.io: https://polygon.io/ - Best quality, requires paid plan for real-time
- Alpha Vantage: https://www.alphavantage.co/ - Free tier: 5 API calls/min
- Yahoo Finance: Free but unofficial API, use yfinance library
- Finnhub: https://finnhub.io/ - Free tier available

Usage:
    python scripts/seed_historical_prices.py [--use-yahoo] [--symbols AAPL,MSFT,GOOGL]
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Base prices for common stocks (approximate as of Dec 2025)
BASE_PRICES = {
    'AAPL': 189.95,
    'MSFT': 415.50,
    'GOOGL': 141.80,
    'AMZN': 178.25,
    'NVDA': 875.25,
    'TSLA': 248.75,
    'META': 505.40,
    'JPM': 198.50,
    'V': 275.80,
    'UNH': 525.40,
    'SPY': 478.25,
    'QQQ': 405.82,
    'DIA': 385.60,
    'IWM': 198.40,
    'AMD': 142.50,
    'CRM': 265.80,
    'NFLX': 485.20,
    'INTC': 45.60,
    'COIN': 178.50,
    'PLTR': 25.40,
}

# Volatility (daily % standard deviation) by symbol
VOLATILITY = {
    'AAPL': 0.018, 'MSFT': 0.016, 'GOOGL': 0.020, 'AMZN': 0.022, 'NVDA': 0.035,
    'TSLA': 0.045, 'META': 0.028, 'JPM': 0.015, 'V': 0.014, 'UNH': 0.016,
    'SPY': 0.010, 'QQQ': 0.014, 'DIA': 0.009, 'IWM': 0.016, 'AMD': 0.038,
    'CRM': 0.025, 'NFLX': 0.030, 'INTC': 0.028, 'COIN': 0.055, 'PLTR': 0.048,
}


def generate_ohlcv_bars(
    symbol: str,
    base_price: float,
    volatility: float,
    n_bars: int,
    timeframe: str = '1m',
    end_time: datetime = None
) -> List[Dict[str, Any]]:
    """
    Generate realistic OHLCV bars using random walk with mean reversion.
    
    Args:
        symbol: Stock symbol
        base_price: Starting price
        volatility: Daily volatility (std dev)
        n_bars: Number of bars to generate
        timeframe: Bar timeframe (1m, 5m, 15m, 1h, 1d)
        end_time: End timestamp
        
    Returns:
        List of OHLCV bar dictionaries
    """
    if end_time is None:
        end_time = datetime.utcnow()
    
    # Time delta per bar
    timeframe_deltas = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1),
    }
    delta = timeframe_deltas.get(timeframe, timedelta(minutes=1))
    
    # Scale volatility based on timeframe
    vol_scaling = {
        '1m': 0.05, '5m': 0.12, '15m': 0.22, '30m': 0.32,
        '1h': 0.45, '4h': 0.80, '1d': 1.0,
    }
    vol = volatility * vol_scaling.get(timeframe, 0.05)
    
    bars = []
    price = base_price
    mean_price = base_price
    
    for i in range(n_bars):
        timestamp = end_time - delta * (n_bars - 1 - i)
        
        # Skip weekends for daily bars
        if timeframe == '1d' and timestamp.weekday() >= 5:
            continue
        
        # Skip non-market hours for intraday bars
        if timeframe in ['1m', '5m', '15m', '30m', '1h']:
            hour = timestamp.hour
            if hour < 9 or hour >= 16:  # Simplified market hours
                continue
        
        # Random return with mean reversion
        mean_reversion = 0.01 * (mean_price - price) / mean_price
        random_return = random.gauss(mean_reversion, vol)
        
        # Generate OHLC
        open_price = price
        change = open_price * random_return
        close_price = open_price + change
        
        # High/Low within the bar
        intrabar_vol = vol * 0.5
        high_deviation = abs(random.gauss(0, intrabar_vol)) * open_price
        low_deviation = abs(random.gauss(0, intrabar_vol)) * open_price
        
        high_price = max(open_price, close_price) + high_deviation
        low_price = min(open_price, close_price) - low_deviation
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Generate volume (higher for volatile moves)
        base_volume = 1000000 if symbol in ['AAPL', 'TSLA', 'NVDA', 'SPY'] else 500000
        volume = int(base_volume * (1 + abs(random_return) * 20) * random.uniform(0.5, 1.5))
        
        bars.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume,
        })
        
        price = close_price
    
    return bars


async def seed_to_questdb(bars: List[Dict[str, Any]]):
    """Seed bars to QuestDB ticks table."""
    from cift.core.database import questdb_manager
    
    # Create table if not exists
    create_table = """
        CREATE TABLE IF NOT EXISTS ticks (
            timestamp TIMESTAMP,
            symbol SYMBOL,
            price DOUBLE,
            volume LONG,
            bid DOUBLE,
            ask DOUBLE
        ) timestamp(timestamp) PARTITION BY DAY;
    """
    
    try:
        await questdb_manager.execute(create_table)
        print("✅ Created ticks table in QuestDB")
    except Exception as e:
        print(f"Table may already exist: {e}")
    
    # Insert data
    for bar in bars:
        # Insert as ticks (using close price)
        insert = """
            INSERT INTO ticks (timestamp, symbol, price, volume, bid, ask)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        spread = bar['close'] * 0.0005  # 5 basis point spread
        await questdb_manager.execute(
            insert,
            bar['timestamp'],
            bar['symbol'],
            bar['close'],
            bar['volume'],
            bar['close'] - spread / 2,
            bar['close'] + spread / 2,
        )
    
    print(f"✅ Inserted {len(bars)} ticks to QuestDB")


async def seed_to_postgres(bars: List[Dict[str, Any]]):
    """Seed bars to PostgreSQL ohlcv_bars table as fallback."""
    from cift.core.database import db_manager
    
    async with db_manager.pool.acquire() as conn:
        # Create table if not exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_bars (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                symbol VARCHAR(10) NOT NULL,
                timeframe VARCHAR(10) NOT NULL DEFAULT '1m',
                open NUMERIC(15, 4) NOT NULL,
                high NUMERIC(15, 4) NOT NULL,
                low NUMERIC(15, 4) NOT NULL,
                close NUMERIC(15, 4) NOT NULL,
                volume BIGINT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_ts 
            ON ohlcv_bars(symbol, timeframe, timestamp DESC);
        """)
        
        # Clear existing data for symbols
        symbols = list(set(bar['symbol'] for bar in bars))
        await conn.execute(
            "DELETE FROM ohlcv_bars WHERE symbol = ANY($1::text[])",
            symbols
        )
        
        # Insert new data
        for bar in bars:
            await conn.execute("""
                INSERT INTO ohlcv_bars (timestamp, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, '1m', $3, $4, $5, $6, $7)
            """, bar['timestamp'], bar['symbol'], bar['open'], bar['high'], bar['low'], bar['close'], bar['volume'])
        
        print(f"✅ Inserted {len(bars)} bars to PostgreSQL ohlcv_bars table")


async def main():
    """Generate and seed historical price data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed historical price data')
    parser.add_argument('--symbols', default='AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,SPY,QQQ',
                       help='Comma-separated symbols')
    parser.add_argument('--days', type=int, default=30, help='Days of history')
    parser.add_argument('--timeframe', default='1m', help='Bar timeframe')
    parser.add_argument('--target', default='postgres', choices=['questdb', 'postgres', 'both'],
                       help='Where to store data')
    parser.add_argument('--use-yahoo', action='store_true', help='Use Yahoo Finance for real data')
    
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    print(f"🚀 Generating {args.days} days of {args.timeframe} data for: {', '.join(symbols)}")
    
    # Calculate number of bars
    bars_per_day = {
        '1m': 390,  # 6.5 hours * 60
        '5m': 78,
        '15m': 26,
        '30m': 13,
        '1h': 7,
        '4h': 2,
        '1d': 1,
    }
    n_bars = args.days * bars_per_day.get(args.timeframe, 390)
    
    all_bars = []
    
    if args.use_yahoo:
        try:
            import yfinance as yf
            print("📊 Fetching real data from Yahoo Finance...")
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=f"{args.days}d", interval=args.timeframe)
                
                for idx, row in hist.iterrows():
                    all_bars.append({
                        'timestamp': idx.to_pydatetime(),
                        'symbol': symbol,
                        'open': round(row['Open'], 2),
                        'high': round(row['High'], 2),
                        'low': round(row['Low'], 2),
                        'close': round(row['Close'], 2),
                        'volume': int(row['Volume']),
                    })
                print(f"  ✓ {symbol}: {len(hist)} bars")
                
        except ImportError:
            print("⚠️ yfinance not installed. Run: pip install yfinance")
            print("   Falling back to generated data...")
            args.use_yahoo = False
    
    if not args.use_yahoo:
        print("📊 Generating synthetic price data...")
        for symbol in symbols:
            base_price = BASE_PRICES.get(symbol, 100.0)
            volatility = VOLATILITY.get(symbol, 0.02)
            
            bars = generate_ohlcv_bars(
                symbol=symbol,
                base_price=base_price,
                volatility=volatility,
                n_bars=n_bars,
                timeframe=args.timeframe,
            )
            all_bars.extend(bars)
            print(f"  ✓ {symbol}: {len(bars)} bars")
    
    print(f"\n📈 Total bars generated: {len(all_bars)}")
    
    # Initialize database connections
    from cift.core.database import db_manager
    await db_manager.initialize()
    
    try:
        if args.target in ['postgres', 'both']:
            await seed_to_postgres(all_bars)
        
        if args.target in ['questdb', 'both']:
            from cift.core.database import questdb_manager
            await questdb_manager.initialize()
            await seed_to_questdb(all_bars)
            
    finally:
        await db_manager.close()
    
    print("\n✅ Historical price data seeding complete!")
    print(f"\nTo use real market data, get an API key from:")
    print("  - Polygon.io: https://polygon.io/")
    print("  - Alpha Vantage: https://www.alphavantage.co/")
    print("  - Yahoo Finance (free): pip install yfinance")


if __name__ == "__main__":
    asyncio.run(main())
