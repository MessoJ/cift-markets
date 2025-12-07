"""
Generate market data ending TODAY (for testing)
"""
import requests
from datetime import datetime, timedelta
import random

QUESTDB_URL = "http://localhost:9000/exec"
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD']

def clear_data():
    """Clear existing data"""
    try:
        response = requests.get(f"{QUESTDB_URL}?query=DELETE FROM ticks;")
        print("✅ Cleared old data")
    except Exception as e:
        print(f"⚠️  Could not clear: {e}")

def generate_ticks():
    """Generate 7 days of 1-minute ticks ending TODAY"""
    
    # End time: NOW (rounded to last minute)
    end_time = datetime.now().replace(second=0, microsecond=0)
    
    # Start time: 7 days ago
    start_time = end_time - timedelta(days=7)
    
    print(f"📊 Generating data from {start_time} to {end_time}")
    
    # Base prices for each symbol
    base_prices = {
        'AAPL': 170.0,
        'MSFT': 380.0,
        'GOOGL': 140.0,
        'AMZN': 150.0,
        'TSLA': 240.0,
        'META': 320.0,
        'NVDA': 480.0,
        'AMD': 130.0,
    }
    
    total_ticks = 0
    batch_size = 100
    batch = []
    
    for symbol in SYMBOLS:
        price = base_prices[symbol]
        current = start_time
        
        while current <= end_time:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(minutes=1)
                continue
            
            # Market hours only (9:30 AM - 4:00 PM EST = 13:30 - 20:00 UTC)
            hour = current.hour
            minute = current.minute
            
            # Convert EST to UTC (add 5 hours for EST to UTC)
            # Market: 09:30-16:00 EST = 14:30-21:00 UTC
            if hour < 14 or hour >= 21:
                current += timedelta(minutes=1)
                continue
            
            if hour == 14 and minute < 30:
                current += timedelta(minutes=1)
                continue
            
            # Random price movement
            price_change = random.uniform(-0.5, 0.5)
            price = max(1.0, price + price_change)
            
            open_price = price
            high_price = price + random.uniform(0, 0.3)
            low_price = price - random.uniform(0, 0.3)
            close_price = price + random.uniform(-0.2, 0.2)
            volume = random.randint(100000, 500000)
            
            # Format timestamp for QuestDB
            ts = current.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
            
            # Build INSERT statement
            insert = f"INSERT INTO ticks VALUES('{symbol}', '{ts}', {open_price:.2f}, {high_price:.2f}, {low_price:.2f}, {close_price:.2f}, {volume});"
            batch.append(insert)
            total_ticks += 1
            
            # Send batch
            if len(batch) >= batch_size:
                query = ' '.join(batch)
                try:
                    requests.get(f"{QUESTDB_URL}?query={query}")
                    print(f"  {symbol}: {total_ticks} ticks", end='\r')
                except Exception as e:
                    print(f"\n❌ Batch failed: {e}")
                batch = []
            
            current += timedelta(minutes=1)
        
        # Send remaining batch
        if batch:
            query = ' '.join(batch)
            try:
                requests.get(f"{QUESTDB_URL}?query={query}")
            except Exception as e:
                print(f"❌ Final batch failed: {e}")
            batch = []
        
        print(f"✅ {symbol}: Done")
    
    print(f"\n🎉 Generated {total_ticks} total ticks")
    print(f"📅 Date range: {start_time.date()} to {end_time.date()}")
    print(f"⏰ Latest timestamp: {end_time}")

if __name__ == "__main__":
    print("=" * 50)
    print("POPULATE MARKET DATA - ENDING TODAY")
    print("=" * 50)
    
    clear_data()
    generate_ticks()
    
    print("\n✅ DONE! Chart should now show data up to NOW")
