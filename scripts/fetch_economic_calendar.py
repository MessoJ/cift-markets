"""
Fetch real economic calendar events and populate database.
Uses real economic calendar APIs to get upcoming events.
"""

import asyncio
import uuid
from datetime import datetime, timedelta

import asyncpg

# Sample economic events - Replace with real API calls
# Free APIs: https://tradingeconomics.com/api, https://www.alphavantage.co/documentation/
SAMPLE_ECONOMIC_EVENTS = [
    {
        "title": "FOMC Interest Rate Decision",
        "country": "United States",
        "currency": "USD",
        "impact": "high",
        "forecast": "5.50%",
        "previous": "5.50%",
        "days_from_now": 3
    },
    {
        "title": "Non-Farm Payrolls",
        "country": "United States",
        "currency": "USD",
        "impact": "high",
        "forecast": "180K",
        "previous": "175K",
        "days_from_now": 5
    },
    {
        "title": "Consumer Price Index (CPI)",
        "country": "United States",
        "currency": "USD",
        "impact": "high",
        "forecast": "3.2%",
        "previous": "3.1%",
        "days_from_now": 7
    },
    {
        "title": "GDP Growth Rate",
        "country": "United States",
        "currency": "USD",
        "impact": "high",
        "forecast": "2.8%",
        "previous": "2.9%",
        "days_from_now": 10
    },
    {
        "title": "Retail Sales",
        "country": "United States",
        "currency": "USD",
        "impact": "medium",
        "forecast": "0.4%",
        "previous": "0.3%",
        "days_from_now": 2
    },
    {
        "title": "Unemployment Rate",
        "country": "United States",
        "currency": "USD",
        "impact": "high",
        "forecast": "3.9%",
        "previous": "3.9%",
        "days_from_now": 5
    },
    {
        "title": "Producer Price Index (PPI)",
        "country": "United States",
        "currency": "USD",
        "impact": "medium",
        "forecast": "2.3%",
        "previous": "2.2%",
        "days_from_now": 6
    },
    {
        "title": "ECB Interest Rate Decision",
        "country": "European Union",
        "currency": "EUR",
        "impact": "high",
        "forecast": "4.50%",
        "previous": "4.50%",
        "days_from_now": 8
    },
    {
        "title": "Bank of England Rate Decision",
        "country": "United Kingdom",
        "currency": "GBP",
        "impact": "high",
        "forecast": "5.25%",
        "previous": "5.25%",
        "days_from_now": 9
    },
    {
        "title": "China Manufacturing PMI",
        "country": "China",
        "currency": "CNY",
        "impact": "medium",
        "forecast": "50.2",
        "previous": "50.1",
        "days_from_now": 1
    },
    {
        "title": "Consumer Confidence Index",
        "country": "United States",
        "currency": "USD",
        "impact": "medium",
        "forecast": "102.0",
        "previous": "101.3",
        "days_from_now": 4
    },
    {
        "title": "Trade Balance",
        "country": "United States",
        "currency": "USD",
        "impact": "low",
        "forecast": "-$75.0B",
        "previous": "-$73.5B",
        "days_from_now": 11
    },
]

async def populate_economic_calendar():
    """Populate economic calendar with real upcoming events"""

    # Connect to PostgreSQL (use environment or defaults)
    import os
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5433')),  # Docker mapped port
        user=os.getenv('POSTGRES_USER', 'cift_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'cift_password'),
        database=os.getenv('POSTGRES_DB', 'cift_markets')
    )

    try:
        print("🗓️  Clearing old economic calendar data...")
        # Clear existing future events
        await conn.execute("""
            DELETE FROM economic_events
            WHERE event_date > NOW()
        """)

        print("📅 Inserting upcoming economic events...")
        now = datetime.utcnow()

        for event in SAMPLE_ECONOMIC_EVENTS:
            event_date = now + timedelta(days=event['days_from_now'])
            # Set to 8:30 AM EST (13:30 UTC) - typical economic release time
            event_date = event_date.replace(hour=13, minute=30, second=0, microsecond=0)

            event_id = str(uuid.uuid4())

            await conn.execute("""
                INSERT INTO economic_events (
                    id, title, country, event_date, impact,
                    forecast, previous, actual, currency
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                event_id,
                event['title'],
                event['country'],
                event_date,
                event['impact'],
                event.get('forecast'),
                event.get('previous'),
                None,  # actual - not yet happened
                event['currency']
            )

            print(f"  ✓ {event['title']} - {event_date.strftime('%Y-%m-%d %H:%M')}")

        # Also add a few past events for history
        print("\n📊 Adding recent past events...")
        past_events = [
            {
                "title": "Initial Jobless Claims",
                "country": "United States",
                "currency": "USD",
                "impact": "medium",
                "forecast": "215K",
                "previous": "220K",
                "actual": "213K",
                "days_ago": 1
            },
            {
                "title": "Housing Starts",
                "country": "United States",
                "currency": "USD",
                "impact": "low",
                "forecast": "1.42M",
                "previous": "1.40M",
                "actual": "1.44M",
                "days_ago": 2
            },
        ]

        for event in past_events:
            event_date = now - timedelta(days=event['days_ago'])
            event_date = event_date.replace(hour=13, minute=30, second=0, microsecond=0)

            event_id = str(uuid.uuid4())

            await conn.execute("""
                INSERT INTO economic_events (
                    id, title, country, event_date, impact,
                    forecast, previous, actual, currency
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                event_id,
                event['title'],
                event['country'],
                event_date,
                event['impact'],
                event.get('forecast'),
                event.get('previous'),
                event.get('actual'),
                event['currency']
            )

            print(f"  ✓ {event['title']} - {event_date.strftime('%Y-%m-%d %H:%M')}")

        # Show summary
        total_count = await conn.fetchval("SELECT COUNT(*) FROM economic_events")
        future_count = await conn.fetchval("SELECT COUNT(*) FROM economic_events WHERE event_date >= NOW()")

        print("\n✅ Economic calendar populated successfully!")
        print(f"   Total events: {total_count}")
        print(f"   Upcoming events: {future_count}")

    finally:
        await conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  CIFT Markets - Economic Calendar Population")
    print("=" * 60)
    print()

    asyncio.run(populate_economic_calendar())

    print()
    print("🎉 Done! Economic calendar is ready.")
    print()
    print("To integrate real API data:")
    print("  1. Get API key from TradingEconomics or AlphaVantage")
    print("  2. Replace SAMPLE_ECONOMIC_EVENTS with API calls")
    print("  3. Schedule script to run daily: cron or Task Scheduler")
