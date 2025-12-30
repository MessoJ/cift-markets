#!/usr/bin/env python3
"""
SYMBOL DATA ENRICHMENT SCRIPT

Fetches comprehensive fundamental data from Finnhub and Yahoo Finance APIs
to populate the symbols table with:
- P/B, P/S ratios
- ROE, ROA, profit margins
- Analyst ratings and price targets
- 52-week high/low
- Volume data
- And more

Usage:
    python scripts/enrich_symbols_data.py
    
Or via Docker:
    docker exec cift-api python scripts/enrich_symbols_data.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
import aiohttp
import asyncpg

# Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d4ojf7pr01qtc1p01m60d4ojf7pr01qtc1p01m6g")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "cift-postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "cift_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "cift_secure_password_2024")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cift_markets")

# Rate limiting for Finnhub free tier (60 req/min)
RATE_LIMIT_DELAY = 1.1  # seconds between requests


async def get_finnhub_data(session: aiohttp.ClientSession, endpoint: str, params: dict = None) -> dict | None:
    """Fetch data from Finnhub API with rate limiting."""
    url = f"https://finnhub.io/api/v1/{endpoint}"
    params = params or {}
    params["token"] = FINNHUB_API_KEY
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 429:
                print(f"  ⚠️  Rate limited, waiting 60s...")
                await asyncio.sleep(60)
                return await get_finnhub_data(session, endpoint, params)
            else:
                print(f"  ❌ API error {response.status} for {endpoint}")
                return None
    except Exception as e:
        print(f"  ❌ Request error: {e}")
        return None


async def get_basic_financials(session: aiohttp.ClientSession, symbol: str) -> dict:
    """Get basic financials from Finnhub (P/B, P/S, ROE, margins, etc.)."""
    data = await get_finnhub_data(session, "stock/metric", {"symbol": symbol, "metric": "all"})
    
    if not data or "metric" not in data:
        return {}
    
    metrics = data["metric"]
    
    return {
        "price_to_book": metrics.get("pbQuarterly") or metrics.get("pbAnnual"),
        "price_to_sales": metrics.get("psQuarterly") or metrics.get("psTTM") or metrics.get("psAnnual"),
        "roe": metrics.get("roeTTM") or metrics.get("roeRfy"),
        "roa": metrics.get("roaTTM") or metrics.get("roaRfy"),
        "profit_margin": metrics.get("netProfitMarginTTM") or metrics.get("netProfitMarginAnnual"),
        "operating_margin": metrics.get("operatingMarginTTM") or metrics.get("operatingMarginAnnual"),
        "revenue": metrics.get("revenueTTM"),
        "net_income": metrics.get("netIncomeTTM"),
        "ebitda": metrics.get("ebitdaTTM"),
        "peg_ratio": metrics.get("pegRatio"),
        "forward_pe": metrics.get("forwardPE"),
        "52_week_high": metrics.get("52WeekHigh"),
        "52_week_low": metrics.get("52WeekLow"),
        "beta": metrics.get("beta"),
        "eps_ttm": metrics.get("epsTTM"),
        "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
    }


async def get_analyst_data(session: aiohttp.ClientSession, symbol: str) -> dict:
    """Get analyst recommendations and price targets."""
    # Get recommendation trends
    rec_data = await get_finnhub_data(session, "stock/recommendation", {"symbol": symbol})
    
    analyst_rating = None
    analyst_count = 0
    
    if rec_data and len(rec_data) > 0:
        latest = rec_data[0]
        # Calculate weighted rating (1=strong buy, 5=strong sell)
        total = (
            latest.get("strongBuy", 0) + 
            latest.get("buy", 0) + 
            latest.get("hold", 0) + 
            latest.get("sell", 0) + 
            latest.get("strongSell", 0)
        )
        if total > 0:
            weighted = (
                latest.get("strongBuy", 0) * 1 +
                latest.get("buy", 0) * 2 +
                latest.get("hold", 0) * 3 +
                latest.get("sell", 0) * 4 +
                latest.get("strongSell", 0) * 5
            ) / total
            
            # Convert to text rating
            if weighted <= 1.5:
                analyst_rating = "Strong Buy"
            elif weighted <= 2.5:
                analyst_rating = "Buy"
            elif weighted <= 3.5:
                analyst_rating = "Hold"
            elif weighted <= 4.5:
                analyst_rating = "Sell"
            else:
                analyst_rating = "Strong Sell"
            
            analyst_count = total
    
    await asyncio.sleep(RATE_LIMIT_DELAY)
    
    # Get price target
    target_data = await get_finnhub_data(session, "stock/price-target", {"symbol": symbol})
    
    analyst_target = None
    if target_data:
        analyst_target = target_data.get("targetMean") or target_data.get("targetMedian")
    
    return {
        "analyst_rating": analyst_rating,
        "analyst_target_price": analyst_target,
        "analyst_count": analyst_count if analyst_count > 0 else None,
    }


async def get_quote_data(session: aiohttp.ClientSession, symbol: str) -> dict:
    """Get current quote data including volume."""
    data = await get_finnhub_data(session, "quote", {"symbol": symbol})
    
    if not data:
        return {}
    
    return {
        "price": data.get("c"),
        "change": data.get("d"),
        "change_pct": data.get("dp"),
        "volume": data.get("v"),  # Note: v might not be in quote endpoint
        "high": data.get("h"),
        "low": data.get("l"),
        "open": data.get("o"),
        "prev_close": data.get("pc"),
    }


async def enrich_symbol(
    pool: asyncpg.Pool, 
    session: aiohttp.ClientSession, 
    symbol: str
) -> bool:
    """Enrich a single symbol with all available data."""
    print(f"\n📊 Enriching {symbol}...")
    
    # Fetch all data sources
    financials = await get_basic_financials(session, symbol)
    await asyncio.sleep(RATE_LIMIT_DELAY)
    
    analyst = await get_analyst_data(session, symbol)
    await asyncio.sleep(RATE_LIMIT_DELAY)
    
    quote = await get_quote_data(session, symbol)
    
    # Update symbols table
    try:
        async with pool.acquire() as conn:
            # Check what we have to update
            updates = []
            params = []
            param_num = 1
            
            field_mapping = {
                # From financials
                "price_to_book": financials.get("price_to_book"),
                "price_to_sales": financials.get("price_to_sales"),
                "roe": financials.get("roe"),
                "roa": financials.get("roa"),
                "profit_margin": financials.get("profit_margin"),
                "operating_margin": financials.get("operating_margin"),
                "revenue": financials.get("revenue"),
                "net_income": financials.get("net_income"),
                "ebitda": financials.get("ebitda"),
                "peg_ratio": financials.get("peg_ratio"),
                "forward_pe": financials.get("forward_pe"),
                # From analyst
                "analyst_rating": analyst.get("analyst_rating"),
                "analyst_target_price": analyst.get("analyst_target_price"),
                "analyst_count": analyst.get("analyst_count"),
            }
            
            # Also update dividend_yield and eps if we got better data
            if financials.get("dividend_yield"):
                field_mapping["dividend_yield"] = financials["dividend_yield"] / 100  # Convert from percentage
            if financials.get("eps_ttm"):
                field_mapping["eps"] = financials["eps_ttm"]
            
            for field, value in field_mapping.items():
                if value is not None:
                    updates.append(f"{field} = ${param_num}")
                    params.append(value)
                    param_num += 1
            
            if updates:
                updates.append(f"data_updated_at = ${param_num}")
                params.append(datetime.utcnow())
                param_num += 1
                
                params.append(symbol)  # For WHERE clause
                
                query = f"""
                    UPDATE symbols 
                    SET {", ".join(updates)}
                    WHERE symbol = ${param_num}
                """
                
                await conn.execute(query, *params)
                print(f"  ✅ Updated {len(updates)-1} fields for {symbol}")
            else:
                print(f"  ⚠️  No new data for {symbol}")
            
            # Update market_data_cache with 52-week data and volume
            if quote.get("price") or financials.get("52_week_high"):
                cache_updates = []
                cache_params = []
                cache_param_num = 1
                
                if quote.get("price"):
                    cache_updates.append(f"price = ${cache_param_num}")
                    cache_params.append(quote["price"])
                    cache_param_num += 1
                
                if quote.get("change"):
                    cache_updates.append(f"change = ${cache_param_num}")
                    cache_params.append(quote["change"])
                    cache_param_num += 1
                
                if quote.get("change_pct"):
                    cache_updates.append(f"change_pct = ${cache_param_num}")
                    cache_params.append(quote["change_pct"])
                    cache_param_num += 1
                
                if financials.get("52_week_high"):
                    cache_updates.append(f"high_52w = ${cache_param_num}")
                    cache_params.append(financials["52_week_high"])
                    cache_param_num += 1
                
                if financials.get("52_week_low"):
                    cache_updates.append(f"low_52w = ${cache_param_num}")
                    cache_params.append(financials["52_week_low"])
                    cache_param_num += 1
                
                if cache_updates:
                    cache_updates.append(f"updated_at = ${cache_param_num}")
                    cache_params.append(datetime.utcnow())
                    cache_param_num += 1
                    
                    cache_params.append(symbol)
                    
                    cache_query = f"""
                        UPDATE market_data_cache 
                        SET {", ".join(cache_updates)}
                        WHERE symbol = ${cache_param_num}
                    """
                    
                    await conn.execute(cache_query, *cache_params)
                    print(f"  ✅ Updated market_data_cache for {symbol}")
            
            return True
            
    except Exception as e:
        print(f"  ❌ Database error for {symbol}: {e}")
        return False


async def main():
    """Main enrichment process."""
    print("=" * 60)
    print("CIFT MARKETS - SYMBOL DATA ENRICHMENT")
    print("=" * 60)
    print(f"\nConnecting to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}")
    
    # Connect to database
    try:
        pool = await asyncpg.create_pool(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            min_size=2,
            max_size=5,
        )
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        sys.exit(1)
    
    print("✅ Connected to database")
    
    # Get list of symbols to enrich
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT symbol FROM symbols 
            WHERE is_active = true 
            ORDER BY market_cap DESC NULLS LAST
        """)
        symbols = [row["symbol"] for row in rows]
    
    print(f"\n📋 Found {len(symbols)} symbols to enrich")
    
    # Create HTTP session
    async with aiohttp.ClientSession() as session:
        success_count = 0
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] ", end="")
            
            success = await enrich_symbol(pool, session, symbol)
            if success:
                success_count += 1
            
            # Rate limiting between symbols
            await asyncio.sleep(RATE_LIMIT_DELAY)
    
    # Close database pool
    await pool.close()
    
    print("\n" + "=" * 60)
    print(f"ENRICHMENT COMPLETE: {success_count}/{len(symbols)} symbols updated")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
