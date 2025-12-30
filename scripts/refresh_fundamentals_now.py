#!/usr/bin/env python3
"""
Trigger immediate fundamental data refresh.
This script fetches latest market cap, P/E, P/B, ROE, etc. from Finnhub.

Usage:
    python scripts/refresh_fundamentals_now.py
    
Or via Docker:
    docker exec cift-api python scripts/refresh_fundamentals_now.py
"""

import asyncio
import os
import aiohttp
import asyncpg
from datetime import datetime

# Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d4ojf7pr01qtc1p01m60d4ojf7pr01qtc1p01m6g")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "cift_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "cift_markets")


async def refresh_symbol(session: aiohttp.ClientSession, pool: asyncpg.Pool, symbol: str) -> bool:
    """Refresh fundamental data for a single symbol."""
    try:
        # Get company profile (includes market cap)
        profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={FINNHUB_API_KEY}"
        async with session.get(profile_url) as resp:
            if resp.status == 200:
                profile = await resp.json()
            else:
                profile = {}

        await asyncio.sleep(0.5)  # Rate limiting

        # Get basic financials (P/E, P/B, ROE, margins)
        metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={FINNHUB_API_KEY}"
        async with session.get(metrics_url) as resp:
            if resp.status == 200:
                metrics_data = await resp.json()
                metrics = metrics_data.get("metric", {})
            else:
                metrics = {}

        await asyncio.sleep(0.5)  # Rate limiting

        # Build update query dynamically
        updates = []
        params = []
        param_num = 1

        # Market cap from profile (in millions, store in actual value)
        market_cap = profile.get("marketCapitalization")
        if market_cap:
            updates.append(f"market_cap = ${param_num}")
            params.append(float(market_cap) * 1_000_000)  # Convert millions to actual
            param_num += 1
            print(f"  → Market Cap: ${market_cap:,.0f}M")

        # Shares outstanding from profile
        shares = profile.get("shareOutstanding")
        if shares:
            updates.append(f"shares_outstanding = ${param_num}")
            params.append(float(shares) * 1_000_000)  # Convert millions
            param_num += 1

        # P/E ratio
        pe = metrics.get("peBasicExclExtraTTM") or metrics.get("peNormalizedAnnual")
        if pe:
            updates.append(f"pe_ratio = ${param_num}")
            params.append(float(pe))
            param_num += 1
            print(f"  → P/E Ratio: {pe:.2f}")

        # EPS
        eps = metrics.get("epsTTM") or metrics.get("epsBasicExclExtraItemsTTM")
        if eps:
            updates.append(f"eps = ${param_num}")
            params.append(float(eps))
            param_num += 1
            print(f"  → EPS: ${eps:.2f}")

        # P/B ratio
        pb = metrics.get("pbQuarterly") or metrics.get("pbAnnual")
        if pb:
            updates.append(f"price_to_book = ${param_num}")
            params.append(float(pb))
            param_num += 1

        # P/S ratio
        ps = metrics.get("psQuarterly") or metrics.get("psTTM")
        if ps:
            updates.append(f"price_to_sales = ${param_num}")
            params.append(float(ps))
            param_num += 1

        # ROE
        roe = metrics.get("roeTTM") or metrics.get("roeRfy")
        if roe:
            updates.append(f"roe = ${param_num}")
            params.append(float(roe))
            param_num += 1

        # ROA
        roa = metrics.get("roaTTM") or metrics.get("roaRfy")
        if roa:
            updates.append(f"roa = ${param_num}")
            params.append(float(roa))
            param_num += 1

        # Profit margin
        margin = metrics.get("netProfitMarginTTM") or metrics.get("netProfitMarginAnnual")
        if margin:
            updates.append(f"profit_margin = ${param_num}")
            params.append(float(margin))
            param_num += 1

        # Dividend yield
        div_yield = metrics.get("dividendYieldIndicatedAnnual")
        if div_yield:
            updates.append(f"dividend_yield = ${param_num}")
            params.append(float(div_yield) / 100)  # Convert to decimal
            param_num += 1

        # 52-week high/low
        week_high = metrics.get("52WeekHigh")
        week_low = metrics.get("52WeekLow")

        if updates:
            updates.append(f"data_updated_at = NOW()")
            params.append(symbol)

            query = f"""
                UPDATE symbols 
                SET {", ".join(updates)}
                WHERE symbol = ${param_num}
            """

            async with pool.acquire() as conn:
                await conn.execute(query, *params)

            # Try to update market_data_cache with 52-week data (may not have column)
            try:
                if week_high or week_low:
                    cache_updates = []
                    cache_params = []
                    cache_num = 1

                    if week_high:
                        cache_updates.append(f"week_52_high = ${cache_num}")
                        cache_params.append(float(week_high))
                        cache_num += 1
                    if week_low:
                        cache_updates.append(f"week_52_low = ${cache_num}")
                        cache_params.append(float(week_low))
                        cache_num += 1

                    if cache_updates:
                        cache_params.append(symbol)
                        cache_query = f"""
                            UPDATE market_data_cache
                            SET {", ".join(cache_updates)}
                            WHERE symbol = ${cache_num}
                        """
                        async with pool.acquire() as conn:
                            await conn.execute(cache_query, *cache_params)
            except Exception:
                pass  # Column doesn't exist, skip

            return True

        return False

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


async def main():
    print("=" * 60)
    print("🔄 FUNDAMENTAL DATA REFRESH")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Connect to database
    pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        database=POSTGRES_DB,
        min_size=1,
        max_size=5
    )

    # Get all active symbols
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT symbol FROM symbols WHERE is_active = true ORDER BY market_cap DESC NULLS LAST"
        )
        symbols = [row["symbol"] for row in rows]

    print(f"Found {len(symbols)} symbols to refresh")
    print()

    updated_count = 0
    async with aiohttp.ClientSession() as session:
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] Refreshing {symbol}...")
            
            if await refresh_symbol(session, pool, symbol):
                updated_count += 1
                print(f"  ✅ Updated")
            else:
                print(f"  ⚠️ No data")
            
            print()

    await pool.close()

    print("=" * 60)
    print(f"✅ Completed: {updated_count}/{len(symbols)} symbols updated")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
