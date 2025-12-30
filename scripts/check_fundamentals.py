#!/usr/bin/env python3
import asyncio
import os
import asyncpg

async def main():
    pool = await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "cift_user"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme123"),
        database=os.getenv("POSTGRES_DB", "cift_markets")
    )
    
    # Check NVDA
    row = await pool.fetchrow(
        "SELECT symbol, market_cap, pe_ratio, eps, roe, profit_margin, data_updated_at FROM symbols WHERE symbol = $1",
        "NVDA"
    )
    print("=" * 60)
    print("NVDA FUNDAMENTAL DATA (from DB)")
    print("=" * 60)
    for k, v in dict(row).items():
        if k == "market_cap" and v:
            print(f"  {k}: ${float(v)/1e12:.2f}T")
        else:
            print(f"  {k}: {v}")
    
    # Compare top 5
    print("\n" + "=" * 60)
    print("TOP 5 BY MARKET CAP")
    print("=" * 60)
    rows = await pool.fetch(
        "SELECT symbol, market_cap, pe_ratio, eps FROM symbols ORDER BY market_cap DESC NULLS LAST LIMIT 5"
    )
    for row in rows:
        mc = row["market_cap"]
        mc_str = f"${float(mc)/1e12:.2f}T" if mc else "N/A"
        pe = row["pe_ratio"] or "N/A"
        eps = row["eps"] or "N/A"
        print(f"  {row['symbol']:6} | Market Cap: {mc_str:>10} | P/E: {pe:>8} | EPS: {eps}")
    
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())
