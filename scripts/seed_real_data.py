#!/usr/bin/env python
"""
CIFT Markets - Real Data Seeding Script

Seeds real market data from Finnhub into the database.
Run this script to populate the database with real stock data.

Usage:
    python scripts/seed_real_data.py [--symbols AAPL,MSFT,...] [--quick]

Options:
    --symbols    Comma-separated list of symbols (default: major stocks)
    --quick      Quick mode - only seed quotes, no historical candles
"""

import argparse
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger


async def main():
    parser = argparse.ArgumentParser(description='Seed real market data from Finnhub')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--quick', action='store_true', help='Quick mode - quotes only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Configure logging
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    try:
        # Initialize database connection
        from cift.core.database import close_all_connections, initialize_all_connections

        logger.info("🔗 Connecting to databases...")
        await initialize_all_connections()
        logger.success("✅ Database connections established")

        # Run migrations to ensure tables exist
        logger.info("📊 Checking database schema...")
        from cift.core.database import db_manager

        migration_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'database', 'migrations', '002_add_company_data.sql'
        )

        if os.path.exists(migration_path):
            async with db_manager.pool.acquire() as conn:
                with open(migration_path) as f:
                    migration_sql = f.read()
                try:
                    await conn.execute(migration_sql)
                    logger.success("✅ Database schema updated")
                except Exception as e:
                    # Some errors are expected if tables/constraints already exist
                    logger.debug(f"Migration note: {e}")

        # Import seeder
        from cift.services.finnhub_data_seeder import FinnhubDataSeeder

        seeder = FinnhubDataSeeder()

        # Parse symbols
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        else:
            symbols = seeder.DEFAULT_SYMBOLS

        logger.info(f"🌱 Seeding data for {len(symbols)} symbols...")
        logger.info(f"   Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

        if args.quick:
            logger.info("⚡ Quick mode: seeding quotes only (no historical candles)")

            for symbol in symbols:
                try:
                    # Just seed profile and quote
                    await seeder.seed_company_profile(symbol)
                    await seeder.seed_quote(symbol)
                except Exception as e:
                    logger.warning(f"Failed to seed {symbol}: {e}")

            logger.success("✅ Quick seeding complete!")
        else:
            # Full seeding with historical candles
            results = await seeder.seed_all(symbols, include_candles=True)

            logger.success("=" * 60)
            logger.success("SEEDING RESULTS")
            logger.success("=" * 60)
            logger.info(f"  Total symbols:   {results['total_symbols']}")
            logger.info(f"  Successful:      {results['successful']}")
            logger.info(f"  Failed:          {results['failed']}")
            logger.info(f"  Total candles:   {results['total_candles']:,}")
            logger.info(f"  Time elapsed:    {results['elapsed_seconds']:.1f} seconds")
            logger.success("=" * 60)

        await seeder.close()

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("💡 Tip: Make sure FINNHUB_API_KEY is set in your .env file")
        logger.info("   Get a FREE API key at: https://finnhub.io/")
        return 1

    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        try:
            await close_all_connections()
        except:
            pass

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
