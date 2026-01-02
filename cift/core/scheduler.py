"""
CIFT Markets - Background Task Scheduler

Handles periodic tasks like:
- KYC document and identity verification processing
- Portfolio snapshot generation
- Market data cleanup
- Risk monitoring
- Notifications
"""

import asyncio
from datetime import datetime, timedelta

from loguru import logger

from cift.core.config import get_settings


class TaskScheduler:
    """Background task scheduler for periodic operations."""

    def __init__(self):
        self.settings = get_settings()
        self.tasks: dict[str, dict] = {}
        self.running = False

        # Register default tasks
        self.register_default_tasks()

    def register_default_tasks(self):
        """Register default periodic tasks."""

        # KYC verification processing (every 30 seconds)
        self.register_task(
            name="kyc_verification_processing",
            func=self.process_kyc_verifications,
            interval_seconds=30,
            description="Process pending KYC document and identity verifications"
        )

        # Market data update from Polygon/Finnhub (every 5 minutes for fresher quotes)
        self.register_task(
            name="polygon_market_data_update",
            func=self.update_polygon_market_data,
            interval_seconds=300,  # Every 5 minutes for more real-time data
            description="Update market quotes and news from Polygon.io/Finnhub"
        )

        # Fundamental data refresh (every 6 hours - market cap, P/E, P/B, ROE, etc.)
        self.register_task(
            name="fundamental_data_refresh",
            func=self.refresh_fundamental_data,
            interval_seconds=21600,  # Every 6 hours
            description="Refresh fundamental data (market cap, P/E, ROE) from Finnhub"
        )

        # Alpaca Sync (every 1 minute)
        self.register_task(
            name="alpaca_sync",
            func=self.sync_alpaca_data,
            interval_seconds=60,
            description="Sync Account and Positions with Alpaca"
        )

        # Portfolio snapshots (daily at 6 PM ET)
        self.register_task(
            name="daily_portfolio_snapshots",
            func=self.generate_portfolio_snapshots,
            interval_seconds=3600,  # Check every hour
            description="Generate end-of-day portfolio snapshots"
        )

        # Market data cleanup (daily at 2 AM ET)
        self.register_task(
            name="market_data_cleanup",
            func=self.cleanup_old_market_data,
            interval_seconds=86400,  # Daily
            description="Clean up old market data and cache"
        )

        # Risk monitoring (every 5 minutes during market hours)
        self.register_task(
            name="risk_monitoring",
            func=self.monitor_risk_limits,
            interval_seconds=300,
            description="Monitor portfolio risk limits and alerts"
        )

        # Price alerts monitoring (every 30 seconds)
        self.register_task(
            name="price_alerts_monitoring",
            func=self.monitor_price_alerts,
            interval_seconds=30,
            description="Monitor and process price alerts"
        )

        # ML Signal Service - Generate trading signals (every 5 minutes)
        self.register_task(
            name="ml_signal_generation",
            func=self.generate_ml_signals,
            interval_seconds=300,
            description="Generate ML-powered trading signals and alerts"
        )

        # Paper Trading Loop - Continuous predictions (every 5 minutes)
        self.register_task(
            name="paper_trading_loop",
            func=self.run_paper_trading_loop,
            interval_seconds=300,
            description="Run paper trading predictions and track performance"
        )

    async def sync_alpaca_data(self):
        """Sync data with Alpaca"""
        try:
            from cift.services.alpaca_sync_service import alpaca_sync_service
            await alpaca_sync_service.sync_all_accounts()
        except Exception as e:
            logger.error(f"Alpaca sync failed: {e}")

    async def generate_ml_signals(self):
        """Generate ML-powered trading signals for watched symbols."""
        try:
            from cift.services.ml_signal_service import MLSignalService
            
            ml_service = MLSignalService()
            
            # Get top traded symbols from watchlists (crypto focus)
            default_symbols = [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT"
            ]
            
            for symbol in default_symbols:
                try:
                    signal = await ml_service.generate_signal(symbol, user_id="system")
                    if signal and signal.confidence >= 0.70:
                        logger.info(f"🧠 ML Signal: {symbol} -> {signal.signal.value} ({signal.confidence:.0%})")
                except Exception as e:
                    logger.warning(f"ML signal generation failed for {symbol}: {e}")
                    
            logger.debug(f"ML signal generation completed for {len(default_symbols)} symbols")
        except Exception as e:
            logger.error(f"ML signal generation task failed: {e}")

    async def run_paper_trading_loop(self):
        """Run paper trading predictions and track performance."""
        try:
            from cift.services.ml_signal_service import MLSignalService
            
            ml_service = MLSignalService()
            
            # Track paper trades for crypto pairs
            paper_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
            
            for symbol in paper_symbols:
                try:
                    signal = await ml_service.generate_signal(symbol, user_id="paper_trader")
                    
                    if signal and signal.signal.value in ["strong_buy", "strong_sell"]:
                        # Execute paper trade
                        trade = await ml_service.execute_paper_trade(
                            symbol=symbol,
                            signal=signal,
                            position_size=1000.0  # $1000 paper position
                        )
                        if trade:
                            logger.info(f"📝 Paper Trade: {trade.side.upper()} {symbol} @ ${trade.entry_price:.2f}")
                except Exception as e:
                    logger.warning(f"Paper trading failed for {symbol}: {e}")
            
            # Calculate and log current Sharpe ratio
            try:
                stats = await ml_service.get_paper_trading_stats()
                if stats and stats.get("total_trades", 0) > 0:
                    sharpe = stats.get("sharpe_ratio", 0)
                    win_rate = stats.get("win_rate", 0)
                    logger.info(f"📊 Paper Trading: {stats['total_trades']} trades, {win_rate:.1%} win rate, Sharpe: {sharpe:.2f}")
            except Exception as e:
                logger.debug(f"Paper trading stats unavailable: {e}")
                
        except Exception as e:
            logger.error(f"Paper trading loop failed: {e}")

    async def process_kyc_verifications(self):
        """Process pending KYC verifications."""
        try:
            from cift.services.kyc_verification import process_pending_verifications
            await process_pending_verifications()
        except Exception as e:
            logger.error(f"KYC verification processing failed: {e}")

    def register_task(
        self,
        name: str,
        func,
        interval_seconds: int,
        description: str = "",
        enabled: bool = True
    ):
        """Register a periodic task."""
        self.tasks[name] = {
            "func": func,
            "interval_seconds": interval_seconds,
            "description": description,
            "enabled": enabled,
            "last_run": None,
            "next_run": datetime.utcnow(),
            "run_count": 0,
            "error_count": 0,
            "last_error": None,
        }

        logger.info(f"Registered task '{name}': {description} (every {interval_seconds}s)")

    async def start(self):
        """Start the task scheduler."""
        self.running = True
        logger.info("🕐 Starting background task scheduler...")

        while self.running:
            try:
                await self.run_due_tasks()
                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    def stop(self):
        """Stop the task scheduler."""
        self.running = False
        logger.info("⏹️ Stopping background task scheduler...")

    async def run_due_tasks(self):
        """Run all tasks that are due."""
        now = datetime.utcnow()

        for task_name, task_info in self.tasks.items():
            if not task_info["enabled"]:
                continue

            if now >= task_info["next_run"]:
                await self.run_task(task_name)

    async def run_task(self, task_name: str):
        """Run a specific task."""
        task_info = self.tasks.get(task_name)
        if not task_info:
            logger.error(f"Task '{task_name}' not found")
            return

        logger.debug(f"Running task: {task_name}")
        start_time = datetime.utcnow()

        try:
            # Run the task function
            await task_info["func"]()

            # Update task status
            task_info["last_run"] = start_time
            task_info["run_count"] += 1
            task_info["next_run"] = start_time + timedelta(seconds=task_info["interval_seconds"])

            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.debug(f"✅ Task '{task_name}' completed in {duration:.2f}s")

        except Exception as e:
            # Update error status
            task_info["error_count"] += 1
            task_info["last_error"] = str(e)
            task_info["next_run"] = start_time + timedelta(seconds=min(task_info["interval_seconds"], 300))

            logger.error(f"❌ Task '{task_name}' failed: {e}")

    def get_task_status(self) -> dict:
        """Get status of all registered tasks."""
        now = datetime.utcnow()

        return {
            "scheduler_running": self.running,
            "task_count": len(self.tasks),
            "tasks": {
                name: {
                    "description": task["description"],
                    "enabled": task["enabled"],
                    "interval_seconds": task["interval_seconds"],
                    "last_run": task["last_run"].isoformat() if task["last_run"] else None,
                    "next_run": task["next_run"].isoformat() if task["next_run"] else None,
                    "next_run_in_seconds": (task["next_run"] - now).total_seconds() if task["next_run"] else None,
                    "run_count": task["run_count"],
                    "error_count": task["error_count"],
                    "last_error": task["last_error"],
                }
                for name, task in self.tasks.items()
            }
        }

    # ========================================================================
    # TASK IMPLEMENTATIONS
    # ========================================================================

    async def update_polygon_market_data(self):
        """Update market data from Polygon.io (hourly)."""
        try:
            from cift.core.config import get_settings
            from cift.services.polygon_realtime_service import PolygonRealtimeService

            settings = get_settings()

            # Skip if no API key configured
            if not settings.polygon_api_key:
                logger.debug("Skipping Polygon update - no API key configured")
                return

            service = PolygonRealtimeService()
            await service.initialize()

            try:
                # Update quotes for popular symbols
                # TSLA must be in first batch since it's heavily traded
                popular_symbols = [
                    "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN",  # First priority batch
                    "NVDA", "META", "SPY", "QQQ", "JPM",
                    "V", "JNJ", "WMT", "PG", "XOM",
                    "UNH", "HD", "MA", "COST", "DIA"
                ]

                # Update more symbols per run (Finnhub fallback has higher rate limits)
                quotes_updated = await service.update_market_cache(symbols=popular_symbols[:10])
                logger.info(f"Updated {quotes_updated} quotes from Polygon")

                # Update news (less frequently - once per hour is fine)
                news_stored = await service.fetch_and_store_news(
                    symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
                    limit=30
                )
                logger.info(f"Stored {news_stored} news articles from Polygon")

            finally:
                await service.close()

        except Exception as e:
            logger.error(f"Polygon market data update failed: {e}")
            # Don't raise - allow scheduler to continue
            raise

    async def refresh_fundamental_data(self):
        """Refresh fundamental data (market cap, P/E, P/B, ROE) for all symbols from Finnhub."""
        try:
            import aiohttp
            from cift.core.config import get_settings
            from cift.core.database import get_postgres_pool

            settings = get_settings()
            finnhub_api_key = settings.finnhub_api_key

            if not finnhub_api_key:
                logger.debug("Skipping fundamental refresh - no Finnhub API key configured")
                return

            pool = await get_postgres_pool()

            # Get all active symbols
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT symbol FROM symbols WHERE is_active = true ORDER BY market_cap DESC NULLS LAST LIMIT 50"
                )
                symbols = [row["symbol"] for row in rows]

            if not symbols:
                logger.debug("No symbols to refresh")
                return

            logger.info(f"Refreshing fundamental data for {len(symbols)} symbols...")
            updated_count = 0

            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    try:
                        # Get company profile (includes market cap)
                        profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={finnhub_api_key}"
                        async with session.get(profile_url) as resp:
                            if resp.status == 200:
                                profile = await resp.json()
                            else:
                                profile = {}

                        await asyncio.sleep(0.5)  # Rate limiting

                        # Get basic financials (P/E, P/B, ROE, margins)
                        metrics_url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={finnhub_api_key}"
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

                        # EPS
                        eps = metrics.get("epsTTM") or metrics.get("epsBasicExclExtraItemsTTM")
                        if eps:
                            updates.append(f"eps = ${param_num}")
                            params.append(float(eps))
                            param_num += 1

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

                            # Also update market_data_cache with 52-week data
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
                                    try:
                                        async with pool.acquire() as conn:
                                            await conn.execute(cache_query, *cache_params)
                                    except Exception:
                                        pass  # Column may not exist, skip

                            updated_count += 1
                            logger.debug(f"Updated fundamentals for {symbol}")

                    except Exception as e:
                        logger.warning(f"Failed to refresh {symbol}: {e}")
                        continue

            logger.info(f"✅ Refreshed fundamental data for {updated_count}/{len(symbols)} symbols")

        except Exception as e:
            logger.error(f"Fundamental data refresh failed: {e}")
            raise

    async def generate_portfolio_snapshots(self):
        """Generate daily portfolio snapshots using enhanced analytics service."""
        try:
            # Run multiple times per day, not just market close
            from cift.services.portfolio_analytics import generate_all_snapshots

            snapshot_count = await generate_all_snapshots()

            if snapshot_count > 0:
                logger.info(f"Generated {snapshot_count} portfolio snapshots")

        except Exception as e:
            logger.error(f"Portfolio snapshot generation failed: {e}")
            raise

    async def create_portfolio_snapshot(self, user_id: str, account_id: str):
        """Create a portfolio snapshot for a specific account."""
        from uuid import uuid4

        from cift.core.database import get_postgres_pool

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Get current portfolio state
            positions = await conn.fetch("""
                SELECT symbol, quantity, current_price, market_value, unrealized_pnl
                FROM positions
                WHERE account_id = $1 AND quantity != 0
            """, account_id)

            account_info = await conn.fetchrow("""
                SELECT cash_balance, equity, buying_power
                FROM accounts
                WHERE id = $1
            """, account_id)

            if not account_info:
                return

            # Calculate portfolio metrics
            positions_value = sum(float(p['market_value'] or 0) for p in positions)
            unrealized_pnl = sum(float(p['unrealized_pnl'] or 0) for p in positions)

            total_value = float(account_info['equity'] or 0)
            cash = float(account_info['cash_balance'] or 0)

            # Create snapshot
            await conn.execute("""
                INSERT INTO portfolio_snapshots (
                    id, user_id, account_id, timestamp, snapshot_type,
                    total_value, cash, positions_value, equity,
                    unrealized_pnl, realized_pnl, created_at
                ) VALUES (
                    $1, $2, $3, $4, 'eod',
                    $5, $6, $7, $5,
                    $8, 0, $4
                )
                ON CONFLICT (id) DO NOTHING
            """,
                uuid4(), user_id, account_id, datetime.utcnow(),
                total_value, cash, positions_value, unrealized_pnl
            )

    async def cleanup_old_market_data(self):
        """Clean up old market data and cache entries."""
        try:
            from cift.core.database import get_postgres_pool

            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                # Clean up old market data cache (keep 30 days)
                await conn.execute("""
                    DELETE FROM market_data_cache
                    WHERE updated_at < NOW() - INTERVAL '30 days'
                """)

                # Clean up old OHLCV data (keep 2 years)
                # Note: This would be done in QuestDB, not PostgreSQL

                logger.info("Cleaned up old market data cache entries")

        except Exception as e:
            logger.error(f"Market data cleanup failed: {e}")
            raise

    async def monitor_risk_limits(self):
        """Monitor portfolio risk limits and generate alerts."""
        try:
            # Only run during market hours (9:30 AM - 4 PM ET)
            now = datetime.utcnow()
            hour_et = (now.hour - 5) % 24  # Rough ET conversion
            minute = now.minute

            # Market hours: 9:30 AM - 4:00 PM ET
            market_open = (hour_et == 9 and minute >= 30) or (10 <= hour_et <= 15)

            if not market_open:
                logger.debug("Skipping risk monitoring - market closed")
                return

            from cift.core.database import get_postgres_pool

            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                # Check accounts with high risk
                high_risk_accounts = await conn.fetch("""
                    SELECT
                        a.id, a.user_id, a.equity, a.buying_power,
                        SUM(ABS(p.market_value)) as total_exposure
                    FROM accounts a
                    LEFT JOIN positions p ON p.account_id = a.id
                    WHERE a.is_active = true
                    GROUP BY a.id, a.user_id, a.equity, a.buying_power
                    HAVING SUM(ABS(p.market_value)) > a.equity * 0.8  -- 80% exposure threshold
                """)

                for account in high_risk_accounts:
                    exposure_ratio = float(account['total_exposure'] or 0) / float(account['equity'] or 1)

                    if exposure_ratio > 0.9:  # 90% threshold
                        logger.warning(f"High risk account {account['id']}: {exposure_ratio:.1%} exposure")

                        # TODO: Send alert notification
                        # await send_risk_alert(account['user_id'], exposure_ratio)

        except Exception as e:
            logger.error(f"Risk monitoring failed: {e}")
            raise

    async def monitor_price_alerts(self):
        """Monitor and process price alerts."""
        try:
            from cift.services.price_alerts import get_alert_service

            alert_service = get_alert_service()

            # Start monitoring if not already running
            if not alert_service.monitoring:
                # Start monitoring in background task
                import asyncio
                asyncio.create_task(alert_service.start_monitoring())
                logger.info("Started price alerts monitoring service")

        except Exception as e:
            logger.error(f"Price alerts monitoring failed: {e}")
            raise


# Global scheduler instance
_scheduler = None


def get_scheduler() -> TaskScheduler:
    """Get the global task scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


async def start_scheduler():
    """Start the background task scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()


def stop_scheduler():
    """Stop the background task scheduler."""
    scheduler = get_scheduler()
    scheduler.stop()


# FastAPI startup integration
async def setup_background_tasks():
    """Setup background tasks for FastAPI app."""
    logger.info("🚀 Setting up background tasks...")

    # Start scheduler in background
    scheduler = get_scheduler()
    asyncio.create_task(scheduler.start())

    logger.success("✅ Background tasks started")


if __name__ == "__main__":
    # Test the scheduler
    async def test_scheduler():
        scheduler = TaskScheduler()

        # Run for 60 seconds
        task = asyncio.create_task(scheduler.start())
        await asyncio.sleep(60)

        scheduler.stop()
        await task

        # Print final status
        status = scheduler.get_task_status()
        print(f"Tasks run: {status}")

    asyncio.run(test_scheduler())
