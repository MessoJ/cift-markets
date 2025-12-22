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

        # Market data update from Polygon (every hour)
        self.register_task(
            name="polygon_market_data_update",
            func=self.update_polygon_market_data,
            interval_seconds=3600,  # Every hour
            description="Update market quotes and news from Polygon.io"
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

    async def sync_alpaca_data(self):
        """Sync data with Alpaca"""
        try:
            from cift.services.alpaca_sync_service import alpaca_sync_service
            await alpaca_sync_service.sync_all_accounts()
        except Exception as e:
            logger.error(f"Alpaca sync failed: {e}")

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
                # Update quotes for popular symbols (limited due to rate limits)
                # Free tier: 5 req/min, so we update ~20 symbols per hour
                popular_symbols = [
                    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                    "TSLA", "META", "SPY", "QQQ", "JPM",
                    "V", "JNJ", "WMT", "PG", "XOM",
                    "UNH", "HD", "MA", "COST", "DIA"
                ]

                quotes_updated = await service.update_market_cache(symbols=popular_symbols[:5])
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
