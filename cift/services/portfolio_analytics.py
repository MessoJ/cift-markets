"""
CIFT Markets - Portfolio Analytics & Historical Snapshots Service

Advanced portfolio analytics with historical tracking, performance metrics,
and comprehensive reporting capabilities.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel

from cift.core.database import get_postgres_pool


class PortfolioSnapshot(BaseModel):
    id: UUID | None = None
    user_id: UUID
    account_id: UUID
    timestamp: datetime
    snapshot_type: str = "eod"  # eod, intraday, weekly, monthly
    total_value: Decimal
    cash: Decimal
    positions_value: Decimal
    equity: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    day_pnl: Decimal
    day_pnl_pct: Decimal
    positions_count: int = 0
    largest_position: str | None = None
    largest_position_value: Decimal = Decimal("0")


class PerformanceMetrics(BaseModel):
    total_return: Decimal
    total_return_pct: Decimal
    annualized_return: Decimal
    volatility: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_pct: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    best_day: Decimal
    worst_day: Decimal


class PortfolioAnalyticsService:
    """Advanced portfolio analytics and historical tracking service."""

    def __init__(self):
        pass

    async def create_snapshot(
        self, user_id: UUID, account_id: UUID, snapshot_type: str = "eod"
    ) -> PortfolioSnapshot:
        """Create a portfolio snapshot for an account."""

        logger.debug(f"Creating {snapshot_type} snapshot for account {account_id}")

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            # Get account information
            account = await conn.fetchrow(
                """
                SELECT cash_balance, equity, buying_power, updated_at
                FROM accounts
                WHERE id = $1 AND user_id = $2
            """,
                account_id,
                user_id,
            )

            if not account:
                raise ValueError(f"Account {account_id} not found for user {user_id}")

            # Get current positions
            positions = await conn.fetch(
                """
                SELECT
                    symbol, quantity, current_price, market_value,
                    unrealized_pnl, realized_pnl, day_pnl, day_pnl_pct,
                    avg_cost, total_cost
                FROM positions
                WHERE account_id = $1 AND quantity != 0
                ORDER BY market_value DESC
            """,
                account_id,
            )

            # Calculate metrics
            total_positions_value = sum(Decimal(str(p["market_value"] or 0)) for p in positions)
            total_unrealized_pnl = sum(Decimal(str(p["unrealized_pnl"] or 0)) for p in positions)
            total_realized_pnl = sum(Decimal(str(p["realized_pnl"] or 0)) for p in positions)
            total_day_pnl = sum(Decimal(str(p["day_pnl"] or 0)) for p in positions)

            cash = Decimal(str(account["cash_balance"] or 0))
            equity = Decimal(str(account["equity"] or 0))
            total_value = cash + total_positions_value

            # Calculate day P&L percentage
            if total_value > 0:
                day_pnl_pct = (total_day_pnl / total_value) * 100
            else:
                day_pnl_pct = Decimal("0")

            # Find largest position
            largest_position = None
            largest_position_value = Decimal("0")

            if positions:
                largest = positions[0]  # Already sorted by market_value DESC
                largest_position = largest["symbol"]
                largest_position_value = Decimal(str(largest["market_value"] or 0))

            # Create snapshot object
            snapshot = PortfolioSnapshot(
                id=uuid4(),
                user_id=user_id,
                account_id=account_id,
                timestamp=datetime.utcnow(),
                snapshot_type=snapshot_type,
                total_value=total_value,
                cash=cash,
                positions_value=total_positions_value,
                equity=equity,
                unrealized_pnl=total_unrealized_pnl,
                realized_pnl=total_realized_pnl,
                day_pnl=total_day_pnl,
                day_pnl_pct=day_pnl_pct,
                positions_count=len(positions),
                largest_position=largest_position,
                largest_position_value=largest_position_value,
            )

            # Store in database
            await conn.execute(
                """
                INSERT INTO portfolio_snapshots (
                    id, user_id, account_id, timestamp, snapshot_type,
                    total_value, cash, positions_value, equity,
                    unrealized_pnl, realized_pnl, day_pnl, day_pnl_pct,
                    positions_count, largest_position, largest_position_value,
                    created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $4
                )
                ON CONFLICT (account_id, DATE(timestamp), snapshot_type) DO UPDATE SET
                    total_value = EXCLUDED.total_value,
                    cash = EXCLUDED.cash,
                    positions_value = EXCLUDED.positions_value,
                    equity = EXCLUDED.equity,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl = EXCLUDED.realized_pnl,
                    day_pnl = EXCLUDED.day_pnl,
                    day_pnl_pct = EXCLUDED.day_pnl_pct,
                    positions_count = EXCLUDED.positions_count,
                    largest_position = EXCLUDED.largest_position,
                    largest_position_value = EXCLUDED.largest_position_value
            """,
                snapshot.id,
                snapshot.user_id,
                snapshot.account_id,
                snapshot.timestamp,
                snapshot.snapshot_type,
                snapshot.total_value,
                snapshot.cash,
                snapshot.positions_value,
                snapshot.equity,
                snapshot.unrealized_pnl,
                snapshot.realized_pnl,
                snapshot.day_pnl,
                snapshot.day_pnl_pct,
                snapshot.positions_count,
                snapshot.largest_position,
                snapshot.largest_position_value,
            )

            logger.success(f"Created {snapshot_type} snapshot: ${snapshot.total_value:,.2f}")
            return snapshot

    async def get_historical_snapshots(
        self, user_id: UUID, account_id: UUID, days: int = 90, snapshot_type: str = "eod"
    ) -> list[PortfolioSnapshot]:
        """Get historical portfolio snapshots."""

        pool = await get_postgres_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM portfolio_snapshots
                WHERE user_id = $1 AND account_id = $2
                AND snapshot_type = $3
                AND timestamp >= $4
                ORDER BY timestamp ASC
            """,
                user_id,
                account_id,
                snapshot_type,
                datetime.utcnow() - timedelta(days=days),
            )

        snapshots = []
        for row in rows:
            snapshots.append(
                PortfolioSnapshot(
                    id=row["id"],
                    user_id=row["user_id"],
                    account_id=row["account_id"],
                    timestamp=row["timestamp"],
                    snapshot_type=row["snapshot_type"],
                    total_value=Decimal(str(row["total_value"])),
                    cash=Decimal(str(row["cash"])),
                    positions_value=Decimal(str(row["positions_value"])),
                    equity=Decimal(str(row["equity"])),
                    unrealized_pnl=Decimal(str(row["unrealized_pnl"])),
                    realized_pnl=Decimal(str(row["realized_pnl"])),
                    day_pnl=Decimal(str(row["day_pnl"])),
                    day_pnl_pct=Decimal(str(row["day_pnl_pct"])),
                    positions_count=row["positions_count"] or 0,
                    largest_position=row["largest_position"],
                    largest_position_value=Decimal(str(row["largest_position_value"] or 0)),
                )
            )

        return snapshots

    async def calculate_performance_metrics(
        self, user_id: UUID, account_id: UUID, days: int = 90
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        snapshots = await self.get_historical_snapshots(user_id, account_id, days)

        if len(snapshots) < 2:
            # Return empty metrics if insufficient data
            return PerformanceMetrics(
                total_return=Decimal("0"),
                total_return_pct=Decimal("0"),
                annualized_return=Decimal("0"),
                volatility=Decimal("0"),
                sharpe_ratio=Decimal("0"),
                max_drawdown=Decimal("0"),
                max_drawdown_pct=Decimal("0"),
                win_rate=Decimal("0"),
                profit_factor=Decimal("0"),
                best_day=Decimal("0"),
                worst_day=Decimal("0"),
            )

        # Calculate returns
        first_value = snapshots[0].total_value
        last_value = snapshots[-1].total_value
        total_return = last_value - first_value
        total_return_pct = (total_return / first_value * 100) if first_value > 0 else Decimal("0")

        # Annualized return
        period_years = Decimal(str(days / 365.25))
        if period_years > 0 and first_value > 0:
            annualized_return = ((last_value / first_value) ** (1 / float(period_years)) - 1) * 100
        else:
            annualized_return = Decimal("0")

        # Daily returns for volatility and other metrics
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i - 1].total_value
            curr_value = snapshots[i].total_value

            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(daily_return)

        # Volatility (standard deviation of daily returns)
        if len(daily_returns) > 1:
            mean_return = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            volatility = (
                (variance ** Decimal("0.5")) * Decimal("252") ** Decimal("0.5") * 100
            )  # Annualized
        else:
            volatility = Decimal("0")

        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = Decimal("0.02")
        if volatility > 0:
            sharpe_ratio = (annualized_return / 100 - risk_free_rate) / (volatility / 100)
        else:
            sharpe_ratio = Decimal("0")

        # Max drawdown
        peak = snapshots[0].total_value
        max_drawdown = Decimal("0")
        max_drawdown_pct = Decimal("0")

        for snapshot in snapshots:
            if snapshot.total_value > peak:
                peak = snapshot.total_value

            drawdown = peak - snapshot.total_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = (drawdown / peak * 100) if peak > 0 else Decimal("0")

        # Win rate and profit factor
        winning_days = sum(1 for r in daily_returns if r > 0)
        sum(1 for r in daily_returns if r < 0)

        win_rate = (
            Decimal(winning_days) / len(daily_returns) * 100 if daily_returns else Decimal("0")
        )

        total_wins = sum(r for r in daily_returns if r > 0)
        total_losses = abs(sum(r for r in daily_returns if r < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else Decimal("0")

        # Best and worst day
        best_day = max(daily_returns) * 100 if daily_returns else Decimal("0")
        worst_day = min(daily_returns) * 100 if daily_returns else Decimal("0")

        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=Decimal(str(annualized_return)),
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            profit_factor=profit_factor,
            best_day=best_day,
            worst_day=worst_day,
        )


# Global analytics service
_analytics_service = None


def get_analytics_service() -> PortfolioAnalyticsService:
    """Get the global analytics service instance."""
    global _analytics_service
    if _analytics_service is None:
        _analytics_service = PortfolioAnalyticsService()
    return _analytics_service


async def generate_all_snapshots() -> int:
    """Generate snapshots for all active accounts."""

    logger.info("🔄 Generating portfolio snapshots for all accounts...")

    pool = await get_postgres_pool()
    analytics = get_analytics_service()
    snapshot_count = 0

    async with pool.acquire() as conn:
        # Get all active accounts that need snapshots today
        accounts = await conn.fetch(
            """
            SELECT DISTINCT user_id, id as account_id
            FROM accounts
            WHERE is_active = true
            AND NOT EXISTS (
                SELECT 1 FROM portfolio_snapshots ps
                WHERE ps.account_id = accounts.id
                AND ps.timestamp::date = CURRENT_DATE
                AND ps.snapshot_type = 'eod'
            )
        """
        )

        for account in accounts:
            try:
                await analytics.create_snapshot(account["user_id"], account["account_id"], "eod")
                snapshot_count += 1

            except Exception as e:
                logger.error(f"Failed to create snapshot for account {account['account_id']}: {e}")

    logger.success(f"✅ Generated {snapshot_count} portfolio snapshots")
    return snapshot_count
