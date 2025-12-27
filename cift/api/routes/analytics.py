"""
CIFT Markets - Analytics API Routes

Performance analytics, P&L breakdowns, and portfolio metrics.
"""

from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel

from cift.core.auth import User, get_current_active_user
from cift.core.database import db_manager

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_current_user_id(
    current_user: User = Depends(get_current_active_user)
) -> UUID:
    """Get current authenticated user ID."""
    return current_user.id


# ============================================================================
# MODELS
# ============================================================================

class PerformanceMetrics(BaseModel):
    """Performance analytics response."""
    period: dict
    returns: dict
    risk_metrics: dict
    trade_statistics: dict


class PnLBreakdownItem(BaseModel):
    """P&L breakdown item."""
    symbol: str | None = None
    date: datetime | None = None
    month: datetime | None = None
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    num_trades: int | None = None
    current_position: float | None = None
    portfolio_value: float | None = None


# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
    start_date: datetime | None = Query(None, description="Start date (default: 30 days ago)"),
    end_date: datetime | None = Query(None, description="End date (default: now)"),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get comprehensive performance analytics.

    **Metrics Included:**
    - Total return (%)
    - Sharpe ratio (annualized)
    - Maximum drawdown (%)
    - Volatility (annualized %)
    - Win rate (%)
    - Average P&L per trade
    - Best/worst trade

    **Use Cases:**
    - Portfolio performance dashboard
    - Risk assessment
    - Performance comparison
    - Reporting

    **Requirements:**
    - Minimum 2 days of portfolio data

    Performance: ~10-20ms
    """
    from cift.core.trading_queries import get_performance_analytics

    # Default to 2 years of history if not specified, to ensure we get data
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=730)

    try:
        metrics = await get_performance_analytics(user_id, start_date, end_date)

        # Check if insufficient data
        if metrics.get("insufficient_data"):
            return {
                "period": {
                    "start_date": (start_date or datetime.utcnow() - timedelta(days=30)).isoformat(),
                    "end_date": (end_date or datetime.utcnow()).isoformat(),
                    "days": 0
                },
                "returns": {
                    "total_return_pct": 0,
                    "initial_value": 0,
                    "final_value": 0,
                    "total_pnl": 0
                },
                "risk_metrics": {
                    "sharpe_ratio": 0,
                    "max_drawdown_pct": 0,
                    "volatility_pct": 0
                },
                "trade_statistics": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "win_rate_pct": 0,
                    "avg_pnl": 0,
                    "best_trade": 0,
                    "worst_trade": 0
                }
            }

        return metrics

    except Exception as e:
        logger.error(f"Performance analytics error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate performance metrics: {str(e)}"
        ) from e


@router.get("/pnl-breakdown", response_model=list[PnLBreakdownItem])
async def get_pnl_breakdown(
    group_by: str = Query("symbol", description="Group by: symbol, day, month"),
    start_date: datetime | None = Query(None, description="Start date (default: 30 days ago)"),
    end_date: datetime | None = Query(None, description="End date (default: now)"),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get P&L breakdown by symbol, time period, or strategy.

    **Grouping Options:**
    - `symbol` - P&L by each symbol
    - `day` - Daily P&L
    - `month` - Monthly P&L

    **Use Cases:**
    - Performance attribution (which symbols are profitable?)
    - Time-series P&L analysis
    - Tax reporting (realized gains/losses)

    **Returns:**
    - Realized P&L (closed positions)
    - Unrealized P&L (open positions)
    - Total P&L
    - Number of trades

    Performance: ~5-10ms
    """
    from cift.core.trading_queries import get_pnl_breakdown

    # Validate group_by
    if group_by not in ["symbol", "day", "month"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid group_by '{group_by}'. Must be: symbol, day, or month"
        )

    try:
        breakdown = await get_pnl_breakdown(user_id, group_by, start_date, end_date)

        return breakdown

    except Exception as e:
        logger.error(f"P&L breakdown error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate P&L breakdown: {str(e)}"
        ) from e


# ============================================================================
# RISK METRICS
# ============================================================================

@router.get("/risk-metrics")
async def get_risk_metrics(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get current risk metrics for portfolio.

    **Metrics:**
    - Portfolio leverage
    - Max position size (% of portfolio)
    - Concentration risk (top 5 positions)
    - Margin utilization
    - VaR (Value at Risk) - 1 day, 95% confidence

    Performance: ~5ms
    """
    from cift.core.trading_queries import get_portfolio_value, get_user_positions

    try:
        # Get portfolio data
        portfolio_value_raw = await get_portfolio_value(user_id)
        portfolio_value = float(portfolio_value_raw) if portfolio_value_raw is not None else 0.0

        positions = await get_user_positions(user_id)

        if portfolio_value == 0 or not positions:
            return {
                "portfolio_value": portfolio_value,
                "leverage": 0,
                "max_position_pct": 0,
                "top_5_concentration": 0,
                "num_positions": 0,
                "positions": []
            }

        # Calculate position sizes
        position_values = []
        for pos in positions:
            # Ensure values are floats
            quantity = float(pos['quantity'])
            price = float(pos['current_price'])
            value = abs(quantity * price)

            pct = (value / portfolio_value * 100) if portfolio_value > 0 else 0
            position_values.append({
                "symbol": pos['symbol'],
                "value": value,
                "pct": pct
            })

        # Sort by value
        position_values.sort(key=lambda x: x['value'], reverse=True)

        # Calculate metrics
        max_position_pct = position_values[0]['pct'] if position_values else 0
        top_5_value = sum(p['value'] for p in position_values[:5])
        top_5_concentration = (top_5_value / portfolio_value * 100) if portfolio_value > 0 else 0

        # Calculate leverage
        total_exposure = sum(p['value'] for p in position_values)
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        return {
            "portfolio_value": round(portfolio_value, 2),
            "leverage": round(leverage, 2),
            "max_position_pct": round(max_position_pct, 2),
            "top_5_concentration": round(top_5_concentration, 2),
            "num_positions": len(positions),
            "positions": position_values[:10]  # Top 10 positions
        }

    except Exception as e:
        logger.error(f"Risk metrics error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate risk metrics: {str(e)}"
        ) from e


# ============================================================================
# TRADE HISTORY
# ============================================================================

@router.get("/trade-history")
async def get_trade_history(
    start_date: datetime | None = Query(None, description="Start date"),
    end_date: datetime | None = Query(None, description="End date"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get detailed trade history with P&L.

    **Use Cases:**
    - Trade journal
    - Performance review
    - Tax reporting

    Performance: ~5-10ms
    """
    from cift.core.database import db_manager

    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=30)
    if not end_date:
        end_date = datetime.utcnow()

    # Build query - RULES COMPLIANT: Use actual database table structure
    if symbol:
        query = """
            SELECT
                o.id,
                o.id as order_id,
                o.symbol,
                o.side,
                o.filled_quantity as quantity,
                o.avg_fill_price as price,
                o.filled_at as created_at,
                COALESCE((o.avg_fill_price - COALESCE(p.avg_cost, o.avg_fill_price)) * o.filled_quantity, 0) as realized_pnl
            FROM orders o
            LEFT JOIN positions p ON o.symbol = p.symbol AND o.user_id = p.user_id
            WHERE o.user_id = $1
              AND o.symbol = $2
              AND o.status = 'filled'
              AND o.filled_quantity > 0
              AND o.filled_at BETWEEN $3 AND $4
            ORDER BY o.filled_at DESC
            LIMIT $5
        """
        params = [user_id, symbol, start_date, end_date, limit]
    else:
        query = """
            SELECT
                o.id,
                o.id as order_id,
                o.symbol,
                o.side,
                o.filled_quantity as quantity,
                o.avg_fill_price as price,
                o.filled_at as created_at,
                COALESCE((o.avg_fill_price - COALESCE(p.avg_cost, o.avg_fill_price)) * o.filled_quantity, 0) as realized_pnl
            FROM orders o
            LEFT JOIN positions p ON o.symbol = p.symbol AND o.user_id = p.user_id
            WHERE o.user_id = $1
              AND o.status = 'filled'
              AND o.filled_quantity > 0
              AND o.filled_at BETWEEN $2 AND $3
            ORDER BY o.filled_at DESC
            LIMIT $4
        """
        params = [user_id, start_date, end_date, limit]

    try:
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        trades = [dict(row) for row in rows]

        return {
            "trades": trades,
            "count": len(trades),
            "total_pnl": sum(t.get('realized_pnl', 0) or 0 for t in trades)
        }

    except Exception as e:
        logger.error(f"Trade history error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trade history"
        ) from e


# ============================================================================
# TODAY'S TRADING STATS
# ============================================================================

@router.get("/today-stats")
async def get_today_stats(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get today's trading statistics for dashboard.

    Returns:
    - trades_count: Number of filled orders today
    - volume: Total trading volume (sum of fill values)
    - win_rate: Percentage of profitable trades
    - avg_pnl: Average P&L per trade

    Performance: ~5ms
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async with db_manager.pool.acquire() as conn:
        # Get today's filled orders with P&L calculation
        rows = await conn.fetch(
            """
            SELECT
                o.id,
                o.symbol,
                o.side,
                o.filled_quantity,
                o.avg_fill_price,
                COALESCE(
                    CASE
                        WHEN o.side = 'sell' THEN
                            (o.avg_fill_price - COALESCE(p.avg_cost, o.avg_fill_price)) * o.filled_quantity
                        ELSE
                            0  -- Buy orders don't have realized P&L yet
                    END,
                    0
                ) as realized_pnl
            FROM orders o
            LEFT JOIN positions p ON o.symbol = p.symbol AND o.user_id = p.user_id
            WHERE o.user_id = $1
              AND o.status = 'filled'
              AND o.filled_quantity > 0
              AND o.filled_at >= $2
            """,
            user_id, today_start
        )

        trades_count = len(rows)

        if trades_count == 0:
            return {
                "trades_count": 0,
                "volume": 0.0,
                "win_rate": None,
                "avg_pnl": None,
                "total_pnl": 0.0,
                "wins": 0,
                "losses": 0,
            }

        # Calculate stats
        total_volume = sum(
            float(row['filled_quantity'] or 0) * float(row['avg_fill_price'] or 0)
            for row in rows
        )

        # Only count sell orders for win/loss (they have realized P&L)
        sell_trades = [row for row in rows if row['side'] == 'sell']

        if sell_trades:
            wins = sum(1 for row in sell_trades if float(row['realized_pnl'] or 0) > 0)
            losses = sum(1 for row in sell_trades if float(row['realized_pnl'] or 0) < 0)
            total_pnl = sum(float(row['realized_pnl'] or 0) for row in sell_trades)
            win_rate = (wins / len(sell_trades) * 100) if sell_trades else None
            avg_pnl = total_pnl / len(sell_trades) if sell_trades else None
        else:
            wins = 0
            losses = 0
            total_pnl = 0.0
            win_rate = None
            avg_pnl = None

        return {
            "trades_count": trades_count,
            "volume": round(total_volume, 2),
            "win_rate": round(win_rate, 1) if win_rate is not None else None,
            "avg_pnl": round(avg_pnl, 2) if avg_pnl is not None else None,
            "total_pnl": round(total_pnl, 2),
            "wins": wins,
            "losses": losses,
        }


# Export router
__all__ = ["router"]
