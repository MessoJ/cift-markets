"""
CIFT Markets - Transaction API Routes

Account transaction history and cash flow analysis.
"""

from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from cift.core.auth import User, get_current_active_user
from cift.core.database import db_manager

router = APIRouter(prefix="/transactions", tags=["Transactions"])


async def get_current_user_id(current_user: User = Depends(get_current_active_user)) -> UUID:
    return current_user.id


# ============================================================================
# TRANSACTION HISTORY
# ============================================================================


@router.get("")
async def get_transactions(
    transaction_type: str | None = Query(
        None, regex="^(deposit|withdrawal|trade|dividend|interest|fee|commission|adjustment)$"
    ),
    symbol: str | None = Query(None, description="Filter by symbol (for trades)"),
    start_date: datetime | None = Query(None),
    end_date: datetime | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get transaction history with filters.

    **Filters:**
    - transaction_type: deposit, withdrawal, trade, etc.
    - symbol: Filter by trading symbol
    - start_date/end_date: Date range
    - limit/offset: Pagination

    Performance: ~5-10ms
    """
    if not start_date:
        start_date = datetime.utcnow() - timedelta(days=90)
    if not end_date:
        end_date = datetime.utcnow()

    # Build query
    query = """
        SELECT
            t.id, t.transaction_type, t.amount, t.balance_after,
            t.symbol, t.description, t.external_ref,
            t.transaction_date, t.created_at, t.order_id,
            o.filled_quantity as quantity,
            o.avg_fill_price as price
        FROM transactions t
        LEFT JOIN orders o ON t.order_id = o.id
        WHERE t.user_id = $1 AND t.transaction_date BETWEEN $2 AND $3
    """
    params = [user_id, start_date, end_date]
    param_idx = 4

    if transaction_type:
        # Support comma-separated types
        types = transaction_type.split(",")
        if len(types) > 1:
            query += f" AND t.transaction_type = ANY(${param_idx})"
            params.append(types)
        else:
            query += f" AND t.transaction_type = ${param_idx}"
            params.append(transaction_type)
        param_idx += 1

    if symbol:
        query += f" AND t.symbol = ${param_idx}"
        params.append(symbol.upper())
        param_idx += 1

    query += f" ORDER BY t.transaction_date DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
    params.extend([limit, offset])

    async with db_manager.pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        # Get total count
        count_query = """
            SELECT COUNT(*) FROM transactions t
            WHERE t.user_id = $1 AND t.transaction_date BETWEEN $2 AND $3
        """
        count_params = [user_id, start_date, end_date]
        count_param_idx = 4

        if transaction_type:
            types = transaction_type.split(",")
            if len(types) > 1:
                count_query += f" AND t.transaction_type = ANY(${count_param_idx})"
                count_params.append(types)
            else:
                count_query += f" AND t.transaction_type = ${count_param_idx}"
                count_params.append(transaction_type)
            count_param_idx += 1

        if symbol:
            count_query += f" AND t.symbol = ${count_param_idx}"
            count_params.append(symbol.upper())

        total = await conn.fetchval(count_query, *count_params)

    return {
        "transactions": [
            {
                **dict(row),
                "quantity": float(row["quantity"]) if row["quantity"] is not None else None,
                "price": float(row["price"]) if row["price"] is not None else None,
                "amount": float(row["amount"]),
                "balance_after": float(row["balance_after"]),
            }
            for row in rows
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
        },
    }


@router.get("/summary")
async def get_transaction_summary(
    days: int = Query(30, ge=1, le=365),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Transaction summary and statistics.
    Performance: ~5-10ms
    """
    start_date = datetime.utcnow() - timedelta(days=days)

    async with db_manager.pool.acquire() as conn:
        # Get aggregations by type
        type_summary = await conn.fetch(
            """
            SELECT
                transaction_type,
                COUNT(*) as count,
                SUM(amount) as total_amount
            FROM transactions
            WHERE user_id = $1 AND transaction_date >= $2
            GROUP BY transaction_type
            ORDER BY transaction_type
        """,
            user_id,
            start_date,
        )

        # Daily aggregations
        daily = await conn.fetch(
            """
            SELECT
                DATE(transaction_date) as date,
                SUM(amount) as net_flow,
                COUNT(*) as count
            FROM transactions
            WHERE user_id = $1 AND transaction_date >= $2
            GROUP BY DATE(transaction_date)
            ORDER BY date DESC
        """,
            user_id,
            start_date,
        )

        # Total stats
        stats = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 0) as total_inflow,
                COALESCE(SUM(CASE WHEN amount < 0 THEN amount ELSE 0 END), 0) as total_outflow,
                COALESCE(SUM(amount), 0) as net_flow
            FROM transactions
            WHERE user_id = $1 AND transaction_date >= $2
        """,
            user_id,
            start_date,
        )

    return {
        "period": {
            "start_date": start_date.isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "days": days,
        },
        "by_type": [
            {
                "transaction_type": row["transaction_type"],
                "count": row["count"],
                "total_amount": float(row["total_amount"]),
            }
            for row in type_summary
        ],
        "daily": [
            {
                "date": row["date"].isoformat(),
                "net_flow": float(row["net_flow"]),
                "count": row["count"],
            }
            for row in daily
        ],
        "stats": {
            "total_inflow": float(stats["total_inflow"]),
            "total_outflow": float(stats["total_outflow"]),
            "net_flow": float(stats["net_flow"]),
        },
    }


@router.get("/cash-flow")
async def get_cash_flow_analysis(
    days: int = Query(90, ge=1, le=365),
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Cash flow analysis: money in vs money out.
    Performance: ~5-10ms (ClickHouse) or ~10-15ms (PostgreSQL)
    """
    start_date = datetime.utcnow() - timedelta(days=days)

    # Try ClickHouse first (Phase 5-7)
    try:
        import polars as pl

        from cift.core.clickhouse_manager import get_clickhouse_manager

        ch = await get_clickhouse_manager()

        query = f"""
            SELECT
                toDate(transaction_date) as date,
                sumIf(amount, amount > 0) as money_in,
                sumIf(amount, amount < 0) as money_out,
                sum(amount) as net_flow
            FROM transactions
            WHERE user_id = '{user_id}' AND transaction_date >= '{start_date.isoformat()}'
            GROUP BY date
            ORDER BY date ASC
            FORMAT JSONEachRow
        """

        result = await ch.query(query)
        df = pl.read_ndjson(result.encode())

        # Calculate cumulative
        df = df.with_columns([pl.col("net_flow").cum_sum().alias("cumulative_flow")])

        logger.info("✅ Cash flow via ClickHouse + Polars")

        return {
            "data": df.to_dicts(),
            "summary": {
                "total_in": round(float(df["money_in"].sum()), 2),
                "total_out": round(float(df["money_out"].sum()), 2),
                "net_flow": round(float(df["net_flow"].sum()), 2),
            },
            "_backend": "clickhouse",
        }

    except Exception as e:
        logger.warning(f"ClickHouse unavailable, using PostgreSQL: {e}")

        # Fallback to PostgreSQL
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    DATE(transaction_date) as date,
                    SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) as money_in,
                    SUM(CASE WHEN amount < 0 THEN ABS(amount) ELSE 0 END) as money_out,
                    SUM(amount) as net_flow
                FROM transactions
                WHERE user_id = $1 AND transaction_date >= $2
                GROUP BY DATE(transaction_date)
                ORDER BY date ASC
            """,
                user_id,
                start_date,
            )

            data = [dict(row) for row in rows]

            # Calculate cumulative
            cumulative = 0
            for item in data:
                cumulative += item["net_flow"]
                item["cumulative_flow"] = round(cumulative, 2)

            total_in = sum(float(d["money_in"]) for d in data)
            total_out = sum(float(d["money_out"]) for d in data)

        return {
            "data": data,
            "summary": {
                "total_in": round(total_in, 2),
                "total_out": round(total_out, 2),
                "net_flow": round(total_in - total_out, 2),
            },
            "_backend": "postgresql",
        }


@router.get("/{transaction_id}")
async def get_transaction_detail(
    transaction_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get single transaction details.
    Performance: ~2-3ms
    """
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM transactions
            WHERE id = $1 AND user_id = $2
        """,
            transaction_id,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="Transaction not found")

        transaction = dict(row)

        # If trade, get order details
        if transaction["order_id"]:
            order = await conn.fetchrow(
                """
                SELECT
                    id, symbol, side, order_type, quantity,
                    filled_quantity, avg_fill_price, status,
                    created_at, filled_at
                FROM orders
                WHERE id = $1
            """,
                transaction["order_id"],
            )

            if order:
                transaction["order_details"] = dict(order)

    return {"transaction": transaction}


__all__ = ["router"]
