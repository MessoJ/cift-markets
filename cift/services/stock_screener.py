"""
CIFT Markets - Advanced Stock Screener Service

Real-time stock screening with technical and fundamental analysis.
Supports complex filtering, custom criteria, and saved screens.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from loguru import logger
from pydantic import BaseModel

from cift.core.database import get_postgres_pool


class ScreenerOperator(str, Enum):
    GT = ">"      # Greater than
    LT = "<"      # Less than
    GTE = ">="    # Greater than or equal
    LTE = "<="    # Less than or equal
    EQ = "="      # Equal to
    NE = "!="     # Not equal to
    BETWEEN = "between"
    IN = "in"
    NOT_IN = "not_in"


@dataclass
class ScreenerCriteria:
    field: str
    operator: ScreenerOperator
    value: Any
    value2: Any | None = None  # For BETWEEN operator


class ScreenerRequest(BaseModel):
    name: str = "Custom Screen"
    criteria: list[dict]  # List of criteria dicts
    sort_by: str = "market_cap"
    sort_order: str = "desc"  # asc or desc
    limit: int = 100
    save_screen: bool = False


class StockScreenerService:
    """Advanced stock screening with real-time data."""

    def __init__(self):
        self.supported_fields = {
            # Price & Volume
            "price": "Current price",
            "change_percent": "Daily change %",
            "volume": "Volume",
            "avg_volume_10d": "10-day avg volume",
            "market_cap": "Market capitalization",

            # Technical Indicators
            "rsi": "RSI (14)",
            "sma_20": "20-day SMA",
            "sma_50": "50-day SMA",
            "sma_200": "200-day SMA",
            "ema_12": "12-day EMA",
            "ema_26": "26-day EMA",
            "macd": "MACD",
            "bollinger_upper": "Bollinger Upper",
            "bollinger_lower": "Bollinger Lower",

            # Fundamentals (mock for now)
            "pe_ratio": "P/E Ratio",
            "pb_ratio": "P/B Ratio",
            "dividend_yield": "Dividend Yield %",
            "eps": "Earnings Per Share",
            "revenue_growth": "Revenue Growth %",

            # Performance
            "return_1d": "1-day return %",
            "return_1w": "1-week return %",
            "return_1m": "1-month return %",
            "return_3m": "3-month return %",
            "return_1y": "1-year return %",

            # Risk Metrics
            "beta": "Beta",
            "volatility_30d": "30-day volatility",
            "max_drawdown": "Max drawdown %",
        }

    async def screen_stocks(self, request: ScreenerRequest, user_id: str = None) -> dict:
        """Execute stock screening with given criteria."""

        logger.info(f"Running stock screen: {request.name} with {len(request.criteria)} criteria")

        try:
            # Build screening query
            query, params = await self._build_screening_query(request)

            # Execute query
            pool = await get_postgres_pool()
            async with pool.acquire() as conn:
                results = await conn.fetch(query, *params)

            # Format results
            formatted_results = [dict(row) for row in results]

            # Calculate additional metrics
            for result in formatted_results:
                await self._enrich_stock_data(result)

            # Save screen if requested
            screen_id = None
            if request.save_screen and user_id:
                screen_id = await self._save_screen(request, user_id)

            return {
                "total_results": len(formatted_results),
                "results": formatted_results,
                "criteria": request.criteria,
                "sort_by": request.sort_by,
                "sort_order": request.sort_order,
                "screen_id": screen_id,
                "generated_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Stock screening failed: {e}")
            raise

    async def _build_screening_query(self, request: ScreenerRequest) -> tuple:
        """Build PostgreSQL query for screening."""

        # Base query with mock data (replace with real market data)
        base_query = """
        WITH stock_data AS (
            SELECT
                symbol,
                price,
                change_percent,
                volume,
                market_cap,
                -- Technical indicators (mock)
                RANDOM() * 100 as rsi,
                price * (0.95 + RANDOM() * 0.1) as sma_20,
                price * (0.90 + RANDOM() * 0.2) as sma_50,
                price * (0.85 + RANDOM() * 0.3) as sma_200,
                -- Fundamentals (mock)
                15 + RANDOM() * 30 as pe_ratio,
                1 + RANDOM() * 5 as pb_ratio,
                RANDOM() * 8 as dividend_yield,
                -- Performance (mock)
                (RANDOM() - 0.5) * 10 as return_1d,
                (RANDOM() - 0.5) * 20 as return_1w,
                (RANDOM() - 0.5) * 30 as return_1m,
                -- Risk metrics (mock)
                0.5 + RANDOM() * 1.5 as beta,
                10 + RANDOM() * 20 as volatility_30d
            FROM market_data_cache
            WHERE symbol IN ('AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE')
        )
        SELECT * FROM stock_data
        """

        where_conditions = []
        params = []

        # Process criteria
        for _i, criteria in enumerate(request.criteria):
            field = criteria['field']
            operator = criteria['operator']
            value = criteria['value']

            if field not in self.supported_fields:
                continue

            param_idx = len(params) + 1

            if operator == ">":
                where_conditions.append(f"{field} > ${param_idx}")
                params.append(value)
            elif operator == "<":
                where_conditions.append(f"{field} < ${param_idx}")
                params.append(value)
            elif operator == ">=":
                where_conditions.append(f"{field} >= ${param_idx}")
                params.append(value)
            elif operator == "<=":
                where_conditions.append(f"{field} <= ${param_idx}")
                params.append(value)
            elif operator == "=":
                where_conditions.append(f"{field} = ${param_idx}")
                params.append(value)
            elif operator == "between":
                where_conditions.append(f"{field} BETWEEN ${param_idx} AND ${param_idx + 1}")
                params.extend([value, criteria.get('value2')])

        # Add WHERE clause
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)

        # Add ORDER BY
        sort_field = request.sort_by if request.sort_by in self.supported_fields else "market_cap"
        sort_order = "DESC" if request.sort_order.upper() == "DESC" else "ASC"
        base_query += f" ORDER BY {sort_field} {sort_order}"

        # Add LIMIT
        base_query += f" LIMIT {min(request.limit, 500)}"

        return base_query, params

    async def _enrich_stock_data(self, stock_data: dict):
        """Add calculated fields and real-time data."""

        # Technical analysis signals
        price = float(stock_data.get('price', 0))
        sma_20 = float(stock_data.get('sma_20', 0))
        sma_50 = float(stock_data.get('sma_50', 0))

        stock_data['trend_signal'] = "bullish" if price > sma_20 > sma_50 else "bearish"
        stock_data['momentum'] = "strong" if abs(stock_data.get('change_percent', 0)) > 3 else "weak"

        # Risk classification
        beta = stock_data.get('beta', 1.0)
        stock_data['risk_level'] = "high" if beta > 1.5 else "medium" if beta > 0.8 else "low"

        # Market cap category
        market_cap = stock_data.get('market_cap', 0)
        if market_cap > 200_000_000_000:
            stock_data['cap_category'] = "mega"
        elif market_cap > 10_000_000_000:
            stock_data['cap_category'] = "large"
        elif market_cap > 2_000_000_000:
            stock_data['cap_category'] = "mid"
        else:
            stock_data['cap_category'] = "small"

    async def _save_screen(self, request: ScreenerRequest, user_id: str) -> str:
        """Save screen criteria for future use."""

        screen_id = f"screen_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO saved_screens (
                    id, user_id, name, criteria, sort_by, sort_order, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (id) DO NOTHING
            """,
                screen_id, user_id, request.name, json.dumps(request.criteria),
                request.sort_by, request.sort_order, datetime.utcnow()
            )

        logger.info(f"Saved screen: {screen_id}")
        return screen_id

    async def get_saved_screens(self, user_id: str) -> list[dict]:
        """Get user's saved screens."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, name, criteria, sort_by, sort_order, created_at, last_run
                FROM saved_screens
                WHERE user_id = $1
                ORDER BY created_at DESC
            """, user_id)

        return [
            {
                "screen_id": row['id'],
                "name": row['name'],
                "criteria": json.loads(row['criteria']),
                "sort_by": row['sort_by'],
                "sort_order": row['sort_order'],
                "created_at": row['created_at'],
                "last_run": row['last_run'],
            }
            for row in rows
        ]

    async def run_saved_screen(self, screen_id: str, user_id: str) -> dict:
        """Run a previously saved screen."""

        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT name, criteria, sort_by, sort_order
                FROM saved_screens
                WHERE id = $1 AND user_id = $2
            """, screen_id, user_id)

            if not row:
                raise ValueError(f"Screen {screen_id} not found")

            # Update last run time
            await conn.execute("""
                UPDATE saved_screens SET last_run = $3
                WHERE id = $1 AND user_id = $2
            """, screen_id, user_id, datetime.utcnow())

        # Build request from saved data
        request = ScreenerRequest(
            name=row['name'],
            criteria=json.loads(row['criteria']),
            sort_by=row['sort_by'],
            sort_order=row['sort_order'],
            save_screen=False
        )

        return await self.screen_stocks(request, user_id)

    async def get_popular_screens(self) -> list[dict]:
        """Get popular/preset screening templates."""

        return [
            {
                "name": "High Growth Momentum",
                "description": "Stocks with strong price momentum and growth",
                "criteria": [
                    {"field": "change_percent", "operator": ">", "value": 5},
                    {"field": "rsi", "operator": "between", "value": 40, "value2": 70},
                    {"field": "market_cap", "operator": ">", "value": 1_000_000_000},
                ],
                "sort_by": "change_percent",
                "sort_order": "desc"
            },
            {
                "name": "Value Opportunities",
                "description": "Undervalued stocks with good fundamentals",
                "criteria": [
                    {"field": "pe_ratio", "operator": "<", "value": 15},
                    {"field": "pb_ratio", "operator": "<", "value": 2},
                    {"field": "dividend_yield", "operator": ">", "value": 2},
                ],
                "sort_by": "pe_ratio",
                "sort_order": "asc"
            },
            {
                "name": "Breakout Candidates",
                "description": "Stocks breaking above key resistance levels",
                "criteria": [
                    {"field": "volume", "operator": ">", "value": 1_000_000},
                    {"field": "rsi", "operator": ">", "value": 60},
                    {"field": "change_percent", "operator": ">", "value": 3},
                ],
                "sort_by": "volume",
                "sort_order": "desc"
            },
            {
                "name": "Dividend Champions",
                "description": "High-quality dividend paying stocks",
                "criteria": [
                    {"field": "dividend_yield", "operator": ">", "value": 3},
                    {"field": "market_cap", "operator": ">", "value": 10_000_000_000},
                    {"field": "beta", "operator": "<", "value": 1.2},
                ],
                "sort_by": "dividend_yield",
                "sort_order": "desc"
            }
        ]


# Global screener instance
_screener_service = None

def get_screener_service() -> StockScreenerService:
    """Get global screener service instance."""
    global _screener_service
    if _screener_service is None:
        _screener_service = StockScreenerService()
    return _screener_service
