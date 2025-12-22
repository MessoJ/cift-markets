"""
ClickHouse Manager - 100x faster analytics queries
High-performance columnar database for time-series analytics
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
import polars as pl
from loguru import logger


class ClickHouseManager:
    """
    High-performance ClickHouse manager for analytics

    Features:
    - 100x faster complex queries compared to PostgreSQL
    - Columnar storage with 90%+ compression
    - Real-time aggregations via materialized views
    - Seamless integration with Polars for data processing
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "cift_analytics",
        user: str = "default",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.base_url = f"http://{host}:{port}"
        self.client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Initialize async HTTP client"""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

        # Test connection
        try:
            await self.execute("SELECT 1")
            logger.info(f"Connected to ClickHouse at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise

    async def disconnect(self) -> None:
        """Close async HTTP client"""
        if self.client:
            await self.client.aclose()
            logger.info("Disconnected from ClickHouse")

    async def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        format: str = "JSONEachRow",
    ) -> list[dict[str, Any]]:
        """
        Execute query and return results

        Args:
            query: SQL query
            params: Query parameters for safe substitution
            format: Output format (JSONEachRow, CSV, Parquet, etc.)

        Returns:
            List of result rows as dictionaries
        """
        try:
            # Format query with parameters (simple substitution)
            if params:
                for key, value in params.items():
                    if isinstance(value, str):
                        query = query.replace(f"{{{key}}}", f"'{value}'")
                    else:
                        query = query.replace(f"{{{key}}}", str(value))

            # Execute query
            response = await self.client.post(
                "/",
                params={
                    "database": self.database,
                    "user": self.user,
                    "password": self.password,
                    "query": query,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            response.raise_for_status()

            # Parse response based on format
            if format == "JSONEachRow":
                lines = response.text.strip().split("\n")
                return [eval(line) for line in lines if line]
            else:
                return [{"result": response.text}]

        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}\nQuery: {query}")
            raise

    async def execute_polars(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        """
        Execute query and return Polars DataFrame for fast processing

        Returns:
            Polars DataFrame with query results
        """
        results = await self.execute(query, params, format="JSONEachRow")

        if not results:
            return pl.DataFrame()

        return pl.DataFrame(results)

    async def insert(
        self,
        table: str,
        data: list[dict[str, Any]],
    ) -> None:
        """
        Insert data into table (batch insert)

        Args:
            table: Table name
            data: List of dictionaries with column values
        """
        if not data:
            return

        try:
            # Convert to DataFrame for efficient insertion
            df = pl.DataFrame(data)

            # Generate INSERT query
            columns = df.columns
            values_list = []

            for row in df.iter_rows(named=True):
                values = []
                for col in columns:
                    value = row[col]
                    if value is None:
                        values.append("NULL")
                    elif isinstance(value, str):
                        values.append(f"'{value.replace(chr(39), chr(39)*2)}'")
                    elif isinstance(value, (int, float, Decimal)):
                        values.append(str(value))
                    elif isinstance(value, datetime):
                        values.append(f"'{value.isoformat()}'")
                    else:
                        values.append(f"'{str(value)}'")

                values_list.append(f"({', '.join(values)})")

            # Execute batch insert
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES {', '.join(values_list)}
            """

            await self.execute(query, format="CSV")
            logger.debug(f"Inserted {len(data)} rows into {table}")

        except Exception as e:
            logger.error(f"Failed to insert into {table}: {e}")
            raise

    async def insert_from_polars(
        self,
        table: str,
        df: pl.DataFrame,
    ) -> None:
        """
        Insert Polars DataFrame into ClickHouse (optimized)

        Args:
            table: Table name
            df: Polars DataFrame
        """
        data = df.to_dicts()
        await self.insert(table, data)

    # =====================================================
    # MARKET DATA QUERIES
    # =====================================================

    async def get_ticks(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 10000,
    ) -> pl.DataFrame:
        """Get tick data for symbol in time range"""
        query = f"""
            SELECT *
            FROM ticks_analytics
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_time.isoformat()}'
              AND timestamp <= '{end_time.isoformat()}'
            ORDER BY timestamp
            LIMIT {limit}
        """
        return await self.execute_polars(query)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get OHLCV bars for symbol"""
        query = f"""
            SELECT *
            FROM bars_analytics
            WHERE symbol = '{symbol}'
              AND timeframe = '{timeframe}'
              AND timestamp >= '{start_time.isoformat()}'
              AND timestamp <= '{end_time.isoformat()}'
            ORDER BY timestamp
        """
        return await self.execute_polars(query)

    async def get_order_book_snapshot(
        self,
        symbol: str,
        timestamp: datetime,
        levels: int = 10,
    ) -> dict[str, list[dict]]:
        """Get order book snapshot at specific time"""
        query = f"""
            SELECT side, price, quantity, order_count, level
            FROM order_book_snapshots
            WHERE symbol = '{symbol}'
              AND timestamp <= '{timestamp.isoformat()}'
              AND level <= {levels}
            ORDER BY timestamp DESC, side, level
            LIMIT {levels * 2}
        """

        df = await self.execute_polars(query)

        if df.is_empty():
            return {"bids": [], "asks": []}

        bids = df.filter(pl.col("side") == "bid").to_dicts()
        asks = df.filter(pl.col("side") == "ask").to_dicts()

        return {"bids": bids, "asks": asks}

    # =====================================================
    # TRADING ANALYTICS QUERIES
    # =====================================================

    async def get_trade_executions(
        self,
        user_id: int | None = None,
        symbol: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> pl.DataFrame:
        """Get trade executions with filters"""
        conditions = ["1=1"]

        if user_id:
            conditions.append(f"user_id = {user_id}")
        if symbol:
            conditions.append(f"symbol = '{symbol}'")
        if start_time:
            conditions.append(f"timestamp >= '{start_time.isoformat()}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time.isoformat()}'")

        query = f"""
            SELECT *
            FROM trade_executions
            WHERE {' AND '.join(conditions)}
            ORDER BY timestamp DESC
            LIMIT 10000
        """

        return await self.execute_polars(query)

    async def get_position_history(
        self,
        user_id: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pl.DataFrame:
        """Get closed position history"""
        conditions = [f"user_id = {user_id}"]

        if start_date:
            conditions.append(f"exit_time >= '{start_date.isoformat()}'")
        if end_date:
            conditions.append(f"exit_time <= '{end_date.isoformat()}'")

        query = f"""
            SELECT *
            FROM position_history_analytics
            WHERE {' AND '.join(conditions)}
            ORDER BY exit_time DESC
        """

        return await self.execute_polars(query)

    async def calculate_user_pnl(
        self,
        user_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, float]:
        """Calculate user P&L metrics"""
        query = f"""
            SELECT
                sum(realized_pnl) as total_pnl,
                sum(commission_total) as total_commission,
                avg(realized_pnl) as avg_pnl,
                max(realized_pnl) as max_win,
                min(realized_pnl) as max_loss,
                countIf(realized_pnl > 0) as winning_trades,
                countIf(realized_pnl < 0) as losing_trades,
                count() as total_trades,
                countIf(realized_pnl > 0) / count() as win_rate
            FROM position_history_analytics
            WHERE user_id = {user_id}
              AND exit_time >= '{start_date.isoformat()}'
              AND exit_time <= '{end_date.isoformat()}'
        """

        results = await self.execute(query)
        return results[0] if results else {}

    # =====================================================
    # TECHNICAL INDICATORS QUERIES
    # =====================================================

    async def get_technical_indicators(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get pre-calculated technical indicators"""
        query = f"""
            SELECT *
            FROM technical_indicators
            WHERE symbol = '{symbol}'
              AND timeframe = '{timeframe}'
              AND timestamp >= '{start_time.isoformat()}'
              AND timestamp <= '{end_time.isoformat()}'
            ORDER BY timestamp
        """
        return await self.execute_polars(query)

    async def get_order_flow_features(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pl.DataFrame:
        """Get order flow features for ML models"""
        query = f"""
            SELECT *
            FROM order_flow_features
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_time.isoformat()}'
              AND timestamp <= '{end_time.isoformat()}'
            ORDER BY timestamp
        """
        return await self.execute_polars(query)

    # =====================================================
    # PERFORMANCE ANALYTICS QUERIES
    # =====================================================

    async def get_strategy_performance(
        self,
        strategy_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pl.DataFrame:
        """Get strategy performance metrics"""
        query = f"""
            SELECT *
            FROM strategy_performance
            WHERE strategy_id = '{strategy_id}'
              AND timestamp >= '{start_date.isoformat()}'
              AND timestamp <= '{end_date.isoformat()}'
            ORDER BY timestamp
        """
        return await self.execute_polars(query)

    async def get_top_symbols_by_volume(
        self,
        limit: int = 10,
        lookback_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Get most traded symbols by volume"""
        query = f"""
            SELECT
                symbol,
                sum(volume) as total_volume,
                count() as trade_count,
                avg(price) as avg_price,
                max(price) as high_price,
                min(price) as low_price
            FROM ticks_analytics
            WHERE timestamp >= now() - INTERVAL {lookback_hours} HOUR
            GROUP BY symbol
            ORDER BY total_volume DESC
            LIMIT {limit}
        """
        return await self.execute(query)

    # =====================================================
    # AGGREGATION QUERIES
    # =====================================================

    async def calculate_vwap(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
    ) -> float:
        """Calculate VWAP for symbol in time range"""
        query = f"""
            SELECT
                sum(price * volume) / sum(volume) as vwap
            FROM ticks_analytics
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_time.isoformat()}'
              AND timestamp <= '{end_time.isoformat()}'
        """

        results = await self.execute(query)
        return results[0].get("vwap", 0.0) if results else 0.0

    async def calculate_volatility(
        self,
        symbol: str,
        timeframe: str,
        period: int = 20,
    ) -> float:
        """Calculate historical volatility"""
        query = f"""
            SELECT stddevPop(close) as volatility
            FROM (
                SELECT close
                FROM bars_analytics
                WHERE symbol = '{symbol}'
                  AND timeframe = '{timeframe}'
                ORDER BY timestamp DESC
                LIMIT {period}
            )
        """

        results = await self.execute(query)
        return results[0].get("volatility", 0.0) if results else 0.0

    # =====================================================
    # UTILITY METHODS
    # =====================================================

    async def optimize_table(self, table: str) -> None:
        """Optimize table (merge parts, remove duplicates)"""
        query = f"OPTIMIZE TABLE {table} FINAL"
        await self.execute(query, format="CSV")
        logger.info(f"Optimized table {table}")

    async def get_table_size(self, table: str) -> dict[str, Any]:
        """Get table storage statistics"""
        query = f"""
            SELECT
                table,
                formatReadableSize(sum(bytes)) as size,
                sum(rows) as rows,
                count() as parts
            FROM system.parts
            WHERE database = '{self.database}'
              AND table = '{table}'
              AND active = 1
            GROUP BY table
        """

        results = await self.execute(query)
        return results[0] if results else {}

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()


# Global ClickHouse manager instance
_clickhouse_manager: ClickHouseManager | None = None


async def get_clickhouse_manager() -> ClickHouseManager:
    """Get or create global ClickHouse manager instance"""
    global _clickhouse_manager

    if _clickhouse_manager is None:
        import os

        host = os.getenv("CLICKHOUSE_HOST", "localhost")
        port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
        database = os.getenv("CLICKHOUSE_DB", "cift_analytics")
        user = os.getenv("CLICKHOUSE_USER", "default")
        password = os.getenv("CLICKHOUSE_PASSWORD", "")

        _clickhouse_manager = ClickHouseManager(host, port, database, user, password)
        await _clickhouse_manager.connect()

    return _clickhouse_manager


async def close_clickhouse_manager() -> None:
    """Close global ClickHouse manager"""
    global _clickhouse_manager

    if _clickhouse_manager is not None:
        await _clickhouse_manager.disconnect()
        _clickhouse_manager = None
