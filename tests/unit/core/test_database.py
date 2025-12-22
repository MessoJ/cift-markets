"""
CIFT Markets - Database Tests

Real database connection tests (NO MOCKS - queries actual databases).
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from cift.core.database import (
    DatabaseManager,
    QuestDBManager,
    RedisManager,
    check_all_connections,
)


class TestDatabaseManager:
    """Test PostgreSQL database manager with real connections."""

    @pytest.mark.asyncio
    async def test_initialize_creates_connection(self):
        """Test that initialization creates working database connection."""
        manager = DatabaseManager()

        await manager.initialize()

        assert manager._is_initialized is True
        assert manager.engine is not None
        assert manager.async_session_maker is not None

        await manager.close()

    @pytest.mark.asyncio
    async def test_health_check_queries_database(self):
        """Test health check executes real query against PostgreSQL."""
        manager = DatabaseManager()
        await manager.initialize()

        # This queries the actual database
        result = await manager.health_check()

        assert result is True
        await manager.close()

    @pytest.mark.asyncio
    async def test_session_commits_transaction(self, db_session):
        """Test that database session commits real transactions."""
        # Execute real query
        result = await db_session.execute(text("SELECT 1 as value"))
        row = result.fetchone()

        assert row.value == 1

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self, db_session):
        """Test that session rolls back on errors."""
        with pytest.raises(SQLAlchemyError):
            # This will fail (invalid SQL)
            await db_session.execute(text("SELECT * FROM nonexistent_table"))

        # Session should still be usable after rollback
        result = await db_session.execute(text("SELECT 1"))
        assert result.scalar() == 1


class TestQuestDBManager:
    """Test QuestDB manager with real connections."""

    @pytest.mark.asyncio
    async def test_initialize_creates_pool(self):
        """Test that initialization creates connection pool."""
        manager = QuestDBManager()

        await manager.initialize()

        assert manager._is_initialized is True
        assert manager.pool is not None

        await manager.close()

    @pytest.mark.asyncio
    async def test_health_check_queries_questdb(self, questdb_conn):
        """Test health check executes real query against QuestDB."""
        # This queries the actual QuestDB instance
        result = await questdb_conn.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_execute_creates_table(self, questdb_conn):
        """Test execute method creates real table in QuestDB."""
        # Create a test table
        await questdb_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_ticks (
                symbol SYMBOL,
                price DOUBLE,
                volume LONG,
                timestamp TIMESTAMP
            ) timestamp(timestamp)
            """
        )

        # Verify table exists by querying it
        result = await questdb_conn.fetchrow("SELECT count() FROM test_ticks")
        assert result is not None

    @pytest.mark.asyncio
    async def test_fetch_returns_real_data(self, questdb_conn):
        """Test fetch returns actual query results."""
        # Create and insert data
        await questdb_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS test_market_data (
                symbol SYMBOL,
                price DOUBLE,
                timestamp TIMESTAMP
            ) timestamp(timestamp)
            """
        )

        # Query the table
        results = await questdb_conn.fetch("SELECT count() as cnt FROM test_market_data")

        assert len(results) > 0
        assert results[0]["cnt"] == 0  # Empty table


class TestRedisManager:
    """Test Redis manager with real connections."""

    @pytest.mark.asyncio
    async def test_initialize_creates_client(self):
        """Test that initialization creates Redis client."""
        manager = RedisManager()

        await manager.initialize()

        assert manager._is_initialized is True
        assert manager.client is not None

        await manager.close()

    @pytest.mark.asyncio
    async def test_health_check_pings_redis(self, redis_client):
        """Test health check pings actual Redis server."""
        # This pings the actual Redis instance
        result = await redis_client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_set_and_get_real_data(self, redis_client):
        """Test setting and getting actual data from Redis."""
        key = "test:market:AAPL"
        value = "150.50"

        # Set value in actual Redis
        await redis_client.set(key, value)

        # Get value from actual Redis
        retrieved = await redis_client.get(key)

        assert retrieved == value

    @pytest.mark.asyncio
    async def test_set_with_expiration(self, redis_client):
        """Test setting value with TTL in Redis."""
        key = "test:temp:data"
        value = "expires"

        # Set with 1 second expiration
        await redis_client.set(key, value, expire=1)

        # Should exist immediately
        result = await redis_client.get(key)
        assert result == value

        # TTL should be set
        ttl = await redis_client.client.ttl(key)
        assert ttl > 0

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, redis_client):
        """Test deleting keys from Redis."""
        key1 = "test:delete:1"
        key2 = "test:delete:2"

        # Set values
        await redis_client.set(key1, "value1")
        await redis_client.set(key2, "value2")

        # Delete keys
        deleted_count = await redis_client.delete(key1, key2)

        assert deleted_count == 2

        # Verify deleted
        assert await redis_client.get(key1) is None
        assert await redis_client.get(key2) is None


class TestConnectionHealth:
    """Test connection health checks across all databases."""

    @pytest.mark.asyncio
    async def test_check_all_connections_returns_status(self):
        """Test that check_all_connections queries all databases."""
        # This makes real connections to all services
        status = await check_all_connections()

        assert "postgres" in status
        assert "questdb" in status
        assert "redis" in status

        # All should be healthy if services are running
        # (test will fail if services aren't up - which is correct behavior)
        assert status["postgres"] == "healthy"
        assert status["questdb"] == "healthy"
        assert status["redis"] == "healthy"
