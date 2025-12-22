"""
CIFT Markets - Pytest Configuration

Fixtures for testing with real database connections (no mocks).
"""

from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from cift.core.database import DatabaseManager, QuestDBManager, RedisManager
from cift.core.models import Base


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide database session for tests with real PostgreSQL connection.
    Each test gets a fresh session with automatic rollback.
    """
    # Use a fresh manager per test to avoid sharing asyncpg/SQLAlchemy state
    # across pytest-asyncio event loops.
    manager = DatabaseManager()
    await manager.initialize()

    try:
        # Create tables
        async with manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Provide session (do not auto-commit; tests control transaction behavior)
        async with manager.async_session_maker() as session:
            try:
                yield session
            finally:
                await session.rollback()

        # Cleanup tables after test
        async with manager.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    finally:
        await manager.close()


@pytest_asyncio.fixture(scope="function")
async def questdb_conn():
    """Provide QuestDB connection for tests."""
    manager = QuestDBManager()
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.close()


@pytest_asyncio.fixture(scope="function")
async def redis_client():
    """Provide Redis client for tests."""
    manager = RedisManager()
    await manager.initialize()
    try:
        yield manager
    finally:
        # Flush test data and close connection.
        await manager.client.flushdb()
        await manager.close()
