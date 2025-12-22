"""
CIFT Markets - Pytest Configuration

Fixtures for testing with real database connections (no mocks).
"""

from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from cift.core.database import db_manager, questdb_manager, redis_manager
from cift.core.models import Base


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Provide database session for tests with real PostgreSQL connection.
    Each test gets a fresh session with automatic rollback.
    """
    # Initialize database
    await db_manager.initialize()

    # Create tables
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Provide session (do not auto-commit; tests control transaction behavior)
    async with db_manager.async_session_maker() as session:
        try:
            yield session
        finally:
            await session.rollback()

    # Cleanup tables after test
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    # Important with pytest-asyncio loop-per-test: ensure pools/engine aren't reused across loops.
    await db_manager.close()


@pytest_asyncio.fixture(scope="function")
async def questdb_conn():
    """Provide QuestDB connection for tests."""
    await questdb_manager.initialize()
    try:
        yield questdb_manager
    finally:
        await questdb_manager.close()


@pytest_asyncio.fixture(scope="function")
async def redis_client():
    """Provide Redis client for tests."""
    await redis_manager.initialize()
    try:
        yield redis_manager
    finally:
        # Flush test data and close connection.
        await redis_manager.client.flushdb()
        await redis_manager.close()
