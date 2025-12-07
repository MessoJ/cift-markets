"""
CIFT Markets - Pytest Configuration

Fixtures for testing with real database connections (no mocks).
"""

import asyncio
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from cift.core.database import (
    db_manager,
    questdb_manager,
    redis_manager,
)
from cift.core.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


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
    
    # Provide session
    async with db_manager.get_session() as session:
        yield session
        await session.rollback()
    
    # Cleanup tables after test
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def questdb_conn():
    """Provide QuestDB connection for tests."""
    await questdb_manager.initialize()
    yield questdb_manager
    # Cleanup is handled by the manager


@pytest_asyncio.fixture(scope="function")
async def redis_client():
    """Provide Redis client for tests."""
    await redis_manager.initialize()
    yield redis_manager
    # Flush test data
    await redis_manager.client.flushdb()
