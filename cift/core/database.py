"""
CIFT Markets - Database Connection Management

Phase 5-7 Ultra-Low-Latency Stack:
- PostgreSQL: Relational data
- QuestDB: Real-time tick data (1.4M rows/sec)
- Dragonfly: Cache (25x faster than Redis)
- ClickHouse: Analytics (100x faster queries)
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from cift.core.config import settings
from cift.core.exceptions import DatabaseConnectionError

# SQLAlchemy Base
Base = declarative_base()


# ============================================================================
# POSTGRESQL CONNECTION (SQLAlchemy)
# ============================================================================

class DatabaseManager:
    """Manage PostgreSQL connections with pooling and health checks."""

    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self.pool = None  # asyncpg pool for raw queries
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize database connection pool with timeout protection."""
        if self._is_initialized:
            return

        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}@"
                f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}",
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=settings.debug_sql,
                connect_args={"timeout": 3},  # 3 second connection timeout
            )

            # Create asyncpg pool for raw queries (used in auth) with timeout
            self.pool = await asyncio.wait_for(
                asyncpg.create_pool(
                    host=settings.postgres_host,
                    port=settings.postgres_port,
                    user=settings.postgres_user,
                    password=settings.postgres_password,
                    database=settings.postgres_db,
                    min_size=5,
                    max_size=20,
                    command_timeout=60,
                    timeout=3,  # 3 second timeout
                ),
                timeout=5.0  # Overall timeout for pool creation
            )

            # Create session factory
            self.async_session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Test connection with timeout
            await asyncio.wait_for(self.health_check(), timeout=2.0)

            self._is_initialized = True
            logger.info("✅ PostgreSQL connection pool initialized (SQLAlchemy + asyncpg)")

        except TimeoutError:
            logger.error("PostgreSQL connection timed out - database may not be running")
            raise DatabaseConnectionError(
                "PostgreSQL connection timeout - database unavailable",
                details={"error": "Connection timeout after 5 seconds"},
            )
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise DatabaseConnectionError(
                "Failed to connect to PostgreSQL",
                details={"error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """Check if database is healthy."""
        try:
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup."""
        if not self._is_initialized:
            await self.initialize()

        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close database connections."""
        if self.pool:
            await self.pool.close()
        if self.engine:
            await self.engine.dispose()
            logger.info("PostgreSQL connection pools closed")
            self._is_initialized = False


# ============================================================================
# QUESTDB CONNECTION (asyncpg)
# ============================================================================

class QuestDBManager:
    """Manage QuestDB connections for time-series data."""

    def __init__(self):
        self.pool = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize QuestDB connection pool."""
        if self._is_initialized:
            return

        try:
            # Create connection pool for QuestDB PostgreSQL wire protocol
            self.pool = await asyncpg.create_pool(
                host=settings.questdb_host,
                port=settings.questdb_pg_port,
                user="admin",
                password="quest",
                database="qdb",
                min_size=5,
                max_size=20,
                command_timeout=60,
            )

            # Test connection
            await self.health_check()

            self._is_initialized = True
            logger.info("✅ QuestDB connection pool initialized")

        except Exception as e:
            logger.error(f"Failed to initialize QuestDB: {e}")
            raise DatabaseConnectionError(
                "Failed to connect to QuestDB",
                details={"error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """Check if QuestDB is healthy."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"QuestDB health check failed: {e}")
            return False

    async def execute(self, query: str, *args) -> None:
        """Execute a query without returning results."""
        if not self._is_initialized:
            await self.initialize()

        async with self.pool.acquire() as conn:
            await conn.execute(query, *args)

    async def fetch(self, query: str, *args):
        """Fetch all results from a query."""
        if not self._is_initialized:
            await self.initialize()

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """Fetch a single row from a query."""
        if not self._is_initialized:
            await self.initialize()

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def close(self) -> None:
        """Close QuestDB connections."""
        if self.pool:
            await self.pool.close()
            logger.info("QuestDB connection pool closed")
            self._is_initialized = False


# ============================================================================
# DRAGONFLY CONNECTION (Redis-compatible, 25x faster)
# ============================================================================

class RedisManager:
    """
    Manage Dragonfly connections for caching and real-time data.

    Dragonfly is 100% Redis API compatible but provides:
    - 25x higher throughput (2.5M ops/sec)
    - 80% less memory usage
    - Vertical scaling

    Note: Called RedisManager for backward compatibility.
    """

    def __init__(self):
        self.client = None
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize Dragonfly connection (Redis-compatible)."""
        if self._is_initialized:
            return

        try:
            # Create Dragonfly client with connection pooling (uses Redis protocol)
            self.client = await redis.from_url(
                settings.dragonfly_url,  # Uses settings.redis_url alias
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
            )

            # Test connection
            await self.health_check()

            self._is_initialized = True
            logger.info("✅ Dragonfly cache connection initialized (25x faster)")

        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise DatabaseConnectionError(
                "Failed to connect to Redis",
                details={"error": str(e)},
            ) from e

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            pong = await self.client.ping()
            return pong is True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def get(self, key: str) -> str | None:
        """Get value from Redis."""
        if not self._is_initialized:
            await self.initialize()
        return await self.client.get(key)

    async def set(
        self,
        key: str,
        value: str,
        expire: int | None = None,
    ) -> bool:
        """Set value in Redis with optional expiration."""
        if not self._is_initialized:
            await self.initialize()
        return await self.client.set(key, value, ex=expire)

    async def delete(self, *keys: str) -> int:
        """Delete keys from Redis."""
        if not self._is_initialized:
            await self.initialize()
        return await self.client.delete(*keys)

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")
            self._is_initialized = False


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Create global instances
db_manager = DatabaseManager()
questdb_manager = QuestDBManager()
redis_manager = RedisManager()  # Actually Dragonfly (Redis-compatible)

# Phase 5-7 Advanced managers (imported lazily)
_clickhouse_manager = None
_nats_manager = None


async def get_clickhouse():
    """Get ClickHouse manager instance (lazy load)."""
    global _clickhouse_manager
    if _clickhouse_manager is None:
        from cift.core.clickhouse_manager import get_clickhouse_manager
        _clickhouse_manager = await get_clickhouse_manager()
    return _clickhouse_manager


async def get_nats():
    """Get NATS manager instance (lazy load)."""
    global _nats_manager
    if _nats_manager is None:
        from cift.core.nats_manager import get_nats_manager
        _nats_manager = await get_nats_manager()
    return _nats_manager


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions."""
    async with db_manager.get_session() as session:
        yield session


async def get_redis() -> redis.Redis:
    """FastAPI dependency for Redis client."""
    if not redis_manager._is_initialized:
        await redis_manager.initialize()
    return redis_manager.client


async def get_postgres_pool() -> asyncpg.Pool:
    """
    FastAPI dependency for PostgreSQL asyncpg pool.

    Returns raw asyncpg pool for direct SQL queries.
    Use this for high-performance queries in new route modules.

    Best Practice: Prefer this over ORM for:
    - Simple CRUD operations
    - High-throughput endpoints
    - Direct SQL queries
    """
    if not db_manager._is_initialized:
        await db_manager.initialize()
    return db_manager.pool


async def get_questdb_pool() -> asyncpg.Pool:
    """
    FastAPI dependency for QuestDB connection pool.

    Returns asyncpg pool for time-series queries.
    Use this for:
    - Real-time market data
    - High-frequency tick data
    - Time-series analytics
    """
    if not questdb_manager._is_initialized:
        await questdb_manager.initialize()
    return questdb_manager.pool


# ============================================================================
# HEALTH CHECK
# ============================================================================

async def check_all_connections() -> dict[str, str]:
    """Check health of all database connections (Phase 5-7)."""
    results = {}

    # PostgreSQL
    try:
        if not db_manager._is_initialized:
            await db_manager.initialize()
        pg_healthy = await db_manager.health_check()
        results["postgres"] = "healthy" if pg_healthy else "unhealthy"
    except Exception as e:
        results["postgres"] = f"error: {str(e)}"

    # QuestDB
    try:
        if not questdb_manager._is_initialized:
            await questdb_manager.initialize()
        qdb_healthy = await questdb_manager.health_check()
        results["questdb"] = "healthy" if qdb_healthy else "unhealthy"
    except Exception as e:
        results["questdb"] = f"error: {str(e)}"

    # Dragonfly (Redis-compatible)
    try:
        if not redis_manager._is_initialized:
            await redis_manager.initialize()
        redis_healthy = await redis_manager.health_check()
        results["dragonfly"] = "healthy" if redis_healthy else "unhealthy"
    except Exception as e:
        results["dragonfly"] = f"error: {str(e)}"

    # ClickHouse (Phase 5-7)
    try:
        ch = await get_clickhouse()
        result = await ch.execute("SELECT 1")
        results["clickhouse"] = "healthy" if result else "unhealthy"
    except Exception as e:
        results["clickhouse"] = f"error: {str(e)}"

    # NATS JetStream (Phase 5-7)
    try:
        nats = await get_nats()
        results["nats"] = "healthy" if nats.is_connected else "unhealthy"
    except Exception as e:
        results["nats"] = f"error: {str(e)}"

    return results


async def initialize_all_connections() -> None:
    """Initialize all database connections on startup with graceful degradation."""
    # Try to initialize each database with timeout protection
    # Allows backend to start even if some databases are unavailable

    # PostgreSQL (critical for auth/trading)
    try:
        await asyncio.wait_for(db_manager.initialize(), timeout=5.0)
    except (TimeoutError, Exception) as e:
        logger.warning(f"⚠️  PostgreSQL unavailable: {e}")
        logger.warning("Auth and trading endpoints will be unavailable")

    # QuestDB (for tick data)
    try:
        await asyncio.wait_for(questdb_manager.initialize(), timeout=3.0)
    except (TimeoutError, Exception) as e:
        logger.warning(f"⚠️  QuestDB unavailable: {e}")
        logger.warning("Real-time market data storage will be unavailable")

    # Redis/Dragonfly (for caching)
    try:
        await asyncio.wait_for(redis_manager.initialize(), timeout=3.0)
    except (TimeoutError, Exception) as e:
        logger.warning(f"⚠️  Redis unavailable: {e}")
        logger.warning("Caching will be unavailable, performance may be degraded")

    # Phase 5-7 advanced services (lazy load)
    logger.info("Phase 5-7 services (ClickHouse, NATS) will initialize on first use")


async def close_all_connections() -> None:
    """Close all database connections on shutdown (Phase 5-7)."""
    # Close core databases
    await asyncio.gather(
        db_manager.close(),
        questdb_manager.close(),
        redis_manager.close(),
    )

    # Close Phase 5-7 services if initialized
    if _clickhouse_manager is not None:
        from cift.core.clickhouse_manager import close_clickhouse_manager
        await close_clickhouse_manager()

    if _nats_manager is not None:
        await _nats_manager.disconnect()
