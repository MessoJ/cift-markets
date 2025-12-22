"""
CIFT Markets - ORM Models Tests

Test database models with real database inserts and queries.
"""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import select

from cift.core.models import (
    Alert,
    APIKey,
    AuditLog,
    Backtest,
    TradingStrategy,
    User,
)


class TestUserModel:
    """Test User model with real database operations."""

    @pytest.mark.asyncio
    async def test_create_user_in_database(self, db_session):
        """Test creating user record in actual PostgreSQL."""
        user = User(
            email="test@ciftmarkets.com",
            username="testuser",
            hashed_password="$2b$12$hash",
            full_name="Test User",
            is_active=True,
            is_superuser=False,
        )

        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Verify user was actually created with ID
        assert user.id is not None
        assert isinstance(user.id, uuid.UUID)
        assert user.email == "test@ciftmarkets.com"
        assert user.created_at is not None

    @pytest.mark.asyncio
    async def test_query_user_from_database(self, db_session):
        """Test querying user from actual database."""
        # Create user
        user = User(email="query@test.com", username="queryuser", hashed_password="hash")
        db_session.add(user)
        await db_session.commit()

        # Query from database
        result = await db_session.execute(select(User).where(User.email == "query@test.com"))
        queried_user = result.scalar_one()

        assert queried_user.email == "query@test.com"
        assert queried_user.username == "queryuser"

    @pytest.mark.asyncio
    async def test_user_relationships_cascade(self, db_session):
        """Test that deleting user cascades to related records."""
        # Create user with API key
        user = User(email="cascade@test.com", username="cascadeuser", hashed_password="hash")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Add API key
        api_key = APIKey(
            user_id=user.id,
            key_hash="hash123",
            name="Test Key",
        )
        db_session.add(api_key)
        await db_session.commit()

        # Delete user - should cascade to API key
        await db_session.delete(user)
        await db_session.commit()

        # Verify API key was also deleted
        result = await db_session.execute(select(APIKey).where(APIKey.key_hash == "hash123"))
        assert result.scalar_one_or_none() is None


class TestTradingModels:
    """Test trading-related models."""

    @pytest.mark.asyncio
    async def test_create_trading_strategy(self, db_session):
        """Test creating trading strategy in database."""
        # Create user first
        user = User(email="trader@test.com", username="trader", hashed_password="hash")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create strategy
        strategy = TradingStrategy(
            user_id=user.id,
            name="Momentum Strategy",
            description="Simple momentum trading",
            config={
                "lookback_period": 20,
                "threshold": 0.02,
                "symbols": ["AAPL", "MSFT"],
            },
        )
        db_session.add(strategy)
        await db_session.commit()
        await db_session.refresh(strategy)

        assert strategy.id is not None
        assert strategy.config["lookback_period"] == 20
        assert "AAPL" in strategy.config["symbols"]

    @pytest.mark.asyncio
    async def test_create_backtest_with_results(self, db_session):
        """Test creating backtest record with JSON results."""
        # Create user
        user = User(email="backtest@test.com", username="backtester", hashed_password="hash")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create backtest
        backtest = Backtest(
            user_id=user.id,
            name="AAPL Backtest 2024",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            symbols=["AAPL", "MSFT"],
            config={"strategy": "momentum"},
            results={
                "sharpe_ratio": 2.5,
                "max_drawdown": 0.12,
                "total_return": 0.45,
            },
            status="completed",
        )
        db_session.add(backtest)
        await db_session.commit()
        await db_session.refresh(backtest)

        assert backtest.results["sharpe_ratio"] == 2.5
        assert backtest.symbols == ["AAPL", "MSFT"]
        assert backtest.status == "completed"


class TestAuditAndAlerts:
    """Test audit logging and alerts."""

    @pytest.mark.asyncio
    async def test_create_audit_log(self, db_session):
        """Test creating audit log entry."""
        # Create user
        user = User(email="audit@test.com", username="audituser", hashed_password="hash")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create audit log
        log = AuditLog(
            user_id=user.id,
            action="user.login",
            resource_type="session",
            details={"ip": "192.168.1.1", "browser": "Chrome"},
            ip_address="192.168.1.1",
        )
        db_session.add(log)
        await db_session.commit()
        await db_session.refresh(log)

        assert log.action == "user.login"
        assert log.details["ip"] == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_create_alert(self, db_session):
        """Test creating user alert."""
        # Create user
        user = User(email="alert@test.com", username="alertuser", hashed_password="hash")
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create alert
        alert = Alert(
            user_id=user.id,
            alert_type="drawdown",
            severity="critical",
            title="Drawdown Exceeded",
            message="Portfolio drawdown exceeded 10%",
            data={"current_drawdown": 0.12, "threshold": 0.10},
        )
        db_session.add(alert)
        await db_session.commit()
        await db_session.refresh(alert)

        assert alert.severity == "critical"
        assert alert.data["current_drawdown"] == 0.12
        assert alert.is_read is False
