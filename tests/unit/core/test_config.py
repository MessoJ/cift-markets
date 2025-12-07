"""
CIFT Markets - Configuration Tests

Tests for settings validation and configuration management.
"""

import pytest
from pydantic import ValidationError

from cift.core.config import Settings, get_settings


class TestSettings:
    """Test Settings configuration."""

    def test_settings_default_values(self):
        """Test that settings have correct default values."""
        settings = Settings()
        
        assert settings.app_name == "CIFT Markets"
        assert settings.app_env == "development"
        assert settings.app_debug is True
        assert settings.postgres_db == "cift_markets"
        assert settings.redis_db == 0

    def test_settings_postgres_url(self):
        """Test PostgreSQL URL construction."""
        settings = Settings()
        url = settings.postgres_url
        
        assert "postgresql://" in url
        assert settings.postgres_user in url
        assert settings.postgres_db in url
        assert str(settings.postgres_port) in url

    def test_settings_questdb_url(self):
        """Test QuestDB URL construction."""
        settings = Settings()
        
        http_url = settings.questdb_http_url
        assert http_url.startswith("http://")
        assert str(settings.questdb_http_port) in http_url
        
        pg_url = settings.questdb_pg_url
        assert "postgresql://" in pg_url
        assert str(settings.questdb_pg_port) in pg_url

    def test_settings_redis_url_no_password(self):
        """Test Redis URL without password."""
        settings = Settings(redis_password=None)
        url = settings.redis_url
        
        assert url.startswith("redis://")
        assert "@" not in url  # No password

    def test_settings_redis_url_with_password(self):
        """Test Redis URL with password."""
        settings = Settings(redis_password="testpass")
        url = settings.redis_url
        
        assert "redis://:" in url  # Password format
        assert "testpass" in url

    def test_secret_key_validation_length(self):
        """Test that secret keys must be at least 32 characters."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(secret_key="short", app_env="production")
        
        assert "at least 32 characters" in str(exc_info.value)

    def test_settings_singleton(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2

    def test_kafka_topic_names(self):
        """Test Kafka topic configuration."""
        settings = Settings()
        
        assert settings.kafka_topic_market_data == "market-data"
        assert settings.kafka_topic_predictions == "predictions"
        assert settings.kafka_topic_orders == "orders"
        assert settings.kafka_topic_positions == "positions"

    def test_mlflow_configuration(self):
        """Test MLflow settings."""
        settings = Settings()
        
        assert settings.mlflow_tracking_uri == "http://localhost:5000"
        assert settings.mlflow_experiment_name == "cift-markets"

    def test_risk_management_defaults(self):
        """Test risk management default values."""
        settings = Settings()
        
        assert settings.max_position_size == 10000.0
        assert settings.max_portfolio_leverage == 2.0
        assert settings.max_drawdown_pct == 15.0
        assert settings.risk_free_rate == 0.045
