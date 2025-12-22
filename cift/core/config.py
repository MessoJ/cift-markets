"""
CIFT Markets - Configuration Management

Centralized configuration using Pydantic Settings for type safety and validation.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "CIFT Markets"
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = True
    app_url: str = "http://localhost:8000"
    api_version: str = "v1"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "cift_markets"
    postgres_user: str = "cift_user"
    postgres_password: str = "changeme123"

    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # QuestDB
    questdb_host: str = "localhost"
    questdb_http_port: int = 9000
    questdb_influx_port: int = 9009
    questdb_pg_port: int = 8812

    @property
    def questdb_http_url(self) -> str:
        """Build QuestDB HTTP URL."""
        return f"http://{self.questdb_host}:{self.questdb_http_port}"

    @property
    def questdb_pg_url(self) -> str:
        """Build QuestDB PostgreSQL wire protocol URL."""
        return f"postgresql://admin:quest@{self.questdb_host}:{self.questdb_pg_port}/qdb"

    # Dragonfly (Redis-compatible, 25x faster)
    dragonfly_host: str = "localhost"
    dragonfly_port: int = 6379
    dragonfly_password: str | None = None
    dragonfly_db: int = 0

    @property
    def dragonfly_url(self) -> str:
        """Build Dragonfly connection URL (Redis-compatible)."""
        if self.dragonfly_password:
            return f"redis://:{self.dragonfly_password}@{self.dragonfly_host}:{self.dragonfly_port}/{self.dragonfly_db}"
        return f"redis://{self.dragonfly_host}:{self.dragonfly_port}/{self.dragonfly_db}"

    # Legacy Redis alias for compatibility
    @property
    def redis_url(self) -> str:
        """Alias for dragonfly_url (backward compatibility)."""
        return self.dragonfly_url

    # NATS JetStream (5-10x lower latency than Kafka)
    nats_url: str = "nats://localhost:4222"
    nats_stream_market_data: str = "MARKET_DATA"
    nats_stream_orders: str = "ORDERS"
    nats_stream_signals: str = "SIGNALS"
    nats_stream_events: str = "EVENTS"

    # ClickHouse (100x faster analytics)
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_db: str = "cift_analytics"
    clickhouse_user: str = "default"
    clickhouse_password: str = ""

    @property
    def clickhouse_url(self) -> str:
        """Build ClickHouse HTTP URL."""
        return f"http://{self.clickhouse_host}:{self.clickhouse_port}"

    # Market Data Providers
    polygon_api_key: str = ""
    polygon_base_url: str = "https://api.polygon.io"

    # OAuth
    github_client_id: str = ""
    github_client_secret: str = ""
    microsoft_client_id: str = ""
    microsoft_client_secret: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    
    # URLs
    frontend_url: str = "http://localhost:3000"
    api_base_url: str = "http://localhost:8000"

    # Trading
    max_order_value: float = 100000.0
    max_daily_loss: float = 5000.0

    alphavantage_api_key: str = ""
    alphavantage_base_url: str = "https://www.alphavantage.co"

    # Finnhub (FREE real-time WebSocket)
    finnhub_api_key: str = ""
    finnhub_base_url: str = "https://finnhub.io/api/v1"

    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    @property
    def is_alpaca_configured(self) -> bool:
        """Check if Alpaca is configured."""
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    # Interactive Brokers
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 1
    ib_account: str = ""

    # Security
    secret_key: str = Field(
        default="change-this-to-a-random-secret-key-min-32-chars",
        min_length=32,
    )
    jwt_secret_key: str = Field(
        default="change-this-to-another-random-secret-key",
        min_length=32,
    )
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    api_key_salt: str = "random-salt-for-api-key-hashing"

    # Email Configuration (SMTP)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str = "noreply@ciftmarkets.com"
    from_name: str = "CIFT Markets"

    # SMS Configuration
    sms_provider: Literal["twilio", "africas_talking", "aws_sns"] = "twilio"
    sms_from_number: str = ""

    # Twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # Africa's Talking
    africas_talking_username: str = ""
    africas_talking_api_key: str = ""

    # Payment Provider Webhooks
    stripe_webhook_secret: str = ""
    paypal_webhook_id: str = ""
    mpesa_consumer_key: str = ""
    mpesa_consumer_secret: str = ""
    mpesa_shortcode: str = ""
    mpesa_passkey: str = ""

    # Monitoring
    prometheus_port: int = 9090
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "admin"
    grafana_port: int = 3001

    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831

    sentry_dsn: str = ""
    sentry_environment: str = "development"
    sentry_traces_sample_rate: float = 1.0

    # MLOps
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "cift-markets"
    dvc_remote_url: str = ""

    # Feast Feature Store
    feast_registry_path: str = "./feature_store/registry.db"
    feast_online_store_path: str = "./feature_store/online_store.db"

    # Trading Configuration
    max_position_size: float = 10000.0
    max_portfolio_leverage: float = 2.0
    max_drawdown_pct: float = 15.0
    risk_free_rate: float = 0.045

    # Execution
    default_slippage_bps: float = 1.0
    maker_fee_bps: float = 0.08
    taker_fee_bps: float = 0.08
    order_timeout_seconds: int = 30

    # Models
    model_prediction_horizon_ms: int = 500
    model_confidence_threshold: float = 0.65
    ensemble_min_agreement: int = 3

    # Performance
    worker_processes: int = 4
    worker_threads: int = 2
    cache_ttl_seconds: int = 300
    cache_max_size_mb: int = 1024

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "json"
    log_file: str = "logs/cift.log"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"

    # Development
    hot_reload: bool = True
    debug_sql: bool = False
    profiling_enabled: bool = False

    @field_validator("secret_key", "jwt_secret_key")
    @classmethod
    def validate_secrets(cls, v: str) -> str:
        """Validate secret keys are strong enough."""
        if v.startswith("change-this") and cls.model_config.get("app_env") == "production":
            raise ValueError("Secret keys must be changed in production!")
        if len(v) < 32:
            raise ValueError("Secret keys must be at least 32 characters long")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
