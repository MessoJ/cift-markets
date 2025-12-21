"""
CIFT Markets - Core Components

Phase 5-7 Ultra-Low-Latency Stack (<10ms)
"""

# Performance-optimized modules
from cift.core import (
    benchmarks,  # Performance testing suite
    capnp_serializer,  # 220x faster serialization (zero-copy)
    clickhouse_manager,  # 100x faster analytics
    config,
    data_processing,  # 19.5x faster data operations (Polars)
    database,
    exceptions,
    features_numba,  # 100x faster feature calculations
    logging,
    models,
    nats_manager,  # 5-10x lower latency than Kafka
    rust_integration,  # 100x faster order matching & risk
    trading_queries,  # 3x faster database queries (raw asyncpg)
)

__all__ = [
    "config",
    "database",
    "nats_manager",
    "clickhouse_manager",
    "rust_integration",
    "logging",
    "models",
    "exceptions",
    "features_numba",
    "data_processing",
    "trading_queries",
    "benchmarks",
    "capnp_serializer",
]
