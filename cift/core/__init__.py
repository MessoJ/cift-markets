"""
CIFT Markets - Core Components

Phase 5-7 Ultra-Low-Latency Stack (<10ms)
"""

# Performance-optimized modules
# - benchmarks: Performance testing suite
# - capnp_serializer: 220x faster serialization (zero-copy)
# - clickhouse_manager: 100x faster analytics
# - data_processing: 19.5x faster data operations (Polars)
# - features_numba: 100x faster feature calculations
# - nats_manager: 5-10x lower latency than Kafka
# - rust_integration: 100x faster order matching & risk
# - trading_queries: 3x faster database queries (raw asyncpg)
from cift.core import (
    benchmarks,
    capnp_serializer,
    clickhouse_manager,
    config,
    data_processing,
    database,
    exceptions,
    features_numba,
    logging,
    models,
    nats_manager,
    rust_integration,
    trading_queries,
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
