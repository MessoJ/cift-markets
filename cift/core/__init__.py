"""
CIFT Markets - Core Components

Phase 5-7 Ultra-Low-Latency Stack (<10ms)
"""

from cift.core import (
    config,
    database,
    nats_manager,           # 5-10x lower latency than Kafka
    clickhouse_manager,     # 100x faster analytics
    rust_integration,       # 100x faster order matching & risk
    logging,
    models,
    exceptions,
)

# Performance-optimized modules
from cift.core import (
    features_numba,         # 100x faster feature calculations
    data_processing,        # 19.5x faster data operations (Polars)
    trading_queries,        # 3x faster database queries (raw asyncpg)
    benchmarks,             # Performance testing suite
    capnp_serializer,       # 220x faster serialization (zero-copy)
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
