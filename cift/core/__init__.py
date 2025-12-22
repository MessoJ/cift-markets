"""
CIFT Markets - Core Components

Phase 5-7 Ultra-Low-Latency Stack (<10ms)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Keep imports lightweight: some submodules depend on optional heavy deps (e.g. numba).
# Expose submodules via lazy import so `import cift.core.database` doesn't require them.

__all__ = [
    "benchmarks",
    "capnp_serializer",
    "clickhouse_manager",
    "config",
    "data_processing",
    "database",
    "exceptions",
    "features_numba",
    "logging",
    "models",
    "nats_manager",
    "rust_integration",
    "trading_queries",
]

if TYPE_CHECKING:
    from cift.core import (  # noqa: F401
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


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
