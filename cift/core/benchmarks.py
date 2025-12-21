"""
CIFT Markets - Performance Benchmarking Suite

Comprehensive benchmarks to validate performance optimizations.

Usage:
    python -m cift.core.benchmarks

This will run all benchmarks and generate a performance report.
"""

import asyncio
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import polars as pl
from loguru import logger

from cift.core.features_numba import (
    calculate_bollinger_bands,
    calculate_ofi,
    calculate_rsi,
    calculate_vwap,
)

# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

class BenchmarkResult:
    """Store benchmark results."""

    def __init__(self, name: str, iterations: int):
        self.name = name
        self.iterations = iterations
        self.times: list[float] = []

    def add_time(self, duration: float):
        """Add execution time in milliseconds."""
        self.times.append(duration * 1000)  # Convert to ms

    @property
    def mean(self) -> float:
        """Mean execution time in ms."""
        return statistics.mean(self.times) if self.times else 0

    @property
    def median(self) -> float:
        """Median execution time in ms."""
        return statistics.median(self.times) if self.times else 0

    @property
    def min(self) -> float:
        """Minimum execution time in ms."""
        return min(self.times) if self.times else 0

    @property
    def max(self) -> float:
        """Maximum execution time in ms."""
        return max(self.times) if self.times else 0

    @property
    def std(self) -> float:
        """Standard deviation in ms."""
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean:.3f}ms\n"
            f"  Median: {self.median:.3f}ms\n"
            f"  Min: {self.min:.3f}ms\n"
            f"  Max: {self.max:.3f}ms\n"
            f"  Std: {self.std:.3f}ms\n"
            f"  Iterations: {self.iterations}"
        )


def benchmark(func: Callable, iterations: int = 1000, warmup: int = 100) -> BenchmarkResult:
    """
    Benchmark a function.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with statistics
    """
    result = BenchmarkResult(func.__name__, iterations)

    # Warmup
    for _ in range(warmup):
        func()

    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        result.add_time(end - start)

    return result


async def benchmark_async(
    func: Callable,
    iterations: int = 1000,
    warmup: int = 100
) -> BenchmarkResult:
    """
    Benchmark an async function.

    Args:
        func: Async function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult with statistics
    """
    result = BenchmarkResult(func.__name__, iterations)

    # Warmup
    for _ in range(warmup):
        await func()

    # Actual benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        end = time.perf_counter()
        result.add_time(end - start)

    return result


# ============================================================================
# NUMBA BENCHMARKS
# ============================================================================

def benchmark_numba_features():
    """Benchmark Numba-optimized feature calculations."""
    logger.info("🔥 Benchmarking Numba-optimized features...")

    # Generate sample data
    np.random.seed(42)
    prices = np.random.randn(10000) * 10 + 100
    volumes = np.random.randint(100, 10000, 10000).astype(float)

    bid_prices = np.array([99.9, 99.8, 99.7, 99.6, 99.5])
    ask_prices = np.array([100.1, 100.2, 100.3, 100.4, 100.5])
    bid_volumes = np.array([1000.0, 800.0, 600.0, 400.0, 200.0])
    ask_volumes = np.array([900.0, 700.0, 500.0, 300.0, 100.0])

    results = {}

    # VWAP
    result = benchmark(lambda: calculate_vwap(prices[:1000], volumes[:1000]), iterations=10000)
    results["VWAP (1K points)"] = result
    logger.info(f"✅ VWAP: {result.mean:.3f}ms")

    # Order Flow Imbalance
    result = benchmark(
        lambda: calculate_ofi(bid_prices, ask_prices, bid_volumes, ask_volumes),
        iterations=10000
    )
    results["OFI (5 levels)"] = result
    logger.info(f"✅ OFI: {result.mean:.3f}ms")

    # RSI
    result = benchmark(lambda: calculate_rsi(prices, period=14), iterations=1000)
    results["RSI (10K points)"] = result
    logger.info(f"✅ RSI: {result.mean:.3f}ms")

    # Bollinger Bands
    result = benchmark(lambda: calculate_bollinger_bands(prices, window=20), iterations=1000)
    results["Bollinger Bands (10K points)"] = result
    logger.info(f"✅ Bollinger Bands: {result.mean:.3f}ms")

    return results


# ============================================================================
# POLARS BENCHMARKS
# ============================================================================

def benchmark_polars_operations():
    """Benchmark Polars data processing operations."""
    logger.info("⚡ Benchmarking Polars operations...")

    # Generate sample OHLCV data
    n_rows = 100000
    df = pl.DataFrame({
        "timestamp": [datetime.utcnow() - timedelta(minutes=i) for i in range(n_rows)],
        "symbol": ["AAPL"] * n_rows,
        "open": np.random.randn(n_rows) * 5 + 150,
        "high": np.random.randn(n_rows) * 5 + 152,
        "low": np.random.randn(n_rows) * 5 + 148,
        "close": np.random.randn(n_rows) * 5 + 150,
        "volume": np.random.randint(1000, 100000, n_rows),
    })

    results = {}

    # GroupBy aggregation
    def groupby_test():
        return df.groupby("symbol").agg([
            pl.col("close").mean().alias("avg_close"),
            pl.col("volume").sum().alias("total_volume"),
        ])

    result = benchmark(groupby_test, iterations=1000)
    results["GroupBy (100K rows)"] = result
    logger.info(f"✅ GroupBy: {result.mean:.3f}ms")

    # Rolling window calculations
    def rolling_test():
        return df.with_columns([
            pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("close").rolling_std(window_size=20).alias("std_20"),
        ])

    result = benchmark(rolling_test, iterations=100)
    results["Rolling Window (100K rows)"] = result
    logger.info(f"✅ Rolling Window: {result.mean:.3f}ms")

    # Join operation
    df2 = pl.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
    })

    def join_test():
        return df.join(df2, on="symbol", how="left")

    result = benchmark(join_test, iterations=1000)
    results["Join (100K rows)"] = result
    logger.info(f"✅ Join: {result.mean:.3f}ms")

    # Filter operation
    def filter_test():
        return df.filter(pl.col("close") > 150)

    result = benchmark(filter_test, iterations=1000)
    results["Filter (100K rows)"] = result
    logger.info(f"✅ Filter: {result.mean:.3f}ms")

    return results


# ============================================================================
# SERIALIZATION BENCHMARKS
# ============================================================================

def benchmark_serialization():
    """Benchmark MessagePack vs JSON serialization."""
    logger.info("📦 Benchmarking serialization...")

    import json

    import msgpack

    # Sample data
    data = {
        "symbol": "AAPL",
        "price": 150.25,
        "volume": 1000000,
        "timestamp": datetime.utcnow().isoformat(),
        "bid": 150.24,
        "ask": 150.26,
        "bid_size": 100,
        "ask_size": 200,
    }

    results = {}

    # JSON encode
    result = benchmark(lambda: json.dumps(data).encode("utf-8"), iterations=10000)
    results["JSON Encode"] = result
    logger.info(f"✅ JSON Encode: {result.mean:.3f}ms")

    # JSON decode
    json_data = json.dumps(data).encode("utf-8")
    result = benchmark(lambda: json.loads(json_data.decode("utf-8")), iterations=10000)
    results["JSON Decode"] = result
    logger.info(f"✅ JSON Decode: {result.mean:.3f}ms")

    # MessagePack encode
    result = benchmark(lambda: msgpack.packb(data), iterations=10000)
    results["MessagePack Encode"] = result
    logger.info(f"✅ MessagePack Encode: {result.mean:.3f}ms")

    # MessagePack decode
    msgpack_data = msgpack.packb(data)
    result = benchmark(lambda: msgpack.unpackb(msgpack_data), iterations=10000)
    results["MessagePack Decode"] = result
    logger.info(f"✅ MessagePack Decode: {result.mean:.3f}ms")

    # Calculate speedup
    json_total = results["JSON Encode"].mean + results["JSON Decode"].mean
    msgpack_total = results["MessagePack Encode"].mean + results["MessagePack Decode"].mean
    speedup = json_total / msgpack_total

    logger.info(f"🚀 MessagePack is {speedup:.2f}x faster than JSON")

    return results


# ============================================================================
# DATABASE BENCHMARKS (Async)
# ============================================================================

async def benchmark_database_queries():
    """Benchmark database query performance."""
    logger.info("🗄️ Benchmarking database queries...")

    from cift.core.database import questdb_manager
    from cift.core.trading_queries import get_latest_price

    results = {}

    # Initialize connections
    try:
        await questdb_manager.initialize()
    except Exception as e:
        logger.warning(f"Could not initialize QuestDB: {e}")
        return results

    # Benchmark raw asyncpg query
    async def raw_query():
        query = "SELECT 1"
        return await questdb_manager.pool.fetchval(query)

    result = await benchmark_async(raw_query, iterations=1000)
    results["Raw asyncpg Query"] = result
    logger.info(f"✅ Raw Query: {result.mean:.3f}ms")

    # Benchmark cached price query
    async def cached_price_query():
        return await get_latest_price("AAPL")

    try:
        result = await benchmark_async(cached_price_query, iterations=100)
        results["Cached Price Query"] = result
        logger.info(f"✅ Cached Price: {result.mean:.3f}ms")
    except Exception as e:
        logger.warning(f"Could not benchmark price query: {e}")

    return results


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

async def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    logger.info("="*70)
    logger.info("CIFT Markets - Performance Benchmark Suite")
    logger.info("="*70)
    logger.info("")

    all_results = {}

    # Run synchronous benchmarks
    logger.info("📊 Running Numba benchmarks...")
    all_results["Numba Features"] = benchmark_numba_features()
    logger.info("")

    logger.info("📊 Running Polars benchmarks...")
    all_results["Polars Operations"] = benchmark_polars_operations()
    logger.info("")

    logger.info("📊 Running Serialization benchmarks...")
    all_results["Serialization"] = benchmark_serialization()
    logger.info("")

    # Run async benchmarks
    logger.info("📊 Running Database benchmarks...")
    all_results["Database Queries"] = await benchmark_database_queries()
    logger.info("")

    # Generate summary report
    logger.info("="*70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("="*70)

    for category, results in all_results.items():
        logger.info(f"\n{category}:")
        logger.info("-" * 70)
        for name, result in results.items():
            logger.info(f"  {name:40s} {result.mean:8.3f}ms ± {result.std:6.3f}ms")

    logger.info("")
    logger.info("="*70)
    logger.info("✅ All benchmarks completed")
    logger.info("="*70)

    return all_results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run benchmarks
    asyncio.run(run_all_benchmarks())
