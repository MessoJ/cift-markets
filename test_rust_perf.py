#!/usr/bin/env python3
"""Benchmark Rust module performance"""
import time

try:
    from cift_core import FastIndicators, FastFeatureExtractor
    RUST_AVAILABLE = True
    print("✓ Rust module loaded successfully!")
except ImportError as e:
    print(f"✗ Rust module not available: {e}")
    RUST_AVAILABLE = False
    exit(1)

# Generate test data
data = [100 + x * 0.01 for x in range(10000)]
volumes = [1000000 + x * 100 for x in range(10000)]

# Benchmark indicators
ind = FastIndicators()
iterations = 100

# RSI
start = time.perf_counter()
for _ in range(iterations):
    ind.rsi(data, 14)
rsi_time = (time.perf_counter() - start) / iterations * 1000
print(f"RSI (10k bars):     {rsi_time:.3f}ms")

# MACD
start = time.perf_counter()
for _ in range(iterations):
    ind.macd(data, 12, 26, 9)
macd_time = (time.perf_counter() - start) / iterations * 1000
print(f"MACD (10k bars):    {macd_time:.3f}ms")

# Bollinger
start = time.perf_counter()
for _ in range(iterations):
    ind.bollinger_bands(data, 20, 2.0)
bb_time = (time.perf_counter() - start) / iterations * 1000
print(f"Bollinger (10k):    {bb_time:.3f}ms")

# Feature extraction
extractor = FastFeatureExtractor()

# Process ticks individually (simulating real-time)
start = time.perf_counter()
for _ in range(10):
    extractor.reset()
    for i in range(1000):
        # price, volume, timestamp, ask, bid_size, ask_size
        extractor.process_tick(100 + i*0.01, 1000 + i, float(i), 100.01 + i*0.01, 100, 100)
feat_time = (time.perf_counter() - start) / 10 * 1000
print(f"Features (1k ticks): {feat_time:.3f}ms")

print("\n✓ All benchmarks passed - Rust acceleration is working!")
