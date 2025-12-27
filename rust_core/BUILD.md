# CIFT Rust Core Build Guide

## Prerequisites

### Windows
1. Install Visual Studio 2022 Build Tools:
   ```powershell
   winget install Microsoft.VisualStudio.2022.BuildTools
   ```
   Or download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
   Select "Desktop development with C++" workload

2. Install Rust:
   ```powershell
   winget install Rustlang.Rust.GNU
   # or
   Invoke-WebRequest -Uri https://win.rustup.rs -OutFile rustup-init.exe; .\rustup-init.exe
   ```

3. Install maturin (Python-Rust build tool):
   ```powershell
   pip install maturin
   ```

### Linux
```bash
# Ubuntu/Debian
sudo apt install build-essential
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
```

### macOS
```bash
xcode-select --install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
```

## Building

### Development Build (with debug info)
```powershell
cd rust_core
maturin develop
```

### Release Build (optimized, ~100x faster)
```powershell
cd rust_core
maturin develop --release
```

### Build Wheel for Distribution
```powershell
cd rust_core
maturin build --release
# Output: target/wheels/cift_core-*.whl
```

## Verifying Installation

```python
from cift.rust_bindings import is_rust_available, benchmark_rust_vs_python

print(f"Rust available: {is_rust_available()}")
print(benchmark_rust_vs_python(10000))
```

Expected output with Rust:
```
Rust available: True
{'rsi_time_ms': 0.5, 'feature_extraction_1000_ticks_ms': 1.2, 'using_rust': True}
```

## Module Contents

### FastFeatureExtractor
28-feature ML vector extraction with rolling statistics.
- `process_tick(price, volume, bid, ask, bid_size, ask_size)` → dict
- `process_tick_array(...)` → list (for NumPy)
- `batch_extract(...)` → list of lists
- `feature_names()` → list of feature names

### FastIndicators
Technical indicators optimized for batch processing.
- `rsi(prices, period=14)` → list
- `macd(prices, fast=12, slow=26, signal=9)` → (line, signal, histogram)
- `bollinger_bands(prices, period=20, std=2.0)` → (upper, mid, lower, bandwidth, %b)
- `atr(high, low, close, period=14)` → list
- `stochastic(high, low, close, k=14, slow=3, d=3)` → (k, d)
- `vwap(high, low, close, volume)` → list
- `all_indicators(high, low, close, volume)` → dict of all

### FastJsonParser
simd-json WebSocket message parsing.
- `parse_finnhub(json_str)` → list of trade dicts
- `parse_finnhub_arrays(json_str)` → (symbols, prices, sizes, timestamps)
- `parse_alpaca_trade(json_str)` → trade dict

### FastOrderBook
High-performance order book (<10μs operations).
- `add_limit_order(id, side, price, qty, user_id)` → (id, fills)
- `add_market_order(id, side, qty, user_id)` → fills
- `cancel_order(id)` → bool
- `best_bid()`, `best_ask()`, `spread()` → float
- `depth(levels)` → (bids, asks)

### FastMarketData
Market microstructure calculations.
- `calculate_vwap(ticks)` → float
- `calculate_ofi(bid_volumes, ask_volumes)` → float
- `calculate_microprice(bid, ask, bid_vol, ask_vol)` → float

### FastRiskEngine
Position and order risk checks.
- `check_order(symbol, side, qty, price, position, account_value)` → (bool, reason)
- `max_order_size(...)` → float

## Performance Benchmarks

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| RSI (10k bars) | 50ms | 0.5ms | 100x |
| MACD (10k bars) | 80ms | 0.8ms | 100x |
| Feature extraction (1k ticks) | 100ms | 1ms | 100x |
| JSON parse (1k msgs) | 20ms | 2ms | 10x |
| Order book update | 100μs | 10μs | 10x |

## Troubleshooting

### "linker `link.exe` not found"
Install Visual Studio Build Tools with C++ workload.

### "cargo not found"
Add Rust to PATH:
```powershell
$env:PATH += ";$HOME\.cargo\bin"
```

### "maturin: command not found"
```powershell
pip install maturin
```

### Module import fails after build
Ensure you're in the right Python environment:
```powershell
which python  # Should point to your venv
pip show cift_core
```
