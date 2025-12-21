import pandas as pd
import numpy as np
import os

def generate_data(output_path="data/sample_data.csv", n_rows=1000):
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=n_rows, freq="D")
    close = 100 + np.cumsum(np.random.randn(n_rows))
    
    # Ensure positive prices
    close = np.maximum(close, 1.0)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "close": close,
        "open": close + np.random.randn(n_rows) * 0.5,
        "high": close + np.abs(np.random.randn(n_rows) * 0.5),
        "low": close - np.abs(np.random.randn(n_rows) * 0.5),
        "volume": np.random.randint(1000, 10000, n_rows)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {n_rows} rows.")

if __name__ == "__main__":
    generate_data()
