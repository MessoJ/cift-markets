import pandas as pd
import os

data_dir = 'data/intraday/1h'
files = os.listdir(data_dir)
if files:
    f = files[0]
    path = os.path.join(data_dir, f)
    print(f"Inspecting {path}")
    df = pd.read_parquet(path)
    print("Columns:", df.columns)
    print("Index:", df.index)
    print("Head:\n", df.head())
    
    # Check what happens when we try to extract Close
    if isinstance(df.columns, pd.MultiIndex):
        print("MultiIndex detected")
        try:
            s = df.xs('Close', axis=1, level=0).iloc[:, 0]
            print("Extracted Series:\n", s.head())
        except Exception as e:
            print("Extraction failed:", e)
else:
    print("No files found")
