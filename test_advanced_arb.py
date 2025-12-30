
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, os.getcwd())

try:
    from cift.ml.advanced_stat_arb import CopulaPairsTrading, MLResidualFilter, HRPAllocation
    print("Imports successful.")
    
    # Test HRP
    print("Testing HRP...")
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = np.random.randn(100, 5)
    df = pd.DataFrame(data, index=dates, columns=['A', 'B', 'C', 'D', 'E'])
    # Add some correlation
    df['B'] = df['A'] * 0.8 + np.random.normal(0, 0.2, 100)
    
    hrp = HRPAllocation()
    weights = hrp.allocate(df)
    print("HRP Weights:")
    print(weights)
    
    # Test ML Filter
    print("\nTesting ML Filter...")
    ml = MLResidualFilter()
    spread = pd.Series(np.random.randn(100))
    score = ml.train(spread)
    print(f"ML Training Score: {score}")
    
    print("\nAll tests passed.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
