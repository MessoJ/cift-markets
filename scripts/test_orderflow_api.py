"""Test the Order Flow Transformer API endpoint."""
import requests
import numpy as np
import json

# Generate test data
np.random.seed(42)
prices = (100 + np.cumsum(np.random.randn(100) * 0.01)).tolist()
volumes = np.random.exponential(1000, 100).tolist()

# Test API
url = "http://localhost:8000/api/v1/inference/orderflow/predict"
payload = {
    "prices": prices,
    "volumes": volumes,
    "timestamp": 1234567890.0
}

print(f"Testing Order Flow Transformer API...")
print(f"Input: {len(prices)} prices, {len(volumes)} volumes")

try:
    response = requests.post(url, json=payload, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except requests.exceptions.ConnectionError:
    print("Server not running. Start with: uvicorn cift.api.main:app --reload")
except Exception as e:
    print(f"Error: {e}")
