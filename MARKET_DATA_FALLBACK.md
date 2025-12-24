# Market Data Fallback System

## Overview
The CIFT platform now includes a robust fallback system for market data. This ensures that the application remains functional even if the primary market data provider (Polygon.io) is unavailable or if the API key is invalid/expired.

## How it Works
The `PolygonRealtimeService` monitors API responses. If it encounters a `403 Forbidden` error (indicating an invalid API key or subscription issue), it automatically switches to "Mock Mode".

### Mock Mode Features
When in Mock Mode, the service generates realistic synthetic data for:
1.  **Real-time Quotes:** Generates price, volume, and change data based on a deterministic seed derived from the symbol name. This ensures that "AAPL" always has a consistent base price, but fluctuates slightly like a real stock.
2.  **Historical Data (Charts):** Generates OHLCV (Open, High, Low, Close, Volume) bars for charts. It uses a random walk algorithm to create realistic-looking price trends over time.

## Benefits
-   **Zero Downtime:** Users can still interact with the platform (view charts, place trades, set alerts) even without a valid data feed.
-   **Development & Testing:** Developers can work on the frontend without needing a paid API subscription.
-   **Demo Mode:** The platform can be demonstrated to potential clients without relying on external API stability.

## Implementation Details
-   **File:** `cift/services/polygon_realtime_service.py`
-   **Trigger:** `_request` method catches 403 status codes.
-   **Generators:**
    -   `_generate_mock_quotes`: Creates snapshot data.
    -   `_generate_mock_aggregates`: Creates historical bar data.

## Future Improvements
-   Add a UI indicator when running in Mock Mode.
-   Support for Alpha Vantage as a secondary real-time provider before falling back to mock data.
