# Chart Data Fix Implementation

## Problem
Users reported missing chart data for many stocks. The system only had data for manually seeded symbols.
The backend returned 404 for missing data instead of fetching it.

## Solution
Implemented "Fetch on Miss" (Lazy Loading) pattern for historical data.

### Changes

1.  **`cift/services/polygon_realtime_service.py`**
    *   Fixed `update_ohlcv_bars` to correctly map Polygon timespans (minute, hour, day) to database timeframes (1m, 1h, 1d).
    *   Fixed a bug where data was always inserted with "1min" timeframe string, which caused queries to fail.
    *   Added support for `multiplier` to fetch different granularities if needed.

2.  **`cift/api/routes/market_data.py`**
    *   Modified `get_bars` endpoint.
    *   If database returns no data, it now triggers `market_data_service.polygon.update_ohlcv_bars` to fetch history from Polygon.
    *   After fetching, it retries the database query to return the data immediately.
    *   Handles different timeframes (1m, 1h, 1d) by fetching appropriate data from Polygon.

3.  **`cift/core/trading_queries.py`**
    *   Updated `_get_ohlcv_from_postgres` to support querying non-1m timeframes directly.
    *   Previously, it only supported aggregating from 1m bars. Now it checks for exact timeframe match first (e.g., 1h, 1d), enabling efficient storage and retrieval of longer timeframes.

## Result
*   **All Stocks Supported**: Any valid symbol requested on the chart page will now automatically fetch its history and display the chart.
*   **Performance**: First load might take 1-2 seconds to fetch from Polygon. Subsequent loads will be instant (served from DB).
*   **Data Persistence**: Fetched data is stored in PostgreSQL `ohlcv_bars` table for future use.

## Deployment
Run `deploy_chart_fix.ps1` to deploy the changes and restart the API.
