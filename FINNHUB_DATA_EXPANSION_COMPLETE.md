# Finnhub Data Expansion & Market Data Fixes

## Overview
This update addresses two main issues:
1.  **Missing Market Data:** Indices (SPY, QQQ) and Crypto (BTC-USD, ETH-USD) were showing $0.00 because they were not being tracked in the real-time update loop.
2.  **Limited Fundamental Data:** The frontend was only showing basic company profile data. The user requested deeper insights including Reported Financials (Balance Sheet, Income Statement) and Earnings Estimates.

## Changes Implemented

### Backend
1.  **`cift/services/polygon_realtime_service.py`**:
    *   Updated `update_market_cache` to explicitly include `INDEX_SYMBOLS` (SPY, QQQ, IWM, DIA) and Crypto pairs (BTC-USD, ETH-USD, etc.) in the background update loop. This ensures these symbols always have fresh data in the cache.

2.  **`cift/services/finnhub_realtime_service.py`**:
    *   Added `get_financials_reported(symbol)`: Fetches "financials-reported" from Finnhub (10-K/10-Q data).
    *   Added `get_earnings_estimates(symbol)`: Fetches "eps-estimate" from Finnhub.

3.  **`cift/services/market_data_service.py`**:
    *   Exposed the new Finnhub methods through the main service orchestrator.

4.  **`cift/api/routes/market_data.py`**:
    *   Added new API endpoints:
        *   `GET /market-data/financials/reported/{symbol}`
        *   `GET /market-data/estimates/{symbol}`

### Frontend
1.  **`frontend/src/lib/api/client.ts`**:
    *   Added `getFinancialsReported(symbol)` and `getEstimates(symbol)` methods to the API client.

2.  **`frontend/src/components/trading/CompanyProfileWidget.tsx`**:
    *   **Logo:** Added company logo display next to the name.
    *   **Earnings Estimates:** Added a section to display EPS estimates (Avg, High, Low) for upcoming periods.
    *   **Reported Financials:** Added a summary of the latest filed report (10-K/10-Q), showing filing date and number of line items available.
    *   **Data Fetching:** Updated `createEffect` to fetch all new data points in parallel.

## Verification
*   **Indices/Crypto:** SPY, QQQ, BTC-USD should now show real-time prices instead of $0.00.
*   **Company Profile:**
    *   Should show the company logo.
    *   Should show "Earnings Estimates" section.
    *   Should show "Latest Report" section with filing details.

## Deployment
*   Deployed using `deploy_full_update.ps1`.
*   Backend restarted.
*   Frontend rebuilt and restarted.
