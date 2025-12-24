# Market Data Persistence Fix

## Issue
The user reported "persistent yet wrong" market data, specifically large discrepancies between the "Price" and the "Bid/Ask" values (e.g., NVDA Price $189 vs Bid/Ask ~$485).

## Diagnosis
1.  **Stale Seed Data:** The database was initially seeded with mock data where NVDA was ~$485.
2.  **Partial Updates:** The `PolygonRealtimeService` was fetching live prices (e.g., NVDA $189) and updating the `price` column in the database, but it was **NOT** updating the `bid` and `ask` columns.
3.  **Result:** The API served the *new* Price ($189) alongside the *old* Bid/Ask ($485), causing the discrepancy.

## Fix Implemented
1.  **Updated `cift/services/polygon_realtime_service.py`**:
    *   Modified `get_quotes_batch` to extract `bid` and `ask` from Polygon snapshots (using `lastQuote` data).
    *   Ensured fallback providers (Finnhub, Alltick) return `bid: None` and `ask: None` if they don't provide them.
    *   Modified `update_market_cache` to explicitly update the `bid` and `ask` columns in the `market_data_cache` table.

## How it Works Now
*   **Live Data:** If Polygon provides live Bid/Ask, they are stored in the DB.
*   **Fallback/Missing Data:** If Bid/Ask are missing (e.g., from Finnhub), `NULL` is written to the DB.
*   **API Logic:** The API (`market_data.py`) handles `NULL` Bid/Ask by calculating them dynamically from the current Price (`Price * 0.9999` / `Price * 1.0001`).
*   **Result:** Bid/Ask will now always be consistent with the Price.

## Verification
*   **NVDA:** Should show Price ~$189 and Bid/Ask around ~$189 (not $485).
*   **AMZN:** Should show Price ~$232 and Bid/Ask around ~$232 (not $155).

## Deployment
*   Deployed backend updates using `deploy_full_update.ps1`.
*   Restarted API service.
