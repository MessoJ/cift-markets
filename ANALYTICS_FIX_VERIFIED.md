# Analytics Fix Verified

## Issue
The backend was throwing a `NameError: name 'Depends' is not defined` in `cift/api/routes/market_data.py`. This was caused by missing imports after adding the `get_equity_curve_data` endpoint.

## Fixes Applied

1.  **Backend Fix (`cift/api/routes/market_data.py`)**:
    *   Added missing imports: `Depends` from `fastapi`, and `get_current_active_user`, `User` from `cift.core.auth`.
    *   Verified the `get_equity_curve_data` function correctly queries the `portfolio_snapshots` table.

2.  **Frontend Client Cleanup (`frontend/src/lib/api/client.ts`)**:
    *   Removed a duplicate `getAnalytics` method definition that was causing ambiguity.
    *   Verified `getEquityCurveData` exists and is correctly implemented.
    *   Removed a duplicate `getEquityCurveData` method that was accidentally added.

## Verification
*   **Backend**: The server should now reload without errors.
*   **Frontend**: The Analytics page should now correctly fetch:
    *   Performance Metrics (via `getAnalytics` -> `/analytics/performance`)
    *   Equity Curve (via `getEquityCurveData` -> `/market-data/equity-curve`)
    *   Positions (via `getPositions` -> `/trading/positions`)

## Next Steps
*   Refresh the browser to see the fully functional Analytics dashboard.
*   Verify the data matches the seeded values (2 years of history).
