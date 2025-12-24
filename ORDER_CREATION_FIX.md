# Order Creation Fix

## Issue
The user reported a `400 Bad Request` when submitting orders.
This was caused by the backend `OrderRequest` model rejecting `stop` and `stop_limit` order types, which the frontend supports and sends.

## Diagnosis
1.  **Frontend:** `TradingPage.tsx` allows users to select "Stop" and "Stop Limit" order types and sends `order_type: "stop"` or `"stop_limit"`.
2.  **Backend:** `cift/api/routes/trading.py` had a validator that explicitly raised an error if `order_type` was not "market" or "limit".
3.  **Database:** The `orders` table schema **already supported** `stop` and `stop_limit` types and the `stop_price` column.
4.  **Query:** `insert_order_fast` in `cift/core/trading_queries.py` was missing the `stop_price` column in the INSERT statement.

## Fix Implemented
1.  **Updated `cift/api/routes/trading.py`**:
    *   Updated `OrderRequest` model to include `stop_price`.
    *   Updated `validate_order_type` to allow `stop` and `stop_limit`.
    *   Added `validate_stop_price` to ensure `stop_price` is provided for stop orders.
    *   Updated `submit_order` to pass `stop_price` to the database function.

2.  **Updated `cift/core/trading_queries.py`**:
    *   Updated `insert_order_fast` to include `stop_price` in the SQL INSERT statement.

## Verification
*   **Market/Limit Orders:** Should still work as before.
*   **Stop/Stop Limit Orders:** Should now be accepted by the backend and stored correctly in the database.

## Deployment
*   Deployed backend updates using `deploy_full_update.ps1`.
*   Restarted API service.
