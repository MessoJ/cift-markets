# Orders Page & Backend Sync Update

## Improvements
1.  **Backend Sync (`cift/api/routes/trading.py`)**:
    *   Updated `GET /orders` to accept a `sync=true` parameter.
    *   Implemented logic to fetch orders from Alpaca (`/v2/orders`) and upsert them into the local PostgreSQL database.
    *   This ensures that even if the local DB is empty, we can pull historical data from the broker.

2.  **Frontend Enhancements (`OrdersPage.tsx`)**:
    *   **"Sync" Button**: Added a dedicated "Sync" button next to the date picker. Clicking this forces a backend sync with Alpaca.
    *   **"Cancel All" Visibility**: The "Cancel All Open" button is now **always visible**. It is disabled (greyed out) if there are no open orders, which is much better UX than hiding it completely.
    *   **Auto-Sync on Load**: The page now attempts to sync with the broker immediately upon loading (`fetchOrders(true)`), ensuring fresh data.

3.  **Client Library (`client.ts`)**:
    *   Updated `getOrders` signature to support the `sync` parameter.

## Verification
1.  **Refresh**: Hard refresh the page.
2.  **Check Buttons**: You should see the "Sync" button and the "Cancel All Open" button (likely disabled if no open orders).
3.  **Data**: If you have an Alpaca account connected with orders, they should appear after the initial load or after clicking "Sync".
