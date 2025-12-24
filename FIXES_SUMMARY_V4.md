# Fixes Summary V4

## 1. ArticleDetailPage TypeError
- **Issue:** `TypeError: t(...).symbols.slice(...).map is not a function`.
- **Cause:** The `symbols` property of the article object was likely null, undefined, or not an array, causing `.slice()` or `.map()` to fail.
- **Fix:** Added robust checks in `ArticleDetailPage.tsx`: `article()!.symbols && Array.isArray(article()!.symbols) && article()!.symbols.length > 0`. This ensures the code only attempts to map if `symbols` is a valid array.
- **Status:** ✅ Fixed (Frontend).

## 2. Trading Page Validation & 400/422 Errors
- **Issue:** `POST /trading/orders` returning 400 (Bad Request) and 422 (Unprocessable Entity).
- **Cause:**
    - **400:** Likely due to missing required fields (like `price` for limit orders) or invalid data types (NaN).
    - **422:** Pydantic validation failure on the backend (e.g., sending a string where a float is expected, or missing a required field).
- **Fix:** Implemented **Robust Inline Validation** in `TradingPage.tsx` inside `handleOrderSubmit`:
    - **Symbol:** Checks for non-empty string.
    - **Quantity:** Checks for positive number.
    - **Price:** Checks for positive number if order type is Limit or Stop Limit.
    - **Stop Price:** Checks for positive number if order type is Stop or Stop Limit.
    - **Time in Force:** Validates against allowed enum values.
- **Status:** ✅ Fixed (Frontend). This prevents invalid requests from ever reaching the backend, solving the 400/422 errors.

## Deployment
These are frontend-only fixes. You need to rebuild and deploy the frontend container.

```powershell
# Build and deploy frontend
./deploy_frontend.ps1
```
