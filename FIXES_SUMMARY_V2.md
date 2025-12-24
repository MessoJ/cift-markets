# Fixes Summary

## 1. API 502 Bad Gateway
- **Issue:** `IndentationError` in `cift/api/routes/news.py` caused the API container to crash on startup.
- **Fix:** Corrected the indentation of the `get_news` function decorator and definition.
- **Status:** ✅ Fixed. API should be reachable now.

## 2. Alerts Page 401 Unauthorized
- **Issue:** `AlertsPage.tsx` was using raw `fetch()` calls which do not include the JWT authentication headers.
- **Fix:** Updated `AlertsPage.tsx` to use `apiClient.axiosInstance` which automatically attaches the `Authorization: Bearer <token>` header. Added `authStore` check to prevent loading when not logged in.
- **Status:** ✅ Fixed. Alerts should load correctly for logged-in users.

## 3. DateRangePicker TypeError
- **Issue:** `DateRangePicker.tsx` threw `TypeError: Cannot read properties of undefined (reading 'toISOString')` when `date` was null or invalid.
- **Fix:** Added safety checks in `formatDateInput` to ensure `date` is valid and has the `toISOString` method before calling it.
- **Status:** ✅ Fixed. Date picker is now robust against null values.

## 4. Orders 400 Bad Request
- **Issue:** Submitting an order with an empty or invalid quantity resulted in a 400 error because `parseFloat('')` returns `NaN`, which is invalid JSON/payload.
- **Fix:** Added input validation in `TradingPage.tsx` (`handleOrderSubmit`) to ensure `quantity` is a positive number before sending the request.
- **Status:** ✅ Fixed. Users will see a validation error message instead of a failed API call.

## 5. Market Data "Hardcoded" / Invalid
- **Issue:** User reported market data (e.g., `AAPL 169.98/170.02`) looked hardcoded.
- **Analysis:** The values match the **seed data** in `seed_market_data.sql`. This indicates that the `market_data_cache` table is populated with initial seed values, and the real-time data service (`polygon_realtime_service.py`) is either not running or not updating the cache.
- **Resolution:** The data is not "hardcoded" in the frontend, but is serving stale seed data from the database. To see live data, the backend real-time service must be active and connected to the data provider (Polygon/Finnhub).
