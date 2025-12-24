# Fix Report - December 23, 2025

## Summary
Successfully resolved critical errors affecting the CIFT Markets platform, including WebSocket connection failures, 500 Internal Server Errors on Alerts and Orders endpoints, and missing database tables.

## Resolved Issues

### 1. WebSocket Connection Failures (403 Forbidden)
- **Issue:** WebSocket connections to `/api/v1/ws/market-data/ws/stream` were failing with 403 Forbidden.
- **Root Cause:** Nginx was forwarding the request path `/api/v1/ws/...` directly to the backend, but the backend expected `/market-data/ws/stream` (via `/api/v1` prefix). The extra `/ws` segment caused a path mismatch or routing issue.
- **Fix:** Updated `frontend/nginx.conf` to correctly strip the `/ws` segment from the URI before passing it to the backend.
  ```nginx
  location ^~ /api/v1/ws/ {
      proxy_pass http://cift-api:8000/api/v1/;
      # ...
  }
  ```
- **Deployment:** Hot-patched the running `cift-frontend` container with the new configuration and reloaded Nginx.
- **Verification:** Logs confirm WebSocket connections are now `[accepted]` with the correct path `/api/v1/market-data/ws/stream`.

### 2. 500 Internal Server Error - Alerts
- **Issue:** Fetching alerts returned 500 Error.
- **Root Cause:** SQL query in `cift/api/routes/alerts.py` referenced `condition_value` but the Pydantic model expected `target_value`. Also, `status` field logic was missing in SQL.
- **Fix:** Updated the SQL query to alias columns correctly (`condition_value as target_value`) and added a `CASE` statement to derive `status` from `triggered_at`.

### 3. 500 Internal Server Error - Trading Orders
- **Issue:** Fetching orders returned 500 Error.
- **Root Cause:** `cift/api/routes/trading.py` called `db_manager.fetch(...)` and `db_manager.fetchval(...)`, but the `DatabaseManager` class in `cift/core/database.py` did not implement these methods.
- **Fix:** Added wrapper methods (`fetch`, `fetchrow`, `fetchval`, `execute`) to `DatabaseManager` class to delegate to the underlying `asyncpg` pool.

### 4. Missing Database Table - QuestDB
- **Issue:** Backend logs showed `table does not exist [ticks]`.
- **Root Cause:** The `ticks` table was missing in QuestDB.
- **Fix:** Created the `ticks` table using the QuestDB REST API.
- **Additional Fix:** Added error handling in `cift/api/routes/news.py` to gracefully handle cases where QuestDB queries fail or return empty results.

## Verification Status
- **WebSockets:** ✅ Connected (Log: `WebSocket connected. Total connections: 1`)
- **API:** ✅ Healthy (Log: `CIFT Markets API started successfully`)
- **Database:** ✅ Connections initialized (PostgreSQL, QuestDB, Dragonfly)

## Next Steps
- The system is now stable.
- If you encounter CSP errors in the browser console, ensure the frontend is connecting via the Nginx proxy (port 3000/80) and not directly to the API port (8000), as the Nginx fix handles the routing correctly.
