# 🔧 Quick Fix: Chart Authentication Issue

## Problem
- You're logged in and authenticated
- Other pages work fine  
- But chart-related endpoints return 401 Unauthorized
- Requests going to `localhost:3000` instead of being proxied to `localhost:8000`

## Root Cause
Frontend dev server proxy isn't active or needs restart.

## Solution

### Step 1: Restart Frontend Dev Server

```bash
# Stop current frontend (Ctrl+C in the terminal running it)

# Start fresh
cd c:\Users\mesof\cift-markets\frontend
npm run dev
# OR
yarn dev
```

### Step 2: Hard Refresh Browser

```
1. Open http://localhost:3000/charts
2. Press Ctrl+Shift+R (hard refresh)
3. Open DevTools (F12) → Network tab
4. Try loading charts page again
5. Check if requests go to localhost:8000 (via proxy)
```

### Step 3: Verify Proxy is Working

In browser console, test the proxy:

```javascript
// This should return data, not 401
fetch('/api/v1/health', { credentials: 'include' })
  .then(r => r.json())
  .then(console.log)
```

Expected: `{status: "healthy", environment: "development", version: "0.1.0"}`

### Step 4: Check Backend Logs

```bash
docker logs cift-api --tail 20 -f
```

When you refresh charts page, you should see:
```
INFO: 127.0.0.1:XXXXX - "GET /api/v1/price-alerts?symbol=AAPL&active_only=false HTTP/1.1" 200 OK
INFO: 127.0.0.1:XXXXX - "GET /api/v1/chart-templates HTTP/1.1" 200 OK
```

---

## Alternative: Check if Frontend is Running

```bash
# Check what's running on port 3000
netstat -ano | findstr :3000
```

If nothing, start the frontend:
```bash
cd frontend
npm run dev
```

---

## If Still Not Working

### Option A: Use Full API URL (Temporary)

Edit `AlertManager.tsx` and `TemplateManager.tsx`:

Change:
```typescript
const response = await fetch(`/api/v1/price-alerts?...`
```

To:
```typescript
const response = await fetch(`http://localhost:8000/api/v1/price-alerts?...`
```

### Option B: Check CORS Settings

The backend should have CORS configured for `localhost:3000`. Verify in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Quick Test

```bash
# Test from command line (replace with your auth cookie)
curl http://localhost:8000/api/v1/price-alerts \
  -H "Cookie: access_token=YOUR_TOKEN" \
  -v
```

Should return 200 OK with JSON array (empty or with alerts).

---

## Summary

**Most likely fix**: Restart frontend dev server  
**Verify**: Proxy forwards `/api` to `localhost:8000`  
**Test**: Browser console fetch to `/api/v1/health`  

After restart, chart features should work! 🚀
