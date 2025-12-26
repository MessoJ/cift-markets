# News Fix Deployment Report

## Status: Deployed
**Date:** 2025-12-24
**Deployment Script:** `deploy_news_fix.ps1`

## Issues Addressed
1. **Stale News Data**: The news feed was showing data from 18+ hours ago.
   - **Root Cause**: Reliance on a single provider (Polygon) which may have been rate-limited or inactive.
   - **Fix**: Integrated **Financial Modeling Prep (FMP)** as a new, robust news provider.
   - **Backend Change**: Updated `NewsService` in `cift/services/news_service.py` to fetch from FMP.

2. **UI Blinking**: The "Breaking News" banner had an annoying blinking animation.
   - **Fix**: Removed `animate-pulse-subtle` class from `NewsPage.tsx`.

3. **Missing Alerts/Breaking News**: Critical news wasn't appearing as "Breaking".
   - **Root Cause**: The "breaking" logic only looked back 1 hour.
   - **Fix**: Extended the window to 24 hours (`86400000` ms) in `NewsPage.tsx`. Added fallback to show the latest article if no specific "breaking" sentiment is found.

4. **Manual Refresh**: Users couldn't force a news update.
   - **Fix**: Added a `POST /api/v1/news/refresh` endpoint and a "Refresh News" button in the UI.

## Deployment Details
- **Backend Files Updated**:
  - `cift/services/news_service.py`
  - `cift/api/routes/news.py`
- **Frontend Files Updated**:
  - `frontend/src/pages/news/NewsPage.tsx`
- **Services Restarted**:
  - `cift-api` (Restarted)
  - `cift-frontend` (Rebuilt and Recreated)

## Critical Next Step
The server needs the FMP API Key to fetch the new data.

**Action Required:**
1. SSH into the server: `ssh -i temp_key azureuser@20.250.40.67`
2. Edit `.env`: `nano ~/cift-markets/.env`
3. Add: `FMP_API_KEY=your_fmp_api_key_here`
4. Restart API: `docker compose restart api`
