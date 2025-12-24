# Multi-Provider Market Data Strategy

## Overview
CIFT Markets now utilizes a robust multi-provider strategy to ensure data availability and global coverage.

## Providers

### 1. Polygon.io (Primary - US Stocks)
-   **Role:** Primary source for Real-time US Stock Quotes, Historical Data, and News.
-   **Strengths:** High reliability, WebSocket support, extensive historical data.
-   **Status:** Active (with Mock Fallback).

### 2. Finnhub (Primary - Fundamentals & Estimates)
-   **Role:**
    -   **Primary:** Company Profiles, Financial Statements, Earnings Estimates.
    -   **Secondary:** Real-time Quotes (Free Tier fallback).
-   **Strengths:** Global fundamental data, institutional-grade estimates.
-   **Key:** `d4ojf7pr01qtc1p01m60d4ojf7pr01qtc1p01m6g`

### 3. Alltick.co (Primary - Global Market Data)
-   **Role:**
    -   **Primary:** Global market data (Asian/European markets) where Polygon lacks coverage.
    -   **Secondary:** Fallback for US Quotes and Charts.
-   **Strengths:** Global exchange connectivity.
-   **Key:** `fd881057aae7a4b2045a1fb659f7a670-c-app`

## Fallback Logic

### Real-Time Quotes (`get_quotes_batch`)
1.  **Mock Mode Check:** If API keys are missing, return Mock Data immediately.
2.  **Polygon Snapshot:** Attempt to fetch from Polygon.
3.  **Finnhub Quote:** If Polygon fails (403/Error), try Finnhub.
4.  **Alltick Quote:** If Finnhub fails, try Alltick.
5.  **Polygon Prev Close:** If all real-time sources fail, try Polygon's previous day close.
6.  **Mock Data:** (Implicit) If all else fails.

### Historical Data (Charts)
-   Currently defaults to Polygon.
-   **Future:** Implement similar fallback chain (Polygon -> Finnhub -> Alltick -> Mock).

### Fundamentals
-   **Source:** Finnhub (Company Profile 2, Metrics).

## Implementation
-   **Service:** `cift/services/polygon_realtime_service.py` (Orchestrator)
-   **Helpers:** `cift/services/finnhub_realtime_service.py`, `cift/services/alltick_service.py`
