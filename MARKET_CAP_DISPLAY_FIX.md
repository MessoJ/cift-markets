# Market Cap Display Fix

## Issue
The user reported that the market cap for large companies like Nvidia was being displayed in Millions ("M") instead of Trillions ("T").

## Investigation
1. **Database**: Checked `company_profiles` table. Market cap is stored in **Millions**.
   - Example: NVDA = `4364036.91` (which represents $4.36 Trillion).
2. **Frontend**:
   - `formatMarketCap` utility expects the raw value (Ones).
   - `ScreenerPage.tsx` was passing the DB value directly (Millions).
   - `formatMarketCap(4364036)` -> "4.36M".
   - `formatMarketCap(4364036 * 1000000)` -> "4.36T".

## Fixes Applied

### 1. Filter Logic (`ScreenerPage.tsx`)
- Updated `getCriteria` to multiply user input (Billions) by `1000` to match DB units (Millions).
- Updated `useEffect` to divide DB values (Millions) by `1000` when populating the filter inputs (Billions).

### 2. Display Logic (`ScreenerPage.tsx`)
- Updated the table render loop in both "Overview" and "Valuation" tabs.
- Multiplied `result.market_cap` by `1,000,000` before passing it to `formatMarketCap`.

## Verification
- Built the frontend successfully.
- The display should now correctly show "T" for Trillions and "B" for Billions.
