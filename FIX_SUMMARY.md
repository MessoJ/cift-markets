# Fix Summary

## 1. Frontend API Client Fix (`frontend/src/lib/api/client.ts`)
- **Issue**: `TypeError: L.post is not a function` causing crashes in News, Orders, and other components.
- **Cause**: The `CIFTApiClient` class was missing generic HTTP wrapper methods (`get`, `post`, `put`, `delete`) that were being called by the compiled frontend code.
- **Fix**: Added the missing methods to `CIFTApiClient` to properly wrap `this.axiosInstance`.

## 2. Payment Verification UI Fix (`frontend/src/pages/funding/tabs/PaymentMethodsTab.tsx`)
- **Issue**: Users could not verify payment methods like M-Pesa and PayPal because the UI always showed "VERIFIED" and lacked a verification trigger.
- **Fix**: 
  - Updated the UI to conditionally show a "VERIFY NOW" button for unverified methods.
  - Implemented `handleVerify` function to call `apiClient.initiatePaymentVerification`.
  - Added handling for OAuth redirects (PayPal) and alert messages (M-Pesa STK Push).

## 3. News Refresh Fix
- **Issue**: "News is still the old news of yesterday".
- **Cause**: The "Refresh" button in `NewsPage` was failing due to the `L.post` error, preventing the backend from fetching fresh news.
- **Fix**: The client fix enables the "Refresh" button to work correctly, triggering `POST /news/refresh` which fetches new data.

## 4. Rate Limiting (429 Errors)
- **Analysis**: 
  - Backend limit is 200 requests/minute.
  - Frontend polling is ~7 requests/minute (News + Orders).
  - The 429 errors were likely exacerbated by the client error causing rapid retries or component crashes.
- **Status**: Should be resolved by the client fix. If issues persist, check if multiple tabs are open or if `slowapi` is misidentifying IP addresses.

## Next Steps
1. Reload the frontend application.
2. Verify that the News page loads and the "Refresh" button works.
3. Go to the Funding page and try adding/verifying M-Pesa or PayPal.
4. Monitor for any further 429 errors.
