# Trading Page v3.0 Verification Plan

## 1. Layout & Design
- [ ] **3-Column Grid:** Verify the layout is split into Left (Market Depth), Center (Chart/Account), and Right (Order/Watchlist).
- [ ] **Responsiveness:** Ensure it handles window resizing gracefully (though it's designed for desktop density).
- [ ] **"Glass" Aesthetic:** Verify the dark mode, semi-transparent backgrounds are applied.

## 2. Data Integration
- [ ] **Quotes:** Verify real-time price updates in the header and order ticket.
- [ ] **Order Book:** Verify the L2 display is rendering (simulated or real).
- [ ] **Time & Sales:** Verify the tape is scrolling.
- [ ] **Chart:** Verify the chart container is present (using the existing Chart component).

## 3. Account Management (Bottom Panel)
- [ ] **Tabs:** Verify switching between "Positions", "Open Orders", and "Order History".
- [ ] **Data Loading:** Verify `apiClient.getPositions()` and `apiClient.getOrders()` are called.

## 4. Order Entry (Right Panel)
- [ ] **Inputs:** Quantity, Price, Stop Price fields should appear based on Order Type.
- [ ] **Order Types:** Limit, Market, Stop, Stop Limit.
- [ ] **TIF:** Day, GTC, IOC, FOK.
- [ ] **Submission:** Verify "Buy" and "Sell" buttons trigger `apiClient.submitOrder()`.

## 5. Watchlist (Right Panel)
- [ ] **Loading:** Verify watchlists are fetched via `apiClient.getWatchlists()`.
- [ ] **Display:** Verify symbols and changes are displayed.

## Status
- **Build:** ✅ Passed
- **Deployment:** ✅ Complete
- **Verification:** ⏳ Pending User Confirmation
