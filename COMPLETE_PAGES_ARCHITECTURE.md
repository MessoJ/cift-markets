# 🏗️ CIFT Markets - Complete Pages Architecture

**Platform:** Professional Institutional Trading Platform  
**Design Standard:** Bloomberg Terminal / AlphaDesk Inspired  
**Status:** Systematic Implementation in Progress  

---

## 📋 **COMPLETE PAGE INVENTORY**

### **✅ COMPLETED** 

1. **Dashboard** (`/dashboard`) - ✅ REDESIGNED
   - Professional grid layout
   - Inline portfolio metrics
   - Active positions table
   - Activity feed
   - Quick stats
   - Backend: `GET /api/v1/portfolio`, `GET /api/v1/positions`, `GET /api/v1/activity`

---

### **🚧 IN PROGRESS - REDESIGNING NOW**

2. **Trading** (`/trading`) - 🚧 REDESIGNING  
   - **Left Panel (40%):** Order Entry Ticket
     - Symbol search with autocomplete
     - Side selector (BUY/SELL) - large buttons
     - Order type tabs (MARKET/LIMIT/STOP)
     - Quantity input
     - Price inputs (limit/stop)
     - Estimated cost calculator
     - Submit button
   - **Center Panel (35%):** Real-Time Quote
     - Current price (large, prominent)
     - Bid/Ask spread
     - Volume
     - Day high/low
     - Open/Close
     - Price chart (mini)
   - **Right Panel (25%):** Recent Orders
     - Live order updates
     - Fill notifications
     - Cancel buttons
   - **Backend:** 
     - `GET /api/v1/quote/:symbol`
     - `POST /api/v1/orders`
     - `GET /api/v1/orders/recent`

3. **Portfolio** (`/portfolio`) - 🚧 TO BUILD
   - **Top:** Portfolio Summary Bar
     - Total value
     - Day P&L
     - Total P&L
     - Cash
     - Buying power
   - **Main:** Positions Table (Full Details)
     - Symbol, Side, Qty, Avg Cost, Current Price
     - Market Value, Unrealized P&L, Realized P&L
     - Day P&L, Total Return %
     - Actions (Add, Close, View Details)
   - **Right Sidebar:** Allocation Charts
     - Sector allocation pie chart
     - Position size bars
     - Asset class breakdown
   - **Backend:**
     - `GET /api/v1/portfolio`
     - `GET /api/v1/positions`
     - `POST /api/v1/positions/:id/close`

4. **Orders** (`/orders`) - 🚧 TO BUILD
   - **Tabs:** Open | Filled | Cancelled | All
   - **Filters:** Date range, Symbol, Side, Status
   - **Table Columns:**
     - Order ID, Time, Symbol, Side
     - Type, Qty, Limit Price
     - Filled Qty, Avg Fill Price
     - Status, Actions
   - **Actions:** Cancel, Modify, View Details
   - **Backend:**
     - `GET /api/v1/orders?status=open`
     - `DELETE /api/v1/orders/:id`
     - `PATCH /api/v1/orders/:id`

---

### **❌ NOT STARTED - TO BUILD**

5. **Analytics** (`/analytics`)
   - **Performance Metrics:**
     - Total return, Sharpe ratio, Max drawdown
     - Win rate, Profit factor
     - Average win/loss
   - **Charts:**
     - Equity curve
     - Monthly returns heatmap
     - Drawdown chart
     - Win/loss distribution
   - **Risk Metrics:**
     - Portfolio beta, VaR
     - Correlation matrix
     - Position concentration
   - **Backend:**
     - `GET /api/v1/analytics/performance`
     - `GET /api/v1/analytics/risk`

6. **Watchlists** (`/watchlists`)
   - **Left:** Watchlist Selector
     - Create new list
     - Rename/delete list
     - List switcher
   - **Main:** Symbols Table
     - Symbol, Name, Price, Change %
     - Volume, Mkt Cap
     - Add to list, Remove, Trade
   - **Quick Add:** Symbol search to add
   - **Backend:**
     - `GET /api/v1/watchlists`
     - `POST /api/v1/watchlists`
     - `POST /api/v1/watchlists/:id/symbols`
     - `DELETE /api/v1/watchlists/:id/symbols/:symbol`

7. **Transactions** (`/transactions`)
   - **Filters:** Date range, Type, Symbol
   - **Types:** Trades, Deposits, Withdrawals, Dividends, Fees
   - **Table:**
     - Date, Type, Symbol, Description
     - Amount, Balance
     - Reference ID
   - **Export:** CSV download
   - **Backend:**
     - `GET /api/v1/transactions?from=...&to=...`

8. **Settings** (`/settings`)
   - **Tabs:**
     - **Profile:** Name, Email, Password
     - **Trading:** Default order type, Confirmations
     - **Notifications:** Email, Push, SMS preferences
     - **API Keys:** Generate, Revoke, View
     - **Security:** 2FA, Login history
   - **Backend:**
     - `GET /api/v1/user/settings`
     - `PATCH /api/v1/user/settings`
     - `POST /api/v1/user/api-keys`

---

## 🔍 **DRILL-DOWN PAGES**

### **Position Detail** (`/position/:symbol`)
- **Header:** Symbol name, current price, day change
- **Tabs:**
  1. **Overview:**
     - Position details (qty, cost basis, current value)
     - P&L breakdown
     - Performance chart (line)
  2. **Orders:**
     - All orders for this symbol
     - Entry orders, Exit orders
  3. **Transactions:**
     - All buys/sells for this symbol
     - Dividend history
- **Actions:** Add to Position, Close Position, Set Alert
- **Backend:**
  - `GET /api/v1/positions/:symbol`
  - `GET /api/v1/orders?symbol=:symbol`
  - `GET /api/v1/transactions?symbol=:symbol`

### **Order Detail** (`/order/:id`)
- **Header:** Order ID, Status, Timestamp
- **Details:**
  - Symbol, Side, Type
  - Quantity, Limit Price
  - Filled Qty, Avg Fill Price
  - Fees, Total Cost
  - Fill history (multi-part fills)
- **Timeline:** Order lifecycle events
- **Actions:** Cancel (if open), Duplicate Order
- **Backend:**
  - `GET /api/v1/orders/:id`
  - `GET /api/v1/orders/:id/fills`

### **Symbol Detail** (`/symbol/:symbol`)
- **Header:** Real-time quote
  - Price, Change, Volume
  - Bid/Ask spread
- **Main Chart:** TradingView-style chart
- **Tabs:**
  1. **Overview:**
     - Company info
     - Key stats
     - Recent news
  2. **Your Activity:**
     - Your position (if any)
     - Your orders
     - Your transactions
  3. **Market Data:**
     - Level 2 (depth)
     - Recent trades
     - Historical data
- **Quick Actions:** Trade, Add to Watchlist, Set Alert
- **Backend:**
  - `GET /api/v1/quote/:symbol`
  - `GET /api/v1/market/:symbol/depth`
  - `GET /api/v1/positions/:symbol`

---

## 🎨 **DESIGN STANDARDS (All Pages)**

### **Layout Rules:**
1. **Terminal Colors:** bg-terminal-950, terminal-900, terminal-850
2. **Spacing:** p-2, p-3 (compact), gap-2
3. **Typography:** font-mono for numbers, tabular-nums
4. **Borders:** border-terminal-750
5. **Accents:** accent-500 (orange) for CTAs

### **Component Usage:**
- **Tables:** Use `<Table compact hoverable />`
- **Headers:** 12px font-mono uppercase text-gray-500
- **Numbers:** Right-aligned, tabular-nums, monospace
- **Colors:** 
  - Green (#22c55e) = Positive/Gains/Buy
  - Red (#ef4444) = Negative/Losses/Sell
  - Orange (#f97316) = Accent/CTA/Brand
  - Blue (#3b82f6) = Interactive

### **Information Density:**
- Maximize data per screen
- Use inline metrics (not cards)
- Dense tables (py-1.5, text-xs)
- Multi-column layouts
- Sticky headers

---

## 🔌 **BACKEND INTEGRATION CHECKLIST**

### **APIs to Implement:**
- ✅ `/auth/login` - Complete
- ✅ `/auth/me` - Complete
- ✅ `/portfolio` - Complete
- ✅ `/positions` - Complete
- ✅ `/activity` - Complete
- ❌ `/quote/:symbol` - TO IMPLEMENT
- ❌ `/orders` (POST, GET, DELETE, PATCH) - TO IMPLEMENT
- ❌ `/orders/:id` - TO IMPLEMENT
- ❌ `/orders/:id/fills` - TO IMPLEMENT
- ❌ `/analytics/*` - TO IMPLEMENT
- ❌ `/watchlists/*` - TO IMPLEMENT
- ❌ `/transactions` - TO IMPLEMENT
- ❌ `/market/:symbol/depth` - TO IMPLEMENT

### **WebSocket Channels:**
- ❌ `quotes/:symbol` - Real-time price updates
- ❌ `orders` - Order status changes
- ❌ `fills` - Fill notifications
- ❌ `portfolio` - Portfolio value updates

---

## 📊 **IMPLEMENTATION PRIORITY**

### **Phase 1: Core Trading (NOW)**
1. ✅ Dashboard
2. 🚧 Trading page (order entry)
3. ❌ Portfolio page (full positions)
4. ❌ Orders page (order management)

### **Phase 2: Analysis & History**
5. ❌ Analytics page
6. ❌ Transactions page
7. ❌ Watchlists page

### **Phase 3: Configuration**
8. ❌ Settings page

### **Phase 4: Drill-Downs**
9. ❌ Position detail
10. ❌ Order detail
11. ❌ Symbol detail

### **Phase 5: Real-Time**
12. ❌ WebSocket integration
13. ❌ Live price updates
14. ❌ Order notifications

---

## 🎯 **CURRENT STATUS**

**Completed:** 1/11 pages (9%)  
**In Progress:** Trading page redesign  
**Next:** Portfolio, Orders, Analytics, etc.  

**The foundation is solid. Now systematically building all pages to the same professional standard.**

---

## 💡 **KEY IMPLEMENTATION NOTES**

1. **NO MOCK DATA** - All data from backend APIs
2. **Error Handling** - Show user-friendly errors
3. **Loading States** - Skeleton loaders, spinners
4. **Responsive** - Mobile-friendly (but desktop-first)
5. **Accessibility** - Proper ARIA labels, keyboard nav
6. **Performance** - Lazy load charts, virtualize large tables
7. **Real-Time** - WebSocket for live data
8. **Validation** - Form validation before submit

---

**Status: Building systematically with professional standards...**
