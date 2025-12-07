# CIFT Markets Frontend Feature Specification

**Status**: Build in progress - Research complete  
**Target**: Institutional-grade algorithmic trading platform  
**Stack**: Next.js 15 + TypeScript + TailwindCSS + shadcn/ui

---

## 📊 Backend Verification Summary

### ✅ **EXISTING BACKEND ENDPOINTS**

| Feature Category | Backend Status | Endpoints Available |
|-----------------|----------------|---------------------|
| **Authentication** | ✅ COMPLETE | Login, Register, Refresh Token, API Keys |
| **Trading** | ✅ CORE READY | Submit Orders, Cancel Orders, Risk Checks |
| **Positions** | ✅ READY | Get Positions, Portfolio Summary |
| **Market Data** | ✅ COMPLETE | Real-time Quotes, Historical Data, WebSocket |
| **Account** | ✅ READY | Buying Power, Account Summary |
| **Monitoring** | ✅ READY | Health, Metrics (Prometheus) |

### ⏳ **MISSING BACKEND (Phase 2-3)**

| Feature | Status | Priority |
|---------|--------|----------|
| **Backtesting API** | NOT IMPLEMENTED | HIGH |
| **ML Model Predictions** | NOT IMPLEMENTED | HIGH |
| **Strategy Management** | NOT IMPLEMENTED | MEDIUM |
| **Performance Analytics** | PARTIAL | MEDIUM |
| **Notifications/Alerts** | NOT IMPLEMENTED | LOW |

---

## 🎯 MUST-HAVE FEATURES (Phase 8 - MVP)

### 1. **Authentication & Onboarding** ✅ Backend Ready

**Pages Required:**
- `/login` - Login page
- `/register` - Sign up page
- `/forgot-password` - Password reset

**Features:**
- JWT authentication with refresh tokens
- API key generation for algorithmic trading
- Session management (30 min expiry)
- Remember me functionality
- OAuth2 integration (future: Google, GitHub)

**UI Requirements:**
- Clean, minimal design (inspired by Robinhood/Coinbase)
- Form validation with real-time feedback
- Loading states for async operations
- Error handling with user-friendly messages

**Backend Endpoints:**
```
✅ POST /api/v1/auth/register
✅ POST /api/v1/auth/login
✅ POST /api/v1/auth/refresh
✅ GET  /api/v1/auth/me
✅ POST /api/v1/auth/logout
✅ POST /api/v1/auth/api-keys
```

---

### 2. **Dashboard (Home Page)** ✅ Backend 80% Ready

**Route:** `/dashboard`

**Sections:**
1. **Portfolio Overview Card**
   - Total account value (large, prominent)
   - Day change ($ and %)
   - Unrealized P&L / Realized P&L
   - Cash vs. Invested breakdown

2. **Positions Table**
   - Symbol, Qty, Avg Cost, Current Price, P&L, P&L%
   - Color coding: Green (profit), Red (loss)
   - Sortable columns
   - Click to view detail

3. **Recent Activity**
   - Latest 10 orders (time, symbol, side, qty, price, status)
   - Real-time updates

4. **Market Overview**
   - Watchlist (customizable)
   - Top movers (gainers/losers)
   - Market indices (S&P 500, NASDAQ, DOW)

5. **Quick Actions**
   - "New Order" button (prominent)
   - "Deposit Funds" button
   - "API Keys" button

**Backend Endpoints:**
```
✅ GET  /api/v1/trading/portfolio
✅ GET  /api/v1/trading/positions
✅ GET  /api/v1/trading/orders
✅ GET  /api/v1/market-data/quotes?symbols[]=AAPL&symbols[]=GOOGL
⏳ GET  /api/v1/trading/activity (Needs implementation)
```

**Performance Target:** <100ms load time

---

### 3. **Trading Interface** ✅ Backend Complete

**Route:** `/trade` or `/trade/:symbol`

**Layout: 3-Column**

#### **Left: Order Entry Panel**
- Symbol selector (autocomplete)
- Side selector (Buy/Sell toggle)
- Order type (Market/Limit radio buttons)
- Quantity input (with max button)
- Price input (limit orders only)
- Time in force (Day/GTC/IOC/FOK)
- **Estimated Cost** calculation
- **Buying Power** display
- **Risk Check** (real-time validation)
- Submit Order button (prominent)

#### **Center: Chart & Data**
- TradingView-style chart (candlestick default)
- Timeframe selector (1m, 5m, 15m, 1h, 4h, 1d)
- Technical indicators (MA, RSI, MACD, Bollinger Bands)
- Drawing tools
- Volume histogram
- Real-time price updates

#### **Right: Order Book & Depth**
- Live order book (bid/ask ladder)
- Market depth visualization
- Recent trades feed
- Current position (if holding)

**Backend Endpoints:**
```
✅ POST /api/v1/trading/orders
✅ POST /api/v1/trading/risk/check
✅ GET  /api/v1/market-data/quote/{symbol}
✅ GET  /api/v1/market-data/bars/{symbol}
✅ WS   /api/v1/market-data/ws/stream
✅ GET  /api/v1/trading/positions/{symbol}
```

**Performance Target:** 
- Order submission: <10ms
- Chart rendering: <50ms
- WebSocket latency: <5ms

---

### 4. **Positions & Portfolio** ✅ Backend Ready

**Route:** `/portfolio`

**Sections:**

#### **Portfolio Summary**
- Total value (with sparkline chart)
- Day change, Week change, Month change
- Cash available
- Buying power
- Total P&L (realized + unrealized)

#### **Holdings Table**
- Symbol, Shares, Avg Cost, Current Price, Market Value, Day P&L, Total P&L, Allocation %
- Search/filter by symbol
- Sort by any column
- Export to CSV

#### **Performance Chart**
- Line chart showing portfolio value over time
- Comparison to benchmarks (S&P 500)
- Drawdown visualization
- Return metrics

#### **P&L Breakdown**
- Realized P&L (closed positions)
- Unrealized P&L (open positions)
- P&L by symbol (pie chart)
- Winners vs Losers ratio

**Backend Endpoints:**
```
✅ GET  /api/v1/trading/portfolio
✅ GET  /api/v1/trading/positions
⏳ GET  /api/v1/analytics/performance (Needs implementation)
⏳ GET  /api/v1/analytics/pnl-breakdown (Needs implementation)
```

---

### 5. **Order Management** ✅ Backend 70% Ready

**Route:** `/orders`

**Features:**

#### **Orders Table**
- All orders (pending, filled, cancelled, rejected)
- Filter by status, symbol, date range
- Columns: Time, Symbol, Side, Type, Qty, Price, Status, Filled Qty
- Color coding by status
- Cancel button (for pending orders)

#### **Order Detail Modal**
- Full order information
- Fill history (partial fills)
- Timestamps (created, updated, filled)
- Fees breakdown
- Cancel/Modify options

#### **Quick Actions**
- Cancel all pending orders
- Duplicate order (re-submit)
- Create stop-loss from position

**Backend Endpoints:**
```
✅ GET    /api/v1/trading/orders
✅ DELETE /api/v1/trading/orders/{order_id}
⏳ PATCH  /api/v1/trading/orders/{order_id} (Needs modify endpoint)
⏳ POST   /api/v1/trading/orders/cancel-all (Needs implementation)
```

---

### 6. **Market Data & Watchlists** ✅ Backend Complete

**Route:** `/markets`

**Features:**

#### **Market Overview**
- Major indices (S&P 500, NASDAQ, DOW, Russell 2000)
- Sector performance heatmap
- Top gainers/losers/volume
- Market sentiment indicators

#### **Symbol Search**
- Fast autocomplete search
- Symbol details (company name, sector, market cap)
- Add to watchlist button

#### **Watchlists**
- Multiple watchlists (customizable)
- Drag-and-drop reordering
- Real-time price updates
- Mini-charts (spark lines)
- Quick trade button

#### **Symbol Detail Page**
- Company profile
- Key stats (P/E, Market Cap, 52-week range)
- News feed
- Earnings calendar
- Analyst ratings

**Backend Endpoints:**
```
✅ GET /api/v1/market-data/quotes
✅ GET /api/v1/market-data/symbols
✅ WS  /api/v1/market-data/ws/stream
⏳ GET /api/v1/market-data/company/{symbol} (Needs implementation)
⏳ GET /api/v1/market-data/news/{symbol} (Needs implementation)
```

---

## 🚀 SHOULD-HAVE FEATURES (Phase 9-10)

### 7. **Backtesting Studio** ⏳ Backend NOT Ready

**Route:** `/backtests`

**Features:**
- Strategy builder (visual or code-based)
- Parameter configuration
- Historical data selection (date range, symbols)
- Run backtest button
- Results visualization:
  - Equity curve
  - Drawdown chart
  - Trade list
  - Performance metrics (Sharpe, Sortino, Max DD, Win Rate)
- Compare multiple backtests
- Export results

**Backend Needed:**
```
❌ POST /api/v1/backtests
❌ GET  /api/v1/backtests
❌ GET  /api/v1/backtests/{id}
❌ GET  /api/v1/backtests/{id}/trades
❌ GET  /api/v1/backtests/{id}/metrics
```

---

### 8. **Strategy Management** ⏳ Backend NOT Ready

**Route:** `/strategies`

**Features:**
- List all strategies
- Create new strategy (IDE-like interface)
- Edit existing strategies
- Live vs Paper toggle
- Strategy parameters configuration
- Performance tracking per strategy
- Start/Stop strategy execution

**Backend Needed:**
```
❌ GET   /api/v1/strategies
❌ POST  /api/v1/strategies
❌ PATCH /api/v1/strategies/{id}
❌ POST  /api/v1/strategies/{id}/start
❌ POST  /api/v1/strategies/{id}/stop
❌ GET   /api/v1/strategies/{id}/performance
```

---

### 9. **ML Model Insights** ⏳ Backend NOT Ready

**Route:** `/insights` or `/predictions`

**Features:**
- AI-powered trade suggestions
- Price predictions (1h, 1d, 1w ahead)
- Order flow imbalance predictions
- Anomaly detection alerts
- Model confidence scores
- Explanation (feature importance)

**Backend Needed:**
```
❌ GET /api/v1/predictions/{symbol}
❌ GET /api/v1/predictions/opportunities
❌ GET /api/v1/models
❌ GET /api/v1/models/{id}/performance
```

---

### 10. **Analytics & Reports** ✅ Backend 50% Ready

**Route:** `/analytics`

**Features:**

#### **Performance Dashboard**
- Return metrics (daily, weekly, monthly, yearly)
- Sharpe ratio, Sortino ratio, Calmar ratio
- Benchmark comparison
- Risk metrics (volatility, max drawdown, VaR)

#### **Trade Analytics**
- Win rate by symbol, strategy, time of day
- Average profit/loss per trade
- Hold time analysis
- Entry/exit efficiency

#### **Tax Reports**
- Realized gains/losses
- Wash sale tracking
- Export for tax filing (CSV, PDF)

**Backend Endpoints:**
```
✅ GET /api/v1/trading/portfolio (basic metrics)
⏳ GET /api/v1/analytics/performance (detailed metrics)
⏳ GET /api/v1/analytics/risk-metrics (VaR, Sharpe, etc.)
⏳ GET /api/v1/analytics/trade-stats (win rate, avg P&L, etc.)
⏳ GET /api/v1/tax/gains-losses (tax reports)
```

---

### 11. **Account & Settings** ✅ Backend 80% Ready

**Route:** `/account`

**Pages:**

#### **Profile**
- Personal information
- Email verification
- Password change
- 2FA setup (future)

#### **API Keys**
- List all API keys
- Create new key
- Revoke keys
- Usage statistics

#### **Funding**
- Deposit funds (ACH, wire)
- Withdraw funds
- Transaction history

#### **Preferences**
- Theme (light/dark)
- Notifications settings
- Trading preferences (default order type, confirmations)
- Data refresh rates

**Backend Endpoints:**
```
✅ GET    /api/v1/auth/me
✅ POST   /api/v1/auth/change-password
✅ POST   /api/v1/auth/api-keys
✅ GET    /api/v1/auth/api-keys
✅ DELETE /api/v1/auth/api-keys/{id}
⏳ POST   /api/v1/account/deposit (Needs implementation)
⏳ POST   /api/v1/account/withdraw (Needs implementation)
⏳ GET    /api/v1/account/transactions (Needs implementation)
```

---

## 🎨 UI/UX REQUIREMENTS

### **Design System**
- **Framework**: Next.js 15 (App Router)
- **Styling**: TailwindCSS + shadcn/ui
- **Icons**: Lucide React
- **Charts**: TradingView Lightweight Charts / Recharts
- **Tables**: TanStack Table (formerly React Table)
- **Forms**: React Hook Form + Zod validation
- **State Management**: Zustand or Jotai
- **Real-time**: WebSocket with auto-reconnect

### **Color Palette** (Institutional Grade)
```css
/* Dark Theme (Primary) */
--background: #0a0a0b;
--surface: #141417;
--surface-elevated: #1c1c21;
--text-primary: #ffffff;
--text-secondary: #a0a0ab;
--accent: #4f46e5; /* Indigo */
--success: #10b981; /* Green */
--danger: #ef4444; /* Red */
--warning: #f59e0b; /* Amber */

/* Light Theme */
--background: #ffffff;
--surface: #f9fafb;
--surface-elevated: #ffffff;
--text-primary: #111827;
--text-secondary: #6b7280;
```

### **Typography**
- **Headings**: Inter (600-700 weight)
- **Body**: Inter (400-500 weight)
- **Monospace** (prices, code): JetBrains Mono

### **Responsiveness**
- Desktop-first (primary use case)
- Tablet support (iPad landscape)
- Mobile: View-only (no trading)

### **Performance Targets**
- Initial page load: <1s
- Route transitions: <100ms
- WebSocket message handling: <5ms
- Chart rendering: <50ms
- Table rendering: <100ms (1000 rows)

---

## 🔥 CRITICAL FRONTEND FEATURES

### **1. Real-Time Data** (CRITICAL)
✅ **Backend Ready:** WebSocket at `/api/v1/market-data/ws/stream`

**Implementation:**
- WebSocket connection with auto-reconnect
- Subscribe to symbols on page load
- Update UI on price changes (no re-renders)
- Heartbeat/ping-pong for connection health
- Graceful degradation (fallback to polling if WS fails)

### **2. Order Confirmation Modal** (CRITICAL)
✅ **Backend Ready:** `/api/v1/trading/risk/check`

**Flow:**
1. User fills order form
2. Click "Review Order"
3. Modal shows:
   - Order details confirmation
   - Estimated cost
   - Risk check results
   - Account impact preview
4. "Confirm & Submit" button
5. Loading state during submission
6. Success/Error feedback

### **3. Notifications System** (HIGH PRIORITY)
⏳ **Backend:** Needs WebSocket events

**Types:**
- Order fills (real-time)
- Order rejections
- Position P&L milestones (+/-5%, +/-10%)
- System alerts (maintenance, outages)
- Strategy signals (Phase 9)

**UI:**
- Bell icon with badge count
- Dropdown panel (click bell)
- Toast notifications (important events)
- Sound alerts (optional, user preference)

### **4. Error Handling** (CRITICAL)
✅ **Backend Ready:** Standard HTTP errors

**Implementation:**
- Global error boundary
- API error interceptor
- User-friendly error messages
- Retry mechanisms (network failures)
- Fallback UI states

### **5. Loading States** (CRITICAL)

**Patterns:**
- Skeleton screens (initial load)
- Spinners (button actions)
- Progress bars (large data loads)
- Shimmer effects (tables, cards)
- Optimistic updates (order submission)

---

## 📊 DATA VISUALIZATION REQUIREMENTS

### **Charts Library:** TradingView Lightweight Charts

**Features Needed:**
- Candlestick charts
- Line charts (portfolio value over time)
- Area charts (filled line charts)
- Volume histogram
- Technical indicators overlay
- Crosshair with data tooltip
- Time scale navigation
- Zoom/pan
- Responsive sizing

### **Alternative:** Recharts (simpler charts)
- Portfolio allocation (pie chart)
- Performance comparison (bar chart)
- P&L breakdown (stacked bar)

### **Tables:** TanStack Table

**Features:**
- Sorting (multi-column)
- Filtering (search, dropdowns)
- Pagination
- Virtual scrolling (large datasets)
- Column resizing
- Column visibility toggle
- Export to CSV

---

## 🔐 SECURITY CONSIDERATIONS

### **Frontend Security**
- Store JWT in httpOnly cookies (not localStorage)
- CSRF protection
- XSS prevention (sanitize user input)
- Rate limiting on forms
- Secure WebSocket (wss://)
- No sensitive data in client logs

### **API Integration**
- Axios interceptor for auth headers
- Automatic token refresh
- Logout on 401 Unauthorized
- API key management (never expose in frontend)

---

## 🚦 DEPLOYMENT CHECKLIST

### **Phase 8 MVP (Weeks 1-4)**
✅ Authentication pages
✅ Dashboard
✅ Trading interface
✅ Portfolio/Positions
✅ Order management
✅ Market data (basic)
✅ Real-time WebSocket

### **Phase 9 (Weeks 5-8)**
⏳ Backtesting studio
⏳ Strategy management
⏳ Advanced analytics
⏳ ML insights

### **Phase 10 (Weeks 9-12)**
⏳ Mobile app (React Native)
⏳ Advanced charting
⏳ Social trading features
⏳ API documentation portal

---

## 📝 NEXT STEPS

### **Immediate Actions (While Build Runs)**

1. ✅ **Backend Audit Complete**
2. ⏳ **Create Frontend Boilerplate**
   ```bash
   npx create-next-app@latest frontend --typescript --tailwind --app
   cd frontend
   npm install shadcn-ui lucide-react zustand axios
   ```

3. ⏳ **Setup Project Structure**
   ```
   frontend/
   ├── app/
   │   ├── (auth)/
   │   │   ├── login/
   │   │   └── register/
   │   ├── (dashboard)/
   │   │   ├── dashboard/
   │   │   ├── trade/
   │   │   ├── portfolio/
   │   │   ├── orders/
   │   │   └── markets/
   │   └── layout.tsx
   ├── components/
   │   ├── ui/ (shadcn components)
   │   ├── charts/
   │   ├── tables/
   │   └── forms/
   ├── lib/
   │   ├── api/ (API client)
   │   ├── websocket/ (WS client)
   │   └── utils/
   └── types/
   ```

4. ⏳ **API Client Setup**
   - Axios instance with interceptors
   - TypeScript types for all endpoints
   - Error handling wrapper
   - Token refresh logic

5. ⏳ **WebSocket Client**
   - Connection manager
   - Auto-reconnect logic
   - Event emitter for updates
   - Subscription management

---

## 🎯 SUCCESS METRICS

### **MVP Launch Criteria**
- [ ] All Phase 8 pages implemented
- [ ] Real-time data working
- [ ] Order submission <10ms
- [ ] Portfolio load <100ms
- [ ] WebSocket stable (no disconnects)
- [ ] Mobile responsive (view-only)
- [ ] Zero critical bugs

### **Performance Benchmarks**
- Lighthouse Score: >90
- First Contentful Paint: <1s
- Time to Interactive: <2s
- WebSocket latency: <5ms
- API response time: <50ms

---

## 📚 REFERENCES

**Inspiration:**
- Robinhood (simplicity)
- TradingView (charts)
- Webull (order flow)
- Interactive Brokers (institutional features)
- Coinbase Pro (clean design)

**Libraries:**
- https://github.com/tradingview/lightweight-charts
- https://ui.shadcn.com/
- https://tanstack.com/table/latest
- https://recharts.org/
- https://zustand.docs.pmnd.rs/

---

**STATUS:** Build in progress (~30 mins remaining)  
**NEXT:** Start frontend scaffold after backend build completes
