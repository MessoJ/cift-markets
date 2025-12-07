# Frontend Data Drilldown - Deep Research & Backend Verification

**Date:** 2025-11-09  
**Purpose:** Comprehensive analysis of all institutional trading platform drilldowns  
**Method:** Analyzed Bloomberg Terminal, Interactive Brokers TWS, TradingView, Alpaca, TD Ameritrade

---

## 🎯 Executive Summary

**Finding:** Backend is **60% ready** for institutional drilldowns. Missing critical tables and 15 endpoints.

### **Critical Missing Components:**

| Component | Status | Impact |
|-----------|--------|--------|
| **Portfolio Snapshots** | ❌ Missing table | Cannot show time-series performance |
| **Order Execution Details** | ❌ Missing fills ref | Cannot analyze execution quality |
| **Watchlists** | ❌ Missing table | Cannot save symbol lists |
| **Intraday P&L** | ❌ Missing endpoint | Cannot show real-time day P&L |
| **Order Book** | ❌ Missing data | Cannot show depth of market |
| **Symbol Stats** | ❌ Missing aggregations | Cannot show per-symbol analytics |

---

## 📊 INSTITUTIONAL DRILLDOWN REQUIREMENTS

### **Level 1: Dashboard View** (Overview)
Shows aggregate metrics, click to drill down

### **Level 2: Category View** (Grouped Data)
Shows data grouped by dimension (symbol, day, strategy)

### **Level 3: Detail View** (Individual Record)
Shows complete details of single entity (order, position, trade)

### **Level 4: Time-Series View** (Historical)
Shows metric evolution over time (P&L curve, drawdown)

---

## 🔍 DRILLDOWN ANALYSIS

---

## 1. **ORDER DRILLDOWNS** 🔴 50% READY

### **Level 1: Orders List**
**Route:** `/orders`  
**View:** Table of all orders

**Required Data:**
- Order ID, Symbol, Side, Type, Status
- Quantity (total, filled, remaining)
- Prices (limit, stop, avg fill)
- Timestamps (created, filled, cancelled)
- Commission, Total value

**Backend Status:**
- ✅ GET `/api/v1/trading/orders` - EXISTS
- ✅ Database: `orders` table - COMPLETE
- ❌ Missing: Filter by date range
- ❌ Missing: Filter by multiple statuses
- ❌ Missing: Pagination

**Required Enhancements:**
```python
GET /api/v1/trading/orders?
    symbol=AAPL&
    status=filled,partial&
    start_date=2025-01-01&
    end_date=2025-01-31&
    page=1&
    limit=50&
    sort_by=created_at&
    sort_order=desc
```

---

### **Level 2: Order Detail** ❌ CRITICAL MISSING
**Route:** `/orders/:order_id`  
**View:** Complete order execution breakdown

**Required Data:**
```json
{
  "order": {
    "id": "uuid",
    "symbol": "AAPL",
    "side": "buy",
    "quantity": 100,
    "status": "filled",
    "created_at": "...",
    "filled_at": "..."
  },
  "fills": [
    {
      "fill_id": "uuid",
      "quantity": 60,
      "price": 150.25,
      "venue": "NASDAQ",
      "timestamp": "...",
      "commission": 0.05,
      "liquidity_flag": "removed"
    },
    {
      "fill_id": "uuid",
      "quantity": 40,
      "price": 150.30,
      "venue": "ARCA",
      "timestamp": "...",
      "commission": 0.03,
      "liquidity_flag": "removed"
    }
  ],
  "execution_quality": {
    "avg_fill_price": 150.27,
    "vwap": 150.26,
    "slippage_bps": 1.2,
    "fill_rate": 100,
    "num_fills": 2,
    "time_to_fill_ms": 245
  },
  "timeline": [
    {"event": "created", "timestamp": "..."},
    {"event": "submitted_to_broker", "timestamp": "..."},
    {"event": "accepted", "timestamp": "..."},
    {"event": "partial_fill", "quantity": 60, "timestamp": "..."},
    {"event": "filled", "quantity": 40, "timestamp": "..."}
  ]
}
```

**Backend Status:**
- ❌ GET `/api/v1/trading/orders/:id` - MISSING
- ✅ Database: `order_fills` table - EXISTS
- ❌ Missing: Execution quality calculations
- ❌ Missing: Timeline/audit trail
- ❌ Missing: VWAP comparison

**Database Gap:**
```sql
-- MISSING: fills reference in orders table
-- Need to query order_fills by order_id

-- MISSING: execution metrics
ALTER TABLE orders ADD COLUMN execution_latency_ms INTEGER;
ALTER TABLE orders ADD COLUMN slippage_bps NUMERIC(10, 4);
ALTER TABLE orders ADD COLUMN vwap_price NUMERIC(15, 4);
```

---

### **Level 3: Symbol Order History** ❌ MISSING
**Route:** `/orders/symbol/:symbol`  
**View:** All orders for a specific symbol

**Required Data:**
- All orders for symbol (last 90 days)
- Win rate (profitable orders %)
- Avg P&L per trade
- Total volume traded
- Execution quality stats

**Backend Status:**
- ❌ GET `/api/v1/trading/orders/symbol/:symbol` - MISSING
- ✅ Database: Can query from `orders` table
- ❌ Missing: Aggregations

---

## 2. **POSITION DRILLDOWNS** 🟡 70% READY

### **Level 1: Positions List**
**Route:** `/positions`  
**View:** All current positions

**Backend Status:**
- ✅ GET `/api/v1/trading/positions` - EXISTS
- ✅ Database: `positions` table - COMPLETE
- ✅ Real-time P&L - EXISTS

---

### **Level 2: Position Detail** ❌ CRITICAL MISSING
**Route:** `/positions/:symbol`  
**View:** Deep dive into single position

**Required Data:**
```json
{
  "position": {
    "symbol": "AAPL",
    "quantity": 150,
    "avg_cost": 148.50,
    "current_price": 155.75,
    "market_value": 23362.50,
    "unrealized_pnl": 1087.50,
    "unrealized_pnl_pct": 7.32,
    "day_pnl": 225.00
  },
  "cost_basis": [
    {"date": "2025-01-05", "quantity": 100, "price": 147.25},
    {"date": "2025-01-12", "quantity": 50, "price": 151.00}
  ],
  "entry_orders": [
    {"order_id": "uuid1", "quantity": 100, "price": 147.25, "date": "..."},
    {"order_id": "uuid2", "quantity": 50, "price": 151.00, "date": "..."}
  ],
  "pnl_timeline": [
    {"date": "2025-01-05", "pnl": 0, "pnl_pct": 0},
    {"date": "2025-01-06", "pnl": 125.00, "pnl_pct": 0.85},
    {"date": "2025-01-07", "pnl": -75.00, "pnl_pct": -0.51},
    {"date": "2025-01-08", "pnl": 1087.50, "pnl_pct": 7.32}
  ],
  "risk_metrics": {
    "portfolio_weight": 15.2,
    "beta": 1.15,
    "volatility_30d": 22.5,
    "var_1d_95": -245.00
  },
  "trade_history": [
    {"type": "buy", "quantity": 100, "price": 147.25, "date": "..."},
    {"type": "buy", "quantity": 50, "price": 151.00, "date": "..."}
  ]
}
```

**Backend Status:**
- ❌ GET `/api/v1/trading/positions/:symbol/detail` - MISSING
- ✅ Database: `positions` table - EXISTS
- ❌ Missing: Cost basis tracking (FIFO/LIFO)
- ❌ Missing: Entry order references
- ❌ Missing: P&L timeline
- ❌ Missing: Risk metrics

**Database Gap:**
```sql
-- MISSING: Cost basis lots tracking
CREATE TABLE position_lots (
    id UUID PRIMARY KEY,
    position_id UUID REFERENCES positions(id),
    quantity NUMERIC(15, 4),
    purchase_price NUMERIC(15, 4),
    purchase_date TIMESTAMP,
    lot_method VARCHAR(10) -- FIFO, LIFO, AvgCost
);

-- MISSING: Position P&L snapshots
CREATE TABLE position_snapshots (
    id UUID PRIMARY KEY,
    position_id UUID REFERENCES positions(id),
    symbol VARCHAR(10),
    quantity NUMERIC(15, 4),
    price NUMERIC(15, 4),
    unrealized_pnl NUMERIC(15, 2),
    timestamp TIMESTAMP,
    INDEX(position_id, timestamp)
);
```

---

### **Level 3: Closed Position Analysis** ❌ PARTIALLY MISSING
**Route:** `/positions/history/:symbol`  
**View:** Historical performance of closed positions

**Backend Status:**
- ✅ Database: `position_history` table - EXISTS
- ❌ GET `/api/v1/trading/positions/history` - MISSING ENDPOINT
- ❌ Missing: Aggregations

---

## 3. **PORTFOLIO DRILLDOWNS** 🔴 40% READY

### **Level 1: Portfolio Summary**
**Route:** `/portfolio`  
**View:** High-level portfolio metrics

**Backend Status:**
- ✅ GET `/api/v1/trading/portfolio` - EXISTS
- ❌ Missing: Time-series data

---

### **Level 2: Portfolio Performance** ❌ CRITICAL MISSING
**Route:** `/portfolio/performance`  
**View:** Time-series portfolio analytics

**Required Data:**
```json
{
  "equity_curve": [
    {"date": "2025-01-01", "value": 100000, "cash": 50000, "positions": 50000},
    {"date": "2025-01-02", "value": 101250, "cash": 48000, "positions": 53250},
    {"date": "2025-01-03", "value": 99875, "cash": 48000, "positions": 51875}
  ],
  "daily_pnl": [
    {"date": "2025-01-02", "pnl": 1250, "pnl_pct": 1.25},
    {"date": "2025-01-03", "pnl": -1375, "pnl_pct": -1.35}
  ],
  "drawdown_curve": [
    {"date": "2025-01-01", "drawdown": 0},
    {"date": "2025-01-02", "drawdown": 0},
    {"date": "2025-01-03", "drawdown": -1.23}
  ],
  "metrics": {
    "total_return": 12.5,
    "sharpe_ratio": 2.1,
    "max_drawdown": -5.3,
    "win_rate": 66.7,
    "avg_win": 425.00,
    "avg_loss": -210.00
  }
}
```

**Backend Status:**
- ❌ GET `/api/v1/analytics/portfolio/equity-curve` - MISSING
- ❌ Database: `portfolio_snapshots` table - MISSING
- ❌ Missing: Daily snapshot mechanism

**Critical Database Gap:**
```sql
-- MISSING: Portfolio snapshots (CRITICAL!)
CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    account_id UUID REFERENCES accounts(id),
    
    -- Values
    total_value NUMERIC(15, 2),
    cash NUMERIC(15, 2),
    positions_value NUMERIC(15, 2),
    
    -- P&L
    unrealized_pnl NUMERIC(15, 2),
    realized_pnl NUMERIC(15, 2),
    day_pnl NUMERIC(15, 2),
    
    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE,
    snapshot_type VARCHAR(20), -- eod, intraday
    
    INDEX(user_id, timestamp),
    INDEX(account_id, timestamp)
);
```

---

### **Level 3: Allocation Breakdown** 🟡 PARTIAL
**Route:** `/portfolio/allocation`  
**View:** Portfolio composition

**Required Data:**
- By sector (Tech 35%, Finance 25%, etc.)
- By asset class (Stocks 80%, Options 20%)
- By position size (Top 10 positions)
- Geographic breakdown

**Backend Status:**
- ❌ GET `/api/v1/analytics/portfolio/allocation` - MISSING
- ✅ Can calculate from positions table
- ❌ Missing: Sector/industry data

---

## 4. **ANALYTICS DRILLDOWNS** 🟡 60% READY

### **Level 1: Performance Dashboard**
**Backend Status:**
- ✅ GET `/api/v1/analytics/performance` - EXISTS (recently added)
- ✅ Sharpe, max drawdown, returns - EXISTS
- ❌ Missing: Intraday metrics

---

### **Level 2: P&L Analysis** 🟡 PARTIAL
**Route:** `/analytics/pnl`  
**View:** Multi-dimensional P&L breakdown

**Required Data:**
- By symbol (which symbols are profitable?)
- By day/week/month (when am I profitable?)
- By time of day (best trading hours?)
- By strategy (which strategies work?)
- By holding period (<1hr, 1-4hr, 1day+)

**Backend Status:**
- ✅ GET `/api/v1/analytics/pnl-breakdown?group_by=symbol` - EXISTS
- ✅ group_by: symbol, day, month - EXISTS
- ❌ Missing: group_by hour, strategy, holding_period
- ❌ Missing: Intraday breakdown

---

### **Level 3: Trade Journal** ❌ MISSING
**Route:** `/analytics/trade-journal`  
**View:** Detailed trade log with notes

**Required Data:**
```json
{
  "trades": [
    {
      "trade_id": "uuid",
      "symbol": "AAPL",
      "entry_date": "2025-01-05 10:30:00",
      "exit_date": "2025-01-05 14:15:00",
      "entry_price": 147.25,
      "exit_price": 149.50,
      "quantity": 100,
      "pnl": 225.00,
      "pnl_pct": 1.53,
      "hold_duration": "3h 45m",
      "strategy": "momentum",
      "notes": "Strong breakout above resistance",
      "tags": ["breakout", "high-volume"],
      "screenshots": ["url1", "url2"]
    }
  ]
}
```

**Backend Status:**
- ❌ GET `/api/v1/analytics/trade-journal` - MISSING
- ✅ Database: `position_history` table - EXISTS (partial)
- ❌ Missing: Notes, tags, screenshots
- ❌ Missing: Strategy classification

**Database Gap:**
```sql
ALTER TABLE position_history 
    ADD COLUMN strategy VARCHAR(50),
    ADD COLUMN tags TEXT[],
    ADD COLUMN notes TEXT,
    ADD COLUMN screenshots JSONB;
```

---

## 5. **MARKET DATA DRILLDOWNS** 🟢 80% READY

### **Level 1: Watchlist**
**Route:** `/watchlist`  
**View:** Saved symbols with real-time prices

**Backend Status:**
- ❌ Database: `watchlists` table - MISSING
- ❌ GET `/api/v1/watchlists` - MISSING
- ✅ Can get prices from market data API

**Database Gap:**
```sql
CREATE TABLE watchlists (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(100),
    symbols TEXT[],
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

---

### **Level 2: Symbol Detail** 🟡 PARTIAL
**Route:** `/symbol/:symbol`  
**View:** Deep dive into symbol

**Required Data:**
- Real-time quote
- Intraday chart
- Historical performance
- Order book (Level 2 data)
- Recent trades
- User's trading history for symbol
- User's P&L for symbol

**Backend Status:**
- ✅ GET `/api/v1/market-data/quote/:symbol` - EXISTS
- ✅ GET `/api/v1/market-data/bars/:symbol` - EXISTS
- ❌ GET `/api/v1/market-data/order-book/:symbol` - MISSING
- ❌ GET `/api/v1/market-data/recent-trades/:symbol` - MISSING
- ❌ GET `/api/v1/analytics/symbol/:symbol/stats` - MISSING

---

### **Level 3: Comparison View** ❌ MISSING
**Route:** `/compare?symbols=AAPL,GOOGL,MSFT`  
**View:** Side-by-side comparison

**Backend Status:**
- ❌ All comparison endpoints - MISSING

---

## 6. **EXECUTION QUALITY DRILLDOWNS** ❌ 20% READY

### **Level 1: Execution Dashboard** ❌ CRITICAL MISSING
**Route:** `/execution/quality`  
**View:** Execution analytics

**Required Data:**
```json
{
  "summary": {
    "avg_slippage_bps": 1.2,
    "fill_rate": 98.5,
    "avg_execution_time_ms": 245,
    "maker_taker_ratio": 0.65
  },
  "by_symbol": [
    {"symbol": "AAPL", "avg_slippage": 0.8, "fill_rate": 99.2},
    {"symbol": "TSLA", "avg_slippage": 2.5, "fill_rate": 95.0}
  ],
  "by_venue": [
    {"venue": "NASDAQ", "volume": 15000, "avg_slippage": 0.9},
    {"venue": "ARCA", "volume": 8000, "avg_slippage": 1.5}
  ],
  "by_time_of_day": [
    {"hour": 9, "avg_slippage": 3.5, "volume": 2000},
    {"hour": 10, "avg_slippage": 1.2, "volume": 5000}
  ]
}
```

**Backend Status:**
- ❌ ALL execution quality endpoints - MISSING
- ❌ Missing: Slippage calculations
- ❌ Missing: Execution latency tracking
- ❌ Missing: Venue analytics

**Database Gap:**
```sql
-- Add execution metrics to order_fills
ALTER TABLE order_fills
    ADD COLUMN submission_latency_ms INTEGER,
    ADD COLUMN execution_latency_ms INTEGER,
    ADD COLUMN slippage_bps NUMERIC(10, 4),
    ADD COLUMN market_price_at_submission NUMERIC(15, 4);

-- Add execution stats table
CREATE TABLE execution_stats (
    id UUID PRIMARY KEY,
    user_id UUID,
    symbol VARCHAR(10),
    date DATE,
    
    -- Aggregations
    total_orders INTEGER,
    filled_orders INTEGER,
    cancelled_orders INTEGER,
    avg_slippage_bps NUMERIC(10, 4),
    avg_execution_time_ms INTEGER,
    total_volume NUMERIC(15, 4),
    maker_fills INTEGER,
    taker_fills INTEGER,
    
    INDEX(user_id, date),
    INDEX(symbol, date)
);
```

---

## 7. **RISK DRILLDOWNS** 🟡 50% READY

### **Level 1: Risk Dashboard**
**Route:** `/risk`  
**View:** Portfolio risk metrics

**Backend Status:**
- ✅ GET `/api/v1/analytics/risk-metrics` - EXISTS (recently added)
- ✅ Leverage, concentration - EXISTS
- ❌ Missing: VaR, stress testing

---

### **Level 2: Risk Detail** ❌ MISSING
**Route:** `/risk/detail`  
**View:** Advanced risk analytics

**Required Data:**
- Value at Risk (VaR) 1-day, 5-day
- Expected Shortfall (CVaR)
- Portfolio Greeks (for options)
- Correlation matrix
- Stress test scenarios
- Margin requirements

**Backend Status:**
- ❌ ALL advanced risk endpoints - MISSING
- ❌ Requires quantitative models

---

## 8. **TRANSACTION DRILLDOWNS** 🟢 80% READY

### **Level 1: Transaction History**
**Route:** `/transactions`  
**View:** All account movements

**Backend Status:**
- ✅ Database: `transactions` table - EXISTS
- ❌ GET `/api/v1/trading/transactions` - MISSING ENDPOINT
- ✅ Data structure supports all types

---

### **Level 2: Cash Flow Analysis** ❌ MISSING
**Route:** `/transactions/cash-flow`  
**View:** Money in vs money out

**Backend Status:**
- ❌ Endpoint missing
- ✅ Can calculate from transactions

---

## 9. **ALERT/NOTIFICATION DRILLDOWNS** 🟡 60% READY

### **Level 1: Alerts List**
**Route:** `/alerts`  
**View:** All notifications

**Backend Status:**
- ✅ Database: `alerts` table - EXISTS
- ❌ GET `/api/v1/alerts` - MISSING ENDPOINT

---

## 10. **STRATEGY DRILLDOWNS** ❌ 30% READY

### **Level 1: Strategies List**
**Route:** `/strategies`  
**View:** All trading strategies

**Backend Status:**
- ✅ Database: `trading_strategies` table - EXISTS
- ❌ GET `/api/v1/strategies` - MISSING ENDPOINT

---

### **Level 2: Strategy Performance** ❌ MISSING
**Route:** `/strategies/:id/performance`  
**View:** Strategy-specific analytics

**Backend Status:**
- ❌ ALL strategy analytics - MISSING
- ❌ Need to link orders to strategies

**Database Gap:**
```sql
ALTER TABLE orders 
    ADD COLUMN strategy_id UUID REFERENCES trading_strategies(id);
```

---

## 📋 SUMMARY: MISSING COMPONENTS

### **🔴 CRITICAL MISSING (Blocks Core Functionality)**

#### **1. Missing Database Tables (5)**
```sql
1. portfolio_snapshots - Daily/intraday portfolio values
2. position_lots - Cost basis tracking (FIFO/LIFO)
3. position_snapshots - Position P&L over time
4. watchlists - Saved symbol lists
5. execution_stats - Execution quality aggregations
```

#### **2. Missing Endpoints (15 HIGH PRIORITY)**
```python
# Orders
GET /api/v1/trading/orders/:id - Order detail with fills
GET /api/v1/trading/orders/symbol/:symbol - Symbol order history

# Positions
GET /api/v1/trading/positions/:symbol/detail - Position deep dive
GET /api/v1/trading/positions/history - Closed positions

# Portfolio
GET /api/v1/analytics/portfolio/equity-curve - Time-series data
GET /api/v1/analytics/portfolio/allocation - Breakdown

# Transactions
GET /api/v1/trading/transactions - Transaction history

# Watchlists
GET /api/v1/watchlists - User watchlists
POST /api/v1/watchlists - Create watchlist

# Execution
GET /api/v1/analytics/execution/quality - Execution stats

# Alerts
GET /api/v1/alerts - User alerts

# Strategies
GET /api/v1/strategies - Strategy list
GET /api/v1/strategies/:id/performance - Strategy analytics

# Symbol Stats
GET /api/v1/analytics/symbol/:symbol/stats - Symbol-level metrics

# Market Data
GET /api/v1/market-data/order-book/:symbol - Level 2 data
```

---

### **🟡 MEDIUM PRIORITY (Nice to Have)**

#### **3. Missing Features**
- Intraday P&L tracking
- Trade journal with notes/tags
- Comparison views
- Advanced risk metrics (VaR, Greeks)
- Stress testing

---

### **🟢 LOW PRIORITY (Phase 9+)**

- Social features
- Community insights
- News integration
- Advanced charting tools
- Mobile app support

---

## ✅ RECOMMENDATION

### **Phase 8A: Critical Drilldown Support (MUST DO NOW)**

**Week 1-2: Database Schema**
1. Create `portfolio_snapshots` table
2. Create `position_lots` table
3. Create `position_snapshots` table
4. Create `watchlists` table
5. Add missing columns to existing tables

**Week 3-4: Backend Endpoints**
1. Order detail endpoint
2. Position detail endpoint
3. Portfolio equity curve
4. Transaction history
5. Watchlists CRUD
6. Alerts endpoint
7. Symbol stats

**Week 5-6: Frontend Implementation**
Start building with complete backend support

---

**Status:** 🔴 **CRITICAL GAPS IDENTIFIED**  
**Action Required:** Implement missing database tables and endpoints before frontend  
**Timeline:** 2-3 weeks for complete drilldown support
