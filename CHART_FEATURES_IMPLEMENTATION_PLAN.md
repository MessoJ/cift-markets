# 📊 Chart Features - Complete Implementation Plan

**Start Time**: 2025-11-16 15:53 UTC+3  
**Estimated Duration**: ~3 hours total  
**Complexity**: Advanced, Production-Ready

---

## 🎯 Five Features to Implement

### **1. Real-time WebSocket Updates** (30 mins)
**Current State**: WebSocket hook exists, subscribing to symbols, but NOT updating chart candles  
**What's Missing**:
- Candle updates on chart (last bar modification)
- Live price ticker UI
- Visual indicator (pulsing dot)
- Performance optimization

**Implementation**:
- Wire `onCandle` callback to update last bar
- Add live price display in chart header
- Connection status indicator
- Efficient re-render (only last candle)

---

### **2. Full Technical Indicators** (45 mins)
**Current State**: MACD/Bollinger calculated in backend, NOT rendered  
**What's Missing**:
- RSI calculation in backend (Polars)
- Separate indicator panels (MACD, RSI below main chart)
- Bollinger Bands rendering on main chart
- Panel resize/collapse

**Backend Work**:
```python
# Add to data_processing.py calculate_technical_indicators()
# RSI calculation using Polars
def calculate_rsi(closes: pl.Series, period: int = 14) -> pl.Series:
    delta = closes.diff()
    gain = delta.clip_min(0)
    loss = -delta.clip_max(0)
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df = df.with_columns([
    calculate_rsi(pl.col("close"), 14).alias("rsi_14"),
])
```

**Frontend Work**:
- Multi-grid ECharts layout (main chart + 2 indicator panels)
- MACD panel: Line (MACD), Line (Signal), Bar (Histogram)
- RSI panel: Line with 30/70 levels
- Sync zoom/pan across all panels
- Bollinger Bands on main chart (3 lines: upper, middle, lower)

---

### **3. Multi-Timeframe Analysis** (40 mins)
**Current State**: Single chart only  
**What's Needed**:
- Grid layout (2x2 or 3x1)
- Each panel independent timeframe
- Synchronized symbol selection
- Shared drawing tools
- Performance: lazy loading

**Layout Options**:
```
┌────────────┬────────────┐
│   1d       │    1h      │  2x2 Grid
├────────────┼────────────┤
│   15m      │    5m      │
└────────────┴────────────┘

OR

┌──────────────────────────┐
│         1d               │  3x1 Stack
├──────────────────────────┤
│         1h               │
├──────────────────────────┤
│         15m              │
└──────────────────────────┘
```

**Implementation**:
- Multi-chart wrapper component
- Shared state for symbol
- Independent state for timeframe, indicators
- Toggle between single/multi view
- LocalStorage for layout preference

---

### **4. Chart Templates** (35 mins)
**Current State**: No save/load functionality  
**What's Needed**:
- PostgreSQL table for templates
- Save current config (indicators, timeframe, chart type, drawings)
- Load template
- Template management UI
- Default templates

**Database Schema**:
```sql
CREATE TABLE chart_templates (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    -- config structure:
    -- {
    --   symbol: 'AAPL',
    --   timeframe: '1d',
    --   chartType: 'candlestick',
    --   indicators: [{id: 'sma_20', enabled: true, color: '#3b82f6'}],
    --   layout: 'single' | 'multi',
    --   multi_timeframes: ['1d', '1h', '15m']
    -- }
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**UI**:
- "Save Template" button in chart controls
- Template dropdown/modal
- "Apply Template" → Loads config
- Template CRUD (create, rename, delete)

---

### **5. Price Alerts** (40 mins)
**Current State**: No alert system  
**What's Needed**:
- Set alerts at price levels
- PostgreSQL storage
- Visual markers on chart
- Alert trigger mechanism
- Notification UI

**Database Schema**:
```sql
CREATE TABLE price_alerts (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(20) NOT NULL, -- 'above', 'below', 'crosses'
    price NUMERIC(12, 4) NOT NULL,
    triggered BOOLEAN DEFAULT false,
    triggered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);
```

**Features**:
- Right-click chart → "Set Alert Here"
- Alert lines on chart (dashed yellow)
- Alert list sidebar
- Real-time checking (WebSocket)
- Browser notification when triggered
- Email notification (optional)

---

## 📋 Implementation Order & Dependencies

```
Phase 1: Real-time Updates (No dependencies)
   ├─ Already 80% done (WebSocket exists)
   └─ Just wire to chart candle updates

Phase 2: Technical Indicators (Depends on Phase 1 for real-time)
   ├─ Add RSI to backend
   ├─ Create multi-panel layout
   └─ Render MACD, RSI, Bollinger

Phase 3: Multi-Timeframe (Depends on Phase 2 for complete chart)
   ├─ Multiple chart instances
   └─ Layout management

Phase 4: Chart Templates (Depends on Phase 2, Phase 3)
   ├─ Database schema
   ├─ Save/load API
   └─ Template UI

Phase 5: Price Alerts (Depends on Phase 1 for real-time checking)
   ├─ Database schema
   ├─ Alert API
   ├─ Chart markers
   └─ Notification system
```

---

## 🚀 Technical Stack

### **Backend**:
- **FastAPI**: WebSocket + REST API
- **PostgreSQL**: Templates, alerts storage
- **Polars**: Indicator calculations (19.5x faster than Pandas)
- **QuestDB**: OHLCV data
- **asyncpg**: High-performance DB queries

### **Frontend**:
- **SolidJS**: Reactive state management
- **ECharts**: Chart rendering (GPU-accelerated)
- **TypeScript**: Type safety
- **TailwindCSS**: Styling
- **WebSocket API**: Real-time data

---

## 📊 Performance Targets

| Feature | Target Performance |
|---------|-------------------|
| WebSocket latency | <50ms tick to chart update |
| Indicator calculation | <10ms for 500 bars |
| Multi-chart render | <200ms for 4 charts |
| Template save/load | <100ms round-trip |
| Alert checking | <5ms per alert per tick |

**Overall**: 60fps smooth chart updates, no jank

---

## 🧪 Testing Plan

### **Phase 1: Real-time**
- Connect WebSocket → See "connected" status
- Watch chart → Last candle updates live
- Change symbol → Unsubscribe old, subscribe new
- Disconnect → Shows "disconnected" + reconnect attempt

### **Phase 2: Indicators**
- Enable MACD → Panel appears below chart
- Enable RSI → Second panel appears
- Enable Bollinger → 3 lines on main chart
- Zoom chart → All panels zoom together

### **Phase 3: Multi-Timeframe**
- Toggle multi-view → 4 charts appear
- Change symbol → All charts update
- Each chart has different timeframe
- Drawings only on active chart

### **Phase 4: Templates**
- Configure chart (indicators, timeframe, etc.)
- Click "Save Template" → Name it "Day Trading"
- Load template → Chart restores config
- Delete template → Removed from list

### **Phase 5: Alerts**
- Right-click chart @ $150 → "Set Alert Above $150"
- Alert appears as yellow dashed line
- Price crosses $150 → Browser notification
- Alert marked as triggered, line turns green

---

## 📁 Files to Create/Modify

### **Create** (8 files):
1. `cift/db/migrations/010_create_chart_templates.sql`
2. `cift/db/migrations/011_create_price_alerts.sql`
3. `cift/api/routes/chart_templates.py`
4. `cift/api/routes/price_alerts.py`
5. `frontend/src/components/charts/IndicatorPanels.tsx`
6. `frontend/src/components/charts/MultiTimeframeView.tsx`
7. `frontend/src/components/charts/TemplateManager.tsx`
8. `frontend/src/components/charts/AlertManager.tsx`

### **Modify** (5 files):
1. `cift/core/data_processing.py` (add RSI)
2. `frontend/src/components/charts/CandlestickChart.tsx` (candle updates)
3. `frontend/src/pages/charts/ChartsPage.tsx` (integrate new features)
4. `frontend/src/lib/api/client.ts` (template, alert APIs)
5. `cift/api/main.py` (register new routes)

**Total**: ~1500 lines of production code

---

## 🎯 Success Criteria

### **Phase 1**: ✅ Chart updates live, <50ms latency
### **Phase 2**: ✅ MACD, RSI, Bollinger all rendering
### **Phase 3**: ✅ 4 timeframes simultaneously, smooth
### **Phase 4**: ✅ Templates save/load all config
### **Phase 5**: ✅ Alerts trigger and notify

**Overall**: Professional-grade charting platform, TradingView competitor

---

**Ready to implement?** Starting with Phase 1... 🚀
