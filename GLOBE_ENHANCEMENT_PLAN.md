# 🌍 Globe Enhancement - Comprehensive Implementation Plan

## 📊 Current State Analysis

### **What We Have** ✅
- 25 active stock exchanges across 22 countries
- Coverage:
  - 🌍 **Africa**: 3 exchanges (Egypt, South Africa, Kenya)
  - 🌎 **Americas**: 4 exchanges (US×2, Canada, Brazil, Mexico)
  - 🌏 **Asia**: 9 exchanges (China×2, India×2, Japan, Hong Kong, Singapore, S.Korea, Taiwan)
  - 🌍 **Europe**: 4 exchanges (UK, France, Germany, Spain, Switzerland)
  - 🌏 **Middle East**: 2 exchanges (UAE, Saudi Arabia)
  - 🌏 **Oceania**: 1 exchange (Australia)

- Political boundaries API endpoint (implemented but disabled)
- Arc connections (news-based connections between exchanges)
- Modal system for exchange details

### **What's Missing** ❌
- More African exchanges (Nigeria, Morocco, Tunisia)
- More European exchanges (Italy, Netherlands, Sweden, Norway)
- More Asian exchanges (Thailand, Malaysia, Indonesia, Philippines)
- Major asset locations (new feature)
- Real-time status indicators
- Filter controls UI

---

## 🎯 Enhancement Objectives

### **1. Expand Exchange Coverage** 📍
**Goal**: Add 15+ new exchanges to reach 40+ total

**Continents to Expand**:
- **Africa** (+4): Nigeria (NSE), Morocco (CSE), Tunisia (BVMT), Botswana (BSE)
- **Europe** (+5): Italy (FTSE MIB), Netherlands (AEX), Sweden (OMX), Norway (OSE), Russia (MOEX)
- **Asia** (+4): Thailand (SET), Malaysia (KLSE), Indonesia (IDX), Philippines (PSE)
- **Americas** (+2): Argentina (BCBA), Chile (BCS)

**Implementation**:
- Update `stock_exchanges_seed.sql` with new exchanges
- Research accurate lat/lng coordinates
- Add market cap data (USD)
- Add timezone information
- Reseed database

### **2. Enable Political Boundaries** 🗺️
**Goal**: Visualize country-level news sentiment

**Current Status**: Already implemented, just disabled

**Implementation**:
- Enable in `NewsPage.tsx`: `showBoundaries={true}`
- Test rendering performance
- Add toggle button in filter panel

**Features**:
- Hex overlay on countries
- Color by sentiment (green/red/blue)
- Hover shows country stats
- Click shows country modal

### **3. Major Asset Locations** 🏛️ (NEW FEATURE)
**Goal**: Real-time monitoring of market-moving locations

**Asset Types**:

1. **Central Banks** 🏦
   - Federal Reserve (Washington DC, USA)
   - European Central Bank (Frankfurt, Germany)
   - Bank of Japan (Tokyo, Japan)
   - Bank of England (London, UK)
   - People's Bank of China (Beijing, China)
   - Swiss National Bank (Zurich, Switzerland)
   - Bank of Canada (Ottawa, Canada)
   - Reserve Bank of Australia (Sydney, Australia)

2. **Commodities Markets** 🛢️
   - COMEX (New York - Gold, Silver, Copper)
   - NYMEX (New York - Oil, Gas)
   - ICE Futures (London - Brent Crude)
   - LME (London - Metals)
   - CBOT (Chicago - Grains, Bonds)

3. **Government/Financial Institutions** 🏛️
   - US Treasury (Washington DC)
   - IMF Headquarters (Washington DC)
   - World Bank (Washington DC)
   - BIS - Bank for International Settlements (Basel)
   - SEC Headquarters (Washington DC)

4. **Major Tech HQs** 🏢
   - Apple (Cupertino, CA)
   - Microsoft (Redmond, WA)
   - Google/Alphabet (Mountain View, CA)
   - Amazon (Seattle, WA)
   - Tesla (Austin, TX)
   - Meta (Menlo Park, CA)
   - NVIDIA (Santa Clara, CA)

5. **Energy Infrastructure** ⚡
   - OPEC Headquarters (Vienna, Austria)
   - Major oil fields (Saudi Arabia, Texas, North Sea)
   - Strategic reserves

**Status Indicators**:
- 🟢 **Green**: Active/Operating (recent news confirms activity)
- ⚪ **Grey**: No recent info / Unknown status
- 🔴 **Red**: Issues/Problems (negative news detected)

**Data Requirements**:
- Real-time news monitoring for each location
- Sentiment analysis on location-specific news
- Last updated timestamp
- Issue detection (strikes, shutdowns, disasters)

---

## 🗄️ Database Schema Design

### **New Table: `asset_locations`**
```sql
CREATE TABLE asset_locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(50) NOT NULL, -- 'central_bank', 'commodity_market', 'government', 'tech_hq', 'energy'
    country VARCHAR(100) NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    city VARCHAR(100),
    lat DECIMAL(10, 8) NOT NULL,
    lng DECIMAL(11, 8) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    description TEXT,
    importance_score INTEGER DEFAULT 50, -- 0-100, how influential this asset is
    website VARCHAR(255),
    icon_url VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_asset_locations_type ON asset_locations(asset_type);
CREATE INDEX idx_asset_locations_country ON asset_locations(country_code);
CREATE INDEX idx_asset_locations_active ON asset_locations(is_active);
```

### **New Table: `asset_status_log`**
```sql
CREATE TABLE asset_status_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES asset_locations(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL, -- 'operational', 'unknown', 'issue'
    sentiment_score DECIMAL(3, 2), -- -1.0 to 1.0
    news_count INTEGER DEFAULT 0,
    last_news_at TIMESTAMP,
    status_reason TEXT, -- Why this status was assigned
    checked_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (asset_id) REFERENCES asset_locations(id)
);

CREATE INDEX idx_asset_status_asset ON asset_status_log(asset_id);
CREATE INDEX idx_asset_status_time ON asset_status_log(checked_at DESC);
```

### **New Table: `asset_news_mentions`**
```sql
CREATE TABLE asset_news_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES asset_locations(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    relevance_score DECIMAL(3, 2), -- 0.0 to 1.0, how relevant the article is
    mentioned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(asset_id, article_id)
);

CREATE INDEX idx_asset_mentions_asset ON asset_news_mentions(asset_id);
CREATE INDEX idx_asset_mentions_article ON asset_news_mentions(article_id);
CREATE INDEX idx_asset_mentions_time ON asset_news_mentions(mentioned_at DESC);
```

---

## 🔌 API Endpoints Design

### **1. GET /api/v1/globe/assets**
**Purpose**: Get all asset locations with current status

**Query Parameters**:
- `timeframe`: 1h, 24h, 7d, 30d (default: 24h)
- `asset_type`: Filter by type (optional)
- `status`: operational, unknown, issue (optional)
- `min_importance`: Minimum importance score (0-100)

**Response**:
```json
{
  "assets": [
    {
      "id": "uuid",
      "code": "FED",
      "name": "Federal Reserve",
      "asset_type": "central_bank",
      "country": "United States",
      "country_code": "US",
      "city": "Washington DC",
      "flag": "🇺🇸",
      "lat": 38.8933,
      "lng": -77.0445,
      "timezone": "America/New_York",
      "importance_score": 100,
      "current_status": "operational",
      "sentiment_score": 0.2,
      "news_count": 15,
      "last_news_at": "2025-11-17T10:30:00Z",
      "categories": ["monetary_policy", "interest_rates"],
      "latest_articles": [...]
    }
  ],
  "total_count": 45,
  "status_summary": {
    "operational": 40,
    "unknown": 3,
    "issue": 2
  }
}
```

### **2. GET /api/v1/globe/assets/{asset_id}**
**Purpose**: Get detailed info for specific asset

**Response**: Same as above but single asset with full article list

### **3. POST /api/v1/globe/assets/update-status**
**Purpose**: Background job to update asset statuses

**Logic**:
1. For each asset, search news in last 24h
2. Calculate sentiment from articles
3. Determine status based on:
   - News count: 0 = unknown, >0 = check sentiment
   - Sentiment: >0.3 = operational, <-0.3 = issue, else unknown
4. Log status change
5. Update `asset_status_log`

---

## 🎨 Frontend Components

### **1. Asset Markers on Globe**
- Different shapes for different asset types:
  - 🏦 Central banks: Cube geometry
  - 🛢️ Commodities: Cylinder geometry
  - 🏛️ Government: Pyramid geometry (TetrahedronGeometry)
  - 🏢 Tech HQs: Box with antenna
  - ⚡ Energy: Cone geometry

- Color by status:
  - 🟢 Green: Operational
  - ⚪ Grey: Unknown
  - 🔴 Red: Issue

- Size by importance (0-100 score)

### **2. Asset Detail Modal**
Similar to exchange modal but with:
- Asset type icon/badge
- Status indicator with timestamp
- Importance score
- Recent news (filtered for this asset)
- "View Full Report" button

### **3. Filter Control Panel**
Floating panel on globe view:

```
┌─────────────────────────┐
│  🔍 Globe Filters       │
├─────────────────────────┤
│  Show:                  │
│  ☑ Exchanges (25)       │
│  ☑ Assets (45)          │
│  ☑ Connections          │
│  ☑ Boundaries           │
├─────────────────────────┤
│  Asset Types:           │
│  ☑ Central Banks (8)    │
│  ☑ Commodities (5)      │
│  ☑ Government (5)       │
│  ☑ Tech HQs (7)         │
│  ☑ Energy (5)           │
├─────────────────────────┤
│  Status:                │
│  ☑ Operational (40)     │
│  ☑ Unknown (3)          │
│  ☑ Issues (2)           │
├─────────────────────────┤
│  [Reset Filters]        │
└─────────────────────────┘
```

Position: Top-right of globe container
Collapsible: Click to expand/collapse
Smooth animations

---

## 📅 Implementation Phases

### **Phase 1: Expand Exchanges** (30 min)
1. ✅ Research new exchange coordinates
2. ✅ Update SQL seed file
3. ✅ Reseed database
4. ✅ Test API response
5. ✅ Verify markers appear

### **Phase 2: Enable Political Boundaries** (15 min)
1. ✅ Update NewsPage to enable boundaries
2. ✅ Test rendering
3. ✅ Add toggle in filter panel

### **Phase 3: Database Schema for Assets** (45 min)
1. ✅ Create migration file
2. ✅ Define asset_locations table
3. ✅ Define asset_status_log table
4. ✅ Define asset_news_mentions table
5. ✅ Create seed file with 40+ assets
6. ✅ Run migration + seed

### **Phase 4: Asset API Endpoints** (60 min)
1. ✅ Create `/api/v1/globe/assets` route
2. ✅ Implement GET handler with filters
3. ✅ Implement status calculation logic
4. ✅ Add asset detail endpoint
5. ✅ Test with curl/Postman

### **Phase 5: Asset Markers Frontend** (60 min)
1. ✅ Create useAssetData hook
2. ✅ Add asset markers to EnhancedGlobe
3. ✅ Different geometries per type
4. ✅ Color by status
5. ✅ Size by importance
6. ✅ Hover tooltips
7. ✅ Click-to-zoom animation

### **Phase 6: Asset Modal** (30 min)
1. ✅ Create AssetDetailModal component
2. ✅ Show status with indicator
3. ✅ Show recent news
4. ✅ Link to full articles

### **Phase 7: Filter Panel** (45 min)
1. ✅ Create GlobeFilterPanel component
2. ✅ Checkboxes for all categories
3. ✅ Real-time filtering
4. ✅ Collapsible UI
5. ✅ Smooth animations
6. ✅ Reset button

### **Phase 8: Real-Time Status Updates** (45 min)
1. ✅ Create background job script
2. ✅ Search news for each asset
3. ✅ Calculate status
4. ✅ Update database
5. ✅ Schedule job (cron or FastAPI scheduler)

---

## ⚡ Performance Considerations

### **Rendering**
- **Target**: 60 FPS with 70+ markers (25 exchanges + 45 assets)
- **Optimization**:
  - Use instanced geometries for same shapes
  - LOD (Level of Detail) based on camera distance
  - Frustum culling
  - Only update changed markers

### **Data Loading**
- **Pagination**: Load assets in chunks if >100
- **Caching**: Cache API responses for 30s
- **Lazy loading**: Load asset details on-demand

### **Database**
- **Indexes**: On all foreign keys and query fields
- **Materialized views**: For expensive aggregations
- **Connection pooling**: Reuse connections

---

## 🎯 Success Criteria

1. ✅ 40+ stock exchanges visible
2. ✅ Political boundaries toggle working
3. ✅ 40+ asset locations with real-time status
4. ✅ Color-coded status indicators accurate
5. ✅ Filter panel controls all visibility
6. ✅ Modal shows relevant info for clicked item
7. ✅ Smooth animations maintained (60fps)
8. ✅ News articles correctly linked to assets
9. ✅ Status updates every 5-15 minutes
10. ✅ All data from database (no mock data)

---

## 🔍 Research Sources

### **Exchange Data**
- World Federation of Exchanges (WFE)
- Each exchange's official website
- Market cap data from Bloomberg/Reuters APIs

### **Asset Locations**
- Official central bank websites
- CME Group for commodities
- Company investor relations pages
- Google Maps for coordinates

### **Status Detection Logic**
- Keyword matching: "operational", "shutdown", "strike", "issue"
- Sentiment analysis on news text
- Frequency: More news = more important
- Recency: Last 24h most relevant

---

**Total Estimated Time**: 6-7 hours
**Complexity**: High (new database tables, API routes, 3D rendering)
**Risk**: Medium (rendering performance with 70+ objects)

Ready to proceed? I'll implement in phases, starting with Phase 1.
