# 🌍 Interactive Financial News Globe - Complete Specification

## Executive Summary

Transform the CIFT Markets globe into an **advanced financial intelligence visualization platform** using `three-globe` library with animated arcs, political boundaries, stock exchange markers, and real-time news search.

---

## 🎯 Core Features

### 1. **Interactive Stock Exchange Markers**
Display major global stock exchanges as clickable markers.

**Implementation:**
```typescript
const stockExchanges = [
  {
    name: 'New York Stock Exchange',
    code: 'NYSE',
    lat: 40.7069,
    lng: -74.0113,
    country: 'United States',
    flag: '🇺🇸',
    newsCount: 150,
    sentiment: 0.25,
    marketCap: '31.9T USD',
    timezone: 'EST'
  },
  {
    name: 'Shanghai Stock Exchange',
    code: 'SSE',
    lat: 31.236750,
    lng: 121.508750,
    country: 'China',
    flag: '🇨🇳',
    newsCount: 89,
    sentiment: -0.10,
    marketCap: '6.8T USD',
    timezone: 'CST'
  },
  {
    name: 'London Stock Exchange',
    code: 'LSE',
    lat: 51.51528,
    lng: -0.09917,
    country: 'United Kingdom',
    flag: '🇬🇧',
    newsCount: 67,
    sentiment: 0.15,
    marketCap: '3.8T USD',
    timezone: 'GMT'
  },
  {
    name: 'Tokyo Stock Exchange',
    code: 'TSE',
    lat: 35.6762,
    lng: 139.6503,
    country: 'Japan',
    flag: '🇯🇵',
    newsCount: 54,
    sentiment: 0.05,
    marketCap: '5.6T USD',
    timezone: 'JST'
  },
  {
    name: 'Hong Kong Stock Exchange',
    code: 'HKEX',
    lat: 22.2783,
    lng: 114.1747,
    country: 'Hong Kong',
    flag: '🇭🇰',
    newsCount: 43,
    sentiment: -0.05,
    marketCap: '4.2T USD',
    timezone: 'HKT'
  },
  {
    name: 'Euronext',
    code: 'ENX',
    lat: 48.8566,
    lng: 2.3522,
    country: 'France',
    flag: '🇫🇷',
    newsCount: 38,
    sentiment: 0.08,
    marketCap: '4.0T USD',
    timezone: 'CET'
  },
  {
    name: 'Shenzhen Stock Exchange',
    code: 'SZSE',
    lat: 22.5431,
    lng: 114.0579,
    country: 'China',
    flag: '🇨🇳',
    newsCount: 35,
    sentiment: -0.08,
    marketCap: '3.7T USD',
    timezone: 'CST'
  },
  {
    name: 'Toronto Stock Exchange',
    code: 'TSX',
    lat: 43.6532,
    lng: -79.3832,
    country: 'Canada',
    flag: '🇨🇦',
    newsCount: 28,
    sentiment: 0.12,
    marketCap: '2.9T USD',
    timezone: 'EST'
  },
  {
    name: 'Bombay Stock Exchange',
    code: 'BSE',
    lat: 18.9294,
    lng: 72.8333,
    country: 'India',
    flag: '🇮🇳',
    newsCount: 31,
    sentiment: 0.18,
    marketCap: '3.4T USD',
    timezone: 'IST'
  },
  {
    name: 'Nairobi Securities Exchange',
    code: 'NSE',
    lat: -1.286389,
    lng: 36.817223,
    country: 'Kenya',
    flag: '🇰🇪',
    newsCount: 12,
    sentiment: 0.05,
    marketCap: '24B USD',
    timezone: 'EAT'
  }
];
```

**Marker Features:**
- 💙 Bright blue markers (#0088ff)
- 📊 Size based on article count
- 🎨 Pulse animation on new news
- 🎯 Click to zoom and show details

---

### 2. **Animated News Flow Arcs**
Visualize news connections between markets with flowing arcs.

**Use Cases:**
- **Trade Relationships**: Arc from NYSE → LSE for cross-border deals
- **Breaking News Spread**: Arc from origin country to affected markets
- **Economic Events**: Central bank decisions affecting multiple markets
- **Earnings Impact**: Company earnings affecting global suppliers

**Implementation:**
```typescript
const newsArcs = [
  {
    startLat: 40.7069,    // NYSE
    startLng: -74.0113,
    endLat: 51.51528,     // LSE
    endLng: -0.09917,
    color: ['#00ff88', '#0088ff'], // Gradient
    title: 'Fed Rate Decision Impact',
    newsCount: 23,
    animated: true,
    dashSpeed: 0.5
  },
  {
    startLat: 35.6762,    // TSE
    startLng: 139.6503,
    endLat: 31.236750,    // SSE
    endLng: 121.508750,
    color: ['#ff8800', '#ff0088'],
    title: 'Asia Market Correlation',
    newsCount: 15,
    animated: true,
    dashSpeed: 0.3
  }
];
```

**Arc Features:**
- 🌈 Color-coded by news category/sentiment
- ⚡ Animated dash flow showing direction
- 📈 Thickness based on news volume
- 🎬 Trigger on specific news events

---

### 3. **Political Boundaries Overlay**
Show country borders for better geographic context.

**Implementation:**
```typescript
// Using GeoJSON with political boundaries
globe
  .hexPolygonsData(countries.features)
  .hexPolygonResolution(3)
  .hexPolygonMargin(0.4)
  .hexPolygonColor(d => {
    const countryNews = newsDataByCountry[d.properties.ISO_A3];
    if (!countryNews) return 'rgba(100, 100, 100, 0.1)';
    
    // Color by sentiment
    if (countryNews.sentiment > 0.2) return 'rgba(0, 255, 100, 0.3)';
    if (countryNews.sentiment < -0.2) return 'rgba(255, 0, 100, 0.3)';
    return 'rgba(100, 150, 255, 0.2)';
  })
  .hexPolygonLabel(d => `
    <b>${d.properties.ADMIN}</b><br/>
    Articles: ${newsDataByCountry[d.properties.ISO_A3]?.count || 0}<br/>
    Sentiment: ${(newsDataByCountry[d.properties.ISO_A3]?.sentiment || 0).toFixed(2)}
  `);
```

**Boundary Features:**
- 🗺️ Country polygons with news heat mapping
- 🎨 Color intensity based on news volume
- 📊 Sentiment-based coloring (green/red)
- 🔍 Hover to show country statistics

---

### 4. **Advanced Search & Filtering**
Real-time search and filter system for news exploration.

**Search UI:**
```tsx
interface SearchControls {
  // Text search
  query: string;              // "Federal Reserve", "Tesla", "Bitcoin"
  
  // Geographic filters
  regions: string[];          // ["North America", "Asia Pacific"]
  countries: string[];        // ["United States", "China"]
  exchanges: string[];        // ["NYSE", "LSE", "SSE"]
  
  // Content filters
  categories: string[];       // ["earnings", "economics", "crypto"]
  sentiments: string[];       // ["positive", "neutral", "negative"]
  dateRange: {
    start: Date;
    end: Date;
  };
  
  // Visualization filters
  minNewsCount: number;       // Show only markers with X+ articles
  showArcs: boolean;          // Toggle arc visibility
  showBoundaries: boolean;    // Toggle political boundaries
  animationSpeed: number;     // Control animation speed
}
```

**Search Features:**
- 🔍 **Real-time search** as you type
- 🏷️ **Tag-based filtering** (click tags to filter)
- 🌍 **Geographic selection** (click country/exchange)
- 📅 **Time-based filtering** (last hour, day, week, custom)
- 🎯 **Multi-criteria** (combine filters)
- 💾 **Save searches** for later use

---

### 5. **Point Labels & Tooltips**
Rich information display on hover and click.

**Hover Tooltip:**
```tsx
<div class="globe-tooltip">
  <div class="tooltip-header">
    <span class="flag">{exchange.flag}</span>
    <strong>{exchange.name}</strong>
  </div>
  <div class="tooltip-body">
    <div class="stat">
      <span class="label">Articles:</span>
      <span class="value">{exchange.newsCount}</span>
    </div>
    <div class="stat">
      <span class="label">Sentiment:</span>
      <span class="value sentiment-{getSentimentClass()}">
        {(exchange.sentiment * 100).toFixed(1)}%
      </span>
    </div>
    <div class="stat">
      <span class="label">Market Cap:</span>
      <span class="value">{exchange.marketCap}</span>
    </div>
  </div>
  <div class="tooltip-footer">
    Click to view details →
  </div>
</div>
```

**Click Modal (Enhanced):**
```tsx
<div class="exchange-modal">
  <div class="modal-hero">
    <img src={exchange.icon} alt={exchange.name} />
    <div>
      <h2>{exchange.name}</h2>
      <p class="meta">
        {exchange.flag} {exchange.country} • {exchange.code}
      </p>
    </div>
  </div>
  
  <div class="modal-stats-grid">
    <StatCard label="Articles" value={exchange.newsCount} />
    <StatCard label="Sentiment" value={exchange.sentiment} type="sentiment" />
    <StatCard label="Market Cap" value={exchange.marketCap} />
    <StatCard label="Timezone" value={exchange.timezone} />
  </div>
  
  <div class="modal-news-feed">
    <h3>Latest News</h3>
    <For each={exchange.latestNews}>
      {article => <ArticleCard article={article} />}
    </For>
  </div>
  
  <div class="modal-actions">
    <button onClick={() => navigateToNews(exchange.code)}>
      View All {exchange.newsCount} Articles →
    </button>
    <button onClick={() => showConnections(exchange.code)}>
      Show Connected Markets
    </button>
  </div>
</div>
```

---

## 🏗️ Technical Architecture

### **Library Choice: `three-globe`**

**Why three-globe?**
- ✅ **Most powerful** for data visualization
- ✅ **Built-in support** for arcs, markers, polygons, labels
- ✅ **Highly performant** (WebGL rendering)
- ✅ **Active maintenance** (1.4k+ stars on GitHub)
- ✅ **Rich API** for customization
- ✅ **TypeScript support**

**Installation:**
```bash
npm install three-globe three
```

### **Component Structure:**

```
components/globe/
├── FinancialGlobe.tsx           # Main globe component
├── GlobeControls.tsx             # Search, filters, settings
├── ExchangeMarkers.tsx           # Stock exchange points
├── NewsArcs.tsx                  # Animated arc connections
├── PoliticalBoundaries.tsx       # Country polygons
├── ExchangeModal.tsx             # Click detail modal
├── SearchPanel.tsx               # Advanced search UI
├── GlobeTooltip.tsx              # Hover tooltips
├── hooks/
│   ├── useGlobeData.ts          # Data fetching
│   ├── useGlobeAnimation.ts     # Animation control
│   └── useGlobeSearch.ts        # Search logic
└── types/
    ├── exchange.ts              # Exchange data types
    ├── arc.ts                   # Arc data types
    └── globe-config.ts          # Configuration types
```

---

## 📊 Database Schema Updates

### **New Tables:**

#### 1. **Stock Exchanges**
```sql
CREATE TABLE stock_exchanges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(10) UNIQUE NOT NULL,    -- NYSE, LSE, SSE
    name VARCHAR(200) NOT NULL,
    country VARCHAR(100) NOT NULL,
    country_code VARCHAR(2) NOT NULL,     -- US, GB, CN
    lat DECIMAL(10, 6) NOT NULL,
    lng DECIMAL(10, 6) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    market_cap_usd BIGINT,
    trading_hours JSONB,                  -- {"open": "09:30", "close": "16:00"}
    website VARCHAR(500),
    icon_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_exchanges_country ON stock_exchanges(country_code);
CREATE INDEX idx_exchanges_location ON stock_exchanges USING GIST (
    ll_to_earth(lat, lng)
);
```

#### 2. **News Geographic Tags**
```sql
CREATE TABLE news_geotags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    exchange_id UUID REFERENCES stock_exchanges(id),
    country_code VARCHAR(2),
    lat DECIMAL(10, 6),
    lng DECIMAL(10, 6),
    relevance_score DECIMAL(3, 2),       -- 0.00 to 1.00
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_geotags_article ON news_geotags(article_id);
CREATE INDEX idx_geotags_exchange ON news_geotags(exchange_id);
CREATE INDEX idx_geotags_country ON news_geotags(country_code);
CREATE INDEX idx_geotags_location ON news_geotags USING GIST (
    ll_to_earth(lat, lng)
);
```

#### 3. **News Connections (for Arcs)**
```sql
CREATE TABLE news_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_exchange_id UUID NOT NULL REFERENCES stock_exchanges(id),
    target_exchange_id UUID NOT NULL REFERENCES stock_exchanges(id),
    article_id UUID NOT NULL REFERENCES news_articles(id),
    connection_type VARCHAR(50),          -- trade, impact, correlation
    strength DECIMAL(3, 2),               -- 0.00 to 1.00
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_connections_source ON news_connections(source_exchange_id);
CREATE INDEX idx_connections_target ON news_connections(target_exchange_id);
CREATE INDEX idx_connections_type ON news_connections(connection_type);
CREATE INDEX idx_connections_created ON news_connections(created_at DESC);
```

---

## 🎨 UI/UX Design

### **Layout:**

```
┌─────────────────────────────────────────────────────────┐
│ [🔍 Search News...]  [Filters ▼]  [Time: 24h ▼]       │  Search Bar
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    🌍 GLOBE VIEW                        │
│                                                         │
│     [Interactive 3D Globe with Markers & Arcs]         │  Main Canvas
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Active Filters: [NYSE ×] [Positive ×] [Clear All]     │  Filter Chips
├─────────────────────────────────────────────────────────┤
│ Legend:                                                 │
│ 🔵 Stock Exchange  🌈 News Flow  🗺️ Sentiment Heat    │  Legend
└─────────────────────────────────────────────────────────┘

┌────────── SIDE PANEL (Toggle) ──────────┐
│ 📊 Quick Stats                           │
│ • 245 Articles Today                     │
│ • 18 Exchanges Active                    │
│ • +15% Positive Sentiment                │
│                                          │
│ 🔥 Trending Markets                      │
│ 1. NYSE (85 articles)                    │
│ 2. SSE (62 articles)                     │
│ 3. LSE (48 articles)                     │
│                                          │
│ 🌐 Regional Activity                     │
│ [Chart: News by Region]                  │
└──────────────────────────────────────────┘
```

### **Interaction Flow:**

1. **Initial Load**
   - Globe auto-rotates slowly
   - Markers pulse based on recent news
   - Arcs animate showing news flow

2. **Search/Filter**
   - Type in search bar → instant globe update
   - Click filter tags → markers/arcs filter
   - Date range selection → historical view

3. **Hover Marker**
   - Cursor → pointer
   - Tooltip appears instantly
   - Marker scales up 20%

4. **Click Marker**
   - Auto-rotation stops
   - Camera zooms smoothly (1s animation)
   - Modal slides in from bottom
   - Show all connected arcs

5. **Modal Interactions**
   - Click "View All Articles" → Navigate to filtered news
   - Click "Show Connections" → Highlight related markets
   - Close button → Camera returns, rotation resumes

---

## 🚀 Implementation Roadmap

### **Phase 1: Foundation (Week 1)**
- [ ] Install `three-globe` and dependencies
- [ ] Create basic FinancialGlobe component
- [ ] Set up stock exchange data
- [ ] Implement basic markers
- [ ] Add hover tooltips

### **Phase 2: Interactivity (Week 2)**
- [ ] Add click-to-zoom functionality
- [ ] Build exchange modal
- [ ] Implement OrbitControls
- [ ] Add distance-based marker scaling
- [ ] Create search UI skeleton

### **Phase 3: Advanced Features (Week 3)**
- [ ] Implement animated arcs
- [ ] Add political boundaries overlay
- [ ] Build news connection logic
- [ ] Implement real-time search
- [ ] Add filter system

### **Phase 4: Database Integration (Week 4)**
- [ ] Create new database tables
- [ ] Build geotag extraction script
- [ ] Implement connection detection algorithm
- [ ] Add API endpoints for globe data
- [ ] Migrate existing news to geotags

### **Phase 5: Polish & Optimization (Week 5)**
- [ ] Performance optimization
- [ ] Add loading states
- [ ] Implement error handling
- [ ] Add animations and transitions
- [ ] Mobile responsiveness
- [ ] User preferences (save settings)

---

## 📝 API Endpoints

### **GET /api/v1/globe/exchanges**
Returns all stock exchanges with news counts.

**Response:**
```json
{
  "exchanges": [
    {
      "id": "uuid",
      "code": "NYSE",
      "name": "New York Stock Exchange",
      "country": "United States",
      "country_code": "US",
      "flag": "🇺🇸",
      "lat": 40.7069,
      "lng": -74.0113,
      "news_count": 150,
      "sentiment_score": 0.25,
      "market_cap_usd": 31900000000000,
      "latest_articles": [...]
    }
  ],
  "total_news_count": 456,
  "last_updated": "2025-11-17T09:30:00Z"
}
```

### **GET /api/v1/globe/arcs**
Returns news connections for arc visualization.

**Query Params:**
- `timeframe`: "1h", "24h", "7d", "30d"
- `min_strength`: 0.0-1.0
- `connection_type`: "trade", "impact", "correlation"

**Response:**
```json
{
  "arcs": [
    {
      "id": "uuid",
      "source": {
        "code": "NYSE",
        "lat": 40.7069,
        "lng": -74.0113
      },
      "target": {
        "code": "LSE",
        "lat": 51.51528,
        "lng": -0.09917
      },
      "article_count": 23,
      "connection_type": "impact",
      "strength": 0.85,
      "primary_article": {...}
    }
  ]
}
```

### **GET /api/v1/globe/boundaries**
Returns country-level news aggregation for political boundaries.

**Response:**
```json
{
  "countries": [
    {
      "country_code": "US",
      "name": "United States",
      "article_count": 234,
      "sentiment_score": 0.18,
      "top_categories": ["economics", "earnings", "market"],
      "exchanges": ["NYSE", "NASDAQ"]
    }
  ]
}
```

### **GET /api/v1/globe/search**
Search and filter globe data.

**Query Params:**
- `q`: Search query
- `exchanges`: Comma-separated codes
- `countries`: Comma-separated codes
- `categories`: Comma-separated categories
- `sentiment`: "positive", "neutral", "negative"
- `date_from`: ISO date
- `date_to`: ISO date

**Response:**
```json
{
  "results": {
    "exchanges": [...],
    "arcs": [...],
    "article_count": 42
  },
  "filters_applied": {...},
  "execution_time_ms": 45
}
```

---

## 🎯 Success Metrics

**User Engagement:**
- Click-through rate on markers → news articles
- Average time spent on globe view
- Number of searches performed
- Filter combinations used

**Technical Performance:**
- Page load time < 2s
- 60 FPS animation
- API response time < 100ms
- Bundle size < 500KB (globe component)

**Business Impact:**
- Increased news discovery
- Longer session duration
- Higher article read rate
- User satisfaction score

---

## 🔮 Future Enhancements

1. **Real-time Updates**
   - WebSocket for live news feed
   - Markers pulse on breaking news
   - Auto-refresh arcs

2. **AI-Powered Features**
   - Auto-detect news connections
   - Smart article clustering
   - Predictive arc suggestions

3. **Social Features**
   - Share custom globe views
   - Collaborative annotations
   - Community highlights

4. **Advanced Visualizations**
   - Heat maps over time
   - 3D bar charts on countries
   - Volume-based terrain elevation

5. **Mobile App**
   - React Native globe
   - AR globe view
   - Gesture controls

---

## 📚 Technical References

**Libraries:**
- `three-globe`: https://github.com/vasturiano/three-globe
- `three.js`: https://threejs.org/
- `@tweenjs/tween.js`: https://github.com/tweenjs/tween.js

**Data Sources:**
- Stock Exchange Coordinates: Wikipedia/Wikidata
- Political Boundaries: Natural Earth Data
- Country GeoJSON: TopoJSON World Atlas

**Inspiration:**
- Shopify BFCM Globe
- GitHub Contribution Globe
- Flight Path Visualizations

---

**Status**: 📝 Specification Complete - Ready for Implementation

This specification provides a complete roadmap for building a world-class interactive financial news globe that will set CIFT Markets apart from competitors.
