# 🌍 Global Market Pulse - Interactive 3D News Globe

**Technical Specification & Implementation Guide**

---

## **Visual Design**

### **Inspired By:**
Nairobi Securities Exchange (NSE) globe visualization with:
- Particle-based 3D Earth
- Data points showing market activity
- Interactive overlays
- Auto-rotation with manual control
- Atmospheric glow effects

### **Adapted For CIFT Markets:**
- Show global financial **news hotspots**
- Display **market sentiment** by region
- Highlight **major exchanges** worldwide
- Real-time **economic events** positioning

---

## **Technology Stack**

### **Core Libraries:**

```json
{
  "three.js": "^0.160.0",        // 3D rendering
  "three-globe": "^2.30.0",      // Globe utilities
  "@react-three/fiber": "^8.15.0", // React + Three.js
  "@react-three/drei": "^9.92.0",  // Three.js helpers
  "d3-geo": "^3.1.0",            // Geographic projections
  "topojson-client": "^3.1.0"    // World map data
}
```

### **Integration:**
- **Frontend:** SolidJS (will use Three.js directly, not React)
- **Backend:** Existing news API endpoints
- **Data:** PostgreSQL (news_articles with geo coordinates)
- **Real-time:** WebSocket for live updates

---

## **Data Schema Extension**

### **Add Geo-Location to News:**

```sql
-- Add columns to news_articles table
ALTER TABLE news_articles 
ADD COLUMN country VARCHAR(100),
ADD COLUMN country_code VARCHAR(3),
ADD COLUMN latitude DECIMAL(10, 8),
ADD COLUMN longitude DECIMAL(11, 8),
ADD COLUMN region VARCHAR(50);  -- Americas, Europe, Asia, Africa, Oceania

-- Index for geo queries
CREATE INDEX idx_news_geo ON news_articles(country_code, published_at DESC);
CREATE INDEX idx_news_region ON news_articles(region, published_at DESC);
```

### **Populate Geo Data:**

Use external service (during news fetch):
- **RestCountries API** - Free, country data
- **GeoNames API** - City/location coordinates
- **Our mapping** - Exchange symbols → Countries

---

## **Feature Breakdown**

### **1. Globe Rendering**

```typescript
// Globe Component Structure

interface GlobePoint {
  lat: number;
  lng: number;
  size: number;        // Number of articles
  color: string;       // Sentiment color
  pulse: boolean;      // Breaking news indicator
  label: string;       // Country/Exchange name
  articles: number;    // Article count
  sentiment: number;   // -1 to 1
}

interface ExchangeMarker {
  lat: number;
  lng: number;
  name: string;        // "NYSE", "NSE", "LSE"
  code: string;        // "US", "KE", "GB"
  logo: string;        // Exchange logo URL
  type: 'stock' | 'forex' | 'crypto';
}
```

### **2. Data Aggregation API**

```python
# New endpoint: /api/v1/news/globe-data

@router.get("/globe-data")
async def get_globe_news_data(
    hours: int = 24,  # Last N hours
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Returns news aggregated by country for globe visualization.
    
    Response:
    {
      "countries": [
        {
          "code": "US",
          "name": "United States",
          "lat": 37.0902,
          "lng": -95.7129,
          "article_count": 45,
          "sentiment": 0.23,  // Average sentiment
          "top_headline": "Fed Holds Rates Steady",
          "exchanges": ["NYSE", "NASDAQ"]
        }
      ],
      "exchanges": [...],
      "breaking_news": [...]  // Last 1 hour
    }
    """
```

### **3. Interactive Features**

#### **Auto-Rotation:**
```typescript
const globeRotation = {
  speed: 0.2,           // Degrees per second
  axis: 'y',            // Vertical rotation
  pauseOnHover: true,
  resumeAfter: 3000     // ms
};
```

#### **User Controls:**
```typescript
const controls = {
  zoom: { min: 100, max: 500 },
  rotate: true,          // Drag to rotate
  tilt: true,           // Mouse wheel to tilt
  autoReturn: 10000     // Return to auto-rotate after 10s idle
};
```

#### **Click Interactions:**

```typescript
// Click country → Show overlay
onCountryClick(countryCode) {
  1. Rotate globe to focus country
  2. Show overlay card (like NSE card in image)
  3. Load top 5 headlines
  4. Display market summary
  5. Show "View All →" link
}

// Click exchange marker → Filter news
onExchangeClick(exchange) {
  1. Navigate to news page
  2. Filter by country/exchange
  3. Highlight related symbols
}

// Click headline in overlay → Full article
onHeadlineClick(articleId) {
  1. Navigate to article detail
  2. Show full content
  3. Related news sidebar
}
```

---

## **Visual Effects**

### **1. Particle Globe:**
```typescript
const globeConfig = {
  particles: {
    count: 50000,      // Dot density
    size: 0.8,         // Dot size
    color: '#4a5568',  // Base gray
    glow: true
  },
  atmosphere: {
    color: '#7c3aed',  // Purple glow
    intensity: 0.6,
    thickness: 15
  },
  background: {
    stars: 1000,       // Star field
    nebula: true       // Purple/cyan gradient
  }
};
```

### **2. Data Point Effects:**

```typescript
const pointEffects = {
  idle: {
    glow: 0.3,
    pulse: false
  },
  active: {
    glow: 0.8,
    pulse: true,
    frequency: 1.5    // Hz
  },
  breaking: {
    glow: 1.0,
    pulse: true,
    frequency: 2.5,
    color: '#ef4444'  // Red for breaking
  }
};
```

### **3. Transition Animations:**

```typescript
const animations = {
  rotateToCountry: {
    duration: 1500,   // ms
    easing: 'easeInOutCubic'
  },
  overlayFadeIn: {
    duration: 300,
    delay: 800        // After rotation
  },
  dataPointPulse: {
    duration: 2000,
    repeat: Infinity
  }
};
```

---

## **Overlay Card Design**

### **Similar to NSE Card:**

```typescript
interface OverlayCard {
  position: 'center' | 'right' | 'left';
  content: {
    logo: string;           // Country flag / Exchange logo
    title: string;          // "United States" or "NYSE"
    subtitle: string;       // "Stocks • Forex • Crypto"
    metrics: {
      articles: number;
      sentiment: number;    // -1 to 1
      volume: string;       // "High", "Medium", "Low"
    };
    headlines: ArticlePreview[];  // Top 3-5
    cta: string;           // "View All News →"
  };
  theme: {
    background: 'rgba(0, 0, 0, 0.9)',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    blur: 20,              // Backdrop blur
    glow: true
  };
}
```

---

## **Performance Optimization**

### **1. Level of Detail (LOD):**
```typescript
const lodConfig = {
  closeView: {
    particles: 50000,
    labels: true,
    dataPoints: 'all'
  },
  mediumView: {
    particles: 20000,
    labels: 'major',
    dataPoints: 'aggregated'
  },
  farView: {
    particles: 5000,
    labels: false,
    dataPoints: 'top10'
  }
};
```

### **2. Data Caching:**
```typescript
const cache = {
  globeData: {
    ttl: 300000,        // 5 minutes
    refresh: 'background'
  },
  countryDetails: {
    ttl: 60000,         // 1 minute
    preload: ['US', 'GB', 'JP', 'CN']  // Major markets
  }
};
```

### **3. WebGL Optimization:**
```typescript
const performance = {
  antialiasing: 'auto',  // Disable on low-end devices
  shadows: false,
  physicallyCorrectLights: false,
  maxAnisotropy: 4
};
```

---

## **Data Flow**

```
┌─────────────────────────────────────────────────┐
│ 1. News Fetcher (External APIs)                │
│    - Finnhub, Alpha Vantage, NewsAPI          │
│    - Fetches articles with country codes       │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 2. Geo-Enrichment Service                      │
│    - Maps symbols/sources → countries          │
│    - Looks up coordinates                       │
│    - Assigns region (Americas, Europe, etc.)   │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 3. PostgreSQL (news_articles)                  │
│    - Stores articles with geo data             │
│    - Indexes by country_code, region           │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 4. Globe Data API (/api/v1/news/globe-data)   │
│    - Aggregates by country                      │
│    - Calculates sentiment scores                │
│    - Returns top headlines per region           │
└──────────────────┬──────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────┐
│ 5. Frontend Globe Component                    │
│    - Three.js renders globe                     │
│    - Plots data points                          │
│    - Handles interactions                       │
└─────────────────────────────────────────────────┘
```

---

## **Implementation Phases**

### **Phase 1: Data Foundation** (1-2 days)
- [ ] Add geo columns to `news_articles` table
- [ ] Create geo-enrichment service
- [ ] Update news fetcher to populate geo data
- [ ] Create `/api/v1/news/globe-data` endpoint

### **Phase 2: Basic Globe** (2-3 days)
- [ ] Install Three.js dependencies
- [ ] Create `GlobalNewsGlobe.tsx` component
- [ ] Render particle-based Earth
- [ ] Add auto-rotation
- [ ] Plot data points (basic circles)

### **Phase 3: Interactions** (2-3 days)
- [ ] Implement click handlers
- [ ] Add overlay card component
- [ ] Rotate-to-country animation
- [ ] Manual rotation controls
- [ ] Zoom controls

### **Phase 4: Visual Polish** (2-3 days)
- [ ] Atmospheric glow effects
- [ ] Pulsing animations for breaking news
- [ ] Sentiment-based colors
- [ ] Exchange markers with logos
- [ ] Space background with stars

### **Phase 5: Performance & UX** (1-2 days)
- [ ] LOD system
- [ ] Caching strategy
- [ ] Mobile responsiveness
- [ ] Loading states
- [ ] Error handling

---

## **Significance & Value**

### **User Benefits:**

1. **Intuitive Geographic Context**
   - See where news is happening globally
   - Understand regional market trends
   - Identify emerging markets

2. **Engaging Visual Experience**
   - Beautiful, modern interface
   - Fun to explore
   - Memorable brand experience

3. **Quick Market Pulse**
   - At-a-glance global sentiment
   - Identify hot regions
   - Spot breaking news instantly

4. **Discovery Tool**
   - Find news about unfamiliar markets
   - Explore exchanges worldwide
   - Learn about global finance

### **Business Value:**

1. **Differentiation**
   - Unique feature competitors don't have
   - Premium feel
   - Shareable on social media

2. **User Engagement**
   - Increases time on site
   - Encourages exploration
   - Reduces bounce rate

3. **Data Visualization Excellence**
   - Showcases technical capability
   - Attracts institutional clients
   - Press-worthy feature

---

## **Alternative: Simpler 2D Version**

If 3D is too complex, consider:

### **Interactive World Map (Mapbox/Leaflet):**
```
- 2D flat map
- Marker clustering
- Heatmap overlay
- Easier to implement
- Better mobile performance
- Still effective
```

**Effort:** 40% of 3D version  
**Impact:** 70% of 3D version

---

## **Recommended Approach**

### **Start Simple, Iterate:**

**Version 1.0** (MVP - 3 days)
- 2D world map with markers
- Click country → filter news
- Basic overlay card

**Version 1.5** (Enhanced - 5 days)
- Add 3D globe option (toggle view)
- Auto-rotation
- Simple data points

**Version 2.0** (Full Feature - 10 days)
- Full 3D globe like NSE
- All visual effects
- Complete interactions
- Performance optimization

---

## **Next Steps**

1. **Decide on approach:**
   - Full 3D (impressive, 10 days)
   - 2D map (practical, 3 days)
   - Hybrid (both options, 8 days)

2. **I can create:**
   - Complete implementation
   - All components
   - API endpoints
   - Database migrations

3. **Your input needed:**
   - Priority level (nice-to-have vs must-have)
   - Timeline constraints
   - Design preferences

**Would you like me to implement this feature? If yes, which version should I start with?**
