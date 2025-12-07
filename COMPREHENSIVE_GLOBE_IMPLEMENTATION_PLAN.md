# 🎯 Comprehensive Globe Implementation Plan - Research-Based

## 📊 Research Summary

Based on research of Three.js globe implementations, best practices found:

### **Political Boundaries (All Countries)**:
- **Source**: Use `natural-earth-data` 110m or 50m resolution GeoJSON
- **Library**: `three-geojson-geometry` for proper polygon extrusion
- **Method**: Raycasting for country click detection
- **Performance**: Simplified geometry for 195+ countries

### **Camera/Zoom**:
- **City Zoom**: TWEEN.js with lat/lng to Vector3 conversion
- **Duration**: 1500-2000ms for smooth animation
- **Easing**: `TWEEN.Easing.Cubic.InOut`
- **LookAt**: Animate both camera position AND controls.target

### **Search**:
- **Library**: Fuse.js for fuzzy search
- **Datasets**: Cities (40k+), Countries (195), Assets (63), Exchanges (40)
- **UI**: Autocomplete dropdown with categories

---

## 🔧 Implementation Tasks

### **1. Political Boundaries - PROPER (All 195 Countries)** 

#### Current Issues:
- ❌ Only renders ~20 countries with news data
- ❌ Africa not broken into 54 countries
- ❌ Uses circles, not real borders
- ❌ No click functionality

#### Solution:
```typescript
// Use proper GeoJSON source
const GEOJSON_URL = 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson';

// Render ALL countries, color only those with news
features.forEach(country => {
  const iso_a2 = country.properties.iso_a2;
  const sentimentData = sentimentMap[iso_a2]; // undefined if no news
  
  // Render all countries with default color
  const color = sentimentData 
    ? getSentimentColor(sentimentData.sentiment)
    : 0x333333; // Grey for countries without news
  
  // Make clickable with raycaster
  countryMesh.userData = {
    type: 'country',
    code: iso_a2,
    name: country.properties.name,
    hasData: !!sentimentData
  };
});
```

#### Files to Modify:
- `EnhancedFinancialGlobe.tsx` - updateBoundaries()
- Add `CountryModal.tsx` - New component

---

### **2. Country Click Modal**

#### Data to Display:
```typescript
interface CountryData {
  name: string;
  code: string;
  flag: string; // emoji
  // Economic Data
  gdp: number; // from World Bank API or database
  gdp_growth: number;
  inflation: number;
  unemployment: number;
  // News
  topNews: NewsArticle; // Most important market-relevant
  newsSummary: {
    positive: number;
    neutral: number;
    negative: number;
  };
  recentNews: NewsArticle[]; // 3 more
  // Map Data
  exchanges: number; // count
  assets: number; // count
  sentiment: number;
}
```

#### Backend Endpoint:
```python
# cift/api/routes/globe.py
@router.get("/countries/{country_code}")
async def get_country_details(
    country_code: str,
    db: asyncpg.Connection = Depends(get_db)
):
    """
    Get comprehensive country data including:
    - Economic indicators (GDP, inflation from cached data)
    - News analysis (top story, counts, sentiment)
    - Exchange/asset counts
    """
    # Query news
    # Query economic indicators table
    # Query exchanges/assets in country
    return CountryDetailResponse(...)
```

#### Database Addition:
```sql
-- Add country economic indicators table
CREATE TABLE country_indicators (
    country_code VARCHAR(2) PRIMARY KEY,
    country_name VARCHAR(255),
    gdp_usd BIGINT,
    gdp_growth_rate DECIMAL(5,2),
    inflation_rate DECIMAL(5,2),
    unemployment_rate DECIMAL(5,2),
    last_updated TIMESTAMP,
    source VARCHAR(100)
);
```

---

### **3. City Zoom Feature**

#### Implementation:
```typescript
interface CityZoomTarget {
  name: string;
  lat: number;
  lng: number;
  altitude: number; // How close to zoom (50-150)
}

function zoomToCity(city: CityZoomTarget) {
  controls.autoRotate = false;
  
  // Convert lat/lng to 3D position
  const targetPos = latLonToVector3(
    city.lat, 
    city.lng, 
    GLOBE_RADIUS + city.altitude
  );
  
  // Calculate lookAt point (on globe surface)
  const lookAtPos = latLonToVector3(
    city.lat, 
    city.lng, 
    GLOBE_RADIUS
  );
  
  // Animate camera
  new TWEEN.Tween(camera.position)
    .to(targetPos, 2000)
    .easing(TWEEN.Easing.Cubic.InOut)
    .start();
    
  // Animate controls target (what camera looks at)
  new TWEEN.Tween(controls.target)
    .to(lookAtPos, 2000)
    .easing(TWEEN.Easing.Cubic.InOut)
    .onUpdate(() => controls.update())
    .start();
}

// Usage:
zoomToCity({
  name: 'New York',
  lat: 40.7128,
  lng: -74.0060,
  altitude: 80
});
```

#### UI Trigger:
- Search result click → zoom to city
- Country modal → "View Capital" button
- Asset detail → "Zoom to Location" button

---

### **4. Search Feature**

#### Component Structure:
```tsx
<div class="globe-search-container">
  <input 
    type="search"
    placeholder="Search cities, countries, assets, exchanges..."
    onInput={handleSearch}
  />
  <div class="search-results">
    <Show when={results().length > 0}>
      <For each={results()}>
        {(result) => (
          <div class="search-result-item" onClick={() => selectResult(result)}>
            <span class="result-icon">{getIcon(result.type)}</span>
            <span class="result-name">{result.name}</span>
            <span class="result-type">{result.type}</span>
          </div>
        )}
      </For>
    </Show>
  </div>
</div>
```

#### Search Implementation:
```typescript
import Fuse from 'fuse.js';

// Combined search dataset
const searchData = [
  ...cities.map(c => ({ type: 'city', ...c })),
  ...countries.map(c => ({ type: 'country', ...c })),
  ...assets.map(a => ({ type: 'asset', ...a })),
  ...exchanges.map(e => ({ type: 'exchange', ...e })),
];

const fuse = new Fuse(searchData, {
  keys: ['name', 'code', 'country', 'city'],
  threshold: 0.3,
  limit: 10
});

function handleSearch(query: string) {
  const results = fuse.search(query);
  setResults(results.map(r => r.item));
}

function selectResult(result: SearchResult) {
  switch(result.type) {
    case 'city':
      zoomToCity(result);
      break;
    case 'country':
      zoomToCountry(result);
      showCountryModal(result);
      break;
    case 'asset':
      zoomToAsset(result);
      showAssetModal(result);
      break;
    case 'exchange':
      zoomToExchange(result);
      showExchangeModal(result);
      break;
  }
}
```

#### City Database:
```sql
-- Add major cities table
CREATE TABLE major_cities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    country VARCHAR(255) NOT NULL,
    lat DECIMAL(10, 8) NOT NULL,
    lng DECIMAL(11, 8) NOT NULL,
    population BIGINT,
    is_capital BOOLEAN DEFAULT false,
    importance_score INTEGER DEFAULT 50
);

-- Seed with top 1000 cities
INSERT INTO major_cities (name, country_code, country, lat, lng, population, is_capital)
VALUES
  ('New York', 'US', 'United States', 40.7128, -74.0060, 8336817, false),
  ('London', 'GB', 'United Kingdom', 51.5074, -0.1278, 8982000, true),
  ('Tokyo', 'JP', 'Japan', 35.6762, 139.6503, 13960000, true),
  -- ... 997 more cities
```

---

### **5. Reduce Modal Size**

#### Current Issue:
- Modals too large, cover too much screen

#### Solution:
```css
/* ExchangeDetailModal.module.css */
.modal-overlay {
  /* Reduce max-width */
  max-width: 600px; /* Was: 800px */
  max-height: 70vh; /* Was: 80vh */
  padding: 1.5rem; /* Was: 2rem */
}

.modal-content {
  font-size: 0.9rem; /* Was: 1rem */
}

.modal-header h2 {
  font-size: 1.5rem; /* Was: 2rem */
}

/* Make scrollable */
.modal-body {
  max-height: 50vh;
  overflow-y: auto;
}
```

Apply to:
- `ExchangeDetailModal.tsx`
- `AssetDetailModal.tsx` (new)
- `CountryModal.tsx` (new)

---

### **6. Fix Globe Cutoff at Bottom**

#### Current Issue:
- Globe cut at bottom, not fully visible

#### Solution:
```css
/* EnhancedFinancialGlobe.module.css or inline styles */
.globe-container {
  width: 100%;
  height: 100%; /* Ensure full height */
  position: relative;
  overflow: hidden; /* Prevent scrollbars */
}

/* Adjust canvas positioning */
canvas {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%); /* Center perfectly */
  width: 100% !important;
  height: 100% !important;
}
```

```typescript
// In init() function
function init() {
  // Set camera aspect ratio properly
  camera = new THREE.PerspectiveCamera(
    45, // FOV
    containerRef!.clientWidth / containerRef!.clientHeight, // Proper aspect
    1,
    2000
  );
  camera.position.z = 250; // Adjust distance to show full globe
  
  // Set renderer size
  renderer.setSize(
    containerRef!.clientWidth,
    containerRef!.clientHeight
  );
}
```

---

### **7. Comprehensive Asset Filters**

#### Current Filters:
- Asset Types (partial)
- Asset Status (exists)
- Show/Hide toggles

#### Enhanced Filters:
```typescript
interface GlobeFilters {
  // Existing
  showExchanges: boolean;
  showAssets: boolean;
  showArcs: boolean;
  showBoundaries: boolean;
  showShips: boolean; // NEW
  
  // Asset Type (enhanced)
  assetTypes: {
    central_bank: boolean;
    commodity_market: boolean;
    government: boolean;
    tech_hq: boolean;
    energy: boolean;
  };
  
  // Asset Status
  assetStatus: {
    operational: boolean;
    unknown: boolean;
    issue: boolean;
  };
  
  // NEW: Country Filter
  countries: string[]; // ISO2 codes, empty = all
  
  // NEW: Importance Filter
  minImportance: number; // 0-100
  
  // NEW: Ship Type Filter
  shipTypes: {
    oil_tanker: boolean;
    lng_carrier: boolean;
    container: boolean;
    bulk_carrier: boolean;
    chemical_tanker: boolean;
  };
  
  // NEW: Geographic Region
  regions: {
    americas: boolean;
    europe: boolean;
    asia: boolean;
    africa: boolean;
    oceania: boolean;
  };
}
```

#### UI Component:
```tsx
// GlobeFilterPanel.tsx - Enhanced
<div class="filter-section">
  <h4>🌍 Regions</h4>
  <label><input type="checkbox" checked={filters().regions.africa} onChange={...} /> Africa</label>
  <label><input type="checkbox" checked={filters().regions.asia} onChange={...} /> Asia</label>
  <!-- etc -->
</div>

<div class="filter-section">
  <h4>🚢 Ships</h4>
  <label><input type="checkbox" checked={filters().showShips} onChange={...} /> Show Ships</label>
  <label><input type="checkbox" checked={filters().shipTypes.oil_tanker} onChange={...} /> Oil Tankers</label>
  <!-- etc -->
</div>

<div class="filter-section">
  <h4>📊 Importance</h4>
  <input 
    type="range" 
    min="0" 
    max="100" 
    value={filters().minImportance}
    onInput={(e) => setFilters({ ...filters(), minImportance: +e.target.value })}
  />
  <span>Min: {filters().minImportance}</span>
</div>
```

---

### **8. Database Seeding & Frontend Integration**

#### Issue:
- Data created but not actually seeded to database
- Frontend may not see the data

#### Solution Steps:

**Step 1: Verify Database Container**
```powershell
docker ps | findstr postgres
# Get actual container name
```

**Step 2: Copy Seed Files**
```powershell
# Copy all seeds
docker cp database/seeds/african_assets_seed.sql <container_name>:/tmp/
docker cp database/seeds/ships_seed.sql <container_name>:/tmp/
docker cp database/migrations/007_create_ship_tracking.sql <container_name>:/tmp/
```

**Step 3: Run Migrations & Seeds**
```powershell
# Run ship migration
docker exec <container_name> psql -U cift_user -d cift_markets -f /tmp/007_create_ship_tracking.sql

# Seed African assets
docker exec <container_name> psql -U cift_user -d cift_markets -f /tmp/african_assets_seed.sql

# Seed ships
docker exec <container_name> psql -U cift_user -d cift_markets -f /tmp/ships_seed.sql
```

**Step 4: Verify Data**
```powershell
# Check asset count
docker exec <container_name> psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM asset_locations WHERE is_active = true;"

# Should return 63 (40 + 23)

# Check African assets specifically
docker exec <container_name> psql -U cift_user -d cift_markets -c "SELECT code, name, country FROM asset_locations WHERE country_code IN ('ZA', 'NG', 'EG', 'KE') ORDER BY country;"

# Should return 15+ results

# Check ships
docker exec <container_name> psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM tracked_ships;"

# Should return 16
```

**Step 5: Test Backend APIs**
```powershell
# Test assets endpoint
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/assets/" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | Select-Object total_count

# Should return 63

# Test ships endpoint
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/ships" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | Select-Object total_count

# Should return 16
```

**Step 6: Update Asset Statuses**
```powershell
# Run enhanced news analysis
python scripts/update_asset_status.py

# Should process 63 assets and set statuses
```

**Step 7: Verify Frontend**
```powershell
cd frontend
npm run dev

# Open browser DevTools console
# Should see logs:
# "✨ Creating asset markers for 63 assets"
# "✨ Creating boundaries for X countries"
```

---

## 📁 Files to Create/Modify

### **New Files** (7):
1. `frontend/src/components/globe/CountryModal.tsx` - Country detail modal
2. `frontend/src/components/globe/GlobeSearch.tsx` - Search component
3. `frontend/src/components/globe/AssetDetailModal.tsx` - Asset modal
4. `database/seeds/major_cities_seed.sql` - 1000+ cities
5. `database/migrations/008_country_indicators.sql` - Economic data table
6. `cift/api/routes/country.py` - Country endpoints
7. `frontend/src/hooks/useGlobeSearch.ts` - Search logic

### **Modified Files** (8):
1. `EnhancedFinancialGlobe.tsx` - All features integration
2. `GlobeFilterPanel.tsx` - Enhanced filters
3. `globe.py` - Add country endpoint
4. `ExchangeDetailModal.module.css` - Reduce size
5. `useGlobeData.ts` - Add country data
6. `update_asset_status.py` - Already enhanced
7. `update_ship_positions.py` - Already created
8. `cift/core/database.py` - Add city lookup functions

---

## 🎯 Priority Order

### **Phase 1: Fix Visibility** (URGENT)
1. ✅ Seed database with African assets (23)
2. ✅ Seed database with ships (16)
3. ✅ Verify backend endpoints return data
4. ✅ Test frontend actually shows data
5. ✅ Fix globe cutoff

### **Phase 2: Political Boundaries** (HIGH)
1. ✅ Use proper GeoJSON for ALL 195 countries
2. ✅ Add raycasting for country clicks
3. ✅ Create country modal with GDP/inflation/news
4. ✅ Add country detail endpoint

### **Phase 3: Search & Zoom** (HIGH)
1. ✅ Add city database (1000+ cities)
2. ✅ Create search component with Fuse.js
3. ✅ Implement zoom-to-city function
4. ✅ Wire up search results to zoom

### **Phase 4: Polish** (MEDIUM)
1. ✅ Reduce modal sizes
2. ✅ Add comprehensive filters
3. ✅ Add ship rendering to globe
4. ✅ Performance optimization

---

## 📊 Success Criteria

- [ ] Can see all 63 assets on globe (40 original + 23 African)
- [ ] Can see 16 ships moving on globe
- [ ] Can click any country and see modal with GDP/inflation/news
- [ ] Can search for "Lagos" and zoom to Nigeria
- [ ] Political boundaries show ALL countries (195), not just 20
- [ ] Africa shows 54 individual countries, not one blob
- [ ] Globe fully visible (not cut at bottom)
- [ ] Modals are appropriately sized
- [ ] Can filter by regions (Africa, Asia, etc.)
- [ ] All data comes from database (no mock data)

---

Next: Start implementing Phase 1 (Fix Visibility) with actual database verification!
