# 🧪 Testing Guide - Phase 9 Features

## What You Should See

### ✅ **Political Boundaries** (Fixed - Now Real Maps!)
- **Before**: Blue circles ❌
- **Now**: Actual country border outlines ✅
- **Colors**: 
  - 🟢 Green = Positive news sentiment
  - 🔴 Red = Negative news sentiment  
  - 🔵 Blue = Neutral sentiment
- **Source**: Real GeoJSON data from world.geo.json

### ✅ **African Assets** (23 New)
- Central Banks: SARB, CBN, CBE, CBK, BOA, BOG
- Oil/Energy: Niger Delta, Angola Oil, Egypt Oil, Mozambique LNG
- Commodities: Johannesburg Gold, Congo Cobalt, Botswana Diamonds
- Tech: Teraco JHB, Lagos Hub, Cairo Center, Kenya Hub
- Ports: Suez Canal, Durban, Lagos, Tema

### ✅ **Enhanced Status System**
- Analyzes news for keywords: "shutdown", "disruption", "outage"
- Updates colors: Green (working), Red (issues), Grey (unknown)

---

## 🚀 Quick Test Steps

### 1. **Start Services** (if not running)
```powershell
cd c:\Users\mesof\cift-markets
docker-compose up -d
```

### 2. **Check Database**
```powershell
# Check if assets table exists
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -c "\dt asset_locations"

# Count current assets
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM asset_locations WHERE is_active = true;"
```

Expected: Should show 40 assets (or 63 if African assets are seeded)

### 3. **Seed African Assets** (if not done)
```powershell
# Copy seed file to container
docker cp database/seeds/african_assets_seed.sql cift-markets_postgres_1:/tmp/

# Run seed
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -f /tmp/african_assets_seed.sql
```

### 4. **Test Backend APIs**
```powershell
# Test boundaries endpoint
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/boundaries?timeframe=24h" | Select-Object -ExpandProperty Content | ConvertFrom-Json | Select-Object -ExpandProperty countries | Measure-Object

# Test assets endpoint
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/assets/" | Select-Object -ExpandProperty Content | ConvertFrom-Json | Select-Object total_count

# Test if African assets are there
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/assets/" | Select-Object -ExpandProperty Content | ConvertFrom-Json | Select-Object -ExpandProperty assets | Where-Object { $_.country_code -in @('ZA', 'NG', 'EG', 'KE') } | Select-Object name, country
```

### 5. **Start Frontend**
```powershell
cd frontend
npm run dev
```

Navigate to: **http://localhost:3000/news** → Click **Globe** view button

---

## 👀 What to Look For

### **Political Boundaries**
1. Open Globe view
2. Click Filter Panel (top-right) → Expand
3. Toggle "Boundaries" ON
4. **You should see**: Actual country outlines (not circles!)
   - US, China, Europe, etc. with real borders
   - Colored by sentiment (green/red/blue)
   - Subtle transparent fill

### **African Assets**
1. Keep Globe view open
2. Look at Africa region:
   - **South Africa**: Cubes (Johannesburg area)
   - **Nigeria**: Cones near Lagos/Port Harcourt
   - **Egypt**: Pyramids near Cairo/Suez
   - **Kenya**: Markers near Nairobi
3. Hover over markers → Should show tooltips
4. Click markers → Should open detail modals

### **Enhanced Status**
1. In Filter Panel → Expand "Asset Status"
2. Should see counts:
   - 🟢 Operational: X
   - ⚪ Unknown: Y
   - 🔴 Issues: Z
3. Toggle filters → Markers appear/disappear

---

## 🐛 Troubleshooting

### Issue: "No boundaries showing"
**Fix:**
```powershell
# Check if boundary data exists
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/boundaries?timeframe=24h" | Select-Object -ExpandProperty Content

# If empty, check if news data exists
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM news_articles;"
```

### Issue: "Only 40 assets, not 63"
**Fix:**
```powershell
# Re-run African assets seed
cd c:\Users\mesof\cift-markets
docker cp database/seeds/african_assets_seed.sql cift-markets_postgres_1:/tmp/
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -f /tmp/african_assets_seed.sql

# Verify
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM asset_locations WHERE country_code IN ('ZA', 'NG', 'EG', 'KE', 'GH', 'DZ');"
```
Expected: Should show 23

### Issue: "Frontend not loading boundaries"
**Check browser console:**
1. Open DevTools (F12)
2. Look for: `🗺️ Creating boundaries for X countries...`
3. Should see: `📥 Loaded GeoJSON data with 177 countries`
4. Should see: `✅ Rendered X country boundaries with real GeoJSON data`

**If error fetching GeoJSON:**
- It will fallback to simplified outlines
- Check network tab for failed requests

### Issue: "Asset colors all grey"
**Fix:**
```powershell
# Run enhanced status update
cd c:\Users\mesof\cift-markets
python scripts/update_asset_status.py
```

This will analyze news and update colors to green/red based on keywords.

---

## 📊 Expected Results

**Total Markers**: 103
- 40 Stock Exchanges ✅
- 63 Assets (40 original + 23 African) ✅
- Political boundaries for ~15-20 countries ✅

**Interactive Features**:
- ✅ Hover on asset → Tooltip with name, status, importance
- ✅ Click asset → Modal with details, news, sentiment
- ✅ Filter panel → Toggle asset types, status, boundaries
- ✅ Boundaries show real country shapes (not circles!)

---

## 🎯 Visual Verification Checklist

- [ ] Can see actual country border outlines (not circles)
- [ ] Boundaries are colored (green/red/blue)
- [ ] Can see markers in Africa (South Africa, Nigeria, Egypt, Kenya)
- [ ] African markers have different shapes (cubes, cones, pyramids)
- [ ] Hovering shows asset details in tooltip
- [ ] Clicking opens detailed modal
- [ ] Filter panel shows correct counts
- [ ] Toggling filters updates display immediately
- [ ] Status colors are working (not all grey)

---

## 🔄 If Nothing Shows

**Full Reset:**
```powershell
# 1. Stop services
docker-compose down

# 2. Start fresh
docker-compose up -d

# 3. Wait for services (30 seconds)
Start-Sleep -Seconds 30

# 4. Re-seed everything
docker cp database/seeds/african_assets_seed.sql cift-markets_postgres_1:/tmp/
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -f /tmp/african_assets_seed.sql

# 5. Update statuses
python scripts/update_asset_status.py

# 6. Restart frontend
cd frontend
npm run dev
```

---

## 📞 Quick Verification Commands

```powershell
# Count assets by country
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -c "SELECT country_code, country, COUNT(*) FROM asset_locations WHERE is_active = true GROUP BY country_code, country ORDER BY COUNT(*) DESC;"

# Check African assets specifically
docker exec cift-markets_postgres_1 psql -U cift_user -d cift_markets -c "SELECT code, name, asset_type, country FROM asset_locations WHERE country_code IN ('ZA', 'NG', 'EG', 'KE', 'GH', 'DZ', 'MA', 'CI', 'BW', 'AO', 'MZ', 'CD') ORDER BY country;"

# Check if boundaries API returns data
Invoke-WebRequest -Uri "http://localhost:8000/api/v1/globe/boundaries?timeframe=24h" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json | Select-Object -ExpandProperty countries | Format-Table country_code, article_count, sentiment_score
```

---

Ready to test! Start with Step 2 (Check Database) to verify current state.
