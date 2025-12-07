# Globe Updates Implementation Plan

## Changes Required

### 1. Asset Markers - Make Clickable & Use Exchange-Style Modal ✅ (Partial)
**Status**: Click handler exists but may need debugging
- [x] Asset click detection exists (line 317-356)
- [ ] Replace current large asset modal with exchange-style small modal
- [ ] Test asset marker clicking

### 2. Capital City Markers (NEW FEATURE)
**Status**: In Progress
- [x] Created `countryCapitals.ts` with 60+ capitals
- [x] Added `CapitalMarkerData` interface
- [x] Added `capitalMarkerGroup` to scene
- [x] Added state signals for capitals
- [ ] Create `updateCapitalMarkers()` function
- [ ] Add capitals to hover detection
- [ ] Add capitals to click detection
- [ ] Create capital modal (small, exchange-style)
- [ ] Show: Country, Capital, GDP, Inflation, Top News, "More News" button

### 3. Remove Country Boundary Clicking
- [ ] Remove boundary click detection (lines 357-368)
- [ ] Keep boundaries visible but not clickable
- [ ] Remove `fetchCountryDetails()` function (or repurpose for capitals)

### 4. Ship Markers - Direction Arrows & Color Coding
**Current**: Ships render as geometric shapes
**Required**: 
- [ ] Replace with directional arrows/pointers
- [ ] Rotate arrow to show `current_course` direction
- [ ] Color coding:
  - Green: No issues
  - Grey: Status unknown
  - Red: Major issues reported
- [ ] Integrate AIS Stream API: `80280c3f82df9c48b7aedbfa46ee7e5445ce3751`

### 5. AIS Stream Integration
- [ ] Create WebSocket connection to AIS Stream
- [ ] Replace static ship data with live AIS data
- [ ] Update ship positions in real-time
- [ ] Parse AIS messages for:
  - Position (lat/lng)
  - Course (direction)
  - Speed
  - Status/issues

## Implementation Order

1. **Capital Markers** (High Priority)
   - Create rendering function
   - Add to hover/click handlers
   - Create capital modal

2. **Remove Boundary Clicking** (Quick Fix)
   - Comment out boundary click handler
   - Keep boundaries visual only

3. **Fix Asset Modal** (Medium Priority)
   - Replace with exchange-style modal
   - Make smaller and cleaner

4. **Ship Direction Arrows** (Complex)
   - Create arrow geometry
   - Add rotation based on course
   - Implement color logic

5. **AIS Stream API** (Advanced)
   - Set up WebSocket
   - Parse AIS data
   - Real-time updates

## File Structure

```
frontend/src/
├── components/globe/
│   ├── EnhancedFinancialGlobe.tsx (main file - needs updates)
│   └── CountryModal.tsx (exists - repurpose for capitals)
├── data/
│   └── countryCapitals.ts (✅ created)
└── hooks/
    └── useAISStream.ts (needs creation)
```

## Next Steps

1. Create `updateCapitalMarkers()` function
2. Add capitals to click/hover handlers  
3. Create capital modal UI
4. Remove boundary clicking
5. Update ship markers
6. Integrate AIS Stream
