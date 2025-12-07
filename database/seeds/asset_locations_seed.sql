-- Asset Locations Seed Data
-- Real coordinates and data for major market-moving locations
-- Date: 2025-11-17

-- Central Banks (8 major ones)
INSERT INTO asset_locations (code, name, asset_type, country, country_code, city, lat, lng, timezone, description, importance_score, website, icon_url) VALUES
('FED', 'Federal Reserve', 'central_bank', 'United States', 'US', 'Washington DC', 38.893220, -77.044500, 'America/New_York', 'Central bank of the United States, controls monetary policy and interest rates', 100, 'https://www.federalreserve.gov', 'https://placehold.co/60x60/003d7a/ffffff?text=FED'),
('ECB', 'European Central Bank', 'central_bank', 'Germany', 'DE', 'Frankfurt', 50.110922, 8.682127, 'Europe/Berlin', 'Central bank for the euro and manages monetary policy for the eurozone', 95, 'https://www.ecb.europa.eu', 'https://placehold.co/60x60/003399/ffffff?text=ECB'),
('BOJ', 'Bank of Japan', 'central_bank', 'Japan', 'JP', 'Tokyo', 35.676248, 139.750347, 'Asia/Tokyo', 'Central bank of Japan, implements monetary policy and issues yen', 90, 'https://www.boj.or.jp/en', 'https://placehold.co/60x60/bc002d/ffffff?text=BOJ'),
('BOE', 'Bank of England', 'central_bank', 'United Kingdom', 'GB', 'London', 51.514167, -0.088889, 'Europe/London', 'Central bank of the United Kingdom, maintains monetary and financial stability', 90, 'https://www.bankofengland.co.uk', 'https://placehold.co/60x60/c8102e/ffffff?text=BOE'),
('PBOC', 'People''s Bank of China', 'central_bank', 'China', 'CN', 'Beijing', 39.916668, 116.397498, 'Asia/Shanghai', 'Central bank of China, formulates and implements monetary policy', 95, 'http://www.pbc.gov.cn/english', 'https://placehold.co/60x60/de2910/ffffff?text=PBC'),
('SNB', 'Swiss National Bank', 'central_bank', 'Switzerland', 'CH', 'Zurich', 47.373878, 8.545094, 'Europe/Zurich', 'Central bank of Switzerland, conducts monetary policy and ensures price stability', 85, 'https://www.snb.ch/en', 'https://placehold.co/60x60/ff0000/ffffff?text=SNB'),
('BOC', 'Bank of Canada', 'central_bank', 'Canada', 'CA', 'Ottawa', 45.421530, -75.697193, 'America/Toronto', 'Central bank of Canada, promotes economic and financial welfare', 80, 'https://www.bankofcanada.ca', 'https://placehold.co/60x60/ff0000/ffffff?text=BOC'),
('RBA', 'Reserve Bank of Australia', 'central_bank', 'Australia', 'AU', 'Sydney', -33.868820, 151.209296, 'Australia/Sydney', 'Central bank of Australia, conducts monetary policy', 75, 'https://www.rba.gov.au', 'https://placehold.co/60x60/00008b/ffffff?text=RBA'),

-- Commodities Markets (10 major trading centers)
('COMEX', 'COMEX - Metals Division', 'commodity_market', 'United States', 'US', 'New York', 40.713050, -74.013220, 'America/New_York', 'World''s primary market for trading metals futures (gold, silver, copper)', 90, 'https://www.cmegroup.com', 'https://placehold.co/60x60/ffd700/000000?text=CMX'),
('NYMEX', 'NYMEX - Energy Division', 'commodity_market', 'United States', 'US', 'New York', 40.713051, -74.013221, 'America/New_York', 'Leading energy futures exchange (crude oil, natural gas, gasoline)', 95, 'https://www.cmegroup.com/markets/energy.html', 'https://placehold.co/60x60/000000/ffd700?text=NYM'),
('ICE_BRENT', 'ICE Futures - Brent Crude', 'commodity_market', 'United Kingdom', 'GB', 'London', 51.515556, -0.088611, 'Europe/London', 'Global benchmark for Brent crude oil futures', 90, 'https://www.theice.com', 'https://placehold.co/60x60/1e3a8a/ffffff?text=ICE'),
('LME', 'London Metal Exchange', 'commodity_market', 'United Kingdom', 'GB', 'London', 51.514583, -0.087778, 'Europe/London', 'World''s largest market for industrial metals trading', 85, 'https://www.lme.com', 'https://placehold.co/60x60/006400/ffffff?text=LME'),
('CBOT', 'Chicago Board of Trade', 'commodity_market', 'United States', 'US', 'Chicago', 41.878114, -87.629798, 'America/Chicago', 'Leading futures exchange for grains and US Treasury bonds', 85, 'https://www.cmegroup.com/markets/agriculture.html', 'https://placehold.co/60x60/ff6347/ffffff?text=CBT'),
('DCE', 'Dalian Commodity Exchange', 'commodity_market', 'China', 'CN', 'Dalian', 38.914003, 121.614682, 'Asia/Shanghai', 'Major commodity futures exchange for agricultural products and iron ore', 75, 'http://www.dce.com.cn/DCE/index.html', 'https://placehold.co/60x60/de2910/ffffff?text=DCE'),
('SHFE', 'Shanghai Futures Exchange', 'commodity_market', 'China', 'CN', 'Shanghai', 31.230391, 121.473701, 'Asia/Shanghai', 'Leading futures exchange for metals, energy, and raw materials in China', 80, 'http://www.shfe.com.cn/bourseService', 'https://placehold.co/60x60/de2910/ffffff?text=SHF'),
('MCX', 'Multi Commodity Exchange', 'commodity_market', 'India', 'IN', 'Mumbai', 19.075984, 72.877656, 'Asia/Kolkata', 'India''s leading commodity derivatives exchange', 70, 'https://www.mcxindia.com', 'https://placehold.co/60x60/ff9933/ffffff?text=MCX'),
('TOCOM', 'Tokyo Commodity Exchange', 'commodity_market', 'Japan', 'JP', 'Tokyo', 35.689487, 139.691706, 'Asia/Tokyo', 'Commodity futures exchange for precious metals and energy', 70, 'https://www.tocom.or.jp/english', 'https://placehold.co/60x60/bc002d/ffffff?text=TCM'),
('SGX_COM', 'SGX Commodities', 'commodity_market', 'Singapore', 'SG', 'Singapore', 1.278676, 103.850971, 'Asia/Singapore', 'Asia''s hub for iron ore, rubber, and freight derivatives', 65, 'https://www.sgx.com/products/commodities', 'https://placehold.co/60x60/ed2939/ffffff?text=SGC'),

-- Government & Financial Institutions (8 key locations)
('USTREAS', 'US Department of Treasury', 'government', 'United States', 'US', 'Washington DC', 38.895850, -77.032200, 'America/New_York', 'Manages US government finances, issues bonds, enforces financial laws', 95, 'https://home.treasury.gov', 'https://placehold.co/60x60/003d7a/ffffff?text=TRS'),
('IMF_HQ', 'International Monetary Fund', 'government', 'United States', 'US', 'Washington DC', 38.899750, -77.043056, 'America/New_York', 'Promotes global monetary cooperation and financial stability', 90, 'https://www.imf.org', 'https://placehold.co/60x60/0072c6/ffffff?text=IMF'),
('WB_HQ', 'World Bank', 'government', 'United States', 'US', 'Washington DC', 38.899167, -77.043056, 'America/New_York', 'Provides financing and knowledge for developing countries', 85, 'https://www.worldbank.org', 'https://placehold.co/60x60/009fdb/ffffff?text=WBK'),
('BIS', 'Bank for International Settlements', 'government', 'Switzerland', 'CH', 'Basel', 47.558861, 7.583056, 'Europe/Zurich', 'International financial institution owned by central banks', 85, 'https://www.bis.org', 'https://placehold.co/60x60/003d7a/ffffff?text=BIS'),
('SEC', 'US Securities and Exchange Commission', 'government', 'United States', 'US', 'Washington DC', 38.893849, -77.021164, 'America/New_York', 'Protects investors and maintains fair securities markets', 80, 'https://www.sec.gov', 'https://placehold.co/60x60/003d7a/ffffff?text=SEC'),
('FCA', 'Financial Conduct Authority', 'government', 'United Kingdom', 'GB', 'London', 51.519444, -0.113889, 'Europe/London', 'UK financial services regulator', 75, 'https://www.fca.org.uk', 'https://placehold.co/60x60/c8102e/ffffff?text=FCA'),
('ESMA', 'European Securities Markets Authority', 'government', 'France', 'FR', 'Paris', 48.856614, 2.352222, 'Europe/Paris', 'EU securities markets regulator', 70, 'https://www.esma.europa.eu', 'https://placehold.co/60x60/0055a4/ffffff?text=ESM'),
('NDRC', 'National Development & Reform Commission', 'government', 'China', 'CN', 'Beijing', 39.904200, 116.407396, 'Asia/Shanghai', 'China''s macroeconomic planning agency', 75, 'https://en.ndrc.gov.cn', 'https://placehold.co/60x60/de2910/ffffff?text=NDR'),

-- Tech Company Headquarters (7 major market movers)
('AAPL_HQ', 'Apple Inc. HQ', 'tech_hq', 'United States', 'US', 'Cupertino', 37.334900, -122.009020, 'America/Los_Angeles', 'World''s largest tech company, consumer electronics and services', 100, 'https://www.apple.com', 'https://placehold.co/60x60/000000/ffffff?text=APL'),
('MSFT_HQ', 'Microsoft Corporation HQ', 'tech_hq', 'United States', 'US', 'Redmond', 47.643920, -122.128570, 'America/Los_Angeles', 'Cloud computing, software, and AI leader', 95, 'https://www.microsoft.com', 'https://placehold.co/60x60/00a4ef/ffffff?text=MSF'),
('GOOGL_HQ', 'Google/Alphabet HQ', 'tech_hq', 'United States', 'US', 'Mountain View', 37.422000, -122.084058, 'America/Los_Angeles', 'Search, advertising, cloud, and AI technology', 95, 'https://www.google.com', 'https://placehold.co/60x60/4285f4/ffffff?text=GOG'),
('AMZN_HQ', 'Amazon.com HQ', 'tech_hq', 'United States', 'US', 'Seattle', 47.615450, -122.338320, 'America/Los_Angeles', 'E-commerce and cloud computing giant (AWS)', 95, 'https://www.amazon.com', 'https://placehold.co/60x60/ff9900/000000?text=AMZ'),
('TSLA_HQ', 'Tesla Inc. HQ', 'tech_hq', 'United States', 'US', 'Austin', 30.267153, -97.743061, 'America/Chicago', 'Electric vehicles and clean energy solutions', 90, 'https://www.tesla.com', 'https://placehold.co/60x60/cc0000/ffffff?text=TSL'),
('META_HQ', 'Meta Platforms HQ', 'tech_hq', 'United States', 'US', 'Menlo Park', 37.485215, -122.148285, 'America/Los_Angeles', 'Social media platforms and VR/AR technology', 85, 'https://www.meta.com', 'https://placehold.co/60x60/0668e1/ffffff?text=MTA'),
('NVDA_HQ', 'NVIDIA Corporation HQ', 'tech_hq', 'United States', 'US', 'Santa Clara', 37.370810, -121.963810, 'America/Los_Angeles', 'GPU and AI chip manufacturer', 95, 'https://www.nvidia.com', 'https://placehold.co/60x60/76b900/000000?text=NVD'),

-- Energy Infrastructure (7 critical locations)
('OPEC_HQ', 'OPEC Headquarters', 'energy', 'Austria', 'AT', 'Vienna', 48.208176, 16.373819, 'Europe/Vienna', 'Organization of oil exporting countries, controls global oil supply', 90, 'https://www.opec.org', 'https://placehold.co/60x60/000000/ffd700?text=OPC'),
('SPR_US', 'US Strategic Petroleum Reserve', 'energy', 'United States', 'US', 'Houston', 29.760427, -95.369803, 'America/Chicago', 'Emergency crude oil stockpile for supply disruptions', 85, 'https://www.energy.gov/spr', 'https://placehold.co/60x60/003d7a/ffffff?text=SPR'),
('GHAWAR', 'Ghawar Oil Field', 'energy', 'Saudi Arabia', 'SA', 'Al-Ahsa', 25.491667, 49.593889, 'Asia/Riyadh', 'World''s largest conventional oil field', 90, 'https://www.aramco.com', 'https://placehold.co/60x60/006c35/ffffff?text=GHW'),
('PERMIAN', 'Permian Basin', 'energy', 'United States', 'US', 'Midland', 31.997345, -102.077915, 'America/Chicago', 'Largest US oil producing region', 85, 'https://www.eia.gov', 'https://placehold.co/60x60/003d7a/ffffff?text=PRM'),
('NSEA_BRENT', 'North Sea Brent Field', 'energy', 'United Kingdom', 'GB', 'Aberdeen', 57.149717, -2.094278, 'Europe/London', 'Major oil field, source of Brent crude benchmark', 80, 'https://www.offshore-technology.com', 'https://placehold.co/60x60/c8102e/ffffff?text=BRT'),
('YAMAL', 'Yamal LNG Project', 'energy', 'Russia', 'RU', 'Sabetta', 71.281111, 72.046944, 'Asia/Yekaterinburg', 'Major natural gas liquefaction facility', 75, 'https://yamallng.ru/en', 'https://placehold.co/60x60/0033a0/ffffff?text=YML'),
('QATARGAS', 'Qatar Gas Fields', 'energy', 'Qatar', 'QA', 'Doha', 25.286106, 51.534817, 'Asia/Qatar', 'World''s largest LNG exporter', 80, 'https://www.qatargas.com', 'https://placehold.co/60x60/8d1b3d/ffffff?text=QTG')

ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    asset_type = EXCLUDED.asset_type,
    country = EXCLUDED.country,
    country_code = EXCLUDED.country_code,
    city = EXCLUDED.city,
    lat = EXCLUDED.lat,
    lng = EXCLUDED.lng,
    timezone = EXCLUDED.timezone,
    description = EXCLUDED.description,
    importance_score = EXCLUDED.importance_score,
    website = EXCLUDED.website,
    icon_url = EXCLUDED.icon_url,
    updated_at = NOW();

-- Initialize status log for all assets (set to 'unknown' initially)
INSERT INTO asset_status_log (asset_id, status, sentiment_score, news_count, status_reason, checked_at)
SELECT 
    id,
    'unknown',
    0.0,
    0,
    'Initial status - no news data analyzed yet',
    NOW()
FROM asset_locations
ON CONFLICT DO NOTHING;

-- Show summary
SELECT 
    asset_type,
    COUNT(*) as count,
    ROUND(AVG(importance_score), 1) as avg_importance
FROM asset_locations
GROUP BY asset_type
ORDER BY COUNT(*) DESC;

SELECT COUNT(*) as total_asset_locations FROM asset_locations;
