-- Stock Exchange Seed Data
-- Real coordinates and data for major global stock exchanges

INSERT INTO stock_exchanges (code, name, country, country_code, lat, lng, timezone, market_cap_usd, website, icon_url) VALUES
-- Americas
('NYSE', 'New York Stock Exchange', 'United States', 'US', 40.706877, -74.011326, 'America/New_York', 31900000000000, 'https://www.nyse.com', 'https://placehold.co/60x60/0078d4/ffffff?text=NYSE'),
('NASDAQ', 'NASDAQ', 'United States', 'US', 40.760779, -73.977638, 'America/New_York', 22400000000000, 'https://www.nasdaq.com', 'https://placehold.co/60x60/0078d4/ffffff?text=NDQ'),
('TSX', 'Toronto Stock Exchange', 'Canada', 'CA', 43.648270, -79.381222, 'America/Toronto', 2900000000000, 'https://www.tsx.com', 'https://placehold.co/60x60/ff0000/ffffff?text=TSX'),
('B3', 'B3 - Brasil Bolsa Balcão', 'Brazil', 'BR', -23.546287, -46.633835, 'America/Sao_Paulo', 1300000000000, 'https://www.b3.com.br', 'https://placehold.co/60x60/009c3b/ffffff?text=B3'),
('BMV', 'Bolsa Mexicana de Valores', 'Mexico', 'MX', 19.426942, -99.131584, 'America/Mexico_City', 530000000000, 'https://www.bmv.com.mx', 'https://placehold.co/60x60/ce1126/ffffff?text=BMV'),

-- Europe
('LSE', 'London Stock Exchange', 'United Kingdom', 'GB', 51.515278, -0.099167, 'Europe/London', 3800000000000, 'https://www.londonstockexchange.com', 'https://placehold.co/60x60/0078d4/ffffff?text=LSE'),
('ENX', 'Euronext', 'France', 'FR', 48.856614, 2.352222, 'Europe/Paris', 4000000000000, 'https://www.euronext.com', 'https://placehold.co/60x60/0055a4/ffffff?text=ENX'),
('DB', 'Deutsche Börse', 'Germany', 'DE', 50.110922, 8.682127, 'Europe/Berlin', 2100000000000, 'https://deutsche-boerse.com', 'https://placehold.co/60x60/000000/ffcc00?text=DB'),
('SIX', 'SIX Swiss Exchange', 'Switzerland', 'CH', 47.373878, 8.545094, 'Europe/Zurich', 1900000000000, 'https://www.six-group.com', 'https://placehold.co/60x60/ff0000/ffffff?text=SIX'),
('BME', 'Bolsas y Mercados Españoles', 'Spain', 'ES', 40.416775, -3.703790, 'Europe/Madrid', 970000000000, 'https://www.bolsasymercados.es', 'https://placehold.co/60x60/c60b1e/ffffff?text=BME'),

-- Asia-Pacific
('SSE', 'Shanghai Stock Exchange', 'China', 'CN', 31.236750, 121.508750, 'Asia/Shanghai', 6800000000000, 'http://english.sse.com.cn', 'https://placehold.co/60x60/de2910/ffffff?text=SSE'),
('SZSE', 'Shenzhen Stock Exchange', 'China', 'CN', 22.543099, 114.057939, 'Asia/Shanghai', 3700000000000, 'http://www.szse.cn/English', 'https://placehold.co/60x60/de2910/ffffff?text=SZSE'),
('TSE', 'Tokyo Stock Exchange', 'Japan', 'JP', 35.676248, 139.650347, 'Asia/Tokyo', 5600000000000, 'https://www.jpx.co.jp/english', 'https://placehold.co/60x60/bc002d/ffffff?text=TSE'),
('HKEX', 'Hong Kong Stock Exchange', 'Hong Kong', 'HK', 22.278326, 114.174695, 'Asia/Hong_Kong', 4200000000000, 'https://www.hkex.com.hk', 'https://placehold.co/60x60/de2910/ffffff?text=HKX'),
('BSE', 'Bombay Stock Exchange', 'India', 'IN', 18.929400, 72.833302, 'Asia/Kolkata', 3400000000000, 'https://www.bseindia.com', 'https://placehold.co/60x60/ff9933/ffffff?text=BSE'),
('NSE', 'National Stock Exchange of India', 'India', 'IN', 19.065148, 72.832222, 'Asia/Kolkata', 3300000000000, 'https://www.nseindia.com', 'https://placehold.co/60x60/ff9933/ffffff?text=NSE'),
('KRX', 'Korea Exchange', 'South Korea', 'KR', 37.565188, 126.977041, 'Asia/Seoul', 2100000000000, 'http://global.krx.co.kr', 'https://placehold.co/60x60/003478/ffffff?text=KRX'),
('ASX', 'Australian Securities Exchange', 'Australia', 'AU', -33.868820, 151.209296, 'Australia/Sydney', 1600000000000, 'https://www.asx.com.au', 'https://placehold.co/60x60/00008b/ffffff?text=ASX'),
('SGX', 'Singapore Exchange', 'Singapore', 'SG', 1.352083, 103.819836, 'Asia/Singapore', 700000000000, 'https://www.sgx.com', 'https://placehold.co/60x60/ed2939/ffffff?text=SGX'),
('TWSE', 'Taiwan Stock Exchange', 'Taiwan', 'TW', 25.033671, 121.565418, 'Asia/Taipei', 1800000000000, 'https://www.twse.com.tw', 'https://placehold.co/60x60/000095/ffffff?text=TWS'),

-- Middle East & Africa
('TADAWUL', 'Saudi Stock Exchange', 'Saudi Arabia', 'SA', 24.713552, 46.675296, 'Asia/Riyadh', 2800000000000, 'https://www.saudiexchange.sa', 'https://placehold.co/60x60/006c35/ffffff?text=TDW'),
('DFM', 'Dubai Financial Market', 'UAE', 'AE', 25.204849, 55.270783, 'Asia/Dubai', 160000000000, 'https://www.dfm.ae', 'https://placehold.co/60x60/00732f/ffffff?text=DFM'),
('JSE', 'Johannesburg Stock Exchange', 'South Africa', 'ZA', -26.204103, 28.047305, 'Africa/Johannesburg', 1200000000000, 'https://www.jse.co.za', 'https://placehold.co/60x60/007a4d/ffffff?text=JSE'),
('NSE_KE', 'Nairobi Securities Exchange', 'Kenya', 'KE', -1.286389, 36.817223, 'Africa/Nairobi', 24000000000, 'https://www.nse.co.ke', 'https://placehold.co/60x60/006600/ffffff?text=NSE'),
('EGX', 'Egyptian Exchange', 'Egypt', 'EG', 30.044420, 31.235712, 'Africa/Cairo', 35000000000, 'https://www.egx.com.eg', 'https://placehold.co/60x60/ce1126/ffffff?text=EGX'),
('NSE_NG', 'Nigerian Exchange Group', 'Nigeria', 'NG', 6.445240, 3.420326, 'Africa/Lagos', 60000000000, 'https://www.ngxgroup.com', 'https://placehold.co/60x60/008751/ffffff?text=NGX'),
('CSE_MA', 'Casablanca Stock Exchange', 'Morocco', 'MA', 33.589886, -7.603869, 'Africa/Casablanca', 60000000000, 'https://www.casablanca-bourse.com', 'https://placehold.co/60x60/c1272d/ffffff?text=CSE'),
('BVMT', 'Bourse de Tunis', 'Tunisia', 'TN', 36.819077, 10.165899, 'Africa/Tunis', 9000000000, 'https://www.bvmt.com.tn', 'https://placehold.co/60x60/e70013/ffffff?text=BVT'),
('BSE_BW', 'Botswana Stock Exchange', 'Botswana', 'BW', -24.653257, 25.906792, 'Africa/Gaborone', 5000000000, 'https://www.bse.co.bw', 'https://placehold.co/60x60/75aadb/ffffff?text=BSE'),

-- Additional Europe
('FTSE_IT', 'Borsa Italiana', 'Italy', 'IT', 45.464664, 9.191383, 'Europe/Rome', 760000000000, 'https://www.borsaitaliana.it', 'https://placehold.co/60x60/009246/ffffff?text=BIT'),
('AEX', 'Euronext Amsterdam', 'Netherlands', 'NL', 52.370216, 4.895168, 'Europe/Amsterdam', 1200000000000, 'https://www.euronext.com', 'https://placehold.co/60x60/ae1c28/ffffff?text=AEX'),
('OMX', 'Nasdaq Stockholm', 'Sweden', 'SE', 59.329323, 18.068581, 'Europe/Stockholm', 850000000000, 'https://www.nasdaqomxnordic.com', 'https://placehold.co/60x60/006aa7/ffffff?text=OMX'),
('OSE', 'Oslo Børs', 'Norway', 'NO', 59.913869, 10.752245, 'Europe/Oslo', 300000000000, 'https://www.oslobors.no', 'https://placehold.co/60x60/ef2b2d/ffffff?text=OSE'),
('MOEX', 'Moscow Exchange', 'Russia', 'RU', 55.755826, 37.617300, 'Europe/Moscow', 600000000000, 'https://www.moex.com', 'https://placehold.co/60x60/0033a0/ffffff?text=MOX'),

-- Additional Asia
('SET', 'Stock Exchange of Thailand', 'Thailand', 'TH', 13.756331, 100.501765, 'Asia/Bangkok', 510000000000, 'https://www.set.or.th', 'https://placehold.co/60x60/a51931/ffffff?text=SET'),
('KLSE', 'Bursa Malaysia', 'Malaysia', 'MY', 3.139003, 101.686855, 'Asia/Kuala_Lumpur', 400000000000, 'https://www.bursamalaysia.com', 'https://placehold.co/60x60/010066/ffffff?text=KLS'),
('IDX', 'Indonesia Stock Exchange', 'Indonesia', 'ID', -6.208763, 106.845599, 'Asia/Jakarta', 530000000000, 'https://www.idx.co.id', 'https://placehold.co/60x60/ff0000/ffffff?text=IDX'),
('PSE', 'Philippine Stock Exchange', 'Philippines', 'PH', 14.599512, 120.984222, 'Asia/Manila', 280000000000, 'https://www.pse.com.ph', 'https://placehold.co/60x60/0038a8/ffffff?text=PSE'),

-- Additional Americas
('BCBA', 'Buenos Aires Stock Exchange', 'Argentina', 'AR', -34.603722, -58.381592, 'America/Argentina/Buenos_Aires', 80000000000, 'https://www.bcba.sba.com.ar', 'https://placehold.co/60x60/74acdf/ffffff?text=BCB'),
('BCS', 'Santiago Stock Exchange', 'Chile', 'CL', -33.436893, -70.650391, 'America/Santiago', 220000000000, 'https://www.bolsadesantiago.com', 'https://placehold.co/60x60/0039a6/ffffff?text=BCS')

ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    country = EXCLUDED.country,
    country_code = EXCLUDED.country_code,
    lat = EXCLUDED.lat,
    lng = EXCLUDED.lng,
    timezone = EXCLUDED.timezone,
    market_cap_usd = EXCLUDED.market_cap_usd,
    website = EXCLUDED.website,
    icon_url = EXCLUDED.icon_url,
    updated_at = NOW();

-- Show summary
SELECT 
    country,
    COUNT(*) as exchange_count,
    SUM(market_cap_usd) / 1000000000000.0 as total_market_cap_trillion_usd
FROM stock_exchanges
GROUP BY country
ORDER BY total_market_cap_trillion_usd DESC;

SELECT COUNT(*) as total_exchanges FROM stock_exchanges;
