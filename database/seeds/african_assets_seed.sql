-- Seed: African Assets
-- Purpose: Add major African market assets (central banks, oil fields, data centers, commodities)
-- Date: 2025-11-17

-- African Central Banks
INSERT INTO asset_locations (
    code, name, asset_type, country, country_code, city,
    lat, lng, timezone, description, importance_score, website
) VALUES
-- Central Banks
('SARB', 'South African Reserve Bank', 'central_bank', 'South Africa', 'ZA', 'Pretoria',
 -25.7479, 28.2293, 'Africa/Johannesburg', 
 'Central bank of South Africa, manages monetary policy for the largest African economy', 85,
 'https://www.resbank.co.za'),

('CBN', 'Central Bank of Nigeria', 'central_bank', 'Nigeria', 'NG', 'Abuja',
 9.0765, 7.3986, 'Africa/Lagos',
 'Central bank of Nigeria, Africa''s largest oil producer', 82,
 'https://www.cbn.gov.ng'),

('CBE', 'Central Bank of Egypt', 'central_bank', 'Egypt', 'EG', 'Cairo',
 30.0444, 31.2357, 'Africa/Cairo',
 'Central bank of Egypt, managing monetary policy for North Africa''s largest economy', 78,
 'https://www.cbe.org.eg'),

('CBK', 'Central Bank of Kenya', 'central_bank', 'Kenya', 'KE', 'Nairobi',
 -1.2864, 36.8172, 'Africa/Nairobi',
 'Central bank of Kenya, hub for East African financial services', 72,
 'https://www.centralbank.go.ke'),

('BOA', 'Bank of Algeria', 'central_bank', 'Algeria', 'DZ', 'Algiers',
 36.7538, 3.0588, 'Africa/Algiers',
 'Central bank of Algeria, managing Africa''s 4th largest economy', 70,
 'https://www.bank-of-algeria.dz'),

('BOG', 'Bank of Ghana', 'central_bank', 'Ghana', 'GH', 'Accra',
 5.6037, -0.187, 'Africa/Accra',
 'Central bank of Ghana, regulating one of Africa''s fastest-growing economies', 68,
 'https://www.bog.gov.gh'),

-- Oil & Energy Assets
('NDA', 'Niger Delta Oil Fields', 'energy', 'Nigeria', 'NG', 'Port Harcourt',
 4.8156, 7.0498, 'Africa/Lagos',
 'Major oil production region in Nigeria, producing 2M barrels/day', 90,
 NULL),

('ANP', 'Angola National Petroleum', 'energy', 'Angola', 'AO', 'Luanda',
 -8.8383, 13.2344, 'Africa/Luanda',
 'Angola''s state oil company, 2nd largest African oil producer', 85,
 'https://www.anp.ao'),

('EGPC', 'Egyptian General Petroleum', 'energy', 'Egypt', 'EG', 'Cairo',
 30.0444, 31.2357, 'Africa/Cairo',
 'Egypt''s state oil company managing Suez Canal oil transit', 82,
 'https://www.egpc.com.eg'),

('LNG', 'Mozambique LNG Project', 'energy', 'Mozambique', 'MZ', 'Palma',
 -10.7169, 40.3555, 'Africa/Maputo',
 'Major liquefied natural gas project worth $20B+', 88,
 NULL),

-- Commodity Markets & Mines
('JHB_GOLD', 'Johannesburg Gold Mines', 'commodity_market', 'South Africa', 'ZA', 'Johannesburg',
 -26.2041, 28.0473, 'Africa/Johannesburg',
 'World''s largest gold mining region, produces 40% of all gold ever mined', 92,
 NULL),

('COBALT_DRC', 'Democratic Republic Congo Cobalt Mines', 'commodity_market', 'DR Congo', 'CD', 'Kolwezi',
 -10.7144, 25.4662, 'Africa/Lubumbashi',
 'Produces 70% of world''s cobalt supply for EV batteries', 95,
 NULL),

('DIAMOND_BOT', 'Botswana Diamond Mines', 'commodity_market', 'Botswana', 'BW', 'Orapa',
 -21.3115, 25.3767, 'Africa/Gaborone',
 'World''s largest diamond producer by value', 80,
 NULL),

('PHOSPHATE_MOR', 'Morocco Phosphate Reserves', 'commodity_market', 'Morocco', 'MA', 'Khouribga',
 32.8811, -6.9063, 'Africa/Casablanca',
 'Holds 75% of world''s phosphate reserves for fertilizer', 85,
 NULL),

('COCOA_CIV', 'Ivory Coast Cocoa Farms', 'commodity_market', 'Ivory Coast', 'CI', 'Yamoussoukro',
 6.8270, -5.2893, 'Africa/Abidjan',
 'World''s largest cocoa producer, 40% of global supply', 75,
 NULL),

-- Tech & Data Centers
('TERACO_JHB', 'Teraco Data Centre Johannesburg', 'tech_hq', 'South Africa', 'ZA', 'Johannesburg',
 -26.1076, 28.0567, 'Africa/Johannesburg',
 'Africa''s largest data center, 13,000+ servers', 78,
 'https://www.teraco.co.za'),

('NOM_LAGOS', 'Nigeria Data Hub Lagos', 'tech_hq', 'Nigeria', 'NG', 'Lagos',
 6.5244, 3.3792, 'Africa/Lagos',
 'West Africa''s primary internet exchange point', 72,
 NULL),

('MDC_CAIRO', 'Mega Data Center Cairo', 'tech_hq', 'Egypt', 'EG', 'Cairo',
 30.0131, 31.2089, 'Africa/Cairo',
 'North Africa''s largest cloud computing facility', 70,
 NULL),

('KENET_NAI', 'Kenya Education Network Hub', 'tech_hq', 'Kenya', 'KE', 'Nairobi',
 -1.3032, 36.7073, 'Africa/Nairobi',
 'East Africa''s primary fiber optic connectivity hub', 68,
 NULL),

-- Strategic Government Assets
('SUEZ', 'Suez Canal Authority', 'government', 'Egypt', 'EG', 'Ismailia',
 30.5833, 32.2667, 'Africa/Cairo',
 'Controls 12% of global trade through canal, $8B+ annual revenue', 98,
 'https://www.suezcanal.gov.eg'),

('DURBAN_PORT', 'Port of Durban', 'government', 'South Africa', 'ZA', 'Durban',
 -29.8587, 31.0218, 'Africa/Johannesburg',
 'Busiest port in Africa, handles 60M+ tons annually', 85,
 NULL),

('LAGOS_PORT', 'Lagos Port Complex', 'government', 'Nigeria', 'NG', 'Lagos',
 6.4426, 3.3903, 'Africa/Lagos',
 'West Africa''s largest port, 70% of Nigeria''s trade', 82,
 NULL),

('TEMA_PORT', 'Port of Tema', 'government', 'Ghana', 'GH', 'Tema',
 5.6698, -0.0167, 'Africa/Accra',
 'Ghana''s main seaport, 1M+ containers annually', 70,
 NULL)

ON CONFLICT (code) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    importance_score = EXCLUDED.importance_score,
    updated_at = NOW();

-- Insert initial status logs for African assets
INSERT INTO asset_status_log (asset_id, status, sentiment_score, news_count, status_reason, checked_at)
SELECT 
    id,
    'unknown' as status,
    0.0 as sentiment_score,
    0 as news_count,
    'Initial status - awaiting news analysis' as status_reason,
    NOW() as checked_at
FROM asset_locations
WHERE code IN (
    'SARB', 'CBN', 'CBE', 'CBK', 'BOA', 'BOG',
    'NDA', 'ANP', 'EGPC', 'LNG',
    'JHB_GOLD', 'COBALT_DRC', 'DIAMOND_BOT', 'PHOSPHATE_MOR', 'COCOA_CIV',
    'TERACO_JHB', 'NOM_LAGOS', 'MDC_CAIRO', 'KENET_NAI',
    'SUEZ', 'DURBAN_PORT', 'LAGOS_PORT', 'TEMA_PORT'
);

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ African assets seeded successfully';
    RAISE NOTICE '📊 Added 23 African assets:';
    RAISE NOTICE '   🏦 6 Central Banks';
    RAISE NOTICE '   ⚡ 4 Oil & Energy';
    RAISE NOTICE '   🛢️ 5 Commodity Markets';
    RAISE NOTICE '   💻 4 Tech/Data Centers';
    RAISE NOTICE '   🏛️ 4 Strategic Ports';
END $$;
