-- Expand symbols table with more major stocks
-- Add S&P 500 companies and major ETFs

-- Additional Technology stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('AVGO', 'Broadcom Inc.', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 750000000000, 35.5, 38.50, 0.0125, true, true),
('ORCL', 'Oracle Corporation', 'Technology', 'Software', 'US', 'NYSE', 'stock', 380000000000, 38.2, 4.75, 0.0095, true, true),
('CRM', 'Salesforce Inc.', 'Technology', 'Software', 'US', 'NYSE', 'stock', 280000000000, 62.4, 4.85, 0.0055, true, true),
('ADBE', 'Adobe Inc.', 'Technology', 'Software', 'US', 'NASDAQ', 'stock', 250000000000, 48.5, 11.20, 0.0, true, true),
('AMD', 'Advanced Micro Devices', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 220000000000, 145.2, 0.95, 0.0, true, true),
('INTC', 'Intel Corporation', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 85000000000, 15.8, 1.35, 0.0285, true, true),
('IBM', 'International Business Machines', 'Technology', 'IT Services', 'US', 'NYSE', 'stock', 195000000000, 24.5, 8.95, 0.0325, true, true),
('CSCO', 'Cisco Systems Inc.', 'Technology', 'Networking', 'US', 'NASDAQ', 'stock', 230000000000, 17.8, 3.25, 0.0265, true, true),
('QCOM', 'Qualcomm Inc.', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 175000000000, 18.2, 8.85, 0.0175, true, true),
('TXN', 'Texas Instruments', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 180000000000, 35.6, 5.55, 0.0265, true, true),
('MU', 'Micron Technology', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 95000000000, 22.5, 4.25, 0.004, true, true),
('NOW', 'ServiceNow Inc.', 'Technology', 'Software', 'US', 'NYSE', 'stock', 175000000000, 75.2, 12.15, 0.0, true, true),
('INTU', 'Intuit Inc.', 'Technology', 'Software', 'US', 'NASDAQ', 'stock', 185000000000, 58.5, 11.25, 0.006, true, true),
('AMAT', 'Applied Materials', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 155000000000, 22.8, 8.15, 0.0075, true, true),
('LRCX', 'Lam Research', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 95000000000, 24.5, 32.50, 0.0105, true, true),
('PANW', 'Palo Alto Networks', 'Technology', 'Cybersecurity', 'US', 'NASDAQ', 'stock', 115000000000, 52.5, 6.25, 0.0, true, true),
('SNPS', 'Synopsys Inc.', 'Technology', 'Software', 'US', 'NASDAQ', 'stock', 85000000000, 55.8, 9.85, 0.0, true, true),
('CDNS', 'Cadence Design Systems', 'Technology', 'Software', 'US', 'NASDAQ', 'stock', 80000000000, 68.5, 4.25, 0.0, true, true),
('KLAC', 'KLA Corporation', 'Technology', 'Semiconductors', 'US', 'NASDAQ', 'stock', 95000000000, 28.5, 24.50, 0.0085, true, true),
('PLTR', 'Palantir Technologies', 'Technology', 'Software', 'US', 'NYSE', 'stock', 65000000000, 185.5, 0.35, 0.0, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Financial stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('BAC', 'Bank of America', 'Financial', 'Banking', 'US', 'NYSE', 'stock', 320000000000, 12.5, 3.25, 0.0235, true, true),
('WFC', 'Wells Fargo & Co.', 'Financial', 'Banking', 'US', 'NYSE', 'stock', 210000000000, 12.8, 4.75, 0.0255, true, true),
('GS', 'Goldman Sachs', 'Financial', 'Investment Banking', 'US', 'NYSE', 'stock', 165000000000, 15.2, 32.50, 0.0225, true, true),
('MS', 'Morgan Stanley', 'Financial', 'Investment Banking', 'US', 'NYSE', 'stock', 175000000000, 16.5, 6.85, 0.0325, true, true),
('BLK', 'BlackRock Inc.', 'Financial', 'Asset Management', 'US', 'NYSE', 'stock', 145000000000, 22.5, 42.50, 0.0215, true, true),
('SCHW', 'Charles Schwab', 'Financial', 'Brokerage', 'US', 'NYSE', 'stock', 135000000000, 28.5, 2.65, 0.0135, true, true),
('AXP', 'American Express', 'Financial', 'Credit Services', 'US', 'NYSE', 'stock', 185000000000, 18.5, 14.25, 0.0095, true, true),
('C', 'Citigroup Inc.', 'Financial', 'Banking', 'US', 'NYSE', 'stock', 130000000000, 10.5, 6.85, 0.0325, true, true),
('USB', 'U.S. Bancorp', 'Financial', 'Banking', 'US', 'NYSE', 'stock', 75000000000, 12.5, 3.95, 0.0425, true, true),
('PNC', 'PNC Financial Services', 'Financial', 'Banking', 'US', 'NYSE', 'stock', 75000000000, 13.2, 13.25, 0.0345, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Healthcare stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('LLY', 'Eli Lilly & Co.', 'Healthcare', 'Pharmaceuticals', 'US', 'NYSE', 'stock', 750000000000, 125.5, 6.25, 0.0065, true, true),
('NVO', 'Novo Nordisk A/S', 'Healthcare', 'Pharmaceuticals', 'DK', 'NYSE', 'stock', 450000000000, 45.5, 2.85, 0.0095, true, true),
('ABBV', 'AbbVie Inc.', 'Healthcare', 'Pharmaceuticals', 'US', 'NYSE', 'stock', 310000000000, 55.2, 3.25, 0.0365, true, true),
('MRK', 'Merck & Co.', 'Healthcare', 'Pharmaceuticals', 'US', 'NYSE', 'stock', 280000000000, 18.5, 5.65, 0.0255, true, true),
('PFE', 'Pfizer Inc.', 'Healthcare', 'Pharmaceuticals', 'US', 'NYSE', 'stock', 145000000000, 12.5, 2.15, 0.0575, true, true),
('TMO', 'Thermo Fisher Scientific', 'Healthcare', 'Life Sciences', 'US', 'NYSE', 'stock', 210000000000, 32.5, 17.25, 0.0025, true, true),
('ABT', 'Abbott Laboratories', 'Healthcare', 'Medical Devices', 'US', 'NYSE', 'stock', 195000000000, 35.2, 3.25, 0.0185, true, true),
('DHR', 'Danaher Corporation', 'Healthcare', 'Life Sciences', 'US', 'NYSE', 'stock', 185000000000, 42.5, 6.15, 0.004, true, true),
('BMY', 'Bristol-Myers Squibb', 'Healthcare', 'Pharmaceuticals', 'US', 'NYSE', 'stock', 115000000000, 18.5, 2.95, 0.0455, true, true),
('AMGN', 'Amgen Inc.', 'Healthcare', 'Biotechnology', 'US', 'NASDAQ', 'stock', 155000000000, 25.5, 11.25, 0.0295, true, true),
('GILD', 'Gilead Sciences', 'Healthcare', 'Biotechnology', 'US', 'NASDAQ', 'stock', 105000000000, 18.5, 4.55, 0.0365, true, true),
('ISRG', 'Intuitive Surgical', 'Healthcare', 'Medical Devices', 'US', 'NASDAQ', 'stock', 175000000000, 75.5, 6.25, 0.0, true, true),
('VRTX', 'Vertex Pharmaceuticals', 'Healthcare', 'Biotechnology', 'US', 'NASDAQ', 'stock', 125000000000, 32.5, 15.25, 0.0, true, true),
('REGN', 'Regeneron Pharmaceuticals', 'Healthcare', 'Biotechnology', 'US', 'NASDAQ', 'stock', 95000000000, 22.5, 38.50, 0.0, true, true),
('MDT', 'Medtronic plc', 'Healthcare', 'Medical Devices', 'IE', 'NYSE', 'stock', 105000000000, 28.5, 3.15, 0.0325, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Consumer stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('NKE', 'Nike Inc.', 'Consumer', 'Apparel', 'US', 'NYSE', 'stock', 115000000000, 25.5, 3.25, 0.0155, true, true),
('MCD', 'McDonalds Corporation', 'Consumer', 'Restaurants', 'US', 'NYSE', 'stock', 215000000000, 25.8, 11.25, 0.0205, true, true),
('SBUX', 'Starbucks Corporation', 'Consumer', 'Restaurants', 'US', 'NASDAQ', 'stock', 105000000000, 28.5, 3.25, 0.0235, true, true),
('TGT', 'Target Corporation', 'Consumer', 'Retail', 'US', 'NYSE', 'stock', 65000000000, 15.5, 8.75, 0.0295, true, true),
('LOW', 'Lowes Companies', 'Consumer', 'Retail', 'US', 'NYSE', 'stock', 145000000000, 18.5, 14.25, 0.0175, true, true),
('TJX', 'TJX Companies', 'Consumer', 'Retail', 'US', 'NYSE', 'stock', 135000000000, 28.5, 3.85, 0.0125, true, true),
('CMG', 'Chipotle Mexican Grill', 'Consumer', 'Restaurants', 'US', 'NYSE', 'stock', 85000000000, 58.5, 52.50, 0.0, true, true),
('ORLY', 'OReilly Automotive', 'Consumer', 'Retail', 'US', 'NASDAQ', 'stock', 75000000000, 28.5, 38.50, 0.0, true, true),
('AZO', 'AutoZone Inc.', 'Consumer', 'Retail', 'US', 'NYSE', 'stock', 55000000000, 22.5, 145.50, 0.0, true, true),
('DG', 'Dollar General', 'Consumer', 'Retail', 'US', 'NYSE', 'stock', 35000000000, 18.5, 8.25, 0.0145, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Industrial stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('CAT', 'Caterpillar Inc.', 'Industrial', 'Machinery', 'US', 'NYSE', 'stock', 185000000000, 18.5, 21.25, 0.0145, true, true),
('DE', 'Deere & Company', 'Industrial', 'Machinery', 'US', 'NYSE', 'stock', 125000000000, 15.5, 28.50, 0.0135, true, true),
('UPS', 'United Parcel Service', 'Industrial', 'Logistics', 'US', 'NYSE', 'stock', 115000000000, 18.5, 8.25, 0.0425, true, true),
('HON', 'Honeywell International', 'Industrial', 'Conglomerate', 'US', 'NASDAQ', 'stock', 145000000000, 22.5, 9.25, 0.0195, true, true),
('RTX', 'RTX Corporation', 'Industrial', 'Aerospace', 'US', 'NYSE', 'stock', 155000000000, 35.5, 3.25, 0.0225, true, true),
('BA', 'Boeing Company', 'Industrial', 'Aerospace', 'US', 'NYSE', 'stock', 125000000000, -15.5, -8.25, 0.0, true, true),
('LMT', 'Lockheed Martin', 'Industrial', 'Defense', 'US', 'NYSE', 'stock', 125000000000, 18.5, 27.25, 0.0255, true, true),
('GE', 'GE Aerospace', 'Industrial', 'Aerospace', 'US', 'NYSE', 'stock', 195000000000, 35.5, 4.85, 0.0065, true, true),
('MMM', '3M Company', 'Industrial', 'Conglomerate', 'US', 'NYSE', 'stock', 75000000000, 15.5, 8.25, 0.0525, true, true),
('FDX', 'FedEx Corporation', 'Industrial', 'Logistics', 'US', 'NYSE', 'stock', 75000000000, 18.5, 15.25, 0.0185, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Energy stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('CVX', 'Chevron Corporation', 'Energy', 'Oil & Gas', 'US', 'NYSE', 'stock', 285000000000, 12.5, 12.25, 0.0415, true, true),
('COP', 'ConocoPhillips', 'Energy', 'Oil & Gas', 'US', 'NYSE', 'stock', 125000000000, 11.5, 9.25, 0.0325, true, true),
('SLB', 'Schlumberger Ltd.', 'Energy', 'Oil Services', 'US', 'NYSE', 'stock', 65000000000, 15.5, 3.25, 0.0225, true, true),
('EOG', 'EOG Resources', 'Energy', 'Oil & Gas', 'US', 'NYSE', 'stock', 75000000000, 10.5, 12.25, 0.0285, true, true),
('PXD', 'Pioneer Natural Resources', 'Energy', 'Oil & Gas', 'US', 'NYSE', 'stock', 55000000000, 9.5, 25.50, 0.0545, true, true),
('OXY', 'Occidental Petroleum', 'Energy', 'Oil & Gas', 'US', 'NYSE', 'stock', 55000000000, 8.5, 6.85, 0.0165, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Communication stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('NFLX', 'Netflix Inc.', 'Communication', 'Streaming', 'US', 'NASDAQ', 'stock', 385000000000, 48.5, 18.25, 0.0, true, true),
('DIS', 'Walt Disney Company', 'Communication', 'Entertainment', 'US', 'NYSE', 'stock', 205000000000, 72.5, 1.55, 0.0085, true, true),
('CMCSA', 'Comcast Corporation', 'Communication', 'Media', 'US', 'NASDAQ', 'stock', 165000000000, 12.5, 3.25, 0.0285, true, true),
('T', 'AT&T Inc.', 'Communication', 'Telecom', 'US', 'NYSE', 'stock', 165000000000, 12.5, 1.85, 0.0565, true, true),
('VZ', 'Verizon Communications', 'Communication', 'Telecom', 'US', 'NYSE', 'stock', 175000000000, 10.5, 4.25, 0.0625, true, true),
('TMUS', 'T-Mobile US', 'Communication', 'Telecom', 'US', 'NASDAQ', 'stock', 265000000000, 28.5, 8.25, 0.0145, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Materials stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('LIN', 'Linde plc', 'Materials', 'Chemicals', 'IE', 'NYSE', 'stock', 215000000000, 32.5, 14.25, 0.0125, true, true),
('APD', 'Air Products & Chemicals', 'Materials', 'Chemicals', 'US', 'NYSE', 'stock', 65000000000, 28.5, 10.25, 0.0235, true, true),
('SHW', 'Sherwin-Williams', 'Materials', 'Chemicals', 'US', 'NYSE', 'stock', 95000000000, 35.5, 10.85, 0.008, true, true),
('FCX', 'Freeport-McMoRan', 'Materials', 'Mining', 'US', 'NYSE', 'stock', 65000000000, 25.5, 1.75, 0.0125, true, true),
('NEM', 'Newmont Corporation', 'Materials', 'Mining', 'US', 'NYSE', 'stock', 55000000000, 18.5, 2.35, 0.0245, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Utilities stocks
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('NEE', 'NextEra Energy', 'Utilities', 'Electric Utilities', 'US', 'NYSE', 'stock', 155000000000, 22.5, 3.25, 0.0255, true, true),
('DUK', 'Duke Energy', 'Utilities', 'Electric Utilities', 'US', 'NYSE', 'stock', 85000000000, 18.5, 5.65, 0.0385, true, true),
('SO', 'Southern Company', 'Utilities', 'Electric Utilities', 'US', 'NYSE', 'stock', 95000000000, 22.5, 3.85, 0.0345, true, true),
('D', 'Dominion Energy', 'Utilities', 'Electric Utilities', 'US', 'NYSE', 'stock', 45000000000, 15.5, 3.25, 0.0485, true, true),
('AEP', 'American Electric Power', 'Utilities', 'Electric Utilities', 'US', 'NASDAQ', 'stock', 55000000000, 18.5, 5.15, 0.0355, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap,
    pe_ratio = EXCLUDED.pe_ratio;

-- Additional ETFs
INSERT INTO symbols (symbol, name, sector, industry, country, exchange, asset_type, market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active) VALUES
('ARKK', 'ARK Innovation ETF', 'Index', 'ETF', 'US', 'NYSE', 'etf', 6000000000, NULL, NULL, 0.0, true, true),
('XLF', 'Financial Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 45000000000, NULL, NULL, 0.0165, true, true),
('XLK', 'Technology Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 65000000000, NULL, NULL, 0.0065, true, true),
('XLE', 'Energy Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 35000000000, NULL, NULL, 0.0335, true, true),
('XLV', 'Health Care Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 40000000000, NULL, NULL, 0.0145, true, true),
('XLI', 'Industrial Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 20000000000, NULL, NULL, 0.0135, true, true),
('XLY', 'Consumer Discretionary Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 22000000000, NULL, NULL, 0.0085, true, true),
('XLP', 'Consumer Staples Select Sector SPDR', 'Index', 'ETF', 'US', 'NYSE', 'etf', 18000000000, NULL, NULL, 0.0255, true, true),
('SCHD', 'Schwab US Dividend Equity ETF', 'Index', 'ETF', 'US', 'NYSE', 'etf', 55000000000, NULL, NULL, 0.0345, true, true),
('VGT', 'Vanguard Information Technology ETF', 'Index', 'ETF', 'US', 'NYSE', 'etf', 75000000000, NULL, NULL, 0.0055, true, true)
ON CONFLICT (symbol) DO UPDATE SET
    name = EXCLUDED.name,
    sector = EXCLUDED.sector,
    market_cap = EXCLUDED.market_cap;

-- Update polygon service DEFAULT_SYMBOLS list should be updated too
-- Verify count
SELECT COUNT(*) as total_symbols FROM symbols;
