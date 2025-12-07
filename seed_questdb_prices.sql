-- Seed market_quotes data in QuestDB for the symbols we added
-- This allows the screener to return actual results

-- Insert recent price data for major stocks
INSERT INTO market_quotes (symbol, price, change, change_percent, volume, timestamp)
VALUES
    -- Technology
    ('AAPL', 182.45, 2.31, 1.28, 52340000, now()),
    ('MSFT', 378.91, 4.23, 1.13, 19234000, now()),
    ('GOOGL', 139.85, -1.23, -0.87, 28450000, now()),
    ('NVDA', 495.22, 8.45, 1.74, 45670000, now()),
    ('META', 342.78, 5.67, 1.68, 12340000, now()),
    
    -- Healthcare
    ('JNJ', 159.87, 0.45, 0.28, 6780000, now()),
    ('UNH', 523.45, 3.21, 0.62, 2340000, now()),
    ('PFE', 28.92, -0.34, -1.16, 34560000, now()),
    
    -- Financial
    ('JPM', 154.67, 1.23, 0.80, 8970000, now()),
    ('BAC', 34.12, 0.23, 0.68, 45670000, now()),
    ('V', 254.32, 2.10, 0.83, 5670000, now()),
    
    -- Consumer
    ('AMZN', 151.23, 2.34, 1.57, 34560000, now()),
    ('TSLA', 242.84, -3.45, -1.40, 98760000, now()),
    ('WMT', 162.45, 0.78, 0.48, 7890000, now()),
    ('HD', 341.23, 1.56, 0.46, 3450000, now()),
    
    -- Energy
    ('XOM', 108.45, 1.23, 1.15, 15670000, now()),
    ('CVX', 148.92, 1.67, 1.13, 8970000, now()),
    
    -- Industrial
    ('BA', 215.67, -2.34, -1.07, 4560000, now()),
    ('CAT', 289.45, 3.12, 1.09, 2340000, now()),
    
    -- Materials
    ('LIN', 405.78, 2.45, 0.61, 1230000, now()),
    
    -- ETFs
    ('SPY', 456.78, 2.34, 0.51, 87600000, now()),
    ('QQQ', 389.45, 3.21, 0.83, 45670000, now()),
    ('IWM', 198.23, 0.98, 0.50, 23450000, now());
