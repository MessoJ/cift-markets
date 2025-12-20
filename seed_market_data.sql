-- Seed market_data_cache with initial stock data
INSERT INTO market_data_cache (symbol, price, bid, ask, volume, change, change_pct, high, low, open)
VALUES 
    ('AAPL', 170.00, 169.98, 170.02, 1000000, 1.50, 0.89, 172.00, 168.50, 169.00),
    ('MSFT', 350.00, 349.95, 350.05, 800000, 2.00, 0.57, 352.00, 348.00, 349.00),
    ('GOOGL', 140.00, 139.98, 140.02, 500000, 1.20, 0.86, 141.50, 138.50, 139.00),
    ('AMZN', 155.00, 154.98, 155.02, 600000, 1.80, 1.17, 157.00, 153.00, 154.00),
    ('TSLA', 245.00, 244.95, 245.05, 900000, 3.50, 1.45, 248.00, 242.00, 243.00),
    ('META', 485.00, 484.95, 485.05, 400000, 4.00, 0.83, 490.00, 480.00, 482.00),
    ('NVDA', 485.00, 484.95, 485.05, 700000, 5.00, 1.04, 492.00, 478.00, 481.00),
    ('AMD', 140.00, 139.98, 140.02, 550000, 2.20, 1.60, 143.00, 137.00, 138.00)
ON CONFLICT (symbol) DO UPDATE SET 
    price = EXCLUDED.price, 
    bid = EXCLUDED.bid, 
    ask = EXCLUDED.ask, 
    volume = EXCLUDED.volume, 
    updated_at = CURRENT_TIMESTAMP;

-- Also seed the market_data table
INSERT INTO market_data (symbol, price, bid, ask, volume)
VALUES 
    ('AAPL', 170.00, 169.98, 170.02, 1000000),
    ('MSFT', 350.00, 349.95, 350.05, 800000),
    ('GOOGL', 140.00, 139.98, 140.02, 500000),
    ('AMZN', 155.00, 154.98, 155.02, 600000),
    ('TSLA', 245.00, 244.95, 245.05, 900000),
    ('META', 485.00, 484.95, 485.05, 400000),
    ('NVDA', 485.00, 484.95, 485.05, 700000),
    ('AMD', 140.00, 139.98, 140.02, 550000);
