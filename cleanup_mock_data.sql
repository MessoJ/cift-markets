-- CIFT Markets - Clean Up Mock/Invalid Data from Database
-- Run this script to remove all mock data and invalid symbols

-- 1. List current invalid symbols in market_data_cache (for review)
SELECT symbol, price, volume, updated_at 
FROM market_data_cache 
WHERE symbol IN ('S&P', 'DOW', 'NASDAQ', 'N', 'NK', 'RUSSELL', 'VIX', 'S&P500', 'DJIA', 'NDX', 'RUT')
   OR symbol ~ '^[A-Z]$'  -- Single letter symbols (likely invalid)
   OR price < 0.01        -- Suspicious zero/near-zero prices
ORDER BY symbol;

-- 2. Delete invalid/mock symbols from market_data_cache
DELETE FROM market_data_cache 
WHERE symbol IN ('S&P', 'DOW', 'NASDAQ', 'N', 'NK', 'RUSSELL', 'VIX', 'S&P500', 'DJIA', 'NDX', 'RUT')
   OR symbol ~ '^[A-Z]$';

-- 3. Delete stale data (older than 7 days without updates)
DELETE FROM market_data_cache 
WHERE updated_at < NOW() - INTERVAL '7 days';

-- 4. Clean up symbols table of invalid entries
DELETE FROM symbols 
WHERE symbol IN ('S&P', 'DOW', 'NASDAQ', 'N', 'NK', 'RUSSELL', 'VIX', 'S&P500', 'DJIA', 'NDX', 'RUT')
   OR symbol ~ '^[A-Z]$';

-- 5. Show remaining valid symbols
SELECT symbol, name, price, volume, change_percent, updated_at 
FROM market_data_cache 
ORDER BY volume DESC NULLS LAST
LIMIT 50;

-- 6. Verify cleanup completed
SELECT COUNT(*) AS remaining_symbols FROM market_data_cache;
