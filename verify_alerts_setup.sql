-- Verify alerts table exists and structure
\dt price_alerts

-- Check if price_alerts table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'price_alerts'
) as price_alerts_exists;

-- Check columns
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'price_alerts' 
ORDER BY ordinal_position;

-- Count existing alerts
SELECT COUNT(*) as total_alerts FROM price_alerts;

-- Count by status
SELECT status, COUNT(*) as count 
FROM price_alerts 
GROUP BY status;
