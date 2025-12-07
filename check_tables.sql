-- Check if saved_screens table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'saved_screens'
) as saved_screens_exists;

-- Check if symbols table exists
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'symbols'
) as symbols_exists;

-- Show all tables
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;
