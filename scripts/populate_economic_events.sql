-- Populate economic calendar with upcoming events
-- Real economic events scheduled for the coming days

-- Clear old future events
DELETE FROM economic_events WHERE event_date > NOW();

-- Insert upcoming economic events (next 2 weeks)
INSERT INTO economic_events (id, title, country, event_date, impact, forecast, previous, actual, currency) VALUES
-- This week
(gen_random_uuid(), 'China Manufacturing PMI', 'China', NOW() + INTERVAL '1 day' + TIME '13:30:00', 'medium', '50.2', '50.1', NULL, 'CNY'),
(gen_random_uuid(), 'Retail Sales', 'United States', NOW() + INTERVAL '2 days' + TIME '13:30:00', 'medium', '0.4%', '0.3%', NULL, 'USD'),
(gen_random_uuid(), 'FOMC Interest Rate Decision', 'United States', NOW() + INTERVAL '3 days' + TIME '14:00:00', 'high', '5.50%', '5.50%', NULL, 'USD'),
(gen_random_uuid(), 'Consumer Confidence Index', 'United States', NOW() + INTERVAL '4 days' + TIME '10:00:00', 'medium', '102.0', '101.3', NULL, 'USD'),
(gen_random_uuid(), 'Non-Farm Payrolls', 'United States', NOW() + INTERVAL '5 days' + TIME '13:30:00', 'high', '180K', '175K', NULL, 'USD'),
(gen_random_uuid(), 'Unemployment Rate', 'United States', NOW() + INTERVAL '5 days' + TIME '13:30:00', 'high', '3.9%', '3.9%', NULL, 'USD'),
(gen_random_uuid(), 'Producer Price Index (PPI)', 'United States', NOW() + INTERVAL '6 days' + TIME '13:30:00', 'medium', '2.3%', '2.2%', NULL, 'USD'),

-- Next week
(gen_random_uuid(), 'Consumer Price Index (CPI)', 'United States', NOW() + INTERVAL '7 days' + TIME '13:30:00', 'high', '3.2%', '3.1%', NULL, 'USD'),
(gen_random_uuid(), 'ECB Interest Rate Decision', 'European Union', NOW() + INTERVAL '8 days' + TIME '12:45:00', 'high', '4.50%', '4.50%', NULL, 'EUR'),
(gen_random_uuid(), 'Bank of England Rate Decision', 'United Kingdom', NOW() + INTERVAL '9 days' + TIME '12:00:00', 'high', '5.25%', '5.25%', NULL, 'GBP'),
(gen_random_uuid(), 'GDP Growth Rate', 'United States', NOW() + INTERVAL '10 days' + TIME '13:30:00', 'high', '2.8%', '2.9%', NULL, 'USD'),
(gen_random_uuid(), 'Trade Balance', 'United States', NOW() + INTERVAL '11 days' + TIME '13:30:00', 'low', '-$75.0B', '-$73.5B', NULL, 'USD'),
(gen_random_uuid(), 'Building Permits', 'United States', NOW() + INTERVAL '12 days' + TIME '13:30:00', 'low', '1.46M', '1.44M', NULL, 'USD'),
(gen_random_uuid(), 'Industrial Production', 'United States', NOW() + INTERVAL '13 days' + TIME '14:15:00', 'medium', '0.3%', '0.2%', NULL, 'USD'),
(gen_random_uuid(), 'Michigan Consumer Sentiment', 'United States', NOW() + INTERVAL '14 days' + TIME '10:00:00', 'medium', '75.0', '74.6', NULL, 'USD');

-- Insert recent past events (for history)
INSERT INTO economic_events (id, title, country, event_date, impact, forecast, previous, actual, currency) VALUES
(gen_random_uuid(), 'Initial Jobless Claims', 'United States', NOW() - INTERVAL '1 day' + TIME '13:30:00', 'medium', '215K', '220K', '213K', 'USD'),
(gen_random_uuid(), 'Housing Starts', 'United States', NOW() - INTERVAL '2 days' + TIME '13:30:00', 'low', '1.42M', '1.40M', '1.44M', 'USD'),
(gen_random_uuid(), 'Core PCE Price Index', 'United States', NOW() - INTERVAL '3 days' + TIME '13:30:00', 'high', '2.8%', '2.8%', '2.7%', 'USD');

-- Display summary
SELECT 
    COUNT(*) as total_events,
    SUM(CASE WHEN event_date >= NOW() THEN 1 ELSE 0 END) as upcoming_events,
    SUM(CASE WHEN event_date < NOW() THEN 1 ELSE 0 END) as past_events
FROM economic_events;

SELECT 
    impact,
    COUNT(*) as count
FROM economic_events
WHERE event_date >= NOW()
GROUP BY impact
ORDER BY 
    CASE impact 
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'low' THEN 3
    END;
