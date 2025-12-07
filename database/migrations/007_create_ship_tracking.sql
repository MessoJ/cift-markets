-- Migration: Create Ship Tracking System
-- Purpose: Track major cargo vessels (oil tankers, container ships) that impact markets
-- Date: 2025-11-17

-- Table: tracked_ships
-- Stores major market-moving vessels (oil tankers, container ships, LNG carriers)
CREATE TABLE IF NOT EXISTS tracked_ships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mmsi VARCHAR(20) UNIQUE NOT NULL, -- Maritime Mobile Service Identity
    imo VARCHAR(20), -- International Maritime Organization number
    ship_name VARCHAR(255) NOT NULL,
    ship_type VARCHAR(50) NOT NULL, -- 'oil_tanker', 'lng_carrier', 'container', 'bulk_carrier'
    flag_country VARCHAR(100),
    flag_country_code VARCHAR(2),
    deadweight_tonnage INTEGER, -- Carrying capacity in tons
    build_year INTEGER,
    current_lat DECIMAL(10, 8),
    current_lng DECIMAL(11, 8),
    current_speed DECIMAL(5, 2), -- Knots
    current_course DECIMAL(5, 2), -- Degrees
    current_status VARCHAR(50), -- 'underway', 'at_anchor', 'moored', 'not_under_command'
    destination VARCHAR(255),
    eta TIMESTAMP,
    cargo_type VARCHAR(100),
    cargo_value_usd BIGINT, -- Estimated cargo value
    last_port VARCHAR(100),
    next_port VARCHAR(100),
    importance_score INTEGER DEFAULT 50 CHECK (importance_score >= 0 AND importance_score <= 100),
    is_active BOOLEAN DEFAULT true,
    last_updated TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for tracked_ships
CREATE INDEX IF NOT EXISTS idx_ships_type ON tracked_ships(ship_type);
CREATE INDEX IF NOT EXISTS idx_ships_status ON tracked_ships(current_status);
CREATE INDEX IF NOT EXISTS idx_ships_active ON tracked_ships(is_active);
CREATE INDEX IF NOT EXISTS idx_ships_importance ON tracked_ships(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_ships_position ON tracked_ships(current_lat, current_lng);

-- Table: ship_position_history
-- Tracks historical positions for route visualization
CREATE TABLE IF NOT EXISTS ship_position_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ship_id UUID NOT NULL REFERENCES tracked_ships(id) ON DELETE CASCADE,
    lat DECIMAL(10, 8) NOT NULL,
    lng DECIMAL(11, 8) NOT NULL,
    speed DECIMAL(5, 2),
    course DECIMAL(5, 2),
    timestamp TIMESTAMP DEFAULT NOW()
);

-- Indexes for ship_position_history
CREATE INDEX IF NOT EXISTS idx_ship_history_ship ON ship_position_history(ship_id);
CREATE INDEX IF NOT EXISTS idx_ship_history_time ON ship_position_history(timestamp DESC);

-- Table: ship_news_mentions
-- Links news articles to ship tracking events
CREATE TABLE IF NOT EXISTS ship_news_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ship_id UUID NOT NULL REFERENCES tracked_ships(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    relevance_score DECIMAL(3, 2) DEFAULT 0.5 CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    mentioned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ship_id, article_id)
);

-- Indexes for ship_news_mentions
CREATE INDEX IF NOT EXISTS idx_ship_mentions_ship ON ship_news_mentions(ship_id);
CREATE INDEX IF NOT EXISTS idx_ship_mentions_article ON ship_news_mentions(article_id);
CREATE INDEX IF NOT EXISTS idx_ship_mentions_time ON ship_news_mentions(mentioned_at DESC);

-- Add CHECK constraints for valid ship types
ALTER TABLE tracked_ships 
ADD CONSTRAINT ship_type_valid 
CHECK (ship_type IN ('oil_tanker', 'lng_carrier', 'container', 'bulk_carrier', 'chemical_tanker'));

-- Function to update last_updated timestamp
CREATE OR REPLACE FUNCTION update_ships_last_updated()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update last_updated
DROP TRIGGER IF EXISTS trigger_update_ships_last_updated ON tracked_ships;
CREATE TRIGGER trigger_update_ships_last_updated
    BEFORE UPDATE ON tracked_ships
    FOR EACH ROW
    EXECUTE FUNCTION update_ships_last_updated();

-- Create view for current ship status with news
CREATE OR REPLACE VIEW ships_current_status AS
SELECT 
    s.id,
    s.mmsi,
    s.imo,
    s.ship_name,
    s.ship_type,
    s.flag_country,
    s.flag_country_code,
    s.deadweight_tonnage,
    s.current_lat,
    s.current_lng,
    s.current_speed,
    s.current_course,
    s.current_status,
    s.destination,
    s.eta,
    s.cargo_type,
    s.cargo_value_usd,
    s.importance_score,
    s.last_updated,
    COUNT(DISTINCT snm.article_id) as news_count,
    COALESCE(
        AVG(CASE 
            WHEN a.sentiment = 'positive' THEN 0.7
            WHEN a.sentiment = 'negative' THEN -0.7
            ELSE 0
        END),
        0
    ) as avg_sentiment
FROM tracked_ships s
LEFT JOIN ship_news_mentions snm ON s.id = snm.ship_id 
    AND snm.mentioned_at >= NOW() - INTERVAL '24 hours'
LEFT JOIN news_articles a ON a.id = snm.article_id
WHERE s.is_active = true
GROUP BY s.id;

-- Grant permissions
GRANT SELECT, INSERT, UPDATE ON tracked_ships TO cift_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ship_position_history TO cift_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ship_news_mentions TO cift_user;
GRANT SELECT ON ships_current_status TO cift_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ Ship tracking system created successfully';
    RAISE NOTICE '📊 Tables: tracked_ships, ship_position_history, ship_news_mentions';
    RAISE NOTICE '👁️  View: ships_current_status';
    RAISE NOTICE '🚢 Ready to track oil tankers, LNG carriers, and container ships';
END $$;
