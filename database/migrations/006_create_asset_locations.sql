-- Migration: Create Asset Locations System
-- Purpose: Track major market-moving locations (central banks, commodities, tech HQs, etc.)
-- Date: 2025-11-17

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: asset_locations
-- Stores major financial assets and institutions that influence markets
CREATE TABLE IF NOT EXISTS asset_locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    code VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(50) NOT NULL, -- 'central_bank', 'commodity_market', 'government', 'tech_hq', 'energy'
    country VARCHAR(100) NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    city VARCHAR(100),
    lat DECIMAL(10, 8) NOT NULL,
    lng DECIMAL(11, 8) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    description TEXT,
    importance_score INTEGER DEFAULT 50 CHECK (importance_score >= 0 AND importance_score <= 100),
    website VARCHAR(255),
    icon_url VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for asset_locations
CREATE INDEX IF NOT EXISTS idx_asset_locations_type ON asset_locations(asset_type);
CREATE INDEX IF NOT EXISTS idx_asset_locations_country ON asset_locations(country_code);
CREATE INDEX IF NOT EXISTS idx_asset_locations_active ON asset_locations(is_active);
CREATE INDEX IF NOT EXISTS idx_asset_locations_importance ON asset_locations(importance_score DESC);

-- Table: asset_status_log
-- Tracks operational status of assets over time based on news analysis
CREATE TABLE IF NOT EXISTS asset_status_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES asset_locations(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL CHECK (status IN ('operational', 'unknown', 'issue')),
    sentiment_score DECIMAL(3, 2) CHECK (sentiment_score >= -1.0 AND sentiment_score <= 1.0),
    news_count INTEGER DEFAULT 0 CHECK (news_count >= 0),
    last_news_at TIMESTAMP,
    status_reason TEXT,
    checked_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for asset_status_log
CREATE INDEX IF NOT EXISTS idx_asset_status_asset ON asset_status_log(asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_status_time ON asset_status_log(checked_at DESC);
CREATE INDEX IF NOT EXISTS idx_asset_status_latest ON asset_status_log(asset_id, checked_at DESC);

-- Table: asset_news_mentions
-- Links news articles to specific assets they mention
CREATE TABLE IF NOT EXISTS asset_news_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    asset_id UUID NOT NULL REFERENCES asset_locations(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    relevance_score DECIMAL(3, 2) DEFAULT 0.5 CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0),
    mentioned_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(asset_id, article_id)
);

-- Indexes for asset_news_mentions
CREATE INDEX IF NOT EXISTS idx_asset_mentions_asset ON asset_news_mentions(asset_id);
CREATE INDEX IF NOT EXISTS idx_asset_mentions_article ON asset_news_mentions(article_id);
CREATE INDEX IF NOT EXISTS idx_asset_mentions_time ON asset_news_mentions(mentioned_at DESC);

-- Add CHECK constraints for valid asset types
ALTER TABLE asset_locations 
ADD CONSTRAINT asset_type_valid 
CHECK (asset_type IN ('central_bank', 'commodity_market', 'government', 'tech_hq', 'energy'));

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_asset_locations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
DROP TRIGGER IF EXISTS trigger_update_asset_locations_updated_at ON asset_locations;
CREATE TRIGGER trigger_update_asset_locations_updated_at
    BEFORE UPDATE ON asset_locations
    FOR EACH ROW
    EXECUTE FUNCTION update_asset_locations_updated_at();

-- Create view for latest asset status
CREATE OR REPLACE VIEW asset_current_status AS
SELECT DISTINCT ON (asl.asset_id)
    al.id,
    al.code,
    al.name,
    al.asset_type,
    al.country,
    al.country_code,
    al.city,
    al.lat,
    al.lng,
    al.timezone,
    al.importance_score,
    al.is_active,
    asl.status,
    asl.sentiment_score,
    asl.news_count,
    asl.last_news_at,
    asl.checked_at
FROM asset_locations al
LEFT JOIN asset_status_log asl ON al.id = asl.asset_id
WHERE al.is_active = true
ORDER BY asl.asset_id, asl.checked_at DESC;

-- Grant permissions (adjust user as needed)
GRANT SELECT, INSERT, UPDATE ON asset_locations TO cift_user;
GRANT SELECT, INSERT, UPDATE ON asset_status_log TO cift_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON asset_news_mentions TO cift_user;
GRANT SELECT ON asset_current_status TO cift_user;

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ Asset locations system created successfully';
    RAISE NOTICE '📊 Tables: asset_locations, asset_status_log, asset_news_mentions';
    RAISE NOTICE '👁️  View: asset_current_status';
END $$;
