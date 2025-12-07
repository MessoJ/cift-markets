-- Add geo-location columns to news_articles table
-- This enables mapping news to countries/regions for globe visualization

ALTER TABLE news_articles 
ADD COLUMN IF NOT EXISTS country VARCHAR(100),
ADD COLUMN IF NOT EXISTS country_code VARCHAR(3),
ADD COLUMN IF NOT EXISTS latitude DECIMAL(10, 8),
ADD COLUMN IF NOT EXISTS longitude DECIMAL(11, 8),
ADD COLUMN IF NOT EXISTS region VARCHAR(50);  -- Americas, Europe, Asia, Africa, Oceania

-- Add indexes for geo queries
CREATE INDEX IF NOT EXISTS idx_news_geo ON news_articles(country_code, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_region ON news_articles(region, published_at DESC);

-- Comment on columns
COMMENT ON COLUMN news_articles.country IS 'Country name where news originated';
COMMENT ON COLUMN news_articles.country_code IS 'ISO 3166-1 alpha-2 country code';
COMMENT ON COLUMN news_articles.latitude IS 'Geographic latitude for globe plotting';
COMMENT ON COLUMN news_articles.longitude IS 'Geographic longitude for globe plotting';
COMMENT ON COLUMN news_articles.region IS 'World region: Americas, Europe, Asia, Africa, Oceania';
