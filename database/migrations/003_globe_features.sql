-- Globe Features Migration
-- Adds tables for stock exchanges, news geotags, and news connections

-- Stock Exchanges Table
CREATE TABLE IF NOT EXISTS stock_exchanges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    country VARCHAR(100) NOT NULL,
    country_code VARCHAR(2) NOT NULL,
    lat DECIMAL(10, 6) NOT NULL,
    lng DECIMAL(10, 6) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    market_cap_usd BIGINT,
    trading_hours JSONB DEFAULT '{"open": "09:30", "close": "16:00"}',
    website VARCHAR(500),
    icon_url VARCHAR(500),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- News Geographic Tags
CREATE TABLE IF NOT EXISTS news_geotags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    exchange_id UUID REFERENCES stock_exchanges(id) ON DELETE SET NULL,
    country_code VARCHAR(2),
    lat DECIMAL(10, 6),
    lng DECIMAL(10, 6),
    relevance_score DECIMAL(3, 2) CHECK (relevance_score BETWEEN 0 AND 1),
    created_at TIMESTAMP DEFAULT NOW()
);

-- News Connections (for animated arcs)
CREATE TABLE IF NOT EXISTS news_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_exchange_id UUID NOT NULL REFERENCES stock_exchanges(id) ON DELETE CASCADE,
    target_exchange_id UUID NOT NULL REFERENCES stock_exchanges(id) ON DELETE CASCADE,
    article_id UUID NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    connection_type VARCHAR(50) NOT NULL,
    strength DECIMAL(3, 2) CHECK (strength BETWEEN 0 AND 1),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_connection UNIQUE (source_exchange_id, target_exchange_id, article_id)
);

-- Indexes for Stock Exchanges
CREATE INDEX IF NOT EXISTS idx_exchanges_code ON stock_exchanges(code);
CREATE INDEX IF NOT EXISTS idx_exchanges_country ON stock_exchanges(country_code);
CREATE INDEX IF NOT EXISTS idx_exchanges_active ON stock_exchanges(is_active) WHERE is_active = true;

-- Indexes for News Geotags
CREATE INDEX IF NOT EXISTS idx_geotags_article ON news_geotags(article_id);
CREATE INDEX IF NOT EXISTS idx_geotags_exchange ON news_geotags(exchange_id);
CREATE INDEX IF NOT EXISTS idx_geotags_country ON news_geotags(country_code);
CREATE INDEX IF NOT EXISTS idx_geotags_relevance ON news_geotags(relevance_score DESC);

-- Indexes for News Connections
CREATE INDEX IF NOT EXISTS idx_connections_source ON news_connections(source_exchange_id);
CREATE INDEX IF NOT EXISTS idx_connections_target ON news_connections(target_exchange_id);
CREATE INDEX IF NOT EXISTS idx_connections_article ON news_connections(article_id);
CREATE INDEX IF NOT EXISTS idx_connections_type ON news_connections(connection_type);
CREATE INDEX IF NOT EXISTS idx_connections_strength ON news_connections(strength DESC);
CREATE INDEX IF NOT EXISTS idx_connections_created ON news_connections(created_at DESC);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for stock_exchanges
DROP TRIGGER IF EXISTS update_stock_exchanges_updated_at ON stock_exchanges;
CREATE TRIGGER update_stock_exchanges_updated_at
    BEFORE UPDATE ON stock_exchanges
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE stock_exchanges IS 'Major global stock exchanges with geographic coordinates';
COMMENT ON TABLE news_geotags IS 'Geographic tags linking news articles to locations and exchanges';
COMMENT ON TABLE news_connections IS 'Connections between markets for arc visualization';

COMMENT ON COLUMN stock_exchanges.code IS 'Exchange code (e.g., NYSE, LSE, SSE)';
COMMENT ON COLUMN stock_exchanges.market_cap_usd IS 'Total market capitalization in USD';
COMMENT ON COLUMN news_geotags.relevance_score IS 'How relevant the location is to the article (0-1)';
COMMENT ON COLUMN news_connections.strength IS 'Connection strength for visual representation (0-1)';
COMMENT ON COLUMN news_connections.connection_type IS 'Type: trade, impact, correlation, etc.';
