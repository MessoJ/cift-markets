-- CIFT Markets - Company Data Tables Migration
-- Adds tables for company profiles, earnings, and enhanced market data

-- ============================================================================
-- COMPANY PROFILES TABLE
-- ============================================================================
-- Stores company fundamental data from Finnhub

CREATE TABLE IF NOT EXISTS company_profiles (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    exchange VARCHAR(100),
    industry VARCHAR(100),
    sector VARCHAR(100),
    market_cap NUMERIC(15, 2),           -- In millions USD
    shares_outstanding NUMERIC(15, 4),   -- In millions
    ipo_date DATE,
    logo_url TEXT,
    website TEXT,
    description TEXT,
    currency VARCHAR(10) DEFAULT 'USD',
    country VARCHAR(10) DEFAULT 'US',
    employees INTEGER,
    ceo VARCHAR(255),
    phone VARCHAR(50),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    zip VARCHAR(20),
    -- Calculated fields
    pe_ratio NUMERIC(10, 4),
    forward_pe NUMERIC(10, 4),
    dividend_yield NUMERIC(10, 4),
    beta NUMERIC(10, 4),
    fifty_two_week_high NUMERIC(15, 4),
    fifty_two_week_low NUMERIC(15, 4),
    avg_volume_10d BIGINT,
    avg_volume_3m BIGINT,
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_company_profiles_industry ON company_profiles(industry);
CREATE INDEX IF NOT EXISTS idx_company_profiles_sector ON company_profiles(sector);
CREATE INDEX IF NOT EXISTS idx_company_profiles_market_cap ON company_profiles(market_cap DESC);

-- ============================================================================
-- EARNINGS CALENDAR TABLE
-- ============================================================================
-- Stores earnings announcements and results

CREATE TABLE IF NOT EXISTS earnings_calendar (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    earnings_date DATE NOT NULL,
    quarter INTEGER,                     -- 1, 2, 3, 4
    year INTEGER,
    eps_estimate NUMERIC(10, 4),
    eps_actual NUMERIC(10, 4),
    eps_surprise NUMERIC(10, 4),
    eps_surprise_pct NUMERIC(10, 4),
    revenue_estimate NUMERIC(15, 2),     -- In millions
    revenue_actual NUMERIC(15, 2),
    revenue_surprise NUMERIC(15, 2),
    revenue_surprise_pct NUMERIC(10, 4),
    report_time VARCHAR(10),             -- 'bmo' (before market open), 'amc' (after market close)
    conference_call_time TIMESTAMP WITH TIME ZONE,
    conference_call_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, earnings_date)
);

CREATE INDEX IF NOT EXISTS idx_earnings_symbol ON earnings_calendar(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_upcoming ON earnings_calendar(earnings_date) WHERE earnings_date >= CURRENT_DATE;

-- ============================================================================
-- PATTERN RECOGNITION TABLE
-- ============================================================================
-- Stores detected chart patterns from Finnhub

CREATE TABLE IF NOT EXISTS chart_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,      -- D, W, M
    pattern_name VARCHAR(100) NOT NULL,  -- e.g., 'Double Top', 'Head and Shoulders'
    pattern_type VARCHAR(50),            -- 'bullish', 'bearish', 'neutral'
    status VARCHAR(50),                  -- 'emerging', 'complete', 'confirmed'
    start_date DATE,
    end_date DATE,
    price_start NUMERIC(15, 4),
    price_end NUMERIC(15, 4),
    target_price NUMERIC(15, 4),
    stop_loss NUMERIC(15, 4),
    confidence NUMERIC(5, 4),            -- 0.0 to 1.0
    notes TEXT,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patterns_symbol ON chart_patterns(symbol);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON chart_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_detected ON chart_patterns(detected_at DESC);

-- ============================================================================
-- SUPPORT/RESISTANCE LEVELS TABLE
-- ============================================================================
-- Stores calculated support and resistance levels

CREATE TABLE IF NOT EXISTS support_resistance_levels (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    level_type VARCHAR(20) NOT NULL,     -- 'support', 'resistance', 'pivot'
    price NUMERIC(15, 4) NOT NULL,
    strength INTEGER DEFAULT 1,          -- Number of touches/tests
    first_tested DATE,
    last_tested DATE,
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sr_symbol ON support_resistance_levels(symbol);
CREATE INDEX IF NOT EXISTS idx_sr_active ON support_resistance_levels(symbol) WHERE is_active = TRUE;

-- ============================================================================
-- COMPANY NEWS TABLE
-- ============================================================================
-- Stores company-specific news for chart overlays

CREATE TABLE IF NOT EXISTS company_news (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source VARCHAR(100),
    url TEXT,
    image_url TEXT,
    category VARCHAR(50),                -- 'earnings', 'merger', 'product', 'general'
    sentiment VARCHAR(20),               -- 'positive', 'negative', 'neutral'
    sentiment_score NUMERIC(5, 4),       -- -1.0 to 1.0
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_symbol ON company_news(symbol);
CREATE INDEX IF NOT EXISTS idx_news_published ON company_news(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_sentiment ON company_news(sentiment);

-- ============================================================================
-- ANALYST RATINGS TABLE
-- ============================================================================
-- Stores analyst recommendations

CREATE TABLE IF NOT EXISTS analyst_ratings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    analyst_firm VARCHAR(255),
    analyst_name VARCHAR(255),
    rating VARCHAR(50) NOT NULL,         -- 'buy', 'hold', 'sell', 'strong_buy', 'strong_sell'
    target_price NUMERIC(15, 4),
    previous_rating VARCHAR(50),
    previous_target NUMERIC(15, 4),
    rating_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, analyst_firm, rating_date)
);

CREATE INDEX IF NOT EXISTS idx_ratings_symbol ON analyst_ratings(symbol);
CREATE INDEX IF NOT EXISTS idx_ratings_date ON analyst_ratings(rating_date DESC);

-- ============================================================================
-- ADD UNIQUE CONSTRAINT TO OHLCV_BARS (for upsert)
-- ============================================================================
-- Note: This may fail if constraint already exists - that's OK

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'ohlcv_bars_symbol_timeframe_timestamp_key'
    ) THEN
        -- First check if ohlcv_bars exists (it might be in QuestDB only)
        IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'ohlcv_bars') THEN
            ALTER TABLE ohlcv_bars ADD CONSTRAINT ohlcv_bars_symbol_timeframe_timestamp_key 
            UNIQUE (symbol, timeframe, timestamp);
        END IF;
    END IF;
END $$;

-- ============================================================================
-- ENHANCED MARKET DATA CACHE
-- ============================================================================
-- Add missing columns if they don't exist

DO $$
BEGIN
    -- Add 52-week high/low columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'market_data_cache' AND column_name = 'high_52w') THEN
        ALTER TABLE market_data_cache ADD COLUMN high_52w NUMERIC(15, 4);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'market_data_cache' AND column_name = 'low_52w') THEN
        ALTER TABLE market_data_cache ADD COLUMN low_52w NUMERIC(15, 4);
    END IF;
    
    -- Add pre/post market prices
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'market_data_cache' AND column_name = 'pre_market_price') THEN
        ALTER TABLE market_data_cache ADD COLUMN pre_market_price NUMERIC(15, 4);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'market_data_cache' AND column_name = 'post_market_price') THEN
        ALTER TABLE market_data_cache ADD COLUMN post_market_price NUMERIC(15, 4);
    END IF;
    
    -- Add average volume
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                   WHERE table_name = 'market_data_cache' AND column_name = 'avg_volume') THEN
        ALTER TABLE market_data_cache ADD COLUMN avg_volume BIGINT;
    END IF;
END $$;

-- ============================================================================
-- UPDATE TRIGGERS
-- ============================================================================

CREATE OR REPLACE FUNCTION update_company_profiles_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_company_profiles ON company_profiles;
CREATE TRIGGER trigger_update_company_profiles
    BEFORE UPDATE ON company_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_company_profiles_timestamp();

-- Grant permissions
GRANT ALL PRIVILEGES ON company_profiles TO cift_user;
GRANT ALL PRIVILEGES ON earnings_calendar TO cift_user;
GRANT ALL PRIVILEGES ON chart_patterns TO cift_user;
GRANT ALL PRIVILEGES ON support_resistance_levels TO cift_user;
GRANT ALL PRIVILEGES ON company_news TO cift_user;
GRANT ALL PRIVILEGES ON analyst_ratings TO cift_user;
