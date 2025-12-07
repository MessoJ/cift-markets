-- ============================================================================
-- SYMBOLS TABLE MIGRATION
-- Create symbols/stocks master table for screener and market data
-- ============================================================================

-- ============================================================================
-- 1. SYMBOLS TABLE (Master list of tradable securities)
-- ============================================================================

CREATE TABLE IF NOT EXISTS symbols (
    -- Primary key
    symbol VARCHAR(20) PRIMARY KEY,
    
    -- Basic information
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Classification
    asset_type VARCHAR(50) CHECK (asset_type IN ('stock', 'etf', 'index', 'crypto', 'forex', 'commodity')),
    sector VARCHAR(100),
    industry VARCHAR(100),
    
    -- Exchange information
    exchange VARCHAR(50),
    currency VARCHAR(10) DEFAULT 'USD',
    country VARCHAR(3),  -- ISO country code
    
    -- Trading status
    is_tradable BOOLEAN DEFAULT true,
    is_active BOOLEAN DEFAULT true,
    
    -- Fundamental data (updated periodically)
    market_cap DECIMAL(20, 2),
    shares_outstanding BIGINT,
    float_shares BIGINT,
    
    -- Valuation metrics
    pe_ratio DECIMAL(10, 2),
    forward_pe DECIMAL(10, 2),
    peg_ratio DECIMAL(10, 2),
    price_to_book DECIMAL(10, 2),
    price_to_sales DECIMAL(10, 2),
    
    -- Profitability metrics
    eps DECIMAL(15, 4),
    revenue DECIMAL(20, 2),
    net_income DECIMAL(20, 2),
    ebitda DECIMAL(20, 2),
    
    -- Returns metrics
    profit_margin DECIMAL(10, 4),
    operating_margin DECIMAL(10, 4),
    roe DECIMAL(10, 4),  -- Return on Equity
    roa DECIMAL(10, 4),  -- Return on Assets
    
    -- Dividend information
    dividend_yield DECIMAL(10, 4),
    dividend_per_share DECIMAL(15, 4),
    payout_ratio DECIMAL(10, 4),
    ex_dividend_date DATE,
    
    -- Analyst data
    analyst_rating VARCHAR(20),  -- Strong Buy, Buy, Hold, Sell, Strong Sell
    analyst_target_price DECIMAL(15, 4),
    analyst_count INTEGER,
    
    -- IPO information
    ipo_date DATE,
    
    -- Timestamps
    data_updated_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- 2. INDEXES
-- ============================================================================

-- Performance indexes for screener queries
CREATE INDEX idx_symbols_tradable ON symbols(is_tradable, is_active) WHERE is_tradable = true AND is_active = true;
CREATE INDEX idx_symbols_sector ON symbols(sector) WHERE is_tradable = true;
CREATE INDEX idx_symbols_industry ON symbols(industry) WHERE is_tradable = true;
CREATE INDEX idx_symbols_exchange ON symbols(exchange);
CREATE INDEX idx_symbols_asset_type ON symbols(asset_type);

-- Fundamental screening indexes
CREATE INDEX idx_symbols_market_cap ON symbols(market_cap) WHERE is_tradable = true;
CREATE INDEX idx_symbols_pe_ratio ON symbols(pe_ratio) WHERE is_tradable = true AND pe_ratio > 0;
CREATE INDEX idx_symbols_dividend_yield ON symbols(dividend_yield) WHERE is_tradable = true AND dividend_yield > 0;

-- Composite indexes for common screener patterns
CREATE INDEX idx_symbols_sector_market_cap ON symbols(sector, market_cap DESC) WHERE is_tradable = true;
CREATE INDEX idx_symbols_industry_pe ON symbols(industry, pe_ratio) WHERE is_tradable = true;

-- ============================================================================
-- 3. UPDATE TRIGGER
-- ============================================================================

CREATE OR REPLACE FUNCTION update_symbols_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_symbols_updated_at
    BEFORE UPDATE ON symbols
    FOR EACH ROW
    EXECUTE FUNCTION update_symbols_updated_at();

-- ============================================================================
-- 4. SEED DATA (Sample stocks for testing)
-- ============================================================================

INSERT INTO symbols (
    symbol, name, asset_type, sector, industry, exchange, 
    market_cap, pe_ratio, eps, dividend_yield, is_tradable, is_active
) VALUES
    -- Technology
    ('AAPL', 'Apple Inc.', 'stock', 'Technology', 'Consumer Electronics', 'NASDAQ', 3000000000000, 28.5, 6.15, 0.0051, true, true),
    ('MSFT', 'Microsoft Corporation', 'stock', 'Technology', 'Software', 'NASDAQ', 2800000000000, 35.2, 9.72, 0.0078, true, true),
    ('GOOGL', 'Alphabet Inc.', 'stock', 'Technology', 'Internet Services', 'NASDAQ', 1900000000000, 26.8, 5.61, 0.00, true, true),
    ('NVDA', 'NVIDIA Corporation', 'stock', 'Technology', 'Semiconductors', 'NASDAQ', 1200000000000, 75.3, 1.29, 0.0003, true, true),
    ('META', 'Meta Platforms Inc.', 'stock', 'Technology', 'Social Media', 'NASDAQ', 950000000000, 30.1, 14.87, 0.00, true, true),
    
    -- Healthcare
    ('JNJ', 'Johnson & Johnson', 'stock', 'Healthcare', 'Pharmaceuticals', 'NYSE', 400000000000, 16.8, 9.63, 0.0285, true, true),
    ('UNH', 'UnitedHealth Group', 'stock', 'Healthcare', 'Healthcare Plans', 'NYSE', 485000000000, 24.5, 22.65, 0.0134, true, true),
    ('PFE', 'Pfizer Inc.', 'stock', 'Healthcare', 'Pharmaceuticals', 'NYSE', 160000000000, 9.2, 3.01, 0.0586, true, true),
    
    -- Financial
    ('JPM', 'JPMorgan Chase & Co.', 'stock', 'Financial', 'Banking', 'NYSE', 450000000000, 11.5, 15.92, 0.0247, true, true),
    ('BAC', 'Bank of America', 'stock', 'Financial', 'Banking', 'NYSE', 270000000000, 10.2, 3.34, 0.0264, true, true),
    ('V', 'Visa Inc.', 'stock', 'Financial', 'Payment Processing', 'NYSE', 520000000000, 32.6, 7.84, 0.0078, true, true),
    
    -- Consumer
    ('AMZN', 'Amazon.com Inc.', 'stock', 'Consumer', 'E-commerce', 'NASDAQ', 1500000000000, 65.4, 2.90, 0.00, true, true),
    ('TSLA', 'Tesla Inc.', 'stock', 'Consumer', 'Automotive', 'NASDAQ', 800000000000, 85.2, 4.07, 0.00, true, true),
    ('WMT', 'Walmart Inc.', 'stock', 'Consumer', 'Retail', 'NYSE', 420000000000, 28.3, 6.29, 0.0142, true, true),
    ('HD', 'The Home Depot', 'stock', 'Consumer', 'Retail', 'NYSE', 350000000000, 23.8, 13.68, 0.0243, true, true),
    
    -- Energy
    ('XOM', 'Exxon Mobil Corporation', 'stock', 'Energy', 'Oil & Gas', 'NYSE', 430000000000, 9.8, 10.75, 0.0342, true, true),
    ('CVX', 'Chevron Corporation', 'stock', 'Energy', 'Oil & Gas', 'NYSE', 290000000000, 11.2, 13.26, 0.0356, true, true),
    
    -- Industrial
    ('BA', 'The Boeing Company', 'stock', 'Industrial', 'Aerospace', 'NYSE', 130000000000, -5.2, -8.74, 0.00, true, true),
    ('CAT', 'Caterpillar Inc.', 'stock', 'Industrial', 'Machinery', 'NYSE', 160000000000, 15.4, 17.21, 0.0198, true, true),
    
    -- Materials
    ('LIN', 'Linde plc', 'stock', 'Materials', 'Chemicals', 'NYSE', 210000000000, 32.1, 13.47, 0.0128, true, true),
    
    -- ETFs
    ('SPY', 'SPDR S&P 500 ETF Trust', 'etf', 'Index', 'Broad Market', 'NYSE', 450000000000, NULL, NULL, 0.0125, true, true),
    ('QQQ', 'Invesco QQQ Trust', 'etf', 'Index', 'Technology', 'NASDAQ', 220000000000, NULL, NULL, 0.0048, true, true),
    ('IWM', 'iShares Russell 2000 ETF', 'etf', 'Index', 'Small Cap', 'NYSE', 65000000000, NULL, NULL, 0.0102, true, true)
ON CONFLICT (symbol) DO NOTHING;

-- ============================================================================
-- 5. COMMENTS
-- ============================================================================

COMMENT ON TABLE symbols IS 'Master table of tradable securities with fundamental data for screening';
COMMENT ON COLUMN symbols.symbol IS 'Unique ticker symbol';
COMMENT ON COLUMN symbols.market_cap IS 'Market capitalization in USD';
COMMENT ON COLUMN symbols.pe_ratio IS 'Price to earnings ratio (TTM)';
COMMENT ON COLUMN symbols.dividend_yield IS 'Annual dividend yield as decimal (e.g., 0.025 = 2.5%)';
COMMENT ON COLUMN symbols.is_tradable IS 'Whether the symbol can be traded';
COMMENT ON COLUMN symbols.data_updated_at IS 'Last time fundamental data was updated';
