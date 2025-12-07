-- CIFT Markets - Drilldown Support Migration
-- Adds critical tables and columns for frontend data drilldowns
-- Date: 2025-11-09

-- ============================================================================
-- 1. PORTFOLIO SNAPSHOTS (CRITICAL - Time-series analytics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    
    -- Portfolio values
    total_value NUMERIC(15, 2) NOT NULL,
    cash NUMERIC(15, 2) NOT NULL,
    positions_value NUMERIC(15, 2) NOT NULL,
    equity NUMERIC(15, 2) NOT NULL,
    
    -- P&L tracking
    unrealized_pnl NUMERIC(15, 2) DEFAULT 0,
    realized_pnl NUMERIC(15, 2) DEFAULT 0,
    day_pnl NUMERIC(15, 2) DEFAULT 0,
    day_pnl_pct NUMERIC(10, 4) DEFAULT 0,
    
    -- Period metrics (for quick lookups)
    week_pnl NUMERIC(15, 2),
    month_pnl NUMERIC(15, 2),
    year_pnl NUMERIC(15, 2),
    
    -- Snapshot metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    snapshot_type VARCHAR(20) NOT NULL DEFAULT 'eod' CHECK (snapshot_type IN ('eod', 'intraday', 'realtime')),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast time-series queries
CREATE INDEX idx_portfolio_snapshots_user_timestamp ON portfolio_snapshots(user_id, timestamp DESC);
CREATE INDEX idx_portfolio_snapshots_account_timestamp ON portfolio_snapshots(account_id, timestamp DESC);
CREATE INDEX idx_portfolio_snapshots_type ON portfolio_snapshots(snapshot_type);
CREATE INDEX idx_portfolio_snapshots_timestamp ON portfolio_snapshots(timestamp DESC);

-- ============================================================================
-- 2. POSITION LOTS (Cost basis tracking - FIFO/LIFO)
-- ============================================================================

CREATE TABLE IF NOT EXISTS position_lots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    
    -- Lot details
    quantity NUMERIC(15, 4) NOT NULL CHECK (quantity > 0),
    remaining_quantity NUMERIC(15, 4) NOT NULL CHECK (remaining_quantity >= 0),
    purchase_price NUMERIC(15, 4) NOT NULL,
    purchase_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Cost basis method
    lot_method VARCHAR(10) NOT NULL DEFAULT 'FIFO' CHECK (lot_method IN ('FIFO', 'LIFO', 'AvgCost', 'SpecID')),
    
    -- Related order
    entry_order_id UUID REFERENCES orders(id),
    
    -- Lot status
    is_closed BOOLEAN DEFAULT FALSE,
    closed_date TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_position_lots_position_id ON position_lots(position_id);
CREATE INDEX idx_position_lots_user_symbol ON position_lots(user_id, symbol);
CREATE INDEX idx_position_lots_purchase_date ON position_lots(purchase_date);
CREATE INDEX idx_position_lots_is_closed ON position_lots(is_closed);

-- ============================================================================
-- 3. POSITION SNAPSHOTS (P&L over time per position)
-- ============================================================================

CREATE TABLE IF NOT EXISTS position_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    
    -- Position state at snapshot
    quantity NUMERIC(15, 4) NOT NULL,
    avg_cost NUMERIC(15, 4) NOT NULL,
    current_price NUMERIC(15, 4) NOT NULL,
    market_value NUMERIC(15, 2) NOT NULL,
    
    -- P&L at snapshot
    unrealized_pnl NUMERIC(15, 2) NOT NULL,
    unrealized_pnl_pct NUMERIC(10, 4) NOT NULL,
    day_pnl NUMERIC(15, 2),
    
    -- Timestamp
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_position_snapshots_position_timestamp ON position_snapshots(position_id, timestamp DESC);
CREATE INDEX idx_position_snapshots_user_symbol_timestamp ON position_snapshots(user_id, symbol, timestamp DESC);
CREATE INDEX idx_position_snapshots_timestamp ON position_snapshots(timestamp DESC);

-- ============================================================================
-- 4. WATCHLISTS (Saved symbol lists)
-- ============================================================================

CREATE TABLE IF NOT EXISTS watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Watchlist details
    name VARCHAR(100) NOT NULL,
    description TEXT,
    symbols TEXT[] NOT NULL DEFAULT '{}',
    
    -- Settings
    is_default BOOLEAN DEFAULT FALSE,
    is_public BOOLEAN DEFAULT FALSE,
    sort_order INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, name)
);

CREATE INDEX idx_watchlists_user_id ON watchlists(user_id);
CREATE INDEX idx_watchlists_is_default ON watchlists(is_default);

-- ============================================================================
-- 5. EXECUTION STATS (Execution quality aggregations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS execution_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    
    -- Aggregation dimensions
    symbol VARCHAR(10),
    venue VARCHAR(50),
    date DATE NOT NULL,
    hour INTEGER CHECK (hour BETWEEN 0 AND 23),
    
    -- Order statistics
    total_orders INTEGER DEFAULT 0,
    filled_orders INTEGER DEFAULT 0,
    partial_fills INTEGER DEFAULT 0,
    cancelled_orders INTEGER DEFAULT 0,
    rejected_orders INTEGER DEFAULT 0,
    
    -- Execution metrics
    total_volume NUMERIC(15, 4) DEFAULT 0,
    total_value NUMERIC(15, 2) DEFAULT 0,
    avg_fill_price NUMERIC(15, 4),
    avg_slippage_bps NUMERIC(10, 4),
    avg_execution_time_ms INTEGER,
    
    -- Liquidity
    maker_fills INTEGER DEFAULT 0,
    taker_fills INTEGER DEFAULT 0,
    maker_volume NUMERIC(15, 4) DEFAULT 0,
    taker_volume NUMERIC(15, 4) DEFAULT 0,
    
    -- Commission
    total_commission NUMERIC(10, 4) DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_execution_stats_user_date ON execution_stats(user_id, date DESC);
CREATE INDEX idx_execution_stats_symbol_date ON execution_stats(symbol, date DESC);
CREATE INDEX idx_execution_stats_venue_date ON execution_stats(venue, date DESC);
CREATE INDEX idx_execution_stats_date_hour ON execution_stats(date, hour);

-- ============================================================================
-- 6. ADD MISSING COLUMNS TO EXISTING TABLES
-- ============================================================================

-- Orders table enhancements
ALTER TABLE orders 
    ADD COLUMN IF NOT EXISTS strategy_id UUID REFERENCES trading_strategies(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS execution_latency_ms INTEGER,
    ADD COLUMN IF NOT EXISTS slippage_bps NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS vwap_price NUMERIC(15, 4),
    ADD COLUMN IF NOT EXISTS market_price_at_submission NUMERIC(15, 4),
    ADD COLUMN IF NOT EXISTS parent_order_id UUID REFERENCES orders(id);

-- Order fills enhancements
ALTER TABLE order_fills
    ADD COLUMN IF NOT EXISTS submission_latency_ms INTEGER,
    ADD COLUMN IF NOT EXISTS execution_latency_ms INTEGER,
    ADD COLUMN IF NOT EXISTS slippage_bps NUMERIC(10, 4),
    ADD COLUMN IF NOT EXISTS market_price_at_submission NUMERIC(15, 4),
    ADD COLUMN IF NOT EXISTS maker_taker_flag VARCHAR(10) CHECK (maker_taker_flag IN ('maker', 'taker', 'unknown'));

-- Position history enhancements
ALTER TABLE position_history
    ADD COLUMN IF NOT EXISTS strategy_id UUID REFERENCES trading_strategies(id),
    ADD COLUMN IF NOT EXISTS tags TEXT[],
    ADD COLUMN IF NOT EXISTS notes TEXT,
    ADD COLUMN IF NOT EXISTS screenshots JSONB,
    ADD COLUMN IF NOT EXISTS entry_orders JSONB,
    ADD COLUMN IF NOT EXISTS exit_orders JSONB;

-- Trading strategies enhancements
ALTER TABLE trading_strategies
    ADD COLUMN IF NOT EXISTS performance_metrics JSONB,
    ADD COLUMN IF NOT EXISTS total_trades INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS winning_trades INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_pnl NUMERIC(15, 2) DEFAULT 0;

-- ============================================================================
-- 7. INDEXES FOR NEW COLUMNS
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_orders_strategy_id ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_parent_order_id ON orders(parent_order_id);
CREATE INDEX IF NOT EXISTS idx_position_history_strategy_id ON position_history(strategy_id);

-- ============================================================================
-- 8. TRIGGERS FOR NEW TABLES
-- ============================================================================

CREATE TRIGGER update_watchlists_updated_at
    BEFORE UPDATE ON watchlists
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_position_lots_updated_at
    BEFORE UPDATE ON position_lots
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_execution_stats_updated_at
    BEFORE UPDATE ON execution_stats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 9. SEED DEFAULT WATCHLIST
-- ============================================================================

-- Create default watchlist for each user
INSERT INTO watchlists (user_id, name, symbols, is_default)
SELECT 
    id,
    'My Watchlist',
    ARRAY['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']::TEXT[],
    TRUE
FROM users
ON CONFLICT (user_id, name) DO NOTHING;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE portfolio_snapshots TO cift_user;
GRANT ALL PRIVILEGES ON TABLE position_lots TO cift_user;
GRANT ALL PRIVILEGES ON TABLE position_snapshots TO cift_user;
GRANT ALL PRIVILEGES ON TABLE watchlists TO cift_user;
GRANT ALL PRIVILEGES ON TABLE execution_stats TO cift_user;
