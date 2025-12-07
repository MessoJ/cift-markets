-- CIFT Markets - PostgreSQL Initialization
-- Database schema for user management, configurations, and metadata

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL,
    name VARCHAR(200),
    description TEXT,
    scopes TEXT[] DEFAULT ARRAY['read'],
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    rate_limit_per_day INTEGER DEFAULT 10000,
    last_used_at TIMESTAMP WITH TIME ZONE,
    last_used_ip INET,
    total_requests BIGINT DEFAULT 0,
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);

-- Trading accounts table
CREATE TABLE IF NOT EXISTS trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    broker VARCHAR(50) NOT NULL,
    account_id VARCHAR(100) NOT NULL,
    account_type VARCHAR(50) NOT NULL, -- paper, live
    credentials_encrypted TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, broker, account_id)
);

-- Model configurations table
CREATE TABLE IF NOT EXISTS model_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Trading strategies table
CREATE TABLE IF NOT EXISTS trading_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Backtests table
CREATE TABLE IF NOT EXISTS backtests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    strategy_id UUID REFERENCES trading_strategies(id) ON DELETE SET NULL,
    name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital NUMERIC(15, 2) NOT NULL,
    symbols TEXT[] NOT NULL,
    config JSONB NOT NULL,
    results JSONB,
    status VARCHAR(50) DEFAULT 'pending', -- pending, running, completed, failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL, -- drawdown, accuracy_drop, service_down, etc.
    severity VARCHAR(20) NOT NULL, -- info, warning, critical
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_trading_accounts_user_id ON trading_accounts(user_id);
CREATE INDEX IF NOT EXISTS idx_model_configs_user_id ON model_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_strategies_user_id ON trading_strategies(user_id);
CREATE INDEX IF NOT EXISTS idx_backtests_user_id ON backtests(user_id);
CREATE INDEX IF NOT EXISTS idx_backtests_status ON backtests(status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at trigger to relevant tables
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_accounts_updated_at
    BEFORE UPDATE ON trading_accounts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_configs_updated_at
    BEFORE UPDATE ON model_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_strategies_updated_at
    BEFORE UPDATE ON trading_strategies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_backtests_updated_at
    BEFORE UPDATE ON backtests
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create default admin user (password: admin - CHANGE IN PRODUCTION!)
-- Password hash is bcrypt hash of "admin"
INSERT INTO users (email, username, hashed_password, full_name, is_superuser)
VALUES (
    'admin@ciftmarkets.com',
    'admin',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5ztP6oBKdvvK.',
    'CIFT Admin',
    TRUE
)
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- TRADING TABLES (Phase 1)
-- ============================================================================

-- Accounts table - User trading accounts with balances
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_number VARCHAR(50) UNIQUE NOT NULL,
    account_type VARCHAR(20) NOT NULL CHECK (account_type IN ('cash', 'margin', 'paper')),
    
    -- Balances
    cash NUMERIC(15, 2) NOT NULL DEFAULT 0,
    buying_power NUMERIC(15, 2) NOT NULL DEFAULT 0,
    portfolio_value NUMERIC(15, 2) NOT NULL DEFAULT 0,
    equity NUMERIC(15, 2) NOT NULL DEFAULT 0,
    
    -- Margin (if applicable)
    margin_used NUMERIC(15, 2) DEFAULT 0,
    maintenance_margin NUMERIC(15, 2) DEFAULT 0,
    
    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'closed')),
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(user_id, account_number)
);

-- Orders table - All order submissions and their lifecycle
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    
    -- Order details
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    time_in_force VARCHAR(10) NOT NULL DEFAULT 'day' CHECK (time_in_force IN ('day', 'gtc', 'ioc', 'fok', 'opg', 'cls')),
    
    -- Quantities and prices
    quantity NUMERIC(15, 4) NOT NULL CHECK (quantity > 0),
    filled_quantity NUMERIC(15, 4) NOT NULL DEFAULT 0,
    remaining_quantity NUMERIC(15, 4) NOT NULL,
    limit_price NUMERIC(15, 4),
    stop_price NUMERIC(15, 4),
    avg_fill_price NUMERIC(15, 4),
    
    -- Status tracking
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'accepted', 'partial', 'filled', 'cancelled', 'rejected', 'expired'
    )),
    
    -- External references
    broker_order_id VARCHAR(100),
    client_order_id VARCHAR(100),
    
    -- Financial
    total_value NUMERIC(15, 2),
    commission NUMERIC(10, 4) DEFAULT 0,
    
    -- Risk and validation
    rejected_reason TEXT,
    notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    submitted_at TIMESTAMP WITH TIME ZONE,
    accepted_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Order fills table - Individual executions of orders
CREATE TABLE IF NOT EXISTS order_fills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    
    -- Fill details
    fill_quantity NUMERIC(15, 4) NOT NULL CHECK (fill_quantity > 0),
    fill_price NUMERIC(15, 4) NOT NULL CHECK (fill_price > 0),
    fill_value NUMERIC(15, 2) NOT NULL,
    commission NUMERIC(10, 4) NOT NULL DEFAULT 0,
    
    -- External reference
    broker_fill_id VARCHAR(100),
    execution_venue VARCHAR(50),
    
    -- Timestamp
    filled_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    liquidity_flag VARCHAR(10) CHECK (liquidity_flag IN ('added', 'removed', 'unknown'))
);

-- Positions table - Current holdings for each account/symbol
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    
    -- Position details
    quantity NUMERIC(15, 4) NOT NULL,
    side VARCHAR(5) NOT NULL CHECK (side IN ('long', 'short')),
    
    -- Cost basis
    avg_cost NUMERIC(15, 4) NOT NULL,
    total_cost NUMERIC(15, 2) NOT NULL,
    
    -- Current valuation
    current_price NUMERIC(15, 4),
    market_value NUMERIC(15, 2),
    
    -- P&L tracking
    unrealized_pnl NUMERIC(15, 2) DEFAULT 0,
    unrealized_pnl_pct NUMERIC(10, 4) DEFAULT 0,
    realized_pnl NUMERIC(15, 2) DEFAULT 0,
    total_pnl NUMERIC(15, 2) DEFAULT 0,
    
    -- Day metrics
    day_pnl NUMERIC(15, 2) DEFAULT 0,
    day_pnl_pct NUMERIC(10, 4) DEFAULT 0,
    
    -- Timestamps
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(account_id, symbol)
);

-- Position history table - Closed positions for performance tracking
CREATE TABLE IF NOT EXISTS position_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    
    -- Position details
    quantity NUMERIC(15, 4) NOT NULL,
    side VARCHAR(5) NOT NULL,
    
    -- Entry/Exit
    avg_entry_price NUMERIC(15, 4) NOT NULL,
    avg_exit_price NUMERIC(15, 4) NOT NULL,
    total_cost NUMERIC(15, 2) NOT NULL,
    total_proceeds NUMERIC(15, 2) NOT NULL,
    
    -- P&L
    realized_pnl NUMERIC(15, 2) NOT NULL,
    realized_pnl_pct NUMERIC(10, 4) NOT NULL,
    commission_paid NUMERIC(10, 4) NOT NULL DEFAULT 0,
    
    -- Duration
    opened_at TIMESTAMP WITH TIME ZONE NOT NULL,
    closed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    hold_duration_seconds INTEGER,
    
    -- Performance metrics
    max_favorable_excursion NUMERIC(15, 2),
    max_adverse_excursion NUMERIC(15, 2),
    
    -- Metadata
    closing_reason VARCHAR(50),
    notes TEXT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table - All account cash movements
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    
    -- Transaction details
    transaction_type VARCHAR(30) NOT NULL CHECK (transaction_type IN (
        'deposit', 'withdrawal', 'trade', 'dividend', 'interest', 
        'fee', 'commission', 'adjustment'
    )),
    
    -- Amount (positive for credits, negative for debits)
    amount NUMERIC(15, 2) NOT NULL,
    balance_after NUMERIC(15, 2) NOT NULL,
    
    -- Related entities
    order_id UUID REFERENCES orders(id),
    symbol VARCHAR(10),
    
    -- Description
    description TEXT NOT NULL,
    external_ref VARCHAR(100),
    
    -- Timestamp
    transaction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Market data cache table - Latest prices for quick access
CREATE TABLE IF NOT EXISTS market_data_cache (
    symbol VARCHAR(10) PRIMARY KEY,
    
    -- Price data
    price NUMERIC(15, 4) NOT NULL,
    bid NUMERIC(15, 4),
    ask NUMERIC(15, 4),
    bid_size INTEGER,
    ask_size INTEGER,
    
    -- Volume and trades
    volume BIGINT,
    trade_count INTEGER,
    
    -- Daily metrics
    open NUMERIC(15, 4),
    high NUMERIC(15, 4),
    low NUMERIC(15, 4),
    close NUMERIC(15, 4),
    prev_close NUMERIC(15, 4),
    
    -- Changes
    change NUMERIC(15, 4),
    change_pct NUMERIC(10, 4),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',
    exchange VARCHAR(20),
    
    -- Timestamps
    last_trade_time TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR TRADING TABLES
-- ============================================================================

-- Accounts indexes
CREATE INDEX IF NOT EXISTS idx_accounts_user_id ON accounts(user_id);
CREATE INDEX IF NOT EXISTS idx_accounts_status ON accounts(status);
CREATE INDEX IF NOT EXISTS idx_accounts_account_type ON accounts(account_type);

-- Orders indexes (critical for performance)
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_account_id ON orders(account_id);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON orders(broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_user_symbol_status ON orders(user_id, symbol, status);

-- Order fills indexes
CREATE INDEX IF NOT EXISTS idx_order_fills_order_id ON order_fills(order_id);
CREATE INDEX IF NOT EXISTS idx_order_fills_filled_at ON order_fills(filled_at);

-- Positions indexes
CREATE INDEX IF NOT EXISTS idx_positions_user_id ON positions(user_id);
CREATE INDEX IF NOT EXISTS idx_positions_account_id ON positions(account_id);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_account_symbol ON positions(account_id, symbol);

-- Position history indexes
CREATE INDEX IF NOT EXISTS idx_position_history_user_id ON position_history(user_id);
CREATE INDEX IF NOT EXISTS idx_position_history_account_id ON position_history(account_id);
CREATE INDEX IF NOT EXISTS idx_position_history_symbol ON position_history(symbol);
CREATE INDEX IF NOT EXISTS idx_position_history_closed_at ON position_history(closed_at);

-- Transactions indexes
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_account_id ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_order_id ON transactions(order_id);

-- Market data cache indexes
CREATE INDEX IF NOT EXISTS idx_market_data_cache_updated_at ON market_data_cache(updated_at);
CREATE INDEX IF NOT EXISTS idx_market_data_cache_status ON market_data_cache(status);

-- ============================================================================
-- TRIGGERS FOR TRADING TABLES
-- ============================================================================

CREATE TRIGGER update_accounts_updated_at
    BEFORE UPDATE ON accounts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- FUNCTIONS FOR TRADING LOGIC
-- ============================================================================

-- Function to calculate position P&L
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    p_quantity NUMERIC,
    p_avg_cost NUMERIC,
    p_current_price NUMERIC
) RETURNS TABLE (
    unrealized_pnl NUMERIC,
    unrealized_pnl_pct NUMERIC,
    market_value NUMERIC
) AS $$
BEGIN
    RETURN QUERY SELECT
        (p_quantity * p_current_price) - (p_quantity * p_avg_cost) AS unrealized_pnl,
        CASE 
            WHEN p_avg_cost > 0 THEN
                ((p_current_price - p_avg_cost) / p_avg_cost) * 100
            ELSE 0
        END AS unrealized_pnl_pct,
        p_quantity * p_current_price AS market_value;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to update account balances after order fill
CREATE OR REPLACE FUNCTION update_account_after_fill()
RETURNS TRIGGER AS $$
DECLARE
    v_order RECORD;
    v_account RECORD;
BEGIN
    -- Get order details
    SELECT * INTO v_order FROM orders WHERE id = NEW.order_id;
    
    -- Get account details
    SELECT * INTO v_account FROM accounts WHERE id = v_order.account_id;
    
    -- Update account based on order side
    IF v_order.side = 'buy' THEN
        -- Deduct cash for buy orders
        UPDATE accounts
        SET 
            cash = cash - (NEW.fill_value + NEW.commission),
            updated_at = CURRENT_TIMESTAMP,
            last_activity_at = CURRENT_TIMESTAMP
        WHERE id = v_order.account_id;
    ELSE
        -- Add cash for sell orders
        UPDATE accounts
        SET 
            cash = cash + (NEW.fill_value - NEW.commission),
            updated_at = CURRENT_TIMESTAMP,
            last_activity_at = CURRENT_TIMESTAMP
        WHERE id = v_order.account_id;
    END IF;
    
    -- Record transaction
    INSERT INTO transactions (
        user_id, account_id, transaction_type, amount, 
        balance_after, order_id, symbol, description
    )
    SELECT
        v_order.user_id,
        v_order.account_id,
        'trade',
        CASE 
            WHEN v_order.side = 'buy' THEN -(NEW.fill_value + NEW.commission)
            ELSE (NEW.fill_value - NEW.commission)
        END,
        (SELECT cash FROM accounts WHERE id = v_order.account_id),
        NEW.order_id,
        v_order.symbol,
        format('Order fill: %s %s shares at $%s', v_order.side, NEW.fill_quantity, NEW.fill_price);
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update account after order fill
CREATE TRIGGER trigger_update_account_after_fill
    AFTER INSERT ON order_fills
    FOR EACH ROW
    EXECUTE FUNCTION update_account_after_fill();

-- ============================================================================
-- SEED DATA FOR TESTING
-- ============================================================================

-- Create default test account for admin user
INSERT INTO accounts (user_id, account_number, account_type, cash, buying_power, portfolio_value, equity)
SELECT 
    id,
    'CIFT-' || SUBSTRING(id::text, 1, 8),
    'paper',
    100000.00,
    100000.00,
    100000.00,
    100000.00
FROM users WHERE email = 'admin@ciftmarkets.com'
ON CONFLICT (user_id, account_number) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cift_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cift_user;
