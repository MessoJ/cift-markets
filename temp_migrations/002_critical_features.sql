-- ============================================================================
-- CRITICAL FEATURES DATABASE MIGRATION
-- Adds tables for: Funding, KYC/Onboarding, Support, News, Screener, 
-- Statements, and Alerts functionality
-- ============================================================================

-- ============================================================================
-- FUNDING TABLES
-- ============================================================================

-- Payment methods table
CREATE TABLE IF NOT EXISTS payment_methods (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50) NOT NULL CHECK (type IN ('bank_account', 'debit_card', 'wire')),
    name VARCHAR(200) NOT NULL,
    last_four VARCHAR(4) NOT NULL,
    account_number_encrypted TEXT,  -- TODO: Encrypt in production
    routing_number_encrypted TEXT,  -- TODO: Encrypt in production
    is_verified BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_payment_methods_user ON payment_methods(user_id) WHERE is_active = true;

-- Funding transactions table
CREATE TABLE IF NOT EXISTS funding_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('deposit', 'withdrawal')),
    method VARCHAR(50) NOT NULL,
    amount DECIMAL(20, 2) NOT NULL,
    fee DECIMAL(20, 2) DEFAULT 0,
    status VARCHAR(20) NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    payment_method_id UUID REFERENCES payment_methods(id),
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    expected_arrival TIMESTAMP,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_funding_txns_user ON funding_transactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_funding_txns_status ON funding_transactions(status) WHERE status IN ('pending', 'processing');

-- User transfer limits table
CREATE TABLE IF NOT EXISTS user_transfer_limits (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    daily_deposit_limit DECIMAL(20, 2) DEFAULT 25000.00,
    daily_withdrawal_limit DECIMAL(20, 2) DEFAULT 25000.00,
    instant_transfer_limit DECIMAL(20, 2) DEFAULT 1000.00,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- KYC/ONBOARDING TABLES
-- ============================================================================

-- KYC profiles table
CREATE TABLE IF NOT EXISTS kyc_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL DEFAULT 'incomplete' CHECK (status IN ('incomplete', 'pending', 'approved', 'rejected')),
    
    -- Personal info
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    middle_name VARCHAR(100),
    date_of_birth DATE,
    ssn_encrypted TEXT,  -- TODO: Encrypt in production
    ssn_last_four VARCHAR(4),
    phone_number VARCHAR(20),
    
    -- Address
    street_address VARCHAR(200),
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    country VARCHAR(50) DEFAULT 'USA',
    
    -- Employment
    employment_status VARCHAR(50),
    employer_name VARCHAR(200),
    occupation VARCHAR(100),
    annual_income VARCHAR(50),
    net_worth VARCHAR(50),
    
    -- Trading experience
    trading_experience VARCHAR(50),
    investment_objectives TEXT[],
    risk_tolerance VARCHAR(50),
    
    -- Documents
    identity_document_uploaded BOOLEAN DEFAULT false,
    address_proof_uploaded BOOLEAN DEFAULT false,
    
    -- Agreements
    terms_accepted BOOLEAN DEFAULT false,
    privacy_accepted BOOLEAN DEFAULT false,
    risk_disclosure_accepted BOOLEAN DEFAULT false,
    
    -- Tracking
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP,
    reviewer_notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_kyc_status ON kyc_profiles(status);

-- KYC documents table
CREATE TABLE IF NOT EXISTS kyc_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_type VARCHAR(50) NOT NULL CHECK (document_type IN ('identity', 'address_proof', 'other')),
    file_name VARCHAR(255) NOT NULL,
    file_size INTEGER NOT NULL,
    file_content BYTEA,  -- Store in S3/blob storage in production
    mime_type VARCHAR(100),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'verified', 'rejected')),
    uploaded_at TIMESTAMP DEFAULT NOW(),
    verified_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_kyc_docs_user ON kyc_documents(user_id, uploaded_at DESC);

-- ============================================================================
-- SCREENER TABLES
-- ============================================================================

-- Saved screening criteria
CREATE TABLE IF NOT EXISTS saved_screens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    criteria JSONB NOT NULL,
    sort_by VARCHAR(50) DEFAULT 'market_cap',
    sort_order VARCHAR(10) DEFAULT 'desc' CHECK (sort_order IN ('asc', 'desc')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_run TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_saved_screens_user ON saved_screens(user_id, created_at DESC);

-- Screener results cache for performance
CREATE TABLE IF NOT EXISTS screener_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    criteria_hash VARCHAR(64) NOT NULL,  -- MD5 hash of criteria
    results JSONB NOT NULL,
    result_count INTEGER NOT NULL,
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_screener_cache_hash ON screener_cache(criteria_hash);
CREATE INDEX IF NOT EXISTS idx_screener_cache_expires ON screener_cache(expires_at);

-- ============================================================================
-- PRICE ALERTS TABLES
-- ============================================================================

-- Price alerts
CREATE TABLE IF NOT EXISTS price_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('price_above', 'price_below', 'price_change_percent', 'volume_spike', 'technical_indicator')),
    condition_value DECIMAL(15,6) NOT NULL,
    condition_value2 DECIMAL(15,6),  -- For range alerts
    notification_methods JSONB DEFAULT '["in_app"]'::jsonb,
    message TEXT,
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    triggered_at TIMESTAMP WITH TIME ZONE,
    trigger_price DECIMAL(15,6),
    verification_details JSONB
);

CREATE INDEX IF NOT EXISTS idx_price_alerts_user ON price_alerts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_price_alerts_symbol ON price_alerts(symbol, is_active);
CREATE INDEX IF NOT EXISTS idx_price_alerts_active ON price_alerts(is_active, triggered_at, expires_at);

-- Alert trigger history
CREATE TABLE IF NOT EXISTS alert_triggers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id UUID NOT NULL REFERENCES price_alerts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    trigger_price DECIMAL(15,6) NOT NULL,
    trigger_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    condition_met TEXT NOT NULL,
    notification_methods JSONB,
    message TEXT
);

CREATE INDEX IF NOT EXISTS idx_alert_triggers_user ON alert_triggers(user_id, trigger_time DESC);
CREATE INDEX IF NOT EXISTS idx_alert_triggers_alert ON alert_triggers(alert_id);

-- ============================================================================
-- NEWS TABLES
-- ============================================================================

-- News articles storage
CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    url TEXT UNIQUE NOT NULL,
    source VARCHAR(100) NOT NULL,
    author VARCHAR(100),
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    symbols JSONB DEFAULT '[]'::jsonb,
    categories JSONB DEFAULT '[]'::jsonb,
    sentiment VARCHAR(20) CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    importance INTEGER DEFAULT 1 CHECK (importance >= 1 AND importance <= 5),
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_articles USING GIN(symbols);
CREATE INDEX IF NOT EXISTS idx_news_categories ON news_articles USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_news_source ON news_articles(source);
CREATE INDEX IF NOT EXISTS idx_news_importance ON news_articles(importance DESC, published_at DESC);

-- Market movers snapshots
CREATE TABLE IF NOT EXISTS market_movers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_date DATE NOT NULL,
    mover_type VARCHAR(20) NOT NULL CHECK (mover_type IN ('gainer', 'loser', 'volume')),
    symbol VARCHAR(10) NOT NULL,
    name VARCHAR(100),
    price DECIMAL(15,6),
    change_amount DECIMAL(15,6),
    change_percent DECIMAL(8,4),
    volume BIGINT,
    market_cap DECIMAL(20,2),
    rank_position INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_market_movers_unique ON market_movers(snapshot_date, mover_type, symbol);
CREATE INDEX IF NOT EXISTS idx_market_movers_date ON market_movers(snapshot_date DESC);

-- Enhanced portfolio snapshots with additional analytics fields
ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS positions_count INTEGER DEFAULT 0;
ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS largest_position VARCHAR(10);
ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS largest_position_value DECIMAL(15,6) DEFAULT 0;

-- Performance metrics cache table
CREATE TABLE IF NOT EXISTS portfolio_performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    period_days INTEGER NOT NULL,
    total_return DECIMAL(15,6),
    total_return_pct DECIMAL(8,4),
    annualized_return DECIMAL(8,4),
    volatility DECIMAL(8,4),
    sharpe_ratio DECIMAL(6,4),
    max_drawdown DECIMAL(15,6),
    max_drawdown_pct DECIMAL(8,4),
    win_rate DECIMAL(6,2),
    profit_factor DECIMAL(6,4),
    best_day DECIMAL(8,4),
    worst_day DECIMAL(8,4),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(account_id, period_days)
);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_account ON portfolio_performance_metrics(account_id, period_days);

-- Generated statements and tax documents
CREATE TABLE IF NOT EXISTS generated_statements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_id UUID NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    statement_type VARCHAR(50) NOT NULL CHECK (statement_type IN ('monthly', 'quarterly', 'annual', 'trade_confirmation', 'tax_1099', 'tax_summary')),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    file_data BYTEA,  -- Store PDF data directly or use file path in production
    file_path TEXT,   -- For external storage (S3, etc.)
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    file_size INTEGER DEFAULT 0,
    page_count INTEGER DEFAULT 1,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_generated_statements_user ON generated_statements(user_id, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_generated_statements_account ON generated_statements(account_id, statement_type);
CREATE INDEX IF NOT EXISTS idx_generated_statements_period ON generated_statements(period_start, period_end);

-- Geographic news events for globe visualization
CREATE TABLE IF NOT EXISTS geo_news_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT UNIQUE NOT NULL,
    source VARCHAR(100) NOT NULL,
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    country VARCHAR(100) NOT NULL,
    country_code VARCHAR(3) NOT NULL,
    latitude DECIMAL(10,8) NOT NULL,
    longitude DECIMAL(11,8) NOT NULL,
    impact_level INTEGER DEFAULT 1 CHECK (impact_level >= 1 AND impact_level <= 5),
    categories JSONB DEFAULT '[]'::jsonb,
    symbols_affected JSONB DEFAULT '[]'::jsonb,
    economic_indicators JSONB DEFAULT '[]'::jsonb,
    sentiment_score DECIMAL(4,3) DEFAULT 0 CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_geo_news_published ON geo_news_events(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_geo_news_country ON geo_news_events(country_code, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_geo_news_impact ON geo_news_events(impact_level DESC, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_geo_news_location ON geo_news_events(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_geo_news_symbols ON geo_news_events USING GIN(symbols_affected);

-- ============================================================================
-- SUPPORT TABLES
-- ============================================================================

-- FAQ items table
CREATE TABLE IF NOT EXISTS faq_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    display_order INTEGER DEFAULT 0,
    is_published BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    search_vector TSVECTOR
);

CREATE INDEX IF NOT EXISTS idx_faq_category ON faq_items(category) WHERE is_published = true;
CREATE INDEX IF NOT EXISTS idx_faq_search ON faq_items USING gin(search_vector);

-- Trigger to update search vector
CREATE OR REPLACE FUNCTION update_faq_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.question, '') || ' ' || COALESCE(NEW.answer, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS faq_search_vector_update ON faq_items;
CREATE TRIGGER faq_search_vector_update
BEFORE INSERT OR UPDATE ON faq_items
FOR EACH ROW EXECUTE FUNCTION update_faq_search_vector();

-- Support tickets table
CREATE TABLE IF NOT EXISTS support_tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL CHECK (category IN ('account', 'trading', 'funding', 'technical', 'billing', 'other')),
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'in_progress', 'waiting', 'resolved', 'closed')),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,
    last_message_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tickets_user ON support_tickets(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON support_tickets(status, priority) WHERE status NOT IN ('resolved', 'closed');

-- Support messages table
CREATE TABLE IF NOT EXISTS support_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id UUID NOT NULL REFERENCES support_tickets(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    staff_id UUID,  -- References staff table (not created yet)
    message TEXT NOT NULL,
    is_internal BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_ticket ON support_messages(ticket_id, created_at ASC);

-- ============================================================================
-- NEWS TABLES
-- ============================================================================

-- News articles table
CREATE TABLE IF NOT EXISTS news_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    summary TEXT NOT NULL,
    content TEXT,
    source VARCHAR(100) NOT NULL,
    url TEXT,
    author VARCHAR(200),
    published_at TIMESTAMP NOT NULL,
    category VARCHAR(50),
    sentiment VARCHAR(20) CHECK (sentiment IN ('positive', 'negative', 'neutral')),
    symbols TEXT[],
    image_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_category ON news_articles(category, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_articles USING gin(symbols);

-- Economic events table
CREATE TABLE IF NOT EXISTS economic_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(200) NOT NULL,
    country VARCHAR(50) NOT NULL,
    event_date TIMESTAMP NOT NULL,
    impact VARCHAR(20) CHECK (impact IN ('high', 'medium', 'low')),
    forecast VARCHAR(100),
    previous VARCHAR(100),
    actual VARCHAR(100),
    currency VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_economic_events_date ON economic_events(event_date);

-- Earnings calendar table
CREATE TABLE IF NOT EXISTS earnings_calendar (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(200),
    earnings_date DATE NOT NULL,
    earnings_time VARCHAR(20) CHECK (earnings_time IN ('bmo', 'amc', 'dmh')),  -- before market open, after market close, during market hours
    eps_estimate DECIMAL(10, 4),
    eps_actual DECIMAL(10, 4),
    revenue_estimate DECIMAL(20, 2),
    revenue_actual DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_earnings_date ON earnings_calendar(earnings_date);
CREATE INDEX IF NOT EXISTS idx_earnings_symbol ON earnings_calendar(symbol);

-- ============================================================================
-- SCREENER TABLES
-- ============================================================================

-- Saved screens table
CREATE TABLE IF NOT EXISTS saved_screens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    criteria JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_run TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_saved_screens_user ON saved_screens(user_id, created_at DESC);

-- ============================================================================
-- STATEMENTS TABLES
-- ============================================================================

-- Account statements table
CREATE TABLE IF NOT EXISTS account_statements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    statement_type VARCHAR(20) NOT NULL CHECK (statement_type IN ('monthly', 'quarterly', 'annual')),
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    generated_at TIMESTAMP DEFAULT NOW(),
    file_url TEXT,
    
    -- Summary data
    starting_balance DECIMAL(20, 2),
    ending_balance DECIMAL(20, 2),
    total_deposits DECIMAL(20, 2),
    total_withdrawals DECIMAL(20, 2),
    total_trades INTEGER,
    realized_gain_loss DECIMAL(20, 2),
    dividends_received DECIMAL(20, 2),
    fees_paid DECIMAL(20, 2)
);

CREATE INDEX IF NOT EXISTS idx_statements_user ON account_statements(user_id, period_end DESC);

-- Tax documents table
CREATE TABLE IF NOT EXISTS tax_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_type VARCHAR(20) NOT NULL CHECK (document_type IN ('1099-B', '1099-DIV', '1099-INT')),
    tax_year INTEGER NOT NULL,
    generated_at TIMESTAMP DEFAULT NOW(),
    file_url TEXT,
    
    -- Summary data
    total_proceeds DECIMAL(20, 2),
    total_cost_basis DECIMAL(20, 2),
    total_gain_loss DECIMAL(20, 2),
    total_dividends DECIMAL(20, 2),
    total_interest DECIMAL(20, 2),
    
    UNIQUE(user_id, document_type, tax_year)
);

CREATE INDEX IF NOT EXISTS idx_tax_docs_user ON tax_documents(user_id, tax_year DESC);

-- ============================================================================
-- ALERTS TABLES
-- ============================================================================

-- Price alerts table
CREATE TABLE IF NOT EXISTS price_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    alert_type VARCHAR(20) NOT NULL CHECK (alert_type IN ('price_above', 'price_below', 'price_change', 'volume')),
    target_value DECIMAL(20, 8) NOT NULL,
    current_value DECIMAL(20, 8),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'triggered', 'cancelled', 'expired')),
    notification_methods TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    triggered_at TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_user ON price_alerts(user_id, status);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON price_alerts(symbol, status) WHERE status = 'active';

-- Notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    notification_type VARCHAR(20) NOT NULL CHECK (notification_type IN ('alert', 'order', 'news', 'system')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    read_at TIMESTAMP,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications(user_id) WHERE is_read = false;

-- Notification settings table
CREATE TABLE IF NOT EXISTS notification_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    email_notifications BOOLEAN DEFAULT true,
    sms_notifications BOOLEAN DEFAULT false,
    push_notifications BOOLEAN DEFAULT true,
    alert_notifications BOOLEAN DEFAULT true,
    order_notifications BOOLEAN DEFAULT true,
    news_notifications BOOLEAN DEFAULT true,
    marketing_notifications BOOLEAN DEFAULT false,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SAMPLE DATA FOR TESTING
-- ============================================================================

-- Insert sample FAQ items
INSERT INTO faq_items (category, question, answer, display_order) VALUES
('account', 'How do I verify my account?', 'To verify your account, complete the KYC process by uploading a government-issued ID and proof of address. Verification typically takes 1-2 business days.', 1),
('account', 'How do I reset my password?', 'Click "Forgot Password" on the login page and follow the instructions sent to your email.', 2),
('trading', 'What order types are supported?', 'We support market orders, limit orders, stop orders, and stop-limit orders for all tradable securities.', 1),
('trading', 'What are trading hours?', 'Regular trading hours are 9:30 AM - 4:00 PM EST, Monday through Friday. Pre-market trading is available from 4:00 AM - 9:30 AM.', 2),
('funding', 'How do I deposit funds?', 'Navigate to the Funding page, select Deposit, choose your payment method, and enter the amount. Instant transfers arrive in minutes, standard ACH takes 3-5 business days.', 1),
('funding', 'Are there any fees for deposits or withdrawals?', 'Standard ACH transfers are free. Instant transfers have a 0.5% fee (minimum $0.50). Wire transfers may incur bank fees.', 2),
('technical', 'Which browsers are supported?', 'We support the latest versions of Chrome, Firefox, Safari, and Edge. For the best experience, we recommend Chrome.', 1),
('billing', 'What are the commission rates?', 'Stock trades are commission-free. Options trades are $0.65 per contract. No account minimums or maintenance fees.', 1)
ON CONFLICT DO NOTHING;

-- Insert sample economic events (next 7 days)
INSERT INTO economic_events (title, country, event_date, impact, currency) VALUES
('Federal Reserve Interest Rate Decision', 'USA', NOW() + INTERVAL '3 days', 'high', 'USD'),
('Non-Farm Payrolls', 'USA', NOW() + INTERVAL '5 days', 'high', 'USD'),
('Consumer Price Index (CPI)', 'USA', NOW() + INTERVAL '2 days', 'high', 'USD'),
('GDP Growth Rate', 'USA', NOW() + INTERVAL '6 days', 'medium', 'USD')
ON CONFLICT DO NOTHING;

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Critical features migration completed successfully';
    RAISE NOTICE '   - Funding tables created';
    RAISE NOTICE '   - KYC/Onboarding tables created';
    RAISE NOTICE '   - Support tables created';
    RAISE NOTICE '   - News tables created';
    RAISE NOTICE '   - Screener tables created';
    RAISE NOTICE '   - Statements tables created';
    RAISE NOTICE '   - Alerts tables created';
    RAISE NOTICE '   - Sample data inserted';
END $$;
