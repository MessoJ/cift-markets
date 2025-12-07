-- ============================================================================
-- CIFT Markets - Database Seed Data
-- Run this AFTER init.sql to populate with realistic sample data
-- ============================================================================

-- ============================================================================
-- CHART DRAWINGS TABLE (if not exists)
-- ============================================================================
CREATE TABLE IF NOT EXISTS chart_drawings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    drawing_type VARCHAR(50) NOT NULL,
    drawing_data JSONB NOT NULL,
    style JSONB NOT NULL DEFAULT '{"color": "#3b82f6", "lineWidth": 2}',
    locked BOOLEAN DEFAULT FALSE,
    visible BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chart_drawings_user_id ON chart_drawings(user_id);
CREATE INDEX IF NOT EXISTS idx_chart_drawings_symbol ON chart_drawings(symbol);
CREATE INDEX IF NOT EXISTS idx_chart_drawings_user_symbol ON chart_drawings(user_id, symbol);

-- ============================================================================
-- WATCHLISTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    symbols TEXT DEFAULT '',
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_watchlists_user_id ON watchlists(user_id);

-- ============================================================================
-- PRICE ALERTS TABLE
-- ============================================================================
CREATE TABLE IF NOT EXISTS price_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(30) NOT NULL CHECK (alert_type IN ('price_above', 'price_below', 'price_change', 'volume')),
    target_value NUMERIC(15, 4) NOT NULL,
    current_value NUMERIC(15, 4),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'triggered', 'cancelled')),
    notification_methods JSONB DEFAULT '["push"]',
    triggered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_price_alerts_user_id ON price_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_price_alerts_symbol ON price_alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_price_alerts_status ON price_alerts(status);

-- ============================================================================
-- SUPPORT TICKETS & FAQ TABLES
-- ============================================================================
CREATE TABLE IF NOT EXISTS support_tickets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    subject VARCHAR(255) NOT NULL,
    category VARCHAR(50) NOT NULL,
    priority VARCHAR(20) DEFAULT 'medium',
    status VARCHAR(20) DEFAULT 'open',
    description TEXT NOT NULL,
    resolution TEXT,
    assigned_to VARCHAR(100),
    messages_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE IF NOT EXISTS support_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticket_id UUID NOT NULL REFERENCES support_tickets(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    message TEXT NOT NULL,
    is_staff BOOLEAN DEFAULT FALSE,
    is_internal BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS faq_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    category VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    helpful_count INTEGER DEFAULT 0,
    views INTEGER DEFAULT 0,
    is_published BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- SEED: Market Data Cache (Real stock prices as of Dec 2025)
-- ============================================================================
INSERT INTO market_data_cache (symbol, price, bid, ask, open, high, low, close, prev_close, volume, change, change_pct, status, exchange)
VALUES 
    ('AAPL', 189.95, 189.92, 189.98, 188.50, 191.20, 187.80, 189.95, 188.08, 52100000, 1.87, 0.99, 'active', 'NASDAQ'),
    ('MSFT', 415.50, 415.45, 415.55, 413.00, 418.75, 412.50, 415.50, 416.25, 18900000, -0.75, -0.18, 'active', 'NASDAQ'),
    ('GOOGL', 141.80, 141.75, 141.85, 140.20, 143.50, 139.80, 141.80, 140.85, 22400000, 0.95, 0.67, 'active', 'NASDAQ'),
    ('AMZN', 178.25, 178.20, 178.30, 176.50, 180.00, 175.80, 178.25, 177.50, 35600000, 0.75, 0.42, 'active', 'NASDAQ'),
    ('NVDA', 875.25, 875.00, 875.50, 860.00, 890.00, 855.00, 875.25, 862.75, 34500000, 12.50, 1.45, 'active', 'NASDAQ'),
    ('TSLA', 248.75, 248.70, 248.80, 252.00, 255.50, 246.00, 248.75, 252.00, 89600000, -3.25, -1.29, 'active', 'NASDAQ'),
    ('META', 505.40, 505.35, 505.45, 500.00, 510.00, 498.50, 505.40, 501.20, 16800000, 4.20, 0.84, 'active', 'NASDAQ'),
    ('JPM', 198.50, 198.45, 198.55, 196.00, 200.25, 195.50, 198.50, 197.75, 8900000, 0.75, 0.38, 'active', 'NYSE'),
    ('V', 275.80, 275.75, 275.85, 273.00, 278.00, 272.50, 275.80, 274.90, 5600000, 0.90, 0.33, 'active', 'NYSE'),
    ('UNH', 525.40, 525.30, 525.50, 520.00, 530.00, 518.00, 525.40, 523.50, 3200000, 1.90, 0.36, 'active', 'NYSE'),
    ('SPY', 478.25, 478.20, 478.30, 475.50, 480.00, 474.80, 478.25, 477.01, 52100000, 1.24, 0.26, 'active', 'NYSE'),
    ('QQQ', 405.82, 405.78, 405.86, 402.50, 408.00, 401.00, 405.82, 403.67, 38200000, 2.15, 0.53, 'active', 'NASDAQ'),
    ('DIA', 385.60, 385.55, 385.65, 383.00, 387.50, 382.00, 385.60, 384.50, 4500000, 1.10, 0.29, 'active', 'NYSE'),
    ('IWM', 198.40, 198.35, 198.45, 196.50, 200.00, 195.80, 198.40, 197.60, 28900000, 0.80, 0.40, 'active', 'NYSE'),
    ('AMD', 142.50, 142.45, 142.55, 140.00, 145.00, 138.50, 142.50, 141.20, 42300000, 1.30, 0.92, 'active', 'NASDAQ'),
    ('CRM', 265.80, 265.75, 265.85, 262.00, 268.50, 260.50, 265.80, 264.50, 5600000, 1.30, 0.49, 'active', 'NYSE'),
    ('NFLX', 485.20, 485.15, 485.25, 480.00, 490.00, 478.00, 485.20, 483.50, 4200000, 1.70, 0.35, 'active', 'NASDAQ'),
    ('INTC', 45.60, 45.55, 45.65, 44.80, 46.20, 44.50, 45.60, 45.20, 35600000, 0.40, 0.88, 'active', 'NASDAQ'),
    ('COIN', 178.50, 178.45, 178.55, 175.00, 182.00, 172.50, 178.50, 176.80, 12500000, 1.70, 0.96, 'active', 'NASDAQ'),
    ('PLTR', 25.40, 25.38, 25.42, 24.80, 26.00, 24.50, 25.40, 25.10, 48900000, 0.30, 1.20, 'active', 'NASDAQ')
ON CONFLICT (symbol) DO UPDATE SET
    price = EXCLUDED.price,
    bid = EXCLUDED.bid,
    ask = EXCLUDED.ask,
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    prev_close = EXCLUDED.prev_close,
    volume = EXCLUDED.volume,
    change = EXCLUDED.change,
    change_pct = EXCLUDED.change_pct,
    updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- SEED: Sample Positions for Admin User
-- ============================================================================
DO $$
DECLARE
    v_user_id UUID;
    v_account_id UUID;
BEGIN
    SELECT id INTO v_user_id FROM users WHERE email = 'admin@ciftmarkets.com';
    SELECT id INTO v_account_id FROM accounts WHERE user_id = v_user_id LIMIT 1;
    
    IF v_account_id IS NOT NULL THEN
        -- Clear existing positions for clean seed
        DELETE FROM positions WHERE account_id = v_account_id;
        
        -- Insert sample positions
        INSERT INTO positions (user_id, account_id, symbol, quantity, side, avg_cost, total_cost, current_price, market_value, unrealized_pnl, unrealized_pnl_pct)
        VALUES
            (v_user_id, v_account_id, 'AAPL', 50, 'long', 175.50, 8775.00, 189.95, 9497.50, 722.50, 8.23),
            (v_user_id, v_account_id, 'MSFT', 25, 'long', 380.00, 9500.00, 415.50, 10387.50, 887.50, 9.34),
            (v_user_id, v_account_id, 'NVDA', 15, 'long', 750.00, 11250.00, 875.25, 13128.75, 1878.75, 16.70),
            (v_user_id, v_account_id, 'GOOGL', 40, 'long', 135.00, 5400.00, 141.80, 5672.00, 272.00, 5.04),
            (v_user_id, v_account_id, 'AMZN', 30, 'long', 165.00, 4950.00, 178.25, 5347.50, 397.50, 8.03),
            (v_user_id, v_account_id, 'META', 20, 'long', 450.00, 9000.00, 505.40, 10108.00, 1108.00, 12.31),
            (v_user_id, v_account_id, 'TSLA', 35, 'long', 265.00, 9275.00, 248.75, 8706.25, -568.75, -6.13),
            (v_user_id, v_account_id, 'SPY', 100, 'long', 465.00, 46500.00, 478.25, 47825.00, 1325.00, 2.85);
        
        -- Update account balances
        UPDATE accounts
        SET 
            cash = 100000.00 - (SELECT COALESCE(SUM(total_cost), 0) FROM positions WHERE account_id = v_account_id),
            portfolio_value = (SELECT COALESCE(SUM(market_value), 0) FROM positions WHERE account_id = v_account_id),
            equity = cash + (SELECT COALESCE(SUM(market_value), 0) FROM positions WHERE account_id = v_account_id),
            buying_power = cash + (SELECT COALESCE(SUM(market_value), 0) FROM positions WHERE account_id = v_account_id)
        WHERE id = v_account_id;
    END IF;
END $$;

-- ============================================================================
-- SEED: Sample Orders for Admin User
-- ============================================================================
DO $$
DECLARE
    v_user_id UUID;
    v_account_id UUID;
BEGIN
    SELECT id INTO v_user_id FROM users WHERE email = 'admin@ciftmarkets.com';
    SELECT id INTO v_account_id FROM accounts WHERE user_id = v_user_id LIMIT 1;
    
    IF v_account_id IS NOT NULL THEN
        -- Insert filled orders (history)
        INSERT INTO orders (user_id, account_id, symbol, side, order_type, time_in_force, quantity, filled_quantity, remaining_quantity, limit_price, avg_fill_price, status, total_value, commission, created_at, filled_at)
        VALUES
            (v_user_id, v_account_id, 'AAPL', 'buy', 'limit', 'day', 50, 50, 0, 175.50, 175.50, 'filled', 8775.00, 0, NOW() - INTERVAL '30 days', NOW() - INTERVAL '30 days'),
            (v_user_id, v_account_id, 'MSFT', 'buy', 'market', 'day', 25, 25, 0, NULL, 380.00, 'filled', 9500.00, 0, NOW() - INTERVAL '25 days', NOW() - INTERVAL '25 days'),
            (v_user_id, v_account_id, 'NVDA', 'buy', 'limit', 'gtc', 15, 15, 0, 750.00, 750.00, 'filled', 11250.00, 0, NOW() - INTERVAL '20 days', NOW() - INTERVAL '20 days'),
            (v_user_id, v_account_id, 'GOOGL', 'buy', 'market', 'day', 40, 40, 0, NULL, 135.00, 'filled', 5400.00, 0, NOW() - INTERVAL '15 days', NOW() - INTERVAL '15 days'),
            (v_user_id, v_account_id, 'AMZN', 'buy', 'limit', 'day', 30, 30, 0, 165.00, 165.00, 'filled', 4950.00, 0, NOW() - INTERVAL '10 days', NOW() - INTERVAL '10 days'),
            (v_user_id, v_account_id, 'META', 'buy', 'market', 'day', 20, 20, 0, NULL, 450.00, 'filled', 9000.00, 0, NOW() - INTERVAL '8 days', NOW() - INTERVAL '8 days'),
            (v_user_id, v_account_id, 'TSLA', 'buy', 'limit', 'day', 35, 35, 0, 265.00, 265.00, 'filled', 9275.00, 0, NOW() - INTERVAL '5 days', NOW() - INTERVAL '5 days'),
            (v_user_id, v_account_id, 'SPY', 'buy', 'market', 'day', 100, 100, 0, NULL, 465.00, 'filled', 46500.00, 0, NOW() - INTERVAL '3 days', NOW() - INTERVAL '3 days'),
            -- Pending orders
            (v_user_id, v_account_id, 'AMD', 'buy', 'limit', 'gtc', 50, 0, 50, 138.00, NULL, 'accepted', 6900.00, 0, NOW() - INTERVAL '1 day', NULL),
            (v_user_id, v_account_id, 'AAPL', 'sell', 'limit', 'gtc', 10, 0, 10, 195.00, NULL, 'accepted', 1950.00, 0, NOW() - INTERVAL '12 hours', NULL);
    END IF;
END $$;

-- ============================================================================
-- SEED: Sample Transactions for Admin User
-- ============================================================================
DO $$
DECLARE
    v_user_id UUID;
    v_account_id UUID;
BEGIN
    SELECT id INTO v_user_id FROM users WHERE email = 'admin@ciftmarkets.com';
    SELECT id INTO v_account_id FROM accounts WHERE user_id = v_user_id LIMIT 1;
    
    IF v_account_id IS NOT NULL THEN
        -- Insert transaction history
        INSERT INTO transactions (user_id, account_id, transaction_type, amount, balance_after, symbol, description, transaction_date)
        VALUES
            (v_user_id, v_account_id, 'deposit', 100000.00, 100000.00, NULL, 'Initial deposit', NOW() - INTERVAL '35 days'),
            (v_user_id, v_account_id, 'trade', -8775.00, 91225.00, 'AAPL', 'Buy 50 AAPL @ $175.50', NOW() - INTERVAL '30 days'),
            (v_user_id, v_account_id, 'trade', -9500.00, 81725.00, 'MSFT', 'Buy 25 MSFT @ $380.00', NOW() - INTERVAL '25 days'),
            (v_user_id, v_account_id, 'trade', -11250.00, 70475.00, 'NVDA', 'Buy 15 NVDA @ $750.00', NOW() - INTERVAL '20 days'),
            (v_user_id, v_account_id, 'trade', -5400.00, 65075.00, 'GOOGL', 'Buy 40 GOOGL @ $135.00', NOW() - INTERVAL '15 days'),
            (v_user_id, v_account_id, 'dividend', 125.00, 65200.00, 'AAPL', 'Quarterly dividend - AAPL', NOW() - INTERVAL '12 days'),
            (v_user_id, v_account_id, 'trade', -4950.00, 60250.00, 'AMZN', 'Buy 30 AMZN @ $165.00', NOW() - INTERVAL '10 days'),
            (v_user_id, v_account_id, 'trade', -9000.00, 51250.00, 'META', 'Buy 20 META @ $450.00', NOW() - INTERVAL '8 days'),
            (v_user_id, v_account_id, 'trade', -9275.00, 41975.00, 'TSLA', 'Buy 35 TSLA @ $265.00', NOW() - INTERVAL '5 days'),
            (v_user_id, v_account_id, 'trade', -46500.00, -4525.00, 'SPY', 'Buy 100 SPY @ $465.00', NOW() - INTERVAL '3 days'),
            (v_user_id, v_account_id, 'deposit', 10000.00, 5475.00, NULL, 'Additional deposit', NOW() - INTERVAL '2 days'),
            (v_user_id, v_account_id, 'interest', 8.50, 5483.50, NULL, 'Cash sweep interest', NOW() - INTERVAL '1 day');
    END IF;
END $$;

-- ============================================================================
-- SEED: Sample Watchlists for Admin User
-- ============================================================================
DO $$
DECLARE
    v_user_id UUID;
BEGIN
    SELECT id INTO v_user_id FROM users WHERE email = 'admin@ciftmarkets.com';
    
    IF v_user_id IS NOT NULL THEN
        -- Insert watchlists (symbols is text[] array type)
        INSERT INTO watchlists (user_id, name, description, symbols, is_default)
        VALUES
            (v_user_id, 'Tech Giants', 'Large cap technology stocks', ARRAY['AAPL','MSFT','GOOGL','AMZN','META','NVDA'], TRUE),
            (v_user_id, 'My Portfolio', 'Stocks I currently own', ARRAY['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','SPY'], FALSE),
            (v_user_id, 'Watchlist', 'Potential buys', ARRAY['AMD','CRM','NFLX','INTC','COIN','PLTR'], FALSE),
            (v_user_id, 'ETFs', 'Index and sector ETFs', ARRAY['SPY','QQQ','DIA','IWM','XLK','XLF'], FALSE)
        ON CONFLICT (user_id, name) DO UPDATE SET 
            symbols = EXCLUDED.symbols,
            updated_at = CURRENT_TIMESTAMP;
    END IF;
END $$;

-- ============================================================================
-- SEED: Sample Price Alerts for Admin User
-- ============================================================================
DO $$
DECLARE
    v_user_id UUID;
BEGIN
    SELECT id INTO v_user_id FROM users WHERE email = 'admin@ciftmarkets.com';
    
    IF v_user_id IS NOT NULL THEN
        INSERT INTO price_alerts (user_id, symbol, alert_type, target_value, current_value, status, notification_methods)
        VALUES
            (v_user_id, 'AAPL', 'price_above', 195.00, 189.95, 'active', ARRAY['push','email']),
            (v_user_id, 'NVDA', 'price_above', 900.00, 875.25, 'active', ARRAY['push']),
            (v_user_id, 'TSLA', 'price_below', 240.00, 248.75, 'active', ARRAY['push','email']),
            (v_user_id, 'AMD', 'price_below', 140.00, 142.50, 'active', ARRAY['push']),
            (v_user_id, 'META', 'price_above', 520.00, 505.40, 'active', ARRAY['email']);
    END IF;
END $$;

-- ============================================================================
-- SEED: Sample Support Ticket for Admin User
-- ============================================================================
DO $$
DECLARE
    v_user_id UUID;
    v_ticket_id UUID;
BEGIN
    SELECT id INTO v_user_id FROM users WHERE email = 'admin@ciftmarkets.com';
    
    IF v_user_id IS NOT NULL THEN
        INSERT INTO support_tickets (user_id, subject, category, priority, status)
        VALUES
            (v_user_id, 'Question about margin requirements', 'trading', 'medium', 'resolved')
        RETURNING id INTO v_ticket_id;
        
        INSERT INTO support_messages (ticket_id, user_id, message, is_staff)
        VALUES
            (v_ticket_id, v_user_id, 'I would like to understand the margin requirements for options trading.', FALSE),
            (v_ticket_id, NULL, 'Thank you for reaching out! For options trading, you need to maintain a minimum of 25% margin for covered positions and 100% for uncovered positions. Let me know if you have more questions.', TRUE),
            (v_ticket_id, v_user_id, 'Thank you for the clarification!', FALSE);
    END IF;
END $$;

-- ============================================================================
-- SEED: FAQ Items
-- ============================================================================
INSERT INTO faq_items (category, question, answer, display_order)
VALUES
    ('account', 'How do I open an account?', 'You can open an account by clicking the "Sign Up" button on our homepage. You will need to provide your personal information and complete identity verification.', 1),
    ('account', 'What are the account types available?', 'We offer three account types: Cash accounts for simple trading, Margin accounts for leveraged trading, and Paper accounts for practice trading with virtual money.', 2),
    ('trading', 'What order types do you support?', 'We support Market orders, Limit orders, Stop orders, and Stop-Limit orders. Each order type has different execution characteristics.', 1),
    ('trading', 'What are your trading hours?', 'Regular trading hours are 9:30 AM - 4:00 PM ET. Extended hours trading is available from 4:00 AM - 9:30 AM and 4:00 PM - 8:00 PM ET.', 2),
    ('funding', 'How do I deposit funds?', 'You can deposit funds via ACH transfer, wire transfer, or debit card. ACH transfers are free and typically take 1-3 business days.', 1),
    ('funding', 'What are the withdrawal limits?', 'Daily withdrawal limit is $50,000 for ACH and $100,000 for wire transfers. You can request limit increases by contacting support.', 2),
    ('technical', 'How do I use the charting tools?', 'Our charts support multiple timeframes, technical indicators, and drawing tools. Click on the chart icon to access the full charting interface.', 1),
    ('billing', 'Are there any account fees?', 'CIFT Markets offers commission-free trading. There are no monthly fees or minimum balance requirements.', 1)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- GRANT PERMISSIONS
-- ============================================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO cift_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO cift_user;
