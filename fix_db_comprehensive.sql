-- Fix missing columns in users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS alpaca_account_id VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone_number VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'user';

-- Fix missing columns in notifications table
CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    category VARCHAR(50) DEFAULT 'info',
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE notifications ADD COLUMN IF NOT EXISTS category VARCHAR(50) DEFAULT 'info';
ALTER TABLE notifications ADD COLUMN IF NOT EXISTS action_link VARCHAR(255);

-- Ensure market_data_cache has all needed columns
CREATE TABLE IF NOT EXISTS market_data_cache (
    symbol VARCHAR(20) PRIMARY KEY,
    price DECIMAL(20, 8),
    change DECIMAL(20, 8),
    change_pct DECIMAL(20, 8),
    volume BIGINT,
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    open DECIMAL(20, 8),
    bid DECIMAL(20, 8),
    ask DECIMAL(20, 8),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Ensure ohlcv_bars exists
CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume BIGINT,
    PRIMARY KEY (symbol, timestamp, timeframe)
);
