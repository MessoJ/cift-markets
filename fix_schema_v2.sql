-- Fix missing columns in database schema

-- 1. Add 'category' to news_articles
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS category TEXT;
CREATE INDEX IF NOT EXISTS idx_news_articles_category ON news_articles(category);

-- 2. Add 'alpaca_account_id' to trading_accounts
ALTER TABLE trading_accounts ADD COLUMN IF NOT EXISTS alpaca_account_id TEXT;

-- 3. Ensure 'status' in accounts table allows 'active' (case insensitive check or update constraint)
-- The user had an issue with "ACTIVE" vs "active". Let's drop the check constraint if it exists and is too strict.
-- Finding the constraint name is hard without querying, but usually it's accounts_status_check.
-- We will try to drop it and recreate it with a looser check or just rely on the application.
ALTER TABLE accounts DROP CONSTRAINT IF EXISTS accounts_status_check;
ALTER TABLE accounts ADD CONSTRAINT accounts_status_check CHECK (status IN ('active', 'inactive', 'suspended', 'closed', 'ACTIVE', 'INACTIVE', 'SUSPENDED', 'CLOSED'));
