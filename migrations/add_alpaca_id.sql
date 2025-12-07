-- Add Alpaca Account ID to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS alpaca_account_id TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS kyc_status TEXT DEFAULT 'pending'; -- pending, submitted, approved, rejected
