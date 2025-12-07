-- ============================================================================
-- EXPAND PAYMENT METHODS MIGRATION
-- Adds support for credit cards, PayPal, M-Pesa, and cryptocurrency
-- ============================================================================

-- Drop the old constraint that only allowed: bank_account, debit_card, wire
ALTER TABLE payment_methods 
DROP CONSTRAINT IF EXISTS payment_methods_type_check;

-- Add new constraint with all payment types
ALTER TABLE payment_methods 
ADD CONSTRAINT payment_methods_type_check 
CHECK (type IN (
    'bank_account', 
    'debit_card', 
    'credit_card',
    'paypal', 
    'mpesa', 
    'crypto_wallet',
    'wire'
));

-- Add new columns for expanded payment methods
ALTER TABLE payment_methods 
ADD COLUMN IF NOT EXISTS bank_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS account_type VARCHAR(20),
ADD COLUMN IF NOT EXISTS routing_number VARCHAR(255),
ADD COLUMN IF NOT EXISTS card_brand VARCHAR(50),
ADD COLUMN IF NOT EXISTS card_exp_month INTEGER,
ADD COLUMN IF NOT EXISTS card_exp_year INTEGER,
ADD COLUMN IF NOT EXISTS paypal_email VARCHAR(255),
ADD COLUMN IF NOT EXISTS mpesa_phone VARCHAR(50),
ADD COLUMN IF NOT EXISTS mpesa_country VARCHAR(2),
ADD COLUMN IF NOT EXISTS crypto_address VARCHAR(255),
ADD COLUMN IF NOT EXISTS crypto_network VARCHAR(50);

-- Add comments for documentation
COMMENT ON COLUMN payment_methods.type IS 'Payment method type: bank_account, debit_card, credit_card, paypal, mpesa, crypto_wallet, wire';
COMMENT ON COLUMN payment_methods.paypal_email IS 'PayPal account email address';
COMMENT ON COLUMN payment_methods.mpesa_phone IS 'M-Pesa phone number (E.164 format)';
COMMENT ON COLUMN payment_methods.mpesa_country IS 'M-Pesa country code (KE, TZ, UG, RW)';
COMMENT ON COLUMN payment_methods.crypto_address IS 'Cryptocurrency wallet address';
COMMENT ON COLUMN payment_methods.crypto_network IS 'Crypto network: bitcoin, ethereum, usdc, usdt, solana';
