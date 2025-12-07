-- Payment Verification System Migration
-- Adds support for tracking payment method verification flows

-- Create payment_verification table
CREATE TABLE IF NOT EXISTS payment_verification (
    payment_method_id UUID PRIMARY KEY REFERENCES payment_methods(id) ON DELETE CASCADE,
    verification_type VARCHAR(50) NOT NULL,  -- 'micro_deposit', 'instant', 'stk_push', 'oauth', 'address'
    verification_data JSONB NOT NULL,        -- Stores verification-specific data (amounts, codes, tokens)
    attempt_count INTEGER DEFAULT 0,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Add verification fields to payment_methods table
ALTER TABLE payment_methods 
ADD COLUMN IF NOT EXISTS verification_status VARCHAR(50) DEFAULT 'pending_verification',
ADD COLUMN IF NOT EXISTS verification_error TEXT,
ADD COLUMN IF NOT EXISTS verification_initiated_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS cashapp_tag VARCHAR(255);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_payment_verification_expires ON payment_verification(expires_at);
CREATE INDEX IF NOT EXISTS idx_payment_methods_verification_status ON payment_methods(verification_status);

-- Add comments
COMMENT ON TABLE payment_verification IS 'Tracks active payment method verification processes';
COMMENT ON COLUMN payment_verification.verification_type IS 'Type of verification: micro_deposit, instant, stk_push, oauth, address';
COMMENT ON COLUMN payment_verification.verification_data IS 'Encrypted verification data (amounts, tokens, codes)';
COMMENT ON COLUMN payment_verification.attempt_count IS 'Number of verification attempts (max 3)';

-- Update existing payment methods to have default status
UPDATE payment_methods 
SET verification_status = CASE 
    WHEN is_verified = true THEN 'verified'
    ELSE 'pending_verification'
END
WHERE verification_status IS NULL;
