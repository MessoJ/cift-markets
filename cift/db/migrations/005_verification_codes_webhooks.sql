-- Verification Codes and Webhook Support Migration
-- Adds support for SMS verification codes and webhook event logging

-- Create verification_codes table for SMS/phone verification
CREATE TABLE IF NOT EXISTS verification_codes (
    id SERIAL PRIMARY KEY,
    phone VARCHAR(20) NOT NULL,
    code VARCHAR(10) NOT NULL,
    purpose VARCHAR(50) NOT NULL,  -- 'phone_verification', '2fa', 'account_recovery'
    expires_at TIMESTAMP NOT NULL,
    verified_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_verification_codes_phone ON verification_codes(phone);
CREATE INDEX IF NOT EXISTS idx_verification_codes_expires ON verification_codes(expires_at);
CREATE INDEX IF NOT EXISTS idx_verification_codes_code ON verification_codes(code);

-- Create webhook_events table for logging webhook callbacks
CREATE TABLE IF NOT EXISTS webhook_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL,  -- 'stripe', 'paypal', 'mpesa'
    event_type VARCHAR(100) NOT NULL,
    event_id VARCHAR(255),  -- External event ID from provider
    payload JSONB NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for webhook events
CREATE INDEX IF NOT EXISTS idx_webhook_events_provider ON webhook_events(provider);
CREATE INDEX IF NOT EXISTS idx_webhook_events_event_type ON webhook_events(event_type);
CREATE INDEX IF NOT EXISTS idx_webhook_events_event_id ON webhook_events(event_id);
CREATE INDEX IF NOT EXISTS idx_webhook_events_processed ON webhook_events(processed);
CREATE INDEX IF NOT EXISTS idx_webhook_events_created ON webhook_events(created_at);

-- Add external IDs to payment_methods for webhook correlation
ALTER TABLE payment_methods 
ADD COLUMN IF NOT EXISTS external_method_id VARCHAR(255),  -- Stripe payment_method_id, PayPal billing_agreement_id
ADD COLUMN IF NOT EXISTS external_customer_id VARCHAR(255);  -- Stripe customer_id, PayPal payer_id

-- Add external transaction ID to funding_transactions
ALTER TABLE funding_transactions
ADD COLUMN IF NOT EXISTS external_transaction_id VARCHAR(255);  -- Stripe payment_intent_id, PayPal capture_id, M-Pesa receipt

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_payment_methods_external_id ON payment_methods(external_method_id);
CREATE INDEX IF NOT EXISTS idx_funding_transactions_external_id ON funding_transactions(external_transaction_id);

-- Add comments
COMMENT ON TABLE verification_codes IS 'Stores SMS/phone verification codes for 2FA and account security';
COMMENT ON TABLE webhook_events IS 'Logs webhook events from payment providers for debugging and replay';
COMMENT ON COLUMN verification_codes.purpose IS 'Purpose: phone_verification, 2fa, account_recovery';
COMMENT ON COLUMN webhook_events.provider IS 'Payment provider: stripe, paypal, mpesa';

-- Clean up expired verification codes (run periodically)
-- DELETE FROM verification_codes WHERE expires_at < NOW() - INTERVAL '7 days';
