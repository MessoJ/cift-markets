-- ============================================================================
-- USER SETTINGS & PREFERENCES DATABASE MIGRATION
-- Adds tables for: User Settings, API Keys, Session Logs, 2FA
-- Critical for Settings Page functionality
-- ============================================================================

-- ============================================================================
-- USER SETTINGS TABLE
-- ============================================================================

-- Main user settings/preferences table
CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    
    -- Profile
    full_name VARCHAR(200),
    phone_number VARCHAR(20),
    
    -- Trading Preferences
    default_order_type VARCHAR(20) DEFAULT 'market' CHECK (default_order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    default_time_in_force VARCHAR(10) DEFAULT 'day' CHECK (default_time_in_force IN ('day', 'gtc', 'ioc', 'fok')),
    require_order_confirmation BOOLEAN DEFAULT true,
    enable_fractional_shares BOOLEAN DEFAULT false,
    
    -- Notification Preferences
    email_notifications BOOLEAN DEFAULT true,
    email_trade_confirms BOOLEAN DEFAULT true,
    email_market_news BOOLEAN DEFAULT false,
    email_price_alerts BOOLEAN DEFAULT true,
    
    sms_notifications BOOLEAN DEFAULT false,
    sms_trade_confirms BOOLEAN DEFAULT false,
    sms_price_alerts BOOLEAN DEFAULT false,
    
    push_notifications BOOLEAN DEFAULT true,
    push_trade_confirms BOOLEAN DEFAULT true,
    push_market_news BOOLEAN DEFAULT false,
    push_price_alerts BOOLEAN DEFAULT true,
    
    -- Quiet Hours
    notification_quiet_hours BOOLEAN DEFAULT false,
    quiet_start_time TIME DEFAULT '22:00:00',
    quiet_end_time TIME DEFAULT '08:00:00',
    quiet_days VARCHAR(20)[] DEFAULT ARRAY['saturday', 'sunday'],
    
    -- UI Preferences
    theme VARCHAR(20) DEFAULT 'dark' CHECK (theme IN ('dark', 'light', 'auto')),
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'America/New_York',
    currency VARCHAR(3) DEFAULT 'USD',
    date_format VARCHAR(20) DEFAULT 'MM/DD/YYYY',
    
    -- Display Options
    show_portfolio_value BOOLEAN DEFAULT true,
    show_buying_power BOOLEAN DEFAULT true,
    show_day_pnl BOOLEAN DEFAULT true,
    compact_mode BOOLEAN DEFAULT false,
    
    -- Data & Privacy
    data_sharing_enabled BOOLEAN DEFAULT false,
    analytics_enabled BOOLEAN DEFAULT true,
    marketing_emails BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_settings_updated ON user_settings(updated_at);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_user_settings_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_user_settings_timestamp ON user_settings;
CREATE TRIGGER trigger_user_settings_timestamp
    BEFORE UPDATE ON user_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_user_settings_timestamp();

-- ============================================================================
-- API KEYS TABLE
-- ============================================================================

-- User-generated API keys for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Key details
    key_hash TEXT NOT NULL UNIQUE,  -- Hashed API key (bcrypt)
    key_prefix VARCHAR(10) NOT NULL,  -- First 8 chars for display (e.g., "sk_live_...")
    name VARCHAR(200),
    description TEXT,
    
    -- Permissions
    scopes TEXT[] DEFAULT ARRAY['read'],  -- ['read', 'trade', 'withdraw']
    
    -- Rate limiting
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    rate_limit_per_day INTEGER DEFAULT 10000,
    
    -- Usage tracking
    last_used_at TIMESTAMP,
    last_used_ip INET,
    total_requests BIGINT DEFAULT 0,
    
    -- Lifecycle
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    revoked_at TIMESTAMP,
    revoked_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);

-- ============================================================================
-- SESSION LOGS TABLE
-- ============================================================================

-- Track user login sessions for security auditing
CREATE TABLE IF NOT EXISTS session_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session details
    session_token_hash TEXT NOT NULL,
    
    -- Login information
    ip_address INET,
    user_agent TEXT,
    device_type VARCHAR(50),  -- 'desktop', 'mobile', 'tablet'
    browser VARCHAR(100),
    os VARCHAR(100),
    
    -- Location (from IP geolocation)
    country VARCHAR(50),
    region VARCHAR(100),
    city VARCHAR(100),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    
    -- Session lifecycle
    login_at TIMESTAMP DEFAULT NOW(),
    logout_at TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    -- Security flags
    is_suspicious BOOLEAN DEFAULT false,
    suspicious_reason TEXT,
    login_method VARCHAR(50) DEFAULT 'password',  -- 'password', '2fa', 'api_key', 'oauth'
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_logs_user ON session_logs(user_id, login_at DESC);
CREATE INDEX IF NOT EXISTS idx_session_logs_active ON session_logs(user_id) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_session_logs_suspicious ON session_logs(is_suspicious) WHERE is_suspicious = true;

-- ============================================================================
-- TWO-FACTOR AUTHENTICATION TABLE
-- ============================================================================

-- 2FA settings and backup codes
CREATE TABLE IF NOT EXISTS two_factor_auth (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    
    -- 2FA Status
    enabled BOOLEAN DEFAULT false,
    method VARCHAR(20) DEFAULT 'totp' CHECK (method IN ('totp', 'sms', 'email')),
    
    -- TOTP (Time-based One-Time Password)
    secret_encrypted TEXT,  -- Encrypted TOTP secret
    
    -- Backup codes (for account recovery)
    backup_codes_encrypted TEXT[],  -- Array of hashed backup codes
    backup_codes_remaining INTEGER DEFAULT 10,
    
    -- SMS 2FA
    phone_number_verified VARCHAR(20),
    phone_verified_at TIMESTAMP,
    
    -- Email 2FA
    email_verified BOOLEAN DEFAULT false,
    
    -- Lifecycle
    verified_at TIMESTAMP,
    last_verified_at TIMESTAMP,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_2fa_enabled ON two_factor_auth(user_id) WHERE enabled = true;

-- ============================================================================
-- SECURITY AUDIT LOG TABLE
-- ============================================================================

-- Comprehensive audit trail for security events
CREATE TABLE IF NOT EXISTS security_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Event details
    event_type VARCHAR(100) NOT NULL,  -- 'login', 'logout', 'password_change', 'api_key_created', etc.
    event_category VARCHAR(50) NOT NULL,  -- 'auth', 'settings', 'trading', 'withdrawal'
    severity VARCHAR(20) DEFAULT 'info',  -- 'info', 'warning', 'critical'
    
    -- Event data
    description TEXT,
    metadata JSONB,  -- Flexible storage for event-specific data
    
    -- Request context
    ip_address INET,
    user_agent TEXT,
    endpoint VARCHAR(200),
    http_method VARCHAR(10),
    
    -- Status
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_log_user ON security_audit_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_event ON security_audit_log(event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_severity ON security_audit_log(severity, created_at DESC) WHERE severity IN ('warning', 'critical');
CREATE INDEX IF NOT EXISTS idx_audit_log_metadata ON security_audit_log USING gin(metadata);

-- ============================================================================
-- PASSWORD RESET TOKENS TABLE
-- ============================================================================

-- Secure password reset token management
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    token_hash TEXT NOT NULL UNIQUE,
    
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    is_valid BOOLEAN DEFAULT true,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_password_reset_user ON password_reset_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_password_reset_valid ON password_reset_tokens(token_hash) WHERE is_valid = true AND used_at IS NULL;

-- ============================================================================
-- EMAIL VERIFICATION TOKENS TABLE
-- ============================================================================

-- Email address verification tokens
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    email VARCHAR(255) NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    
    expires_at TIMESTAMP NOT NULL,
    verified_at TIMESTAMP,
    is_valid BOOLEAN DEFAULT true,
    
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_email_verification_user ON email_verification_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_email_verification_valid ON email_verification_tokens(token_hash) WHERE is_valid = true AND verified_at IS NULL;

-- ============================================================================
-- DEFAULT USER SETTINGS INSERTION
-- ============================================================================

-- Create default settings for existing users
INSERT INTO user_settings (user_id, full_name)
SELECT id, full_name FROM users
WHERE NOT EXISTS (SELECT 1 FROM user_settings WHERE user_settings.user_id = users.id);

-- ============================================================================
-- VIEWS FOR CONVENIENCE
-- ============================================================================

-- View: Active sessions for security dashboard
CREATE OR REPLACE VIEW active_sessions AS
SELECT 
    sl.id,
    sl.user_id,
    u.email,
    u.username,
    sl.ip_address,
    sl.device_type,
    sl.browser,
    sl.city,
    sl.country,
    sl.login_at,
    sl.last_activity_at,
    EXTRACT(EPOCH FROM (NOW() - sl.last_activity_at)) / 60 AS minutes_idle
FROM session_logs sl
JOIN users u ON sl.user_id = u.id
WHERE sl.is_active = true
ORDER BY sl.last_activity_at DESC;

-- View: API key usage statistics
CREATE OR REPLACE VIEW api_key_stats AS
SELECT 
    ak.id,
    ak.user_id,
    u.email,
    ak.name,
    ak.key_prefix,
    ak.scopes,
    ak.total_requests,
    ak.last_used_at,
    ak.created_at,
    CASE 
        WHEN ak.expires_at IS NULL THEN 'never'
        WHEN ak.expires_at < NOW() THEN 'expired'
        ELSE 'active'
    END as expiry_status
FROM api_keys ak
JOIN users u ON ak.user_id = u.id
WHERE ak.is_active = true
ORDER BY ak.last_used_at DESC NULLS LAST;

-- ============================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- ============================================================================

-- Function: Log security event
CREATE OR REPLACE FUNCTION log_security_event(
    p_user_id UUID,
    p_event_type VARCHAR,
    p_event_category VARCHAR,
    p_description TEXT DEFAULT NULL,
    p_severity VARCHAR DEFAULT 'info',
    p_metadata JSONB DEFAULT NULL,
    p_ip_address INET DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_log_id UUID;
BEGIN
    INSERT INTO security_audit_log (
        user_id, event_type, event_category, 
        description, severity, metadata, ip_address
    ) VALUES (
        p_user_id, p_event_type, p_event_category,
        p_description, p_severity, p_metadata, p_ip_address
    ) RETURNING id INTO v_log_id;
    
    RETURN v_log_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Clean up expired tokens
CREATE OR REPLACE FUNCTION cleanup_expired_tokens()
RETURNS INTEGER AS $$
DECLARE
    v_deleted INTEGER := 0;
BEGIN
    -- Delete expired password reset tokens
    DELETE FROM password_reset_tokens
    WHERE expires_at < NOW() - INTERVAL '7 days';
    GET DIAGNOSTICS v_deleted = ROW_COUNT;
    
    -- Delete expired email verification tokens
    DELETE FROM email_verification_tokens
    WHERE expires_at < NOW() - INTERVAL '7 days';
    
    -- Delete old inactive sessions (keep for 90 days)
    DELETE FROM session_logs
    WHERE is_active = false 
    AND logout_at < NOW() - INTERVAL '90 days';
    
    RETURN v_deleted;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- COMMENTS FOR DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE user_settings IS 'User preferences and application settings';
COMMENT ON TABLE api_keys IS 'User-generated API keys for programmatic access';
COMMENT ON TABLE session_logs IS 'Login session tracking for security auditing';
COMMENT ON TABLE two_factor_auth IS 'Two-factor authentication settings';
COMMENT ON TABLE security_audit_log IS 'Comprehensive security event audit trail';
COMMENT ON TABLE password_reset_tokens IS 'Secure password reset tokens';
COMMENT ON TABLE email_verification_tokens IS 'Email address verification tokens';

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE '✅ Migration 003_user_settings.sql completed successfully';
    RAISE NOTICE '   - Created user_settings table';
    RAISE NOTICE '   - Created api_keys table';
    RAISE NOTICE '   - Created session_logs table';
    RAISE NOTICE '   - Created two_factor_auth table';
    RAISE NOTICE '   - Created security_audit_log table';
    RAISE NOTICE '   - Created password_reset_tokens table';
    RAISE NOTICE '   - Created email_verification_tokens table';
    RAISE NOTICE '   - Created helper views and functions';
    RAISE NOTICE '   - Total new tables: 7';
END $$;
