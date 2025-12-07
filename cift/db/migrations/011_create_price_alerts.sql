-- Price Alerts System
-- Allows users to set alerts when price reaches certain levels

CREATE TABLE IF NOT EXISTS price_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(20) NOT NULL CHECK (alert_type IN ('above', 'below', 'crosses_above', 'crosses_below')),
    price NUMERIC(18, 8) NOT NULL,
    message TEXT,
    triggered BOOLEAN DEFAULT false,
    triggered_at TIMESTAMPTZ,
    triggered_price NUMERIC(18, 8),
    notification_sent BOOLEAN DEFAULT false,
    enabled BOOLEAN DEFAULT true,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT positive_price CHECK (price > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_price_alerts_user ON price_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_price_alerts_symbol ON price_alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_price_alerts_active ON price_alerts(user_id, symbol, enabled, triggered) 
    WHERE enabled = true AND triggered = false;
CREATE INDEX IF NOT EXISTS idx_price_alerts_triggered ON price_alerts(triggered_at DESC) WHERE triggered = true;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_price_alerts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_price_alerts_timestamp
    BEFORE UPDATE ON price_alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_price_alerts_updated_at();

COMMENT ON TABLE price_alerts IS 'Price alerts for notifying users when prices reach specific levels';
COMMENT ON COLUMN price_alerts.alert_type IS 'Type of alert: above, below, crosses_above, crosses_below';
COMMENT ON COLUMN price_alerts.triggered IS 'Whether the alert has been triggered';
COMMENT ON COLUMN price_alerts.notification_sent IS 'Whether notification was successfully sent to user';
COMMENT ON COLUMN price_alerts.expires_at IS 'Optional expiration date for the alert';
