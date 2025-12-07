-- Chart Templates for Saving/Loading Chart Configurations
-- Allows users to save indicator setups, drawings, and chart settings

CREATE TABLE IF NOT EXISTS chart_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    -- config structure:
    -- {
    --   "symbol": "AAPL",
    --   "timeframe": "1d",
    --   "chartType": "candlestick",
    --   "indicators": [{"id": "sma_20", "enabled": true, "color": "#3b82f6"}],
    --   "viewMode": "single" | "multi",
    --   "multiLayout": "2x2" | "3x1" | "4x1",
    --   "multiTimeframes": ["1d", "1h", "15m", "5m"],
    --   "drawingIds": ["uuid1", "uuid2"]
    -- }
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT unique_user_template_name UNIQUE(user_id, name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chart_templates_user ON chart_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_chart_templates_default ON chart_templates(user_id, is_default) WHERE is_default = true;
CREATE INDEX IF NOT EXISTS idx_chart_templates_created ON chart_templates(created_at DESC);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_chart_templates_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_chart_templates_timestamp
    BEFORE UPDATE ON chart_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_chart_templates_updated_at();

COMMENT ON TABLE chart_templates IS 'User-saved chart configurations for quick loading';
COMMENT ON COLUMN chart_templates.config IS 'JSONB containing all chart settings, indicators, and drawing references';
COMMENT ON COLUMN chart_templates.is_default IS 'Whether this template should load by default for the user';
