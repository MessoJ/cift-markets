-- Migration: Chart Drawings and State Persistence
-- Stores user drawings (trendlines, Fibonacci, etc.) and chart configurations

-- ============================================================================
-- CHART DRAWINGS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS chart_drawings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Drawing metadata
    drawing_type VARCHAR(50) NOT NULL CHECK (
        drawing_type IN (
            'trendline', 'horizontal_line', 'vertical_line',
            'fibonacci', 'rectangle', 'text', 'arrow'
        )
    ),
    
    -- Drawing data (stored as JSONB for flexibility)
    drawing_data JSONB NOT NULL,
    
    -- Style configuration
    style JSONB NOT NULL DEFAULT '{
        "color": "#3b82f6",
        "lineWidth": 2,
        "lineType": "solid"
    }'::jsonb,
    
    -- State flags
    locked BOOLEAN DEFAULT FALSE,
    visible BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes for fast lookup
    CONSTRAINT valid_drawing_data CHECK (jsonb_typeof(drawing_data) = 'object')
);

-- Indexes for performance
CREATE INDEX idx_chart_drawings_user_symbol ON chart_drawings(user_id, symbol);
CREATE INDEX idx_chart_drawings_symbol_timeframe ON chart_drawings(symbol, timeframe);
CREATE INDEX idx_chart_drawings_created_at ON chart_drawings(created_at DESC);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_chart_drawings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_chart_drawings_updated_at
    BEFORE UPDATE ON chart_drawings
    FOR EACH ROW
    EXECUTE FUNCTION update_chart_drawings_updated_at();

-- ============================================================================
-- CHART STATES TABLE (saved chart configurations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chart_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Chart configuration
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Active indicators
    indicators JSONB DEFAULT '[]'::jsonb,
    
    -- Chart settings (colors, overlays, etc.)
    settings JSONB DEFAULT '{}'::jsonb,
    
    -- Template flag (can be shared/reused)
    is_template BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure user has unique names
    CONSTRAINT unique_user_chart_name UNIQUE (user_id, name)
);

-- Indexes
CREATE INDEX idx_chart_states_user ON chart_states(user_id);
CREATE INDEX idx_chart_states_template ON chart_states(is_template) WHERE is_template = TRUE;
CREATE INDEX idx_chart_states_accessed ON chart_states(last_accessed_at DESC);

-- Update timestamp trigger
CREATE TRIGGER trigger_chart_states_updated_at
    BEFORE UPDATE ON chart_states
    FOR EACH ROW
    EXECUTE FUNCTION update_chart_drawings_updated_at();

-- ============================================================================
-- CHART TEMPLATES (predefined configurations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS chart_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    category VARCHAR(50) DEFAULT 'general',
    
    -- Template configuration
    indicators JSONB NOT NULL DEFAULT '[]'::jsonb,
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    
    -- Popularity tracking
    usage_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_chart_templates_category ON chart_templates(category);
CREATE INDEX idx_chart_templates_popularity ON chart_templates(usage_count DESC);

-- ============================================================================
-- INSERT DEFAULT TEMPLATES
-- ============================================================================

INSERT INTO chart_templates (name, description, category, indicators, settings) VALUES
    (
        'Trend Following',
        'Moving averages for trend identification',
        'trend',
        '["sma_20", "sma_50", "sma_200", "ema_12"]'::jsonb,
        '{"showVolume": true, "showGrid": true}'::jsonb
    ),
    (
        'Momentum Trading',
        'MACD and RSI for momentum signals',
        'momentum',
        '["macd", "rsi_14", "ema_12", "ema_26"]'::jsonb,
        '{"showVolume": true, "showMACD": true}'::jsonb
    ),
    (
        'Volatility Analysis',
        'Bollinger Bands for volatility tracking',
        'volatility',
        '["bb_bands", "sma_20", "volatility_20"]'::jsonb,
        '{"showVolume": true, "showBollingerBands": true}'::jsonb
    ),
    (
        'Clean Chart',
        'Minimal chart with just price action',
        'general',
        '[]'::jsonb,
        '{"showVolume": false, "showGrid": false}'::jsonb
    )
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE chart_drawings IS 'User drawings on charts (trendlines, Fibonacci, shapes)';
COMMENT ON TABLE chart_states IS 'Saved chart configurations and indicator sets';
COMMENT ON TABLE chart_templates IS 'Predefined chart templates for quick setup';

COMMENT ON COLUMN chart_drawings.drawing_data IS 'JSONB containing points, coordinates, and drawing-specific data';
COMMENT ON COLUMN chart_drawings.style IS 'JSONB containing color, line width, fill, etc.';
COMMENT ON COLUMN chart_states.indicators IS 'Array of active indicator IDs';
COMMENT ON COLUMN chart_states.settings IS 'Chart UI settings (theme, colors, panels)';
