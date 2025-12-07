-- Chart Drawings Table
-- Stores user-drawn annotations on charts (trendlines, Fibonacci, shapes)

CREATE TABLE IF NOT EXISTS chart_drawings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    
    -- Drawing type
    drawing_type VARCHAR(50) NOT NULL, -- 'trendline', 'fibonacci', 'horizontal_line', 'rectangle', 'text', 'arrow'
    
    -- Drawing data (JSON)
    data JSONB NOT NULL,
    -- Structure:
    -- Trendline: {x1: timestamp, y1: price, x2: timestamp, y2: price, color: '#xxx', width: 2}
    -- Fibonacci: {startX: timestamp, startY: price, endX: timestamp, endY: price, levels: [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]}
    -- Rectangle: {x1: timestamp, y1: price, x2: timestamp, y2: price, fillColor: '#xxx', borderColor: '#xxx'}
    -- Text: {x: timestamp, y: price, text: 'annotation', fontSize: 14, color: '#xxx'}
    -- Arrow: {x1: timestamp, y1: price, x2: timestamp, y2: price, color: '#xxx'}
    
    -- Styling
    color VARCHAR(20) DEFAULT '#3b82f6',
    line_width INTEGER DEFAULT 2,
    line_style VARCHAR(20) DEFAULT 'solid', -- 'solid', 'dashed', 'dotted'
    
    -- Visibility
    is_visible BOOLEAN DEFAULT true,
    locked BOOLEAN DEFAULT false,
    
    -- Metadata
    name VARCHAR(100),
    notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast retrieval
CREATE INDEX idx_chart_drawings_user_symbol ON chart_drawings(user_id, symbol);
CREATE INDEX idx_chart_drawings_user_symbol_timeframe ON chart_drawings(user_id, symbol, timeframe);
CREATE INDEX idx_chart_drawings_type ON chart_drawings(drawing_type);
CREATE INDEX idx_chart_drawings_visible ON chart_drawings(is_visible) WHERE is_visible = true;

-- Trigger to update updated_at
CREATE TRIGGER update_chart_drawings_updated_at
    BEFORE UPDATE ON chart_drawings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE chart_drawings IS 'User-drawn chart annotations (trendlines, Fibonacci retracements, shapes)';
COMMENT ON COLUMN chart_drawings.data IS 'JSONB containing drawing coordinates and properties';
COMMENT ON COLUMN chart_drawings.locked IS 'If true, drawing cannot be edited or deleted';
