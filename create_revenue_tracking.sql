-- Create platform_revenue table to track CIFT Markets earnings
CREATE TABLE IF NOT EXISTS platform_revenue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(50) NOT NULL CHECK (source_type IN ('trading_commission', 'funding_fee', 'subscription', 'other')),
    
    -- Amount details
    amount NUMERIC(15, 4) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    
    -- Reference to the source of revenue
    reference_id UUID NOT NULL, -- order_id or transaction_id
    user_id UUID REFERENCES users(id),
    account_id UUID REFERENCES accounts(id),
    
    -- Metadata
    description TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for reporting
CREATE INDEX IF NOT EXISTS idx_platform_revenue_created_at ON platform_revenue(created_at);
CREATE INDEX IF NOT EXISTS idx_platform_revenue_source_type ON platform_revenue(source_type);
CREATE INDEX IF NOT EXISTS idx_platform_revenue_user_id ON platform_revenue(user_id);

-- Create a view for daily revenue summary
CREATE OR REPLACE VIEW daily_revenue_summary AS
SELECT 
    DATE_TRUNC('day', created_at) as revenue_date,
    source_type,
    COUNT(*) as transaction_count,
    SUM(amount) as total_revenue
FROM platform_revenue
GROUP BY 1, 2
ORDER BY 1 DESC, 2;
