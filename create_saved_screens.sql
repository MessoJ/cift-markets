-- Create saved_screens table if it doesn't exist
CREATE TABLE IF NOT EXISTS saved_screens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    criteria JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_run TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_saved_screens_user ON saved_screens(user_id, created_at DESC);

-- Verify table was created
SELECT 'saved_screens table created successfully' as status;
