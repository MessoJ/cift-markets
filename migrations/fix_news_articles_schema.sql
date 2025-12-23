-- Add missing columns to news_articles table
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS category VARCHAR(50);
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS symbols JSONB DEFAULT '[]';
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS categories JSONB DEFAULT '[]';
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS sentiment VARCHAR(50);
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS importance INTEGER DEFAULT 1;
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS image_url TEXT;

-- Re-create indexes
DROP INDEX IF EXISTS idx_news_category;
CREATE INDEX idx_news_category ON news_articles(category);
