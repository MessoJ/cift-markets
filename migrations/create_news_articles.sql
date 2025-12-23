-- Create news_articles table
CREATE TABLE IF NOT EXISTS news_articles (
    id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    url TEXT NOT NULL,
    source VARCHAR(255),
    author VARCHAR(255),
    published_at TIMESTAMP WITH TIME ZONE,
    symbols JSONB DEFAULT '[]',
    categories JSONB DEFAULT '[]',
    category VARCHAR(50), -- Added for backward compatibility/simple queries
    sentiment VARCHAR(50),
    importance INTEGER DEFAULT 1,
    image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_category ON news_articles(category);
