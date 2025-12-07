-- Recategorize news articles based on content analysis
-- Run this after adding new categorization logic

-- Show current distribution
SELECT category, COUNT(*) as count
FROM news_articles
GROUP BY category
ORDER BY count DESC;

-- Categorize as EARNINGS
UPDATE news_articles
SET category = 'earnings'
WHERE category != 'earnings'
  AND (
    LOWER(title || ' ' || summary) LIKE '%earnings%' OR
    LOWER(title || ' ' || summary) LIKE '%eps%' OR
    LOWER(title || ' ' || summary) LIKE '%revenue%' OR
    LOWER(title || ' ' || summary) LIKE '%profit%' OR
    LOWER(title || ' ' || summary) LIKE '%quarterly results%' OR
    LOWER(title || ' ' || summary) LIKE '%q1%' OR
    LOWER(title || ' ' || summary) LIKE '%q2%' OR
    LOWER(title || ' ' || summary) LIKE '%q3%' OR
    LOWER(title || ' ' || summary) LIKE '%q4%' OR
    LOWER(title || ' ' || summary) LIKE '%fiscal%' OR
    LOWER(title || ' ' || summary) LIKE '%guidance%' OR
    LOWER(title || ' ' || summary) LIKE '%beats estimate%' OR
    LOWER(title || ' ' || summary) LIKE '%misses estimate%' OR
    LOWER(title || ' ' || summary) LIKE '%earnings call%' OR
    LOWER(title || ' ' || summary) LIKE '%financial results%'
  );

-- Categorize as ECONOMICS
UPDATE news_articles
SET category = 'economics'
WHERE category NOT IN ('earnings', 'economics')
  AND (
    LOWER(title || ' ' || summary) LIKE '%fed%' OR
    LOWER(title || ' ' || summary) LIKE '%federal reserve%' OR
    LOWER(title || ' ' || summary) LIKE '%interest rate%' OR
    LOWER(title || ' ' || summary) LIKE '%inflation%' OR
    LOWER(title || ' ' || summary) LIKE '%cpi%' OR
    LOWER(title || ' ' || summary) LIKE '%unemployment%' OR
    LOWER(title || ' ' || summary) LIKE '%gdp%' OR
    LOWER(title || ' ' || summary) LIKE '%economic%' OR
    LOWER(title || ' ' || summary) LIKE '%central bank%' OR
    LOWER(title || ' ' || summary) LIKE '%monetary policy%' OR
    LOWER(title || ' ' || summary) LIKE '%fiscal policy%' OR
    LOWER(title || ' ' || summary) LIKE '%recession%' OR
    LOWER(title || ' ' || summary) LIKE '%treasury%' OR
    LOWER(title || ' ' || summary) LIKE '%bond yield%' OR
    LOWER(title || ' ' || summary) LIKE '%jobs report%' OR
    LOWER(title || ' ' || summary) LIKE '%non-farm payroll%' OR
    LOWER(title || ' ' || summary) LIKE '%pmi%' OR
    LOWER(title || ' ' || summary) LIKE '%consumer confidence%'
  );

-- Categorize as CRYPTO
UPDATE news_articles
SET category = 'crypto'
WHERE category NOT IN ('earnings', 'economics', 'crypto')
  AND (
    LOWER(title || ' ' || summary) LIKE '%bitcoin%' OR
    LOWER(title || ' ' || summary) LIKE '%btc%' OR
    LOWER(title || ' ' || summary) LIKE '%ethereum%' OR
    LOWER(title || ' ' || summary) LIKE '%eth%' OR
    LOWER(title || ' ' || summary) LIKE '%crypto%' OR
    LOWER(title || ' ' || summary) LIKE '%cryptocurrency%' OR
    LOWER(title || ' ' || summary) LIKE '%blockchain%' OR
    LOWER(title || ' ' || summary) LIKE '%defi%' OR
    LOWER(title || ' ' || summary) LIKE '%nft%' OR
    LOWER(title || ' ' || summary) LIKE '%web3%' OR
    LOWER(title || ' ' || summary) LIKE '%altcoin%' OR
    LOWER(title || ' ' || summary) LIKE '%dogecoin%' OR
    LOWER(title || ' ' || summary) LIKE '%binance%' OR
    LOWER(title || ' ' || summary) LIKE '%coinbase%' OR
    LOWER(title || ' ' || summary) LIKE '%solana%' OR
    LOWER(title || ' ' || summary) LIKE '%cardano%'
  );

-- Categorize as TECHNOLOGY
UPDATE news_articles
SET category = 'technology'
WHERE category NOT IN ('earnings', 'economics', 'crypto', 'technology')
  AND (
    LOWER(title || ' ' || summary) LIKE '%artificial intelligence%' OR
    LOWER(title || ' ' || summary) LIKE '% ai %' OR
    LOWER(title || ' ' || summary) LIKE '%machine learning%' OR
    LOWER(title || ' ' || summary) LIKE '% ml %' OR
    LOWER(title || ' ' || summary) LIKE '%technology%' OR
    LOWER(title || ' ' || summary) LIKE '% tech %' OR
    LOWER(title || ' ' || summary) LIKE '%software%' OR
    LOWER(title || ' ' || summary) LIKE '%hardware%' OR
    LOWER(title || ' ' || summary) LIKE '%cloud computing%' OR
    LOWER(title || ' ' || summary) LIKE '%semiconductor%' OR
    LOWER(title || ' ' || summary) LIKE '%chip%' OR
    LOWER(title || ' ' || summary) LIKE '%processor%' OR
    LOWER(title || ' ' || summary) LIKE '%innovation%' OR
    LOWER(title || ' ' || summary) LIKE '%startup%' OR
    LOWER(title || ' ' || summary) LIKE '%venture capital%' OR
    LOWER(title || ' ' || summary) LIKE '% vc %' OR
    LOWER(title || ' ' || summary) LIKE '%tech sector%' OR
    LOWER(title || ' ' || summary) LIKE '%saas%' OR
    LOWER(title || ' ' || summary) LIKE '%platform%'
  );

-- Show new distribution
SELECT 
    category, 
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
FROM news_articles
GROUP BY category
ORDER BY count DESC;

-- Show sample of each category
SELECT category, title
FROM (
    SELECT category, title, ROW_NUMBER() OVER (PARTITION BY category ORDER BY published_at DESC) as rn
    FROM news_articles
) sub
WHERE rn <= 2
ORDER BY category;
