"""
Recategorize existing news articles based on intelligent content analysis.
Updates the category field for all articles in the database.
"""

import asyncio
import asyncpg
import os
from loguru import logger


def categorize_article(title: str, summary: str) -> str:
    """
    Intelligently categorize article based on content.
    
    Categories: earnings, economics, technology, crypto, market
    """
    text = (title + " " + summary).lower()
    
    # Check for earnings-related keywords
    earnings_keywords = [
        "earnings", "eps", "revenue", "profit", "quarterly results",
        "q1", "q2", "q3", "q4", "fiscal", "guidance", "beats estimate",
        "misses estimate", "earnings call", "financial results"
    ]
    if any(keyword in text for keyword in earnings_keywords):
        return "earnings"
    
    # Check for economics/macro keywords
    economics_keywords = [
        "fed", "federal reserve", "interest rate", "inflation", "cpi",
        "unemployment", "gdp", "economic", "central bank", "monetary policy",
        "fiscal policy", "recession", "treasury", "bond yield", "jobs report",
        "non-farm payroll", "pmi", "consumer confidence"
    ]
    if any(keyword in text for keyword in economics_keywords):
        return "economics"
    
    # Check for crypto keywords
    crypto_keywords = [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
        "blockchain", "defi", "nft", "web3", "altcoin", "dogecoin",
        "binance", "coinbase", "solana", "cardano"
    ]
    if any(keyword in text for keyword in crypto_keywords):
        return "crypto"
    
    # Check for technology keywords
    tech_keywords = [
        "artificial intelligence", "ai", "machine learning", "ml",
        "technology", "tech", "software", "hardware", "cloud computing",
        "semiconductor", "chip", "processor", "innovation", "startup",
        "venture capital", "vc", "tech sector", "saas", "platform"
    ]
    if any(keyword in text for keyword in tech_keywords):
        return "technology"
    
    # Default to market
    return "market"


async def recategorize_articles():
    """Recategorize all existing articles"""
    
    # Connect to PostgreSQL
    conn = await asyncpg.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', '5432')),
        user=os.getenv('POSTGRES_USER', 'cift_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'cift_password'),
        database=os.getenv('POSTGRES_DB', 'cift_markets')
    )
    
    try:
        logger.info("Fetching all articles...")
        
        # Get all articles
        rows = await conn.fetch("""
            SELECT id, title, summary, category
            FROM news_articles
            ORDER BY published_at DESC
        """)
        
        total = len(rows)
        logger.info(f"Found {total} articles to recategorize")
        
        # Count by current category
        current_categories = {}
        for row in rows:
            cat = row['category']
            current_categories[cat] = current_categories.get(cat, 0) + 1
        
        logger.info(f"Current distribution: {current_categories}")
        
        # Recategorize and update
        updated = 0
        new_categories = {}
        
        for i, row in enumerate(rows, 1):
            article_id = row['id']
            title = row['title'] or ""
            summary = row['summary'] or ""
            old_category = row['category']
            
            # Determine new category
            new_category = categorize_article(title, summary)
            
            # Count new categories
            new_categories[new_category] = new_categories.get(new_category, 0) + 1
            
            # Update if changed
            if new_category != old_category:
                await conn.execute("""
                    UPDATE news_articles
                    SET category = $1
                    WHERE id = $2
                """, new_category, article_id)
                updated += 1
                
                if updated % 10 == 0:
                    logger.info(f"Progress: {i}/{total} processed, {updated} updated")
        
        logger.success(f"✅ Recategorization complete!")
        logger.info(f"   Total articles: {total}")
        logger.info(f"   Updated: {updated}")
        logger.info(f"   Unchanged: {total - updated}")
        logger.info(f"\nNew distribution:")
        for cat, count in sorted(new_categories.items()):
            logger.info(f"   {cat}: {count} articles")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  CIFT Markets - News Recategorization")
    logger.info("=" * 60)
    logger.info("")
    
    asyncio.run(recategorize_articles())
    
    logger.info("")
    logger.info("🎉 Done! Refresh the news page to see categorized articles.")
