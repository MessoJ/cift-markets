"""
CIFT Markets - News & Market Intelligence Service

Real-time financial news aggregation and analysis.
Supports multiple news providers and intelligent categorization.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

import aiohttp
from loguru import logger
from pydantic import BaseModel

from cift.core.database import get_postgres_pool
from cift.core.config import get_settings


class NewsArticle(BaseModel):
    id: Optional[str] = None
    title: str
    summary: Optional[str] = None
    content: Optional[str] = None
    url: str
    source: str
    author: Optional[str] = None
    published_at: datetime
    symbols: List[str] = []
    categories: List[str] = []
    sentiment: Optional[str] = None  # positive, negative, neutral
    importance: int = 1  # 1-5 scale
    image_url: Optional[str] = None


class NewsService:
    """Advanced financial news aggregation and intelligence service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.session = None
        
        # Check for API keys
        self.polygon_api_key = getattr(self.settings, 'polygon_api_key', '')
        self.alphavantage_api_key = getattr(self.settings, 'alphavantage_api_key', '')
        
        # News provider configurations
        self.providers = {
            "polygon": {
                "enabled": bool(self.polygon_api_key),  # Enable if API key exists
                "url": "https://api.polygon.io/v2/reference/news",
                "api_key": self.polygon_api_key
            },
            "alpha_vantage": {
                "enabled": bool(self.alphavantage_api_key),  # Enable if API key exists
                "url": "https://www.alphavantage.co/query",
                "params": {"function": "NEWS_SENTIMENT", "apikey": self.alphavantage_api_key}
            },
            "finnhub": {
                "enabled": False,  # Requires API key
                "url": "https://finnhub.io/api/v1/news",
                "headers": {"X-Finnhub-Token": ""}
            },
            "mock": {
                "enabled": not bool(self.polygon_api_key) and not bool(self.alphavantage_api_key),  # Only use mock if no real keys
                "url": None
            }
        }
        
        # News categories
        self.categories = [
            "earnings", "mergers", "ipo", "analyst_ratings", 
            "regulatory", "market_outlook", "economic_data",
            "technology", "healthcare", "finance", "energy"
        ]
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_latest_news(
        self, 
        symbols: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """Fetch latest financial news from multiple sources."""
        
        logger.info(f"Fetching latest news (symbols={symbols}, categories={categories})")
        
        all_articles = []
        
        # Try each enabled provider
        for provider_name, config in self.providers.items():
            if not config["enabled"]:
                continue
                
            try:
                articles = await self._fetch_from_provider(
                    provider_name, symbols, categories, limit
                )
                all_articles.extend(articles)
                
            except Exception as e:
                logger.warning(f"Failed to fetch from {provider_name}: {e}")
        
        # Deduplicate and sort by importance/recency
        unique_articles = self._deduplicate_articles(all_articles)
        sorted_articles = sorted(
            unique_articles,
            key=lambda x: (x.importance, x.published_at),
            reverse=True
        )
        
        # Store in database
        await self._store_articles(sorted_articles[:limit])
        
        return sorted_articles[:limit]
    
    async def _fetch_from_provider(
        self,
        provider: str,
        symbols: Optional[List[str]],
        categories: Optional[List[str]],
        limit: int
    ) -> List[NewsArticle]:
        """Fetch news from a specific provider."""
        
        if provider == "polygon":
            return await self._fetch_polygon_news(symbols, limit)
        elif provider == "mock":
            return await self._fetch_mock_news(symbols, categories, limit)
        elif provider == "alpha_vantage":
            return await self._fetch_alpha_vantage_news(symbols, limit)
        elif provider == "finnhub":
            return await self._fetch_finnhub_news(symbols, limit)
        else:
            return []
    
    async def _fetch_polygon_news(
        self,
        symbols: Optional[List[str]],
        limit: int
    ) -> List[NewsArticle]:
        """Fetch news from Polygon.io API."""
        
        articles = []
        
        try:
            # Build URL with API key
            base_url = "https://api.polygon.io/v2/reference/news"
            params = {
                "apiKey": self.polygon_api_key,
                "limit": min(limit, 100),  # Polygon max is 100
                "order": "desc",
                "sort": "published_utc"
            }
            
            # Add ticker filter if specified
            if symbols and len(symbols) > 0:
                params["ticker"] = symbols[0]  # Polygon only supports one ticker at a time
            
            async with self.session.get(base_url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"Polygon news API returned {response.status}")
                    return []
                
                data = await response.json()
                results = data.get("results", [])
                
                for item in results:
                    try:
                        # Parse published time
                        published_str = item.get("published_utc", "")
                        if published_str:
                            published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                        else:
                            published_at = datetime.utcnow()
                        
                        # Get tickers mentioned
                        tickers = item.get("tickers", [])
                        
                        # Determine sentiment from insights
                        insights = item.get("insights", [])
                        sentiment = "neutral"
                        if insights:
                            sentiments = [i.get("sentiment", "neutral") for i in insights]
                            if "bullish" in sentiments or "positive" in sentiments:
                                sentiment = "positive"
                            elif "bearish" in sentiments or "negative" in sentiments:
                                sentiment = "negative"
                        
                        # Basic sentiment from title if no insights
                        if sentiment == "neutral":
                            title_lower = item.get("title", "").lower()
                            if any(w in title_lower for w in ["surge", "rally", "gain", "beat", "growth"]):
                                sentiment = "positive"
                            elif any(w in title_lower for w in ["fall", "drop", "miss", "loss", "crash"]):
                                sentiment = "negative"
                        
                        article = NewsArticle(
                            id=str(item.get("id", str(uuid4()))),
                            title=item.get("title", ""),
                            summary=item.get("description", ""),
                            content=item.get("description", ""),
                            url=item.get("article_url", ""),
                            source=item.get("publisher", {}).get("name", "Unknown"),
                            author=item.get("author", ""),
                            published_at=published_at,
                            symbols=tickers,
                            categories=["market_news"],
                            sentiment=sentiment,
                            importance=3 if len(tickers) > 0 else 2,
                            image_url=item.get("image_url", "")
                        )
                        
                        articles.append(article)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing Polygon article: {e}")
                
                logger.success(f"Fetched {len(articles)} news articles from Polygon.io")
                
        except Exception as e:
            logger.error(f"Failed to fetch Polygon news: {e}")
        
        return articles
    
    async def _fetch_mock_news(
        self,
        symbols: Optional[List[str]],
        categories: Optional[List[str]], 
        limit: int
    ) -> List[NewsArticle]:
        """Generate realistic mock news for testing."""
        
        # Mock news templates
        templates = [
            {
                "title": "{symbol} Reports Strong Q4 Earnings, Beats Estimates",
                "summary": "{symbol} exceeded analyst expectations with strong revenue growth.",
                "categories": ["earnings"],
                "sentiment": "positive",
                "importance": 4
            },
            {
                "title": "Analyst Upgrades {symbol} to Buy on Strong Outlook",
                "summary": "Major investment firm raises price target following positive developments.",
                "categories": ["analyst_ratings"],
                "sentiment": "positive", 
                "importance": 3
            },
            {
                "title": "{symbol} Faces Regulatory Scrutiny Over Recent Practices",
                "summary": "Government agencies investigating potential compliance issues.",
                "categories": ["regulatory"],
                "sentiment": "negative",
                "importance": 3
            },
            {
                "title": "Breaking: {symbol} Announces Major Acquisition Deal",
                "summary": "Company expands market presence through strategic acquisition.",
                "categories": ["mergers"],
                "sentiment": "positive",
                "importance": 5
            },
            {
                "title": "Market Outlook: Technology Sector Shows Resilience",
                "summary": "Analysts remain optimistic about tech stocks despite volatility.",
                "categories": ["market_outlook", "technology"],
                "sentiment": "positive",
                "importance": 2
            }
        ]
        
        # Generate articles
        articles = []
        target_symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
        
        for i in range(min(limit, len(templates) * 3)):
            template = templates[i % len(templates)]
            symbol = target_symbols[i % len(target_symbols)]
            
            # Skip if categories filter doesn't match
            if categories and not any(cat in template["categories"] for cat in categories):
                continue
            
            article = NewsArticle(
                id=str(uuid4()),
                title=template["title"].format(symbol=symbol),
                summary=template["summary"].format(symbol=symbol),
                url=f"https://example-news.com/article-{i}",
                source="Mock Financial News",
                published_at=datetime.utcnow() - timedelta(hours=i, minutes=i*15),
                symbols=[symbol] if "{symbol}" in template["title"] else [],
                categories=template["categories"],
                sentiment=template["sentiment"],
                importance=template["importance"]
            )
            
            articles.append(article)
        
        return articles
    
    async def _fetch_alpha_vantage_news(
        self, 
        symbols: Optional[List[str]], 
        limit: int
    ) -> List[NewsArticle]:
        """Fetch from Alpha Vantage News API."""
        config = self.providers["alpha_vantage"]
        params = config["params"].copy()
        
        # Add tickers if specified
        if symbols:
            params["tickers"] = ",".join(symbols)
        else:
            params["topics"] = "financial_markets"
            
        params["limit"] = str(limit)
        params["sort"] = "LATEST"
        
        try:
            async with self.session.get(config["url"], params=params) as response:
                if response.status != 200:
                    logger.warning(f"Alpha Vantage API returned {response.status}")
                    return []
                
                data = await response.json()
                
                if "feed" not in data:
                    logger.warning(f"Alpha Vantage response missing 'feed': {data.get('Note', 'Unknown')}")
                    return []
                
                articles = []
                for item in data["feed"]:
                    # Parse time
                    time_str = item.get("time_published", "")
                    try:
                        published_at = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                    except:
                        published_at = datetime.utcnow()
                    
                    # Sentiment
                    sentiment_score = float(item.get("overall_sentiment_score", 0))
                    if sentiment_score > 0.15:
                        sentiment = "positive"
                    elif sentiment_score < -0.15:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"
                        
                    # Symbols
                    item_symbols = [t.get("ticker") for t in item.get("ticker_sentiment", [])]
                    
                    article = NewsArticle(
                        id=str(uuid4()),
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        content=item.get("summary", ""),
                        url=item.get("url", ""),
                        source=item.get("source", "Alpha Vantage"),
                        author=", ".join(item.get("authors", [])),
                        published_at=published_at,
                        symbols=item_symbols,
                        sentiment=sentiment,
                        image_url=item.get("banner_image")
                    )
                    articles.append(article)
                    
                return articles
                
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")
            return []
    
    async def _fetch_finnhub_news(
        self,
        symbols: Optional[List[str]],
        limit: int
    ) -> List[NewsArticle]:
        """Fetch from Finnhub News API.""" 
        # TODO: Implement Finnhub integration
        logger.info("Finnhub news integration not implemented")
        return []
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Simple deduplication by title
            title_key = article.title.lower().strip()
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    async def _store_articles(self, articles: List[NewsArticle]):
        """Store articles in database."""
        
        if not articles:
            return
            
        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            for article in articles:
                try:
                    await conn.execute("""
                        INSERT INTO news_articles (
                            id, title, summary, content, url, source, author,
                            published_at, symbols, categories, sentiment, 
                            importance, image_url, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                        ON CONFLICT (url) DO UPDATE SET
                            title = EXCLUDED.title,
                            summary = EXCLUDED.summary,
                            symbols = EXCLUDED.symbols,
                            categories = EXCLUDED.categories,
                            sentiment = EXCLUDED.sentiment,
                            importance = EXCLUDED.importance,
                            updated_at = NOW()
                    """,
                        article.id or str(uuid4()),
                        article.title,
                        article.summary, 
                        article.content,
                        article.url,
                        article.source,
                        article.author,
                        article.published_at,
                        json.dumps(article.symbols),
                        json.dumps(article.categories),
                        article.sentiment,
                        article.importance,
                        article.image_url,
                        datetime.utcnow()
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to store article {article.title}: {e}")
        
        logger.success(f"Stored {len(articles)} news articles")
    
    async def get_news_by_symbol(
        self, 
        symbol: str, 
        limit: int = 20
    ) -> List[NewsArticle]:
        """Get news articles related to a specific symbol."""
        
        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM news_articles
                WHERE symbols::text LIKE $1
                ORDER BY published_at DESC, importance DESC
                LIMIT $2
            """, f'%"{symbol}"%', limit)
        
        articles = []
        for row in rows:
            articles.append(NewsArticle(
                id=row['id'],
                title=row['title'],
                summary=row['summary'],
                content=row['content'],
                url=row['url'],
                source=row['source'],
                author=row['author'],
                published_at=row['published_at'],
                symbols=json.loads(row['symbols'] or '[]'),
                categories=json.loads(row['categories'] or '[]'),
                sentiment=row['sentiment'],
                importance=row['importance'],
                image_url=row['image_url']
            ))
        
        return articles
    
    async def get_news_by_category(
        self,
        category: str,
        limit: int = 20
    ) -> List[NewsArticle]:
        """Get news articles by category."""
        
        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM news_articles
                WHERE categories::text LIKE $1
                ORDER BY published_at DESC, importance DESC
                LIMIT $2
            """, f'%"{category}"%', limit)
        
        articles = []
        for row in rows:
            articles.append(NewsArticle(
                id=row['id'],
                title=row['title'],
                summary=row['summary'],
                content=row['content'],
                url=row['url'],
                source=row['source'],
                author=row['author'],
                published_at=row['published_at'],
                symbols=json.loads(row['symbols'] or '[]'),
                categories=json.loads(row['categories'] or '[]'),
                sentiment=row['sentiment'],
                importance=row['importance'],
                image_url=row['image_url']
            ))
        
        return articles
    
    async def get_market_movers_news(self) -> Dict[str, List[NewsArticle]]:
        """Get news for market movers (gainers/losers)."""
        
        # Get top gainers/losers from market data
        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            # Get top gainers
            gainers = await conn.fetch("""
                SELECT symbol FROM market_data_cache
                WHERE change_percent > 5
                ORDER BY change_percent DESC
                LIMIT 10
            """)
            
            # Get top losers
            losers = await conn.fetch("""
                SELECT symbol FROM market_data_cache  
                WHERE change_percent < -5
                ORDER BY change_percent ASC
                LIMIT 10
            """)
        
        result = {"gainers": [], "losers": []}
        
        # Get news for gainers
        for row in gainers:
            news = await self.get_news_by_symbol(row['symbol'], 3)
            result["gainers"].extend(news)
        
        # Get news for losers
        for row in losers:
            news = await self.get_news_by_symbol(row['symbol'], 3)
            result["losers"].extend(news)
        
        return result
    
    async def cleanup_old_news(self, days_to_keep: int = 30):
        """Clean up old news articles."""
        
        pool = await get_postgres_pool()
        
        async with pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM news_articles
                WHERE published_at < NOW() - INTERVAL '%s days'
            """, days_to_keep)
            
            logger.info(f"Cleaned up old news articles: {result}")


# Global news service instance
_news_service = None

def get_news_service() -> NewsService:
    """Get the global news service instance."""
    global _news_service
    if _news_service is None:
        _news_service = NewsService()
    return _news_service


# Background task for news fetching
async def fetch_news_background():
    """Background task to fetch latest news periodically."""
    
    logger.info("Starting news background fetch...")
    
    try:
        async with NewsService() as news_service:
            # Fetch general market news
            await news_service.fetch_latest_news(limit=100)
            
            # Fetch news for popular symbols
            popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]
            for symbol in popular_symbols:
                try:
                    await news_service.fetch_latest_news(symbols=[symbol], limit=10)
                except Exception as e:
                    logger.warning(f"Failed to fetch news for {symbol}: {e}")
    
    except Exception as e:
        logger.error(f"News background fetch failed: {e}")


if __name__ == "__main__":
    # Test the news service
    async def test_news():
        async with NewsService() as service:
            # Test mock news generation
            articles = await service.fetch_latest_news(
                symbols=["AAPL", "MSFT"], 
                limit=10
            )
            
            print(f"Fetched {len(articles)} articles:")
            for article in articles[:3]:
                print(f"- {article.title}")
                print(f"  Source: {article.source}")
                print(f"  Symbols: {article.symbols}")
                print(f"  Sentiment: {article.sentiment}")
                print()
    
    asyncio.run(test_news())
