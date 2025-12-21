"""
Real-Time Financial News Fetcher

Fetches live financial news from multiple sources and stores in PostgreSQL.
NO MOCK DATA - All news fetched from real APIs.

Supported APIs:
1. Finnhub (free tier) - General market news
2. Alpha Vantage (free tier) - Market news & sentiment
3. NewsAPI (free tier) - Financial news articles

Usage:
    python scripts/fetch_news.py --api finnhub --api-key YOUR_KEY
    python scripts/fetch_news.py --api alphavantage --api-key YOUR_KEY
    python scripts/fetch_news.py --api newsapi --api-key YOUR_KEY

Free API Keys:
- Finnhub: https://finnhub.io/register
- Alpha Vantage: https://www.alphavantage.co/support/#api-key
- NewsAPI: https://newsapi.org/register
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta

import aiohttp
import asyncpg
from loguru import logger

# Database configuration
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "user": os.getenv("POSTGRES_USER", "cift_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "cift_pass"),
    "database": os.getenv("POSTGRES_DB", "cift_markets"),
}


class NewsAPIClient:
    """Base class for news API clients"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_news(self) -> list[dict]:
        raise NotImplementedError


class FinnhubClient(NewsAPIClient):
    """Finnhub API client for market news"""

    BASE_URL = "https://finnhub.io/api/v1"

    async def fetch_news(self, category: str = "general") -> list[dict]:
        """
        Fetch market news from Finnhub.

        Categories: general, forex, crypto, merger
        Docs: https://finnhub.io/docs/api/market-news
        """
        url = f"{self.BASE_URL}/news"
        params = {
            "category": category,
            "token": self.api_key
        }

        logger.info(f"Fetching Finnhub news (category: {category})...")

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Finnhub API error: {response.status}")
                return []

            data = await response.json()
            logger.info(f"Received {len(data)} articles from Finnhub")

            # Transform to our schema
            articles = []
            for item in data:
                articles.append({
                    "title": item.get("headline", "")[:500],
                    "summary": item.get("summary", "")[:1000],
                    "content": item.get("summary", ""),  # Finnhub doesn't provide full content
                    "source": item.get("source", "Finnhub"),
                    "url": item.get("url"),
                    "author": None,
                    "published_at": datetime.fromtimestamp(item.get("datetime", 0)),
                    "category": category,
                    "sentiment": self._analyze_sentiment(item.get("summary", "")),
                    "symbols": self._extract_symbols(item.get("related", "")),
                    "image_url": item.get("image"),
                })

            return articles

    async def fetch_company_news(self, symbol: str, days: int = 7) -> list[dict]:
        """
        Fetch company-specific news from Finnhub.

        Docs: https://finnhub.io/docs/api/company-news
        """
        url = f"{self.BASE_URL}/company-news"

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            "symbol": symbol,
            "from": from_date,
            "to": to_date,
            "token": self.api_key
        }

        logger.info(f"Fetching Finnhub company news for {symbol}...")

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"Finnhub API error for {symbol}: {response.status}")
                return []

            data = await response.json()
            logger.info(f"Received {len(data)} articles for {symbol}")

            articles = []
            for item in data:
                articles.append({
                    "title": item.get("headline", "")[:500],
                    "summary": item.get("summary", "")[:1000],
                    "content": item.get("summary", ""),
                    "source": item.get("source", "Finnhub"),
                    "url": item.get("url"),
                    "author": None,
                    "published_at": datetime.fromtimestamp(item.get("datetime", 0)),
                    "category": "company",
                    "sentiment": self._analyze_sentiment(item.get("summary", "")),
                    "symbols": [symbol],
                    "image_url": item.get("image"),
                })

            return articles

    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis (can be improved with NLP)"""
        if not text:
            return "neutral"

        text_lower = text.lower()
        positive_words = ["gain", "rise", "surge", "profit", "growth", "success", "record", "beat"]
        negative_words = ["loss", "fall", "drop", "decline", "miss", "warning", "concern", "risk"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _extract_symbols(self, related: str) -> list[str]:
        """Extract stock symbols from related field"""
        if not related:
            return []
        return [s.strip() for s in related.split(",") if s.strip()]


class AlphaVantageClient(NewsAPIClient):
    """Alpha Vantage API client for news & sentiment"""

    BASE_URL = "https://www.alphavantage.co/query"

    async def fetch_news(self, topics: str = "financial_markets") -> list[dict]:
        """
        Fetch news and sentiment from Alpha Vantage.

        Topics: blockchain, earnings, ipo, mergers_and_acquisitions,
                financial_markets, economy_fiscal, economy_monetary, etc.
        Docs: https://www.alphavantage.co/documentation/#news-sentiment
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": topics,
            "apikey": self.api_key,
            "sort": "LATEST",
            "limit": 50
        }

        logger.info(f"Fetching Alpha Vantage news (topics: {topics})...")

        async with self.session.get(self.BASE_URL, params=params) as response:
            if response.status != 200:
                logger.error(f"Alpha Vantage API error: {response.status}")
                return []

            data = await response.json()

            if "feed" not in data:
                logger.error(f"Alpha Vantage error: {data.get('Note', 'Unknown error')}")
                return []

            feed = data["feed"]
            logger.info(f"Received {len(feed)} articles from Alpha Vantage")

            articles = []
            for item in feed:
                # Parse time published
                time_str = item.get("time_published", "")
                try:
                    published_at = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                except ValueError:
                    published_at = datetime.utcnow()

                # Get sentiment score
                sentiment_score = float(item.get("overall_sentiment_score", 0))
                if sentiment_score > 0.15:
                    sentiment = "positive"
                elif sentiment_score < -0.15:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                # Extract symbols from ticker sentiment
                symbols = []
                for ticker_item in item.get("ticker_sentiment", []):
                    symbols.append(ticker_item.get("ticker", ""))

                articles.append({
                    "title": item.get("title", "")[:500],
                    "summary": item.get("summary", "")[:1000],
                    "content": item.get("summary", ""),
                    "source": item.get("source", "Alpha Vantage"),
                    "url": item.get("url"),
                    "author": ", ".join(item.get("authors", [])) or None,
                    "published_at": published_at,
                    "category": self._map_category(item.get("topics", [])),
                    "sentiment": sentiment,
                    "symbols": [s for s in symbols if s],
                    "image_url": item.get("banner_image"),
                })

            return articles

    def _map_category(self, topics: list[dict]) -> str:
        """Map Alpha Vantage topics to our categories"""
        if not topics:
            return "market"

        topic_name = topics[0].get("topic", "").lower()

        mapping = {
            "earnings": "earnings",
            "ipo": "ipo",
            "mergers": "merger",
            "financial_markets": "market",
            "economy": "economics",
            "blockchain": "crypto",
            "technology": "tech",
        }

        for key, value in mapping.items():
            if key in topic_name:
                return value

        return "market"


class NewsAPIorgClient(NewsAPIClient):
    """NewsAPI.org client for general news"""

    BASE_URL = "https://newsapi.org/v2"

    async def fetch_news(self, query: str = "stock market OR trading OR finance") -> list[dict]:
        """
        Fetch news from NewsAPI.org.

        Docs: https://newsapi.org/docs/endpoints/everything
        """
        url = f"{self.BASE_URL}/everything"

        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
            "pageSize": 50,
            "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        }

        logger.info(f"Fetching NewsAPI.org articles (query: {query})...")

        async with self.session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"NewsAPI.org error: {response.status}")
                return []

            data = await response.json()

            if data.get("status") != "ok":
                logger.error(f"NewsAPI.org error: {data.get('message', 'Unknown error')}")
                return []

            articles_data = data.get("articles", [])
            logger.info(f"Received {len(articles_data)} articles from NewsAPI.org")

            articles = []
            for item in articles_data:
                # Parse published date (remove timezone for PostgreSQL compatibility)
                pub_str = item.get("publishedAt", "")
                try:
                    published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                    # Convert to naive UTC datetime
                    if published_at.tzinfo is not None:
                        published_at = published_at.replace(tzinfo=None)
                except Exception:
                    published_at = datetime.utcnow()

                # Extract symbols from title/description
                title = item.get("title") or ""
                description = item.get("description") or ""
                symbols = self._extract_symbols_from_text(title + " " + description)

                # Safely get values with None handling
                title_val = (item.get("title") or "")[:500]
                summary_val = (description or "")[:1000]
                content_val = (item.get("content") or description or "")

                # Filter out CORS-blocked image domains
                image_url = item.get("urlToImage")
                cors_blocked_domains = ['cryptoslate.com', 'medium.com', 'substack.com']
                if image_url and any(domain in image_url for domain in cors_blocked_domains):
                    image_url = None

                # Categorize based on content
                category = self._categorize_article(title_val, summary_val)

                articles.append({
                    "title": title_val,
                    "summary": summary_val,
                    "content": content_val,
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "url": item.get("url"),
                    "author": item.get("author"),
                    "published_at": published_at,
                    "category": category,
                    "sentiment": self._analyze_sentiment(description),
                    "symbols": symbols,
                    "image_url": image_url,
                })

            return articles

    def _categorize_article(self, title: str, summary: str) -> str:
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

    def _extract_symbols_from_text(self, text: str) -> list[str]:
        """Extract stock symbols from text (basic implementation)"""
        # Common symbols to look for
        common_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "AMD"]

        symbols = []
        text_upper = text.upper()
        for symbol in common_symbols:
            if symbol in text_upper:
                symbols.append(symbol)

        return symbols

    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis"""
        if not text:
            return "neutral"

        text_lower = text.lower()
        positive_words = ["gain", "rise", "surge", "profit", "growth", "success", "beat", "strong"]
        negative_words = ["loss", "fall", "drop", "decline", "miss", "warning", "concern", "weak"]

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"


async def save_articles_to_db(articles: list[dict], pool: asyncpg.Pool):
    """Save news articles to PostgreSQL database"""

    if not articles:
        logger.warning("No articles to save")
        return

    logger.info(f"Saving {len(articles)} articles to database...")

    # Insert articles (ignore duplicates based on URL)
    saved_count = 0
    skipped_count = 0

    async with pool.acquire() as conn:
        for article in articles:
            try:
                # Check if URL already exists
                existing = await conn.fetchval(
                    "SELECT id FROM news_articles WHERE url = $1",
                    article["url"]
                )

                if existing:
                    skipped_count += 1
                    continue

                # Insert article
                await conn.execute(
                    """
                    INSERT INTO news_articles
                    (title, summary, content, source, url, author, published_at,
                     category, sentiment, symbols, image_url)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    article["title"],
                    article["summary"],
                    article["content"],
                    article["source"],
                    article["url"],
                    article["author"],
                    article["published_at"],
                    article["category"],
                    article["sentiment"],
                    article["symbols"],
                    article["image_url"]
                )

                saved_count += 1

            except Exception as e:
                logger.error(f"Error saving article: {e}")
                continue

    logger.info(f"✅ Saved {saved_count} new articles, skipped {skipped_count} duplicates")


async def fetch_and_store_news(api_type: str, api_key: str):
    """Main function to fetch and store news"""

    # Create database connection pool
    pool = await asyncpg.create_pool(**DB_CONFIG, min_size=1, max_size=5)

    try:
        all_articles = []

        if api_type == "finnhub":
            async with FinnhubClient(api_key) as client:
                # Fetch general market news
                articles = await client.fetch_news("general")
                all_articles.extend(articles)

                await asyncio.sleep(1)  # Rate limiting

                # Fetch forex news
                forex_articles = await client.fetch_news("forex")
                all_articles.extend(forex_articles)

                await asyncio.sleep(1)

                # Fetch crypto news
                crypto_articles = await client.fetch_news("crypto")
                all_articles.extend(crypto_articles)

                # Fetch company-specific news for major symbols
                symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
                for symbol in symbols:
                    await asyncio.sleep(1)  # Rate limiting
                    company_articles = await client.fetch_company_news(symbol, days=7)
                    all_articles.extend(company_articles)

        elif api_type == "alphavantage":
            async with AlphaVantageClient(api_key) as client:
                # Fetch different topic categories
                topics = [
                    "financial_markets",
                    "earnings",
                    "economy_fiscal",
                    "technology"
                ]

                for topic in topics:
                    articles = await client.fetch_news(topic)
                    all_articles.extend(articles)
                    await asyncio.sleep(12)  # Alpha Vantage rate limit: 5 req/min

        elif api_type == "newsapi":
            async with NewsAPIorgClient(api_key) as client:
                # Fetch different queries
                queries = [
                    "stock market",
                    "trading",
                    "S&P 500",
                    "NASDAQ",
                    "Wall Street"
                ]

                for query in queries:
                    articles = await client.fetch_news(query)
                    all_articles.extend(articles)
                    await asyncio.sleep(1)  # Rate limiting

        else:
            logger.error(f"Unknown API type: {api_type}")
            return

        # Save to database
        await save_articles_to_db(all_articles, pool)

        logger.info(f"🎉 Successfully fetched and stored {len(all_articles)} articles!")

    finally:
        await pool.close()


def main():
    """CLI entry point"""

    parser = argparse.ArgumentParser(
        description="Fetch real financial news from APIs and store in database"
    )
    parser.add_argument(
        "--api",
        choices=["finnhub", "alphavantage", "newsapi"],
        required=True,
        help="News API to use"
    )
    parser.add_argument(
        "--api-key",
        required=False,
        help="API key for the selected service (or set via env: NEWSAPI_KEY, FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY)"
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        env_key_map = {
            "newsapi": "NEWSAPI_KEY",
            "finnhub": "FINNHUB_API_KEY",
            "alphavantage": "ALPHAVANTAGE_API_KEY"
        }
        env_var = env_key_map.get(args.api)
        api_key = os.getenv(env_var)

        if not api_key:
            logger.error(f"API key not provided. Use --api-key or set {env_var} environment variable")
            sys.exit(1)

    logger.info(f"Starting news fetch from {args.api}...")

    asyncio.run(fetch_and_store_news(args.api, api_key))


if __name__ == "__main__":
    main()
