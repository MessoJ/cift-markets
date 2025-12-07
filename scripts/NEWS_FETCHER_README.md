# 📰 Real-Time Financial News Fetcher

Fetch live financial news from real APIs and store in your PostgreSQL database.

**NO MOCK DATA** - All news comes from legitimate financial news sources.

---

## 🚀 Quick Start

### Option 1: Finnhub (Recommended - Easiest to Set Up)

**Get Free API Key:**
1. Visit: https://finnhub.io/register
2. Sign up (takes 30 seconds)
3. Copy your API key

**Run:**
```bash
cd c:\Users\mesof\cift-markets
docker-compose exec api python scripts/fetch_news.py --api finnhub --api-key YOUR_KEY_HERE
```

**What You Get:**
- ✅ General market news
- ✅ Forex news
- ✅ Crypto news
- ✅ Company-specific news (AAPL, GOOGL, MSFT, AMZN, TSLA)
- ✅ ~50-100 articles per run

---

### Option 2: Alpha Vantage (Best Sentiment Analysis)

**Get Free API Key:**
1. Visit: https://www.alphavantage.co/support/#api-key
2. Fill form (instant approval)
3. Copy your API key

**Run:**
```bash
docker-compose exec api python scripts/fetch_news.py --api alphavantage --api-key YOUR_KEY_HERE
```

**What You Get:**
- ✅ Market news with AI sentiment scores
- ✅ Topic categorization (earnings, IPO, M&A, etc.)
- ✅ Related stock tickers
- ✅ ~50 articles per run
- ⚠️ Rate limit: 5 requests/minute (script handles this automatically)

---

### Option 3: NewsAPI.org (Most Coverage)

**Get Free API Key:**
1. Visit: https://newsapi.org/register
2. Verify email
3. Copy your API key

**Run:**
```bash
docker-compose exec api python scripts/fetch_news.py --api newsapi --api-key YOUR_KEY_HERE
```

**What You Get:**
- ✅ Broad financial news coverage
- ✅ Multiple sources (Reuters, Bloomberg, WSJ, etc.)
- ✅ ~50 articles per run
- ⚠️ Free tier has some limitations on historical data

---

## 📊 What Gets Stored

All articles are saved to `news_articles` table with:

| Field | Description |
|-------|-------------|
| **title** | Article headline |
| **summary** | Short description (1-2 sentences) |
| **content** | Full article text (when available) |
| **source** | News source (Reuters, Bloomberg, etc.) |
| **url** | Original article URL |
| **author** | Article author |
| **published_at** | Publication timestamp |
| **category** | market, earnings, crypto, forex, etc. |
| **sentiment** | positive, negative, or neutral |
| **symbols** | Related stock tickers (e.g., ["AAPL", "MSFT"]) |
| **image_url** | Featured image |

---

## 🔄 Automation

### Run Daily with Cron

Create a cron job to fetch news automatically:

```bash
# Edit crontab
crontab -e

# Add this line to run every day at 9 AM
0 9 * * * cd /path/to/cift-markets && docker-compose exec -T api python scripts/fetch_news.py --api finnhub --api-key YOUR_KEY >> /var/log/news_fetch.log 2>&1
```

### Run Hourly for Real-Time Updates

```bash
# Run every hour
0 * * * * cd /path/to/cift-markets && docker-compose exec -T api python scripts/fetch_news.py --api finnhub --api-key YOUR_KEY
```

---

## 🔍 Verify Data

Check articles were saved:

```bash
# Check count
docker exec -i cift-postgres psql -U cift_user -d cift_markets -c "SELECT COUNT(*) FROM news_articles;"

# View latest 10 articles
docker exec -i cift-postgres psql -U cift_user -d cift_markets -c "SELECT title, source, published_at, sentiment FROM news_articles ORDER BY published_at DESC LIMIT 10;"

# View articles for specific symbol
docker exec -i cift-postgres psql -U cift_user -d cift_markets -c "SELECT title, sentiment FROM news_articles WHERE 'AAPL' = ANY(symbols) ORDER BY published_at DESC LIMIT 5;"
```

---

## 🎯 Advanced Usage

### Fetch News for Specific Symbols

Edit `fetch_news.py` line 482 to customize which stocks to track:

```python
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "AMD"]
```

### Combine Multiple APIs

Run all three APIs for maximum coverage:

```bash
# Morning: Finnhub
docker-compose exec api python scripts/fetch_news.py --api finnhub --api-key FINNHUB_KEY

# Afternoon: Alpha Vantage (wait 12 hours due to rate limit)
docker-compose exec api python scripts/fetch_news.py --api alphavantage --api-key ALPHAVANTAGE_KEY

# Evening: NewsAPI
docker-compose exec api python scripts/fetch_news.py --api newsapi --api-key NEWSAPI_KEY
```

---

## 📈 Expected Results

After running the script, your news page should show:

✅ **Latest financial news** in "News Feed" section  
✅ **Filtered by category** (market, earnings, crypto)  
✅ **Sentiment indicators** (green = positive, red = negative)  
✅ **Related symbols** shown for each article  
✅ **Source attribution** (Reuters, Bloomberg, etc.)

---

## 🐛 Troubleshooting

### "API error: 401"
- ❌ Invalid API key
- ✅ Double-check your API key is correct

### "API error: 429"
- ❌ Rate limit exceeded
- ✅ Wait a few minutes and try again
- ✅ Alpha Vantage: Max 5 requests/minute
- ✅ NewsAPI: Max 100 requests/day (free tier)

### "No articles saved"
- ❌ All articles were duplicates (already in database)
- ✅ This is normal! Script skips duplicates
- ✅ Try again tomorrow for fresh content

### "Database connection error"
- ❌ PostgreSQL not running
- ✅ Run: `docker-compose up -d postgres`

---

## 🎉 Success!

Once articles are fetched, refresh your browser:

```
http://localhost:3000/news
```

You should see real financial news populated! 📰✨

---

## 📝 Notes

- **Deduplication**: Script automatically skips articles already in database (based on URL)
- **Rate Limits**: Script respects API rate limits with automatic delays
- **Sentiment**: Basic keyword-based analysis (can be enhanced with NLP)
- **Symbols**: Extracted from article text and API metadata
- **Free Tier Limits**: 
  - Finnhub: 60 API calls/minute
  - Alpha Vantage: 5 API calls/minute, 500/day
  - NewsAPI: 100 requests/day

---

## 🔗 API Documentation

- **Finnhub**: https://finnhub.io/docs/api
- **Alpha Vantage**: https://www.alphavantage.co/documentation/#news-sentiment
- **NewsAPI**: https://newsapi.org/docs

---

## 🛡️ Security

**IMPORTANT**: Never commit your API keys to git!

Store in environment variables:
```bash
export FINNHUB_API_KEY=your_key_here
export ALPHAVANTAGE_API_KEY=your_key_here
export NEWSAPI_KEY=your_key_here

# Then use
docker-compose exec api python scripts/fetch_news.py --api finnhub --api-key $FINNHUB_API_KEY
```

Or create `.env` file (already in `.gitignore`):
```bash
FINNHUB_API_KEY=your_key_here
ALPHAVANTAGE_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
```
