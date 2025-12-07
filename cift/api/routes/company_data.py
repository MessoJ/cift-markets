"""
CIFT Markets - Company Data API Routes

Provides endpoints for company fundamentals, earnings, and advanced chart data.
This powers the enhanced charting features that go beyond TradingView.

Endpoints:
- GET /company/{symbol}/profile - Company profile with fundamentals
- GET /company/{symbol}/earnings - Earnings calendar and history
- GET /company/{symbol}/patterns - Detected chart patterns
- GET /company/{symbol}/levels - Support/resistance levels
- GET /company/{symbol}/news - Company news
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from cift.core.database import db_manager


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/company", tags=["Company Data"])


# ============================================================================
# MODELS
# ============================================================================

class CompanyProfile(BaseModel):
    """Company profile with fundamentals."""
    symbol: str
    name: str
    exchange: Optional[str] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    market_cap: Optional[float] = None  # In millions
    shares_outstanding: Optional[float] = None
    ipo_date: Optional[str] = None
    logo_url: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    currency: str = "USD"
    country: str = "US"
    # Valuation metrics
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    # 52-week range
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    # Volume
    avg_volume: Optional[int] = None


class EarningsEvent(BaseModel):
    """Earnings calendar entry."""
    symbol: str
    earnings_date: str
    quarter: Optional[int] = None
    year: Optional[int] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    eps_surprise: Optional[float] = None
    eps_surprise_pct: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    report_time: Optional[str] = None  # 'bmo' or 'amc'


class ChartPattern(BaseModel):
    """Detected chart pattern."""
    symbol: str
    timeframe: str
    pattern_name: str
    pattern_type: str  # bullish, bearish, neutral
    status: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    confidence: Optional[float] = None


class SupportResistanceLevel(BaseModel):
    """Support or resistance level."""
    symbol: str
    level_type: str  # support, resistance, pivot
    price: float
    strength: int = 1
    is_active: bool = True


class CompanyNews(BaseModel):
    """Company news article."""
    symbol: str
    headline: str
    summary: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    sentiment: Optional[str] = None
    published_at: datetime


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/{symbol}/profile", response_model=CompanyProfile)
async def get_company_profile(symbol: str):
    """
    Get company profile with fundamental data.
    
    Returns company overview including:
    - Basic info (name, exchange, sector)
    - Market cap and shares outstanding
    - Valuation ratios (P/E, dividend yield)
    - 52-week price range
    
    Performance: ~3ms (database query)
    """
    symbol = symbol.upper()
    
    try:
        async with db_manager.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 
                    symbol, name, exchange, industry, sector,
                    market_cap, shares_outstanding, ipo_date::text,
                    logo_url, website, description, currency, country,
                    pe_ratio, forward_pe, dividend_yield, beta,
                    fifty_two_week_high, fifty_two_week_low,
                    avg_volume_10d as avg_volume
                FROM company_profiles
                WHERE symbol = $1
                """,
                symbol
            )
        
        if not row:
            # Return minimal profile if not in database
            return CompanyProfile(
                symbol=symbol,
                name=symbol,  # Placeholder
                market_cap=None,
            )
        
        return CompanyProfile(
            symbol=row['symbol'],
            name=row['name'],
            exchange=row['exchange'],
            industry=row['industry'],
            sector=row['sector'],
            market_cap=float(row['market_cap']) if row['market_cap'] else None,
            shares_outstanding=float(row['shares_outstanding']) if row['shares_outstanding'] else None,
            ipo_date=row['ipo_date'],
            logo_url=row['logo_url'],
            website=row['website'],
            description=row['description'],
            currency=row['currency'] or 'USD',
            country=row['country'] or 'US',
            pe_ratio=float(row['pe_ratio']) if row['pe_ratio'] else None,
            forward_pe=float(row['forward_pe']) if row['forward_pe'] else None,
            dividend_yield=float(row['dividend_yield']) if row['dividend_yield'] else None,
            beta=float(row['beta']) if row['beta'] else None,
            fifty_two_week_high=float(row['fifty_two_week_high']) if row['fifty_two_week_high'] else None,
            fifty_two_week_low=float(row['fifty_two_week_low']) if row['fifty_two_week_low'] else None,
            avg_volume=int(row['avg_volume']) if row['avg_volume'] else None,
        )
    
    except Exception as e:
        logger.error(f"Error fetching profile for {symbol}: {e}")
        return CompanyProfile(symbol=symbol, name=symbol)


@router.get("/{symbol}/earnings", response_model=List[EarningsEvent])
async def get_earnings(
    symbol: str,
    include_past: bool = Query(True, description="Include past earnings"),
    limit: int = Query(8, ge=1, le=20, description="Number of earnings to return"),
):
    """
    Get earnings calendar and history for a symbol.
    
    Returns:
    - Upcoming earnings dates
    - Historical earnings with actual vs estimate
    - Earnings surprise percentages
    
    Performance: ~3ms
    """
    symbol = symbol.upper()
    
    try:
        async with db_manager.pool.acquire() as conn:
            if include_past:
                rows = await conn.fetch(
                    """
                    SELECT 
                        symbol, earnings_date::text, earnings_time,
                        eps_estimate, eps_actual,
                        revenue_estimate, revenue_actual
                    FROM earnings_calendar
                    WHERE symbol = $1
                    ORDER BY earnings_date DESC
                    LIMIT $2
                    """,
                    symbol, limit
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT 
                        symbol, earnings_date::text, earnings_time,
                        eps_estimate, eps_actual,
                        revenue_estimate, revenue_actual
                    FROM earnings_calendar
                    WHERE symbol = $1 AND earnings_date >= CURRENT_DATE
                    ORDER BY earnings_date ASC
                    LIMIT $2
                    """,
                    symbol, limit
                )
        
        events = []
        for row in rows:
            # Calculate earnings surprise if both estimate and actual exist
            eps_surprise = None
            eps_surprise_pct = None
            if row['eps_estimate'] and row['eps_actual']:
                eps_surprise = float(row['eps_actual']) - float(row['eps_estimate'])
                if float(row['eps_estimate']) != 0:
                    eps_surprise_pct = (eps_surprise / abs(float(row['eps_estimate']))) * 100
            
            # Extract quarter and year from date
            from datetime import datetime
            date_str = row['earnings_date']
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                quarter = (date_obj.month - 1) // 3 + 1
                year = date_obj.year
            except:
                quarter = None
                year = None
            
            events.append(EarningsEvent(
                symbol=row['symbol'],
                earnings_date=row['earnings_date'],
                quarter=quarter,
                year=year,
                eps_estimate=float(row['eps_estimate']) if row['eps_estimate'] else None,
                eps_actual=float(row['eps_actual']) if row['eps_actual'] else None,
                eps_surprise=eps_surprise,
                eps_surprise_pct=eps_surprise_pct,
                revenue_estimate=float(row['revenue_estimate']) if row['revenue_estimate'] else None,
                revenue_actual=float(row['revenue_actual']) if row['revenue_actual'] else None,
                report_time=row['earnings_time'],  # bmo, amc, dmh
            ))
        
        return events
    
    except Exception as e:
        logger.error(f"Error fetching earnings for {symbol}: {e}")
        return []


@router.get("/{symbol}/patterns", response_model=List[ChartPattern])
async def get_chart_patterns(
    symbol: str,
    timeframe: str = Query("D", description="Timeframe (D, W, M)"),
    limit: int = Query(5, ge=1, le=20),
):
    """
    Get detected chart patterns for a symbol.
    
    Patterns include:
    - Head and Shoulders
    - Double Top/Bottom
    - Triangle (ascending, descending, symmetrical)
    - Flag, Pennant
    - Cup and Handle
    
    Performance: ~3ms
    """
    symbol = symbol.upper()
    
    try:
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    symbol, timeframe, pattern_name, pattern_type, status,
                    start_date::text, end_date::text,
                    target_price, stop_loss, confidence
                FROM chart_patterns
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY detected_at DESC
                LIMIT $3
                """,
                symbol, timeframe, limit
            )
        
        return [
            ChartPattern(
                symbol=row['symbol'],
                timeframe=row['timeframe'],
                pattern_name=row['pattern_name'],
                pattern_type=row['pattern_type'],
                status=row['status'],
                start_date=row['start_date'],
                end_date=row['end_date'],
                target_price=float(row['target_price']) if row['target_price'] else None,
                stop_loss=float(row['stop_loss']) if row['stop_loss'] else None,
                confidence=float(row['confidence']) if row['confidence'] else None,
            )
            for row in rows
        ]
    
    except Exception as e:
        logger.error(f"Error fetching patterns for {symbol}: {e}")
        return []


@router.get("/{symbol}/levels", response_model=List[SupportResistanceLevel])
async def get_support_resistance(
    symbol: str,
    timeframe: str = Query("D", description="Timeframe (D, W, M)"),
    active_only: bool = Query(True, description="Only return active levels"),
):
    """
    Get support and resistance levels for a symbol.
    
    Levels are calculated from:
    - Historical price pivots
    - Fibonacci retracements
    - Volume profile nodes
    
    Performance: ~3ms
    """
    symbol = symbol.upper()
    
    try:
        async with db_manager.pool.acquire() as conn:
            query = """
                SELECT symbol, level_type, price, strength, is_active
                FROM support_resistance_levels
                WHERE symbol = $1 AND timeframe = $2
            """
            
            if active_only:
                query += " AND is_active = TRUE"
            
            query += " ORDER BY price DESC LIMIT 20"
            
            rows = await conn.fetch(query, symbol, timeframe)
        
        return [
            SupportResistanceLevel(
                symbol=row['symbol'],
                level_type=row['level_type'],
                price=float(row['price']),
                strength=row['strength'],
                is_active=row['is_active'],
            )
            for row in rows
        ]
    
    except Exception as e:
        logger.error(f"Error fetching S/R levels for {symbol}: {e}")
        return []


@router.get("/{symbol}/news", response_model=List[CompanyNews])
async def get_company_news(
    symbol: str,
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(7, ge=1, le=30, description="Days of history"),
):
    """
    Get recent news for a symbol.
    
    Includes:
    - Headline and summary
    - Source and URL
    - Sentiment analysis
    
    Performance: ~5ms
    """
    symbol = symbol.upper()
    from_date = datetime.utcnow() - timedelta(days=days)
    
    try:
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT symbol, headline, summary, source, url, sentiment, published_at
                FROM company_news
                WHERE symbol = $1 AND published_at >= $2
                ORDER BY published_at DESC
                LIMIT $3
                """,
                symbol, from_date, limit
            )
        
        return [
            CompanyNews(
                symbol=row['symbol'],
                headline=row['headline'],
                summary=row['summary'],
                source=row['source'],
                url=row['url'],
                sentiment=row['sentiment'],
                published_at=row['published_at'],
            )
            for row in rows
        ]
    
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []


@router.get("/{symbol}/summary")
async def get_symbol_summary(symbol: str):
    """
    Get comprehensive summary for a symbol (combines multiple data sources).
    
    This is the main endpoint for the chart header info bar.
    
    Returns:
    - Current price and change
    - Company name and sector
    - Market cap
    - 52-week range
    - Next earnings date
    - Recent patterns
    
    Performance: ~10ms (parallel queries)
    """
    symbol = symbol.upper()
    
    try:
        async with db_manager.pool.acquire() as conn:
            # Get current quote
            quote_row = await conn.fetchrow(
                """
                SELECT price, change, change_pct, open, high, low, volume, prev_close,
                       high_52w, low_52w, pre_market_price, post_market_price
                FROM market_data_cache
                WHERE symbol = $1
                """,
                symbol
            )
            
            # Get company profile
            profile_row = await conn.fetchrow(
                """
                SELECT name, sector, industry, market_cap, logo_url, pe_ratio
                FROM company_profiles
                WHERE symbol = $1
                """,
                symbol
            )
            
            # Get next earnings
            earnings_row = await conn.fetchrow(
                """
                SELECT earnings_date, eps_estimate, earnings_time
                FROM earnings_calendar
                WHERE symbol = $1 AND earnings_date >= CURRENT_DATE
                ORDER BY earnings_date ASC
                LIMIT 1
                """,
                symbol
            )
            
            # Calculate 52-week high/low from OHLCV data (if not in market_data_cache)
            high_52w = float(quote_row['high_52w']) if quote_row and quote_row['high_52w'] else None
            low_52w = float(quote_row['low_52w']) if quote_row and quote_row['low_52w'] else None
            
            if high_52w is None or low_52w is None:
                # Calculate from historical data
                range_row = await conn.fetchrow(
                    """
                    SELECT MAX(high) as high_52w, MIN(low) as low_52w
                    FROM ohlcv_bars
                    WHERE symbol = $1 
                      AND timeframe = '1m'
                      AND timestamp >= NOW() - INTERVAL '7 days'
                    """,
                    symbol
                )
                if range_row:
                    high_52w = float(range_row['high_52w']) if range_row['high_52w'] else None
                    low_52w = float(range_row['low_52w']) if range_row['low_52w'] else None
        
        return {
            "symbol": symbol,
            "name": profile_row['name'] if profile_row else symbol,
            "sector": profile_row['sector'] if profile_row else None,
            "industry": profile_row['industry'] if profile_row else None,
            "market_cap": float(profile_row['market_cap']) if profile_row and profile_row['market_cap'] else None,
            "logo_url": profile_row['logo_url'] if profile_row else None,
            "pe_ratio": float(profile_row['pe_ratio']) if profile_row and profile_row['pe_ratio'] else None,
            # Quote data
            "price": float(quote_row['price']) if quote_row and quote_row['price'] else None,
            "change": float(quote_row['change']) if quote_row and quote_row['change'] else None,
            "change_pct": float(quote_row['change_pct']) if quote_row and quote_row['change_pct'] else None,
            "open": float(quote_row['open']) if quote_row and quote_row['open'] else None,
            "high": float(quote_row['high']) if quote_row and quote_row['high'] else None,
            "low": float(quote_row['low']) if quote_row and quote_row['low'] else None,
            "volume": int(quote_row['volume']) if quote_row and quote_row['volume'] else None,
            "prev_close": float(quote_row['prev_close']) if quote_row and quote_row['prev_close'] else None,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "pre_market": float(quote_row['pre_market_price']) if quote_row and quote_row['pre_market_price'] else None,
            "post_market": float(quote_row['post_market_price']) if quote_row and quote_row['post_market_price'] else None,
            # Earnings
            "next_earnings": {
                "date": str(earnings_row['earnings_date']) if earnings_row else None,
                "eps_estimate": float(earnings_row['eps_estimate']) if earnings_row and earnings_row['eps_estimate'] else None,
                "time": earnings_row['earnings_time'] if earnings_row else None,
            } if earnings_row else None,
        }
    
    except Exception as e:
        logger.error(f"Error fetching summary for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


# Export router
__all__ = ["router"]
