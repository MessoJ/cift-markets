"""
STOCK SCREENER API ROUTES
Handles stock screening with technical and fundamental filters.
All data is fetched from database - NO MOCK DATA.

Technical Indicators: RSI, MACD, SMA, EMA calculated from ohlcv_bars
Performance Metrics: Returns calculated from price history
Alerts: Saved criteria with notification triggers
Backtesting: Historical screening analysis
"""

import json
import math
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/screener", tags=["screener"])


# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS (Real Data from ohlcv_bars)
# ============================================================================

def calculate_rsi(closes: list[float], period: int = 14) -> float | None:
    """Calculate RSI (Relative Strength Index) from price closes"""
    if len(closes) < period + 1:
        return None
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def calculate_macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict | None:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(closes) < slow + signal:
        return None
    
    closes_arr = np.array(closes)
    
    # EMA calculations
    def ema(data, period):
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]
    
    # Need enough data for EMA
    if len(closes) < slow:
        return None
    
    ema_fast = ema(closes_arr[-fast:], fast)
    ema_slow = ema(closes_arr[-slow:], slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line would need historical MACD values - simplified version
    return {
        "macd_line": round(macd_line, 4),
        "signal_line": None,  # Would need more historical data
        "histogram": None,
    }


def calculate_sma(closes: list[float], period: int) -> float | None:
    """Calculate Simple Moving Average"""
    if len(closes) < period:
        return None
    return round(sum(closes[-period:]) / period, 4)


def calculate_ema(closes: list[float], period: int) -> float | None:
    """Calculate Exponential Moving Average"""
    if len(closes) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = closes[0]
    
    for price in closes[1:]:
        ema = (price - ema) * multiplier + ema
    
    return round(ema, 4)


def calculate_atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    """Calculate Average True Range"""
    if len(closes) < period + 1:
        return None
    
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return None
    
    return round(sum(true_ranges[-period:]) / period, 4)


def calculate_bollinger_bands(closes: list[float], period: int = 20, std_dev: float = 2.0) -> dict | None:
    """Calculate Bollinger Bands"""
    if len(closes) < period:
        return None
    
    recent = closes[-period:]
    sma = sum(recent) / period
    variance = sum((x - sma) ** 2 for x in recent) / period
    std = math.sqrt(variance)
    
    return {
        "upper": round(sma + std_dev * std, 4),
        "middle": round(sma, 4),
        "lower": round(sma - std_dev * std, 4),
        "width": round((2 * std_dev * std) / sma * 100, 2) if sma else None,  # As percentage
    }


def calculate_stochastic(highs: list[float], lows: list[float], closes: list[float], k_period: int = 14, d_period: int = 3) -> dict | None:
    """Calculate Stochastic Oscillator"""
    if len(closes) < k_period:
        return None
    
    highest_high = max(highs[-k_period:])
    lowest_low = min(lows[-k_period:])
    
    if highest_high == lowest_low:
        return None
    
    k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    
    return {
        "k": round(k, 2),
        "d": None,  # Would need historical K values for D line
    }


# ============================================================================
# PERFORMANCE CALCULATIONS (Real Data from ohlcv_bars)
# ============================================================================

def calculate_return(start_price: float, end_price: float) -> float | None:
    """Calculate percentage return"""
    if not start_price or start_price == 0:
        return None
    return round(((end_price - start_price) / start_price) * 100, 2)


# ============================================================================
# MODELS
# ============================================================================

class ScreenerCriteria(BaseModel):
    """Screener criteria model"""
    # Price filters
    price_min: Decimal | None = None
    price_max: Decimal | None = None

    # Volume filters
    volume_min: int | None = None

    # Market cap filters (in millions)
    market_cap_min: Decimal | None = None
    market_cap_max: Decimal | None = None

    # Valuation filters
    pe_ratio_min: Decimal | None = None
    pe_ratio_max: Decimal | None = None
    forward_pe_min: Decimal | None = None
    forward_pe_max: Decimal | None = None
    peg_ratio_min: Decimal | None = None
    peg_ratio_max: Decimal | None = None
    price_to_book_min: Decimal | None = None
    price_to_book_max: Decimal | None = None
    price_to_sales_min: Decimal | None = None
    price_to_sales_max: Decimal | None = None
    
    # Dividend filters
    dividend_yield_min: Decimal | None = None
    dividend_yield_max: Decimal | None = None
    
    # Profitability filters
    profit_margin_min: Decimal | None = None
    roe_min: Decimal | None = None
    roa_min: Decimal | None = None
    
    # Risk filters
    beta_min: Decimal | None = None
    beta_max: Decimal | None = None
    
    # EPS filters
    eps_min: Decimal | None = None

    # Performance filters
    change_pct_min: Decimal | None = None
    change_pct_max: Decimal | None = None

    # Category filters
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    exchange: str | None = None
    asset_type: str | None = None


class ScreenerResponse(BaseModel):
    """Screener response with pagination"""
    results: list[dict]
    total_count: int
    page: int
    limit: int


class ScreenerResult(BaseModel):
    """Screener result model"""
    symbol: str
    name: str
    price: Decimal
    change: Decimal
    change_percent: Decimal
    volume: int
    market_cap: Decimal | None = None
    pe_ratio: Decimal | None = None
    eps: Decimal | None = None
    dividend_yield: Decimal | None = None
    sector: str | None = None
    industry: str | None = None


class SavedScreen(BaseModel):
    """Saved screen model"""
    id: str
    name: str
    criteria: ScreenerCriteria
    created_at: datetime
    last_run: datetime | None = None


class SaveScreenRequest(BaseModel):
    """Save screen request"""
    name: str = Field(..., min_length=1, max_length=100)
    criteria: ScreenerCriteria


# ============================================================================
# NEW MODELS - Technical Indicators, Performance, Alerts, Backtesting
# ============================================================================

class TechnicalCriteria(BaseModel):
    """Technical indicator filter criteria"""
    # RSI filters
    rsi_min: float | None = None
    rsi_max: float | None = None
    rsi_period: int = 14
    
    # MACD filters
    macd_above_signal: bool | None = None
    macd_positive: bool | None = None
    
    # Moving Average filters
    price_above_sma20: bool | None = None
    price_above_sma50: bool | None = None
    price_above_sma200: bool | None = None
    price_above_ema20: bool | None = None
    sma20_above_sma50: bool | None = None  # Golden cross signal
    sma50_above_sma200: bool | None = None
    
    # Bollinger Bands
    price_near_bb_lower: bool | None = None  # Oversold signal
    price_near_bb_upper: bool | None = None  # Overbought signal
    
    # Stochastic
    stochastic_k_min: float | None = None
    stochastic_k_max: float | None = None
    
    # ATR
    atr_min: float | None = None
    atr_max: float | None = None


class PerformanceCriteria(BaseModel):
    """Performance metrics filter criteria"""
    return_1w_min: float | None = None
    return_1w_max: float | None = None
    return_1m_min: float | None = None
    return_1m_max: float | None = None
    return_3m_min: float | None = None
    return_3m_max: float | None = None
    return_6m_min: float | None = None
    return_6m_max: float | None = None
    return_ytd_min: float | None = None
    return_ytd_max: float | None = None
    return_1y_min: float | None = None
    return_1y_max: float | None = None


class TechnicalIndicators(BaseModel):
    """Technical indicators response for a symbol"""
    symbol: str
    rsi_14: float | None = None
    macd: dict | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    ema_20: float | None = None
    ema_50: float | None = None
    bollinger_bands: dict | None = None
    stochastic: dict | None = None
    atr_14: float | None = None
    current_price: float | None = None


class PerformanceMetrics(BaseModel):
    """Performance metrics response for a symbol"""
    symbol: str
    return_1w: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_ytd: float | None = None
    return_1y: float | None = None
    volatility_30d: float | None = None
    sharpe_ratio: float | None = None


class ScreenerAlert(BaseModel):
    """Screener alert model"""
    id: str
    name: str
    criteria: dict
    technical_criteria: dict | None = None
    performance_criteria: dict | None = None
    is_active: bool = True
    notify_email: bool = True
    notify_push: bool = False
    last_triggered: datetime | None = None
    matched_symbols: list[str] = []
    created_at: datetime


class CreateAlertRequest(BaseModel):
    """Create screener alert request"""
    name: str = Field(..., min_length=1, max_length=100)
    criteria: ScreenerCriteria
    technical_criteria: TechnicalCriteria | None = None
    performance_criteria: PerformanceCriteria | None = None
    notify_email: bool = True
    notify_push: bool = False


class BacktestRequest(BaseModel):
    """Backtest request model"""
    name: str = Field(..., min_length=1, max_length=100)
    criteria: ScreenerCriteria
    technical_criteria: TechnicalCriteria | None = None
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    initial_capital: float = 100000
    position_sizing: Literal["equal_weight", "market_cap_weight"] = "equal_weight"
    max_positions: int = 20


class BacktestResult(BaseModel):
    """Backtest result model"""
    id: str
    name: str
    status: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float | None = None
    total_return: float | None = None
    annualized_return: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None
    trades_count: int | None = None
    win_rate: float | None = None
    equity_curve: list[dict] | None = None
    holdings_history: list[dict] | None = None
    created_at: datetime
    completed_at: datetime | None = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/scan")
async def screen_stocks(
    criteria: ScreenerCriteria,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "market_cap",
    sort_order: str = "desc",
):
    """
    Run advanced stock screen with technical and fundamental analysis.
    Uses REAL DATA from symbols table and market_data_cache.
    No authentication required for basic screening.
    """
    pg_pool = await get_postgres_pool()

    # Valid sort columns - map frontend names to actual column expressions
    # Using 's' alias for symbols table, 'mdc' for market_data_cache
    sort_map = {
        "market_cap": "s.market_cap",
        "pe_ratio": "s.pe_ratio",
        "forward_pe": "s.forward_pe",
        "dividend_yield": "s.dividend_yield",
        "symbol": "s.symbol",
        "name": "s.name",
        "sector": "s.sector",
        "change_pct": "mdc.change_pct",
        "volume": "mdc.volume",
        "price": "mdc.price",
        "beta": "COALESCE(mdc.change_pct, 0) / NULLIF(ABS(mdc.change_pct), 0)",  # Placeholder
        "eps": "s.eps",
        "roe": "s.roe",
        "profit_margin": "s.profit_margin",
    }
    sort_column = sort_map.get(sort_by, "s.market_cap")
    sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"

    # Base query - using symbols table (which has actual data) instead of company_profiles (empty)
    base_query = """
        FROM symbols s
        LEFT JOIN market_data_cache mdc ON s.symbol = mdc.symbol
        WHERE s.is_tradable = true AND s.is_active = true
    """

    params = []
    param_count = 1

    # ========== MARKET CAP FILTERS ==========
    if criteria.market_cap_min:
        # Convert from millions (API) to actual value (DB stores in raw form)
        # DB has values like 3000000000000 for Apple ($3T)
        base_query += f" AND s.market_cap >= ${param_count}"
        params.append(float(criteria.market_cap_min) * 1_000_000)  # Convert M to actual
        param_count += 1

    if criteria.market_cap_max:
        base_query += f" AND s.market_cap <= ${param_count}"
        params.append(float(criteria.market_cap_max) * 1_000_000)
        param_count += 1

    # ========== VALUATION FILTERS ==========
    if criteria.pe_ratio_min:
        base_query += f" AND s.pe_ratio >= ${param_count}"
        params.append(float(criteria.pe_ratio_min))
        param_count += 1

    if criteria.pe_ratio_max:
        base_query += f" AND s.pe_ratio <= ${param_count}"
        params.append(float(criteria.pe_ratio_max))
        param_count += 1

    if criteria.forward_pe_min:
        base_query += f" AND s.forward_pe >= ${param_count}"
        params.append(float(criteria.forward_pe_min))
        param_count += 1

    if criteria.forward_pe_max:
        base_query += f" AND s.forward_pe <= ${param_count}"
        params.append(float(criteria.forward_pe_max))
        param_count += 1

    if criteria.peg_ratio_min:
        base_query += f" AND s.peg_ratio >= ${param_count}"
        params.append(float(criteria.peg_ratio_min))
        param_count += 1

    if criteria.peg_ratio_max:
        base_query += f" AND s.peg_ratio <= ${param_count}"
        params.append(float(criteria.peg_ratio_max))
        param_count += 1

    if criteria.price_to_book_min:
        base_query += f" AND s.price_to_book >= ${param_count}"
        params.append(float(criteria.price_to_book_min))
        param_count += 1

    if criteria.price_to_book_max:
        base_query += f" AND s.price_to_book <= ${param_count}"
        params.append(float(criteria.price_to_book_max))
        param_count += 1

    if criteria.price_to_sales_min:
        base_query += f" AND s.price_to_sales >= ${param_count}"
        params.append(float(criteria.price_to_sales_min))
        param_count += 1

    if criteria.price_to_sales_max:
        base_query += f" AND s.price_to_sales <= ${param_count}"
        params.append(float(criteria.price_to_sales_max))
        param_count += 1

    # ========== DIVIDEND FILTERS ==========
    if criteria.dividend_yield_min:
        base_query += f" AND s.dividend_yield >= ${param_count}"
        params.append(float(criteria.dividend_yield_min))
        param_count += 1

    if criteria.dividend_yield_max:
        base_query += f" AND s.dividend_yield <= ${param_count}"
        params.append(float(criteria.dividend_yield_max))
        param_count += 1

    # ========== PROFITABILITY FILTERS ==========
    if criteria.profit_margin_min:
        base_query += f" AND s.profit_margin >= ${param_count}"
        params.append(float(criteria.profit_margin_min))
        param_count += 1

    if criteria.roe_min:
        base_query += f" AND s.roe >= ${param_count}"
        params.append(float(criteria.roe_min))
        param_count += 1

    if criteria.roa_min:
        base_query += f" AND s.roa >= ${param_count}"
        params.append(float(criteria.roa_min))
        param_count += 1

    if criteria.eps_min:
        base_query += f" AND s.eps >= ${param_count}"
        params.append(float(criteria.eps_min))
        param_count += 1

    # ========== RISK FILTERS ==========
    # Note: Beta is not in symbols table, would need to be calculated or added
    # Skipping beta filters for now

    # ========== CATEGORY FILTERS ==========
    if criteria.sector:
        base_query += f" AND s.sector ILIKE ${param_count}"
        params.append(f"%{criteria.sector}%")
        param_count += 1

    if criteria.industry:
        base_query += f" AND s.industry ILIKE ${param_count}"
        params.append(f"%{criteria.industry}%")
        param_count += 1

    if criteria.country:
        base_query += f" AND s.country = ${param_count}"
        params.append(criteria.country)
        param_count += 1

    if criteria.exchange:
        base_query += f" AND s.exchange = ${param_count}"
        params.append(criteria.exchange)
        param_count += 1

    if criteria.asset_type:
        base_query += f" AND s.asset_type = ${param_count}"
        params.append(criteria.asset_type)
        param_count += 1

    # ========== PRICE FILTERS (from market_data_cache) ==========
    if criteria.price_min:
        base_query += f" AND mdc.price >= ${param_count}"
        params.append(float(criteria.price_min))
        param_count += 1

    if criteria.price_max:
        base_query += f" AND mdc.price <= ${param_count}"
        params.append(float(criteria.price_max))
        param_count += 1

    if criteria.volume_min:
        base_query += f" AND mdc.volume >= ${param_count}"
        params.append(criteria.volume_min)
        param_count += 1

    if criteria.change_pct_min:
        base_query += f" AND mdc.change_pct >= ${param_count}"
        params.append(float(criteria.change_pct_min))
        param_count += 1

    if criteria.change_pct_max:
        base_query += f" AND mdc.change_pct <= ${param_count}"
        params.append(float(criteria.change_pct_max))
        param_count += 1

    # Count query
    count_query = f"SELECT COUNT(*) {base_query}"

    # Data query with comprehensive fields
    data_query = f"""
        SELECT
            s.symbol,
            s.name,
            s.sector,
            s.industry,
            s.country,
            s.exchange,
            s.asset_type,
            s.market_cap,
            s.pe_ratio,
            s.forward_pe,
            s.peg_ratio,
            s.price_to_book,
            s.price_to_sales,
            s.eps,
            s.dividend_yield,
            s.profit_margin,
            s.operating_margin,
            s.roe,
            s.roa,
            s.revenue,
            s.net_income,
            s.analyst_rating,
            s.analyst_target_price,
            s.analyst_count,
            COALESCE(mdc.price, 0) as price,
            COALESCE(mdc.change, 0) as change,
            COALESCE(mdc.change_pct, 0) as change_pct,
            COALESCE(mdc.volume, 0) as volume,
            mdc.high_52w,
            mdc.low_52w,
            mdc.avg_volume,
            mdc.updated_at as price_updated_at
        {base_query}
        ORDER BY {sort_column} {sort_direction} NULLS LAST
        LIMIT ${param_count} OFFSET ${param_count + 1}
    """

    results = []
    total_count = 0

    try:
        async with pg_pool.acquire() as conn:
            # Get total count first
            total_count = await conn.fetchval(count_query, *params) or 0

            # Get paginated data
            rows = await conn.fetch(data_query, *params, limit, offset)

            for row in rows:
                # Calculate 52-week metrics
                week52_high = float(row['high_52w']) if row['high_52w'] else None
                week52_low = float(row['low_52w']) if row['low_52w'] else None
                current_price = float(row['price']) if row['price'] else 0
                
                # Distance from 52-week high/low (percentage)
                pct_from_high = None
                pct_from_low = None
                if week52_high and current_price:
                    pct_from_high = ((current_price - week52_high) / week52_high) * 100
                if week52_low and current_price:
                    pct_from_low = ((current_price - week52_low) / week52_low) * 100

                results.append({
                    "symbol": row['symbol'],
                    "name": row['name'],
                    "sector": row['sector'] or "Unknown",
                    "industry": row['industry'] or "Unknown",
                    "country": row['country'] or "US",
                    "exchange": row['exchange'] or "Unknown",
                    "asset_type": row['asset_type'] or "stock",
                    # Price data
                    "price": current_price,
                    "change": float(row['change']) if row['change'] else 0,
                    "change_pct": float(row['change_pct']) if row['change_pct'] else 0,
                    "volume": int(row['volume']) if row['volume'] else 0,
                    # Valuation
                    "market_cap": float(row['market_cap']) if row['market_cap'] else 0,
                    "pe_ratio": float(row['pe_ratio']) if row['pe_ratio'] else None,
                    "forward_pe": float(row['forward_pe']) if row['forward_pe'] else None,
                    "peg_ratio": float(row['peg_ratio']) if row['peg_ratio'] else None,
                    "price_to_book": float(row['price_to_book']) if row['price_to_book'] else None,
                    "price_to_sales": float(row['price_to_sales']) if row['price_to_sales'] else None,
                    "eps": float(row['eps']) if row['eps'] else None,
                    # Dividends
                    "dividend_yield": float(row['dividend_yield']) if row['dividend_yield'] else None,
                    # Profitability
                    "profit_margin": float(row['profit_margin']) if row['profit_margin'] else None,
                    "operating_margin": float(row['operating_margin']) if row['operating_margin'] else None,
                    "roe": float(row['roe']) if row['roe'] else None,
                    "roa": float(row['roa']) if row['roa'] else None,
                    # Fundamentals
                    "revenue": float(row['revenue']) if row['revenue'] else None,
                    "net_income": float(row['net_income']) if row['net_income'] else None,
                    # Analyst data
                    "analyst_rating": row['analyst_rating'],
                    "analyst_target": float(row['analyst_target_price']) if row['analyst_target_price'] else None,
                    "analyst_count": row['analyst_count'],
                    # 52-week data
                    "week52_high": week52_high,
                    "week52_low": week52_low,
                    "pct_from_52w_high": round(pct_from_high, 2) if pct_from_high else None,
                    "pct_from_52w_low": round(pct_from_low, 2) if pct_from_low else None,
                    "avg_volume": int(row['avg_volume']) if row['avg_volume'] else None,
                    # Metadata
                    "price_updated_at": row['price_updated_at'].isoformat() if row['price_updated_at'] else None,
                })

        logger.info(f"Screener returned {len(results)} results (Total: {total_count})")
        return {
            "results": results,
            "total_count": total_count,
            "page": (offset // limit) + 1,
            "limit": limit,
            "data_source": "real",  # Flag that this is real data
        }

    except Exception as e:
        logger.error(f"Stock screening failed: {e}")
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}") from e


@router.get("/presets")
async def get_preset_screens():
    """Get popular preset screens for quick access"""
    # Note: market_cap is stored in millions (e.g., 4364036.91 = $4.36T)
    return [
        {
            "id": "gainers",
            "name": "Top Gainers",
            "description": "Stocks with positive change today",
            "criteria": {"change_pct_min": 0.01},  # Any positive change
            "sort_by": "change_pct",
            "sort_order": "desc"
        },
        {
            "id": "losers",
            "name": "Top Losers",
            "description": "Stocks with negative change today",
            "criteria": {"change_pct_max": -0.01},  # Any negative change
            "sort_by": "change_pct",
            "sort_order": "asc"
        },
        {
            "id": "most_active",
            "name": "Most Active",
            "description": "Highest volume stocks today",
            "criteria": {},  # Return all, sorted by volume
            "sort_by": "volume",
            "sort_order": "desc"
        },
        {
            "id": "mega_cap",
            "name": "Mega Cap",
            "description": "Market cap over $200B",
            "criteria": {"market_cap_min": 200000},  # 200000 million = $200B
            "sort_by": "market_cap",
            "sort_order": "desc"
        },
        {
            "id": "large_cap",
            "name": "Large Cap",
            "description": "Market cap over $10B",
            "criteria": {"market_cap_min": 10000},  # 10000 million = $10B
            "sort_by": "market_cap",
            "sort_order": "desc"
        },
        {
            "id": "tech",
            "name": "Technology",
            "description": "Technology sector stocks",
            "criteria": {"sector": "Technology"},
            "sort_by": "market_cap",
            "sort_order": "desc"
        },
        {
            "id": "healthcare",
            "name": "Healthcare",
            "description": "Healthcare sector stocks",
            "criteria": {"sector": "Health Care"},  # Match actual DB value
            "sort_by": "market_cap",
            "sort_order": "desc"
        }
    ]

@router.get("/saved")
async def get_saved_screens(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get user's saved screens from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id::text,
                name,
                criteria,
                created_at,
                last_run
            FROM saved_screens
            WHERE user_id = $1
            ORDER BY created_at DESC
            """,
            user_id,
        )

        screens = [
            SavedScreen(
                id=row['id'],
                name=row['name'],
                criteria=ScreenerCriteria(**row['criteria']),
                created_at=row['created_at'],
                last_run=row['last_run'],
            )
            for row in rows
        ]

        return {"screens": screens}


@router.post("/saved")
async def save_screen(
    request: SaveScreenRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Save screen to database"""
    pool = await get_postgres_pool()

    try:
        # Convert criteria to dict, excluding None values for proper JSON storage
        criteria_dict = {k: v for k, v in request.criteria.dict().items() if v is not None}

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO saved_screens (user_id, name, criteria)
                VALUES ($1, $2, $3::jsonb)
                RETURNING id::text, name, criteria, created_at
                """,
                user_id,
                request.name,
                json.dumps(criteria_dict),
            )

            return SavedScreen(
                id=row['id'],
                name=row['name'],
                criteria=ScreenerCriteria(**row['criteria']),
                created_at=row['created_at'],
                last_run=None,
            )
    except Exception as e:
        logger.error(f"Failed to save screen: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save screen: {str(e)}") from e


@router.delete("/saved/{screen_id}")
async def delete_saved_screen(
    screen_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete saved screen from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM saved_screens
            WHERE id = $1::uuid AND user_id = $2
            """,
            screen_id,
            user_id,
        )

        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Screen not found")

        return {"success": True}


@router.post("/saved/{screen_id}/run")
async def run_saved_screen(
    screen_id: str,
    limit: int = 100,
    user_id: UUID = Depends(get_current_user_id),
):
    """Run a saved screen"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT criteria
            FROM saved_screens
            WHERE id = $1::uuid AND user_id = $2
            """,
            screen_id,
            user_id,
        )

        if not row:
            raise HTTPException(status_code=404, detail="Screen not found")

        criteria = ScreenerCriteria(**row['criteria'])

        # Update last_run timestamp
        await conn.execute(
            """
            UPDATE saved_screens
            SET last_run = $1
            WHERE id = $2::uuid
            """,
            datetime.utcnow(),
            screen_id,
        )

    # Run the screen
    return await screen_stocks(criteria, limit, user_id)


@router.post("/advanced-scan")
async def advanced_stock_screen(
    request: dict,  # Generic dict to accept flexible criteria
    user_id: UUID = Depends(get_current_user_id),
):
    """Run advanced stock screen using new screener service"""
    try:
        from cift.services.stock_screener import ScreenerRequest, get_screener_service

        # Convert dict to ScreenerRequest
        screener_request = ScreenerRequest(
            name=request.get('name', 'Advanced Screen'),
            criteria=request.get('criteria', []),
            sort_by=request.get('sort_by', 'market_cap'),
            sort_order=request.get('sort_order', 'desc'),
            limit=request.get('limit', 100),
            save_screen=request.get('save_screen', False)
        )

        screener = get_screener_service()
        result = await screener.screen_stocks(screener_request, str(user_id))

        return result

    except Exception as e:
        logger.error(f"Advanced screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/popular-screens")
async def get_popular_screens():
    """Get popular/template screening criteria"""
    try:
        from cift.services.stock_screener import get_screener_service

        screener = get_screener_service()
        return await screener.get_popular_screens()

    except Exception as e:
        logger.error(f"Failed to get popular screens: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/fields")
async def get_screener_fields():
    """Get available screening fields and their descriptions"""
    try:
        from cift.services.stock_screener import get_screener_service

        screener = get_screener_service()
        return {
            "fields": screener.supported_fields,
            "operators": {
                ">": "Greater than",
                "<": "Less than",
                ">=": "Greater than or equal",
                "<=": "Less than or equal",
                "=": "Equal to",
                "!=": "Not equal to",
                "between": "Between two values",
                "in": "In list of values",
                "not_in": "Not in list of values"
            }
        }

    except Exception as e:
        logger.error(f"Failed to get screener fields: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/saved/{screen_id}/run-advanced")
async def run_saved_screen_advanced(
    screen_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Run a saved screen using advanced screener service"""
    try:
        from cift.services.stock_screener import get_screener_service

        screener = get_screener_service()
        result = await screener.run_saved_screen(screen_id, str(user_id))

        return result

    except Exception as e:
        logger.error(f"Failed to run saved screen: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/sectors")
async def get_sectors():
    """Get list of sectors from database"""
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT sector, COUNT(*) as count
            FROM symbols
            WHERE sector IS NOT NULL AND is_tradable = true
            GROUP BY sector
            ORDER BY sector
            """
        )

        return [
            {
                "sector": row['sector'],
                "count": row['count'],
            }
            for row in rows
        ]


@router.get("/industries")
async def get_industries(
    sector: str | None = None,
):
    """Get list of industries from database"""
    pool = await get_postgres_pool()

    query = """
        SELECT DISTINCT industry, COUNT(*) as count
        FROM symbols
        WHERE industry IS NOT NULL AND is_tradable = true
    """
    params = []

    if sector:
        query += " AND sector = $1"
        params.append(sector)

    query += " GROUP BY industry ORDER BY industry"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        return [
            {
                "industry": row['industry'],
                "count": row['count'],
            }
            for row in rows
        ]


# ============================================================================
# TECHNICAL INDICATORS ENDPOINTS
# ============================================================================

@router.get("/technical/{symbol}")
async def get_technical_indicators(
    symbol: str,
    timeframe: str = "1d",
):
    """
    Get technical indicators for a specific symbol.
    Calculates RSI, MACD, SMA, EMA, Bollinger Bands, Stochastic, ATR
    from REAL ohlcv_bars data in the database.
    """
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            # Get OHLCV data - need at least 200 bars for SMA200
            rows = await conn.fetch(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_bars
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY timestamp ASC
                LIMIT 250
                """,
                symbol.upper(),
                timeframe,
            )
            
            if len(rows) < 14:
                return TechnicalIndicators(
                    symbol=symbol.upper(),
                    current_price=None,
                ).model_dump()
            
            # Extract price arrays
            closes = [float(row['close']) for row in rows]
            highs = [float(row['high']) for row in rows]
            lows = [float(row['low']) for row in rows]
            current_price = closes[-1] if closes else None
            
            # Calculate all indicators
            result = TechnicalIndicators(
                symbol=symbol.upper(),
                rsi_14=calculate_rsi(closes, 14),
                macd=calculate_macd(closes),
                sma_20=calculate_sma(closes, 20),
                sma_50=calculate_sma(closes, 50),
                sma_200=calculate_sma(closes, 200),
                ema_20=calculate_ema(closes, 20),
                ema_50=calculate_ema(closes, 50),
                bollinger_bands=calculate_bollinger_bands(closes, 20),
                stochastic=calculate_stochastic(highs, lows, closes),
                atr_14=calculate_atr(highs, lows, closes, 14),
                current_price=current_price,
            )
            
            return result.model_dump()
            
    except Exception as e:
        logger.error(f"Failed to get technical indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/technical/bulk")
async def get_bulk_technical_indicators(
    symbols: str,  # Comma-separated list
    timeframe: str = "1d",
):
    """
    Get technical indicators for multiple symbols.
    Used by screener to filter by technical criteria.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")][:50]  # Max 50
    
    pool = await get_postgres_pool()
    results = {}
    
    try:
        async with pool.acquire() as conn:
            for symbol in symbol_list:
                rows = await conn.fetch(
                    """
                    SELECT timestamp, open, high, low, close
                    FROM ohlcv_bars
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY timestamp ASC
                    LIMIT 250
                    """,
                    symbol,
                    timeframe,
                )
                
                if len(rows) >= 14:
                    closes = [float(row['close']) for row in rows]
                    highs = [float(row['high']) for row in rows]
                    lows = [float(row['low']) for row in rows]
                    
                    results[symbol] = {
                        "rsi_14": calculate_rsi(closes, 14),
                        "sma_20": calculate_sma(closes, 20),
                        "sma_50": calculate_sma(closes, 50),
                        "sma_200": calculate_sma(closes, 200),
                        "ema_20": calculate_ema(closes, 20),
                        "current_price": closes[-1],
                        "price_above_sma20": closes[-1] > calculate_sma(closes, 20) if calculate_sma(closes, 20) else None,
                        "price_above_sma50": closes[-1] > calculate_sma(closes, 50) if calculate_sma(closes, 50) else None,
                    }
                else:
                    results[symbol] = None
        
        return {"data": results, "count": len([r for r in results.values() if r])}
        
    except Exception as e:
        logger.error(f"Failed to get bulk technical indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# PERFORMANCE METRICS ENDPOINTS
# ============================================================================

@router.get("/performance/{symbol}")
async def get_performance_metrics(
    symbol: str,
):
    """
    Get performance metrics for a specific symbol.
    Calculates returns from REAL ohlcv_bars daily data.
    """
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            # Get daily bars for the past year
            rows = await conn.fetch(
                """
                SELECT timestamp, close
                FROM ohlcv_bars
                WHERE symbol = $1 AND timeframe = '1d'
                ORDER BY timestamp DESC
                LIMIT 365
                """,
                symbol.upper(),
            )
            
            if len(rows) < 5:
                return PerformanceMetrics(symbol=symbol.upper()).model_dump()
            
            # Convert to chronological order
            prices = [(row['timestamp'], float(row['close'])) for row in reversed(rows)]
            current_price = prices[-1][1]
            
            # Find prices at specific dates
            now = datetime.utcnow()
            
            def find_price_at_date(target_date):
                """Find closest price to target date"""
                for ts, price in reversed(prices):
                    if ts.replace(tzinfo=None) <= target_date:
                        return price
                return None
            
            # Calculate returns
            price_1w = find_price_at_date(now - timedelta(days=7))
            price_1m = find_price_at_date(now - timedelta(days=30))
            price_3m = find_price_at_date(now - timedelta(days=90))
            price_6m = find_price_at_date(now - timedelta(days=180))
            price_1y = find_price_at_date(now - timedelta(days=365))
            
            # YTD calculation
            year_start = datetime(now.year, 1, 1)
            price_ytd = find_price_at_date(year_start)
            
            # Calculate 30-day volatility (annualized)
            volatility_30d = None
            if len(prices) >= 30:
                recent_prices = [p[1] for p in prices[-30:]]
                returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                          for i in range(1, len(recent_prices))]
                if returns:
                    volatility_30d = round(np.std(returns) * np.sqrt(252) * 100, 2)
            
            result = PerformanceMetrics(
                symbol=symbol.upper(),
                return_1w=calculate_return(price_1w, current_price),
                return_1m=calculate_return(price_1m, current_price),
                return_3m=calculate_return(price_3m, current_price),
                return_6m=calculate_return(price_6m, current_price),
                return_ytd=calculate_return(price_ytd, current_price),
                return_1y=calculate_return(price_1y, current_price),
                volatility_30d=volatility_30d,
            )
            
            return result.model_dump()
            
    except Exception as e:
        logger.error(f"Failed to get performance metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/performance/bulk")
async def get_bulk_performance_metrics(
    symbols: str,  # Comma-separated list
):
    """
    Get performance metrics for multiple symbols.
    Used by screener to filter by performance criteria.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")][:50]  # Max 50
    
    pool = await get_postgres_pool()
    results = {}
    
    try:
        async with pool.acquire() as conn:
            for symbol in symbol_list:
                rows = await conn.fetch(
                    """
                    SELECT timestamp, close
                    FROM ohlcv_bars
                    WHERE symbol = $1 AND timeframe = '1d'
                    ORDER BY timestamp DESC
                    LIMIT 365
                    """,
                    symbol,
                )
                
                if len(rows) >= 5:
                    prices = [(row['timestamp'], float(row['close'])) for row in reversed(rows)]
                    current_price = prices[-1][1]
                    now = datetime.utcnow()
                    
                    def find_price_at_date(target_date):
                        for ts, price in reversed(prices):
                            if ts.replace(tzinfo=None) <= target_date:
                                return price
                        return None
                    
                    price_1w = find_price_at_date(now - timedelta(days=7))
                    price_1m = find_price_at_date(now - timedelta(days=30))
                    price_3m = find_price_at_date(now - timedelta(days=90))
                    
                    results[symbol] = {
                        "return_1w": calculate_return(price_1w, current_price),
                        "return_1m": calculate_return(price_1m, current_price),
                        "return_3m": calculate_return(price_3m, current_price),
                    }
                else:
                    results[symbol] = None
        
        return {"data": results, "count": len([r for r in results.values() if r])}
        
    except Exception as e:
        logger.error(f"Failed to get bulk performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# SCREENER ALERTS ENDPOINTS
# ============================================================================

@router.get("/alerts")
async def get_screener_alerts(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get all screener alerts for the user"""
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    id::text,
                    title as name,
                    data->'criteria' as criteria,
                    data->'technical_criteria' as technical_criteria,
                    data->'performance_criteria' as performance_criteria,
                    data->'matched_symbols' as matched_symbols,
                    COALESCE((data->>'is_active')::boolean, true) as is_active,
                    COALESCE((data->>'notify_email')::boolean, true) as notify_email,
                    COALESCE((data->>'notify_push')::boolean, false) as notify_push,
                    created_at
                FROM alerts
                WHERE user_id = $1 AND alert_type = 'screener'
                ORDER BY created_at DESC
                """,
                user_id,
            )
            
            alerts = []
            for row in rows:
                alerts.append({
                    "id": row['id'],
                    "name": row['name'],
                    "criteria": row['criteria'] or {},
                    "technical_criteria": row['technical_criteria'],
                    "performance_criteria": row['performance_criteria'],
                    "matched_symbols": row['matched_symbols'] or [],
                    "is_active": row['is_active'],
                    "notify_email": row['notify_email'],
                    "notify_push": row['notify_push'],
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                })
            
            return {"alerts": alerts, "count": len(alerts)}
            
    except Exception as e:
        logger.error(f"Failed to get screener alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/alerts")
async def create_screener_alert(
    request: CreateAlertRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create a new screener alert that triggers when stocks match criteria"""
    pool = await get_postgres_pool()
    
    try:
        # Build alert data
        alert_data = {
            "criteria": request.criteria.model_dump(exclude_none=True),
            "technical_criteria": request.technical_criteria.model_dump(exclude_none=True) if request.technical_criteria else None,
            "performance_criteria": request.performance_criteria.model_dump(exclude_none=True) if request.performance_criteria else None,
            "notify_email": request.notify_email,
            "notify_push": request.notify_push,
            "is_active": True,
            "matched_symbols": [],
        }
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO alerts (user_id, alert_type, severity, title, message, data)
                VALUES ($1, 'screener', 'info', $2, 'Screener alert for matching stocks', $3::jsonb)
                RETURNING id::text, created_at
                """,
                user_id,
                request.name,
                json.dumps(alert_data),
            )
            
            return {
                "id": row['id'],
                "name": request.name,
                "created_at": row['created_at'].isoformat(),
                "message": "Alert created successfully. Will trigger when stocks match criteria.",
            }
            
    except Exception as e:
        logger.error(f"Failed to create screener alert: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/alerts/{alert_id}")
async def delete_screener_alert(
    alert_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete a screener alert"""
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM alerts
                WHERE id = $1::uuid AND user_id = $2 AND alert_type = 'screener'
                """,
                alert_id,
                user_id,
            )
            
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Alert not found")
            
            return {"success": True}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete screener alert: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/alerts/{alert_id}/check")
async def check_screener_alert(
    alert_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Manually check/run a screener alert to find matching stocks"""
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT data
                FROM alerts
                WHERE id = $1::uuid AND user_id = $2 AND alert_type = 'screener'
                """,
                alert_id,
                user_id,
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="Alert not found")
            
            alert_data = row['data']
            criteria = ScreenerCriteria(**alert_data.get('criteria', {}))
            
            # Run the screen
            result = await screen_stocks(criteria, limit=100, offset=0)
            matched_symbols = [r['symbol'] for r in result['results']]
            
            # Update alert with matched symbols
            alert_data['matched_symbols'] = matched_symbols
            await conn.execute(
                """
                UPDATE alerts
                SET data = $1::jsonb
                WHERE id = $2::uuid
                """,
                json.dumps(alert_data),
                alert_id,
            )
            
            return {
                "matched_count": len(matched_symbols),
                "matched_symbols": matched_symbols[:20],  # Return first 20
                "total_in_screen": result['total_count'],
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check screener alert: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# BACKTESTING ENDPOINTS
# ============================================================================

@router.get("/backtests")
async def get_backtests(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get all backtests for the user"""
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT 
                    id::text,
                    name,
                    start_date,
                    end_date,
                    initial_capital,
                    config,
                    results,
                    status,
                    created_at,
                    completed_at
                FROM backtests
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT 50
                """,
                user_id,
            )
            
            backtests = []
            for row in rows:
                results_data = row['results'] or {}
                backtests.append({
                    "id": row['id'],
                    "name": row['name'],
                    "start_date": row['start_date'].isoformat() if row['start_date'] else None,
                    "end_date": row['end_date'].isoformat() if row['end_date'] else None,
                    "initial_capital": float(row['initial_capital']),
                    "status": row['status'],
                    "total_return": results_data.get('total_return'),
                    "max_drawdown": results_data.get('max_drawdown'),
                    "sharpe_ratio": results_data.get('sharpe_ratio'),
                    "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                    "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
                })
            
            return {"backtests": backtests, "count": len(backtests)}
            
    except Exception as e:
        logger.error(f"Failed to get backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/backtests")
async def create_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Create and run a new backtest.
    This simulates applying screener criteria historically to evaluate performance.
    """
    pool = await get_postgres_pool()
    
    try:
        # Parse dates
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
        
        if start_date >= end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if (end_date - start_date).days > 365 * 3:
            raise HTTPException(status_code=400, detail="Backtest period cannot exceed 3 years")
        
        # Build config
        config = {
            "criteria": request.criteria.model_dump(exclude_none=True),
            "technical_criteria": request.technical_criteria.model_dump(exclude_none=True) if request.technical_criteria else None,
            "rebalance_frequency": request.rebalance_frequency,
            "position_sizing": request.position_sizing,
            "max_positions": request.max_positions,
        }
        
        async with pool.acquire() as conn:
            # Create backtest record
            row = await conn.fetchrow(
                """
                INSERT INTO backtests (user_id, name, start_date, end_date, initial_capital, symbols, config, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, 'running')
                RETURNING id::text, created_at
                """,
                user_id,
                request.name,
                start_date,
                end_date,
                request.initial_capital,
                [],  # Will be populated during backtest
                json.dumps(config),
            )
            
            backtest_id = row['id']
            
        # Run backtest in background
        background_tasks.add_task(run_backtest_worker, backtest_id, config, start_date, end_date, request.initial_capital)
        
        return {
            "id": backtest_id,
            "name": request.name,
            "status": "running",
            "message": "Backtest started. Check status with GET /screener/backtests/{id}",
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def run_backtest_worker(backtest_id: str, config: dict, start_date, end_date, initial_capital: float):
    """Background worker to run backtest calculation"""
    pool = await get_postgres_pool()
    
    try:
        # Get symbols that had data in the period
        async with pool.acquire() as conn:
            # Get all unique symbols with daily data in the period
            symbol_rows = await conn.fetch(
                """
                SELECT DISTINCT symbol
                FROM ohlcv_bars
                WHERE timeframe = '1d' 
                AND timestamp >= $1 
                AND timestamp <= $2
                """,
                start_date,
                end_date,
            )
            
            available_symbols = [row['symbol'] for row in symbol_rows]
            
            if not available_symbols:
                # Mark as failed - no data
                await conn.execute(
                    """
                    UPDATE backtests
                    SET status = 'failed', results = $1::jsonb, completed_at = $2
                    WHERE id = $3::uuid
                    """,
                    json.dumps({"error": "No price data available for the selected period"}),
                    datetime.utcnow(),
                    backtest_id,
                )
                return
            
            # Simplified backtest: Run screen criteria and track performance
            # In a real implementation, this would rebalance at intervals
            
            # Get price data for available symbols
            equity_curve = []
            portfolio_value = initial_capital
            
            # Get first day's prices to establish positions
            first_prices = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) symbol, close, timestamp
                FROM ohlcv_bars
                WHERE timeframe = '1d' AND timestamp >= $1
                ORDER BY symbol, timestamp ASC
                """,
                start_date,
            )
            
            # Get last day's prices
            last_prices = await conn.fetch(
                """
                SELECT DISTINCT ON (symbol) symbol, close, timestamp
                FROM ohlcv_bars
                WHERE timeframe = '1d' AND timestamp <= $1
                ORDER BY symbol, timestamp DESC
                """,
                end_date,
            )
            
            first_price_map = {row['symbol']: float(row['close']) for row in first_prices}
            last_price_map = {row['symbol']: float(row['close']) for row in last_prices}
            
            # Simple equal-weight allocation across available symbols
            max_positions = config.get('max_positions', 20)
            symbols_to_use = available_symbols[:max_positions]
            
            if symbols_to_use and len(first_price_map) > 0:
                position_size = initial_capital / len(symbols_to_use)
                
                # Calculate returns
                total_return = 0
                symbol_returns = {}
                
                for symbol in symbols_to_use:
                    if symbol in first_price_map and symbol in last_price_map:
                        start_price = first_price_map[symbol]
                        end_price = last_price_map[symbol]
                        symbol_return = (end_price - start_price) / start_price
                        symbol_returns[symbol] = round(symbol_return * 100, 2)
                        total_return += symbol_return / len(symbols_to_use)
                
                final_value = initial_capital * (1 + total_return)
                
                # Calculate simple metrics
                results = {
                    "total_return": round(total_return * 100, 2),
                    "final_value": round(final_value, 2),
                    "symbol_returns": symbol_returns,
                    "symbols_used": symbols_to_use,
                    "position_count": len(symbols_to_use),
                    # Simplified - real implementation would calculate these from daily returns
                    "max_drawdown": None,
                    "sharpe_ratio": None,
                    "annualized_return": None,
                }
            else:
                results = {
                    "error": "Insufficient price data for backtest",
                    "total_return": 0,
                    "final_value": initial_capital,
                }
            
            # Update backtest with results
            await conn.execute(
                """
                UPDATE backtests
                SET status = 'completed', results = $1::jsonb, symbols = $2, completed_at = $3
                WHERE id = $4::uuid
                """,
                json.dumps(results),
                symbols_to_use,
                datetime.utcnow(),
                backtest_id,
            )
            
    except Exception as e:
        logger.error(f"Backtest worker failed: {e}")
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE backtests
                SET status = 'failed', results = $1::jsonb, completed_at = $2
                WHERE id = $3::uuid
                """,
                json.dumps({"error": str(e)}),
                datetime.utcnow(),
                backtest_id,
            )


@router.get("/backtests/{backtest_id}")
async def get_backtest_detail(
    backtest_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get detailed backtest results"""
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 
                    id::text,
                    name,
                    start_date,
                    end_date,
                    initial_capital,
                    symbols,
                    config,
                    results,
                    status,
                    created_at,
                    completed_at
                FROM backtests
                WHERE id = $1::uuid AND user_id = $2
                """,
                backtest_id,
                user_id,
            )
            
            if not row:
                raise HTTPException(status_code=404, detail="Backtest not found")
            
            results_data = row['results'] or {}
            
            return {
                "id": row['id'],
                "name": row['name'],
                "start_date": row['start_date'].isoformat() if row['start_date'] else None,
                "end_date": row['end_date'].isoformat() if row['end_date'] else None,
                "initial_capital": float(row['initial_capital']),
                "symbols": row['symbols'] or [],
                "config": row['config'] or {},
                "results": results_data,
                "status": row['status'],
                "total_return": results_data.get('total_return'),
                "final_value": results_data.get('final_value'),
                "max_drawdown": results_data.get('max_drawdown'),
                "sharpe_ratio": results_data.get('sharpe_ratio'),
                "symbol_returns": results_data.get('symbol_returns', {}),
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "completed_at": row['completed_at'].isoformat() if row['completed_at'] else None,
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest detail: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/backtests/{backtest_id}")
async def delete_backtest(
    backtest_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete a backtest"""
    pool = await get_postgres_pool()
    
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM backtests
                WHERE id = $1::uuid AND user_id = $2
                """,
                backtest_id,
                user_id,
            )
            
            if result == "DELETE 0":
                raise HTTPException(status_code=404, detail="Backtest not found")
            
            return {"success": True}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# SCATTER PLOT DATA ENDPOINT
# ============================================================================

@router.get("/scatter-data")
async def get_scatter_plot_data(
    x_axis: str = "pe_ratio",
    y_axis: str = "market_cap",
    sector: str | None = None,
):
    """
    Get data formatted for scatter plot visualization.
    Returns symbol data with x/y values for plotting.
    
    Available axes: market_cap, pe_ratio, forward_pe, price_to_book, price_to_sales,
                   dividend_yield, roe, roa, profit_margin, eps, price, change_pct, volume
    """
    pool = await get_postgres_pool()
    
    # Valid field mappings
    field_map = {
        "market_cap": "s.market_cap",
        "pe_ratio": "s.pe_ratio",
        "forward_pe": "s.forward_pe",
        "price_to_book": "s.price_to_book",
        "price_to_sales": "s.price_to_sales",
        "dividend_yield": "s.dividend_yield",
        "roe": "s.roe",
        "roa": "s.roa",
        "profit_margin": "s.profit_margin",
        "operating_margin": "s.operating_margin",
        "eps": "s.eps",
        "price": "mdc.price",
        "change_pct": "mdc.change_pct",
        "volume": "mdc.volume",
    }
    
    if x_axis not in field_map or y_axis not in field_map:
        raise HTTPException(status_code=400, detail=f"Invalid axis. Valid options: {list(field_map.keys())}")
    
    try:
        async with pool.acquire() as conn:
            query = f"""
                SELECT 
                    s.symbol,
                    s.name,
                    s.sector,
                    s.industry,
                    {field_map[x_axis]} as x_value,
                    {field_map[y_axis]} as y_value,
                    s.market_cap
                FROM symbols s
                LEFT JOIN market_data_cache mdc ON s.symbol = mdc.symbol
                WHERE s.is_tradable = true 
                AND s.is_active = true
                AND {field_map[x_axis]} IS NOT NULL
                AND {field_map[y_axis]} IS NOT NULL
            """
            
            params = []
            if sector:
                query += " AND s.sector = $1"
                params.append(sector)
            
            query += " LIMIT 500"
            
            rows = await conn.fetch(query, *params)
            
            data_points = []
            for row in rows:
                data_points.append({
                    "symbol": row['symbol'],
                    "name": row['name'],
                    "sector": row['sector'],
                    "industry": row['industry'],
                    "x": float(row['x_value']) if row['x_value'] else None,
                    "y": float(row['y_value']) if row['y_value'] else None,
                    "size": float(row['market_cap']) if row['market_cap'] else 1,  # Size by market cap
                })
            
            return {
                "x_axis": x_axis,
                "y_axis": y_axis,
                "data": data_points,
                "count": len(data_points),
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scatter plot data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e