"""
STOCK SCREENER API ROUTES
Handles stock screening with technical and fundamental filters.
All data is fetched from database - NO MOCK DATA.
"""

import json
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/screener", tags=["screener"])


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
