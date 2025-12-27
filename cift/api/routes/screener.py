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

    # Market cap filters
    market_cap_min: Decimal | None = None
    market_cap_max: Decimal | None = None

    # Fundamental filters
    pe_ratio_min: Decimal | None = None
    pe_ratio_max: Decimal | None = None
    eps_min: Decimal | None = None
    dividend_yield_min: Decimal | None = None
    beta_min: Decimal | None = None
    beta_max: Decimal | None = None

    # Performance filters
    change_pct_min: Decimal | None = None
    change_pct_max: Decimal | None = None

    # Category filters
    sector: str | None = None
    industry: str | None = None
    country: str | None = None


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
    No authentication required for basic screening.
    """
    pg_pool = await get_postgres_pool()

    # Valid sort columns - map frontend names to actual column expressions
    sort_map = {
        "market_cap": "cp.market_cap",
        "pe_ratio": "cp.pe_ratio",
        "dividend_yield": "cp.dividend_yield",
        "symbol": "cp.symbol",
        "name": "cp.name",
        "sector": "cp.sector",
        "change_pct": "mdc.change_pct",
        "volume": "mdc.volume",
        "price": "mdc.price",
        "beta": "cp.beta",
        "avg_volume": "cp.avg_volume_10d",
    }
    sort_column = sort_map.get(sort_by, "cp.market_cap")
    sort_direction = "DESC" if sort_order.lower() == "desc" else "ASC"

    # Base query
    base_query = """
        FROM company_profiles cp
        LEFT JOIN market_data_cache mdc ON cp.symbol = mdc.symbol
        WHERE 1=1
    """

    params = []
    param_count = 1

    # Apply fundamental filters from company_profiles
    if criteria.market_cap_min:
        base_query += f" AND cp.market_cap >= ${param_count}"
        params.append(float(criteria.market_cap_min))
        param_count += 1

    if criteria.market_cap_max:
        base_query += f" AND cp.market_cap <= ${param_count}"
        params.append(float(criteria.market_cap_max))
        param_count += 1

    if criteria.pe_ratio_min:
        base_query += f" AND cp.pe_ratio >= ${param_count}"
        params.append(float(criteria.pe_ratio_min))
        param_count += 1

    if criteria.pe_ratio_max:
        base_query += f" AND cp.pe_ratio <= ${param_count}"
        params.append(float(criteria.pe_ratio_max))
        param_count += 1

    if criteria.dividend_yield_min:
        base_query += f" AND cp.dividend_yield >= ${param_count}"
        params.append(float(criteria.dividend_yield_min))
        param_count += 1

    if criteria.beta_min:
        base_query += f" AND cp.beta >= ${param_count}"
        params.append(float(criteria.beta_min))
        param_count += 1

    if criteria.beta_max:
        base_query += f" AND cp.beta <= ${param_count}"
        params.append(float(criteria.beta_max))
        param_count += 1

    if criteria.sector:
        base_query += f" AND cp.sector ILIKE ${param_count}"
        params.append(f"%{criteria.sector}%")
        param_count += 1

    if criteria.industry:
        base_query += f" AND cp.industry ILIKE ${param_count}"
        params.append(f"%{criteria.industry}%")
        param_count += 1

    if criteria.country:
        base_query += f" AND cp.country = ${param_count}"
        params.append(criteria.country)
        param_count += 1

    # Apply price filters from market_data_cache
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

    # Data query
    data_query = f"""
        SELECT
            cp.symbol,
            cp.name,
            cp.sector,
            cp.industry,
            cp.country,
            cp.market_cap,
            cp.pe_ratio,
            cp.forward_pe,
            cp.dividend_yield,
            cp.beta,
            cp.fifty_two_week_high,
            cp.fifty_two_week_low,
            cp.avg_volume_10d,
            COALESCE(mdc.price, 0) as price,
            COALESCE(mdc.change, 0) as change,
            COALESCE(mdc.change_pct, 0) as change_percent,
            COALESCE(mdc.volume, 0) as volume
        {base_query}
        ORDER BY {sort_column} {sort_direction} NULLS LAST
        LIMIT ${param_count} OFFSET ${param_count + 1}
    """

    results = []
    total_count = 0

    try:
        async with pg_pool.acquire() as conn:
            # Get total count first
            total_count = await conn.fetchval(count_query, *params)

            # Get paginated data
            rows = await conn.fetch(data_query, *params, limit, offset)

            for row in rows:
                results.append({
                    "symbol": row['symbol'],
                    "name": row['name'],
                    "price": float(row['price']) if row['price'] else 0,
                    "change": float(row['change']) if row['change'] else 0,
                    "change_pct": float(row['change_percent']) if row['change_percent'] else 0,
                    "volume": int(row['volume']) if row['volume'] else 0,
                    "market_cap": float(row['market_cap']) if row['market_cap'] else 0,
                    "pe_ratio": float(row['pe_ratio']) if row['pe_ratio'] else None,
                    "forward_pe": float(row['forward_pe']) if row['forward_pe'] else None,
                    "dividend_yield": float(row['dividend_yield']) if row['dividend_yield'] else None,
                    "beta": float(row['beta']) if row['beta'] else None,
                    "week52_high": float(row['fifty_two_week_high']) if row['fifty_two_week_high'] else None,
                    "week52_low": float(row['fifty_two_week_low']) if row['fifty_two_week_low'] else None,
                    "avg_volume": int(row['avg_volume_10d']) if row['avg_volume_10d'] else None,
                    "sector": row['sector'] or "Unknown",
                    "industry": row['industry'] or "Unknown",
                    "country": row['country'] or "Unknown",
                })

        logger.info(f"Screener returned {len(results)} results (Total: {total_count})")
        return {
            "results": results,
            "total_count": total_count,
            "page": (offset // limit) + 1,
            "limit": limit
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
