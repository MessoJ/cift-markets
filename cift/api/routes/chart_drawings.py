"""
Chart Drawings API

Endpoints for managing user drawings on charts (trendlines, Fibonacci, etc.).
All drawings persist to PostgreSQL database.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/chart-drawings", tags=["Chart Drawings"])


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class DrawingPoint(BaseModel):
    """Point in price-time space."""
    timestamp: int
    price: float


class DrawingStyle(BaseModel):
    """Drawing style configuration."""
    color: str = "#3b82f6"
    lineWidth: int = 2
    lineType: str = "solid"
    fillColor: str | None = None
    fillOpacity: float | None = None


class DrawingCreate(BaseModel):
    """Create a new drawing."""
    symbol: str = Field(..., max_length=20)
    timeframe: str = Field(..., max_length=10)
    drawing_type: str = Field(
        ...,
        description="Type of drawing",
    )
    drawing_data: dict = Field(..., description="Drawing-specific data (points, text, etc.)")
    style: DrawingStyle = Field(default_factory=DrawingStyle)
    locked: bool = False
    visible: bool = True


class DrawingUpdate(BaseModel):
    """Update an existing drawing."""
    drawing_data: dict | None = None
    style: DrawingStyle | None = None
    locked: bool | None = None
    visible: bool | None = None


class DrawingResponse(BaseModel):
    """Drawing response model."""
    id: str
    user_id: str
    symbol: str
    timeframe: str
    drawing_type: str
    drawing_data: dict
    style: DrawingStyle
    locked: bool
    visible: bool
    created_at: datetime
    updated_at: datetime


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=list[DrawingResponse])
async def get_drawings(
    symbol: str,
    timeframe: str | None = None,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get all drawings for a symbol (and optionally timeframe).

    Performance: ~5-10ms for typical user (< 100 drawings)
    """
    pool = await get_postgres_pool()

    if timeframe:
        query = """
            SELECT
                id::text,
                user_id::text,
                symbol,
                timeframe,
                drawing_type,
                drawing_data,
                style,
                locked,
                visible,
                created_at,
                updated_at
            FROM chart_drawings
            WHERE user_id = $1 AND symbol = $2 AND timeframe = $3 AND visible = TRUE
            ORDER BY created_at DESC
        """
        rows = await pool.fetch(query, user_id, symbol, timeframe)
    else:
        query = """
            SELECT
                id::text,
                user_id::text,
                symbol,
                timeframe,
                drawing_type,
                drawing_data,
                style,
                locked,
                visible,
                created_at,
                updated_at
            FROM chart_drawings
            WHERE user_id = $1 AND symbol = $2 AND visible = TRUE
            ORDER BY created_at DESC
        """
        rows = await pool.fetch(query, user_id, symbol)

    return [dict(row) for row in rows]


@router.post("/", response_model=DrawingResponse)
async def create_drawing(
    drawing: DrawingCreate,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Create a new drawing.

    Performance: ~5ms (single INSERT with RETURNING)
    """
    pool = await get_postgres_pool()

    query = """
        INSERT INTO chart_drawings (
            user_id, symbol, timeframe, drawing_type,
            drawing_data, style, locked, visible
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING
            id::text,
            user_id::text,
            symbol,
            timeframe,
            drawing_type,
            drawing_data,
            style,
            locked,
            visible,
            created_at,
            updated_at
    """

    try:
        row = await pool.fetchrow(
            query,
            user_id,
            drawing.symbol,
            drawing.timeframe,
            drawing.drawing_type,
            drawing.drawing_data,
            drawing.style.dict(),
            drawing.locked,
            drawing.visible,
        )

        logger.info(f"Drawing created: {drawing.drawing_type} on {drawing.symbol} by user {user_id}")
        return dict(row)

    except Exception as e:
        logger.error(f"Failed to create drawing: {e}")
        raise HTTPException(status_code=500, detail="Failed to create drawing") from e


@router.put("/{drawing_id}", response_model=DrawingResponse)
async def update_drawing(
    drawing_id: UUID,
    update: DrawingUpdate,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Update an existing drawing.

    Performance: ~5ms
    """
    pool = await get_postgres_pool()

    # Check ownership
    check_query = "SELECT id FROM chart_drawings WHERE id = $1 AND user_id = $2"
    exists = await pool.fetchrow(check_query, drawing_id, user_id)

    if not exists:
        raise HTTPException(status_code=404, detail="Drawing not found or not owned by user")

    # Build update query dynamically
    updates = []
    params = [drawing_id, user_id]
    param_idx = 3

    if update.drawing_data is not None:
        updates.append(f"drawing_data = ${param_idx}")
        params.append(update.drawing_data)
        param_idx += 1

    if update.style is not None:
        updates.append(f"style = ${param_idx}")
        params.append(update.style.dict())
        param_idx += 1

    if update.locked is not None:
        updates.append(f"locked = ${param_idx}")
        params.append(update.locked)
        param_idx += 1

    if update.visible is not None:
        updates.append(f"visible = ${param_idx}")
        params.append(update.visible)
        param_idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    query = f"""
        UPDATE chart_drawings
        SET {', '.join(updates)}
        WHERE id = $1 AND user_id = $2
        RETURNING
            id::text,
            user_id::text,
            symbol,
            timeframe,
            drawing_type,
            drawing_data,
            style,
            locked,
            visible,
            created_at,
            updated_at
    """

    try:
        row = await pool.fetchrow(query, *params)
        logger.info(f"Drawing updated: {drawing_id} by user {user_id}")
        return dict(row)

    except Exception as e:
        logger.error(f"Failed to update drawing: {e}")
        raise HTTPException(status_code=500, detail="Failed to update drawing") from e


@router.delete("/{drawing_id}")
async def delete_drawing(
    drawing_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Delete a drawing (soft delete by setting visible=false).

    Performance: ~3ms
    """
    pool = await get_postgres_pool()

    query = """
        UPDATE chart_drawings
        SET visible = FALSE
        WHERE id = $1 AND user_id = $2
        RETURNING id
    """

    row = await pool.fetchrow(query, drawing_id, user_id)

    if not row:
        raise HTTPException(status_code=404, detail="Drawing not found or not owned by user")

    logger.info(f"Drawing deleted: {drawing_id} by user {user_id}")
    return {"status": "success", "id": str(drawing_id)}


@router.delete("/symbol/{symbol}")
async def delete_all_drawings(
    symbol: str,
    timeframe: str | None = None,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Delete all drawings for a symbol (soft delete).

    Performance: ~5-10ms
    """
    pool = await get_postgres_pool()

    if timeframe:
        query = """
            UPDATE chart_drawings
            SET visible = FALSE
            WHERE user_id = $1 AND symbol = $2 AND timeframe = $3
            RETURNING id
        """
        rows = await pool.fetch(query, user_id, symbol, timeframe)
    else:
        query = """
            UPDATE chart_drawings
            SET visible = FALSE
            WHERE user_id = $1 AND symbol = $2
            RETURNING id
        """
        rows = await pool.fetch(query, user_id, symbol)

    count = len(rows)
    logger.info(f"Deleted {count} drawings for {symbol} by user {user_id}")

    return {"status": "success", "deleted_count": count}


# Export router
__all__ = ["router"]
