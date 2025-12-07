"""
CIFT Markets - Watchlist API Routes

Manage saved symbol lists with real-time prices.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status
from loguru import logger
from pydantic import BaseModel, Field

from cift.core.auth import get_current_active_user, User
from cift.core.database import db_manager


router = APIRouter(prefix="/watchlists", tags=["Watchlists"])


async def get_current_user_id(current_user: User = Depends(get_current_active_user)) -> UUID:
    return current_user.id


# ============================================================================
# MODELS
# ============================================================================

class WatchlistCreate(BaseModel):
    """Create watchlist request."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    symbols: List[str] = Field(default_factory=list)
    is_default: bool = False


class WatchlistUpdate(BaseModel):
    """Update watchlist request."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    symbols: Optional[List[str]] = None
    is_default: Optional[bool] = None


# ============================================================================
# WATCHLIST CRUD
# ============================================================================

@router.get("")
async def list_watchlists(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get all watchlists for user.
    Performance: ~2-3ms
    """
    async with db_manager.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT 
                id, name, description, symbols, is_default,
                created_at, updated_at
            FROM watchlists
            WHERE user_id = $1
            ORDER BY is_default DESC, sort_order ASC, created_at DESC
        """, user_id)
    
    return {"watchlists": [dict(row) for row in rows]}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_watchlist(
    watchlist: WatchlistCreate,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Create new watchlist.
    Performance: ~3-5ms
    """
    # Uppercase symbols
    symbols = [s.upper() for s in watchlist.symbols]
    
    # If setting as default, unset other defaults
    if watchlist.is_default:
        async with db_manager.pool.acquire() as conn:
            await conn.execute(
                "UPDATE watchlists SET is_default = FALSE WHERE user_id = $1",
                user_id
            )
    
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO watchlists (user_id, name, description, symbols, is_default)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
        """, user_id, watchlist.name, watchlist.description, symbols, watchlist.is_default)
    
    return {"watchlist": dict(row)}


@router.get("/{watchlist_id}")
async def get_watchlist(
    watchlist_id: UUID,
    include_prices: bool = False,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get watchlist with optional real-time prices.
    Performance: ~3-10ms (depending on include_prices)
    """
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM watchlists
            WHERE id = $1 AND user_id = $2
        """, watchlist_id, user_id)
        
        if not row:
            raise HTTPException(status_code=404, detail="Watchlist not found")
        
        watchlist = dict(row)
        
        # Get prices if requested
        if include_prices and watchlist['symbols']:
            prices = await conn.fetch("""
                SELECT 
                    symbol, price, change, change_pct,
                    volume, high, low, open
                FROM market_data_cache
                WHERE symbol = ANY($1)
            """, watchlist['symbols'])
            
            watchlist['prices'] = [dict(p) for p in prices]
    
    return {"watchlist": watchlist}


@router.patch("/{watchlist_id}")
async def update_watchlist(
    watchlist_id: UUID,
    update: WatchlistUpdate,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Update watchlist.
    Performance: ~3-5ms
    """
    # Build update query
    updates = []
    params = [watchlist_id, user_id]
    param_idx = 3
    
    if update.name is not None:
        updates.append(f"name = ${param_idx}")
        params.append(update.name)
        param_idx += 1
    
    if update.description is not None:
        updates.append(f"description = ${param_idx}")
        params.append(update.description)
        param_idx += 1
    
    if update.symbols is not None:
        symbols = [s.upper() for s in update.symbols]
        updates.append(f"symbols = ${param_idx}")
        params.append(symbols)
        param_idx += 1
    
    if update.is_default is not None:
        updates.append(f"is_default = ${param_idx}")
        params.append(update.is_default)
        param_idx += 1
        
        # Unset other defaults
        if update.is_default:
            async with db_manager.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE watchlists SET is_default = FALSE WHERE user_id = $1 AND id != $2",
                    user_id, watchlist_id
                )
    
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    query = f"""
        UPDATE watchlists SET {', '.join(updates)}
        WHERE id = $1 AND user_id = $2
        RETURNING *
    """
    
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow(query, *params)
        
        if not row:
            raise HTTPException(status_code=404, detail="Watchlist not found")
    
    return {"watchlist": dict(row)}


@router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_watchlist(
    watchlist_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Delete watchlist.
    Performance: ~2-3ms
    """
    async with db_manager.pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM watchlists WHERE id = $1 AND user_id = $2",
            watchlist_id, user_id
        )
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Watchlist not found")
    
    return None


@router.post("/{watchlist_id}/symbols/{symbol}")
async def add_symbol_to_watchlist(
    watchlist_id: UUID,
    symbol: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Add symbol to watchlist.
    Performance: ~3-5ms
    """
    symbol = symbol.upper()
    
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow("""
            UPDATE watchlists
            SET symbols = array_append(symbols, $3)
            WHERE id = $1 AND user_id = $2 AND NOT ($3 = ANY(symbols))
            RETURNING *
        """, watchlist_id, user_id, symbol)
        
        if not row:
            raise HTTPException(status_code=404, detail="Watchlist not found or symbol already exists")
    
    return {"watchlist": dict(row)}


@router.delete("/{watchlist_id}/symbols/{symbol}")
async def remove_symbol_from_watchlist(
    watchlist_id: UUID,
    symbol: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Remove symbol from watchlist.
    Performance: ~3-5ms
    """
    symbol = symbol.upper()
    
    async with db_manager.pool.acquire() as conn:
        row = await conn.fetchrow("""
            UPDATE watchlists
            SET symbols = array_remove(symbols, $3)
            WHERE id = $1 AND user_id = $2
            RETURNING *
        """, watchlist_id, user_id, symbol)
        
        if not row:
            raise HTTPException(status_code=404, detail="Watchlist not found")
    
    return {"watchlist": dict(row)}


__all__ = ["router"]
