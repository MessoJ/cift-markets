"""
CIFT Markets - Enhanced Price Alerts API Routes

Advanced price monitoring with real-time alerts, multiple conditions,
and comprehensive notification system.
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.services.price_alerts import AlertType, NotificationMethod

router = APIRouter(prefix="/price-alerts", tags=["Price Alerts"])


# ============================================================================
# MODELS
# ============================================================================

class CreateAlertRequest(BaseModel):
    """Enhanced request model for creating price alert"""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    alert_type: AlertType = Field(..., description="Type of price alert")
    condition_value: float = Field(..., gt=0, description="Primary condition value")
    condition_value2: float | None = Field(None, description="Secondary value for range alerts")
    notification_methods: list[NotificationMethod] = Field(default_factory=lambda: [NotificationMethod.IN_APP])
    message: str | None = Field(None, max_length=200, description="Custom alert message")
    expires_at: datetime | None = Field(None, description="Alert expiration time")


class PriceAlertUpdate(BaseModel):
    """Request model for updating price alert"""
    price: float | None = Field(None, gt=0)
    message: str | None = None
    enabled: bool | None = None
    expires_at: datetime | None = None


class PriceAlertResponse(BaseModel):
    """Response model for price alert"""
    id: str
    user_id: str
    symbol: str
    alert_type: str
    price: float
    current_price: float | None = None  # Added field
    message: str | None
    triggered: bool
    triggered_at: datetime | None
    triggered_price: float | None
    notification_sent: bool
    is_active: bool = Field(alias="enabled") # Map enabled to is_active if needed, or just use enabled
    enabled: bool
    created_at: datetime
    expires_at: datetime | None



# ============================================================================
# ENHANCED ENDPOINTS
# ============================================================================

@router.get("", response_model=list[PriceAlertResponse])
async def get_alerts(
    symbol: str | None = None,
    active_only: bool = True,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Get all price alerts for the authenticated user.

    Query Parameters:
    - symbol: Filter by symbol (optional)
    - active_only: Only return non-triggered, enabled alerts (default: true)
    """
    pool = await get_postgres_pool()

    # Build query dynamically
    conditions = ["user_id = $1"]
    params = [user_id]
    param_idx = 2

    if symbol:
        conditions.append(f"symbol = ${param_idx}")
        params.append(symbol)
        param_idx += 1

    if active_only:
        conditions.append("enabled = true AND triggered = false")

    query = f"""
        SELECT id, user_id, symbol, alert_type, price, message, triggered, triggered_at,
               triggered_price, notification_sent, enabled, expires_at, created_at, updated_at
        FROM price_alerts
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        # Fetch current prices

        from cift.core.trading_queries import get_latest_price

        results = []
        for row in rows:
            current_price = await get_latest_price(row['symbol'])

            results.append(PriceAlertResponse(
                id=str(row['id']),
                user_id=str(row['user_id']),
                symbol=row['symbol'],
                alert_type=row['alert_type'],
                price=float(row['price']),
                current_price=current_price,
                message=row['message'],
                triggered=row['triggered'],
                triggered_at=row['triggered_at'],
                triggered_price=float(row['triggered_price']) if row['triggered_price'] else None,
                notification_sent=row['notification_sent'],
                enabled=row['enabled'],
                is_active=row['enabled'], # Map for frontend compatibility
                expires_at=row['expires_at'],
                created_at=row['created_at']
            ))

        return results


@router.get("/{alert_id}", response_model=PriceAlertResponse)
async def get_alert(
    alert_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Get a specific price alert by ID.
    """
    pool = await get_postgres_pool()

    query = """
        SELECT id, user_id, symbol, alert_type, price, message, triggered, triggered_at,
               triggered_price, notification_sent, enabled, expires_at, created_at, updated_at
        FROM price_alerts
        WHERE id = $1 AND user_id = $2
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, alert_id, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Get current price
        from cift.core.trading_queries import get_latest_price
        current_price = await get_latest_price(row['symbol'])

        return PriceAlertResponse(
            id=str(row['id']),
            user_id=str(row['user_id']),
            symbol=row['symbol'],
            alert_type=row['alert_type'],
            price=float(row['price']),
            current_price=current_price,
            message=row['message'],
            triggered=row['triggered'],
            triggered_at=row['triggered_at'],
            triggered_price=float(row['triggered_price']) if row['triggered_price'] else None,
            notification_sent=row['notification_sent'],
            enabled=row['enabled'],
            is_active=row['enabled'],
            expires_at=row['expires_at'],
            created_at=row['created_at']
        )


@router.post("", response_model=PriceAlertResponse, status_code=201)
async def create_alert(
    alert: CreateAlertRequest,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Create a new price alert.
    """
    pool = await get_postgres_pool()

    query = """
        INSERT INTO price_alerts (user_id, symbol, alert_type, price, message, expires_at)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, user_id, symbol, alert_type, price, message, triggered, triggered_at,
                  triggered_price, notification_sent, enabled, expires_at, created_at, updated_at
    """

    async with pool.acquire() as conn:
        try:
            row = await conn.fetchrow(
                query,
                user_id,
                alert.symbol.upper(),
                alert.alert_type,
                alert.condition_value,
                alert.message,
                alert.expires_at
            )

            logger.info(f"Created price alert: {alert.symbol} {alert.alert_type} ${alert.condition_value} for user {user_id}")

            # Get current price
            from cift.core.trading_queries import get_latest_price
            current_price = await get_latest_price(alert.symbol.upper())

            return PriceAlertResponse(
                id=str(row['id']),
                user_id=str(row['user_id']),
                symbol=row['symbol'],
                alert_type=row['alert_type'],
                price=float(row['price']),
                current_price=current_price,
                message=row['message'],
                triggered=row['triggered'],
                triggered_at=row['triggered_at'],
                triggered_price=float(row['triggered_price']) if row['triggered_price'] else None,
                notification_sent=row['notification_sent'],
                enabled=row['enabled'],
                is_active=row['enabled'],
                expires_at=row['expires_at'],
                created_at=row['created_at']
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.patch("/{alert_id}", response_model=PriceAlertResponse)
async def update_alert(
    alert_id: UUID,
    alert_update: PriceAlertUpdate,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Update an existing price alert.
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        # Check if alert exists and belongs to user
        existing = await conn.fetchrow(
            "SELECT id FROM price_alerts WHERE id = $1 AND user_id = $2",
            alert_id, user_id
        )

        if not existing:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Build dynamic update query
        updates = []
        params = [alert_id, user_id]
        param_idx = 3

        if alert_update.price is not None:
            updates.append(f"price = ${param_idx}")
            params.append(alert_update.price)
            param_idx += 1

        if alert_update.message is not None:
            updates.append(f"message = ${param_idx}")
            params.append(alert_update.message)
            param_idx += 1

        if alert_update.enabled is not None:
            updates.append(f"enabled = ${param_idx}")
            params.append(alert_update.enabled)
            param_idx += 1

        if alert_update.expires_at is not None:
            updates.append(f"expires_at = ${param_idx}")
            params.append(alert_update.expires_at)
            param_idx += 1

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        query = f"""
            UPDATE price_alerts
            SET {', '.join(updates)}
            WHERE id = $1 AND user_id = $2
            RETURNING id, user_id, symbol, alert_type, price, message, triggered, triggered_at,
                      triggered_price, notification_sent, enabled, expires_at, created_at, updated_at
        """

        try:
            row = await conn.fetchrow(query, *params)

            logger.info(f"Updated price alert: {alert_id} for user {user_id}")

            return PriceAlertResponse(
                id=str(row['id']),
                user_id=str(row['user_id']),
                symbol=row['symbol'],
                alert_type=row['alert_type'],
                price=float(row['price']),
                message=row['message'],
                triggered=row['triggered'],
                triggered_at=row['triggered_at'],
                triggered_price=float(row['triggered_price']) if row['triggered_price'] else None,
                notification_sent=row['notification_sent'],
                enabled=row['enabled'],
                expires_at=row['expires_at'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update alert: {str(e)}")


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: UUID,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Delete a price alert.
    """
    pool = await get_postgres_pool()

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM price_alerts WHERE id = $1 AND user_id = $2",
            alert_id, user_id
        )

        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Alert not found")

        logger.info(f"Deleted price alert: {alert_id} for user {user_id}")

        return {"message": "Alert deleted successfully"}


@router.post("/{alert_id}/trigger")
async def trigger_alert(
    alert_id: UUID,
    triggered_price: float,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Mark an alert as triggered (called by alert checking system).
    """
    pool = await get_postgres_pool()

    query = """
        UPDATE price_alerts
        SET triggered = true, triggered_at = NOW(), triggered_price = $3
        WHERE id = $1 AND user_id = $2 AND triggered = false
        RETURNING id
    """

    async with pool.acquire() as conn:
        result = await conn.fetchrow(query, alert_id, user_id, triggered_price)

        if not result:
            raise HTTPException(status_code=404, detail="Alert not found or already triggered")

        logger.info(f"Triggered price alert: {alert_id} at ${triggered_price} for user {user_id}")

        return {"message": "Alert triggered successfully", "triggered_price": triggered_price}


@router.get("/check/{symbol}")
async def check_alerts_for_symbol(
    symbol: str,
    current_price: float,
    user_id: UUID = Depends(get_current_user_id)
):
    """
    Check if any alerts should be triggered for a symbol at current price.
    Returns list of triggered alert IDs.
    """
    pool = await get_postgres_pool()

    # Get active alerts for this symbol
    query = """
        SELECT id, alert_type, price
        FROM price_alerts
        WHERE user_id = $1 AND symbol = $2 AND enabled = true AND triggered = false
    """

    async with pool.acquire() as conn:
        alerts = await conn.fetch(query, user_id, symbol.upper())

        triggered_ids = []

        for alert in alerts:
            should_trigger = False

            if alert['alert_type'] == 'above' and current_price > float(alert['price']):
                should_trigger = True
            elif alert['alert_type'] == 'below' and current_price < float(alert['price']):
                should_trigger = True
            # 'crosses_above' and 'crosses_below' would need previous price tracking

            if should_trigger:
                # Trigger the alert
                await conn.execute(
                    """
                    UPDATE price_alerts
                    SET triggered = true, triggered_at = NOW(), triggered_price = $1
                    WHERE id = $2
                    """,
                    current_price, alert['id']
                )
                triggered_ids.append(str(alert['id']))
                logger.info(f"Auto-triggered alert {alert['id']} at ${current_price}")

        return {"triggered_count": len(triggered_ids), "triggered_ids": triggered_ids}
