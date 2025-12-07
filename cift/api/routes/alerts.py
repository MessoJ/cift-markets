"""
PRICE ALERTS & NOTIFICATIONS API ROUTES
Handles price alerts, order notifications, and alert management.
All data is fetched from database - NO MOCK DATA.
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/alerts", tags=["alerts"])


# ============================================================================
# MODELS
# ============================================================================

class PriceAlert(BaseModel):
    """Price alert model"""
    id: str
    user_id: str
    symbol: str
    alert_type: str  # 'price_above', 'price_below', 'price_change', 'volume'
    target_value: Decimal
    current_value: Optional[Decimal] = None
    status: str  # 'active', 'triggered', 'cancelled', 'expired'
    notification_methods: List[str] = []  # ['email', 'sms', 'push']
    created_at: datetime
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class CreateAlertRequest(BaseModel):
    """Create alert request"""
    symbol: str = Field(..., min_length=1, max_length=10)
    alert_type: str = Field(..., pattern="^(price_above|price_below|price_change|volume)$")
    target_value: Optional[Decimal] = Field(None, gt=0)
    condition_value: Optional[Decimal] = Field(None, gt=0, description="Alias for target_value")
    notification_methods: List[str] = Field(default=['email', 'push'])
    expires_in_days: Optional[int] = Field(default=30, ge=1, le=365)
    message: Optional[str] = None

    def __init__(self, **data):
        # Handle alias for target_value
        if 'condition_value' in data and 'target_value' not in data:
            data['target_value'] = data['condition_value']
        super().__init__(**data)


class Notification(BaseModel):
    """Notification model"""
    id: str
    user_id: str
    notification_type: str  # 'alert', 'order', 'news', 'system'
    title: str
    message: str
    is_read: bool
    created_at: datetime
    read_at: Optional[datetime] = None
    metadata: Optional[dict] = None


# ============================================================================
# ENDPOINTS - PRICE ALERTS
# ============================================================================

@router.get("")
async def get_alerts(
    status: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get user's price alerts from database"""
    pool = await get_postgres_pool()
    
    query = """
        SELECT 
            id::text,
            user_id::text,
            symbol,
            alert_type,
            target_value,
            current_value,
            status,
            notification_methods,
            created_at,
            triggered_at,
            expires_at
        FROM price_alerts
        WHERE user_id = $1
    """
    params = [user_id]
    param_count = 2
    
    if status:
        query += f" AND status = ${param_count}"
        params.append(status)
        param_count += 1
    
    if symbol:
        query += f" AND symbol = ${param_count}"
        params.append(symbol.upper())
        param_count += 1
    
    query += f" ORDER BY created_at DESC LIMIT ${param_count}"
    params.append(limit)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        return [
            PriceAlert(
                id=row['id'],
                user_id=row['user_id'],
                symbol=row['symbol'],
                alert_type=row['alert_type'],
                target_value=row['target_value'],
                current_value=row['current_value'],
                status=row['status'],
                notification_methods=row['notification_methods'] or [],
                created_at=row['created_at'],
                triggered_at=row['triggered_at'],
                expires_at=row['expires_at'],
            )
            for row in rows
        ]


@router.get("/{alert_id}")
async def get_alert(
    alert_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get single alert detail from database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 
                id::text,
                user_id::text,
                symbol,
                alert_type,
                target_value,
                current_value,
                status,
                notification_methods,
                created_at,
                triggered_at,
                expires_at
            FROM price_alerts
            WHERE id = $1::uuid AND user_id = $2
            """,
            alert_id,
            user_id,
        )
        
        if not row:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return PriceAlert(
            id=row['id'],
            user_id=row['user_id'],
            symbol=row['symbol'],
            alert_type=row['alert_type'],
            target_value=row['target_value'],
            current_value=row['current_value'],
            status=row['status'],
            notification_methods=row['notification_methods'] or [],
            created_at=row['created_at'],
            triggered_at=row['triggered_at'],
            expires_at=row['expires_at'],
        )


@router.post("")
async def create_alert(
    request: CreateAlertRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create price alert in database"""
    pool = await get_postgres_pool()
    
    # Validate notification methods
    valid_methods = {'email', 'sms', 'push'}
    if not all(m in valid_methods for m in request.notification_methods):
        raise HTTPException(status_code=400, detail="Invalid notification method")
    
    # Calculate expiration
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
    
    async with pool.acquire() as conn:
        # Verify symbol exists
        symbol_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM symbols WHERE symbol = $1)",
            request.symbol.upper(),
        )
        
        if not symbol_exists:
            raise HTTPException(status_code=404, detail="Symbol not found")
        
        # Check alert limit (max 50 active alerts per user)
        active_count = await conn.fetchval(
            "SELECT COUNT(*) FROM price_alerts WHERE user_id = $1 AND status = 'active'",
            user_id,
        )
        
        if active_count >= 50:
            raise HTTPException(status_code=400, detail="Maximum alert limit reached (50)")
        
        row = await conn.fetchrow(
            """
            INSERT INTO price_alerts (
                user_id, symbol, alert_type, target_value,
                notification_methods, status, expires_at
            ) VALUES ($1, $2, $3, $4, $5, 'active', $6)
            RETURNING id::text, created_at
            """,
            user_id,
            request.symbol.upper(),
            request.alert_type,
            request.target_value,
            request.notification_methods,
            expires_at,
        )
        
        logger.info(f"Price alert created: id={row['id']}, user_id={user_id}, symbol={request.symbol}")
        
        return {
            "alert_id": row['id'],
            "created_at": row['created_at'],
            "message": "Alert created successfully",
        }


@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete (cancel) price alert"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE price_alerts 
            SET status = 'cancelled' 
            WHERE id = $1::uuid AND user_id = $2 AND status = 'active'
            """,
            alert_id,
            user_id,
        )
        
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail="Alert not found or already inactive")
        
        logger.info(f"Price alert cancelled: id={alert_id}, user_id={user_id}")
        
        return {"success": True, "message": "Alert cancelled"}


@router.post("/bulk-delete")
async def bulk_delete_alerts(
    alert_ids: List[str],
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete multiple alerts"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE price_alerts 
            SET status = 'cancelled' 
            WHERE id = ANY($1::uuid[]) AND user_id = $2 AND status = 'active'
            """,
            alert_ids,
            user_id,
        )
        
        count = int(result.split()[-1])
        
        logger.info(f"Bulk alert cancellation: user_id={user_id}, count={count}")
        
        return {
            "success": True,
            "cancelled_count": count,
            "message": f"{count} alert(s) cancelled",
        }


# ============================================================================
# ENDPOINTS - NOTIFICATIONS
# ============================================================================

@router.get("/notifications")
async def get_notifications(
    is_read: Optional[bool] = None,
    notification_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get user's notifications from database"""
    pool = await get_postgres_pool()
    
    query = """
        SELECT 
            id::text,
            user_id::text,
            notification_type,
            title,
            message,
            is_read,
            created_at,
            read_at,
            metadata
        FROM notifications
        WHERE user_id = $1
    """
    params = [user_id]
    param_count = 2
    
    if is_read is not None:
        query += f" AND is_read = ${param_count}"
        params.append(is_read)
        param_count += 1
    
    if notification_type:
        query += f" AND notification_type = ${param_count}"
        params.append(notification_type)
        param_count += 1
    
    query += f" ORDER BY created_at DESC LIMIT ${param_count} OFFSET ${param_count + 1}"
    params.extend([limit, offset])
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        notifications = [
            Notification(
                id=row['id'],
                user_id=row['user_id'],
                notification_type=row['notification_type'],
                title=row['title'],
                message=row['message'],
                is_read=row['is_read'],
                created_at=row['created_at'],
                read_at=row['read_at'],
                metadata=row['metadata'],
            )
            for row in rows
        ]
        
        # Get unread count
        unread_count = await conn.fetchval(
            "SELECT COUNT(*) FROM notifications WHERE user_id = $1 AND is_read = false",
            user_id,
        )
        
        return {
            "notifications": notifications,
            "unread_count": unread_count,
            "total": len(notifications),
        }


@router.put("/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Mark notification as read"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE notifications 
            SET is_read = true, read_at = $1 
            WHERE id = $2::uuid AND user_id = $3 AND is_read = false
            """,
            datetime.utcnow(),
            notification_id,
            user_id,
        )
        
        if result == "UPDATE 0":
            # Check if notification exists
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM notifications WHERE id = $1::uuid AND user_id = $2)",
                notification_id,
                user_id,
            )
            if not exists:
                raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"success": True}


@router.post("/notifications/mark-all-read")
async def mark_all_notifications_read(
    user_id: UUID = Depends(get_current_user_id),
):
    """Mark all notifications as read"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE notifications 
            SET is_read = true, read_at = $1 
            WHERE user_id = $2 AND is_read = false
            """,
            datetime.utcnow(),
            user_id,
        )
        
        count = int(result.split()[-1])
        
        return {
            "success": True,
            "marked_count": count,
            "message": f"{count} notification(s) marked as read",
        }


@router.delete("/notifications/{notification_id}")
async def delete_notification(
    notification_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Delete notification"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM notifications 
            WHERE id = $1::uuid AND user_id = $2
            """,
            notification_id,
            user_id,
        )
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Notification not found")
        
        return {"success": True}


# ============================================================================
# ENDPOINTS - SETTINGS
# ============================================================================

@router.get("/settings")
async def get_notification_settings(
    user_id: UUID = Depends(get_current_user_id),
):
    """Get user's notification settings from database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT 
                email_notifications,
                sms_notifications,
                push_notifications,
                alert_notifications,
                order_notifications,
                news_notifications,
                marketing_notifications
            FROM notification_settings
            WHERE user_id = $1
            """,
            user_id,
        )
        
        if not row:
            # Return defaults if not set
            return {
                "email_notifications": True,
                "sms_notifications": False,
                "push_notifications": True,
                "alert_notifications": True,
                "order_notifications": True,
                "news_notifications": True,
                "marketing_notifications": False,
            }
        
        return dict(row)


@router.put("/settings")
async def update_notification_settings(
    settings: dict,
    user_id: UUID = Depends(get_current_user_id),
):
    """Update notification settings in database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO notification_settings (
                user_id, email_notifications, sms_notifications, push_notifications,
                alert_notifications, order_notifications, news_notifications, marketing_notifications
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (user_id) DO UPDATE SET
                email_notifications = EXCLUDED.email_notifications,
                sms_notifications = EXCLUDED.sms_notifications,
                push_notifications = EXCLUDED.push_notifications,
                alert_notifications = EXCLUDED.alert_notifications,
                order_notifications = EXCLUDED.order_notifications,
                news_notifications = EXCLUDED.news_notifications,
                marketing_notifications = EXCLUDED.marketing_notifications
            """,
            user_id,
            settings.get('email_notifications', True),
            settings.get('sms_notifications', False),
            settings.get('push_notifications', True),
            settings.get('alert_notifications', True),
            settings.get('order_notifications', True),
            settings.get('news_notifications', True),
            settings.get('marketing_notifications', False),
        )
        
        return {"success": True, "message": "Settings updated"}
