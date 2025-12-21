"""
CIFT Markets - Notifications API Routes

User notifications for trades, alerts, system messages, etc.
NO MOCK DATA - All data from PostgreSQL database.

Endpoints:
- GET /notifications - List user notifications
- PUT /notifications/{id}/read - Mark notification as read
- PUT /notifications/read-all - Mark all as read
- GET /notifications/unread-count - Get unread count
"""

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from cift.core.auth import get_current_user_id
from cift.core.database import get_postgres_pool

# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/notifications", tags=["Notifications"])


# ============================================================================
# MODELS
# ============================================================================

class Notification(BaseModel):
    """User notification model"""
    id: str
    user_id: str
    type: str  # alert, order, news, system
    title: str
    message: str
    is_read: bool = False
    created_at: datetime
    read_at: datetime | None = None
    metadata: dict | None = None


class UnreadCount(BaseModel):
    """Unread notification count"""
    count: int


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("", response_model=list[Notification])
async def get_notifications(
    limit: int = 50,
    unread_only: bool = False,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get user notifications from database.

    Returns recent notifications with ability to filter to unread only.
    """
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'notifications')"
            )

            if not table_exists:
                logger.warning("notifications table does not exist")
                return []

            # Build query based on filters
            if unread_only:
                query = """
                    SELECT
                        id::text, user_id::text, notification_type as type, title, message,
                        is_read, created_at, read_at, metadata
                    FROM notifications
                    WHERE user_id = $1 AND is_read = false
                    ORDER BY created_at DESC
                    LIMIT $2;
                """
            else:
                query = """
                    SELECT
                        id::text, user_id::text, notification_type as type, title, message,
                        is_read, created_at, read_at, metadata
                    FROM notifications
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2;
                """

            rows = await conn.fetch(query, user_id, limit)

            return [Notification(**dict(row)) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching notifications for user {user_id}: {e}")
        return []


@router.get("/unread-count", response_model=UnreadCount)
async def get_unread_count(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Get count of unread notifications from database.

    Used for badge display in header.
    """
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'notifications')"
            )

            if not table_exists:
                return UnreadCount(count=0)

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM notifications WHERE user_id = $1 AND is_read = false",
                user_id
            )

            return UnreadCount(count=count or 0)
    except Exception as e:
        logger.error(f"Error fetching unread count for user {user_id}: {e}")
        return UnreadCount(count=0)


@router.put("/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
async def mark_notification_read(
    notification_id: UUID,
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Mark a notification as read in database.

    Only works for notifications belonging to the current user.
    """
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'notifications')"
            )

            if not table_exists:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="Notifications not available"
                )

            result = await conn.execute(
                """
                UPDATE notifications
                SET is_read = true, read_at = NOW()
                WHERE id = $1 AND user_id = $2 AND is_read = false
                """,
                notification_id,
                user_id
            )

            if result == "UPDATE 0":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Notification not found or already read"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking notification {notification_id} as read: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notification as read"
        )


@router.put("/read-all", status_code=status.HTTP_204_NO_CONTENT)
async def mark_all_notifications_read(
    user_id: UUID = Depends(get_current_user_id),
):
    """
    Mark all user notifications as read in database.

    Useful for "mark all as read" button.
    """
    try:
        pool = await get_postgres_pool()
        async with pool.acquire() as conn:
            # Check if table exists
            table_exists = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'notifications')"
            )

            if not table_exists:
                return  # Silently succeed if table doesn't exist

            await conn.execute(
                """
                UPDATE notifications
                SET is_read = true, read_at = NOW()
                WHERE user_id = $1 AND is_read = false
                """,
                user_id
            )
    except Exception as e:
        logger.error(f"Error marking all notifications as read for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to mark notifications as read"
        )
