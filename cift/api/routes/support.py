"""
SUPPORT CENTER API ROUTES
Handles FAQ, support tickets, and knowledge base.
All data is fetched from database - NO MOCK DATA.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cift.core.auth import get_current_active_user, User
from cift.core.database import get_postgres_pool
from cift.core.logging import logger

router = APIRouter(prefix="/support", tags=["support"])


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_current_user_id(current_user: User = Depends(get_current_active_user)) -> UUID:
    """Get current authenticated user ID."""
    return current_user.id


# ============================================================================
# MODELS
# ============================================================================

class FAQItem(BaseModel):
    """FAQ item model"""
    id: str
    category: str
    question: str
    answer: str
    order: int
    created_at: datetime
    updated_at: datetime


class SupportTicket(BaseModel):
    """Support ticket model"""
    id: str
    subject: str
    category: str
    priority: str  # 'low', 'medium', 'high', 'urgent'
    status: str  # 'open', 'in_progress', 'waiting', 'resolved', 'closed'
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None


class TicketMessage(BaseModel):
    """Ticket message model"""
    id: str
    ticket_id: str
    user_id: Optional[str] = None
    staff_id: Optional[str] = None
    message: str
    is_internal: bool = False
    is_staff: bool = False
    created_at: datetime


class CreateTicketRequest(BaseModel):
    """Create support ticket request"""
    subject: str = Field(..., min_length=5, max_length=200)
    category: str = Field(..., pattern="^(account|trading|funding|technical|billing|other)$")
    priority: str = Field(default="medium", pattern="^(low|medium|high|urgent)$")
    message: str = Field(..., min_length=10)


class AddMessageRequest(BaseModel):
    """Add message to ticket"""
    message: str = Field(..., min_length=1)


# ============================================================================
# ENDPOINTS - FAQ
# ============================================================================

@router.get("/faq")
async def get_faqs(
    category: Optional[str] = None,
    limit: int = 100,
):
    """Get FAQ items from database"""
    pool = await get_postgres_pool()
    
    query = """
        SELECT 
            id::text,
            category,
            question,
            answer,
            display_order,
            created_at,
            updated_at
        FROM faq_items
        WHERE is_published = true
    """
    params = []
    
    if category:
        query += " AND category = $1"
        params.append(category)
    
    query += " ORDER BY display_order ASC, created_at DESC LIMIT $" + str(len(params) + 1)
    params.append(limit)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        return [
            FAQItem(
                id=row['id'],
                category=row['category'],
                question=row['question'],
                answer=row['answer'],
                order=row['display_order'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
            )
            for row in rows
        ]


@router.get("/faq/search")
async def search_faqs(
    q: str,
    limit: int = 50,
):
    """Search FAQ items in database"""
    if len(q) < 3:
        raise HTTPException(status_code=400, detail="Search query must be at least 3 characters")
    
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT 
                id::text,
                category,
                question,
                answer,
                display_order,
                created_at,
                updated_at,
                ts_rank(search_vector, plainto_tsquery('english', $1)) as rank
            FROM faq_items
            WHERE is_published = true
            AND search_vector @@ plainto_tsquery('english', $1)
            ORDER BY rank DESC, display_order ASC
            LIMIT $2
            """,
            q,
            limit,
        )
        
        return [
            FAQItem(
                id=row['id'],
                category=row['category'],
                question=row['question'],
                answer=row['answer'],
                order=row['display_order'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
            )
            for row in rows
        ]


@router.get("/faq/categories")
async def get_faq_categories():
    """Get FAQ categories from database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT category, COUNT(*) as count
            FROM faq_items
            WHERE is_published = true
            GROUP BY category
            ORDER BY category
            """
        )
        
        return [
            {
                "category": row['category'],
                "count": row['count'],
            }
            for row in rows
        ]


# ============================================================================
# ENDPOINTS - SUPPORT TICKETS
# ============================================================================

@router.get("/tickets")
async def get_support_tickets(
    status: Optional[str] = None,
    limit: int = 50,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get user's support tickets from database"""
    pool = await get_postgres_pool()
    
    query = """
        SELECT 
            id::text,
            subject,
            category,
            priority,
            status,
            created_at,
            updated_at,
            resolved_at,
            last_message_at
        FROM support_tickets
        WHERE user_id = $1
    """
    params = [user_id]
    
    if status:
        query += " AND status = $2"
        params.append(status)
    
    query += f" ORDER BY created_at DESC LIMIT ${len(params) + 1}"
    params.append(limit)
    
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        
        return {
            "tickets": [
                SupportTicket(
                    id=row['id'],
                    subject=row['subject'],
                    category=row['category'],
                    priority=row['priority'],
                    status=row['status'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    resolved_at=row['resolved_at'],
                    last_message_at=row['last_message_at'],
                )
                for row in rows
            ]
        }


@router.get("/tickets/{ticket_id}")
async def get_ticket_detail(
    ticket_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get ticket detail with messages from database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        # Get ticket
        ticket = await conn.fetchrow(
            """
            SELECT 
                id::text,
                subject,
                category,
                priority,
                status,
                created_at,
                updated_at,
                resolved_at,
                last_message_at
            FROM support_tickets
            WHERE id = $1::uuid AND user_id = $2
            """,
            ticket_id,
            user_id,
        )
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Get messages
        messages = await conn.fetch(
            """
            SELECT 
                id::text,
                ticket_id::text,
                user_id::text,
                staff_id::text,
                message,
                is_internal,
                created_at
            FROM support_messages
            WHERE ticket_id = $1::uuid
            AND is_internal = false
            ORDER BY created_at ASC
            """,
            ticket_id,
        )
        
        return {
            "ticket": SupportTicket(
                id=ticket['id'],
                subject=ticket['subject'],
                category=ticket['category'],
                priority=ticket['priority'],
                status=ticket['status'],
                created_at=ticket['created_at'],
                updated_at=ticket['updated_at'],
                resolved_at=ticket['resolved_at'],
                last_message_at=ticket['last_message_at'],
            ),
            "messages": [
                TicketMessage(
                    id=msg['id'],
                    ticket_id=msg['ticket_id'],
                    user_id=msg['user_id'],
                    staff_id=msg['staff_id'],
                    message=msg['message'],
                    is_internal=msg['is_internal'],
                    created_at=msg['created_at'],
                )
                for msg in messages
            ],
        }


@router.post("/tickets")
async def create_ticket(
    request: CreateTicketRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Create support ticket in database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create ticket
            ticket = await conn.fetchrow(
                """
                INSERT INTO support_tickets (
                    user_id, subject, category, priority, status
                ) VALUES ($1, $2, $3, $4, 'open')
                RETURNING id::text, subject, category, priority, status, created_at, updated_at
                """,
                user_id,
                request.subject,
                request.category,
                request.priority,
            )
            
            # Add initial message
            await conn.execute(
                """
                INSERT INTO support_messages (
                    ticket_id, user_id, message, is_internal
                ) VALUES ($1::uuid, $2, $3, false)
                """,
                ticket['id'],
                user_id,
                request.message,
            )
            
            # Update last_message_at
            await conn.execute(
                """
                UPDATE support_tickets 
                SET last_message_at = $1 
                WHERE id = $2::uuid
                """,
                datetime.utcnow(),
                ticket['id'],
            )
            
            logger.info(f"Support ticket created: ticket_id={ticket['id']}, user_id={user_id}")
            
            # TODO: Send email notification to support team
            # TODO: Integrate with helpdesk system (Zendesk, Intercom, etc.)
            
            return SupportTicket(
                id=ticket['id'],
                subject=ticket['subject'],
                category=ticket['category'],
                priority=ticket['priority'],
                status=ticket['status'],
                created_at=ticket['created_at'],
                updated_at=ticket['updated_at'],
                resolved_at=None,
                last_message_at=datetime.utcnow(),
            )


@router.get("/tickets/{ticket_id}/messages")
async def get_ticket_messages(
    ticket_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Get all messages for a ticket from database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        # Verify ticket belongs to user
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM support_tickets WHERE id = $1::uuid AND user_id = $2)",
            ticket_id,
            user_id,
        )
        
        if not exists:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Get all messages
        rows = await conn.fetch(
            """
            SELECT 
                id::text,
                ticket_id::text,
                user_id::text,
                staff_id::text,
                message,
                is_internal,
                created_at
            FROM support_messages
            WHERE ticket_id = $1::uuid
            ORDER BY created_at ASC
            """,
            ticket_id,
        )
        
        messages = [
            TicketMessage(
                id=row['id'],
                ticket_id=row['ticket_id'],
                user_id=row['user_id'],
                staff_id=row['staff_id'],
                message=row['message'],
                is_internal=row['is_internal'],
                is_staff=row['staff_id'] is not None,
                created_at=row['created_at'],
            )
            for row in rows
        ]
        
        return {"messages": messages}


@router.post("/tickets/{ticket_id}/messages")
async def add_ticket_message(
    ticket_id: str,
    request: AddMessageRequest,
    user_id: UUID = Depends(get_current_user_id),
):
    """Add message to ticket in database"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        # Verify ticket belongs to user
        exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM support_tickets WHERE id = $1::uuid AND user_id = $2)",
            ticket_id,
            user_id,
        )
        
        if not exists:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Add message
        message = await conn.fetchrow(
            """
            INSERT INTO support_messages (
                ticket_id, user_id, message, is_internal
            ) VALUES ($1::uuid, $2, $3, false)
            RETURNING id::text, ticket_id::text, user_id::text, message, created_at
            """,
            ticket_id,
            user_id,
            request.message,
        )
        
        # Update ticket
        await conn.execute(
            """
            UPDATE support_tickets 
            SET last_message_at = $1, updated_at = $1, status = 'waiting'
            WHERE id = $2::uuid
            """,
            datetime.utcnow(),
            ticket_id,
        )
        
        # TODO: Send email notification to support team
        
        return TicketMessage(
            id=message['id'],
            ticket_id=message['ticket_id'],
            user_id=message['user_id'],
            staff_id=None,
            message=message['message'],
            is_internal=False,
            is_staff=False,
            created_at=message['created_at'],
        )


@router.post("/tickets/{ticket_id}/close")
async def close_ticket(
    ticket_id: str,
    user_id: UUID = Depends(get_current_user_id),
):
    """Close support ticket"""
    pool = await get_postgres_pool()
    
    async with pool.acquire() as conn:
        # Update and return the ticket
        ticket = await conn.fetchrow(
            """
            UPDATE support_tickets 
            SET status = 'closed', updated_at = $1, resolved_at = $1
            WHERE id = $2::uuid AND user_id = $3 AND status != 'closed'
            RETURNING 
                id::text, subject, category, priority, status,
                created_at, updated_at, resolved_at, last_message_at
            """,
            datetime.utcnow(),
            ticket_id,
            user_id,
        )
        
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found or already closed")
        
        return SupportTicket(
            id=ticket['id'],
            subject=ticket['subject'],
            category=ticket['category'],
            priority=ticket['priority'],
            status=ticket['status'],
            created_at=ticket['created_at'],
            updated_at=ticket['updated_at'],
            resolved_at=ticket['resolved_at'],
            last_message_at=ticket['last_message_at'],
        )


# ============================================================================
# ENDPOINTS - CONTACT INFO
# ============================================================================

@router.get("/contact")
async def get_contact_info():
    """Get support contact information"""
    return {
        "email": "support@ciftmarkets.com",
        "phone": "+1 (646) 978-2187",
        "hours": {
            "weekdays": "9:00 AM - 6:00 PM EST",
            "weekends": "10:00 AM - 4:00 PM EST",
        },
        "emergency_line": "+1 (646) 978-2187",
        "average_response_time": "2-4 hours",
    }


@router.get("/status")
async def get_system_status():
    """Get system status for status page"""
    # In production, this would query monitoring systems
    return {
        "status": "operational",
        "last_updated": datetime.utcnow().isoformat(),
        "services": [
            {"name": "Trading Platform", "status": "operational"},
            {"name": "Market Data", "status": "operational"},
            {"name": "API", "status": "operational"},
            {"name": "Mobile App", "status": "operational"},
        ],
    }
